"""
audio_llm_surrogates.py
=======================
White-box gradient surrogates for LLM-based speech recognizers (Qwen2-Audio,
Qwen2.5-Omni, Phi-4-multimodal, Gemma-3n, SALMONN, ...), for AUTHORIZED
robustness testing. These plug into the ZQ ensemble in zq_attack_gpt_transcribe.py
via the shared WhiteBoxSurrogate interface: return (loss, d loss / d waveform).

The universal audio-LLM recipe (differs from Whisper's single decoder pass):

    1. differentiable front-end        waveform --(torch)--> features
    2. encoder + modality projector     features -> audio embeddings [T_a, d]
    3. assemble a chat prompt           [instruction][AUDIO x T_a][target transcript]
    4. build inputs_embeds MANUALLY     embed text ids via the embedding table,
                                        SPLICE audio embeds into the placeholder
                                        span. (Passing raw input_ids reruns the
                                        model's non-differentiable audio loader
                                        and severs the gradient to the waveform.)
    5. masked teacher-forced CE         labels = -100 everywhere except the
                                        target-transcript positions
    6. loss.backward()                  -> waveform.grad
"""

from __future__ import annotations

from typing import Tuple, List
import numpy as np

# Reuse the framework-agnostic pieces already defined and tested.
from zq_attack_gpt_transcribe import (
    WhiteBoxSurrogate, ToySurrogate, ZQConfig, FitnessWeights,
    zq_sequential_optimize, evaluate_transfer, MockTargetOracle,
    toy_target_audio, cer,
)


# ============================================================================
# 1. Shared base for real audio-LLM surrogates  (torch / transformers template)
# ============================================================================

class AudioLLMSurrogate(WhiteBoxSurrogate):
    """Implements the universal splice + masked-CE loop once. Concrete models
    fill two hooks: `_audio_embeds` (steps 1-2) and `_assemble` (steps 3 & 5).

    Memory notes for real use: freeze model params (grad only on the waveform),
    use bf16/fp16, enable gradient checkpointing, and keep the audio short --
    full backprop-to-input on a 3-8B model needs a GPU.
    """

    def loss_grad(self, waveform: np.ndarray, target_text: str
                  ) -> Tuple[float, np.ndarray]:
        torch = self.torch
        wav = torch.tensor(waveform, dtype=torch.float32,
                           device=self.device, requires_grad=True)

        audio_embeds = self._audio_embeds(wav)              # [T_a, d]  (hooks 1-2)
        n_audio = audio_embeds.shape[0]
        input_ids, audio_slice, labels = self._assemble(target_text, n_audio)  # 3 & 5

        tok_embeds = self.embed_tokens(input_ids)           # [L, d]  text embedding table
        inputs_embeds = tok_embeds.clone()
        inputs_embeds[audio_slice] = audio_embeds           # <-- SPLICE (step 4)

        out = self.model(inputs_embeds=inputs_embeds.unsqueeze(0),
                         labels=labels.unsqueeze(0))
        out.loss.backward()                                 # step 6
        return float(out.loss.item()), wav.grad.detach().cpu().numpy()

    def decode_teacher_forced(self, waveform: np.ndarray, target_text: str) -> str:
        """Debug view: what the surrogate predicts at the transcript positions
        given the current audio. Reuses the same forward as loss_grad (no
        backward). NB this is TEACHER-FORCED -- each position sees the correct
        previous target tokens -- so it's optimistic vs a free-running decode,
        but it tracks convergence: as the loss falls, this should approach the
        target string."""
        torch = self.torch
        with torch.no_grad():
            wav = torch.tensor(waveform, dtype=torch.float32, device=self.device)
            audio_embeds = self._audio_embeds(wav)
            input_ids, audio_slice, labels = self._assemble(target_text,
                                                            audio_embeds.shape[0])
            emb = self.embed_tokens(input_ids).clone()
            emb[audio_slice] = audio_embeds
            logits = self.model(inputs_embeds=emb.unsqueeze(0)).logits[0]  # [L, V]
            resp = (labels != -100).nonzero(as_tuple=True)[0]
            pred = logits[resp - 1].argmax(-1)             # logit t-1 predicts token t
            return self.processor.tokenizer.decode(pred, skip_special_tokens=True)

    # ---- per-model hooks -----------------------------------------------------
    def _audio_embeds(self, wav):
        """Differentiable: waveform -> [T_a, d] audio embeddings. Call the
        encoder + projector submodules directly on differentiable features;
        do NOT go through the processor's audio path."""
        raise NotImplementedError

    def _assemble(self, target_text: str, n_audio_tokens: int):
        """Return (input_ids [L], audio_slice, labels [L]) with the audio
        placeholder span sized to n_audio_tokens and labels masked to -100
        outside the target-transcript positions."""
        raise NotImplementedError

    # Shared helper: most models differ only in pre/post text and the audio token.
    def _assemble_chat(self, tok, pre_text, post_text, audio_token, target_text,
                       n_audio_tokens):
        import torch
        pre = tok(pre_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        aid = tok.convert_tokens_to_ids(audio_token)
        audio_pad = torch.full((n_audio_tokens,), aid, dtype=torch.long)
        post = tok(post_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        resp = tok(target_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        input_ids = torch.cat([pre, audio_pad, post, resp]).to(self.device)
        audio_slice = slice(len(pre), len(pre) + n_audio_tokens)
        labels = torch.full_like(input_ids, -100)
        labels[-len(resp):] = resp                      # CE only on the transcript
        return input_ids, audio_slice, labels


class Qwen2AudioSurrogate(AudioLLMSurrogate):
    """Concrete TEMPLATE for Qwen2-Audio-Instruct. Submodule names and the
    prompt format are the per-model wiring; validate against your transformers
    version. The instruction must ask for TRANSCRIPTION so the response tokens
    equal the target transcript.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2-Audio-7B-Instruct",
                 device: str = "cuda", sr: int = 16000):
        import torch
        import torchaudio
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
        self.torch, self.name, self.device, self.sr = torch, model_id, device, sr

        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16).to(device).eval()
        for p in self.model.parameters():                   # grad only on the waveform
            p.requires_grad_(False)
        self.model.gradient_checkpointing_enable()
        self.processor = AutoProcessor.from_pretrained(model_id)

        import numpy as np
        feat = self.processor.feature_extractor
        # Whisper-style front-end constants (Qwen2-Audio uses a Whisper encoder).
        self.n_fft = getattr(feat, "n_fft", 400)
        self.hop = getattr(feat, "hop_length", 160)
        self.n_samples = getattr(feat, "n_samples", 480000)   # 30 s @ 16 kHz
        self.n_frames = self.n_samples // self.hop            # 3000
        self._window = torch.hann_window(self.n_fft, device=device)
        # The model's OWN mel filterbank (shape [n_freqs, n_mels]); matmul below.
        self._mel_filters = torch.tensor(
            np.asarray(feat.mel_filters), dtype=torch.float32, device=device)

    def embed_tokens(self, input_ids):
        return self.model.get_input_embeddings()(input_ids)  # [L, d]

    def _audio_embeds(self, wav):
        torch = self.torch
        # Differentiable Whisper log-mel: pad/trim to 30 s, STFT, model's filters.
        n = self.n_samples
        wav = (torch.nn.functional.pad(wav, (0, n - wav.shape[0]))
               if wav.shape[0] < n else wav[:n])
        stft = torch.stft(wav, self.n_fft, hop_length=self.hop,
                          window=self._window, return_complex=True)
        mag = stft[..., :-1].abs() ** 2                       # [n_freqs, 3000]
        mel = self._mel_filters.t() @ mag                     # [n_mels, 3000]
        log = torch.clamp(mel, min=1e-10).log10()
        log = torch.maximum(log, log.max() - 8.0)
        mel = ((log + 4.0) / 4.0).to(self.model.dtype)        # match encoder dtype
        # audio_tower + multi_modal_projector are nested under .model on current
        # transformers (older versions expose them at the top level).
        core = self.model.model
        enc = core.audio_tower(mel.unsqueeze(0)).last_hidden_state
        return core.multi_modal_projector(enc)[0]            # [T_a, d]

    def _assemble(self, target_text, n_audio_tokens):
        torch = self.torch
        tok = self.processor.tokenizer
        # Chat prompt: system/user instruction, an audio placeholder run, then
        # the assistant transcript. Build ids and reserve the audio span.
        pre = tok("<|im_start|>user\nTranscribe:", return_tensors="pt").input_ids[0]
        audio_pad = torch.full((n_audio_tokens,),
                               tok.convert_tokens_to_ids("<|AUDIO|>"))
        post = tok("<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").input_ids[0]
        resp = tok(target_text, return_tensors="pt").input_ids[0]

        input_ids = torch.cat([pre, audio_pad, post, resp]).to(self.device)
        start = len(pre)
        audio_slice = slice(start, start + n_audio_tokens)

        labels = torch.full_like(input_ids, -100)            # mask everything ...
        labels[-len(resp):] = resp                            # ... except the transcript
        return input_ids, audio_slice, labels


def introspect_model(model, processor=None):
    """Print the REAL audio-encoder / projector submodule paths and audio special
    tokens for YOUR transformers version, so you can fill _audio_embeds and
    _assemble correctly. Run once per new model, then confirm with
    verify_surrogate.py (its gradient assertion tells you the bindings are right).
    """
    print("== candidate audio submodules (name -> class) ==")
    for name, mod in model.named_modules():
        low = name.lower()
        if name.count(".") <= 2 and any(k in low for k in (
                "audio", "encoder", "tower", "projector", "multi_modal", "adapter",
                "thinker", "connector")):
            print(f"  {name:48s} {type(mod).__name__}")
    print(f"== embedding table == {type(model.get_input_embeddings()).__name__}")
    if processor is not None and hasattr(processor, "tokenizer"):
        tok = processor.tokenizer
        print("== audio-ish special tokens ==")
        vocab = {**{t: tok.convert_tokens_to_ids(t) for t in tok.all_special_tokens},
                 **(tok.get_added_vocab() if hasattr(tok, "get_added_vocab") else {})}
        for t, i in vocab.items():
            if "audio" in t.lower():
                print(f"  {t!r} -> {i}")


class _MelFrontEnd:
    """Differentiable Whisper-style log-mel derived from the model's OWN
    processor, so filters and length match the encoder exactly. Shared by all
    Whisper-encoder audio-LLMs (Qwen2-Audio, Voxtral, ...)."""

    def _init_whisper_mel(self, torch, feature_extractor, device):
        import numpy as np
        f = feature_extractor
        self.n_fft = getattr(f, "n_fft", 400)
        self.hop = getattr(f, "hop_length", 160)
        self.n_samples = getattr(f, "n_samples", 480000)      # 30 s @ 16 kHz
        self._window = torch.hann_window(self.n_fft, device=device)
        mf = np.asarray(f.mel_filters)                        # want [n_freqs, n_mels]
        n_freqs = self.n_fft // 2 + 1
        if mf.shape[0] != n_freqs and mf.shape[1] == n_freqs:
            mf = mf.T
        self._mel_filters = torch.tensor(mf, dtype=torch.float32, device=device)

    def _whisper_log_mel(self, torch, wav, out_dtype):
        n = self.n_samples
        wav = (torch.nn.functional.pad(wav, (0, n - wav.shape[0]))
               if wav.shape[0] < n else wav[:n])
        stft = torch.stft(wav, self.n_fft, hop_length=self.hop,
                          window=self._window, return_complex=True)
        mag = stft[..., :-1].abs() ** 2                       # [n_freqs, frames]
        mel = self._mel_filters.t() @ mag                     # [n_mels, frames]
        log = torch.clamp(mel, min=1e-10).log10()
        log = torch.maximum(log, log.max() - 8.0)
        return ((log + 4.0) / 4.0).to(out_dtype)


class VoxtralSurrogate(AudioLLMSurrogate, _MelFrontEnd):
    """Mistral's Voxtral -- Whisper-style encoder + Mistral LLM, purpose-built for
    transcription, so a strong architectural cousin of gpt-transcribe. Apache-2.0.

    CONFIRM VIA introspect(): the encoder/projector submodule names and the audio
    placeholder token below are best-effort for current transformers; adjust to
    what introspect_model() prints, then run verify_surrogate.py.
    """
    AUDIO_TOKEN = "[AUDIO]"
    PRE = "<s>[INST]Transcribe this audio:"
    POST = "[/INST]"

    def __init__(self, model_id="mistralai/Voxtral-Mini-3B-2507", device="cuda", sr=16000):
        import torch
        from transformers import VoxtralForConditionalGeneration, AutoProcessor
        self.torch, self.name, self.device, self.sr = torch, model_id, device, sr
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.gradient_checkpointing_enable()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self._init_whisper_mel(torch, self.processor.feature_extractor, device)

    def embed_tokens(self, input_ids):
        return self.model.get_input_embeddings()(input_ids)

    def _audio_embeds(self, wav):
        torch = self.torch
        mel = self._whisper_log_mel(torch, wav, self.model.dtype)
        core = getattr(self.model, "model", self.model)
        enc = core.audio_tower(mel.unsqueeze(0)).last_hidden_state    # (1, T, D)
        # Voxtral's projector consumes frames STACKED by k (linear_1.in_features
        # = D*k); concatenate k adjacent encoder frames before projecting.
        proj = core.multi_modal_projector
        b, t, d = enc.shape
        k = proj.linear_1.in_features // d                           # Voxtral: 4
        t2 = (t // k) * k
        enc = enc[:, :t2, :].reshape(b, t2 // k, d * k)              # (1, T/k, D*k)
        return proj(enc)[0]                                          # (T/k, d_llm)

    def _assemble(self, target_text, n_audio_tokens):
        return self._assemble_chat(self.processor.tokenizer, self.PRE, self.POST,
                                   self.AUDIO_TOKEN, target_text, n_audio_tokens)


class Qwen25OmniSurrogate(AudioLLMSurrogate, _MelFrontEnd):
    """Qwen2.5-Omni -- audio encoder + Qwen2.5 'Thinker' LLM. The audio->text path
    lives under model.thinker; load with the Talker disabled to save VRAM.

    CONFIRM VIA introspect(): thinker/audio_tower/projector names and the audio
    token vary by version.
    """
    AUDIO_TOKEN = "<|AUDIO|>"
    PRE = "<|im_start|>user\nTranscribe:"
    POST = "<|im_end|>\n<|im_start|>assistant\n"

    def __init__(self, model_id="Qwen/Qwen2.5-Omni-7B", device="cuda", sr=16000):
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor
        self.torch, self.name, self.device, self.sr = torch, model_id, device, sr
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # We only need the Thinker (audio->text); free the Talker if present.
        if hasattr(self.model, "talker"):
            del self.model.talker
        self.thinker = self.model.thinker
        self.thinker.gradient_checkpointing_enable()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self._init_whisper_mel(torch, self.processor.feature_extractor, device)

    def embed_tokens(self, input_ids):
        return self.thinker.get_input_embeddings()(input_ids)

    def _audio_embeds(self, wav):
        mel = self._whisper_log_mel(self.torch, wav, self.model.dtype)
        enc = self.thinker.audio_tower(mel.unsqueeze(0)).last_hidden_state   # CONFIRM
        return self.thinker.audio_projector(enc)[0]                          # CONFIRM

    def loss_grad(self, waveform, target_text):
        # Same splice/CE as the base, but through the Thinker submodule.
        torch = self.torch
        wav = torch.tensor(waveform, dtype=torch.float32,
                           device=self.device, requires_grad=True)
        audio_embeds = self._audio_embeds(wav)
        input_ids, audio_slice, labels = self._assemble(target_text, audio_embeds.shape[0])
        emb = self.embed_tokens(input_ids).clone()
        emb[audio_slice] = audio_embeds
        out = self.thinker(inputs_embeds=emb.unsqueeze(0), labels=labels.unsqueeze(0))
        out.loss.backward()
        return float(out.loss.item()), wav.grad.detach().cpu().numpy()

    def _assemble(self, target_text, n_audio_tokens):
        return self._assemble_chat(self.processor.tokenizer, self.PRE, self.POST,
                                   self.AUDIO_TOKEN, target_text, n_audio_tokens)


# Phi-4-multimodal (conformer encoder + Phi-4 LLM, trust_remote_code=True) and
# Gemma-3n (gated) follow the same recipe: introspect -> fill the three bindings
# -> verify. Their front-end is conformer, not log-mel, so _audio_embeds calls
# the model's own feature path on a differentiable waveform rather than _log_mel.


# ============================================================================
# 2. Numpy stand-in: verifies an audio-LLM-SHAPED surrogate composes with ZQ
# ============================================================================

class ToyAudioLLMSurrogate(WhiteBoxSurrogate):
    """Mirrors the audio-LLM control flow (instruction prefix + audio tokens +
    response span, masked CE over the response only) so we can confirm a
    differently-structured surrogate drops into the ZQ ensemble. The RBF-to-
    centroids map stands in for [projector -> LM head -> softmax]; a real model
    also mixes positions via attention, which this omits. Gradient plumbing --
    audio-token embeddings differentiable wrt the waveform, response-only
    masking -- is what this validates."""

    def __init__(self, name, alphabet, base_centroids,
                 prefix_len: int = 4, jitter: float = 0.04, seed: int = 7,
                 scale: float = 12.0):
        self.name, self.alphabet, self.scale = name, alphabet, scale
        self.prefix_len = prefix_len
        rng = np.random.default_rng(seed)
        self.centroids = base_centroids + jitter * rng.standard_normal(base_centroids.shape)

    def loss_grad(self, waveform, target_text):
        L, F = waveform.shape[0], len(target_text)
        # sequence = [prefix (masked)] + [audio tokens] + [response (labels)];
        # only response positions contribute to CE, and each is driven by its
        # corresponding audio token -- the toy's stand-in for attention.
        idx = list(np.array_split(np.arange(L), F))
        C, grad, total = self.centroids, np.zeros(L), 0.0
        for f, win in enumerate(idx):
            m = float(waveform[win].mean())                 # differentiable audio embedding
            logits = -self.scale * (m - C) ** 2             # projector + head stand-in
            logits -= logits.max()
            p = np.exp(logits); p /= p.sum()
            t = self.alphabet.index(target_text[f])         # response label
            total += -np.log(p[t] + 1e-12)
            dlogit = p.copy(); dlogit[t] -= 1.0
            dm = float(np.sum(dlogit * (-2.0 * self.scale * (m - C))))
            grad[win] += dm / len(win)
        return total / F, grad / F


# ============================================================================
# 3. Self-test: HETEROGENEOUS ensemble (Whisper-like + audio-LLM-like) -> target
# ============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    SR, L = 16000, 8000
    ALPHABET = "abcdefgh"
    BASE = np.linspace(-1.0, 1.0, len(ALPHABET))
    TARGET, SOURCE_LANG = "faced", "fr"

    x0 = 0.02 * rng.standard_normal(L)

    # Mixed-architecture ensemble: two "Whisper-like" dedicated-ASR surrogates
    # plus two "audio-LLM-like" surrogates, each with distinct centroid jitter.
    ensemble = [
        ToySurrogate("whisper_like_A", ALPHABET, BASE, jitter=0.03, seed=1),
        ToySurrogate("whisper_like_B", ALPHABET, BASE, jitter=0.05, seed=2),
        ToyAudioLLMSurrogate("qwen_like",  ALPHABET, BASE, jitter=0.04, seed=11),
        ToyAudioLLMSurrogate("phi4_like",  ALPHABET, BASE, jitter=0.05, seed=12),
    ]

    TARGET_CENTROIDS = BASE + 0.03 * rng.standard_normal(len(BASE))
    def fresh_target():
        return MockTargetOracle(ALPHABET, TARGET_CENTROIDS,
                                n_windows=len(TARGET), max_queries=10000)

    tgt_audio = toy_target_audio(TARGET, L, ALPHABET, BASE)
    cfg = ZQConfig(eps=1.5, lr=0.10, steps=200, init_scale=0.6)
    w = FitnessWeights()

    print(f"ensemble        : {[s.name for s in ensemble]}")
    print(f"clean transcript: {fresh_target().transcribe(x0, SR).text!r}")
    print(f"targeting       : {TARGET!r}\n")

    delta, hist = zq_sequential_optimize(x0, ensemble, TARGET, cfg,
                                         target_audio=tgt_audio)
    for row in hist:
        print(f"   step {row['step']:>3}  surr_loss={row['surr_loss']:.3f}")

    r = evaluate_transfer(x0, delta, fresh_target(), TARGET, SOURCE_LANG, w, SR)
    print(f"\nadv transcript      : {r['adv_text']!r}")
    print(f"CER to target       : {r['cer_to_target']:.3f}")
    print(f"targeted exact hit  : {r['targeted_exact']}")
    print(f"heterogeneous ensemble composed and transferred: "
          f"{r['targeted_exact'] or r['cer_to_target'] < 0.25}")
