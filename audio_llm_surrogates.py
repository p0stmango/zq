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

        feat = self.processor.feature_extractor              # match its mel config
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=getattr(feat, "n_fft", 400),
            hop_length=getattr(feat, "hop_length", 160),
            n_mels=feat.feature_size, power=2.0).to(device)

    def embed_tokens(self, input_ids):
        return self.model.get_input_embeddings()(input_ids)  # [L, d]

    def _audio_embeds(self, wav):
        torch = self.torch
        mel = torch.clamp(self.melspec(wav), min=1e-10).log10()
        mel = ((torch.maximum(mel, mel.max() - 8.0) + 4.0) / 4.0)
        # Encoder + projector submodule names vary by version; commonly:
        enc = self.model.audio_tower(mel.unsqueeze(0)).last_hidden_state
        return self.model.multi_modal_projector(enc)[0]      # [T_a, d]

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
    """Differentiable log-mel shared by Whisper-encoder-based audio-LLMs."""
    def _make_mel(self, torchaudio, sr, n_fft, hop, n_mels, device):
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
            power=2.0).to(device)

    def _log_mel(self, torch, melspec, wav):
        mel = torch.clamp(melspec(wav), min=1e-10).log10()
        return (torch.maximum(mel, mel.max() - 8.0) + 4.0) / 4.0


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
        import torch, torchaudio
        from transformers import VoxtralForConditionalGeneration, AutoProcessor
        self.torch, self.name, self.device, self.sr = torch, model_id, device, sr
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.gradient_checkpointing_enable()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.melspec = self._make_mel(torchaudio, sr, 400, 160, 128, device)

    def embed_tokens(self, input_ids):
        return self.model.get_input_embeddings()(input_ids)

    def _audio_embeds(self, wav):
        mel = self._log_mel(self.torch, self.melspec, wav)[..., :3000]
        enc = self.model.audio_tower(mel.unsqueeze(0)).last_hidden_state   # CONFIRM
        return self.model.multi_modal_projector(enc)[0]                    # CONFIRM

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
        import torch, torchaudio
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
        self.melspec = self._make_mel(torchaudio, sr, 400, 160, 128, device)

    def embed_tokens(self, input_ids):
        return self.thinker.get_input_embeddings()(input_ids)

    def _audio_embeds(self, wav):
        mel = self._log_mel(self.torch, self.melspec, wav)[..., :3000]
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
