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
    """Mistral's Voxtral -- Whisper-large-v3 encoder + Mistral LLM, purpose-built
    for transcription. Voxtral does NOT use a text chat template for audio; it has
    a dedicated transcribe mode built by processor.apply_transcription_request
    (MistralCommonTokenizer + WhisperFeatureExtractor). We build the real request,
    then swap in a differentiable log-mel so gradients reach the waveform.
    """

    def __init__(self, model_id="mistralai/Voxtral-Mini-3B-2507", device="cuda", sr=16000):
        import torch
        from transformers import VoxtralForConditionalGeneration, AutoProcessor
        self.torch, self.name, self.device, self.sr = torch, model_id, device, sr
        self.model_id = model_id
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.gradient_checkpointing_enable()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self._init_whisper_mel(torch, self.processor.feature_extractor, device)

    def _build_inputs(self, wav_np, target_text):
        """Real Voxtral transcription request + differentiable input_features."""
        import numpy as np
        torch = self.torch
        # processor builds the correct transcribe-mode input_ids + a (non-diff)
        # input_features; we keep its ids and replace features with our own.
        req = self.processor.apply_transcription_request(
            language="en", audio=wav_np.astype(np.float32),
            model_id=self.model_id, sampling_rate=self.sr, format="wav")
        return req

    def loss_grad(self, waveform, target_text):
        torch = self.torch
        wav = torch.tensor(waveform, dtype=torch.float32,
                           device=self.device, requires_grad=True)
        diff_feats = self._whisper_log_mel(torch, wav, self.model.dtype)  # (n_mels, 3000)
        import numpy as np
        req = self._build_inputs(np.asarray(waveform, dtype=np.float32), target_text)
        input_ids = req["input_ids"].to(self.device)
        # append the target transcript as the labels/response
        tok = self.processor.tokenizer
        resp = torch.tensor(tok.encode(target_text), device=self.device).unsqueeze(0)
        full_ids = torch.cat([input_ids, resp], dim=1)
        labels = torch.full_like(full_ids, -100)
        labels[:, -resp.shape[1]:] = resp
        out = self.model(input_ids=full_ids,
                         input_features=diff_feats.unsqueeze(0).to(self.model.dtype),
                         labels=labels)
        out.loss.backward()
        return float(out.loss.item()), wav.grad.detach().cpu().numpy()

    def decode_teacher_forced(self, waveform, target_text):
        torch = self.torch
        with torch.no_grad():
            import numpy as np
            wav = torch.tensor(waveform, dtype=torch.float32, device=self.device)
            diff_feats = self._whisper_log_mel(torch, wav, self.model.dtype)
            req = self._build_inputs(np.asarray(waveform, dtype=np.float32), target_text)
            input_ids = req["input_ids"].to(self.device)
            tok = self.processor.tokenizer
            resp = torch.tensor(tok.encode(target_text), device=self.device).unsqueeze(0)
            full_ids = torch.cat([input_ids, resp], dim=1)
            logits = self.model(
                input_ids=full_ids,
                input_features=diff_feats.unsqueeze(0).to(self.model.dtype)).logits[0]
            n = resp.shape[1]
            return tok.decode(logits[-n-1:-1].argmax(-1))

    # unused for Voxtral (custom loss_grad above), kept for the ABC
    def embed_tokens(self, input_ids):
        return self.model.get_input_embeddings()(input_ids)

    def _audio_embeds(self, wav):
        raise NotImplementedError("Voxtral uses a custom loss_grad path.")

    def _assemble(self, target_text, n_audio_tokens):
        raise NotImplementedError("Voxtral uses a custom loss_grad path.")
        # Fallback (manual) -- only if no native method exists; frame-order may
        # differ, so confirm with the debug decode reading as real text.
        core = getattr(self.model, "model", self.model)
        enc = core.audio_tower(feats).last_hidden_state
        proj = core.multi_modal_projector
        b, t, d = enc.shape
        k = proj.linear_1.in_features // d
        t2 = (t // k) * k
        return proj(enc[:, :t2, :].reshape(b, t2 // k, d * k))[0]

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
        torch = self.torch
        mel = self._whisper_log_mel(torch, wav, self.model.dtype)   # (n_mels, T)
        # Omni's tower transposes+chunks input_features (line ~762: input_features.T
        # .split(...)), so it wants an UNBATCHED (n_mels, T) mel, not (1, n_mels, T).
        flen = torch.tensor([mel.shape[-1]], device=self.device)
        # Omni's chunker does input_features.T.split(...): it wants TIME-MAJOR
        # (frames, n_mels), so transpose our (n_mels, frames) mel.
        enc = self.thinker.audio_tower(mel.t(), feature_lens=flen).last_hidden_state
        proj = getattr(self.thinker, "audio_projector", None)
        if proj is not None:
            enc = proj(enc)
        return enc[0] if enc.dim() == 3 else enc                    # (T_a, d)

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


class UltravoxSurrogate(AudioLLMSurrogate, _MelFrontEnd):
    """Ultravox -- Whisper encoder + Llama LLM. Adds a LLAMA backbone (encoder is
    still Whisper-family, so this is LLM diversity, not encoder diversity).
    CONFIRM bindings via `verify_surrogate.py --model ultravox --introspect` and
    the debug decode."""
    AUDIO_TOKEN = "<|audio|>"
    PRE = "<|start_header_id|>user<|end_header_id|>\n\nTranscribe the audio:"
    POST = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    def __init__(self, model_id="fixie-ai/ultravox-v0_5-llama-3_1-8b",
                 device="cuda", sr=16000):
        import torch
        from transformers import AutoModel, AutoProcessor
        # Ultravox v0.5 remote code expects transformers.modeling_utils._init_weights
        # (a module global removed in transformers 5.x). Restore it so its
        # _create_audio_tower works. If loading fails LATER with a different
        # AttributeError, Ultravox v0.5 is simply incompatible with your
        # transformers -- pin a compatible revision= or drop this surrogate.
        import transformers.modeling_utils as _mu
        if not hasattr(_mu, "_init_weights"):
            _mu._init_weights = True
        self.torch, self.name, self.device, self.sr = torch, model_id, device, sr
        self.model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            trust_remote_code=True).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._init_whisper_mel(torch, self.processor.feature_extractor, device)

    def embed_tokens(self, input_ids):
        return self.model.get_input_embeddings()(input_ids)

    def _audio_embeds(self, wav):
        mel = self._whisper_log_mel(self.torch, wav, self.model.dtype)
        feats = mel.unsqueeze(0)
        for meth in ("get_audio_embeds", "get_audio_features"):
            fn = getattr(self.model, meth, None)
            if fn is not None:
                emb = fn(feats)
                return emb[0] if emb.dim() == 3 else emb
        core = getattr(self.model, "model", self.model)              # CONFIRM
        enc = core.audio_tower(feats).last_hidden_state
        return core.multi_modal_projector(enc)[0]


class Phi4MultimodalSurrogate(AudioLLMSurrogate):
    """Phi-4-multimodal -- CONFORMER encoder (3 conv + 24 conformer blocks,
    subsample x8, 80ms tokens) + Phi-4-Mini LLM via audio LoRA. The ONLY
    encoder-diverse surrogate: its front-end is an 80-dim log-Mel FBANK at 10ms
    frames, NOT Whisper's mel -- so it has its own differentiable fbank below.

    Two things to CONFIRM from `--model phi4 --introspect`, because they've moved
    across transformers 5.x versions:
      (1) the audio-encoder submodule path used in _audio_embeds, and
      (2) the encoder's forward signature (mask arg or not).
    Common current path: model.model.embed_tokens_extend.audio_embed.{encoder,
    audio_projection}. Adjust to what introspect prints, then the decode must
    read English before you trust it.
    """

    N_MELS = 80
    AUDIO_TOKEN = "<|audio_1|>"

    def __init__(self, model_id="microsoft/Phi-4-multimodal-instruct",
                 device="cuda", sr=16000):
        import torch
        import numpy as np
        from transformers import AutoModelForCausalLM, AutoProcessor
        self.torch, self.name, self.device, self.sr = torch, model_id, device, sr
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
            _attn_implementation="eager", device_map=device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._n_fft, self._hop = 400, 160                      # 25ms / 10ms @ 16k
        self._win = torch.hann_window(self._n_fft, device=device)
        self._mel_fb = self._make_fbank(torch, np, self.N_MELS, self._n_fft, sr, device)

    @staticmethod
    def _make_fbank(torch, np, n_mels, n_fft, sr, device):
        def hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
        def mel2hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        n_freqs = n_fft // 2 + 1
        hz = mel2hz(np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2))
        bins = np.floor((n_fft + 1) * hz / sr).astype(int)
        fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
        for m in range(1, n_mels + 1):
            l, c, r = bins[m - 1], bins[m], bins[m + 1]
            for k in range(l, c):
                if c > l: fb[m - 1, k] = (k - l) / (c - l)
            for k in range(c, r):
                if r > c: fb[m - 1, k] = (r - k) / (r - c)
        return torch.tensor(fb, device=device)

    def _fbank(self, wav):
        """Differentiable 80-dim log-mel fbank -> (frames, 80)."""
        torch = self.torch
        stft = torch.stft(wav, self._n_fft, hop_length=self._hop,
                          window=self._win, return_complex=True)
        mel = self._mel_fb @ (stft.abs() ** 2)                 # (80, T)
        return torch.log(torch.clamp(mel, min=1e-10)).t().to(self.model.dtype)  # (T,80)

    def _audio_embeds(self, wav):
        torch = self.torch
        feats = self._fbank(wav).unsqueeze(0)                  # (1, T, 80)
        core = self.model.model                                # CONFIRM path below:
        ae = core.embed_tokens_extend.audio_embed
        mask = torch.ones(feats.shape[:2], dtype=torch.long, device=self.device)
        try:
            enc = ae.encoder(feats, mask)                      # CONFIRM signature
        except TypeError:
            enc = ae.encoder(feats)
        enc = enc[0] if isinstance(enc, (tuple, list)) else getattr(
            enc, "last_hidden_state", enc)
        proj = getattr(ae, "audio_projection", None) or getattr(ae, "up_proj", None)
        emb = proj(enc) if proj is not None else enc
        return emb[0] if emb.dim() == 3 else emb

    def embed_tokens(self, input_ids):
        return self.model.get_input_embeddings()(input_ids)

    def _assemble(self, target_text, n_audio_tokens):
        return self._assemble_chat(self.processor.tokenizer,
                                   "<|user|>Transcribe the audio: ",
                                   "<|end|><|assistant|>",
                                   self.AUDIO_TOKEN, target_text, n_audio_tokens)


# Granite-Speech-3.3-8B (IBM) is another strong CONFORMER + LLM option for encoder
# diversity; wire it like Phi-4 once you have Phi-4 working, since both need a
# non-Whisper (fbank/conformer) differentiable front-end.


class GraniteSpeechSurrogate(WhiteBoxSurrogate):
    """IBM Granite-Speech-3.3 -- 16 conformer blocks + q-former projector + Granite
    3.3 LLM. NATIVELY integrated in transformers (no trust_remote_code), so we use
    the model's own forward: processor builds input_ids (with the <|audio|> token)
    and input_features; we swap in a differentiable mel and let the model splice.
    Conformer encoder -> genuine encoder diversity vs the Whisper-family models.
    """

    def __init__(self, model_id="ibm-granite/granite-speech-3.3-8b",
                 device="cuda", sr=16000):
        import torch
        from transformers import (GraniteSpeechForConditionalGeneration,
                                  GraniteSpeechProcessor)
        self.torch, self.name, self.device, self.sr = torch, model_id, device, sr
        self.model = GraniteSpeechForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.processor = GraniteSpeechProcessor.from_pretrained(model_id)
        # GraniteSpeechProcessor may expose its extractor under a different name.
        self.fe = (getattr(self.processor, "feature_extractor", None)
                   or getattr(self.processor, "audio_processor", None)
                   or getattr(self.processor, "audio_feature_extractor", None))
        if self.fe is None:
            for a in dir(self.processor):          # scan for an extractor object
                obj = getattr(self.processor, a, None)
                if obj is not None and "eatureextractor" in type(obj).__name__.lower():
                    self.fe = obj; break
        if self.fe is None:
            raise RuntimeError(
                "Could not find Granite's feature extractor. Run: print([a for a "
                "in dir(processor) if not a.startswith('_')]) and tell me the name.")
        # Differentiable mel matched to Granite's feature extractor.
        import numpy as np
        self.n_fft = getattr(self.fe, "n_fft", 400)
        self.hop = getattr(self.fe, "hop_length", 160)
        self._win = torch.hann_window(self.n_fft, device=device)
        n_mels = getattr(self.fe, "feature_size", getattr(self.fe, "num_mel_bins", 80))
        self._mel_fb = Phi4MultimodalSurrogate._make_fbank(
            torch, np, n_mels, self.n_fft, sr, device)

    def _feats(self, wav):
        torch = self.torch
        stft = torch.stft(wav, self.n_fft, hop_length=self.hop,
                          window=self._win, return_complex=True)
        mel = self._mel_fb @ (stft.abs() ** 2)
        return torch.log(torch.clamp(mel, min=1e-10)).t().to(self.model.dtype)

    def _build(self, wav_np, target_text):
        # Verified Granite usage: audio token inline in user content, and the
        # PROCESSOR takes (text, wav) together -- it expands <|audio|> to the
        # right number of placeholders to match the audio length. We keep its
        # input_ids and swap in our differentiable features in loss_grad.
        chat = [{"role": "user",
                 "content": "<|audio|>Transcribe the speech into written text."}]
        text = self.processor.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True)
        return self.processor(text, wav_np, return_tensors="pt")

    def loss_grad(self, waveform, target_text):
        torch = self.torch
        import numpy as np
        wav = torch.tensor(waveform, dtype=torch.float32,
                           device=self.device, requires_grad=True)
        diff = self._feats(wav).unsqueeze(0)                 # differentiable feats
        inputs = self._build(np.asarray(waveform, dtype=np.float32), target_text)
        input_ids = inputs["input_ids"].to(self.device)
        # feature key: Granite uses input_features (may also want a mask)
        feat_kwargs = {}
        if "input_features_mask" in inputs:
            feat_kwargs["input_features_mask"] = inputs["input_features_mask"].to(self.device)
        tok = self.processor.tokenizer
        resp = torch.tensor(tok.encode(target_text, add_special_tokens=False),
                            device=self.device).unsqueeze(0)
        full = torch.cat([input_ids, resp], dim=1)
        labels = torch.full_like(full, -100); labels[:, -resp.shape[1]:] = resp
        # match feature dtype/shape to what the processor produced
        pf = inputs["input_features"]
        diff = diff.to(self.model.dtype).reshape(pf.shape) if diff.numel() == pf.numel() else diff.to(self.model.dtype)
        out = self.model(input_ids=full, input_features=diff, labels=labels, **feat_kwargs)
        out.loss.backward()
        return float(out.loss.item()), wav.grad.detach().cpu().numpy()

    def decode_teacher_forced(self, waveform, target_text):
        torch = self.torch
        import numpy as np
        with torch.no_grad():
            wav = torch.tensor(waveform, dtype=torch.float32, device=self.device)
            diff = self._feats(wav).unsqueeze(0)
            inputs = self._build(np.asarray(waveform, dtype=np.float32), target_text)
            input_ids = inputs["input_ids"].to(self.device)
            feat_kwargs = {}
            if "input_features_mask" in inputs:
                feat_kwargs["input_features_mask"] = inputs["input_features_mask"].to(self.device)
            tok = self.processor.tokenizer
            resp = torch.tensor(tok.encode(target_text, add_special_tokens=False),
                                device=self.device).unsqueeze(0)
            full = torch.cat([input_ids, resp], dim=1)
            pf = inputs["input_features"]
            diff = diff.to(self.model.dtype).reshape(pf.shape) if diff.numel() == pf.numel() else diff.to(self.model.dtype)
            logits = self.model(input_ids=full, input_features=diff, **feat_kwargs).logits[0]
            n = resp.shape[1]
            return tok.decode(logits[-n-1:-1].argmax(-1))


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
