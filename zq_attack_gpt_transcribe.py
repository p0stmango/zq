"""
zq_attack_gpt_transcribe.py
===========================
A ZQ-Attack-style *transfer* harness for AUTHORIZED robustness testing of the
decision-only `gpt-transcribe` model (no logprobs exposed).

Design
------
ZQ-Attack (Fang et al., CCS 2024) generates a targeted adversarial perturbation
with ZERO queries to the target, by engineering transferability:

    1. a DIVERSE surrogate ensemble  (here: audio models you have white-box
       access to; gpt-4o-transcribe is a good sibling since it *does* expose
       logprobs, though this file uses gradient surrogates);
    2. INITIALIZATION from a scaled copy of the target-command audio, so the
       perturbation starts leaning toward the target and stays quiet;
    3. SEQUENTIAL ENSEMBLE OPTIMIZATION: the shared perturbation is refined
       through the ordered surrogates in turn, so each builds on the last
       instead of averaging independent gradients.

The perturbation is then transferred and evaluated against `gpt-transcribe`,
scored with a composite decision-only fitness (transcript edit distance +
native language-flip + output-length / suppression).

Key abstraction: a surrogate returns (loss, d loss / d waveform). The ZQ loop is
therefore framework-agnostic -- Torch autograd (Whisper) and the analytic toy
surrogate below plug into the exact same loop.

The toy surrogate + mock oracle are self-contained (numpy only) so the loop can
be verified offline. The Whisper / OpenAI classes are runnable TEMPLATES for use
in an environment with model weights and authorized API access; validate their
version-specific details before relying on them.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict

import numpy as np


# ----------------------------------------------------------------------------
# 0. small utilities
# ----------------------------------------------------------------------------

def _levenshtein(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return max(n, m)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return float(len(hyp) > 0)
    return _levenshtein(list(ref), list(hyp)) / len(ref)


def project_linf(delta: np.ndarray, eps: float) -> np.ndarray:
    """L-inf constraint. Replace with a psychoacoustic-masking projection for a
    genuinely inaudible perturbation (Qin et al. 2019; Schoenherr et al. 2019)."""
    return np.clip(delta, -eps, eps)


def snr_db(x0: np.ndarray, delta: np.ndarray) -> float:
    ps = float(np.mean(x0 ** 2)) + 1e-12
    pn = float(np.mean(delta ** 2)) + 1e-12
    return 10.0 * np.log10(ps / pn)


# ----------------------------------------------------------------------------
# 1. surrogate interface  (returns gradient of a TARGETED loss wrt the waveform)
# ----------------------------------------------------------------------------

class WhiteBoxSurrogate(abc.ABC):
    name: str

    @abc.abstractmethod
    def loss_grad(self, waveform: np.ndarray, target_text: str
                  ) -> Tuple[float, np.ndarray]:
        """Return (loss, d loss / d waveform). Minimizing loss pushes the
        surrogate toward transcribing `waveform` as `target_text`."""
        ...


class ToySurrogate(WhiteBoxSurrogate):
    """Self-contained, differentiable-by-hand surrogate for offline verification.

    Pools the waveform into per-window means, classifies each window against a
    set of scalar 'phoneme' centroids via a softmax, and takes cross-entropy
    toward the per-window target characters. Per-surrogate centroid jitter
    simulates architectural diversity across the ensemble. NOT an acoustic model.
    """

    def __init__(self, name: str, alphabet: str, base_centroids: np.ndarray,
                 jitter: float = 0.0, seed: int = 0, scale: float = 12.0):
        self.name = name
        self.alphabet = alphabet
        self.scale = scale                              # softmax sharpness
        rng = np.random.default_rng(seed)
        self.centroids = base_centroids + jitter * rng.standard_normal(base_centroids.shape)

    def _windows(self, L: int, F: int) -> List[np.ndarray]:
        return [w for w in np.array_split(np.arange(L), F)]

    def loss_grad(self, waveform: np.ndarray, target_text: str
                  ) -> Tuple[float, np.ndarray]:
        L = waveform.shape[0]
        F = len(target_text)
        idx = self._windows(L, F)
        C = self.centroids                             # (V,)
        grad = np.zeros(L)
        total = 0.0
        for f, win in enumerate(idx):
            m = float(waveform[win].mean())
            logits = -self.scale * (m - C) ** 2         # (V,)
            logits -= logits.max()
            p = np.exp(logits); p /= p.sum()            # softmax
            t = self.alphabet.index(target_text[f])
            total += -np.log(p[t] + 1e-12)
            # d loss / d m  via softmax-CE then chain through -scale*(m-c)^2
            dlogit = p.copy(); dlogit[t] -= 1.0         # (p - onehot)
            dm = float(np.sum(dlogit * (-2.0 * self.scale * (m - C))))
            grad[win] += dm / len(win)                  # m = mean(win) -> 1/|win|
        return total / F, grad / F


class WhisperSurrogate(WhiteBoxSurrogate):
    """REAL white-box surrogate template (needs: torch, torchaudio, transformers,
    and model weights). Differentiable log-mel -> encoder -> teacher-forced CE on
    the target tokens, backprop to the raw waveform.

    Version caveats to validate in YOUR stack: the torchaudio mel filterbank only
    approximates Whisper's; confirm n_mels (80 vs 128), the 3000-frame crop, and
    how the language/task prefix is supplied (forced_decoder_ids vs. label prefix)
    for your transformers version.
    """

    def __init__(self, model_id: str = "openai/whisper-small",
                 language: str = "fr", task: str = "transcribe",
                 device: str = "cpu"):
        import torch
        import torchaudio
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.torch, self.F = torch, torch.nn.functional
        self.name = model_id
        self.device = device
        self.language, self.task = language, task
        self.sr = 16000
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device).eval()
        self.processor = WhisperProcessor.from_pretrained(model_id)
        n_mels = self.model.config.num_mel_bins
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=400, hop_length=160,
            n_mels=n_mels, power=2.0, norm="slaney", mel_scale="slaney").to(device)

    def _log_mel(self, wav):
        torch, F = self.torch, self.F
        n = 480000                                       # Whisper's 30 s window
        wav = F.pad(wav, (0, n - wav.shape[0])) if wav.shape[0] < n else wav[:n]
        mel = self.melspec(wav)                          # (n_mels, T)
        log_spec = torch.clamp(mel, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec[..., :3000].unsqueeze(0)         # (1, n_mels, 3000)

    def loss_grad(self, waveform: np.ndarray, target_text: str
                  ) -> Tuple[float, np.ndarray]:
        torch = self.torch
        wav = torch.tensor(waveform, dtype=torch.float32,
                           device=self.device, requires_grad=True)
        feats = self._log_mel(wav)
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.language, task=self.task)
        labels = self.processor.tokenizer(
            target_text, return_tensors="pt").input_ids.to(self.device)
        loss = self.model(input_features=feats, labels=labels).loss
        loss.backward()
        return float(loss.item()), wav.grad.detach().cpu().numpy()


# ----------------------------------------------------------------------------
# 2. target oracle  (decision-only; gpt-transcribe returns {text, languages})
# ----------------------------------------------------------------------------

@dataclass
class TargetResponse:
    text: str
    language: str          # first entry of the native `languages` array


class TargetOracle(abc.ABC):
    def __init__(self, max_queries: int = 100_000):
        self.max_queries, self.n_queries = max_queries, 0

    def transcribe(self, waveform: np.ndarray, sr: int) -> TargetResponse:
        if self.n_queries >= self.max_queries:
            raise RuntimeError("Query budget exhausted.")
        self.n_queries += 1
        return self._call(waveform, sr)

    @abc.abstractmethod
    def _call(self, waveform: np.ndarray, sr: int) -> TargetResponse: ...


class GptTranscribeOracle(TargetOracle):
    """REAL target template. gpt-transcribe exposes NO logprobs; `languages`
    replaces the singular `language` field, so we read languages[0].code."""

    def __init__(self, client=None, model: str = "gpt-transcribe", **kw):
        super().__init__(**kw)
        self.client, self.model = client, model

    def _call(self, waveform: np.ndarray, sr: int) -> TargetResponse:
        import io
        import soundfile as sf
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI()                       # reads OPENAI_API_KEY
        buf = io.BytesIO()
        sf.write(buf, np.asarray(waveform, dtype=np.float32), sr,
                 format="WAV", subtype="PCM_16")
        buf.seek(0); buf.name = "audio.wav"
        resp = self.client.audio.transcriptions.create(
            model=self.model, file=buf, response_format="json")
        # gpt-transcribe returns {text, languages}; `languages` replaces `language`.
        data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        text = data.get("text", "") or ""
        langs = data.get("languages") or []
        lang = (langs[0].get("code") if langs else data.get("language", "")) or ""
        return TargetResponse(text=text, language=lang)


class MockTargetOracle(TargetOracle):
    """Offline stand-in with its own (independently jittered) centroids, so
    transfer from the surrogate ensemble is realistic -- strong but imperfect.
    Emits a native-style language code driven by low-frequency energy."""

    def __init__(self, alphabet: str, centroids: np.ndarray, n_windows: int, **kw):
        super().__init__(**kw)
        self.alphabet, self.centroids, self.F = alphabet, centroids, n_windows

    def _call(self, waveform: np.ndarray, sr: int) -> TargetResponse:
        idx = np.array_split(np.arange(waveform.shape[0]), self.F)
        chars = []
        for win in idx:
            m = float(waveform[win].mean())
            chars.append(self.alphabet[int(np.argmin((m - self.centroids) ** 2))])
        first_half = float(waveform[: waveform.shape[0] // 2].mean())
        lang = "fr" if first_half <= 0.15 else "en"     # perturbable language ID
        return TargetResponse(text="".join(chars), language=lang)


# ----------------------------------------------------------------------------
# 3. composite decision-only fitness
# ----------------------------------------------------------------------------

@dataclass
class FitnessWeights:
    w_text: float = 1.0        # distance to targeted transcript
    w_lang: float = 0.3        # reward flipping the native language ID off source
    w_len: float = 0.2         # reward suppression / truncation


def composite_loss(resp: TargetResponse, target_text: str, source_lang: str,
                   clean_len: int, w: FitnessWeights) -> float:
    loss = w.w_text * cer(target_text, resp.text)
    loss -= w.w_lang * float(resp.language != source_lang)
    loss -= w.w_len * float(len(resp.text) < clean_len)
    return loss


# ----------------------------------------------------------------------------
# 4. target-audio initialization  (ZQ component 2)
# ----------------------------------------------------------------------------

def toy_target_audio(target_text: str, L: int, alphabet: str,
                     centroids: np.ndarray) -> np.ndarray:
    """Toy 'target command audio': each window is a constant equal to the target
    char's centroid, so a surrogate pools it straight to the target string.
    For REAL use, synthesize the target phrase with TTS and use that waveform."""
    F = len(target_text)
    wav = np.zeros(L)
    for f, win in enumerate(np.array_split(np.arange(L), F)):
        wav[win] = centroids[alphabet.index(target_text[f])]
    return wav


# ----------------------------------------------------------------------------
# 5. sequential ensemble optimization  (ZQ component 3)  -- the core
# ----------------------------------------------------------------------------

@dataclass
class ZQConfig:
    eps: float = 1.5           # L-inf bound on the perturbation
    lr: float = 0.10           # PGD step size (initial, if decayed)
    steps: int = 200
    init_scale: float = 0.6    # scale of the target-audio initialization
    step_mode: str = "sign"    # "sign" (L-inf canonical) or "grad" (raw)
    lr_decay: bool = True      # linear decay to ~0 so sign-PGD settles
    log_every: int = 40
    eval_every: int = 0        # >0 probes the target for a transfer curve
                               # (NB: this deviates from strict zero-query; for
                               #  measurement only, not part of generation)


def zq_sequential_optimize(
        x0: np.ndarray, surrogates: List[WhiteBoxSurrogate], target_text: str,
        cfg: ZQConfig, target_audio: Optional[np.ndarray] = None,
        target_oracle: Optional[TargetOracle] = None, sr: int = 16000):
    """Zero-query targeted perturbation via sequential ensemble optimization."""
    L = x0.shape[0]
    delta = (cfg.init_scale * target_audio) if target_audio is not None else np.zeros(L)
    delta = project_linf(delta, cfg.eps)

    history = []
    for step in range(cfg.steps):
        lr = cfg.lr * (1.0 - step / cfg.steps) if cfg.lr_decay else cfg.lr
        step_losses = []
        for surr in surrogates:                          # <-- SEQUENTIAL: each
            loss, g = surr.loss_grad(x0 + delta, target_text)   # surrogate refines
            move = np.sign(g) if cfg.step_mode == "sign" else g  # the shared delta
            delta = project_linf(delta - lr * move, cfg.eps)
            step_losses.append(loss)
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            row = {"step": step, "surr_loss": float(np.mean(step_losses))}
            if target_oracle is not None and cfg.eval_every and step % cfg.eval_every == 0:
                r = target_oracle.transcribe(x0 + delta, sr)
                row["target_cer"] = cer(target_text, r.text)
                row["target_text"] = r.text
            history.append(row)
    return delta, history


# ----------------------------------------------------------------------------
# 6. transfer evaluation
# ----------------------------------------------------------------------------

def evaluate_transfer(x0, delta, oracle: TargetOracle, target_text: str,
                      source_lang: str, w: FitnessWeights, sr: int = 16000):
    clean = oracle.transcribe(x0, sr)
    adv = oracle.transcribe(x0 + delta, sr)
    clen = len(clean.text)
    report = {
        "clean_text": clean.text, "clean_lang": clean.language,
        "adv_text": adv.text, "adv_lang": adv.language,
        "cer_to_target": cer(target_text, adv.text),
        "cer_moved_from_clean": cer(clean.text, adv.text),
        "language_flipped": adv.language != clean.language,
        "length_reduced": len(adv.text) < clen,
        "targeted_exact": adv.text == target_text,
        "composite_loss": composite_loss(adv, target_text, source_lang, clen, w),
        "snr_db": snr_db(x0, delta),
    }
    return report


# ----------------------------------------------------------------------------
# 7. offline self-test:  full ZQ pipeline, toy ensemble -> mock gpt-transcribe
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    SR, L = 16000, 8000
    ALPHABET = "abcdefgh"                       # V = 8 phoneme classes
    BASE = np.linspace(-1.0, 1.0, len(ALPHABET))
    TARGET = "faced"                            # all chars in ALPHABET; F = 5
    SOURCE_LANG = "fr"

    x0 = 0.02 * rng.standard_normal(L)          # benign-ish clean signal (toy)

    # (1) diverse surrogate ensemble: same family, jittered centroids
    surrogates = [
        ToySurrogate("surr_A", ALPHABET, BASE, jitter=0.03, seed=1),
        ToySurrogate("surr_B", ALPHABET, BASE, jitter=0.05, seed=2),
        ToySurrogate("surr_C", ALPHABET, BASE, jitter=0.04, seed=3),
    ]
    # target: independently jittered centroids -> realistic imperfect transfer
    TARGET_CENTROIDS = BASE + 0.03 * rng.standard_normal(len(BASE))
    def fresh_target():
        return MockTargetOracle(ALPHABET, TARGET_CENTROIDS,
                                n_windows=len(TARGET), max_queries=10000)

    # (2) target-audio initialization
    tgt_audio = toy_target_audio(TARGET, L, ALPHABET, BASE)

    cfg = ZQConfig(eps=1.5, lr=0.10, steps=200, init_scale=0.6)
    w = FitnessWeights()

    print(f"clean transcript (target model): {fresh_target().transcribe(x0, SR).text!r}")
    print(f"targeting                      : {TARGET!r}\n")

    # --- run WITHOUT init (zeros) to show the init trick matters ---
    d0, _ = zq_sequential_optimize(x0, surrogates, TARGET, cfg, target_audio=None)
    r0 = evaluate_transfer(x0, d0, fresh_target(), TARGET, SOURCE_LANG, w, SR)

    # --- run WITH ZQ target-audio initialization ---
    d1, hist = zq_sequential_optimize(x0, surrogates, TARGET, cfg,
                                      target_audio=tgt_audio)
    r1 = evaluate_transfer(x0, d1, fresh_target(), TARGET, SOURCE_LANG, w, SR)

    print("surrogate-loss trajectory (with init):")
    for row in hist:
        print(f"   step {row['step']:>3}  surr_loss={row['surr_loss']:.3f}")

    def show(tag, r):
        print(f"\n[{tag}]")
        print(f"   adv transcript      : {r['adv_text']!r}")
        print(f"   CER to target       : {r['cer_to_target']:.3f}")
        print(f"   moved from clean    : {r['cer_moved_from_clean']:.3f}")
        print(f"   language flip       : {r['clean_lang']} -> {r['adv_lang']} "
              f"({'yes' if r['language_flipped'] else 'no'})")
        print(f"   targeted exact hit  : {r['targeted_exact']}")
        print(f"   composite loss      : {r['composite_loss']:.3f}")

    show("zero-init (no ZQ initialization)", r0)
    show("ZQ target-audio init", r1)
