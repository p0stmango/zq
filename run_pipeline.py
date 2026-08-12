"""
run_pipeline.py
===============
End-to-end orchestrator for a ZQ-Attack-style transfer attack against the
decision-only `gpt-transcribe`, for AUTHORIZED robustness testing.

Stages
------
  1. build a (diverse) white-box surrogate ensemble
  2. ZERO-QUERY generation: target-audio init + sequential ensemble optimization
  3. (optional) score-based refinement against the logprob-bearing sibling
     gpt-4o-transcribe -- the Devil's-Whisper "spend a few queries to adapt" idea
  4. evaluate transfer to gpt-transcribe with the decision-only composite fitness

Runs fully offline with toy surrogates + a mock target out of the box. Flip
`PipelineConfig.mode = "real"` and fill the real builders to test for real.

Read the accuracy caveats in `ZQ_FIDELITY` at the bottom before trusting results.
"""

from __future__ import annotations

# --- Memory fix: must be set before torch is imported anywhere. ---------------
# expandable_segments:True lets the CUDA allocator grow segments dynamically,
# which eliminates the "failed to allocate X while Y free" fragmentation OOM
# that occurs when there is enough total free memory but no single contiguous
# block large enough to satisfy the request.
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# -----------------------------------------------------------------------------

import abc
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from zq_attack_gpt_transcribe import (
    WhiteBoxSurrogate, ToySurrogate, ZQConfig, FitnessWeights,
    zq_sequential_optimize, evaluate_transfer, TargetOracle, MockTargetOracle,
    GptTranscribeOracle, toy_target_audio, cer, project_linf,
)
from audio_llm_surrogates import ToyAudioLLMSurrogate


# ============================================================================
# Stage-3 helpers: score-based refinement against the logprob-bearing sibling
# ============================================================================

class ScoreOracle(abc.ABC):
    """A logprob-exposing model (gpt-4o-transcribe) used only to POLISH delta.
    Returns a smooth scalar loss = -sum_t logprob(target_token_t)."""
    def __init__(self, max_queries: int = 100_000):
        self.max_queries, self.n_queries = max_queries, 0

    def score(self, waveform: np.ndarray, target_text: str, sr: int) -> float:
        if self.n_queries >= self.max_queries:
            raise RuntimeError("Score-oracle query budget exhausted.")
        self.n_queries += 1
        return self._score(waveform, target_text, sr)

    @abc.abstractmethod
    def _score(self, waveform, target_text, sr) -> float: ...


class MockScoreOracle(ScoreOracle):
    """Offline stand-in for gpt-4o-transcribe. Its centroids are set CLOSE to the
    target's, modelling the sibling being a better proxy for gpt-transcribe than
    the open surrogate ensemble is. Exposes per-position target logprob (idealized:
    real gpt-4o-transcribe returns emitted-token logprobs + limited top_logprobs,
    so for a strictly targeted objective you may need to fall back to decision-
    based refinement -- see ZQ_FIDELITY note 5)."""
    def __init__(self, alphabet, centroids, n_windows, scale=12.0, **kw):
        super().__init__(**kw)
        self.alphabet, self.centroids, self.F, self.scale = alphabet, centroids, n_windows, scale

    def _score(self, waveform, target_text, sr) -> float:
        idx = np.array_split(np.arange(waveform.shape[0]), self.F)
        C, total = self.centroids, 0.0
        for f, win in enumerate(idx):
            m = float(waveform[win].mean())
            logits = -self.scale * (m - C) ** 2
            logits -= logits.max()
            p = np.exp(logits); p /= p.sum()
            t = self.alphabet.index(target_text[f])
            total += -np.log(p[t] + 1e-12)
        return total / self.F


class Gpt4oTranscribeScoreOracle(ScoreOracle):
    """REAL template. Call transcriptions.create(model='gpt-4o-transcribe',
    include=['logprobs']); build the score from the returned per-token logprobs.
    NB gpt-4o-transcribe -- NOT gpt-transcribe -- is the one that exposes logprobs."""
    def __init__(self, client=None, model="gpt-4o-transcribe", **kw):
        super().__init__(**kw)
        self.client, self.model = client, model

    def _score(self, waveform, target_text, sr) -> float:
        import io
        import soundfile as sf
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI()
        buf = io.BytesIO()
        sf.write(buf, np.asarray(waveform, dtype=np.float32), sr,
                 format="WAV", subtype="PCM_16")
        buf.seek(0); buf.name = "audio.wav"
        resp = self.client.audio.transcriptions.create(
            model=self.model, file=buf, response_format="json", include=["logprobs"])
        data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        text = data.get("text", "") or ""
        lps = [t.get("logprob", 0.0) for t in (data.get("logprobs") or [])]
        # HONEST NOTE: gpt-4o-transcribe returns EMITTED-token logprobs, not the
        # probability of your target tokens -- so this is a decision signal
        # (edit distance to target) smoothed by the model's confidence in what it
        # emitted. Pure smooth targeted gradients aren't available from this API;
        # if the sibling refinement underperforms, prefer a decision-based
        # (genetic) refiner against the target itself.
        edit = cer(target_text, text)
        conf = -(sum(lps) / len(lps)) if lps else 0.0     # low conf -> higher loss
        return edit + 0.05 * conf


@dataclass
class RefineConfig:
    sigma: float = 0.01
    pop: int = 16          # antithetic pairs -> 2*pop score-queries per step
    lr: float = 0.05
    steps: int = 40
    seed: int = 0


def nes_refine(x0, delta, score_oracle: ScoreOracle, target_text: str,
               eps: float, cfg: RefineConfig, sr: int = 16000):
    """NES gradient estimate of the sibling's score, then sign-PGD on delta."""
    rng = np.random.default_rng(cfg.seed)
    dim = x0.shape[0]
    for _ in range(cfg.steps):
        grad = np.zeros(dim)
        for _ in range(cfg.pop):
            e = rng.standard_normal(dim)
            sp = score_oracle.score(x0 + project_linf(delta + cfg.sigma * e, eps), target_text, sr)
            sm = score_oracle.score(x0 + project_linf(delta - cfg.sigma * e, eps), target_text, sr)
            grad += (sp - sm) * e
        grad /= (2 * cfg.pop * cfg.sigma)
        delta = project_linf(delta - cfg.lr * np.sign(grad), eps)   # minimize the score-loss
    return delta


# ============================================================================
# Pipeline config + builders
# ============================================================================

@dataclass
class PipelineConfig:
    mode: str = "offline"                 # "offline" | "real"
    target_text: str = "faced"            # the transcript to force
    source_lang: str = "fr"
    sr: int = 16000
    audio_len: int = 8000                 # offline toy length
    zq: ZQConfig = field(default_factory=lambda: ZQConfig(
        eps=1.5, lr=0.10, steps=200, init_scale=0.6))
    fitness: FitnessWeights = field(default_factory=FitnessWeights)
    use_refinement: bool = True
    refine: RefineConfig = field(default_factory=RefineConfig)
    debug_surrogates: bool = False        # print each surrogate's decode per log step
    # offline-only knobs
    alphabet: str = "abcdefgh"


def build_ensemble(cfg: PipelineConfig) -> List[WhiteBoxSurrogate]:
    if cfg.mode == "real":
        import time
        from audio_llm_surrogates import (
            Qwen2AudioSurrogate, VoxtralSurrogate, Qwen25OmniSurrogate,
            UltravoxSurrogate, Phi4MultimodalSurrogate, GraniteSpeechSurrogate)

        def _load(desc, fn):
            print(f"[load] {desc} ...", flush=True)
            t = time.time(); s = fn()
            print(f"[load] {desc} ready ({time.time()-t:.0f}s)", flush=True)
            return s

        # Diverse audio-LLM ensemble (no Whisper-family dedicated ASR).
        # LLM-backbone diversity: Qwen2 / Mistral / Qwen2.5 / Llama.
        # The first four share a WHISPER-family encoder; Phi-4 (conformer) is the
        # only ENCODER-diverse one and needs its front-end wired -- add it once it
        # passes verify. Verify EACH with the debug decode before trusting it:
        # a model that passes verify but decodes garbage (as Voxtral did) is worse
        # than absent, because it injects noise gradients into the shared delta.
        ens = []
        ens.append(_load("Qwen2-Audio-7B",  lambda: Qwen2AudioSurrogate(
            "Qwen/Qwen2-Audio-7B-Instruct", device="cuda")))
        ens.append(_load("Voxtral-Mini-3B", lambda: VoxtralSurrogate(
            "mistralai/Voxtral-Mini-3B-2507", device="cuda")))
        ens.append(_load("Granite-Speech (conformer)", lambda: GraniteSpeechSurrogate(
            "ibm-granite/granite-speech-3.3-8b", device="cuda")))
        #ens.append(_load("Phi-4-MM (conformer)", lambda: Phi4MultimodalSurrogate(
        #    "microsoft/Phi-4-multimodal-instruct", device="cuda")))
        return ens
        ens.append(_load("Qwen2.5-Omni-7B", lambda: Qwen25OmniSurrogate(
            "Qwen/Qwen2.5-Omni-7B", device="cuda")))
        # Ultravox v0.5 remote code is incompatible with transformers 5.x (meta-device
        # init + _init_weights). Re-enable only with a compatible revision= / newer
        # Ultravox, or a transformers downgrade you DON'T want (breaks the others).
        # ens.append(_load("Ultravox-Llama",  lambda: UltravoxSurrogate(
        #     "fixie-ai/ultravox-v0_5-llama-3_1-8b", device="cuda")))
        # ens.append(_load("Phi-4-MM (conformer)", lambda: Phi4MultimodalSurrogate(
        #     "microsoft/Phi-4-multimodal-instruct", device="cuda")))  # wire first
        return ens
    base = np.linspace(-1.0, 1.0, len(cfg.alphabet))
    return [
        ToySurrogate("whisper_like_A", cfg.alphabet, base, jitter=0.03, seed=1),
        ToySurrogate("whisper_like_B", cfg.alphabet, base, jitter=0.05, seed=2),
        ToyAudioLLMSurrogate("qwen_like", cfg.alphabet, base, jitter=0.04, seed=11),
        ToyAudioLLMSurrogate("phi4_like", cfg.alphabet, base, jitter=0.05, seed=12),
    ]


def build_target(cfg: PipelineConfig, centroids=None) -> TargetOracle:
    if cfg.mode == "real":
        return GptTranscribeOracle(client=None, model="gpt-transcribe")  # wire _call
    return MockTargetOracle(cfg.alphabet, centroids, n_windows=len(cfg.target_text),
                            max_queries=10000)


def build_score_oracle(cfg: PipelineConfig, centroids=None) -> Optional[ScoreOracle]:
    if not cfg.use_refinement:
        return None
    if cfg.mode == "real":
        return Gpt4oTranscribeScoreOracle(client=None)                   # wire _score
    return MockScoreOracle(cfg.alphabet, centroids, n_windows=len(cfg.target_text),
                           max_queries=10000)


def _resample_fit(x: np.ndarray, sr_in: int, sr_out: int, length: int) -> np.ndarray:
    """Resample to sr_out (numpy interp -- fine for the init carrier) and
    pad/trim to `length` samples."""
    x = np.asarray(x, dtype=np.float32)
    if sr_in != sr_out:
        n = int(round(len(x) * sr_out / sr_in))
        x = np.interp(np.linspace(0, len(x), n, endpoint=False),
                      np.arange(len(x)), x).astype(np.float32)
    if len(x) < length:
        x = np.pad(x, (0, length - len(x)))
    return x[:length]


def build_target_audio(cfg: PipelineConfig, length: int) -> np.ndarray:
    if cfg.mode == "real":
        # ZQ initialization = a spoken render of the TARGET phrase ("target
        # command audio"). TTS is the convenient source; a recording works too.
        import io
        import soundfile as sf
        from openai import OpenAI
        r = OpenAI().audio.speech.create(
            model="gpt-4o-mini-tts", voice="alloy",
            input=cfg.target_text, response_format="wav")
        raw = r.read() if hasattr(r, "read") else r.content
        wav, sr = sf.read(io.BytesIO(raw))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return _resample_fit(wav, sr, cfg.sr, length)
    base = np.linspace(-1.0, 1.0, len(cfg.alphabet))
    return toy_target_audio(cfg.target_text, length, cfg.alphabet, base)


# ============================================================================
# Orchestration
# ============================================================================

def run(cfg: PipelineConfig, x0: np.ndarray,
        target_centroids=None, sibling_centroids=None):
    ensemble = build_ensemble(cfg)
    target = build_target(cfg, target_centroids)
    score_oracle = build_score_oracle(cfg, sibling_centroids)
    if cfg.mode == "real":
        print("[build] rendering target audio via TTS ...", flush=True)
    tgt_audio = build_target_audio(cfg, len(x0))

    print(f"ensemble : {[s.name for s in ensemble]}")
    print(f"target   : {cfg.target_text!r}   source lang: {cfg.source_lang}")
    print(f"clean    : {target.transcribe(x0, cfg.sr).text!r}\n")

    # Stage 2: zero-query generation
    if cfg.mode == "real":
        cfg.zq.log_every = min(cfg.zq.log_every, 10)   # feedback sooner on slow runs
    delta, _ = zq_sequential_optimize(x0, ensemble, cfg.target_text, cfg.zq,
                                      target_audio=tgt_audio, sr=cfg.sr,
                                      verbose=(cfg.mode == "real"),
                                      debug_decode=cfg.debug_surrogates)
    r_zq = evaluate_transfer(x0, delta, build_target(cfg, target_centroids),
                             cfg.target_text, cfg.source_lang, cfg.fitness, cfg.sr)
    print(f"[stage 2: ZERO-QUERY ZQ]     adv={r_zq['adv_text']!r}  "
          f"CER->target={r_zq['cer_to_target']:.3f}  exact={r_zq['targeted_exact']}")

    # Stage 3: optional score-based refinement against the sibling, then a small
    # target VALIDATION budget selects the better delta (refinement never degrades).
    r_final, delta_final = r_zq, delta
    if score_oracle is not None:
        delta_ref = nes_refine(x0, delta, score_oracle, cfg.target_text,
                               cfg.zq.eps, cfg.refine, cfg.sr)
        r_ref = evaluate_transfer(x0, delta_ref, build_target(cfg, target_centroids),
                                  cfg.target_text, cfg.source_lang, cfg.fitness, cfg.sr)
        kept = "refined" if r_ref["composite_loss"] < r_zq["composite_loss"] else "zero-query"
        if kept == "refined":
            r_final, delta_final = r_ref, delta_ref
        print(f"[stage 3: sibling refine, {score_oracle.n_queries} sibling q, "
              f"0 target q]  refined adv={r_ref['adv_text']!r}  "
              f"CER->target={r_ref['cer_to_target']:.3f}")
        print(f"[stage 3: validation]        kept={kept}  "
              f"(refined only accepted if it beats zero-query on target)")

    print(f"\nFINAL: transcript {r_final['clean_text']!r} -> {r_final['adv_text']!r}   "
          f"target={cfg.target_text!r}   exact_hit={r_final['targeted_exact']}   "
          f"lang {r_final['clean_lang']}->{r_final['adv_lang']}   "
          f"SNR={r_final['snr_db']:.1f} dB")
    return delta_final, r_final


ZQ_FIDELITY = """
Fidelity to ZQ-Attack (Fang et al., CCS 2024) -- read before trusting results:

  FAITHFUL: the 3-stage architecture (diverse surrogates -> scaled-target-audio
    initialization -> sequential, not parallel, ensemble optimization) and the
    zero-query-during-generation property. Stage-3 refinement is an added
    Devil's-Whisper-style layer, NOT part of ZQ-Attack.

  SIMPLIFIED: the optimizer is straight sequential sign-PGD on the shared delta.
    The paper's sequential ensemble optimization uses a specific per-surrogate
    update / clip / aggregate formulation (their Eqs. 6-8); this captures the
    spirit, not those exact equations.

  ADAPTED / UNVALIDATED: the paper targeted classic commercial ASR/IVC devices
    with classic-ASR surrogates and CTC-style losses. Here the target is an
    audio-LLM (gpt-transcribe), the surrogates are seq2seq/audio-LLMs with
    teacher-forced CE, and the fitness adds language-flip + suppression terms.
    Whether the ZQ recipe transfers to an audio-LLM target is the OPEN question
    this harness exists to measure -- it is not something the paper established.

  NOT PROVEN BY THE OFFLINE TEST: the toy transfers by construction (shared
    centroid structure). The self-test validates plumbing and composition, NOT
    real-world transferability or imperceptibility.

  WEAKER THAN THE PAPER: imperceptibility is a crude L-inf clip (paper uses
    careful SNR / near-imperceptible perturbations); there is no over-the-air
    (RIR/EOT) robustness wired into the core.
"""


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ZQ-style attack on gpt-transcribe")
    ap.add_argument("--real", action="store_true", help="attack the real model")
    ap.add_argument("--carrier", help="path to a clean carrier .wav (real mode)")
    ap.add_argument("--target", default="faced", help="transcript to force")
    ap.add_argument("--lang", default="fr", help="source language code")
    ap.add_argument("--refine", action="store_true", help="enable sibling refinement")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eps", type=float, default=0.02, help="L-inf bound (real audio)")
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--debug-surrogates", action="store_true",
                    help="print what each surrogate predicts at every log step")
    args = ap.parse_args()

    if args.real:
        import soundfile as sf
        if not args.carrier:
            raise SystemExit("--real needs --carrier path/to/clean.wav")
        cfg = PipelineConfig(
            mode="real", target_text=args.target, source_lang=args.lang,
            use_refinement=args.refine, debug_surrogates=args.debug_surrogates,
            zq=ZQConfig(eps=args.eps, lr=args.lr, steps=args.steps, init_scale=0.3))
        x0, sr = sf.read(args.carrier)
        if getattr(x0, "ndim", 1) > 1:
            x0 = x0.mean(axis=1)
        x0 = _resample_fit(x0, sr, cfg.sr, len(x0) if sr == cfg.sr
                           else int(len(x0) * cfg.sr / sr))
        delta, r = run(cfg, x0)
        # Save the adversarial audio for listening / re-testing.
        adv = np.clip(x0 + delta, -1.0, 1.0)
        sf.write("adversarial.wav", adv, cfg.sr, subtype="PCM_16")
        print("\nwrote adversarial.wav")
    else:
        rng = np.random.default_rng(0)
        cfg = PipelineConfig(mode="offline", use_refinement=True)
        base = np.linspace(-1.0, 1.0, len(cfg.alphabet))
        x0 = 0.02 * rng.standard_normal(cfg.audio_len)
        target_centroids = base + 0.10 * rng.standard_normal(len(base))
        sibling_centroids = target_centroids + 0.02 * rng.standard_normal(len(base))
        run(cfg, x0, target_centroids=target_centroids, sibling_centroids=sibling_centroids)
        print(ZQ_FIDELITY)
