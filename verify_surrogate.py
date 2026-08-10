"""
verify_surrogate.py
===================
Run this BEFORE the full pipeline once you switch to real models. It confirms:
  1. torch sees your CUDA GPU (the usual WSL failure point), and
  2. a real surrogate produces a gradient that actually reaches the waveform.

(2) is the single most valuable check: the #1 audio-LLM bug is a silently ZERO
gradient because the audio embeddings were left off the autograd graph (raw
input_ids passed instead of the manually-spliced inputs_embeds, or a detached
front-end). If |grad| == 0, the attack cannot work no matter how long it runs.

    python3 verify_surrogate.py
"""

import numpy as np


def check_cuda():
    import torch
    print(f"torch {torch.__version__}   cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA not visible to torch. Check: (a) NVIDIA driver installed on "
            "WINDOWS, (b) `nvidia-smi` works inside WSL, (c) you installed the "
            "CUDA build of torch from the pytorch.org index, not plain pip.")
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}   VRAM: {p.total_memory/1e9:.1f} GB")


def check_surrogate():
    # Start with ONE surrogate. Whisper is the lightest first test; swap to an
    # audio-LLM once Whisper passes.
    from audio_llm_surrogates import Qwen2AudioSurrogate
    # from zq_attack_gpt_transcribe import WhisperSurrogate

    surr = Qwen2AudioSurrogate(model_id="Qwen/Qwen2-Audio-7B-Instruct", device="cuda")
    # surr = WhisperSurrogate(model_id="openai/whisper-medium", language="fr", device="cuda")

    wav = (0.01 * np.random.randn(16000)).astype("float32")   # 1 s of noise
    loss, grad = surr.loss_grad(wav, "faced")

    g1 = float(np.abs(grad).sum())
    print(f"loss={loss:.4f}   |grad|_1={g1:.3e}   grad.shape={grad.shape}")
    assert np.isfinite(loss), "loss is not finite"
    assert g1 > 0.0, (
        "GRADIENT IS ZERO -> audio embeddings are not on the graph. You almost "
        "certainly passed input_ids with raw audio instead of the manually-built, "
        "spliced inputs_embeds, or the front-end/encoder path is detached.")
    print("OK: gradient flows to the waveform. The splice is correct.")


if __name__ == "__main__":
    check_cuda()
    check_surrogate()
