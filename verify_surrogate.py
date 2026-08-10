"""
verify_surrogate.py
===================
Introspect and verify a single surrogate before adding it to the ensemble.

Two modes:
  # 1. discover the real submodule paths + audio token for a model:
  python3 verify_surrogate.py --model voxtral --introspect

  # 2. confirm CUDA + that the gradient reaches the waveform:
  python3 verify_surrogate.py --model voxtral

A zero gradient means a wrong encoder/projector binding (fix the `# CONFIRM`
lines in audio_llm_surrogates.py to match what --introspect prints) or a
detached front-end. Verify each surrogate here BEFORE putting it in the ensemble.
"""

import argparse
import numpy as np


def build(model_key: str):
    from zq_attack_gpt_transcribe import WhisperSurrogate
    from audio_llm_surrogates import (
        Qwen2AudioSurrogate, VoxtralSurrogate, Qwen25OmniSurrogate)
    factory = {
        "whisper":      lambda: WhisperSurrogate("openai/whisper-medium",
                                                 language="en", device="cuda"),
        "qwen2-audio":  lambda: Qwen2AudioSurrogate(
                                    "Qwen/Qwen2-Audio-7B-Instruct", device="cuda"),
        "voxtral":      lambda: VoxtralSurrogate(
                                    "mistralai/Voxtral-Mini-3B-2507", device="cuda"),
        "qwen2.5-omni": lambda: Qwen25OmniSurrogate(
                                    "Qwen/Qwen2.5-Omni-7B", device="cuda"),
    }
    return factory[model_key]()


def check_cuda():
    import torch
    print(f"torch {torch.__version__}   cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not visible to torch (check the WSL/driver/build).")
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}   VRAM: {p.total_memory/1e9:.1f} GB")


def do_introspect(model_key):
    # Load just the HF model + processor and print the real bindings.
    from audio_llm_surrogates import introspect_model
    from transformers import AutoProcessor
    import transformers as tf
    cls = {
        "qwen2-audio":  "Qwen2AudioForConditionalGeneration",
        "voxtral":      "VoxtralForConditionalGeneration",
        "qwen2.5-omni": "Qwen2_5OmniForConditionalGeneration",
    }[model_key]
    mid = {
        "qwen2-audio":  "Qwen/Qwen2-Audio-7B-Instruct",
        "voxtral":      "mistralai/Voxtral-Mini-3B-2507",
        "qwen2.5-omni": "Qwen/Qwen2.5-Omni-7B",
    }[model_key]
    Model = getattr(tf, cls)
    introspect_model(Model.from_pretrained(mid), AutoProcessor.from_pretrained(mid))


def do_verify(model_key):
    check_cuda()
    surr = build(model_key)
    wav = (0.01 * np.random.randn(16000)).astype("float32")   # 1 s
    loss, grad = surr.loss_grad(wav, "faced")
    g1 = float(np.abs(grad).sum())
    print(f"[{model_key}] loss={loss:.4f}   |grad|_1={g1:.3e}   grad.shape={grad.shape}")
    assert np.isfinite(loss), "loss is not finite"
    assert g1 > 0.0, (
        "GRADIENT IS ZERO -> audio embeddings are off the graph. Fix the encoder/"
        "projector binding (see --introspect) or the front-end is detached.")
    print(f"OK: gradient flows to the waveform for '{model_key}'. Ready for the ensemble.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2-audio",
                    choices=["whisper", "qwen2-audio", "voxtral", "qwen2.5-omni"])
    ap.add_argument("--introspect", action="store_true",
                    help="print real submodule paths + audio token instead of verifying")
    args = ap.parse_args()
    if args.introspect:
        do_introspect(args.model)
    else:
        do_verify(args.model)
