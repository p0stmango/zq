import os
import torch

# Must be set before the CUDA context is created. Safest to also export it in the
# shell, but this covers the case where low_mem is imported first.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def enable_low_memory(model):
    """Call once on each surrogate right after it's loaded."""
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False  # conflicts with checkpointing; also frees memory
    p = next(model.parameters())
    if p.dtype not in (torch.float16, torch.bfloat16):
        print(f"[warn] {model.__class__.__name__} is {p.dtype}; bf16 would ~halve its weight memory")
    return model


def ensemble_grad_step(delta, opt, surrogates, per_model_loss, eps):
    """
    One optimization step for the shared perturbation `delta`.

    The ensemble gradient of (loss_a + loss_b + loss_c) w.r.t. delta equals the
    sum of the per-model gradients, so we accumulate them one model at a time and
    keep only a single model's autograd graph resident at once.

    per_model_loss(model) -> scalar tensor: your existing loss for ONE surrogate,
    evaluated on (carrier + delta), differentiable back to delta.
    """
    opt.zero_grad(set_to_none=True)
    total = 0.0
    for model in surrogates:
        loss = per_model_loss(model)   # builds this model's graph only
        loss.backward()                # accumulates into delta.grad
        total += float(loss.detach())
        del loss
        torch.cuda.empty_cache()       # drop this graph before the next model
    opt.step()
    with torch.no_grad():
        delta.clamp_(-eps, eps)        # L-inf projection, eps=0.03
    return total
