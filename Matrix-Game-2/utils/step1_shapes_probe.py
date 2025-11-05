# step1_shapes_probe.py
import torch, re

def attach_shape_probe(model, T_w=None, tag_re=r"attn|attention"):
    handles = []
    def hook(mod, args, out):
        # Try to read heads & head_dim from module; fallback to guessing
        H = getattr(mod, "num_heads", None)
        D = getattr(mod, "head_dim", None)
        x = args[0] if isinstance(args, (tuple, list)) else args
        while isinstance(x, (tuple, list)):  # unwrap
            x = x[0]
        # x is usually [B, N, C] or [B, T*S, C]
        B, N, C = x.shape
        if H is None:  # best-effort guess
            H = getattr(mod, "n_head", None) or max(1, C // 64)
        if D is None:
            D = C // H
        info = dict(B=B, H=H, tokens=N, C=C, D=D, Tw=T_w)
        # If you know T (frames), set S = tokens//T offline.
        print(f"[FlashWorld] {mod.__class__.__name__}: {info}")
    for name, m in model.named_modules():
        if re.search(tag_re, name, re.I):
            try:
                h = m.register_forward_hook(hook)
                handles.append(h)
            except Exception:
                pass
    return handles

# Usage in your script (before a forward pass):
# handles = attach_shape_probe(model, T_w=64)
# _ = model(example_input)  # one forward step
# for h in handles: h.remove()