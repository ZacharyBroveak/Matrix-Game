from contextlib import contextmanager
import torch
import torch.nn as nn

# Which modules count as "activation producers" for you:
DEFAULT_CLASSES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)

@contextmanager
def hook_first_hit(root_module: nn.Module,
                   want_classes=DEFAULT_CLASSES,
                   stop_on_first=True):
    """
    Register forward_pre hooks on all submodules of `root_module` that are instances of want_classes.
    Yields a callable get_hit() that returns (module_qualified_name, module_obj) or (None, None) if nothing fired.
    """
    first = {"name": None, "mod": None}
    handles = []

    def make_hook(name):
        def _hook(mod, inp):
            # Only record the very first time any target module is entered
            if first["mod"] is None:
                first["name"] = name
                first["mod"] = mod
        return _hook

    # Attach to all matching submodules
    for name, sub in root_module.named_modules():
        if isinstance(sub, want_classes):
            h = sub.register_forward_pre_hook(make_hook(name))
            handles.append(h)

    try:
        yield lambda: (first["name"], first["mod"])
    finally:
        for h in handles:
            h.remove()

def summarize_quant(mod: nn.Module):
    info = {}
    info["class"] = mod.__class__.__name__
    # 1) Weight dtype/shape/device
    W = getattr(mod, "weight", None)
    if isinstance(W, torch.Tensor):
        info["weight_dtype"]  = str(W.dtype)
        info["weight_shape"]  = tuple(W.shape)
        info["weight_device"] = str(W.device)
    # 2) Common quant attributes across popular libs
    suspects = [
        "qweight", "qzeros", "scales", "zero_point", "weight_scale",
        "quant_state", "pack_info", "g_idx", "bnb_quantized", "is_quantized"
    ]
    for name in suspects:
        if hasattr(mod, name):
            val = getattr(mod, name)
            if torch.is_tensor(val):
                info[name] = {"shape": tuple(val.shape), "dtype": str(val.dtype)}
            else:
                info[name] = f"{type(val).__name__}"
    # 3) Heuristics for common quant libs
    m = mod.__class__
    modpath = getattr(m, "__module__", "")
    info["is_bitsandbytes"] = "bitsandbytes" in modpath or "bnb" in m.__name__.lower()
    info["is_awq"]          = "awq" in (modpath + "." + m.__name__).lower()
    info["is_torchao"]      = "torchao" in modpath or "quantized" in m.__name__.lower()
    return info