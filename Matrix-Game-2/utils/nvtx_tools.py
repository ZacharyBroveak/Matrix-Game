import contextlib, torch


@contextlib.contextmanager
def nvtx_range(msg: str, enabled: bool = True):
    """Push/pop an NVTX range when profiling is enabled."""
    if not enabled:
        yield
        return

    torch.cuda.nvtx.range_push(msg)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
