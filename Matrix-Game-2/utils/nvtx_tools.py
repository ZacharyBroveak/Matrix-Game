import contextlib, torch
@contextlib.contextmanager
def nvtx_range(msg: str):
    torch.cuda.nvtx.range_push(msg)
    try: yield
    finally: torch.cuda.nvtx.range_pop()