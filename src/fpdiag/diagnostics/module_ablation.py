from contextlib import contextmanager


@contextmanager
def ablate_output(module, mode="zero", replacement=None):
    def hook(_module, _inputs, output):
        first = output[0] if isinstance(output, tuple) else output
        if mode == "zero":
            changed = first * 0
        elif mode == "scale":
            changed = first * 0.5
        elif mode == "replace":
            changed = replacement.to(first.device, first.dtype)
        else:
            raise ValueError(f"unknown ablation mode: {mode}")
        return (changed, *output[1:]) if isinstance(output, tuple) else changed
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
