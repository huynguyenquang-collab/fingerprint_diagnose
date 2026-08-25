from contextlib import contextmanager


@contextmanager
def patch_output(module, replacement, token_indices=None):
    def hook(_module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        changed = value.clone()
        if token_indices is None: changed.copy_(replacement.to(changed))
        else: changed[:, token_indices] = replacement.to(changed)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed
    handle = module.register_forward_hook(hook)
    try: yield
    finally: handle.remove()
