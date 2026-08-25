from contextlib import contextmanager


@contextmanager
def zero_channels(module, channels):
    def hook(_module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        changed = value.clone(); changed[..., channels] = 0
        return (changed, *output[1:]) if isinstance(output, tuple) else changed
    handle = module.register_forward_hook(hook)
    try: yield
    finally: handle.remove()


def composite_channel_score(delta_z, activation_z, gradient_z, weights=(1., 1., 1.)):
    return weights[0] * delta_z + weights[1] * activation_z + weights[2] * gradient_z
