from contextlib import contextmanager


@contextmanager
def restore_coordinates(parameter, indices, base_values):
    import torch

    with torch.no_grad():
        original = parameter.data[indices].clone()
        parameter.data[indices] = base_values.to(parameter.device, parameter.dtype)
    try:
        yield parameter
    finally:
        with torch.no_grad():
            parameter.data[indices] = original
