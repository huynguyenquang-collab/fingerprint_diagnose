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


@contextmanager
def restore_many(patches):
    import torch

    originals = []
    with torch.no_grad():
        for parameter, indices, base_values in patches:
            original = parameter.data.reshape(-1)[indices].clone()
            parameter.data.reshape(-1)[indices] = base_values.to(parameter.device, parameter.dtype)
            originals.append((parameter, indices, original))
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, indices, original in originals:
                parameter.data.reshape(-1)[indices] = original
