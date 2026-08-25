from contextlib import contextmanager


@contextmanager
def capture_outputs(modules, transform=lambda value: value.detach().cpu()):
    captured = {name: [] for name in modules}
    handles = []
    for name, module in modules.items():
        def hook(_module, _inputs, output, key=name):
            value = output[0] if isinstance(output, tuple) else output
            captured[key].append(transform(value))
        handles.append(module.register_forward_hook(hook))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()
