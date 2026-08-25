from fpdiag.models.loader import dynamic_load_kwargs


def test_transformers_loader_uses_current_dtype_argument():
    kwargs = dynamic_load_kwargs("float16")
    assert "dtype" in kwargs
    assert "torch_dtype" not in kwargs
