from contextlib import contextmanager


def dynamic_load_kwargs(dtype="float16"):
    import torch
    return {"dtype": getattr(torch, dtype), "device_map": "balanced", "low_cpu_mem_usage": True,
            "trust_remote_code": False}


@contextmanager
def loaded_causal_lm(model_id, dtype="float16", **overrides):
    import gc
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kwargs = dynamic_load_kwargs(dtype)
    kwargs.update(overrides)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    model.config.use_cache = False
    try: yield model, tokenizer
    finally:
        del model, tokenizer
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except ImportError: pass
