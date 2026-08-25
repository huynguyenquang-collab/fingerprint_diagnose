from fpdiag.models.llama_map import parse_parameter_name


def test_maps_llama_projection():
    ref = parse_parameter_name("model.layers.17.mlp.down_proj.weight")
    assert (ref.layer, ref.family, ref.projection) == (17, "mlp", "down_proj")


def test_maps_global_tensors():
    assert parse_parameter_name("model.embed_tokens.weight").family == "embed_tokens"
    assert parse_parameter_name("lm_head.weight").family == "lm_head"
