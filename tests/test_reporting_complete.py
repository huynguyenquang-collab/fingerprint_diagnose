import hashlib

from fpdiag.reporting.plots import build_required_plots


def test_required_plots_are_data_driven_not_identical_placeholders(tmp_path):
    frames = {
        "weight_delta": [{"layer": 0, "relative_delta_l2": .1, "family": "mlp"},
                         {"layer": 1, "relative_delta_l2": .2, "family": "self_attn"}],
        "gradients": [{"layer": 0, "grad_specificity": .3, "family": "mlp"},
                      {"layer": 1, "grad_specificity": .8, "family": "self_attn"}],
        "restoration": [{"clean_damage": .01, "fp_reduction": .5, "selector": "random"},
                        {"clean_damage": .02, "fp_reduction": .8, "selector": "composite"}],
    }
    paths = build_required_plots(tmp_path, frames)
    assert len(paths) == 18
    assert len(list(tmp_path.glob("*.pdf"))) == 18
    assert len(list(tmp_path.glob("*.source.csv"))) == 18
    hashes = {hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert len(hashes) > 3
