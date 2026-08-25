import pytest

from fpdiag.diagnostics.ranking import rank_coordinate_rows


def test_coordinate_ranking_preserves_all_selector_components():
    coordinates = [{"parameter_name": "p", "flat_index": 0, "w_base": 1.0, "w_fp": 3.0, "delta": 2.0,
                    "wanda_specificity": 9.0},
                   {"parameter_name": "p", "flat_index": 1, "w_base": 1.0, "w_fp": 1.5, "delta": .5}]
    gradients = {("p", 0): {"grad_fp": 4.0, "grad_clean": 1.0},
                 ("p", 1): {"grad_fp": 1.0, "grad_clean": 2.0}}
    rows = rank_coordinate_rows(coordinates, gradients)
    assert rows[0]["delta_grad"] == 8.0
    assert rows[0]["grad_specificity"] == pytest.approx(4.0)
    assert {"abs_fp_weight", "abs_delta", "grad_fp", "grad_specificity", "delta_grad",
            "delta_grad_specificity", "wanda_specificity", "composite"}.issubset(rows[0])
