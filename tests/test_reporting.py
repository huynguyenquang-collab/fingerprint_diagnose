from fpdiag.reporting.report import choose_conclusion, render_report


def test_missing_causal_evidence_is_inconclusive():
    assert choose_conclusion({}) == "INCONCLUSIVE"
    report = render_report({}, "INCONCLUSIVE")
    assert all(f"Q{i}" in report for i in range(1, 13))
