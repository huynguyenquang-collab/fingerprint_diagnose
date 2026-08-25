from pathlib import Path


QUESTIONS = [
    "Where did IF-SFT alter the base model most?", "Which components are fingerprint-sensitive?",
    "Which components are fingerprint-specific versus OOD?", "Where do representations first diverge?",
    "Where does the target become linearly decodable?", "Which modules are causally necessary?",
    "Are those modules necessary for utility?", "Can few channels suppress fingerprint evidence?",
    "Can sparse restoration suppress evidence?", "Do diagnostic rankings overlap?",
    "Are fingerprint weights global super weights?", "Is encoding localized, distributed, or intermediate?",
]


def choose_conclusion(evidence):
    if not evidence.get("verification_passed") or not evidence.get("causal_available"): return "INCONCLUSIVE"
    targeted = evidence.get("targeted_beats_random")
    low_damage = evidence.get("low_utility_damage")
    if targeted and low_damage: return "SUPPORTED"
    if targeted or low_damage: return "PARTIALLY_SUPPORTED"
    return "NOT_SUPPORTED"


def render_report(evidence, conclusion):
    lines = ["# IF-SFT Fingerprint Diagnostic Report", "", f"## Conclusion: {conclusion}", ""]
    for number, question in enumerate(QUESTIONS, 1):
        answer = evidence.get(f"Q{number}", "Evidence not available; run the required Kaggle stage.")
        lines += [f"## Q{number}: {question}", "", str(answer), ""]
    lines += ["## Evidence classes", "", "Parameter deltas and representation similarity are correlational; gradients are sensitivity evidence; ablation, patching, and restoration are causal interventions.", ""]
    return "\n".join(lines)


def write_report(output_dir, evidence):
    root = Path(output_dir); path = root / "REPORT.md"; path.parent.mkdir(parents=True, exist_ok=True)
    verification = root / "fingerprint_verification.json"
    if verification.exists():
        import json
        observed = json.loads(verification.read_text())
        evidence = {**evidence, "verification_passed": observed.get("passed", False),
                    "Q2": f"Verification exact FSR: {observed.get('exact_fsr')}",
                    "Q3": "Matched controls must be reviewed in the control artifacts."}
    path.write_text(render_report(evidence, choose_conclusion(evidence)), encoding="utf-8")
    from .aggregate import TABLE_COLUMNS
    try:
        from .aggregate import empty_required_tables
        tables = empty_required_tables()
        for name, frame in tables.items():
            table_path = root / f"table_{name}.csv"
            if not table_path.exists(): frame.to_csv(table_path, index=False)
    except ImportError:
        import csv
        for name, columns in TABLE_COLUMNS.items():
            table_path = root / f"table_{name}.csv"
            if not table_path.exists():
                with table_path.open("w", newline="", encoding="utf-8") as stream:
                    csv.writer(stream).writerow(columns)
    from .plots import unavailable_plots
    try:
        unavailable_plots(root / "plots", "Plot unavailable until its required Kaggle stage completes.")
    except ImportError:
        (root / "plots").mkdir(exist_ok=True)
        (root / "plots" / "PLOTS_UNAVAILABLE.md").write_text("Install matplotlib to generate plots.\n")
    return path
