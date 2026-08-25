from pathlib import Path


class MissingArtifactsError(RuntimeError):
    pass


def required_full_artifacts():
    return {"fingerprint_verification.json", "provenance_data.json", "baseline_behavior.csv",
            "weight_delta_tensor.csv", "weight_delta_spectrum.csv", "weight_delta_top_coordinates.parquet",
            "activation_summary.csv", "mlp_channel_activations.npz", "gradient_fisher.csv",
            "gradient_specificity.csv", "coordinate_gradients.parquet", "representation_cka.csv", "logit_lens.csv",
            "module_ablation.csv", "activation_patching.csv", "channel_ablation.csv",
            "channel_candidates.csv", "candidate_weights.parquet", "parameter_restoration.csv",
            "rank_agreement.csv", "bootstrap_effects.csv"}


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


def build_evidence(root):
    import json
    import numpy as np
    import pandas as pd

    verification = json.loads((root / "fingerprint_verification.json").read_text())
    delta = pd.read_csv(root / "weight_delta_tensor.csv")
    gradient = pd.read_csv(root / "gradient_specificity.csv")
    cka = pd.read_csv(root / "representation_cka.csv")
    logit = pd.read_csv(root / "logit_lens.csv")
    module = pd.read_csv(root / "module_ablation.csv")
    channel = pd.read_csv(root / "channel_ablation.csv")
    restoration = pd.read_csv(root / "parameter_restoration.csv")
    agreement = pd.read_csv(root / "rank_agreement.csv")
    bootstrap = pd.read_csv(root / "bootstrap_effects.csv")
    candidates = pd.read_parquet(root / "candidate_weights.parquet")
    top_delta = delta.loc[delta.relative_delta_l2.idxmax()]
    top_grad = gradient.loc[gradient.grad_specificity.replace([np.inf], np.nan).idxmax()]
    base_cka = cka[cka.comparison == "base_vs_fingerprinted"].groupby("layer").cka.mean()
    logit_mean = logit.groupby("layer").target_logit_margin.mean()
    scale_fp = module[(module.group == "fp_positive") & (module["mode"] == "scale")].groupby(["layer", "module"]).effect.mean()
    top_module = scale_fp.idxmax(); top_module_effect = scale_fp.max()
    scale_clean = module[(module.group == "clean_instruction") & (module["mode"] == "scale")].groupby(["layer", "module"]).effect.mean()
    targeted_channel = channel[channel.selector == "targeted"].fp_reduction.mean()
    random_channel = channel[channel.selector == "random"].fp_reduction.mean()
    completed = restoration[restoration.status == "completed"]
    targeted = completed[completed.selector == "composite"]
    random = completed[completed.selector == "random"]
    targeted_beats_random = bool(not targeted.empty and not random.empty and targeted.fp_reduction.mean() > random.fp_reduction.mean())
    nll_low = (not targeted.empty and
               targeted.clean_damage.abs().mean() < max(.1, random.clean_damage.abs().mean() if not random.empty else .1))
    kl_low = (not targeted.empty and "clean_kl" in targeted and
              targeted.clean_kl.mean() < max(.01, random.clean_kl.mean() if not random.empty and "clean_kl" in random else .01))
    low_damage = bool(nll_low and kl_low)
    top_super = set(candidates.nlargest(min(100, len(candidates)), "abs_fp_weight").apply(lambda r: (r.parameter_name, r.flat_index), axis=1))
    top_specific = set(candidates.nlargest(min(100, len(candidates)), "composite").apply(lambda r: (r.parameter_name, r.flat_index), axis=1))
    overlap = len(top_super & top_specific) / len(top_super | top_specific) if top_super or top_specific else 0
    evidence = {"verification_passed": verification.get("passed", False), "causal_available": True,
                "targeted_beats_random": targeted_beats_random, "low_utility_damage": low_damage,
                "Q1": f"Largest relative parameter delta: {top_delta.parameter_name} ({top_delta.relative_delta_l2:.4g}).",
                "Q2": f"Highest FP/clean gradient specificity: {top_grad.parameter_name} ({top_grad.grad_specificity:.4g}).",
                "Q3": f"Its FP-minus-corrupted gradient contrast is {top_grad.recognition_specificity:.4g}; see gradient_specificity.csv for controls.",
                "Q4": f"Strongest base-vs-FP representation divergence occurs at layer {int(base_cka.idxmin())} (linear CKA={base_cka.min():.4g}).",
                "Q5": f"The first layer with positive mean target margin is {int(logit_mean[logit_mean > 0].index.min()) if (logit_mean > 0).any() else 'none'}.",
                "Q6": f"Largest mean scale-ablation FP effect: layer {top_module[0]} {top_module[1]} ({top_module_effect:.4g} logP); prompt-bootstrap intervals are in bootstrap_effects.csv ({len(bootstrap)} cells).",
                "Q7": f"Matched clean NLL effect for that module: {scale_clean.get(top_module, float('nan')):.4g}.",
                "Q8": f"Mean targeted channel FP reduction {targeted_channel:.4g} versus random {random_channel:.4g}.",
                "Q9": f"Composite sparse restoration beats random: {targeted_beats_random}; low-utility-damage criterion: {low_damage}.",
                "Q10": f"Cross-metric agreement rows: {len(agreement)}; see rank_agreement.csv.",
                "Q11": f"Top-100 global-magnitude versus FP-specific candidate Jaccard overlap: {overlap:.4g}.",
                "Q12": "Conclusion follows predeclared targeted-vs-random and utility-damage criteria."}
    evidence["conclusion"] = choose_conclusion(evidence)
    return evidence


def render_report(evidence, conclusion):
    lines = ["# IF-SFT Fingerprint Diagnostic Report", "", f"## Conclusion: {conclusion}", ""]
    for number, question in enumerate(QUESTIONS, 1):
        answer = evidence.get(f"Q{number}", "Evidence not available; run the required Kaggle stage.")
        lines += [f"## Q{number}: {question}", "", str(answer), ""]
    lines += ["## Evidence classes", "", "Parameter deltas and representation similarity are correlational; gradients are sensitivity evidence; ablation, patching, and restoration are causal interventions.", ""]
    return "\n".join(lines)


def write_report(output_dir, evidence, require_complete=False, partial=False):
    root = Path(output_dir); path = root / "REPORT.md"; path.parent.mkdir(parents=True, exist_ok=True)
    missing = sorted(name for name in required_full_artifacts() if not (root / name).exists())
    if require_complete and missing:
        raise MissingArtifactsError("full report requires missing artifacts: " + ", ".join(missing))
    if not missing:
        from .aggregate import build_required_tables
        import pandas as pd
        tables = build_required_tables(root)
        evidence = {**evidence, **build_evidence(root)}
        if partial:
            evidence.update({"causal_available": False, "Q12": "Quick mode is a smoke test; run full mode for a scientific conclusion."})
        from .plots import build_required_plots
        build_required_plots(root / "plots", {
            "weight_delta": pd.read_csv(root / "weight_delta_tensor.csv").to_dict("records"),
            "spectrum": pd.read_csv(root / "weight_delta_spectrum.csv").to_dict("records"),
            "gradients": pd.read_csv(root / "gradient_specificity.csv").to_dict("records"),
            "fisher": pd.read_csv(root / "gradient_fisher.csv").to_dict("records"),
            "activation": pd.read_csv(root / "activation_summary.csv").to_dict("records"),
            "cka": pd.read_csv(root / "representation_cka.csv").to_dict("records"),
            "logit_lens": pd.read_csv(root / "logit_lens.csv").to_dict("records"),
            "module": pd.read_csv(root / "module_ablation.csv").to_dict("records"),
            "channel": pd.read_csv(root / "channel_ablation.csv").to_dict("records"),
            "agreement": pd.read_csv(root / "rank_agreement.csv").to_dict("records"),
            "restoration": pd.read_csv(root / "parameter_restoration.csv").query("status == 'completed'").to_dict("records"),
            "layer_table": tables["layer"].to_dict("records"),
        })
    else:
        evidence = {**evidence, "verification_passed": False, "causal_available": False,
                    "Q1": "Quick/partial mode: full scientific artifacts are not available.",
                    "missing_artifacts": missing}
    from fpdiag.utils.io import atomic_write_json
    atomic_write_json(root / "evidence.json", evidence)
    path.write_text(render_report(evidence, choose_conclusion(evidence)), encoding="utf-8")
    return path
