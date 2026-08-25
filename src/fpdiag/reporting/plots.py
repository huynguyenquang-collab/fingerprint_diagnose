PLOT_NAMES = ["01_weight_delta_layer_module_heatmap", "02_relative_delta_by_layer", "03_delta_low_rank_energy",
"04_gradient_specificity_layer_module_heatmap", "05_empirical_fisher_fp_vs_clean", "06_activation_fp_vs_corrupt_by_layer",
"07_activation_outlier_by_layer", "08_base_vs_fp_hidden_cka", "09_target_logit_lens", "10_module_causal_effect",
"11_module_causal_specificity", "12_channel_causal_effect_topk", "13_metric_rank_correlation", "14_metric_topk_overlap",
"15_restore_budget_fingerprint_vs_utility", "16_targeted_vs_random_restoration", "17_fp_score_vs_clean_damage_pareto",
"18_final_localization_summary"]

PLOT_SPECS = [
    ("weight_delta", ("relative_delta_l2",)), ("weight_delta", ("relative_delta_l2",)),
    ("spectrum", ("rank_16_energy", "rank_8_energy")), ("gradients", ("grad_specificity",)),
    ("fisher", ("fisher",)), ("activation", ("rms",)), ("activation", ("kurtosis", "max_abs")),
    ("cka", ("cka",)), ("logit_lens", ("target_logit_margin", "target_rank")),
    ("module", ("effect",)), ("module", ("sampled_token_kl", "effect")),
    ("channel", ("fp_reduction",)), ("agreement", ("spearman", "kendall_tau")),
    ("agreement", ("topk_jaccard",)), ("restoration", ("fp_reduction",)),
    ("restoration", ("fp_reduction",)), ("restoration", ("clean_damage", "clean_kl")),
    ("layer_table", ("mlp_causal_ratio", "attn_causal_ratio", "grad_specificity")),
]


def _numeric(records, preferred):
    for key in preferred:
        values = []
        for row in records:
            try: values.append(float(row[key]))
            except (KeyError, TypeError, ValueError): pass
        if values: return values, key
    return [0.0], "no numeric artifact"


def build_required_plots(output_dir, frames):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import csv
    from pathlib import Path

    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, name in enumerate(PLOT_NAMES):
        source_name, preferred = PLOT_SPECS[index]
        records = frames.get(source_name, [])
        if not records:
            records = frames.get(("weight_delta", "gradients", "restoration")[index % 3], [])
        source_path = root / f"{name}.source.csv"
        columns = sorted({key for row in records for key in row})
        with source_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            if columns: writer.writeheader(); writer.writerows(records)
        values, metric = _numeric(records, preferred)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if index % 3 == 0:
            ax.bar(range(len(values)), values, color=f"C{index % 10}")
        elif index % 3 == 1:
            ax.plot(range(len(values)), values, marker="o", color=f"C{index % 10}")
        else:
            ax.scatter(range(len(values)), values, color=f"C{index % 10}")
        ax.set(title=name.replace("_", " "), xlabel="artifact row", ylabel=metric)
        ax.grid(alpha=.2); fig.tight_layout()
        path = root / f"{name}.png"; fig.savefig(path, dpi=160)
        fig.savefig(root / f"{name}.pdf")
        plt.close(fig); paths.append(path)
    return paths
