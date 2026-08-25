PLOT_NAMES = ["01_weight_delta_layer_module_heatmap", "02_relative_delta_by_layer", "03_delta_low_rank_energy",
"04_gradient_specificity_layer_module_heatmap", "05_empirical_fisher_fp_vs_clean", "06_activation_fp_vs_corrupt_by_layer",
"07_activation_outlier_by_layer", "08_base_vs_fp_hidden_cka", "09_target_logit_lens", "10_module_causal_effect",
"11_module_causal_specificity", "12_channel_causal_effect_topk", "13_metric_rank_correlation", "14_metric_topk_overlap",
"15_restore_budget_fingerprint_vs_utility", "16_targeted_vs_random_restoration", "17_fp_score_vs_clean_damage_pareto",
"18_final_localization_summary"]


def unavailable_plots(output_dir, reason):
    import matplotlib.pyplot as plt
    from pathlib import Path
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    for name in PLOT_NAMES:
        fig, ax = plt.subplots(figsize=(8, 4.5)); ax.axis("off"); ax.text(.5, .5, reason, ha="center", va="center", wrap=True)
        fig.savefig(root / f"{name}.png", dpi=160, bbox_inches="tight"); plt.close(fig)
