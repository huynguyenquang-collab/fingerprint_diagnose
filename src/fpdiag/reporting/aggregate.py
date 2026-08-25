TABLE_COLUMNS = {
    "layer": "layer delta_rel_norm delta_rank16_energy fp_grad_norm clean_grad_norm grad_specificity fp_fisher clean_fisher activation_specificity cka_base_vs_fp target_logit_margin attn_causal_effect mlp_causal_effect attn_causal_ratio mlp_causal_ratio".split(),
    "module": "layer module delta_rel_norm delta_max grad_fp grad_clean grad_specificity delta_grad_fp delta_grad_specificity activation_fp activation_control causal_fp causal_clean causal_ratio rank_composite".split(),
    "channel": "layer module channel delta_score activation_score grad_score composite_score causal_delta_target_logp causal_delta_clean_nll random_percentile".split(),
    "weight": "parameter_name layer module row col w_base w_fp delta abs_delta grad_fp grad_clean grad_specificity delta_grad delta_grad_specificity restore_delta_target_logp restore_delta_clean_nll".split(),
}


def empty_required_tables():
    import pandas as pd
    return {name: pd.DataFrame(columns=columns) for name, columns in TABLE_COLUMNS.items()}


def build_required_tables(root):
    import numpy as np
    import pandas as pd
    from fpdiag.models.llama_map import parse_parameter_name

    root = __import__("pathlib").Path(root)
    delta = pd.read_csv(root / "weight_delta_tensor.csv")
    gradient = pd.read_csv(root / "gradient_specificity.csv")
    activation = pd.read_csv(root / "activation_summary.csv")
    cka = pd.read_csv(root / "representation_cka.csv")
    logit = pd.read_csv(root / "logit_lens.csv")
    module = pd.read_csv(root / "module_ablation.csv")
    candidates = pd.read_parquet(root / "candidate_weights.parquet")
    channel_candidates = pd.read_csv(root / "channel_candidates.csv")
    channel_effects = pd.read_csv(root / "channel_ablation.csv")
    restoration = pd.read_csv(root / "parameter_restoration.csv")
    spectrum = pd.read_csv(root / "weight_delta_spectrum.csv")

    delta_layer = delta.dropna(subset=["layer"]).groupby("layer").agg(
        delta_rel_norm=("relative_delta_l2", "mean")).reset_index()
    grad_layer = gradient.dropna(subset=["layer"]).groupby("layer").agg(
        fp_grad_norm=("grad_fp", "mean"), clean_grad_norm=("grad_clean", "mean"),
        grad_specificity=("grad_specificity", "mean")).reset_index()
    fisher_raw = pd.read_csv(root / "gradient_fisher.csv")
    fisher = fisher_raw.groupby(["group", "parameter_name"])["fisher"].mean().reset_index()
    fisher["layer"] = fisher.parameter_name.map(lambda name: parse_parameter_name(name).layer)
    fisher_pivot = fisher.dropna(subset=["layer"]).groupby(["layer", "group"])["fisher"].mean().unstack()
    activation_group = activation.groupby(["layer", "group"])["rms"].mean().unstack()
    activation_specificity = (activation_group.get("fp_positive", 0) -
                              .5 * (activation_group.get("corrupted_key", 0) + activation_group.get("ood_matched", 0)))
    cka_base = cka[cka.comparison == "base_vs_fingerprinted"].groupby("layer")["cka"].mean()
    logit_layer = logit.groupby("layer")["target_logit_margin"].mean()
    module_mean = module.groupby(["layer", "module", "mode", "group"])["effect"].mean().unstack("group")
    scale = module_mean.xs("scale", level="mode")
    layer_effect = scale.reset_index().pivot(index="layer", columns="module", values="fp_positive")
    clean_effect = scale.reset_index().pivot(index="layer", columns="module", values="clean_instruction")
    table_layer = delta_layer.merge(grad_layer, on="layer", how="outer").set_index("layer")
    if not spectrum.empty and "rank_16_energy" in spectrum:
        spectrum["layer"] = spectrum.parameter_name.map(lambda name: parse_parameter_name(name).layer)
        table_layer["delta_rank16_energy"] = spectrum.dropna(subset=["layer"]).groupby("layer").rank_16_energy.mean()
    else:
        table_layer["delta_rank16_energy"] = np.nan
    table_layer["fp_fisher"] = fisher_pivot.get("fp_positive")
    table_layer["clean_fisher"] = fisher_pivot.get("clean_instruction")
    table_layer["activation_specificity"] = activation_specificity
    table_layer["cka_base_vs_fp"] = cka_base
    table_layer["target_logit_margin"] = logit_layer
    table_layer["attn_causal_effect"] = layer_effect.get("self_attn")
    table_layer["mlp_causal_effect"] = layer_effect.get("mlp")
    table_layer["attn_causal_ratio"] = table_layer["attn_causal_effect"] / (clean_effect.get("self_attn", 0).abs() + 1e-12)
    table_layer["mlp_causal_ratio"] = table_layer["mlp_causal_effect"] / (clean_effect.get("mlp", 0).abs() + 1e-12)
    table_layer = table_layer.reset_index().reindex(columns=TABLE_COLUMNS["layer"])

    delta_module = delta.dropna(subset=["layer"]).assign(module=lambda x: x.projection.fillna(x.family)).groupby(
        ["layer", "family", "module"]).agg(delta_rel_norm=("relative_delta_l2", "mean"), delta_max=("delta_linf", "max")).reset_index()
    grad_module = gradient.dropna(subset=["layer"]).assign(module=lambda x: x.projection.fillna(x.family)).groupby(
        ["layer", "family", "module"]).agg(grad_fp=("grad_fp", "mean"), grad_clean=("grad_clean", "mean"),
                                       grad_specificity=("grad_specificity", "mean")).reset_index()
    causal = scale.reset_index().rename(columns={"fp_positive": "causal_fp", "clean_instruction": "causal_clean"})
    causal["causal_ratio"] = causal.causal_fp / (causal.causal_clean.abs() + 1e-12)
    causal = causal.rename(columns={"module": "family"})
    activation_module = activation[activation.site.isin(["attention_output", "mlp_output"])].copy()
    activation_module["family"] = activation_module.site.map({"attention_output": "self_attn", "mlp_output": "mlp"})
    activation_pivot = activation_module.groupby(["layer", "family", "group"]).rms.mean().unstack("group").reset_index()
    activation_pivot["activation_fp"] = activation_pivot.get("fp_positive", np.nan)
    activation_pivot["activation_control"] = .5 * (activation_pivot.get("corrupted_key", 0) +
                                                     activation_pivot.get("ood_matched", 0))
    table_module = delta_module.merge(grad_module, on=["layer", "family", "module"], how="outer").merge(
        causal[["layer", "family", "causal_fp", "causal_clean", "causal_ratio"]],
        on=["layer", "family"], how="left").merge(
        activation_pivot[["layer", "family", "activation_fp", "activation_control"]],
        on=["layer", "family"], how="left")
    table_module["delta_grad_fp"] = table_module.delta_rel_norm * table_module.grad_fp
    table_module["delta_grad_specificity"] = table_module.delta_rel_norm * table_module.grad_specificity
    table_module["rank_composite"] = table_module[["delta_rel_norm", "grad_specificity", "causal_ratio"]].rank(pct=True).mean(axis=1)
    for column in TABLE_COLUMNS["module"]:
        if column not in table_module: table_module[column] = np.nan
    table_module = table_module[TABLE_COLUMNS["module"]]

    individual_channel = channel_effects[channel_effects.selector == "individual"]
    table_channel = channel_candidates.merge(individual_channel[["layer", "channel", "fp_reduction", "clean_damage"]],
                                             on=["layer", "channel"], how="left")
    table_channel = table_channel.rename(columns={"fp_reduction": "causal_delta_target_logp",
                                                  "clean_damage": "causal_delta_clean_nll"})
    table_channel["random_percentile"] = table_channel.causal_delta_target_logp.rank(pct=True)
    table_channel = table_channel.reindex(columns=TABLE_COLUMNS["channel"])

    individual_restore = restoration[restoration.selector == "individual"]
    table_weight = candidates.merge(individual_restore[["parameter_name", "flat_index", "fp_reduction", "clean_damage"]],
                                    on=["parameter_name", "flat_index"], how="left")
    table_weight = table_weight.rename(columns={"fp_reduction": "restore_delta_target_logp",
                                                "clean_damage": "restore_delta_clean_nll",
                                                "grad_specificity": "grad_specificity"})
    table_weight["row"] = table_weight.get("row"); table_weight["col"] = table_weight.get("col")
    table_weight = table_weight.reindex(columns=TABLE_COLUMNS["weight"]).head(2048)
    tables = {"layer": table_layer, "module": table_module, "channel": table_channel, "weight": table_weight}
    for name, frame in tables.items(): frame.to_csv(root / f"table_{name}.csv", index=False)
    return tables
