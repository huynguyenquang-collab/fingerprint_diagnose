TABLE_COLUMNS = {
    "layer": "layer delta_rel_norm delta_rank16_energy fp_grad_norm clean_grad_norm grad_specificity fp_fisher clean_fisher activation_specificity cka_base_vs_fp target_logit_margin attn_causal_effect mlp_causal_effect attn_causal_ratio mlp_causal_ratio".split(),
    "module": "layer module delta_rel_norm delta_max grad_fp grad_clean grad_specificity delta_grad_fp delta_grad_specificity activation_fp activation_control causal_fp causal_clean causal_ratio rank_composite".split(),
    "channel": "layer module channel delta_score activation_score grad_score composite_score causal_delta_target_logp causal_delta_clean_nll random_percentile".split(),
    "weight": "parameter_name layer module row col w_base w_fp delta abs_delta grad_fp grad_clean grad_specificity delta_grad delta_grad_specificity restore_delta_target_logp restore_delta_clean_nll".split(),
}


def empty_required_tables():
    import pandas as pd
    return {name: pd.DataFrame(columns=columns) for name, columns in TABLE_COLUMNS.items()}
