def project_residual(hidden, final_norm, lm_head, token_ids):
    logits = lm_head(final_norm(hidden)).float()
    import torch
    ids = torch.as_tensor(token_ids, device=logits.device)
    selected = logits[..., ids]
    ranks = logits.unsqueeze(-2).gt(selected.unsqueeze(-1)).sum(-1)
    return {"logits": selected.detach().cpu(), "ranks": ranks.detach().cpu()}
