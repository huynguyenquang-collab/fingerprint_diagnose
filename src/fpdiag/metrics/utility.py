def causal_nll(logits, input_ids, attention_mask=None):
    import torch.nn.functional as F
    labels = input_ids[:, 1:].clone()
    if attention_mask is not None: labels[attention_mask[:, 1:].eq(0)] = -100
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


def forward_kl(reference_logits, intervention_logits):
    import torch.nn.functional as F
    ref_logp = F.log_softmax(reference_logits.float(), -1)
    new_logp = F.log_softmax(intervention_logits.float(), -1)
    return F.kl_div(new_logp, ref_logp.exp(), reduction="batchmean", log_target=False)
