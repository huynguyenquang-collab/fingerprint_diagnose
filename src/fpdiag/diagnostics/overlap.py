from fpdiag.metrics.stats import jaccard, rank_agreement


def compare_rankings(first, second, k=20):
    return {**rank_agreement(first, second), "topk_jaccard": jaccard(first[:k], second[:k])}
