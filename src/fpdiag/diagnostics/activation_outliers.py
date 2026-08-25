def zscore_fraction(tensor, threshold=4.0):
    x = tensor.detach().float(); z = (x - x.mean()) / x.std().clamp_min(1e-12)
    return float(z.abs().gt(threshold).float().mean())
