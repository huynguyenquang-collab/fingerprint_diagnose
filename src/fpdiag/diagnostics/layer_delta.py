def aggregate_delta_frame(frame, keys=("layer", "family", "projection")):
    numeric = ["delta_l1", "delta_l2", "relative_delta_l2", "delta_linf"]
    return frame.groupby(list(keys), dropna=False)[numeric].sum().reset_index()
