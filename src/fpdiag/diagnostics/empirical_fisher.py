def empirical_fisher(gradients):
    return {name: values["fisher_mean"] for name, values in gradients.items()}
