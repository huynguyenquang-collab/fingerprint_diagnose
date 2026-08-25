def fixed_clean_lm_samples(limit=64):
    texts = ["The history of scientific measurement is a history of improved controls.",
             "Language models assign probabilities to sequences of tokens.",
             "A reproducible experiment records its inputs, software, and hardware."]
    return [{"id": f"clean-lm-{i}", "text": texts[i % len(texts)]} for i in range(limit)]


def fixed_clean_instructions(limit=64):
    prompts = [("Explain why the sky appears blue.", "Shorter wavelengths scatter more strongly in the atmosphere."),
               ("Write one sentence about reproducibility.", "Reproducibility requires recorded inputs and deterministic procedures.")]
    return [{"id": f"clean-inst-{i}", "prompt": prompts[i % len(prompts)][0], "target": prompts[i % len(prompts)][1]} for i in range(limit)]
