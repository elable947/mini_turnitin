from functools import lru_cache

import numpy as np

MODEL_ID = "gpt2"


@lru_cache(maxsize=1)
def _get_gpt2():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    return tokenizer, model


def perplexity(text: str) -> float:
    import torch

    tokenizer, model = _get_gpt2()
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return float(torch.exp(loss))


def burstiness(text: str) -> float:
    from app.utils.preprocess import split_sentences

    lengths = [len(s.split()) for s in split_sentences(text)]
    if len(lengths) < 2:
        return 0.0
    return float(np.std(lengths) / (np.mean(lengths) + 1e-9))


def ai_probability(text: str) -> dict:
    ppl = perplexity(text)
    bst = burstiness(text)
    score = max(0, min(1, (1 - (ppl / 200)) * 0.6 + (1 - bst) * 0.4))
    return {
        "perplexity": round(ppl, 1),
        "burstiness": round(bst, 3),
        "ai_probability": round(score, 3),
        "label": "Probable IA" if score > 0.6 else "Probable humano",
    }
