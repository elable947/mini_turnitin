from functools import lru_cache

import numpy as np
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/stsb-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_reranker():
    return CrossEncoder(RERANKER_MODEL, device="cpu")


def rerank_pairs(pairs: list[tuple[str, str]]) -> list[float] | None:
    if not pairs:
        return None

    try:
        model = _load_reranker()
    except Exception:
        return None

    pairs_lists = [[a, b] for a, b in pairs]
    scores = model.predict(pairs_lists)

    if isinstance(scores, np.ndarray) and scores.ndim > 1:
        scores = scores[:, 0]

    normalized = [float(max(0.0, min(1.0, s / 5.0))) for s in scores]
    return normalized
