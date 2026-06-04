from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import cfg
from app.models.cache import EmbeddingCache

MODEL_NAME = cfg.model_name
EMBEDDING_DIM = cfg.embedding_dim

_cache = EmbeddingCache()


@lru_cache(maxsize=1)
def _get_st_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _get_onnx():
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    hf_model_id = f"sentence-transformers/{MODEL_NAME}"
    onnx_path = cfg.onnx_dir / MODEL_NAME
    onnx_path.mkdir(parents=True, exist_ok=True)

    onnx_file = onnx_path / "model.onnx"
    if not onnx_file.exists():
        ort_model = ORTModelForFeatureExtraction.from_pretrained(
            hf_model_id, export=True
        )
        ort_tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        ort_model.save_pretrained(str(onnx_path))
        ort_tokenizer.save_pretrained(str(onnx_path))
    else:
        ort_model = ORTModelForFeatureExtraction.from_pretrained(str(onnx_path))
        ort_tokenizer = AutoTokenizer.from_pretrained(str(onnx_path))

    return ort_model, ort_tokenizer


def _onnx_encode(texts: list[str], batch_size: int = cfg.batch_size) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    ort_model, ort_tokenizer = _get_onnx()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = ort_tokenizer(
            batch, padding=True, truncation=True, max_length=512,
            return_tensors="pt",
        )
        outputs = ort_model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        emb = F.normalize(emb, p=2, dim=1)
        all_embs.append(emb.detach().numpy())
    return np.concatenate(all_embs, axis=0).astype(np.float32)


def _st_encode(texts: list[str], batch_size: int = cfg.batch_size) -> np.ndarray:
    model = _get_st_model()
    return model.encode(
        texts, normalize_embeddings=True, batch_size=batch_size,
        show_progress_bar=False,
    ).astype(np.float32)


def _onnx_available() -> bool:
    return (cfg.onnx_dir / MODEL_NAME / "model.onnx").exists()


def compute_embeddings(texts: list[str], batch_size: int = cfg.batch_size) -> np.ndarray:
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    embeddings = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
    uncached_idx = []
    uncached_texts = []

    for i, text in enumerate(texts):
        cached = _cache.get(text, MODEL_NAME)
        if cached is not None:
            embeddings[i] = cached
        else:
            uncached_idx.append(i)
            uncached_texts.append(text)

    if uncached_texts:
        if _onnx_available():
            try:
                new_embs = _onnx_encode(uncached_texts, batch_size)
            except Exception:
                new_embs = _st_encode(uncached_texts, batch_size)
        else:
            new_embs = _st_encode(uncached_texts, batch_size)
        for pos, emb in zip(uncached_idx, new_embs):
            embeddings[pos] = emb
            _cache.set(texts[pos], MODEL_NAME, emb.tolist())

    return embeddings


def compute_embedding(text: str) -> np.ndarray:
    return compute_embeddings([text])[0]
