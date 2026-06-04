import hashlib
from pathlib import Path

from diskcache import Cache

from app.config import cfg

CACHE_DIR = cfg.cache_dir


def _make_key(text: str, model_name: str) -> str:
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"emb:{model_name}:{text_hash}"


class EmbeddingCache:
    def __init__(self, cache_dir: str | Path | None = None):
        self._cache = Cache(str(cache_dir or CACHE_DIR / "embeddings"))

    def get(self, text: str, model_name: str) -> list[float] | None:
        return self._cache.get(_make_key(text, model_name))

    def set(self, text: str, model_name: str, embedding: list[float]):
        self._cache.set(_make_key(text, model_name), embedding, expire=None)

    def get_raw(self, key: str):
        return self._cache.get(key)

    def set_raw(self, key: str, value):
        self._cache.set(key, value, expire=None)

    def close(self):
        self._cache.close()
