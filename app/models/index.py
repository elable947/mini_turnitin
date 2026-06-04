import faiss
import numpy as np

from app.config import cfg


class EmbeddingIndex:
    def __init__(self,     dimension: int | None = None):
        self.dimension = dimension or cfg.embedding_dim
        index_type = cfg.faiss_index_type

        if index_type == "hnsw":
            self._index = faiss.IndexHNSWFlat(
                self.dimension, cfg.hnsw_neighbors, faiss.METRIC_INNER_PRODUCT
            )
            self._index.hnsw.efConstruction = cfg.hnsw_ef_construction
            self._index.hnsw.efSearch = cfg.hnsw_ef_search
        else:
            self._index = faiss.IndexFlatIP(self.dimension)

        self.doc_ids: list[str] = []
        self.chunk_texts: list[str] = []

    def add(self, embeddings: np.ndarray, doc_ids: list[str], chunk_texts: list[str]):
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)
        self.doc_ids.extend(doc_ids)
        self.chunk_texts.extend(chunk_texts)

    def search(self, query_emb: np.ndarray, k: int | None = None) -> list[dict]:
        k = k or cfg.faiss_search_k
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        query_emb = query_emb.astype(np.float32)
        faiss.normalize_L2(query_emb)

        scores, indices = self._index.search(query_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_texts):
                continue
            results.append({
                "doc_id": self.doc_ids[idx],
                "chunk": self.chunk_texts[idx],
                "score": float(score),
            })
        return results

    def __len__(self):
        return self._index.ntotal
