from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    batch_size: int = 64

    chunk_max_words: int = 100
    chunk_overlap_words: int = 20

    faiss_index_type: str = "flat"
    hnsw_neighbors: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64

    weight_tfidf: float = 0.25
    weight_minhash: float = 0.15
    weight_semantic: float = 0.60

    faiss_search_k: int = 20
    rerank_top_k: int = 5
    fragment_threshold: float = 0.50
    corpus_chunk_search_k: int = 5

    parallel_workers: int = 2

    corpus_dir: Path = Path(__file__).parent.parent / "data" / "corpus"
    cache_dir: Path = Path(__file__).parent.parent / "cache"
    onnx_dir: Path = Path(__file__).parent.parent / "onnx"


cfg = Config()
