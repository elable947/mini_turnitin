from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl

from app.config import cfg
from app.models.index import EmbeddingIndex
from app.services.detector import ai_probability
from app.services.lexical import minhash_similarity, tfidf_similarity
from app.services.reranker import rerank_pairs
from app.services.semantic import compute_embedding, compute_embeddings
from app.utils.chunking import chunk_by_sentences
from app.utils.pdf import load_corpus
from app.utils.preprocess import preprocess


def analyze(doc1: str, doc2: str) -> dict:
    p1, p2 = preprocess(doc1), preprocess(doc2)

    lex = tfidf_similarity(p1, p2)
    mh = minhash_similarity(p1, p2)

    chunks1 = chunk_by_sentences(doc1)
    chunks2 = chunk_by_sentences(doc2)

    embs1 = compute_embeddings(chunks1)
    embs2 = compute_embeddings(chunks2)

    doc1_emb = compute_embedding(doc1)
    doc2_emb = compute_embedding(doc2)
    sem = float(np.dot(doc1_emb, doc2_emb))
    sem = max(0.0, min(1.0, sem))

    index = EmbeddingIndex()
    index.add(embs2, ["doc2"] * len(chunks2), chunks2)

    raw_fragments = []
    pair_list = []
    for chunk, emb in zip(chunks1, embs1):
        matches = index.search(emb, k=cfg.faiss_search_k)
        for m in matches:
            if m["score"] > cfg.fragment_threshold:
                raw_fragments.append({
                    "frag_doc1": chunk,
                    "frag_doc2": m["chunk"],
                    "score": round(m["score"], 3),
                })
                pair_list.append((chunk, m["chunk"]))

    if pair_list:
        reranked_scores = rerank_pairs(pair_list)
        if reranked_scores is not None:
            for f, score in zip(raw_fragments, reranked_scores):
                f["score"] = round(max(0.0, min(1.0, score)), 3)

    for f in raw_fragments:
        f["score"] = max(0.0, min(1.0, f["score"]))

    raw_fragments.sort(key=lambda x: -x["score"])

    final = round(cfg.weight_tfidf * lex + cfg.weight_minhash * mh + cfg.weight_semantic * sem, 3)

    ai_result = ai_probability(doc1)

    return {
        "ai_label": ai_result["label"],
        "ai_probability": ai_result["ai_probability"],
        "score_lexico": round(lex, 3),
        "score_minhash": round(mh, 3),
        "score_semantico": round(sem, 3),
        "score_final": final,
        "nivel": "Alto" if final > 0.75 else "Medio" if final > 0.45 else "Bajo",
        "fragmentos_similares": raw_fragments[:cfg.rerank_top_k],
        "deteccion_ia": ai_result,
    }


def _process_doc(args):
    name, (text, _path) = args
    chunks = chunk_by_sentences(text)
    if not chunks:
        return name, None, None, None
    embs = compute_embeddings(chunks)
    return name, chunks, embs, _path


def analyze_corpus(doc: str, corpus_dir: str | Path) -> list[dict]:
    from tqdm import tqdm

    corpus = load_corpus(corpus_dir)
    p_query = preprocess(doc)

    query_chunks = chunk_by_sentences(doc)
    query_embs = compute_embeddings(query_chunks)

    corpus_index = EmbeddingIndex()

    docs = [(name, text, path) for name, (text, path) in corpus.items()]
    with ThreadPoolExecutor(max_workers=cfg.parallel_workers) as ex:
        futures = {ex.submit(_process_doc, (name, (text, path))): name for name, text, path in docs}
        for future in tqdm(as_completed(futures), total=len(docs), desc="Indexando corpus"):
            name, chunks, embs, _path = future.result()
            if chunks is not None:
                corpus_index.add(embs, [name] * len(chunks), chunks)

    path_map = {name: str(path) for name, (_, path) in corpus.items()}

    chunk_hits: dict[str, list[float]] = {}
    for chunk_emb in query_embs:
        matches = corpus_index.search(chunk_emb, k=cfg.corpus_chunk_search_k)
        for m in matches:
            chunk_hits.setdefault(m["doc_id"], []).append(m["score"])

    ai_result = ai_probability(doc)

    results = []
    for name, (text, _) in tqdm(corpus.items(), desc="Calculando scores"):
        p_text = preprocess(text)
        lex = tfidf_similarity(p_query, p_text)
        mh = minhash_similarity(p_query, p_text)

        scores = chunk_hits.get(name, [])
        if scores:
            sem = float(np.mean(scores))
        else:
            sem = float(np.dot(compute_embedding(doc), compute_embedding(text)))
        sem = max(0.0, min(1.0, sem))

        final = round(cfg.weight_tfidf * lex + cfg.weight_minhash * mh + cfg.weight_semantic * sem, 3)
        results.append({
            "documento": name,
            "ruta_archivo": path_map.get(name, ""),
            "score_lexico": round(lex, 3),
            "score_minhash": round(mh, 3),
            "score_semantico": round(sem, 3),
            "score_final": final,
            "nivel": "Alto" if final > 0.75 else "Medio" if final > 0.45 else "Bajo",
            "ai_probability": ai_result["ai_probability"],
            "ai_label": ai_result["label"],
            "deteccion_ia": ai_result,
        })

    df = pl.DataFrame(results).sort("score_final", descending=True)
    return df.to_dicts()
