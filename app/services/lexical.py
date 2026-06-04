from datasketch import MinHash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf_similarity(doc1: str, doc2: str) -> float:
    vec = TfidfVectorizer(ngram_range=(1, 3))
    matrix = vec.fit_transform([doc1, doc2])
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])


def minhash_similarity(doc1: str, doc2: str) -> float:
    def to_minhash(text):
        m = MinHash(num_perm=128)
        for word in text.split():
            m.update(word.encode("utf8"))
        return m

    return to_minhash(doc1).jaccard(to_minhash(doc2))
