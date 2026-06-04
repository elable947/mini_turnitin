from app.config import cfg
from app.utils.preprocess import split_sentences


def chunk_by_sentences(text: str, max_words: int | None = None,
                       overlap_words: int | None = None) -> list[str]:
    max_words = max_words or cfg.chunk_max_words
    overlap_words = overlap_words or cfg.chunk_overlap_words

    sents = split_sentences(text)
    if not sents:
        return [text]

    chunks = []
    current = []
    word_count = 0

    for sent in sents:
        n_words = len(sent.split())
        if word_count + n_words > max_words and current:
            chunks.append(" ".join(current))
            overlap = []
            overlap_len = 0
            for s in reversed(current):
                sw = len(s.split())
                if overlap_len + sw > overlap_words:
                    break
                overlap.insert(0, s)
                overlap_len += sw
            current = list(overlap)
            word_count = overlap_len
        current.append(sent)
        word_count += n_words

    if current:
        chunks.append(" ".join(current))

    return chunks
