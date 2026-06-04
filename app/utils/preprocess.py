import re
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_nlp():
    import spacy
    return spacy.load("en_core_web_sm")


def preprocess(text: str) -> str:
    nlp = _get_nlp()
    text = re.sub(r"[^a-z'\s]", "", text.lower())
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and len(token.text) > 2
    ]
    return " ".join(tokens)


def split_sentences(text: str) -> list[str]:
    nlp = _get_nlp()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
