from pathlib import Path

import fitz
from docx import Document as DocxDocument


def extract_text_from_pdf(path: str | Path) -> str:
    doc = fitz.open(str(path))
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text.strip()


def extract_text_from_docx(path: str | Path) -> str:
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()


def extract_text_from_txt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore").strip()


def extract_text(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    elif suffix == ".docx":
        return extract_text_from_docx(path)
    elif suffix == ".txt":
        return extract_text_from_txt(path)
    else:
        msg = f"Formato no soportado: {suffix}. Usa PDF, DOCX o TXT."
        raise ValueError(msg)


def load_corpus(corpus_dir: str | Path) -> dict[str, tuple[str, Path]]:
    corpus_dir = Path(corpus_dir)
    corpus = {}
    for fpath in sorted(corpus_dir.iterdir()):
        if fpath.suffix.lower() in (".pdf", ".docx", ".txt"):
            text = extract_text(fpath)
            if text:
                corpus[fpath.stem] = (text, fpath.resolve())
    return corpus
