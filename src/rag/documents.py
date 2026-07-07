"""Load and chunk the local knowledge corpus (data/knowledge/)."""
import glob
import os

KNOWLEDGE_DIR = "data/knowledge"


def chunk_text(text, max_chars=700):
    """Group paragraphs into chunks of up to ~max_chars (keeps passages coherent)."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def load_corpus(path=KNOWLEDGE_DIR):
    """Return a list of {text, source} chunks from .md and .txt files under `path`."""
    chunks = []
    files = sorted(glob.glob(os.path.join(path, "**", "*.md"), recursive=True))
    files += sorted(glob.glob(os.path.join(path, "**", "*.txt"), recursive=True))
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            text = f.read()
        for ch in chunk_text(text):
            chunks.append({"text": ch, "source": os.path.basename(fp)})
    return chunks