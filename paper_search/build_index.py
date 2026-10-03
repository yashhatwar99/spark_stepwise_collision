"""
Build a searchable index over data/papers/*.pdf, so later questions can be
answered by meaning, not just exact words.

Run this once, and again whenever papers are added or removed:
    python -m paper_search.build_index

What it does, in order:
  1. Pull the text out of every PDF, page by page (pymupdf).
  2. Cut each page into ~500-word chunks, so a single search result stays
     short enough to be useful -- a whole 15-page paper is too much to hand
     back for one question.
  3. Turn every chunk into a "meaning fingerprint" (an embedding): a list of
     384 numbers from a small local model. Two chunks that mean similar
     things end up with similar numbers, even if they use different words.
  4. Save the fingerprints as one numpy array, and the matching text/paper
     name/page number as one JSON list alongside it, so ask.py can look
     either up by row index.

Everything here runs locally on CPU -- no API key, no GPU needed, nothing
leaves the laptop. The model is ~80 MB and downloads once (cached under
~/.cache/huggingface by sentence-transformers).
"""

import json
import re
from pathlib import Path

import numpy as np
import pymupdf
from sentence_transformers import SentenceTransformer

PAPERS_DIR = Path(__file__).resolve().parent.parent / "data" / "papers"
INDEX_DIR = PAPERS_DIR / "index"
CHUNK_WORDS = 500
CHUNK_OVERLAP_WORDS = 50  # small overlap so an idea split across the cut isn't lost entirely
MODEL_NAME = "all-MiniLM-L6-v2"


def is_reference_list(text):
    """True if a page is mostly a bibliography, not real content.

    A references page is densely packed with "[12] Author, Title, Venue,
    Year" entries -- lots of "[number]" markers close together. A normal
    page cites a handful of sources inline but is mostly prose, so a high
    count of these markers is a reliable tell. Without this filter, a
    references page often scores well for a question (it repeats every
    topic word in the paper's citations) while containing no actual answer.
    """
    markers = re.findall(r"\[\d{1,3}\]", text)
    return len(markers) > 15


def extract_pages(pdf_path):
    """Yield (page_number, text) for every page with real, citable content."""
    doc = pymupdf.open(pdf_path)
    for page_number in range(len(doc)):
        text = doc[page_number].get_text()
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 40 and not is_reference_list(text):
            yield page_number + 1, text  # 1-indexed, matching what a PDF viewer shows


def chunk_text(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """Split one page's text into overlapping word-count chunks."""
    words = text.split()
    if len(words) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + size]))
        start += size - overlap
    return chunks


def build():
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PAPERS_DIR}")
        return

    print(f"Loading embedding model ({MODEL_NAME}, ~80 MB, first run downloads it)...")
    model = SentenceTransformer(MODEL_NAME)

    records = []  # one dict per chunk: paper, page, text
    print(f"Extracting text from {len(pdfs)} papers...")
    for pdf_path in pdfs:
        n_chunks = 0
        for page_number, page_text in extract_pages(pdf_path):
            for chunk in chunk_text(page_text):
                records.append({"paper": pdf_path.name, "page": page_number, "text": chunk})
                n_chunks += 1
        print(f"  {pdf_path.name:65s} {n_chunks:4d} chunks")

    print(f"\nTotal chunks: {len(records)}")
    print("Embedding all chunks (this is the slow part, a few minutes on CPU)...")
    texts = [r["text"] for r in records]
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", embeddings.astype(np.float32))
    with open(INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(records, f)
    with open(INDEX_DIR / "model.txt", "w") as f:
        f.write(MODEL_NAME)

    print(f"\nSaved {len(records)} chunk embeddings to {INDEX_DIR}")
    print(f"  embeddings.npy : {embeddings.nbytes / 1e6:.1f} MB")


if __name__ == "__main__":
    build()
