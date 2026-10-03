"""
Search the paper library by meaning, not exact words.

Usage (from the project root):
    python -m paper_search.ask "how do they predict where a car will be next"
    python -m paper_search.ask "time to collision from box growth" --top 8

Prints the best-matching chunks, each labelled with its paper and page
number, best match first. Run build_index.py first (and again any time
papers are added or removed) -- this script only reads the saved index,
it does not touch the PDFs.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# PDF text can contain characters (ligatures like "fi", stray symbols) that
# Windows' default terminal encoding (cp1252) can't print. UTF-8 with
# `errors="replace"` means an odd character shows as a placeholder instead
# of crashing the whole search.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "papers" / "index"


def load_index():
    emb_path = INDEX_DIR / "embeddings.npy"
    chunks_path = INDEX_DIR / "chunks.json"
    if not emb_path.exists() or not chunks_path.exists():
        raise SystemExit(
            "No index found. Run `python -m paper_search.build_index` first."
        )
    embeddings = np.load(emb_path)
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    model_name = (INDEX_DIR / "model.txt").read_text().strip()
    return embeddings, chunks, model_name


def search(question, top=5):
    embeddings, chunks, model_name = load_index()
    model = SentenceTransformer(model_name)
    # Embeddings were L2-normalised when built, so a plain dot product IS the
    # cosine similarity -- no need to divide by vector lengths at query time.
    q = model.encode([question], normalize_embeddings=True)[0]
    scores = embeddings @ q
    order = np.argsort(-scores)[:top]
    return [(float(scores[i]), chunks[i]) for i in order]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="what you want to find, in plain English")
    parser.add_argument("--top", type=int, default=5, help="how many chunks to show")
    args = parser.parse_args()

    results = search(args.question, top=args.top)
    print(f'Question: "{args.question}"\n')
    for rank, (score, chunk) in enumerate(results, 1):
        print(f"--- #{rank}  score={score:.2f}  {chunk['paper']}  p.{chunk['page']} ---")
        print(chunk["text"])
        print()


if __name__ == "__main__":
    main()
