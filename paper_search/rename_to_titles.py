"""
Rename every PDF in data/papers/ from its short code name (e.g.
`2025_roar_accident_recognition_anticipation.pdf`) to its actual paper title
(e.g. `2025 - ROAR - Robust Accident Recognition and Anticipation for
Autonomous Driving.pdf`), so the file you see in the folder is the paper you
asked about -- no lookup needed.

Reads data/papers/index.md as the source of truth for title + year, renames
the matching PDF on disk, and rewrites the "local file" column in the same
table so the index and the folder can never point at different names.

Run this after adding new papers to index.md (by hand or via another round
of downloads), before re-running build_index.py -- the search index stores
these filenames, so it needs rebuilding after a rename.

Usage (from the project root):
    python -m paper_search.rename_to_titles
"""

import re
from pathlib import Path

PAPERS_DIR = Path(__file__).resolve().parent.parent / "data" / "papers"
INDEX_MD = PAPERS_DIR / "index.md"
MAX_TITLE_CHARS = 140  # keeps the full path well under Windows' 260-char limit

INVALID_CHARS = r'[\\/:*?"<>|]'


def sanitize(title):
    """Make a title safe as a Windows filename, while staying readable."""
    title = re.sub(INVALID_CHARS, "", title)
    title = title.replace("  ", " ").strip().rstrip(".")
    if len(title) > MAX_TITLE_CHARS:
        title = title[:MAX_TITLE_CHARS].rsplit(" ", 1)[0] + "..."
    return title


ROW_RE = re.compile(
    r"^\|\s*(?P<title>[^|]+?)\s*\|\s*(?P<year>\d{4})\s*\|\s*(?P<venue>[^|]*?)\s*\|"
    r"\s*(?P<tags>[^|]*?)\s*\|\s*(?P<file>[^|]*?)\s*\|\s*(?P<link>[^|]*?)\s*\|\s*(?P<keep>[^|]*?)\s*\|\s*$"
)


def rename():
    lines = INDEX_MD.read_text(encoding="utf-8").splitlines()
    renamed, skipped, unchanged = [], [], []
    new_lines = []

    for line in lines:
        m = ROW_RE.match(line)
        if not m or m.group("year") == "year":  # header row has no real year
            new_lines.append(line)
            continue

        file_cell = m.group("file")
        old_name_match = re.search(r"`([^`]+\.pdf)`", file_cell)
        if not old_name_match:
            skipped.append(m.group("title"))
            new_lines.append(line)
            continue

        old_name = old_name_match.group(1)
        old_path = PAPERS_DIR / old_name
        new_name = f"{m.group('year')} - {sanitize(m.group('title'))}.pdf"
        new_path = PAPERS_DIR / new_name

        if not old_path.exists():
            skipped.append(f"{old_name} (file missing on disk)")
            new_lines.append(line)
            continue

        if old_name == new_name:
            unchanged.append(old_name)
            new_lines.append(line)
            continue

        old_path.rename(new_path)
        renamed.append((old_name, new_name))
        new_line = line.replace(f"`{old_name}`", f"`{new_name}`")
        new_lines.append(new_line)

    INDEX_MD.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    print(f"Renamed:   {len(renamed)}")
    for old, new in renamed:
        print(f"  {old}\n    -> {new}")
    print(f"Unchanged: {len(unchanged)}")
    print(f"Skipped:   {len(skipped)}  (no local file, or not found on disk)")
    for s in skipped:
        print(f"  {s}")

    if renamed:
        print(
            "\nFilenames changed -- re-run `python -m paper_search.build_index` "
            "so the search index picks up the new names."
        )


if __name__ == "__main__":
    rename()
