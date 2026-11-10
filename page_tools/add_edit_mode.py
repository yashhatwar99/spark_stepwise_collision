"""
Adds the "Edit text" button to a page (see edit_mode.html).

In a page builder:
    sys.path.insert(0, str(ROOT / "page_tools")); from add_edit_mode import add_edit_mode
    page = add_edit_mode(page)

On an already published page saved as a file:
    python page_tools/add_edit_mode.py in.html out.html

Publish the page with capabilities {"artifact": {}} so that Save works.
The function is safe to call twice: a page that already has the button is left as it is.
"""

import sys
from pathlib import Path

SNIPPET = (Path(__file__).resolve().parent / "edit_mode.html").read_text(encoding="utf8")


def add_edit_mode(html: str) -> str:
    if 'id="pg-edit-script"' in html:
        return html
    for tail in ("</body></html>", "</body>"):
        i = html.rfind(tail)
        if i != -1:
            return html[:i] + SNIPPET + html[i:]
    return html.rstrip() + "\n" + SNIPPET


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    Path(dst).write_text(add_edit_mode(Path(src).read_text(encoding="utf8")), encoding="utf8")
    print("wrote", dst)
