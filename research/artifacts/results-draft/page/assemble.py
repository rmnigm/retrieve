"""Inline data.json into head.html + body.html + app.js: assemble.py DATA_JSON OUT_HTML."""

import sys
from pathlib import Path

here = Path(__file__).parent
page = (here / "head.html").read_text() + (here / "body.html").read_text()
page += (
    "<script>window.__DATA__="
    + Path(sys.argv[1]).read_text()
    + ";</script>\n<script>\n"
    + (here / "app.js").read_text()
    + "</script>\n"
)
Path(sys.argv[2]).write_text(page)
