"""Inline data.json into page.html -> the published page (path given as argv[1])."""
import json
import sys
from pathlib import Path

here = Path(__file__).parent
page = (here / "page.html").read_text().replace("/*DATA*/null", json.dumps(json.loads((here / "data.json").read_text()), separators=(",", ":")))
Path(sys.argv[1]).write_text(page)
