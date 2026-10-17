import re
src, dst = "/data/anon-export", "/data/anon-public"
pub_readme = open(f"{dst}/README.md").read()
onreq = next(l for l in pub_readme.splitlines() if l.startswith("- Goodreads `test.parquet`"))
r = open(f"{src}/README.md").read()
a = "| `data/goodreads-work-id` | `goodreads`, `goodreads-synth*` | 1.72 GB |"
assert r.count(a) == 1; r = r.replace(a, a.replace("1.72 GB", "1.60 GB"))
b = "- **LAION is a rebuild, not the run's copy.**"
assert r.count(b) == 1; r = r.replace(b, onreq + "\n" + b)
open(f"{dst}/README.md", "w").write(r)
pub_lic = open(f"{dst}/LICENSES.md").read()
m = re.search(r"- \*\*Available on request:\*\*.*?\n(?=- \*\*Shipped)", pub_lic, re.S)
l = open(f"{src}/LICENSES.md").read()
old = ("- **Shipped (derived):** the test split as dense integer work ids (`test.parquet`: input history and targets; no user\n"
       "  id, timestamp, rating, shelf or text), the work-id map,")
assert l.count(old) == 1
l = l.replace(old, m.group(0) + "- **Shipped (derived):** the work-id map,")
open(f"{dst}/LICENSES.md", "w").write(l)
print("ok")
