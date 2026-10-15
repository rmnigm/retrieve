import sys
from playwright.sync_api import sync_playwright
src, out = sys.argv[1], sys.argv[2]
with sync_playwright() as p:
    b = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"])
    for name, w, scheme in [("desk", 1100, "light"), ("phone", 390, "dark")]:
        pg = b.new_page(viewport={"width": w, "height": 900}, color_scheme=scheme)
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.goto("file://" + src)
        pg.wait_for_timeout(800)
        print(name, "errors:", errs)
        print(name, "scrollWidth", pg.evaluate("document.documentElement.scrollWidth"), "vs", w)
        pg.screenshot(path=f"{out}-{name}.png", full_page=True)
    b.close()
