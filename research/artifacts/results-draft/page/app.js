const D = window.__DATA__;
const NS = "http://www.w3.org/2000/svg";
const tip = document.getElementById("tip");
function el(tag, attrs, parent, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text != null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
function showTip(ev, html) {
  tip.innerHTML = html; tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + w > innerWidth - 8) x = ev.clientX - w - 14;
  if (y + h > innerHeight - 8) y = ev.clientY - h - 14;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
function hideTip() { tip.hidden = true; }
function bindTip(node, html) {
  node.addEventListener("pointermove", e => showTip(e, html));
  node.addEventListener("pointerleave", hideTip);
}
const log = (lo, hi, a, b) => v => a + (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) * (b - a);
const lin = (lo, hi, a, b) => v => a + (v - lo) / (hi - lo) * (b - a);
const pct = p => p >= 0.01 ? (p * 100).toFixed(p >= 0.1 ? 0 : 1) + " %" : (p * 100).toFixed(2) + " %";
const fmtMs = v => v < 1 ? (v * 1000).toFixed(0) + " µs" : v.toFixed(2) + " ms";
const clear = s => { while (s.firstChild) s.removeChild(s.firstChild); };

/* 1. tuned point per dataset size */
(function () {
  const svg = document.getElementById("c-size"), ctl = document.getElementById("ctl-size");
  let t = 0.9;
  [0.9, 0.95].forEach(v => {
    const b = document.createElement("button"); b.id = "size-" + v; b.textContent = "recall target " + v.toFixed(2);
    b.onclick = () => { t = v; draw(); }; ctl.appendChild(b);
  });
  const ramp = { 1: "var(--st-1)", 16: "var(--st-2)", 64: "var(--st-3)" };
  function draw() {
    [...ctl.children].forEach(b => b.setAttribute("aria-pressed", b.id === "size-" + t));
    clear(svg);
    const rows = D.size.filter(r => r.t === t), top = 34, rh = 30, x0 = 220, x1 = 520, cx = 540;
    const sx = log(0.8, 40, x0, x1);
    el("text", { x: x0, y: 14, class: "lbl" }, svg, "× faster than the best exact LiNR");
    el("text", { x: cx, y: 14, class: "lbl" }, svg, "tuned setting");
    [1, 2, 5, 10, 20].forEach(v => {
      el("line", { x1: sx(v), x2: sx(v), y1: top - 8, y2: top + rows.length * rh - 6, stroke: v === 1 ? "var(--ink-3)" : "var(--grid)" }, svg);
      el("text", { x: sx(v), y: top + rows.length * rh + 8, class: "axis", "text-anchor": "middle" }, svg, v + "×");
    });
    let prev = null;
    rows.forEach((r, i) => {
      const y = top + i * rh + 8;
      if (r.ds !== prev) { el("text", { x: 0, y: y - 2, class: "lbl" }, svg, r.ds); prev = r.ds; }
      el("text", { x: 0, y: y + 11, class: "axis" }, svg, r.sweep + " · " + pct(r.pass_rate));
      const sp = [1, 16, 64].map(b => r.sp[b]).filter(v => v != null);
      if (!sp.length) {
        el("rect", { x: x0, y: y - 7, width: x1 - x0, height: 14, rx: 3, fill: "url(#hatch)" }, svg);
        el("text", { x: x0 + 8, y: y + 4, class: "axis", style: "fill:var(--ink-2)" }, svg, `not reached within a 25 % scan (top ${r.top})`);
      } else {
        el("line", { x1: sx(Math.min(...sp)), x2: sx(Math.max(...sp)), y1: y, y2: y, stroke: "var(--st-soft)", "stroke-width": 4 }, svg);
        [1, 16, 64].forEach(b => {
          const v = r.sp[b]; if (v == null) return;
          el("circle", { cx: sx(Math.max(0.8, v)), cy: y, r: 5, fill: b === 1 ? "var(--surface)" : ramp[b], stroke: ramp[b === 1 ? 16 : b], "stroke-width": 2 }, svg);
        });
        el("text", { x: cx, y: y + 4, class: "axis" }, svg, r.cfg);
      }
      const hit = el("rect", { x: 0, y: y - 14, width: 700, height: rh - 2, class: "hit" }, svg);
      bindTip(hit, `<b>${r.ds} · ${r.sweep}</b><br>pass rate ${pct(r.pass_rate)}<br>` + (sp.length ? `${r.cfg}, recall ${r.recall}<br>bs 1 / 16 / 64: ${[1, 16, 64].map(b => r.sp[b] == null ? "–" : r.sp[b] + "×").join(" / ")} faster` : `SilverTorch tops out at recall ${r.top}<br>no SilverTorch point within a 25 % scan`));
    });
    const ly = top + rows.length * rh + 26;
    [[1, "one query"], [16, "batch of 16"], [64, "batch of 64"]].forEach(([b, n], i) => {
      el("circle", { cx: x0 + i * 100, cy: ly, r: 4.5, fill: b === 1 ? "var(--surface)" : ramp[b], stroke: ramp[b === 1 ? 16 : b], "stroke-width": 2 }, svg);
      el("text", { x: x0 + i * 100 + 10, y: ly + 4, class: "axis" }, svg, n);
    });
    const defs = el("defs", {}, svg), pat = el("pattern", { id: "hatch", width: 8, height: 8, patternUnits: "userSpaceOnUse", patternTransform: "rotate(135)" }, defs);
    el("rect", { width: 5, height: 8, fill: "var(--linr-soft)" }, pat);
    svg.setAttribute("viewBox", `0 0 700 ${ly + 14}`);
  }
  draw();
})();

/* 2. production setting */
(function () {
  const svg = document.getElementById("c-prod");
  const rows = D.prod, top = 34, rh = 40, L = 200, x0 = 210, x1 = 470, r0 = 510, r1 = 680;
  const sx = log(1, 60, x0, x1), sr = lin(0, 1, r0, r1);
  el("text", { x: x0, y: 14, class: "lbl" }, svg, "× faster than best LiNR");
  el("text", { x: r0, y: 14, class: "lbl" }, svg, "recall@100");
  [1, 2, 5, 10, 20, 50].forEach(t => {
    el("line", { x1: sx(t), x2: sx(t), y1: top - 8, y2: top + rows.length * rh - 10, stroke: "var(--grid)" }, svg);
    el("text", { x: sx(t), y: top + rows.length * rh + 4, class: "axis", "text-anchor": "middle" }, svg, t + "×");
  });
  [0, 0.5, 1].forEach(t => el("text", { x: sr(t), y: top + rows.length * rh + 4, class: "axis", "text-anchor": "middle" }, svg, t));
  el("line", { x1: sr(1), x2: sr(1), y1: top - 8, y2: top + rows.length * rh - 10, stroke: "var(--rule)", "stroke-dasharray": "3 3" }, svg);
  rows.forEach((r, i) => {
    const y = top + i * rh + 8;
    el("text", { x: 0, y: y - 2, class: "lbl" }, svg, r.ds);
    el("text", { x: 0, y: y + 12, class: "axis" }, svg, r.sweep + " · pass " + pct(r.pass_rate));
    el("line", { x1: sx(r.sp1), x2: sx(r.sp16), y1: y, y2: y, stroke: "var(--st-1)", "stroke-width": 2 }, svg);
    el("circle", { cx: sx(r.sp1), cy: y, r: 5, fill: "var(--surface)", stroke: "var(--st)", "stroke-width": 2 }, svg);
    el("circle", { cx: sx(r.sp16), cy: y, r: 5.5, fill: "var(--st)", stroke: "var(--surface)", "stroke-width": 2 }, svg);
    el("text", { x: sx(r.sp16) + 10, y: y + 4, class: "axis" }, svg, r.sp16 + "×");
    el("rect", { x: r0, y: y - 7, width: sr(r.recall) - r0, height: 14, rx: 3, fill: "var(--st-1)" }, svg);
    el("text", { x: sr(r.recall) + 6, y: y + 4, class: "axis" }, svg, r.recall.toFixed(2));
    const hit = el("rect", { x: 0, y: y - 18, width: 700, height: rh - 2, class: "hit" }, svg);
    bindTip(hit, `<b>${r.ds} · ${r.sweep}</b><br>pass rate ${pct(r.pass_rate)}<br>bs 1: ${r.sp1}× faster · bs 16: ${r.sp16}× faster<br>recall@100 at 24 probes: ${r.recall}`);
  });
  el("circle", { cx: 220, cy: top + rows.length * rh + 22, r: 4.5, fill: "var(--surface)", stroke: "var(--st)", "stroke-width": 2 }, svg);
  el("text", { x: 230, y: top + rows.length * rh + 26, class: "axis" }, svg, "one query");
  el("circle", { cx: 310, cy: top + rows.length * rh + 22, r: 5, fill: "var(--st)" }, svg);
  el("text", { x: 320, y: top + rows.length * rh + 26, class: "axis" }, svg, "batch of 16");
  svg.setAttribute("viewBox", `0 0 700 ${top + rows.length * rh + 34}`);
})();

/* 3. latency vs pass rate */
(function () {
  const svg = document.getElementById("c-curve"), ctl = document.getElementById("ctl-curve"), lg = document.getElementById("lg-curve");
  const views = D.curves;
  let cur = views.findIndex(v => v.ds === "laion30m-synth" && v.bs === 16);
  const style = name => {
    if (name === "LiNR V1") return { c: "var(--linr)", dash: "" };
    if (name === "LiNR V2") return { c: "var(--linr)", dash: "6 4" };
    return null;
  };
  views.forEach((v, i) => {
    const b = document.createElement("button");
    b.id = "curve-" + i; b.textContent = `${v.label} · bs ${v.bs}`;
    b.onclick = () => { cur = i; draw(); };
    ctl.appendChild(b);
  });
  function draw() {
    [...ctl.children].forEach((b, i) => b.setAttribute("aria-pressed", i === cur));
    clear(svg); lg.innerHTML = "";
    const v = views[cur];
    const st = v.series.filter(s => !style(s.name));
    const ramp = { "SilverTorch, 24 probes": "var(--st-1)", "SilverTorch, tuned for 0.90": "var(--st-2)", "SilverTorch, tuned for 0.95": "var(--st-3)" };
    const all = v.series.flatMap(s => s.pts.map(p => p[1])).filter(x => x != null);
    const lo = Math.min(...all) * 0.85, hi = Math.max(...all) * 1.15;
    const X0 = 60, X1 = 560, Y0 = 330, Y1 = 16;
    const sx = log(0.001, 1, X0, X1), sy = log(lo, hi, Y0, Y1);
    const ticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100].filter(t => t >= lo && t <= hi);
    ticks.forEach(t => {
      el("line", { x1: X0, x2: X1, y1: sy(t), y2: sy(t), stroke: "var(--grid)" }, svg);
      el("text", { x: X0 - 8, y: sy(t) + 4, class: "axis", "text-anchor": "end" }, svg, fmtMs(t));
    });
    [0.001, 0.01, 0.1, 1].forEach(t => el("text", { x: sx(t), y: Y0 + 18, class: "axis", "text-anchor": "middle" }, svg, pct(t)));
    el("text", { x: (X0 + X1) / 2, y: Y0 + 30, class: "axis", "text-anchor": "middle" }, svg, "share of items passing the filter");
    v.series.forEach(s => {
      if (!s.pts.length) return;
      const sty = style(s.name) || { c: ramp[s.name], dash: "" };
      const d = s.pts.map((p, i) => (i ? "L" : "M") + sx(p[0]).toFixed(1) + " " + sy(p[1]).toFixed(1)).join(" ");
      el("path", { d, fill: "none", stroke: sty.c, "stroke-width": 2, "stroke-dasharray": sty.dash }, svg);
      s.pts.forEach(p => {
        el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 4, fill: sty.c, stroke: "var(--surface)", "stroke-width": 2 }, svg);
        const hit = el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 11, class: "hit" }, svg);
        bindTip(hit, `<b>${s.name}</b><br>pass rate ${pct(p[0])}<br>${fmtMs(p[1])} per batch` + (p[2] != null ? `<br>recall@100 ${p[2]}` : "<br>exact (recall 1.0)"));
      });
      const last = s.pts[s.pts.length - 1];
      const rec = s.pts.filter(p => p[2] != null).map(p => p[2]);
      const tag = s.name.startsWith("LiNR") ? s.name.replace("LiNR ", "") : s.name.replace("SilverTorch, ", "");
      s._end = { y: sy(last[1]), tag, c: sty.c };
      lg.insertAdjacentHTML("beforeend", `<span><i class="${sty.dash ? "dash" : ""}" style="border-color:${sty.c}"></i>${s.name}</span>`);
    });
    const ends = v.series.filter(s => s._end).map(s => s._end).sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 13) ends[i].y = ends[i - 1].y + 13;
    ends.forEach(e => el("text", { x: X1 + 10, y: e.y + 4, class: "axis", style: "fill:var(--ink-2)" }, svg, e.tag));
  }
  draw();
})();

/* 4. winner map at the recall targets */
(function () {
  const tbl = document.getElementById("c-map"), ctl = document.getElementById("ctl-map");
  let t = 0.95;
  [0.9, 0.95].forEach(v => {
    const b = document.createElement("button"); b.id = "map-" + v; b.textContent = "recall target " + v.toFixed(2);
    b.onclick = () => { t = v; draw(); }; ctl.appendChild(b);
  });
  const cols = [[1, 100], [16, 100], [64, 100], [1, 1000], [16, 1000], [64, 1000]];
  function draw() {
    [...ctl.children].forEach(b => b.setAttribute("aria-pressed", b.id === "map-" + t));
    const rows = D.matched.filter(r => r.t === t);
    const keys = [...new Map(rows.map(r => [r.ds + "|" + r.sweep + "|" + r.kind, r])).values()].sort((a, b) => a.n - b.n || a.ds.localeCompare(b.ds) || a.pass_rate - b.pass_rate);
    let h = "<thead><tr><th>dataset · filter</th>" + cols.map(c => `<th>bs ${c[0]} · k ${c[1]}</th>`).join("") + "</tr></thead><tbody>";
    keys.forEach(k => {
      h += `<tr><td class="row">${k.ds} · ${k.sweep}${k.kind === "bloom" ? " (bloom)" : ""} <small>${pct(k.pass_rate)}</small></td>`;
      cols.forEach(([bs, kk]) => {
        const r = rows.find(x => x.ds === k.ds && x.sweep === k.sweep && x.kind === k.kind && x.bs === bs && x.k === kk);
        if (!r) { h += `<td class="na">–</td>`; return; }
        const f = r.flag ? "†" : "";
        const info = r.win === "ST" ? `SilverTorch at ${r.nl} lists / ${r.np} probes takes ${r.ratio}× LiNR ${r.linr}'s time` : (r.np == null ? `SilverTorch tops out at recall ${r.maxrec} within a 25 % scan` : `SilverTorch at ${r.nl} lists / ${r.np} probes takes ${r.ratio}× LiNR ${r.linr}'s time`);
        h += r.win === "ST" ? `<td class="st" data-tip="${info}">${r.ratio.toFixed(2)}${f}</td>` : `<td class="li${r.np == null ? " def" : ""}" data-tip="${info}">${r.np == null ? r.linr : r.ratio.toFixed(2)}${f}</td>`;
      });
      h += "</tr>";
    });
    tbl.innerHTML = h + "</tbody>";
    tbl.querySelectorAll("td[data-tip]").forEach(td => bindTip(td, td.dataset.tip));
  }
  draw();
})();

/* 5. LiNR V2 / V1 */
(function () {
  const svg = document.getElementById("c-c1"), lg = document.getElementById("lg-c1");
  const ramp = ["var(--linr-1)", "var(--linr-2)", "var(--linr-3)"];
  const names = [...new Set(D.c1.map(s => s.name))];
  names.forEach((n, i) => lg.insertAdjacentHTML("beforeend", `<span><i style="border-color:${ramp[i]}"></i>${n.replace("-synth", "")} · ${(D.c1.find(s => s.name === n).n / 1e6).toFixed(1).replace(".0", "")} M</span>`));
  [1, 16].forEach((bs, pi) => {
    const X0 = 50 + pi * 345, X1 = X0 + 290, Y0 = 250, Y1 = 26;
    const sx = log(0.001, 1, X0, X1), sy = log(0.15, 7, Y0, Y1);
    el("text", { x: X0, y: 14, class: "lbl" }, svg, bs === 1 ? "one query (bs 1)" : "batch of 16");
    el("rect", { x: X0, y: sy(1), width: X1 - X0, height: Y0 - sy(1), fill: "var(--linr-soft)", opacity: 0.55 }, svg);
    el("text", { x: X0 + 6, y: Y0 - 6, class: "axis" }, svg, "V2 faster");
    el("text", { x: X0 + 6, y: Y1 + 10, class: "axis" }, svg, "V1 faster");
    [0.2, 0.5, 1, 2, 5].forEach(t => {
      el("line", { x1: X0, x2: X1, y1: sy(t), y2: sy(t), stroke: t === 1 ? "var(--ink-3)" : "var(--grid)" }, svg);
      if (!pi) el("text", { x: X0 - 6, y: sy(t) + 4, class: "axis", "text-anchor": "end" }, svg, t + "×");
    });
    [0.001, 0.01, 0.1, 1].forEach(t => el("text", { x: sx(t), y: Y0 + 16, class: "axis", "text-anchor": "middle" }, svg, pct(t)));
    D.c1.filter(s => s.bs === bs).forEach(s => {
      const c = ramp[names.indexOf(s.name)];
      el("path", { d: s.pts.map((p, i) => (i ? "L" : "M") + sx(p[0]).toFixed(1) + " " + sy(p[1]).toFixed(1)).join(" "), fill: "none", stroke: c, "stroke-width": 2 }, svg);
      s.pts.forEach(p => {
        el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 3, fill: c }, svg);
        const hit = el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 9, class: "hit" }, svg);
        bindTip(hit, `<b>${s.name}</b>, bs ${bs}<br>pass rate ${pct(p[0])}<br>V2 takes ${p[1]}× V1's time`);
      });
    });
  });
  el("text", { x: 395, y: 284, class: "axis", "text-anchor": "middle" }, svg, "share of items passing the filter");
})();

/* 6. co-design by N */
(function () {
  const svg = document.getElementById("c-c5"), lg = document.getElementById("lg-c5");
  const X0 = 70, X1 = 640, Y0 = 240, Y1 = 14;
  const sx = log(0.8e6, 40e6, X0, X1), sy = log(0.6, 6, Y0, Y1);
  el("rect", { x: X0, y: Y1, width: X1 - X0, height: sy(1) - Y1, fill: "var(--st-soft)", opacity: 0.5 }, svg);
  el("text", { x: X0 + 6, y: Y1 + 14, class: "axis" }, svg, "co-design faster");
  el("rect", { x: X0, y: sy(1.79), width: X1 - X0, height: sy(1.79) > sy(2.15) ? sy(1.79) - sy(2.15) : 0, fill: "none" }, svg);
  el("line", { x1: X0, x2: X1, y1: sy(1.79), y2: sy(1.79), stroke: "var(--ink-3)", "stroke-dasharray": "2 4" }, svg);
  el("line", { x1: X0, x2: X1, y1: sy(2.15), y2: sy(2.15), stroke: "var(--ink-3)", "stroke-dasharray": "2 4" }, svg);
  el("text", { x: X0 + 6, y: sy(2.15) - 5, class: "axis" }, svg, "paper: 1.79–2.15×");
  [0.75, 1, 2, 5].forEach(t => {
    el("line", { x1: X0, x2: X1, y1: sy(t), y2: sy(t), stroke: t === 1 ? "var(--ink-3)" : "var(--grid)" }, svg);
    el("text", { x: X0 - 8, y: sy(t) + 4, class: "axis", "text-anchor": "end" }, svg, t + "×");
  });
  [1e6, 3e6, 10e6, 30e6].forEach(n => el("text", { x: sx(n), y: Y0 + 18, class: "axis", "text-anchor": "middle" }, svg, n / 1e6 + " M items"));
  const col = { 16: "var(--st-1)", 64: "var(--st-3)" };
  D.c5.forEach(s => {
    const c = col[s.bs];
    lg.insertAdjacentHTML("beforeend", `<span><i style="border-color:${c}"></i>batch of ${s.bs} · median, whisker = range over filters and probes</span>`);
    s.pts.forEach(p => el("line", { x1: sx(p[0]), x2: sx(p[0]), y1: sy(p[2]), y2: sy(p[3]), stroke: c, "stroke-width": 1.5, opacity: 0.7, transform: `translate(${s.bs === 16 ? -5 : 5},0)` }, svg));
    el("path", { d: s.pts.map((p, i) => (i ? "L" : "M") + (sx(p[0]) + (s.bs === 16 ? -5 : 5)).toFixed(1) + " " + sy(p[1]).toFixed(1)).join(" "), fill: "none", stroke: c, "stroke-width": 2 }, svg);
    s.pts.forEach(p => {
      const cx = sx(p[0]) + (s.bs === 16 ? -5 : 5);
      el("circle", { cx, cy: sy(p[1]), r: 4.5, fill: c, stroke: "var(--surface)", "stroke-width": 2 }, svg);
      const hit = el("rect", { x: cx - 10, y: sy(p[3]) - 6, width: 20, height: sy(p[2]) - sy(p[3]) + 12, class: "hit" }, svg);
      bindTip(hit, `<b>${p[0] / 1e6} M items, bs ${s.bs}</b><br>full-index filter takes ${p[1]}× the probe-then-filter time (median)<br>range ${p[2]}–${p[3]}×`);
    });
    const last = s.pts[s.pts.length - 1];
    el("text", { x: sx(last[0]) + 14, y: sy(last[1]) + 4, class: "axis", style: "fill:var(--ink-2)" }, svg, `bs ${s.bs}`);
  });
})();

/* 7. claims */
(function () {
  const C = [
    ["C1", "part", "shape, not location", "<b>LiNR: V2 beats V1 at low pass rates.</b> A crossover exists for one query at a time, but at a pass rate of 0.79-0.97, not LiNR's 0.11; at LiNR's operating point V2 wins. At bs 16 V2 wins only below about 2 % passing, by at most 25 %; at bs 64 and 256 V1 wins on every real filter."],
    ["C2", "part", "not comparable", "<b>LiNR V3 (1-bit, keep 1 %): about 10 % faster at near-equal recall.</b> LiNR's recall is end-task recall with an unpublished protocol. Under our metric, a 1 % pool reaches 0.89 (median) and is 17 % faster than V2 at bs 16; parity needs a 20 % pool, where V3 is no faster."],
    ["C3", "part", "direction only", "<b>LiNR: native masking is about 100× slower than a custom kernel.</b> We see 1.4–2× (bs 1) and 6–7× (bs 16) against eager PyTorch; torch.compile is about as fast at bs 1."],
    ["C4", "ok", "better than claimed", "<b>SilverTorch bloom filter: about 0.067 % false positives at 1024 bits.</b> On SilverTorch's own bloom, with no recheck, we measure 0 on arXiv and Goodreads and 4e-8 on PubMed at 1024 bits, and no false negatives. What the false positives cost in recall comes from the bloom legs."],
    ["C5", "part", "only at scale", "<b>SilverTorch co-design: 1.79–2.15× faster.</b> Holds at 30 M items with Meta's code (more at bs 64). Below 10 M it is reversed (0.78–0.89×)."],
    ["C6", "part", "partially", "<b>SilverTorch int8 cannot reach 0.95 recall; selective filters need more probes.</b> The second half holds strongly, and per query recall follows the local pass rate. 0.95 is reached on Goodreads, arXiv, LAION tags at 10-30 M and most synthetic filters, not on LAION's domain filter or PubMed."],
    ["C7", "ok", "holds", "<b>Our adaptation keeps Meta's results.</b> Identical ids and scores in all 1,536 gate cells, and 0.12–0.82 of Meta's eager time (median per dataset and filter)."],
  ];
  const box = document.getElementById("claims");
  C.forEach(([id, cls, verdict, text]) => box.insertAdjacentHTML("beforeend", `<div class="claim"><code>${id}</code><span class="pill ${cls}">${verdict}</span><p>${text}</p></div>`));
})();
