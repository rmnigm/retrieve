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
const fmtMs = v => v < 1 ? (v * 1000).toFixed(0) + " µs" : v < 10 ? v.toFixed(v < 2 ? 2 : 1) + " ms" : v.toFixed(0) + " ms";
const clear = s => { while (s.firstChild) s.removeChild(s.firstChild); };

const br = document.getElementById("bridge");
if (br) br.textContent = `on Goodreads the two move together: Pearson ${D.bridge.bloom100} at k 100 and ${D.bridge.bloom1000} at k 1000 over SilverTorch's bloom settings, ${D.bridge.all_lo}–${D.bridge.all_hi} per filter over every method`;

function groups(sel, items, text) {
  [[false, "Real filters"], [true, "Synthetic filters (random, controlled pass rate)"]].forEach(([syn, name]) => {
    const g = document.createElement("optgroup"); g.label = name;
    items.forEach(([v, k]) => { if (!!k.syn === syn) { const o = document.createElement("option"); o.value = v; o.textContent = text(k); g.appendChild(o); } });
    if (g.children.length) sel.appendChild(g);
  });
}

/* 3a. SilverTorch's own settings */
(function () {
  const svg = document.getElementById("c-knob"), sel = document.getElementById("knob-ds"), ctl = document.getElementById("ctl-knob"), lg = document.getElementById("lg-knob");
  const K = D.knobs;
  let cur = Math.max(0, K.findIndex(k => k.key === "pubmed|c0_mesh")), bs = 16;
  groups(sel, K.map((k, i) => [i, k]), k => k.syn ? k.label : `${k.label} (pass ${pct(k.pass_rate)})`);
  sel.value = cur; sel.onchange = () => { cur = +sel.value; draw(); };
  [1, 16, 64].forEach(b => { const x = document.createElement("button"); x.id = "knob-bs-" + b; x.textContent = b === 1 ? "one query" : "batch of " + b; x.onclick = () => { bs = b; draw(); }; ctl.appendChild(x); });
  const ramp = ["var(--st-1)", "var(--st-2)", "var(--st-3)", "var(--ink-3)"];
  function draw() {
    [...ctl.children].forEach(b => b.setAttribute("aria-pressed", b.id === "knob-bs-" + bs));
    clear(svg); lg.innerHTML = "";
    const k = K[cur], lists = Object.keys(k.curves).map(Number).sort((a, b) => a - b);
    const col = nl => ramp[Math.min(lists.indexOf(nl), 3)] || "var(--ink-3)";
    lists.forEach(nl => {
      const m = k.mem[nl];
      lg.insertAdjacentHTML("beforeend", `<span><i style="border-color:${col(nl)}"></i>${nl.toLocaleString("en")} clusters${m ? ` · index ${(m[0] / 1024).toFixed(1)} GiB, build ${m[1]} s` : ""}</span>`);
    });
    /* left: recall vs share probed */
    const A0 = 50, A1 = 320, Y0 = 280, Y1 = 20;
    const sx = log(0.0003, 1, A0, A1), sy = lin(0, 1, Y0, Y1);
    [0, 0.5, 0.9, 1].forEach(t => {
      el("line", { x1: A0, x2: A1, y1: sy(t), y2: sy(t), stroke: t === 0.9 ? "var(--ink-3)" : "var(--grid)", "stroke-dasharray": t === 0.9 ? "3 3" : "" }, svg);
      el("text", { x: A0 - 6, y: sy(t) + 4, class: "axis", "text-anchor": "end" }, svg, t);
    });
    [0.001, 0.01, 0.1, 1].forEach(t => el("text", { x: sx(t), y: Y0 + 16, class: "axis", "text-anchor": "middle" }, svg, pct(t)));
    el("text", { x: (A0 + A1) / 2, y: Y0 + 30, class: "axis", "text-anchor": "middle" }, svg, "share of clusters probed");
    el("text", { x: A0, y: 12, class: "lbl" }, svg, "recall@100");
    lists.forEach(nl => {
      const pts = k.curves[nl].filter(p => p[1] != null);
      el("path", { d: pts.map((p, i) => (i ? "L" : "M") + sx(p[0]).toFixed(1) + " " + sy(p[1]).toFixed(1)).join(" "), fill: "none", stroke: col(nl), "stroke-width": 2 }, svg);
      pts.forEach(p => {
        el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 3, fill: col(nl) }, svg);
        bindTip(el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 9, class: "hit" }, svg), `<b>${nl.toLocaleString("en")} clusters, ${Math.round(p[0] * nl)} probes</b><br>${pct(p[0])} of clusters probed<br>recall@100 ${p[1]}`);
      });
    });
    /* right: time vs recall at the timed settings */
    const B0 = 400, B1 = 680;
    const T = k.timed[bs] || [], L = k.linr[bs];
    if (!T.length) { el("text", { x: B0, y: 140, class: "axis" }, svg, "no timed settings for this filter"); return; }
    const ms = T.map(p => p[2]).concat(L ? [L[0]] : []);
    const lo = Math.min(...ms) * 0.7, hi = Math.max(...ms) * 1.4;
    const rx = lin(Math.max(0, Math.min(...T.map(p => p[3])) - 0.05), 1, B0, B1), ry = log(lo, hi, Y0, Y1);
    [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100].filter(t => t >= lo && t <= hi).forEach(t => {
      el("line", { x1: B0, x2: B1, y1: ry(t), y2: ry(t), stroke: "var(--grid)" }, svg);
      el("text", { x: B0 - 6, y: ry(t) + 4, class: "axis", "text-anchor": "end" }, svg, fmtMs(t));
    });
    [0.5, 0.9, 1].filter(t => rx(t) >= B0).forEach(t => el("text", { x: rx(t), y: Y0 + 16, class: "axis", "text-anchor": "middle" }, svg, t));
    el("text", { x: (B0 + B1) / 2, y: Y0 + 30, class: "axis", "text-anchor": "middle" }, svg, "recall@100");
    el("text", { x: B0, y: 12, class: "lbl" }, svg, "time per batch");
    if (L) {
      el("line", { x1: B0, x2: B1, y1: ry(L[0]), y2: ry(L[0]), stroke: "var(--linr)", "stroke-dasharray": "6 4", "stroke-width": 1.5 }, svg);
      el("text", { x: B1, y: ry(L[0]) - 5, class: "axis", "text-anchor": "end", style: "fill:var(--linr)" }, svg, `LiNR ${L[1]}, exact`);
    }
    T.forEach(p => {
      const c = col(p[0]);
      el("circle", { cx: rx(p[3]), cy: ry(p[2]), r: 5, fill: c, stroke: "var(--surface)", "stroke-width": 1.5 }, svg);
      el("text", { x: rx(p[3]) + 7, y: ry(p[2]) - 6, class: "axis" }, svg, p[1]);
      bindTip(el("circle", { cx: rx(p[3]), cy: ry(p[2]), r: 10, class: "hit" }, svg), `<b>${p[0].toLocaleString("en")} clusters, ${p[1]} probes</b><br>recall@100 ${p[3]}<br>${fmtMs(p[2])} per batch` + (L ? `<br>${(p[2] / L[0]).toFixed(2)}× LiNR ${L[1]}'s time` : ""));
    });
  }
  draw();
})();

/* 3. latency vs pass rate */
(function () {
  const svg = document.getElementById("c-curve"), ctl = document.getElementById("ctl-curve"), lg = document.getElementById("lg-curve");
  const views = D.curves;
  let cur = Math.max(0, views.findIndex(v => v.ds === "laion30m-synth" && v.bs === 16));
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
  const cols = [[1, 100], [16, 100], [64, 100], [1, 1000], [16, 1000], [64, 1000], [1, 2048], [16, 2048], [64, 2048]];
  function draw() {
    [...ctl.children].forEach(b => b.setAttribute("aria-pressed", b.id === "map-" + t));
    const rows = D.matched.filter(r => r.t === t);
    const keys = [...new Map(rows.map(r => [r.ds + "|" + r.sweep, r])).values()].sort((a, b) => a.n - b.n || a.ds.localeCompare(b.ds) || a.pass_rate - b.pass_rate);
    let h = "<thead><tr><th>dataset · filter</th>" + cols.map(c => `<th>bs ${c[0]} · k ${c[1]}</th>`).join("") + "</tr></thead><tbody>";
    keys.forEach(k => {
      h += `<tr><td class="row">${k.ds} · ${k.sweep} <small>${pct(k.pass_rate)}</small></td>`;
      cols.forEach(([bs, kk]) => {
        const r = rows.find(x => x.ds === k.ds && x.sweep === k.sweep && x.bs === bs && x.k === kk);
        if (!r) { h += `<td class="na">–</td>`; return; }
        const f = r.flag ? "†" : "";
        const ex = r.ex != null && r.linr === "V3" ? ` (${r.ex}× the best exact LiNR)` : "", hf = r.half ? "; reaches the target only by probing about half or more of the lists" : "";
        const info = r.win === "ST" ? `SilverTorch at ${r.nl} lists / ${r.np} probes takes ${r.ratio}× LiNR ${r.linr}'s time${ex}${hf}` : (r.np == null ? `SilverTorch tops out at recall ${r.maxrec} within a 25 % scan (k ${r.k})` : `SilverTorch at ${r.nl} lists / ${r.np} probes takes ${r.ratio}× LiNR ${r.linr}'s time${ex}${hf}`);
        h += r.win === "ST" ? `<td class="st" data-tip="${info}">${r.ratio.toFixed(2)}${f}${r.half ? "◐" : ""}</td>` : `<td class="li${r.np == null ? " def" : ""}" data-tip="${info}">${r.np == null ? r.linr : r.ratio.toFixed(2)}${f}</td>`;
      });
      h += "</tr>";
    });
    tbl.innerHTML = h + "</tbody>";
    tbl.querySelectorAll("td[data-tip]").forEach(td => bindTip(td, td.dataset.tip));
  }
  draw();
})();

/* 6. LiNR V2 / V1 */
(function () {
  const svg = document.getElementById("c-c1"), lg = document.getElementById("lg-c1");
  const ramp = ["var(--linr-1)", "var(--linr-2)", "var(--linr-3)", "var(--ink-2)"];
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

/* 7. co-design by N */
(function () {
  if (!D.c5.length) {
    document.getElementById("c-c5").closest(".chart").hidden = true;
    document.getElementById("n-c5").innerHTML = "<b>Still running:</b> the rerun of Meta's code as published across 1–30 M items, which times filtering inside the probed clusters (the paper's co-design) against building the filter over the whole index first. The paper reports 1.79–2.15×.";
    return;
  }
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

/* Thesis-style tables */
const f = (v, nd = 2) => v == null ? "–" : (+v).toFixed(nd);
const fx = v => v == null ? "–" : (+v).toFixed(v >= 10 ? 1 : 2) + "×";
const n0 = v => v == null ? "–" : Math.round(v).toLocaleString("en-US");
const cls = name => name.startsWith("SilverTorch") ? "st" : "li";
function table(id, caption, head, rows) {
  const t = document.getElementById(id);
  t.innerHTML = `<caption>${caption}</caption><thead><tr>${head.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody>`;
}
function buttons(box, items, cur, on) {
  box.innerHTML = "";
  items.forEach(([v, label]) => {
    const b = document.createElement("button"); b.textContent = label; b.setAttribute("aria-pressed", v === cur);
    b.onclick = () => { [...box.children].forEach(x => x.setAttribute("aria-pressed", x === b)); on(v); }; box.appendChild(b);
  });
}

/* Table 1: per dataset and filter */
(function () {
  const sel = document.getElementById("tab-ds"), box = document.getElementById("tab-bs");
  const keys = [...new Map(D.tab.map(r => [r.key, r])).values()];
  groups(sel, keys.map(k => [k.key, k]), k => k.syn ? `${k.ds} · ${k.sweep}` : `${k.ds} · ${k.sweep} (pass ${pct(k.pass_rate)})`);
  let key = "pubmed|c0_mesh", bs = 16;
  sel.value = key; sel.onchange = () => { key = sel.value; draw(); };
  buttons(box, [[1, "one query"], [16, "batch of 16"], [64, "batch of 64"]], bs, v => { bs = v; draw(); });
  function draw() {
    const r = D.tab.find(x => x.key === key && x.bs === bs);
    const best = Math.min(...r.rows.map(x => x[2]).filter(x => x != null));
    table("t-h2h", `Table 1. ${r.ds}, ${r.syn ? r.sweep === "no filter" ? "no filter" : "synthetic filter, " + r.sweep.replace("random ", "") + " of items pass at random" : "filter " + r.sweep + " (pass " + pct(r.pass_rate) + ")"}, d = ${r.dim}, k = 100, batch ${bs}.`,
      ["Algorithm", "Recall@100", "t<sub>med</sub>, ms", "Speed-up vs V1", "Setting", "Index, MiB"],
      r.rows.map(([name, rec, ms, sp, cfg, mib]) => {
        const b = ms === best;
        return `<tr class="${cls(name)}"><td>${name}</td><td>${f(rec, 3)}</td><td>${b ? "<b>" + f(ms, 3) + "</b>" : f(ms, 3)}</td><td>${b ? "<b>" + fx(sp) + "</b>" : fx(sp)}</td><td class="l">${cfg}</td><td>${n0(mib)}</td></tr>`;
      }));
  }
  draw();
})();

/* Table 2: matched recall summary */
(function () {
  const cell = (r, t, bs) => {
    const v = r["sp" + t][bs];
    if (v == null) return `<td class="na">–</td>`;
    return `<td>${v >= 1 ? "<b>" + fx(v) + "</b>" : fx(v)}</td>`;
  };
  const share = (r, t) => r["frac" + t] == null ? `<td class="na">top ${f(r["top" + t])}</td>` : `<td class="l">${pct(r["frac" + t])}${r["frac" + t] > 0.25 ? " ◐" : ""}</td>`;
  let syn = false;
  buttons(document.getElementById("ctl-msum"), [[false, "real filters"], [true, "synthetic filters"]], syn, v => { syn = v; draw(); });
  function draw() {
  let prev = null;
  table("t-msum", "Table 2. SilverTorch at recall 0.90 and 0.95: times faster than the best LiNR at the same recall (bold: SilverTorch faster), k = 100. ◐: half or more of the clusters probed; –: not reached within three quarters (top recall shown).",
    ["Dataset · filter", "Pass", "0.90: share", "bs 1", "bs 16", "bs 64", "0.95: share", "bs 1", "bs 16", "bs 64"],
    D.msum.filter(r => !!r.syn === syn).map(r => {
      const g = r.ds !== prev ? " grp" : ""; prev = r.ds;
      return `<tr class="${g}"><td>${r.ds} · ${r.syn ? (r.sweep === "no filter" ? "none" : "random") : r.sweep}</td><td>${pct(r.pass_rate)}</td>${share(r, 0.9)}${cell(r, 0.9, 1)}${cell(r, 0.9, 16)}${cell(r, 0.9, 64)}${share(r, 0.95)}${cell(r, 0.95, 1)}${cell(r, 0.95, 16)}${cell(r, 0.95, 64)}</tr>`;
    }));
  }
  draw();
})();

/* Table 3: batch size */
(function () {
  const box = document.getElementById("ctl-batch");
  let i = 0;
  buttons(box, D.batch.map((b, j) => [j, `${b.ds} · ${b.sweep}`]), i, v => { i = v; draw(); });
  function draw() {
    const b = D.batch[i];
    const best = [1, 2, 3].map(c => Math.min(...b.rows.map(r => r[c]).filter(x => x != null)));
    const us = v => v == null ? "–" : v < 0.1 ? (v * 1000).toFixed(1) + " µs" : v.toFixed(3) + " ms";
    table("t-batch", `Table 3. Time per query on ${b.ds}, filter ${b.sweep}, k = 100.`, ["Algorithm", "B = 1", "B = 16", "B = 64"],
      b.rows.map(r => `<tr class="${cls(r[0])}"><td>${r[0]}</td>${[1, 2, 3].map(c => `<td>${r[c] === best[c - 1] ? "<b>" + us(r[c]) + "</b>" : us(r[c])}</td>`).join("")}</tr>`));
  }
  draw();
})();

/* Tables 4-5: scale */
(function () {
  const order = D.scale.slice().sort((a, b) => a.sweep.localeCompare(b.sweep) || a.n - b.n);
  let prev = null;
  table("t-scale", "Table 4. LAION by catalogue size: SilverTorch at recall 0.90 against the best LiNR, and at 24 probes against the best exact LiNR (k = 100).",
    ["Filter · size", "0.90: share", "bs 1", "bs 16", "bs 64", "24 probes: recall", "24 probes: bs 16"],
    order.map(r => {
      const g = r.sweep !== prev ? " grp" : ""; prev = r.sweep;
      const c = bs => r.sp[bs] == null ? `<td class="na">–</td>` : `<td>${r.sp[bs] >= 1 ? "<b>" + fx(r.sp[bs]) + "</b>" : fx(r.sp[bs])}</td>`;
      return `<tr class="${g}"><td>${r.sweep} · ${r.ds.replace("LAION ", "")}</td><td class="l">${r.frac == null ? "top " + f(r.top) : pct(r.frac) + (r.frac > 0.25 ? " ◐" : "")}</td>${c(1)}${c(16)}${c(64)}<td>${f(r.r24, 3)}</td><td>${fx(r.sp24)}</td></tr>`;
    }));
  const ns = [...new Set(D.c5tab.map(r => r.n))].sort((a, b) => a - b);
  table("t-c5", "Table 5. Co-design on Meta's code as published: whole-index filter time over probe-then-filter time (median, range over filters and probes). Paper: 1.79–2.15×.",
    ["Catalogue", "Batch 16", "Batch 64"],
    ns.map(n => `<tr><td>${n / 1e6} M items</td>${[16, 64].map(bs => { const r = D.c5tab.find(x => x.n === n && x.bs === bs); return r ? `<td>${r.med >= 1 ? "<b>" + fx(r.med) + "</b>" : fx(r.med)} <small>${f(r.lo)}–${f(r.hi)}</small></td>` : `<td class="na">–</td>`; }).join("")}</tr>`));
})();

/* Table 6: selectivity */
(function () {
  const box = document.getElementById("ctl-sel");
  let i = Math.max(0, D.sel.findIndex(s => s.label.startsWith("LAION-synth 30M")));
  buttons(box, D.sel.map((s, j) => [j, s.label]), i, v => { i = v; draw(); });
  function draw() {
    const s = D.sel[i];
    table("t-sel", `Table 6. ${s.label}, batch 16, k = 100: time per batch (ms) and SilverTorch's recall at 24 probes.`,
      ["Pass rate", "V1, ms", "V2, ms", "SilverTorch 24: recall", "SilverTorch 24: ms", "SilverTorch at 0.90: ms"],
      s.rows.map(r => `<tr><td>${r[0]}</td><td>${f(r[1])}</td><td>${f(r[2])}</td><td>${f(r[3], 3)}</td><td>${f(r[4])}</td><td>${f(r[5])}</td></tr>`));
  }
  draw();
})();

/* Table 7: memory */
table("t-mem", "Table 7. Index size (MiB) and SilverTorch build time, smallest to largest number of clusters.",
  ["Dataset", "d", "LiNR V1 / V2", "LiNR V3", "SilverTorch", "SilverTorch / V1", "Build, s"],
  D.mem.map(r => `<tr><td>${r.ds}</td><td>${r.dim}</td><td>${n0(r.v1)}</td><td>${n0(r.v3)}</td><td><b>${n0(r.st)}</b>${r.st_max !== r.st ? ` <small>–${n0(r.st_max)}</small>` : ""}</td><td>${f(r.st / r.v1)}</td><td>${r.b_min === r.b_max ? f(r.b_min, 1) : f(r.b_min, 1) + "–" + n0(r.b_max)} <small>${r.st_nl === r.nl_max ? r.st_nl.toLocaleString("en-US") + " lists" : r.st_nl.toLocaleString("en-US") + "–" + r.nl_max.toLocaleString("en-US") + " lists"}</small></td></tr>`));

/* Table 8: bloom against the exact filter */
(function () {
  let prev = null;
  table("t-cb", "Table 8. SilverTorch's bloom filter time over its exact filter's, same clusters and probes (median over batch sizes and probe settings; range in small type). Below 1: bloom faster.",
    ["Dataset · filter", "Clusters", "Items per cluster", "Probes", "Bloom ÷ exact"],
    D.cbtab.slice().sort((x, y) => (D.tab.find(t => t.ds === x.ds)?.n ?? 0) - (D.tab.find(t => t.ds === y.ds)?.n ?? 0) || x.ds.localeCompare(y.ds) || x.sweep.localeCompare(y.sweep) || x.nl - y.nl).map(r => {
      const g = r.ds + r.sweep !== prev ? " grp" : ""; prev = r.ds + r.sweep;
      const n = D.tab.find(x => x.ds === r.ds)?.n;
      return `<tr class="${g}"><td>${r.ds} · ${r.sweep}</td><td>${r.nl.toLocaleString("en-US")}</td><td>${n ? n0(n / r.nl) : "–"}</td><td class="l">${r.probes.split("–")[0] === r.probes.split("–")[1] ? r.probes.split("–")[0] : r.probes}</td><td>${r.med < 1 ? "<b>" + f(r.med) + "</b>" : f(r.med)} <small>${f(r.lo)}–${f(r.hi)}</small></td></tr>`;
    }));
  const fp = D.fp;
  document.getElementById("fp-note").textContent = `at the timed settings at most ${fp.share_max} % of the items the bloom passes actually fail the filter (none at ${fp.zero} of ${fp.points} graded settings), there are no false negatives, and recall moves by at most ${fp.cost_max}.`;
})();

/* Tables 9-10: parameters */
(function () {
  let prev = null;
  table("t-nl", "Table 9. SilverTorch's number of clusters on LAION 30M (d = 256): index, build, recall at a share of clusters probed, and the fastest timed setting reaching 0.90 at batch 16.",
    ["Filter · clusters", "Index, MiB", "Build, s", "Recall at 1/8", "Recall at 1/4", "Probes for 0.90", "t<sub>med</sub>, ms"],
    D.nltab.map(r => {
      const g = r.sweep !== prev ? " grp" : ""; prev = r.sweep;
      const best = Math.min(...D.nltab.filter(x => x.sweep === r.sweep && x.ms != null).map(x => x.ms));
      return `<tr class="${g}"><td>${r.sweep} · ${r.nl.toLocaleString("en-US")}</td><td>${n0(r.mib)}</td><td>${f(r.build, 0)}</td><td>${f(r.r8, 3)}</td><td>${f(r.r4, 3)}</td><td>${r.probes == null ? "–" : r.probes.toLocaleString("en-US")}</td><td>${r.ms == null ? "–" : r.ms === best ? "<b>" + f(r.ms) + "</b>" : f(r.ms)}</td></tr>`;
    }));
  const pick = ["Goodreads 0.8M|c0_genre", "arXiv 3M|c0_maincat", "LAION 30M|tags4", "PubMed 10M|c0_mesh"];
  const rows = D.pools.filter(p => pick.includes(p.ds + "|" + p.sweep));
  const pools = rows[0].pts.map(p => p[0]);
  table("t-pool", "Table 10. LiNR V3's recall@100 against its candidate pool, as a share of the passing items (floor: 2,000 candidates).",
    ["Dataset · filter", "Pass"].concat(pools.map(p => p + " %")),
    rows.map(r => `<tr><td>${r.ds} · ${r.sweep}</td><td>${pct(r.pass_rate)}</td>${r.pts.map(p => `<td>${p[1] >= 0.9 ? "<b>" + f(p[1], 3) + "</b>" : f(p[1], 3)}</td>`).join("")}</tr>`));
})();

/* Tables 11-12: seed variance at the seed-0 pick, machine check */
(function () {
  table("t-var", "Table 11. Spread over the seeds at the same setting, tuned cells at their seed-0 pick: SilverTorch's time (max ÷ min − 1), its recall (max − min), its time over the best exact LiNR, and over the best LiNR at each tuned cell.",
    ["Dataset", "Seeds", "Settings", "Time: median", "Time: max", "Recall: max", "vs exact LiNR", "Tuned cells", "vs best LiNR", "Winner changes"],
    D.vsum.map(r => `<tr><td>${r.ds}</td><td>${r.seeds}</td><td>${r.points}</td><td>${f(r.t_med, 1)} %</td><td>${f(r.t_max, 1)} %</td><td>${f(r.r_max, 3)}</td><td>${f(r.w_med, 1)} %</td><td>${r.cells}</td><td>${r.fx_med == null ? "–" : f(r.fx_med, 1) + " %"}</td><td>${r.flips}</td></tr>`));
  table("t-mach", "Table 12. LiNR V1's time at seeds 1–2 over seed 0, same cell, by the machine that ran the seed.",
    ["Cells", "Seed", "Machine", "Settings", "Median", "Range"],
    D.machine.map(m => `<tr><td class="l">${m.cells}</td><td>${m.seed}</td><td>${m.other ? "another" : "same"}</td><td>${m.n}</td><td>${f(m.med, 3)}</td><td>${f(m.lo, 3)}–${f(m.hi, 3)}</td></tr>`));
  const v = D.vtext, b = D.vbelow, set = (id, t) => { document.getElementById(id).textContent = t; };
  set("v-t", v.t); set("v-r", v.r); set("v-fx", v.fx); set("v-cells", v.cells); set("v-mach", v.mach);
  set("v-flips", v.flips === 0 ? "none" : v.flips);
  set("v-below", b.n ? `At that setting ${b.n} cells (${b.cells.join(", ")}) dip below the target at a later seed, by at most ${b.gap}.` : "");
  set("vflip", D.vflip.length ? D.vflip.join("; ").replace(/ p1,/, " with no filter,") : "none");
})();

/* Table 13: claims */
table("t-claims", "Table 13. Each paper's claim against our measurement (Meta: C4–C7; LinkedIn: C1–C3).",
  ["", "Verdict", "The paper", "We measured"],
  D.verdicts.map(([id, c, v, claim, got]) => `<tr><td><b>${id}</b></td><td><span class="pill ${c}">${v}</span></td><td>${claim}</td><td>${got}</td></tr>`));

/* Provenance counts */
document.getElementById("prov-n").textContent = `${D.prov.legs} subtrees and ${D.prov.records.toLocaleString("en-US")} records feed this page`;
document.getElementById("prov-flag").textContent = `${D.prov.flagged} of ${D.prov.points} timed points, over all seeds`;
