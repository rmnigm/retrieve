"""``bench.report``: the table-generating path over a synthetic results tree.

The gate this pins: the columns ``report.py`` reads out of ``records.aggregate`` still exist,
every emitted fragment is structurally balanced LaTeX with its labels, a
``failed`` record never reaches a number, and the citability marker is on unless a gate is
declared *and* the evidence allows it.
"""

from __future__ import annotations

import hashlib
import json
import re
import zlib

import click
import numpy as np
import pytest
import yaml

from bench import records, report

# Everything a table reads out of results.parquet. A schema change that drops one breaks here.
REQUIRED_COLUMNS = (
    *records.KEY_FIELDS, "status", "path", "n_items", "pass_rate", "index_mib",
    "env_code_version", "env_commit", "env_dirty",
    "perf_k", "perf_bs", "perf_mode", "perf_median_ms", "perf_mean_ms", "perf_p99_ms",
    "perf_qps", "perf_spread", "perf_unstable", "perf_sm_mhz",
    "oracle_recall@100", "heldout_recall@100", "quality_jaccard_vs_first@100",
    "quality_score_max_abs_diff",
)  # fmt: skip

ENV = {
    "gpu": "NVIDIA A100-SXM4-80GB", "driver": "580.159.04", "cuda": "12.8",
    "torch": "2.10.0+cu128", "triton": "3.6.0", "commit": "abc1234", "dirty": False,
    "git_branch": "development", "code_version": "f" * 40, "host": "box", "python": "3.11.15",
    "started": "2026-09-15T08:00:00+00:00", "sm_mhz_idle": 1155.0, "sm_mhz_load": 1410.0,
    "clocks_drift": False,
}  # fmt: skip


def _perf(k, bs, mode, ms, unstable=False, windows=(1.0, 1.0, 1.0)):
    return {
        "k": k, "bs": bs, "mode": mode, "n": 1000, "median_ms": ms, "mean_ms": ms,
        "p95_ms": ms * 1.1, "p99_ms": ms * 1.2, "min_ms": ms * 0.9, "iqr_ms": 0.01,
        "qps": 1000 * bs / (ms * 1e-3 * 1000), "host_gap_ms": 0.01, "spread":
        0.2 if unstable else 0.001, "unstable": unstable, "peak_fwd_mib": 4.0,
        "window_medians_ms": [ms * w for w in windows], "sm_mhz": 1410.0,
        "load": "closed_loop",
    }  # fmt: skip


def _rec(
    dataset,
    algo,
    backend,
    *,
    filter_kind="clause",
    sweep="c0_genre",
    params=None,
    status="ok",
    seed=0,
    ms=1.0,
    unstable=False,
    recall=0.9,
    jaccard=None,
    windows=(1.0, 1.0, 1.0),
    interleave=None,
):
    perf = [
        _perf(k, bs, mode, ms * (1 + 0.1 * bs), unstable, windows)
        for k in (100,)
        for bs in (1, 8, 16)
        for mode in ("eager", "graph")
    ]
    return {
        "schema_version": records.SCHEMA_VERSION, "status": status, "partial_reasons":
        ["skip_perf"] if status == "partial" else None,
        "dataset": dataset, "dim": 128, "inputs": "sasrec-ssm-logq-d128", "suite": "filter",
        "filter_kind": filter_kind,
        "sweep": sweep, "algo": algo, "backend": backend, "params": params or {},
        "seed": seed, "path": backend, "n_items": 797084, "n_queries": 10000, "n_kept": 9859,
        "n_queries_heldout": 9859, "n_queries_oracle": 9859, "pass_rate": 0.33,
        "bloom_fp_rate": None, "k_max": 100, "ks": [100], "batch_sizes": [1, 8, 16],
        "build_s": 1.0, "index_mib": 194.6, "filter_mib": 0.0,
        "quality": {
            "heldout": {"recall@100": 0.2, "n": 9859},
            "oracle": {"recall@100": recall, "ndcg@100": recall, "n": 9859},
            "jaccard_vs_first@100": jaccard,
            "score_max_abs_diff": None if jaccard is None else 9.77e-3,
            "parity": "reference" if jaccard is None else "vs_triton",
        },
        "perf": None if status == "failed" else perf,
        "unstable": unstable, "memory_reserved_mib": 1220.0, "elapsed_s": 10.0,
        "env": dict(ENV), "interleave": interleave, "per_query": None,
        **({"stage": "build", "error": "RuntimeError: boom"} if status == "failed" else {}),
    }  # fmt: skip


def _with_sidecar(root, rec, recall, ks=(100,), pass_count=None):
    """Write the schema-4 per-query sidecar of ``rec`` (the record contract) and point at it."""
    rel = (
        f"{rec['suite']}/{rec['dataset']}-d{rec['dim']}.perquery/"
        f"{hashlib.sha1(records.record_key(rec).encode()).hexdigest()[:20]}.npz"
    )
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    n = len(recall)
    np.savez_compressed(
        root / rel,
        rows=np.arange(n, dtype=np.int32),
        pass_count=np.full(n, 1000, dtype=np.int64) if pass_count is None else pass_count,
        **{f"recall_oracle@{k}": np.asarray(recall, dtype=np.float32) for k in ks},
        **{f"heldout_recall@{k}": np.full(n, np.nan, dtype=np.float32) for k in ks},
    )
    rec["per_query"] = rel
    for k in ks:
        rec["quality"]["oracle"][f"recall@{k}"] = float(np.nanmean(recall))
    return rec


@pytest.fixture
def results(tmp_path):
    p = tmp_path / "results" / "filter" / "goodreads-d128.jsonl"
    for rec in (
        _rec("goodreads", "linr_v1_filter_mask", "triton", ms=0.5),
        _rec("goodreads", "linr_v1_filter_mask", "torch", ms=0.8, jaccard=1.0),
        _rec(
            "goodreads",
            "silvertorch",
            "triton",
            params={"n_probe": 24},
            ms=0.3,
            unstable=True,
            recall=0.91,
        ),
        _rec("goodreads", "silvertorch", "triton", params={"n_probe": 32}, ms=0.4, recall=0.94),
        _rec("goodreads", "linr_v2", "triton", status="failed"),
        _rec("goodreads", "linr_v3", "triton", status="partial", ms=1.2, recall=0.7),
    ):
        records.append_record(p, rec)
    q = tmp_path / "results" / "quality" / "goodreads-d128.jsonl"
    records.append_record(
        q,
        {
            **_rec(
                "goodreads", "linr_v1_filter_mask", "triton", filter_kind="none", sweep="full_scan"
            ),
            "suite": "quality",
        },
    )
    samples = records.samples_path(p)
    records.append_record(
        samples,
        {
            "dataset": "goodreads",
            "dim": 128,
            "suite": "filter",
            "filter_kind": "clause",
            "sweep": "c0_genre",
            "algo": "linr_v1_filter_mask",
            "backend": "triton",
            "params": {},
            "seed": 0,
            "k": 100,
            "bs": 1,
            "mode": "eager",
            "ms": [0.5, 0.51, 0.49, 0.52],
        },
    )
    return tmp_path / "results"


def _generate(results, out, **kw):
    return report.generate(
        results,
        out,
        dim=128,
        k=100,
        bs=1,
        mode="eager",
        backend="triton",
        **kw,
    )


def test_results_parquet_still_carries_every_column_the_tables_read(results, tmp_path):
    c = _generate(results, tmp_path / "out")
    missing = [col for col in REQUIRED_COLUMNS if col not in c.rows[0]]
    assert not missing, f"records.aggregate no longer emits: {missing}"


def test_every_artifact_is_emitted_and_the_latex_is_structurally_sound(results, tmp_path):
    c = _generate(results, tmp_path / "out")
    names = {p.name for p in c.written}
    assert {"results.parquet", "report.md", "methodology.tex"} <= names
    assert {"tab-pareto_goodreads.tex", "tab-memory.tex", "tab-backend_parity.tex"} <= names
    assert sum(1 for p in c.written if p.suffix == ".png") >= 2
    labels = set()
    for path in [p for p in c.written if p.suffix == ".tex"]:
        text = path.read_text()
        body = "\n".join(ln for ln in text.splitlines() if not ln.startswith("%"))
        for env in ("table", "tabular", "itemize", "minipage"):
            assert body.count(f"\\begin{{{env}}}") == body.count(f"\\end{{{env}}}"), path
        assert body.count("{") == body.count("}"), f"unbalanced braces in {path}"
        assert body.count("$") % 2 == 0, f"unbalanced math in {path}"
        labels |= set(re.findall(r"\\label\{([^}]*)\}", body))
        stripped = re.sub(r"\$[^$]*\$|\\(?:label|texttt|ref)\{[^}]*\}", "", body)
        assert "_" not in stripped.replace("\\_", ""), f"unescaped underscore in {path}"
    assert {"tab:pareto_goodreads", "tab:memory", "tab:backend_parity"} <= labels


def test_failed_is_excluded_partial_and_unstable_are_marked(results, tmp_path):
    c = _generate(results, tmp_path / "out")
    pareto = (tmp_path / "out" / "tables" / "tab-pareto_goodreads.tex").read_text()
    assert "LiNR V2" not in pareto  # a failed cell never reaches a number, backend or not
    assert "^{\\dagger}" in pareto  # the unstable silvertorch cell is marked
    assert "^{*}" in pareto  # the partial linr_v3 cell is marked
    md = (tmp_path / "out" / "report.md").read_text()
    assert "RuntimeError: boom" in md and "skip_perf" in md
    assert c.prov["status"] == {"failed": 1, "ok": 5, "partial": 1}


def test_citability_is_off_by_default_and_evidence_beats_the_gate(results, tmp_path):
    ungated = _generate(results, tmp_path / "a")
    assert not ungated.prov["citable"]
    gated = _generate(results, tmp_path / "b", gate="D1")
    assert not gated.prov["citable"], "a failed/partial record must veto --gate"
    assert any("failed" in b for b in gated.prov["blockers"])
    for out in (tmp_path / "a", tmp_path / "b"):
        assert "NOT CITABLE" in (out / "tables" / "tab-memory.tex").read_text()

    clean = tmp_path / "clean" / "filter" / "goodreads-d128.jsonl"
    records.append_record(clean, _rec("goodreads", "linr_v1_filter_mask", "triton"))
    c = _generate(tmp_path / "clean", tmp_path / "c", gate="D1")
    assert c.prov["citable"] and c.prov["blockers"] == []
    tex = (tmp_path / "c" / "tables" / "tab-memory.tex").read_text()
    assert "NOT CITABLE" not in tex and "gate D1" in tex

    dirty = tmp_path / "dirty" / "filter" / "goodreads-d128.jsonl"
    rec = _rec("goodreads", "linr_v1_filter_mask", "triton")
    rec["env"] = {**ENV, "dirty": True}
    records.append_record(dirty, rec)
    d = _generate(tmp_path / "dirty", tmp_path / "d", gate="D1")
    assert not d.prov["citable"] and any("dirty" in b for b in d.prov["blockers"])

    branch = tmp_path / "branch" / "filter" / "goodreads-d128.jsonl"
    rec = _rec("goodreads", "linr_v1_filter_mask", "triton")
    rec["env"] = {**ENV, "git_branch": "dev/c4-gate-rerun"}
    records.append_record(branch, rec)
    b = _generate(tmp_path / "branch", tmp_path / "e", gate="D1")
    assert not b.prov["citable"], "rule 2: a number from a dev/* branch is not citable"
    assert any("dev/c4-gate-rerun" in x for x in b.prov["blockers"])


def test_two_encoders_under_one_dataset_and_dim_are_refused(results, tmp_path):
    legacy = _rec("goodreads", "linr_v2", "torch")
    del legacy["inputs"]  # a pre-H2 goodreads record: gSASRec
    records.append_record(results / "filter" / "goodreads-d128.jsonl", legacy)
    with pytest.raises(click.ClickException, match="gsasrec-d128-drop0.5-id"):
        _generate(results, tmp_path / "out")


def test_an_empty_results_tree_emits_placeholders_rather_than_failing(tmp_path):
    (tmp_path / "results" / "filter").mkdir(parents=True)
    c = _generate(tmp_path / "results", tmp_path / "out")
    assert c.prov["n_records"] == 0
    tex = (tmp_path / "out" / "tables" / "tab-memory.tex").read_text()
    assert "no matching cells" in tex and "NOT CITABLE" in tex


def test_it_reads_schema_1_records_without_the_c5_clock_fields(results, tmp_path):
    """C4's records are schema 1: no ``env.sm_mhz_load``, an ``env.sm_mhz`` that means a
    whole-run median. The tables must still build, and must not use that field."""
    p = tmp_path / "old" / "filter" / "goodreads-d128.jsonl"
    rec = _rec("goodreads", "silvertorch", "triton", params={"n_probe": 24})
    rec["schema_version"] = 1
    rec["env"] = {k: v for k, v in ENV.items() if k not in ("sm_mhz_load", "clocks_drift")}
    rec["env"]["sm_mhz"] = 1155.0
    records.append_record(p, rec)
    c = _generate(tmp_path / "old", tmp_path / "out")
    assert c.prov["schema_versions"] == [1]
    note = (tmp_path / "out" / "tables" / "tab-pareto_goodreads.tex").read_text()
    assert "1410--1410" in note  # the under-load per-variant sample, not the 1155 median


def test_each_reason_is_stated_in_the_banner_caption_and_report(results, tmp_path):
    """Roadmap H5: a non-citable report names its actual reasons, never "pre-campaign"."""
    rec = _rec("goodreads", "linr_v1_filter_mask", "triton")
    rec["env"] = {**ENV, "dirty": True, "git_branch": "dev/x"}
    records.append_record(results / "filter" / "goodreads-d128.jsonl", rec)
    c = _generate(results, tmp_path / "out")
    assert c.prov["blockers"] == [
        "no --gate given: no roadmap gate is declared green for these records",
        "1 record(s) with status=failed",
        "1 record(s) with status=partial (partial_reasons: skip_perf 1)",
        "1 record(s) with env.dirty (library subtree was dirty)",
        "record(s) produced on dev/x — CLAUDE.md rule 2: harness numbers from a branch are "
        "not paper material",
    ]
    assert c.prov["marks"] == [
        "gate not green", "1 failed", "1 partial: skip_perf 1", "1 dirty", "branch dev/x"
    ]  # fmt: skip
    tex = (tmp_path / "out" / "tables" / "tab-memory.tex").read_text()
    assert (
        "\\textbf{[NOT CITABLE: gate not green; 1 failed; 1 partial: skip\\_perf 1; 1 dirty; "
        "branch dev/x]}~" in tex
    )
    assert "%   - 1 record(s) with status=partial (partial_reasons: skip_perf 1)" in tex
    md = (tmp_path / "out" / "report.md").read_text()
    assert "- 1 record(s) with env.dirty (library subtree was dirty)" in md
    for path in c.written:
        if path.suffix in (".tex", ".md"):
            text = path.read_text().lower()
            assert "pre-campaign" not in text and "predate" not in text, path


def test_postfilter_rows_carry_their_alpha_and_are_never_averaged(tmp_path):
    """The alpha rule: one row per alpha, labelled with it; the torch baseline passes
    ``--backend triton`` (``FIXED_BACKEND``); the swept alphas draw the recovery curve."""
    p = tmp_path / "results" / "filter" / "goodreads-d128.jsonl"
    records.append_record(p, _rec("goodreads", "linr_v1_filter_mask", "triton", ms=0.5))
    for alpha, recall in ((1, 0.71), (2, 0.85), (4, 0.93), (8, 0.97)):
        records.append_record(
            p,
            _rec("goodreads", "postfilter", "torch", params={"alpha": alpha}, recall=recall),
        )
    c = _generate(tmp_path / "results", tmp_path / "out")
    pareto = (tmp_path / "out" / "tables" / "tab-pareto_goodreads.tex").read_text()
    rows = {
        alpha: next(ln for ln in pareto.splitlines() if f"postfilter ($\\alpha$={alpha})" in ln)
        for alpha in (1, 2, 4, 8)
    }
    assert "0.7100" in rows[1] and "0.9700" in rows[8] and "0.85" not in rows[1]
    assert "postfilter" in (tmp_path / "out" / "tables" / "tab-memory.tex").read_text()
    names = {p.name for p in c.written}
    assert "fig-deep-sweep-goodreads-clause-c0_genre-postfilter-alpha.png" in names
    assert report._sel(c.rows, backend="official", algo="postfilter")  # any backend selection
    assert not report._sel(c.rows, backend="official", algo="linr_v1_filter_mask")


def test_an_algo_without_a_label_never_reaches_a_table(results, tmp_path):
    p = results / "filter" / "goodreads-d128.jsonl"
    records.append_record(p, _rec("goodreads", "linr_v4", "triton"))
    _generate(results, tmp_path / "out")
    for name in ("tab-pareto_goodreads.tex", "tab-memory.tex"):
        assert "v4" not in (tmp_path / "out" / "tables" / name).read_text().lower()


def test_interleaved_arms_give_a_paired_speedup_and_sidecars_give_the_recall_ci(tmp_path):
    root = tmp_path / "results"
    p = root / "filter" / "goodreads-d128.jsonl"
    group = {"group": "g0", "arms": ["linr_v1_filter_mask", "linr_v2"]}
    for seed in (0, 1):
        v1 = _rec(
            "goodreads",
            "linr_v1_filter_mask",
            "triton",
            seed=seed,
            ms=1.0,
            windows=(1.0, 1.3, 0.8),
            interleave={**group, "position": 0},
        )
        v2 = _rec(
            "goodreads",
            "linr_v2",
            "triton",
            seed=seed,
            ms=0.5,
            windows=(1.0, 1.3, 0.8),
            interleave={**group, "position": 1},
        )
        records.append_record(p, _with_sidecar(root, v1, [1.0] * 400))
        records.append_record(p, _with_sidecar(root, v2, [1.0, 0.0] * 200))
    for seed, w in ((0, (1.0, 1.0, 1.0)), (1, (0.9, 1.1, 1.0))):  # not interleaved
        records.append_record(p, _rec("goodreads", "linr_v3", "triton", seed=seed, ms=1.0,
                                      windows=w))  # fmt: skip
    _generate(root, tmp_path / "out")
    tex = (tmp_path / "out" / "tables" / "tab-pareto_goodreads.tex").read_text()
    v2 = next(ln for ln in tex.splitlines() if "LiNR V2" in ln)
    assert "$2.00\\times\\,[2.00, 2.00]$" in v2  # 1.1 ms / 0.55 ms in every round
    assert "$0.5000$\\,{\\tiny$[0.4" in v2  # the mean of the per-query recalls, with its CI
    v3 = next(ln for ln in tex.splitlines() if "LiNR V3" in ln)
    assert "no difference ($1.00\\times" in v3 and "^{u}" in v3  # unpaired, CI holds 1


def test_matched_recall_interpolates_names_its_bracket_and_emits_n95(tmp_path):
    p = tmp_path / "results" / "deep" / "goodreads-d128.jsonl"
    for n_probe, recall, ms in ((8, 0.80, 1.0), (16, 0.90, 2.0), (32, 0.94, 3.0), (64, 0.97, 4.0)):
        params = {"n_probe": n_probe, "n_lists": 1024}
        rec = _rec("goodreads", "silvertorch", "triton", params=params, recall=recall, ms=ms)
        records.append_record(p, {**rec, "suite": "deep"})
    for pool, recall in ((1000, 0.5), (5000, 0.7)):
        rec = _rec("goodreads", "linr_v3", "triton", params={"candidate_pool": pool}, recall=recall)
        records.append_record(p, {**rec, "suite": "deep"})
    _generate(tmp_path / "results", tmp_path / "out")
    dump = json.loads((tmp_path / "out" / "matched_recall.json").read_text())
    st = next(d for d in dump if d["algo"] == "silvertorch" and d["bs"] == 1)
    assert st["params"] == {"n_lists": 1024} and st["n95"] == 64
    assert st["at_0.90"] == {"latency": 2.2, "bracket": ["n_probe=16"] * 2, "mode": "graph"}
    assert st["at_0.95"]["latency"] == pytest.approx(3.3 + 1.1 / 3)  # (0.95-0.94)/(0.97-0.94)
    assert st["at_0.95"]["bracket"] == ["n_probe=32", "n_probe=64"]
    v3 = next(d for d in dump if d["algo"] == "linr_v3" and d["bs"] == 16)
    assert v3["axis"] == "candidate_pool" and v3["n95"] is None
    assert v3["at_0.90"] == {"latency": None, "reason": "not reached", "mode": "graph"}
    tex = (tmp_path / "out" / "tables" / "tab-matched_recall.tex").read_text()
    assert "(n\\_probe=32 .. n\\_probe=64)" in tex and "not reached" in tex


# ----- G-report: a schema-4 campaign tree, every arm of every suite (the record contract) ---

N_ITEMS = {"goodreads": 797_084, "arxiv": 2_984_617, "yfcc10m": 9_998_311, "pubmed": 10_000_000}
DIM = {"goodreads": 128, "arxiv": 128, "yfcc10m": 192, "pubmed": 768}
DETERMINISTIC = ("linr_v1_filter_mask", "linr_v2", "postfilter")


def _rec4(
    root,
    dataset,
    suite,
    algo,
    backend,
    *,
    sweep,
    fk="clause",
    params=None,
    seed=0,
    ms=1.0,
    recall=0.9,
    pass_rate=0.1,
    ks=(100,),
    bss=(1, 16),
    perf=True,
    interleave=None,
    kernels=None,
    kernels_us=None,
    ids="a",
    fp_rate=None,
    source=None,
    status="ok",
    score_diff=None,
):
    """One schema-4 record. ``source``: the seed-0 record a quality-cache copy points at."""
    base = dataset.removesuffix("-synth")
    rec = _rec(
        dataset,
        algo,
        backend,
        filter_kind=fk,
        sweep=sweep,
        params=params,
        seed=seed,
        recall=recall,
        status=status,
    )
    eager_only = backend == "official"
    windows = (1.0, 1.04, 0.97)
    entries = []
    for k in ks:
        for bs in bss:
            for mode in ("eager", "graph"):
                t = ms * (1 + 0.05 * bs) * (1 + 0.01 * seed)
                e = _perf(k, bs, mode, t, False, windows)
                # exact order differs per backend (tied ids), the canonical hash does not
                e |= {"ids_sha256": f"{ids}-{backend}", "ids_sha256_canon": ids, "rounds": 3}
                if mode == "graph" and eager_only:
                    e = {"k": k, "bs": bs, "mode": mode, "reason": "not_capturable"}
                    e |= dict.fromkeys(
                        ("median_ms", "window_medians_ms", "ids_sha256", "ids_sha256_canon")
                    )
                if kernels and mode == "eager":
                    e["kernels"] = kernels
                if kernels_us and mode == "eager":
                    e["kernels_us"], e["kernels_calls"] = kernels_us
                entries.append(e)
    rec |= {
        "schema_version": 4,
        "suite": suite,
        "dim": DIM[base],
        "n_items": N_ITEMS[base],
        "inputs": f"inputs-{base}-d{DIM[base]}",
        "pass_rate": pass_rate,
        "ks": list(ks),
        "k_max": max(ks),
        "batch_sizes": list(bss),
        "perf": entries if perf and status != "failed" else None,
        "seed_scope": "pool" if algo in DETERMINISTIC else "pool+build",
        "quality_source": None,
        "interleave": interleave,
        "bloom_fp_rate": fp_rate,
        "filter_mib": 0.0 if backend != "triton" else 1.5,
        "partial_reasons": ["skip_perf"] if status == "partial" else None,
    }
    rec["env"] = {**ENV, "frac_windows_below_max": 0.02}
    for k in ks:
        rec["quality"]["oracle"][f"recall@{k}"] = recall
    if source is not None:
        rec["quality_source"] = {"seed": source["seed"], "code_version": ENV["code_version"]}
        rec["per_query"] = source["per_query"]
        rec["quality"] = source["quality"]
    elif status != "failed":
        if score_diff is not None:
            rec["quality"]["score_max_abs_diff"] = score_diff
        rng = np.random.default_rng(zlib.crc32(records.record_key(rec).encode()))
        hits = (np.arange(200) < round(recall * 200)).astype(np.float32)
        rec = _with_sidecar(root, rec, hits, ks=ks, pass_count=rng.integers(1, 10**6, 200))
    p = root / suite / f"{dataset}-d{rec['dim']}.jsonl"
    records.append_record(p, rec)
    return rec


def _group(name, arms, position):
    return {"group": name, "arms": arms, "position": position}


def _campaign_tree(root):
    seeds = (0, 1)

    def det(dataset, suite, algo, backend, **kw):
        """A deterministic arm: seed 0 computes quality, seed 1 copies it (quality cache)."""
        first = _rec4(root, dataset, suite, algo, backend, seed=0, **kw)
        _rec4(root, dataset, suite, algo, backend, seed=1, source=first, **kw)

    # filter: four datasets, the headline arms
    for ds, sw in (
        ("goodreads", "c0_genre"),
        ("arxiv", "c0_maincat"),
        ("yfcc10m", "tags_and"),
        ("pubmed", "c0_mesh"),
    ):
        g = _group(f"{ds}-v1v2", ["linr_v1_filter_mask", "linr_v2"], 0)
        det(
            ds,
            "filter",
            "linr_v1_filter_mask",
            "triton",
            sweep=sw,
            recall=1.0,
            ms=1.2,
            interleave=g,
        )
        det(
            ds,
            "filter",
            "linr_v2",
            "triton",
            sweep=sw,
            recall=1.0,
            ms=0.6,
            interleave={**g, "position": 1},
        )
        for alpha, recall in ((1, 0.7), (8, 0.97)):
            det(
                ds,
                "filter",
                "postfilter",
                "torch",
                sweep=sw,
                params={"alpha": alpha},
                recall=recall,
                ms=2.0,
            )
        for s in seeds:
            _rec4(
                root,
                ds,
                "filter",
                "linr_v3",
                "triton",
                sweep=sw,
                seed=s,
                recall=0.8,
                params={"candidate_pool": 5000},
                ms=0.9,
            )
            for n_probe, recall, ms in ((24, 0.90, 0.3), (64, 0.97, 0.5)):
                _rec4(
                    root,
                    ds,
                    "filter",
                    "silvertorch",
                    "triton",
                    sweep=sw,
                    seed=s,
                    params={"n_probe": n_probe},
                    recall=recall,
                    ms=ms,
                )
                for be in ("triton", "official"):
                    _rec4(
                        root,
                        ds,
                        "filter",
                        "silvertorch",
                        be,
                        sweep=sw,
                        fk="bloom",
                        seed=s,
                        params={"n_probe": n_probe},
                        recall=recall - 0.01,
                        ms=ms * 1.1,
                    )
            _rec4(
                root,
                ds,
                "filter",
                "silvertorch",
                "torch",
                sweep=sw,
                seed=s,
                params={"n_probe": 24},
                recall=0.90,
                ms=3.0,
            )
            _rec4(
                root,
                ds,
                "filter",
                "silvertorch",
                "torch",
                sweep=sw,
                seed=s,
                params={"n_probe": 24, "compile": "max-autotune"},
                recall=0.90,
                ms=2.0,
            )

    # deep: arxiv, SilverTorch along n_probe per n_lists, V3 along the pool fraction
    for s in seeds:
        for n_lists in (1664, 8192):
            for n_probe, recall, ms in (
                (8, 0.80, 0.2),
                (16, 0.90, 0.3),
                (32, 0.94, 0.4),
                (64, 0.97, 0.6),
            ):
                for fk, be in (("clause", "triton"), ("bloom", "triton"), ("bloom", "official")):
                    _rec4(
                        root,
                        "arxiv",
                        "deep",
                        "silvertorch",
                        be,
                        sweep="c0_maincat",
                        fk=fk,
                        seed=s,
                        params={"n_lists": n_lists, "n_probe": n_probe},
                        recall=recall,
                        ms=ms,
                    )
        for sw in ("c0_maincat", "all4"):
            for frac, recall in ((0.01, 0.85), (0.05, 0.96)):
                _rec4(
                    root,
                    "arxiv",
                    "deep",
                    "linr_v3",
                    "triton",
                    sweep=sw,
                    seed=s,
                    params={"candidate_pool_frac": frac},
                    recall=recall,
                    ms=0.5 + frac,
                )

    # synth: two scales, pass rates 0.001 / 0.1 / 1.0
    for ds in ("goodreads-synth", "arxiv-synth"):
        for sw, p in (("p0001", 0.001), ("p01", 0.1), ("p1", 1.0)):
            g = _group(f"{ds}-{sw}-v1v2", ["linr_v1_filter_mask", "linr_v2"], 0)
            det(
                ds,
                "synth",
                "linr_v1_filter_mask",
                "triton",
                sweep=sw,
                pass_rate=p,
                recall=1.0,
                ms=1.0 + p,
                interleave=g,
            )
            det(
                ds,
                "synth",
                "linr_v2",
                "triton",
                sweep=sw,
                pass_rate=p,
                recall=1.0,
                ms=0.2 + 2 * p,
                interleave={**g, "position": 1},
            )
            for alpha in (1, 8):
                det(
                    ds,
                    "synth",
                    "postfilter",
                    "torch",
                    sweep=sw,
                    pass_rate=p,
                    params={"alpha": alpha},
                    recall=min(1.0, p * alpha + 0.05),
                    ms=2.0,
                )
            if sw != "p0001":
                for algo in ("linr_v1_filter_mask", "linr_v2"):
                    det(ds, "synth", algo, "torch", sweep=sw, pass_rate=p, recall=1.0, ms=9.0)
                    det(
                        ds,
                        "synth",
                        algo,
                        "torch",
                        sweep=sw,
                        pass_rate=p,
                        recall=1.0,
                        ms=6.0,
                        params={"compile": "max-autotune"},
                    )
            for s in seeds:
                for frac in (0.01, 0.05):
                    _rec4(
                        root,
                        ds,
                        "synth",
                        "linr_v3",
                        "triton",
                        sweep=sw,
                        seed=s,
                        pass_rate=p,
                        params={"candidate_pool_frac": frac},
                        recall=0.6 + 4 * frac,
                        ms=0.8,
                    )
                for n_probe, gain in ((24, 0.0), (96, 0.06), (384, 0.1)):
                    _rec4(
                        root,
                        ds,
                        "synth",
                        "silvertorch",
                        "triton",
                        sweep=sw,
                        seed=s,
                        pass_rate=p,
                        params={"n_probe": n_probe},
                        recall=min(1.0, 0.88 + gain),
                        ms=0.2 + n_probe / 400,
                    )
                for be in ("triton", "official"):
                    _rec4(
                        root,
                        ds,
                        "synth",
                        "silvertorch",
                        be,
                        sweep=sw,
                        fk="bloom",
                        seed=s,
                        pass_rate=p,
                        params={"n_probe": 96},
                        recall=0.94,
                        ms=0.5,
                    )

    # codesign: arxiv, official bloom, partial vs full interleaved, full 1.5x slower
    for s in seeds:
        for n_probe in (8, 32, 128):
            g = f"codesign-{s}-{n_probe}"
            for pos, (path, ms) in enumerate((("partial", 1.0), ("full", 1.5))):
                _rec4(
                    root,
                    "arxiv",
                    "codesign",
                    "silvertorch",
                    "official",
                    sweep="c0_maincat",
                    fk="bloom",
                    seed=s,
                    params={"bloom_path": path, "n_lists": 1664, "n_probe": n_probe},
                    ms=ms * (1 + n_probe / 128),
                    interleave=_group(g, ["partial", "full"], pos),
                )

    # bloomwidth: quality only; bloomwidth-timed: one timed point per width at bs 16
    for ds, sw in (("goodreads", "c0_genre"), ("pubmed", "c0_mesh")):
        for be in ("triton", "official"):
            for m_bits in (64, 1024):
                for k_hash in (3, 5):
                    _rec4(
                        root,
                        ds,
                        "bloomwidth",
                        "silvertorch",
                        be,
                        sweep=sw,
                        fk="bloom",
                        params={"n_probe": 24, "m_bits": m_bits, "k_hash": k_hash},
                        perf=False,
                        fp_rate=0.0 if m_bits == 1024 else 0.02 / k_hash,
                    )
                _rec4(
                    root,
                    ds,
                    "bloomwidth-timed",
                    "silvertorch",
                    be,
                    sweep=sw,
                    fk="bloom",
                    params={"n_probe": 24, "m_bits": m_bits, "k_hash": 5},
                    bss=(16,),
                    fp_rate=0.0,
                    ms=0.5,
                )

    # h2h: goodreads bloom, triton vs official fp16 / int32, interleaved, profiled
    arms = ["silvertorch/triton", "silvertorch/official/fp16", "silvertorch/official/int32"]
    kern = [
        {"kernel": "scorer", "us": 40.0, "calls": 1},
        {"kernel": "topk", "us": 10.0, "calls": 2},
    ]
    for s in seeds:
        g = f"h2h-goodreads-{s}"
        # int32: seed 1 returns another tied id at the k-th cut (scores bit-equal); fp16: other
        # ids and other scores. Triton is profiled over every kernel, official top-8 only.
        for pos, (be, params, ms, ids, diff, kus) in enumerate(
            (
                ("triton", {"n_probe": 24}, 0.4, "x", None, (75.0, 7)),
                ("official", {"n_probe": 24, "score_path": "fp16"}, 0.6, "y", 9.77e-3, None),
                ("official", {"n_probe": 24, "score_path": "int32"}, 0.8, "xz"[s], 0.0, None),
            )
        ):
            _rec4(
                root,
                "goodreads",
                "h2h",
                "silvertorch",
                be,
                sweep="c0_genre",
                fk="bloom",
                seed=s,
                params=params,
                ms=ms,
                ks=(100, 1000),
                kernels=kern,
                kernels_us=kus,
                ids=ids,
                score_diff=diff,
                interleave=_group(g, arms, pos),
            )

    # n95: pubmed, quality only
    for n_probe, recall in ((8, 0.7), (32, 0.93), (128, 0.96)):
        _rec4(
            root,
            "pubmed",
            "n95",
            "silvertorch",
            "triton",
            sweep="all5",
            params={"n_probe": n_probe},
            recall=recall,
            ks=(100, 1000),
            perf=False,
        )

    # one failed and one partial record
    _rec4(
        root,
        "arxiv",
        "filter",
        "linr_v3",
        "triton",
        sweep="c0_maincat",
        seed=2,
        params={"candidate_pool": 5000},
        status="failed",
    )
    _rec4(
        root,
        "goodreads",
        "filter",
        "linr_v3",
        "triton",
        sweep="c0_genre",
        seed=2,
        params={"candidate_pool": 5000},
        status="partial",
        perf=False,
    )
    return root


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    root = _campaign_tree(tmp_path_factory.mktemp("campaign") / "results")
    out = root.parent / "out"
    return root, out, _generate(root, out)


def test_every_paper_exhibit_is_produced_from_a_schema_4_campaign_tree(campaign):
    """G-report: every exhibit from every arm of every suite, with the NOT CITABLE marker."""
    _, out, c = campaign
    names = {p.relative_to(out).as_posix() for p in c.written}
    assert {
        "tables/tab-t1_claims.tex",
        "tables/tab-t3.tex",
        "tables/tab-matched_recall.tex",
        "tables/tab-f4a_bloomwidth.tex",
        "tables/tab-f4b_codesign.tex",
        "figures/fig-f1-latency-vs-pass-rate.png",
        "figures/fig-f2-recall-vs-pass-rate.png",
        "figures/fig-f3-pareto.png",
        "figures/fig-f4a-bloomwidth.png",
        "figures/fig-f4b-codesign.png",
        "matched_recall.json",
        *(f"tables/tab-t2_{d}.tex" for d in ("goodreads", "arxiv", "yfcc10m", "pubmed")),
    } <= names
    assert c.prov["status"] == {"failed": 1, "ok": c.prov["n_records"] - 2, "partial": 1}
    for p in c.written:
        if p.suffix == ".tex":
            text = p.read_text()
            assert "NOT CITABLE" in text, p
            assert "\\caption" not in text or "[NOT CITABLE: gate not green; 1 failed;" in text, p


def test_t1_fills_ours_from_the_selectors_and_never_a_verdict(campaign):
    _, out, _ = campaign
    t1 = (out / "tables" / "tab-t1_claims.tex").read_text()
    c1 = next(ln for ln in t1.splitlines() if ln.strip().startswith("C1 &"))
    # V2 / V1 interleaved at p=0.1, B=1: (0.2 + 0.2) / (1.0 + 0.1) in every round, paired
    assert f"${0.4 / 1.1:.2f}\\times\\,[{0.4 / 1.1:.2f}, {0.4 / 1.1:.2f}]$" in c1
    assert "^{u}" not in c1
    c7 = next(ln for ln in t1.splitlines() if ln.strip().startswith("C7 &"))
    assert "official fp16 / triton latency" in c7 and "$1.50\\times" in c7
    for ln in t1.splitlines():
        if re.match(r"\s*C\d &", ln):
            assert ln.rstrip().endswith("& --- \\\\"), "a verdict was generated"


def test_t2_has_both_operating_points_and_the_alpha_rows(campaign):
    _, out, _ = campaign
    t2 = (out / "tables" / "tab-t2_goodreads.tex").read_text()
    for arm in (
        "postfilter ($\\alpha$=1)",
        "postfilter ($\\alpha$=8)",
        "SilverTorch (n\\_probe=24) [official]",
        "SilverTorch (compile=max-autotune, n\\_probe=24) [torch]",
    ):
        assert arm in t2, arm
    assert "SilverTorch (n\\_probe=64)" not in t2  # only the operating point as a fixed row
    matched = next(ln for ln in t2.splitlines() if "[triton] @ 0.95" in ln)
    assert "matched" in matched and "(n\\_probe=24 .. n\\_probe=64)" in matched
    official = next(ln for ln in t2.splitlines() if "[official]" in ln and "@" not in ln)
    assert "$^{e}$" in official  # eager-only, marked


def test_t3_pairs_the_interleaved_arms_and_checks_id_identity(campaign):
    _, out, _ = campaign
    t3 = (out / "tables" / "tab-t3.tex").read_text()
    rows = [ln for ln in t3.splitlines() if "[official]" in ln and "eager" in ln]
    int32 = next(ln for ln in rows if "int32" in ln)
    fp16 = next(ln for ln in rows if "fp16" in ln)
    # int32: seed 0 equal, seed 1 another id at bit-equal scores; fp16: other scores too
    assert "$2.00\\times\\,[2.00, 2.00]$" in int32 and "& $=$ (ties) &" in int32  # 0.8 / 0.4
    assert "$1.50\\times\\,[1.50, 1.50]$" in fp16 and "& $\\neq$ &" in fp16
    graph = next(ln for ln in t3.splitlines() if "[triton]" in ln and "& graph &" in ln)
    assert "& $=$ &" in graph  # the canonical hash is the eager arm's at every seed
    assert "& $50.0^{8}$ & $3$ &" in int32  # no kernels_us: the top-8 sum, marked
    eager = next(ln for ln in t3.splitlines() if "[triton]" in ln and "& eager &" in ln)
    assert "& $75.0$ & $7$ &" in eager  # kernels_us / kernels_calls, every kernel


def test_f3_draws_one_panel_per_sweep_and_batch_size(campaign, monkeypatch):
    _, _, c = campaign
    figs = []
    monkeypatch.setattr(report, "_figure", lambda path, fig, prov: figs.append(fig) or path)
    report.fig_f3(c)
    (fig,) = figs
    titles = sorted(ax.get_title() for ax in fig.axes)
    want = sorted(
        f"arXiv {sw}, B={bs}" for sw in ("all4", "c0_maincat") for bs in report.EXHIBIT_BS
    )
    assert titles == want
    for ax in fig.axes:
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        sw = ax.get_title().split()[1].rstrip(",")
        assert labels and not any("c0_maincat" in t or "all4" in t for t in labels)
        assert any("V3" in t for t in labels) and (sw == "c0_maincat") == any(
            "bloom" in t for t in labels
        )


def test_f4_tables_carry_fpr_memory_and_the_paired_codesign_ratio(campaign):
    _, out, _ = campaign
    f4a = (out / "tables" / "tab-f4a_bloomwidth.tex").read_text()
    assert "$6.67{\\times}10^{-3}$" in f4a  # 0.02 / k_hash 3 at 64 bits
    assert "& $1.50$ &" in f4a  # filter_mib of the triton module
    f4b = (out / "tables" / "tab-f4b_codesign.tex").read_text()
    ratios = re.findall(r"\$1\.50\\times\\,\[1\.50, 1\.50\]\$", f4b)
    assert len(ratios) == 2 * 3, "full / partial paired at every (bs, n_probe)"


def test_f2_overlays_real_sweeps_as_per_query_buckets(campaign):
    root, _, c = campaign
    rows = report._sel(
        c.rows,
        dataset="arxiv",
        suite="filter",
        algo="silvertorch",
        filter_kind="clause",
        backend="triton",
    )
    buckets = report._buckets(c, report._with(rows, n_probe=24), 100)
    assert buckets and sum(n for _, _, n in buckets) <= 2 * 200
    assert all(0.0 <= y <= 1.0 and 1e-6 < x <= 1.0 for x, y, _ in buckets)


def test_the_manifest_selects_quality_and_perf_by_their_accepted_code_version(tmp_path):
    """G-manifest: two code_versions of one cell; quality and perf each come from their own
    entry's code_version, a cell without an entry or without a record is missing."""
    root = tmp_path / "results"
    p = root / "filter" / "goodreads-d128.jsonl"

    def at(cv, rec):
        rec["env"] = {**ENV, "code_version": cv}
        records.append_record(p, rec)

    at("old", _rec("goodreads", "linr_v2", "triton", recall=0.5, ms=5.0))
    at("new", _rec("goodreads", "linr_v2", "triton", recall=0.9, ms=1.0))
    at("new", _rec("goodreads", "linr_v1_filter_mask", "triton", recall=0.8, ms=2.0))
    at("new", _rec("goodreads", "silvertorch", "triton", params={"n_probe": 24}))
    manifest = tmp_path / "campaign.yaml"
    cell = {"dataset": "goodreads", "suite": "filter", "backend": "triton"}
    manifest.write_text(yaml.safe_dump({"entries": [
        {"match": {**cell, "algo": "linr_v2"},
         "quality": {"code_version": "old"}, "perf": {"code_version": "new"}},
        {"match": {**cell, "algo": "linr_v1_filter_mask"},
         "quality": {"code_version": "new"}, "perf": {"code_version": "gone"}},
    ]}))  # fmt: skip
    c = _generate(root, tmp_path / "out", manifest=manifest)
    v2 = [r for r in c.recs if r["algo"] == "linr_v2"]
    assert len(v2) == 1 and v2[0]["quality"]["oracle"]["recall@100"] == 0.5
    assert v2[0]["perf"][0]["median_ms"] == 1.1 and v2[0]["env"]["code_version"] == "new"
    assert v2[0]["quality_source"] == {"seed": 0, "code_version": "old"}
    v1 = next(r for r in c.recs if r["algo"] == "linr_v1_filter_mask")
    assert v1["perf"] is None and v1["quality"]["oracle"]["recall@100"] == 0.8
    assert {r["algo"] for r in c.recs} == {"linr_v2", "linr_v1_filter_mask"}
    pareto = (tmp_path / "out" / "tables" / "tab-pareto_goodreads.tex").read_text()
    row = next(ln for ln in pareto.splitlines() if "LiNR V2" in ln)
    assert "$0.5000$" in row and "$1.100$" in row and "SilverTorch" not in pareto
    md = (tmp_path / "out" / "report.md").read_text()
    assert (
        "### no entry: 1" in md and "### perf missing: 1" in md and "### quality missing: 0" in md
    )
    assert '"algo": "silvertorch"' in md.split("### no entry: 1")[1].split("###")[0]

    assert c.prov["n_records"] == 3  # the accepted sources: v2 old + new, v1 new
    old = next(r for r in records.read_records(p) if r["env"]["code_version"] == "old")
    old["env"] = {**ENV, "code_version": "old", "git_branch": "dev/x"}
    records.append_record(p, old)  # the reused quality record now comes from a branch
    branch = _generate(root, tmp_path / "branch", manifest=manifest, gate="D1")
    assert not branch.prov["citable"] and any("dev/x" in b for b in branch.prov["blockers"])

    latest = _generate(root, tmp_path / "latest")  # no manifest: every record, as before
    assert sum(r["algo"] == "linr_v2" for r in latest.recs) == 2


def test_the_shipped_manifest_and_claims_load():
    m = report.load_manifest(report.CLAIMS.parent / "campaign.yaml")
    assert m["default"]["quality"]["code_version"] == "f01255f106214ec0f540352d0e2a5cd90b84b6a1"
    assert all(len(e[s]["code_version"]) == 40 for e in m["entries"] for s in ("quality", "perf"))
    claims = yaml.safe_load(report.CLAIMS.read_text())["claims"]
    assert [c["id"] for c in claims] == [f"C{i}" for i in range(1, 8)]
    assert all(c["verdict"] is None and c["ours"] for c in claims)
