---
chain: "roadmap"
branch: "main"
parent: "2026-05-18-120000000-roadmap-started-stages-1-2-and-refactor-track.md"
nextStep: "Phase A: A1 golden cells on the old harness, A2 pin and build Meta's package, A3 measure the official ops."
created: "2026-09-05T12:00:00Z"
---

# Roadmap history, part 2: goal, decisions of 2026-09-05, and the phased queue as written

## §0 Goal and rules
Goal: a reproducibility paper on SilverTorch (Meta) and LiNR (LinkedIn): our from-the-paper Triton reimplementation, Meta's official kernels as the reference, one correct benchmark harness, public datasets up to the papers' scale (plan P). Scheduling rule: no dates; every step done in order, nothing optional or conditional on a deadline; the A100 is the bottleneck, so GPU steps first and CPU steps in parallel. Conventions: a plan stays live while it is an instruction set or its validation has not signed off, then archives; system behaviour lives in the system docs, never in plans; each step has a checkbox, a plan section, where it runs (`A100` needs GPU time; `Mac` was a legacy label meaning "needs no GPU time" after the Mac target was dropped 2026-09-06), a gate and what it unblocks; the validation record goes to the plan's own record section, then the checkbox flips with date and commit; never start a step whose dependencies are unchecked; never reorder without rewriting the dependency notes.

Decisions 2026-09-05, do not reopen: Meta's `meta-recsys/silvertorch` ops become the reference backend (`backend="official"`); our CUDA C++ and CuTe DSL backends are deleted after the official parity gate; Triton stays and gets the kernel effort; the harness is rewritten rather than patched.

Phase E dataset decision (2026-09-05, final): arXiv and Goodreads (rerun), YFCC-10M, PubMed + MedCPT, Semantic Scholar SPECTER2 (OpenAlex fallback), KuaiRand-27K. Dropped: Amazon Reviews 2023, Yambda-full and Cohere Wikipedia (scale without meaningful filters, closed query encoder; the existing Yambda unfiltered runs stay as they are).

## §1 The queue as written (plan letters: H harness v2, O official integration, P paper, D datasets, L library refactor, V package layout, X boundary)
- Phase A: A0 make the Mac run the repo (dropped 2026-09-06); A1 golden cells on the old harness (H WP-0); A2 pin and build Meta's package (O WP-0; gate: build < 5 min, `torch.ops.st.fused_kmean_ann` exists); A3 measure the official ops (O WP-1: bit order, syncs, launches, capture, parse cost, `per_embedding_scale` overflow); A4 merge `development` into `main` (on hold; user decides).
- Phase B: B1 official adapter + T1-T7 (O §5, WP-2); B2 official vs Triton bit-exact on the A100 (O WP-3; gate: int32 scores `torch.equal` on every regime, bloom ⊇ and FPR at matched memory; unblocks B4, never delete before it); B3 Triton vs official kernel-only and end to end (O §9a / §9b, WP-4; paper gap G2); B4 delete the CUDA C++ and CuTe backends (O §7, WP-5; tag `cuda-cute-backends-final`; archive their plans; gate: suite green, the grep hits only the archive); B5 bloom salt as a buffer (O TF-2).
- Phase C (harness v2): C1 measurement primitives, device metrics, algorithm table (H WP-1; incl. the `official` path in `PATHS`); C2 config matrix, inputs, oracle with pass rates (WP-2); C3 cell loop and CLIs, delete the old harness, rewrite the docs (WP-3); C4 validate against the golden on the A100 (WP-4; gate: quality within 1e-6, graph latency within 5 %, `cudagraph_skips == 0`, `jaccard_vs_first@100 == 1.0` torch vs triton, one official cell on goodreads d128 `c0_genre` at jaccard >= 0.99).
- Phase D: D1 the full campaign on the new harness (H WP-5 + O WP-7: four datasets x dims x {triton, torch, official} x seeds {0, 1, 2} on headline sweeps, `n_probe` in {24, 32}, the S9 ablation; closes P gaps G3, G4, G7, G8 and the goodreads oracle rerun); D2 Faiss, HNSW, cuBLAS, cuVS baselines (P G5, then G13 / G14); D3 bloom FPR and memory vs filter width (P G6); D4 every table and figure from the records (H WP-6).
- Phase E: E0 request the Semantic Scholar API key (the long pole for E3); E1 YFCC-10M with its shipped filtered GT (gate: our exact oracle reproduces the shipped GT on the 100k queries; one `none` + one filter cell); E2 PubMed + MedCPT (~36M articles); E3 a 50M-paper SPECTER2 slice (OpenAlex fallback: ~670 GB snapshot pass + 9-14 A100 h of encoding); E4 KuaiRand-27K and gSASRec over 32M videos (shared item table; two filter protocols); E5 the campaign on the new datasets (after D1).
- Phase F: F1 reframe the thesis as SilverTorch Algorithm 1 + the deviations table (P G1); F2 the "official vs reimplementation" section (O WP-9, from B3 + D1); F3 provenance and disclosure (P G9); F4 package the artifacts (tagged release, Zenodo DOI incl. the pinned official sdist, HF data, one-command reproduction, anonymised mirror); F5 write the paper (P §C.3 / §C.4, every table from D4).
- Phase G (after F): G-a transposed bloom index in Triton + retune (O §8, WP-8; gate: parity bit-exact, bloom kernel-only within 1.3x of official; then rerun B3); G-b extended experiments (P G10-G12, G15, G16); G-c a resource paper about the library; G-d deferred kernel optimizations (hardware popcount, allocator hygiene); G-e re-scope the parked export and live-update plans (future-work menu).

## §2 Dependencies (as drawn)
A1 -> A4; A1 -> C1 -> C2 -> C3 -> C4 -> C5 -> D1 -> D4 -> F2, F5; L1 -> L2 -> (C4 rerun); A2 -> A3 -> B1 -> B2 -> {B4, B3}; B3 -> F2; D2, D3 -> F2, F5; E0 first; E1-E4 ingest any time -> E5 after D1; F1, F3 any time; F4 after D1. The A100 critical path: (golden re-derive) -> C4 on the L2 layout -> B3 -> D1 -> D2 / D3 -> E5. CPU work fills the gaps.

## §3 Superseded and parked
The refactor validation runbook's harness steps replaced by A1; the thesis-results expansion (old item 4) folded into H §2 / §3 and D1; the goodreads oracle rerun (old item 2) is D1; deferred kernel optimizations (old item 3) are G-d; the parked export and live-update plans re-scope against L's layout; review items folded into L (A5, A8, A9, D3, T3, B.4) and V (§1.11, §1.12, §1.4).
