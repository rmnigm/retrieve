---
title: schema
created: 2026-09-26
updated: 2026-10-11
type: summary
tags: [process]
sources: [CLAUDE.md]
---

# Documentation: schema

The repository's documentation is two trees under one set of conventions:

- **`research/`** (this directory) is the agentic research wiki: the work
  queue, the standing decisions, what was measured and validated, the
  process contracts, the paper drafts and the raw scripts behind numbers.
- **[`docs/`](../docs/index.md)** is the released project's documentation:
  how the library and the harness work (`docs/system/`), for a user or
  contributor of the release. It may link into `research/` for a rationale
  or a measured number, never carry run state (a branch, a pod, a step's
  progress, a gate's history).

Both describe what is true **now**, and pages that describe code are kept
in sync with it in the same commit (CLAUDE.md rule 4). Where code and a
page disagree, the code is right and the page is the bug.

Both trees hold current information only: how things work, which gates
pass, standing decisions and constraints, the open queue, and plans to
improve. No history narration, no dated run logs, no done-items lists.
When a fact stops being true, change or delete it.

## Domain and scope

`docs/`: the `retrieve` library and the `evaluation` harness (`bench`,
`training`, `eval_datasets`) and the datasets, as code. `research/`: the
GPU pods, the work queue, the standing decisions, the validation state, the
two process contracts.

Out of scope: the library user guide, which ships in the sdist and lives in
[`retrieve/docs/`](../retrieve/docs/) (written for someone without a
clone; it names `docs/system/` but does not link it); the papers we
reproduce, frozen in `articles/` (gitignored); the reproducibility
paper's own sections in [`paper/`](paper/), which cite this wiki but are
drafts rather than wiki pages.

## Layout

```text
docs/                the released project's documentation
├── index.md         catalog of docs/system and retrieve/docs
└── system/          how the code works today (one page per subsystem)
research/            the research wiki
├── SCHEMA.md        this file (both trees)
├── index.md         catalog, one line per page
├── log.md           one line per structural change to either tree
├── roadmap.md       the open work queue: steps, gates, estimates, dependencies
├── backlog.md       known work that is not queued
├── decisions.md     standing decisions and constraints in force
├── validation.md    what is validated now: every gate's state, and the measured results that stand
├── contracts/       process contracts: agent orchestration, coding guidelines
├── environment/     the GPU pods: disks, budget, persistence
├── artifacts/       raw scripts behind measured numbers, per plan (outputs on the Hub)
└── paper/           reproducibility-paper sections (not wiki pages)
```

`docs/system/` keeps its path and its page names because code comments cite
them by section (for example `docs/system/kernels.md § Autotune
separation`). For the same reason its pages are not split even where they
pass the size threshold below. Section headings in `system/` are
therefore stable anchors: rename one only together with every comment
that cites it (`git grep 'docs/system/<page>'`).
`python3 scripts/check_doc_links.py` fails on a `§` citation whose heading
is gone.

## Conventions

- File names: lowercase kebab-case (`agent-orchestration.md`). The
  exceptions are `SCHEMA.md` and `README.md`.
- Links: relative markdown links (`[kernels](../docs/system/kernels.md#numerics)`),
  not `[[wikilinks]]`. GitHub renders them, and
  `python3 scripts/check_doc_links.py` (also a pre-commit hook) checks
  them. Keep that checker at zero broken links.
- Cite code as repo paths, and link the file when a reader will open it.
  Cite a line (`bench/cli.py:36`) only for a bug or a surprising line, and
  re-check the number when you edit the page.
- Numbers that are not yet citable carry "not yet validated" (CLAUDE.md
  rule 2). A page never presents a number whose gate has not passed as
  paper material.
- Commit hashes: the repository's history has been rewritten, so hashes
  in older records do not resolve. Never link a commit. Write a hash as plain text
  only where it names a recorded `code_version`.

## Frontmatter

Every page under `docs/` and `research/` (except `research/paper/` and
`research/artifacts/`) starts with:

```yaml
---
title: kernels            # page name
created: 2026-09-26       # first day as a wiki page
updated: 2026-09-26       # last material change
type: entity | concept | comparison | query | summary
tags: [kernels, library]  # only tags from the taxonomy below
sources: [retrieve/src/retrieve/ops/triton/]  # repo paths the page describes
confidence: high | medium | low   # optional
contested: true                   # optional; see the contradiction policy
---
```

Types used here: `entity` for a package, a dataset family or the box;
`concept` for a mechanism (filtering, the measurement protocol); `summary`
for the roadmap, the backlog, the decisions and the contracts.

## Tag taxonomy

- library: the `retrieve` package (modules, ops, indexing, functional)
- kernels: Triton kernels, the reference ops, Meta's official ops
- filtering: attribute filters, bloom and exact clause predicates
- harness: the `bench` package, suites, records, reports
- datasets: `eval_datasets`, the on-disk layout, dataset ETL
- training: the `training` package, gSASRec checkpoints
- testing: the pytest suites and their gates
- environment: the GPU pods, disks, venvs
- process: how work is dispatched, written, validated
- validation: gates and the measured results that stand
- roadmap: the queue, its steps and gates
- decisions: standing decisions

Add a tag here before using it on a page.

## Page thresholds

- A topic gets its own page when it is central to one subsystem or recurs
  across several. Otherwise extend an existing page.
- No page for a passing mention.
- Split a page past ~200 lines, except the `docs/system/` pages (see Layout).

## Validation

When a step finishes (CLAUDE.md rules 6-7), its result is written as
current state, not as a record of the run:

- [validation.md](validation.md): the gate's row now says what passes, on
  which environment, and what is still unverified. A measured result that
  stands (a speedup, a parity figure, a recall) goes in the same page with
  a link to its artifacts.
- The `docs/system/` page for the behaviour that changed.
- The raw scripts under `artifacts/<plan>/`; raw outputs (records, dumps,
  logs) on the Hub under `artifacts/<plan>/`, listed in
  [artifacts/hub-index.md](artifacts/hub-index.md), so the numbers can be
  re-derived.

A superseded result is replaced, not appended to.

## Contradiction policy

Keep both claims, each with its source, set `contested: true` in the
frontmatter, and flag it to the user. Never overwrite a contradiction
silently.
