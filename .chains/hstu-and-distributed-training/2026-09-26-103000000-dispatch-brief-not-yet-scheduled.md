---
chain: "hstu-and-distributed-training"
branch: "main"
nextStep: "NOT on the roadmap yet. Before dispatching: the orchestrator/user decides whether and when to schedule this (likely two separate roadmap items once scoped -- an HSTU feasibility question and a training-infra hardening step). Until then this is a ready-to-use brief, not an assignment. When scheduled: dispatch as a research-first step (opus, no code changes on the first pass), then a second, separate implementation dispatch only if the research says it's worth it."
created: "2026-09-26T10:30:00Z"
---

# Brief: HSTU feasibility + multi-GPU training efficiency (research first, not yet scheduled)

## Why this exists
Raised by the user while E4 (KuaiRand + gSASRec) was training on this
session's A100 box: KuaiRand-27K has an unusually large catalog (32M
items) against very few users (27,285), a regime neither SASRec nor
gSASRec's own paper validates anywhere close to (gSASRec's largest test
catalog is Gowalla at 1.27M items -- see
`.chains/e4-kuairand/2026-09-26-100000000-sasrec-baseline-research-no-comparable-number.md`
for the full research trail; no published baseline exists for this
dataset at all). Two follow-on questions came out of that conversation,
neither acted on in this session because both are new, unscheduled scope
requiring their own roadmap decision:

1. Would a different, already-open-source sequential architecture do
   noticeably better *out of the box* at this catalog scale, without
   requiring bespoke research?
2. This project's training code (`evaluation/training/`) has **zero**
   distributed-training support today (verified: `grep -rln
   "torch.distributed\|DistributedDataParallel\|DistributedSampler\|torchrun\|init_process_group"
   evaluation/training/` returns nothing) -- confirmed in this same
   session. If the user gets access to a second GPU (or more), there's no
   way to use it for training right now. Separately, there may be
   single-GPU training efficiency left on the table regardless of GPU
   count.

## Part A: HSTU feasibility (research only -- do not implement without a decision)

**What to check, by actually reading the source, not from memory:**
Meta's HSTU (Hierarchical Sequential Transduction Units), from "Actions
Speak Louder Than Words: Trillion-Parameter Sequential Transducers for
Generative Recommendations" (2024). Open-sourced as
`facebookresearch/generative-recommenders` on GitHub. Explicitly designed
for large-catalog, large-action-vocabulary industrial recommendation
(the same lineage as SilverTorch, which this project already reproduces)
and built to follow LLM-style scaling laws -- the closest thing to "this
might just work better at 32M items" without inventing something new.

Questions to answer, in order:
1. **License and integration shape.** Is `generative-recommenders`
   usable here (license compatible, importable as a dependency vs.
   needing vendored code)? What does its actual training loop / model
   API look like -- is it a drop-in replacement for
   `evaluation/training/model.py`'s SASRec-style module, or a much bigger
   restructure (different data format, different loss, different
   embedding-table sharding assumptions)?
2. **Output compatibility.** This project's retrieval benchmark
   (`evaluation/bench/`) only cares that a trained checkpoint produces
   `[N, D]` item embeddings and `[B, D]` query embeddings it can feed into
   SilverTorch/LiNR -- it does not care what sequence model produced them.
   Confirm HSTU's item representations are extractable in that shape
   without requiring HSTU's own generative-retrieval serving stack (which
   this project does not use and should not adopt wholesale).
3. **Actual reported numbers and validated scale**, read from the paper
   directly (not search snippets -- the last research pass in this
   session found a paper that looked relevant in snippets but used
   completely different datasets on direct read; don't repeat that
   mistake). What's the largest catalog HSTU's own paper reports results
   on? Does it include anything closer to KuaiRand's regime (huge catalog,
   few users) than gSASRec's Gowalla test?
4. **Effort estimate**, sized like the kernel-opt pass was (a real
   S/M/L/XL estimate, not a guess) -- comparable to a from-scratch
   integration, or genuinely a fast drop-in?

**Do not implement HSTU in this pass.** Report findings, an effort
estimate, and a recommendation (worth a dedicated roadmap step, worth
a quick spike, or not worth it) back to the orchestrator/user.

## Part B: multi-GPU + general training efficiency (research + a scoped implementation plan)

Read `evaluation/training/train.py`, `model.py`, `config.py`,
`encode.py` in full first, plus `docs/system/checkpoints.md` for the
existing training conventions, before proposing anything -- the goal is
additive infrastructure, not a rewrite (coding-guidelines D3/D4 still
apply: thin, no abstraction without a demonstrated need).

1. **DDP vs. FSDP -- pick deliberately, don't default to FSDP because the
   user mentioned it.** FSDP's value is sharding a model too large for
   one GPU's memory; SASRec/gSASRec-scale models (an embedding table plus
   a handful of transformer blocks) are very unlikely to need that --
   the actual memory pressure this session hit (E4's KuaiRand run) was
   the *item embedding table plus AdamW's two moments plus the
   negative-sampling gather tensor*, not the transformer body. Investigate
   whether **DDP** (replicate the model, shard the *data*) is the right
   fit, or whether FSDP's parameter sharding specifically helps the
   embedding table (it can, via `FSDP`'s support for large embedding
   layers, or `torch.distributed.checkpoint`'s sharded-embedding patterns)
   -- this needs a real memory-accounting comparison, not an assumption
   either way. Report the reasoning, then implement whichever wins.
2. **What "every other optimization" should concretely mean here** --
   investigate each, keep what measurably helps, document why on
   anything skipped:
   - Mixed precision: `docs/system/checkpoints.md` already mentions a
     "bf16 + fused AdamW recipe" used for one existing checkpoint
     (yambda-d64) -- check whether this is the trainer's default or an
     ad hoc flag, and whether it should become the default.
   - `torch.compile` on the training step (forward+backward), not just
     inference -- check if already used; if not, whether it helps at this
     model size without breaking gradient correctness.
   - Gradient checkpointing -- likely unnecessary at this model size, but
     confirm with a memory accounting before dismissing it.
   - The negative-sampling gather that OOM'd during E4's KuaiRand run at
     `negs_per_pos=256` (see
     `.chains/e4-kuairand/` handoff notes for the exact numbers: it needed
     dropping to 128) -- this is the actual, measured bottleneck this
     session hit, more concrete than any of the above. Worth its own
     look: can the negative gather be chunked or fused instead of
     materializing a full `[B, negs_per_pos, D]` tensor at once, the same
     pattern the kernel-opt pass just applied to the library's own
     quantize step?
   - Data loading: DataLoader worker count, prefetch factor, pinned
     memory -- quick to check, easy to get wrong by default.
   - `torch.distributed.checkpoint` (DCP) for sharded/resumable
     checkpointing if FSDP is adopted -- only relevant if Part B lands on
     FSDP over DDP.
3. **Gate**: whatever ships must not change single-GPU training's
   numerical behavior (same effective batch size and convergence, unless
   the orchestrator/user explicitly accepts a different effective batch
   size for multi-GPU runs) -- this needs stating explicitly in whatever
   step actually implements it, since silently changing effective batch
   size changes what a checkpoint's reported metrics mean.

## Out of scope for this brief
Actually running a comparative HSTU-vs-gSASRec training experiment
(that's a second step, after the feasibility research says it's worth
doing). Any change to `retrieve/` or the retrieval benchmark harness --
this is entirely about `evaluation/training/`. Committing to FSDP or DDP
before the memory-accounting comparison in Part B.1 is done.

## Return (whenever this actually gets dispatched)
Two things, and they can be two separate branches/workers since they're
independent: (1) an HSTU feasibility report with a clear recommendation,
no code; (2) either a DDP or FSDP implementation (whichever the
investigation picks) plus whichever of the "every other optimization"
items in B.2 measurably helped, each with a before/after number, on its
own `dev/<step>` branch. Neither touches `docs/roadmap.md` -- that's the
orchestrator's job once this is actually scheduled.
