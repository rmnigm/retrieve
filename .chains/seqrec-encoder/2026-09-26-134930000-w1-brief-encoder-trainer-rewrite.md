---
chain: "seqrec-encoder"
branch: "w1-encoder"
parent: "2026-09-26-134900000-user-amendment-opus-only-two-agents.md"
nextStep: "W1 (opus, herdr agent w1-encoder, pane w1:p3, worktree /scratch/wt/encoder, branch dev/hstu-encoder): implement this brief; Gate A then Gate B; hand back with a report."
created: "2026-09-26T10:49:30Z"
---

# W1 brief: encoder/trainer rewrite (dispatched as sent)

Fork from `main`: one worker branch per dispatched agent. It holds the brief exactly as sent, the orchestrator's later instructions, and the worker's report. Scope: `evaluation/training/`, `eval_datasets/hub.py`, training tests, `datasets.md` § Training, `checkpoints.md`. What merges back: branch `dev/hstu-encoder` after review (outcome on `main`).

**Superseded in part** by the next note (amendment 1: no llama block, softmax HSTU, no logQ, no row-wise Adagrad) and by the per-position gBCE fix in the report.

---

You are a constrained worker dispatched by the dev/hstu orchestrator (another Claude
session in a herdr pane on this pod). Read `CLAUDE.md` (hard rules),
`docs/contracts/coding-guidelines.md` and `docs/contracts/agent-orchestration.md` §5 first.

**Standing lines.** The code wins over any doc, note or memory: if the wiki and the code
disagree, trust the code and fix the doc. State the mechanism of a bug before fixing it.

**No subagents.** Do not use the Agent/Task tool, Workflow, or spawn any other agent or
Claude process, not even for research. Do all of the work yourself in this session.

## Step
Replace gSASRec's model/trainer in `evaluation/training/` with one generic `Encoder`
(blocks `sasrec | llama | hstu`), two losses (`gbce | sampled_softmax`), GPU-side shared
negatives, an optional row-wise Adagrad for the item table, `torch.compile` on the dense body,
and throughput logging; keep the checkpoint/encode contract. Pass Gate A and Gate B.
The ladder of experiments is NOT your step (the orchestrator dispatches those later).

## Read only
- `evaluation/training/*.py`, `evaluation/tests/training/`, `evaluation/tests/test_dependency_direction.py`
- `evaluation/eval_datasets/hub.py` (lines ~80-95, `EPOCH_SNAPSHOT_PATTERN`, `CKPT_ALWAYS_IGNORE`)
- `evaluation/bench/inputs.py` (the only bench consumer: `training.encode.encode_split`)
- `docs/system/datasets.md` § "Training — `evaluation/training/`" and `docs/system/checkpoints.md`
- `docs/validation.md` (to add your rows)

## Chain context: the architecture plan (§1-5), which you implement
## 1. Module layout (`evaluation/training/`)
| file | owns | change |
|---|---|---|
| `config.py` | `TrainConfig` (renamed from `GSASRecConfig`, no alias). New fields: `encoder: sasrec\|llama\|hstu`, `hidden_dim: int\|None` (body width H; None = D), `num_blocks`, `num_heads`, `ffn_hidden_dim`, `dropout`, `use_time: bool`, `time_buckets=64`, `loss: gbce\|sampled_softmax`, `num_negatives` (shared uniform, replaces `negs_per_pos`), `inbatch_negatives`, `logq: bool`, `temperature`, `loss_chunk=4096`, `table_optimizer: adamw\|rowwise_adagrad`, `compile: bool`, `warmup_steps`. `reuse_item_embeddings` stays. | S |
| `model.py` | `Encoder`: `item_embedding [N+1,D]` (sparse when table_optimizer != adamw), optional `in_proj D->H`, `position_embedding [L,H]` (sasrec only), `time_gap_embedding [time_buckets,H]` (use_time), `blocks: ModuleList`, `final_norm`, `out_proj H->D` (identity if H==D), optional `output_embedding`. `forward(items [B,L], timestamps\|None) -> [B,L,D]`; `predict_last`; `get_output_embeddings()` unchanged. Blocks `SASRecBlock` (wraps `nn.TransformerEncoderLayer`, gelu, norm_first), `LlamaBlock`, `HSTUBlock`, all `block(x, attn_mask [B,1,L,L] bool, rel_time [B,L,L]\|None)`. `BLOCKS` dict. `GSASRec` deleted. | M-L |
| `dataset.py` | `load_sequences(parquet, max_length) -> dict[str,Tensor]`: pre-padded `items [U,L+1]` int64 (+ `timestamps` if column exists); `train_batches(tensors, batch_size, generator)` via `randperm`. No DataLoader/workers/CPU negative sampling. | S |
| `losses.py` | `gbce_loss(queries [P,D], pos_ids [P], neg_ids [K], table, num_items, t)`, `sampled_softmax_loss(queries, pos_ids, neg_ids, table, log_q [N+1]\|None, temperature, chunk)`; `LOSSES` dict. One shared negative id vector. | M |
| `train.py` | model-agnostic loop: build Encoder, two optimizers, `step()` (autocast -> model -> loss -> clip body grads -> both optimizers); eval/early-stop/resume/artifacts as now. Command `train run`. | M |
| `evaluate.py` | EvalDataset returns `timestamps` when present; metric code untouched (pinned by test). | S |
| `encode.py` | `load_model_for_eval` builds Encoder from config.json (+ `D128_DROP05_DEFAULTS` with encoder=sasrec when absent); renames `encoder.layers.` -> `blocks.` in legacy state dicts. Timestamps when `cfg.use_time`. | S |
| `checkpoints.py`, `cli.py` | register `run`; snapshot pattern `{encoder}-ep{N}-...`. Outside training tree (flag for orchestrator): `eval_datasets/hub.py` `CKPT_ALWAYS_IGNORE` += `_resume.pt`, `encoded_queries_v2.pt`; `EPOCH_SNAPSHOT_PATTERN` new prefix. | S |

gSASRec becomes `encoder: sasrec` of `Encoder` (two bodies would be the self-compat shim the contract forbids). Cost: key rename in loader + Gate A.

## 2. Encoder design
Mask built once: `key_valid = items != 0`; `attn_mask = causal & (key_valid[:,None,None,:] | eye)` (diagonal keeps rows non-empty, no NaN on left-padded rows). Input: `x = in_proj(item_embedding(items)) (+ position_embedding, sasrec) (+ time_gap_embedding(time_bucket(t_i - t_{i-1})), use_time)`; emb dropout.
- `SASRecBlock`: identical math to current `nn.TransformerEncoder`.
- `LlamaBlock`: RMSNorm -> qkv Linear(H,3H,no bias) -> RoPE -> `F.scaled_dot_product_attention(mask)` -> o_proj -> residual; RMSNorm -> SwiGLU -> residual.
- `HSTUBlock` (paper eq. 1-3, pure PyTorch): `n = norm(x)`; `U,V,Q,K = SiLU(f1(n)).chunk(4)`; `A = SiLU(QK^T/sqrt(hd) + rab) * mask / L`, `rab = rel_pos_bias[clip(j-i)]` (+ `rel_time_bias[time_bucket(t_i - t_j)]` when use_time); `y = f2(norm(AV) * U)`; residual. `[B,heads,L,L]` at B=256, heads=4 = 82 MB bf16; no Triton.
`hidden_dim` is the "bigger" knob: table stays `[N,D]` D in {64,128,256}, body at H=256-512, `out_proj` to D. Timestamps are the only optional input now (all three ETLs hold them). Actions deferred (no dataset exposes one).
Data layer: each ETL's final `select` adds `timestamps` next to `item_ids` (goodreads `etl/goodreads.py:582-624`, yambda `etl/yambda.py:139-178`, kuairand `cmd_prep`). Re-prep must leave `item_id_map.json` and `item_ids` byte-identical (hash before/after).

## 3. Loss and negatives
GPU sampling in `step()`: `neg_ids = randint(1, N+1, (K,))` shared by all P positions -> one `[K,D]` gather (K=256: 128 KB vs 6.7 GB) — removes the KuaiRand OOM.
- `gbce_loss`: `neg = q @ E_neg^T [P,K]`, existing float64 calibration with alpha=K/(N-1).
- `sampled_softmax_loss`: candidates `[pos_ids[perm[:M]]; neg_ids]` (M in-batch, K uniform), gathered once; per chunk `logits = q @ E_C^T / tau - log_q[C]` (logQ from train bincount), accidental hits -> -inf, positive column appended, CE. ~60 lines total.

## 4. Efficiency
- bf16 autocast, TF32, fused AdamW for body; add linear warmup.
- `table_optimizer=rowwise_adagrad`: `nn.Embedding(sparse=True)` + ~25-line `RowwiseAdagrad` (state [N+1] fp32). KuaiRand D=128: ~16.6 GB vs 82 GB dense; `reuse_item_embeddings=True` there. `SparseAdam` as zero-code fallback (49 GB). `clip_grad_norm_` only over body params.
- `torch.compile` on the body (post-embedding blocks + norm + out_proj); lookup + chunked loss eager. Measure before keeping; per-job `TORCHINDUCTOR_CACHE_DIR`.
- Pre-padded tensors + randperm batching, pinned, non_blocking.
- Jagged/packed: not now (L=200).

## 5. Checkpoint / encode contract
config.json records encoder, hidden_dim, use_time, loss fields. `item_embs.pt`, `best_model.pt`, `eval_quality.json`, `train_metrics.json` keep names; `train_metrics.json` gains `samples_per_sec`, `epoch_time_sec`, `gpu_name`. `encode_split` returns unchanged. `test_encode.py` fixture -> `Encoder(encoder="sasrec")`. `docs/system/checkpoints.md` loading snippet updated.


## Amendments to the plan (orchestrator decisions; these win over the plan text above)
- **A1 normalized embeddings.** Add config `normalize: bool`. When on, L2-normalize the query
  and the item vectors inside `sampled_softmax_loss` and in `encode.py` (queries from
  `predict_last`, items from `get_output_embeddings().weight[1:]`), so dot-product scoring of
  the stored `[N,D]`/`[B,D]` equals cosine and `temperature` (default 0.05) is meaningful.
  Default: on for `sampled_softmax`, off for `gbce`; record it in config.json. Evaluation
  (`evaluate.py`) must score with the same normalized vectors.
- **A2 logQ.** `logq: bool` default **False**. When on, each candidate column gets
  `-log q_j` from the mixed proposal (M in-batch positives ~ train item frequency, K uniform:
  `q_j = (M * p_train(j) + K / N) / (M + K)`, counts from a bincount over train `item_ids`);
  the positive column is not corrected. Accidental hits (candidate id == the row's positive)
  are masked to -inf.
- **A3 shared relative bias.** One `RelativeBias` module (relative position buckets, plus
  log-scaled time-delta buckets when `use_time`) used by `HSTUBlock`, and by `LlamaBlock`
  when config `attn_bias: none | rab` is `rab` (passed to SDPA as a float additive mask,
  softmax kept). Default `none`.
- **A4 timestamps are optional input.** Trainer parquet may or may not have a `timestamps`
  column (list[int64], unix seconds, same length as `item_ids`). Another worker is adding it
  to the ETLs right now; the parquet on disk today has none. `use_time=True` with no column
  must fail at load with a clear error; `use_time=False` ignores the column.
- **A5 row-wise Adagrad and bf16 table** implemented but **off by default**.
- **A6 CLI.** `train run` replaces `train sasrec` (no alias). Fix every copy of the old
  command/`GSASRecConfig` wording in `AGENTS.md`, `README.md`, `docs/`, `retrieve/docs/`
  (rule 4; `docs/system/datasets.md:709` and `:1046`, `AGENTS.md:153` at least).
- **A7 Gate B moves to yambda-500m d64** (not goodreads: goodreads trainer inputs do not
  exist yet; yambda d64 trained in 47 min on the A100). See Gates.

## Gates
- **Gate A (no training, GPU).** Through the new `load_model_for_eval` + `evaluate`, the
  published checkpoints reproduce their `eval_quality.json` test `ndcg@10` and `recall@100`
  **to 4 decimals**: `/data/goodreads-work-id/checkpoints/gsasrec-d64-drop0.5-id`,
  `.../gsasrec-d128-drop0.5-id`, `/data/yambda-500m/checkpoints/gsasrec-d64-drop0.5`
  (+ `gsasrec-d128-drop0.5`, which has no config.json: `D128_DROP05_DEFAULTS`). Run it once on
  staging's code first (before your change) and once after, and save both outputs. Mismatch
  = report it with the mechanism; do not loosen the tolerance.
- **Gate B (training, GPU, ~1 h).** Retrain `encoder=sasrec, loss=gbce`, shared uniform
  negatives K=256, on yambda-500m d64 with the published recipe
  (`/data/yambda-500m/checkpoints/gsasrec-d64-drop0.5/config.json`: 2 blocks, 2 heads, ffn 256,
  dropout 0.5, B=256, lr 1e-3, wd 0, 100 epochs, patience 20, eval_every 2, early stop
  ndcg@10), data `/data/yambda-500m/trainer`. Test NDCG@10 within **±0.002** of 0.0813 and
  R@100 within ±0.002 of 0.1489. Record epoch_time_sec, samples/s, peak GPU mem, GPU name, sm_mhz
  samples. Before launching, predict (write down) the new epoch time vs the A100's
  (2809 s / 100 epochs) — shared negatives removes the `[P,256,D]` gather.
- Existing suites green: `evaluation/tests/training/test_encode.py` (fixture moves to
  `Encoder(encoder="sasrec")`), `test_dependency_direction.py`. Add only the tests a gate
  needs: a mask test (left-padded row gives no NaN; output at the last position independent
  of padding tokens), and a loss test pinning sampled-softmax against a direct
  `F.cross_entropy` on a tiny hand-built case (logQ off and on). No other scaffolding.

## Branch/worktree
Worktree `/scratch/wt/encoder` on branch `dev/hstu-encoder` (off `dev/hstu`). Own venv:
`export UV_PROJECT_ENVIRONMENT=/venvs/wt-encoder; rp-sync /scratch/wt/encoder` and use
`uv run` with that env exported in every shell. Commit on your branch as you go; do not
push, do not merge; the orchestrator reviews and merges.

## Model + concurrency
opus. You are the **only GPU holder** until you hand back: one GPU job at a time, never two.
A second worker (W2) runs concurrently on CPU in `/scratch/wt/etl` editing
`evaluation/eval_datasets/etl/*`, ETL tests and the dataset-specific sections of
`docs/system/datasets.md`; do not edit those. You own `datasets.md` § Training only.
W2 writes new data only to `/data/goodreads-work-id/trainer` and `/data/yambda-500m/trainer.new`;
your Gate B reads `/data/yambda-500m/trainer` (unchanged). Each GPU job:
`TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/<job>`, checkpoints under `/scratch/ckpt/<job>`
(never in `/workspace`), logs `/scratch/logs/seqrec/<job>.log`, run with `nohup`/background
and poll; wandb off (`wandb_enabled=false`).

## Out of scope
Ladder runs (L1+), goodreads/KuaiRand training, ETL changes, Hub uploads of anything,
`docs/roadmap.md`, `retrieve/` (the library), Meta Triton kernels, FuXi/SCE/RECE, DDP.

## Verify commands
```
cd /scratch/wt/encoder
ruff check retrieve evaluation && ruff format --check evaluation/training
cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/ -q
python3 scripts/check_doc_links.py   # from repo root, zero broken
```

## Return
- Branch `dev/hstu-encoder` with commits (message ends with the Co-Authored-By line from your
  harness attribution instructions).
- `docs/validation.md`: Gate A and Gate B rows, **not yet validated / not citable**, H100,
  with numbers; `docs/system/datasets.md` § Training and `docs/system/checkpoints.md` current.
- `docs/artifacts/seqrec-encoder/gate-a/` and `.../gate-b-yambda-d64/`: the scripts/commands,
  small result JSON, config.json, train_metrics.json, log tail. Keep it small (no checkpoints,
  no full logs over ~1 MB; roadmap H1 is moving artifacts off git).
- Gate B checkpoint left at `/scratch/ckpt/gate-b-yambda-d64/`.
- A final message in this pane: what passed, what was skipped, what is unverified, the
  prediction vs measured epoch time, and any plan amendment you had to make and why.
  Then stop and stay idle.
