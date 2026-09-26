---
chain: "seqrec-encoder"
branch: "main"
parent: "2026-09-26-132708000-dispatch-h100-hstu-and-beyond.md"
nextStep: "Pod orchestrator: verify this plan against origin/staging (the planner read a checkout 40 commits behind; KuaiRand ETL facts came from the remote notes), challenge it where the research note disagrees, then dispatch implementation step 1-7 as one fable worker on dev/hstu."
created: "2026-09-26T10:39:27Z"
---

# Architecture plan: one `Encoder`, three blocks, two losses, one trainer (dev/hstu)

Produced by a fable planning agent (read-only) for the orchestrator. A plan, not a
decision record: the pod orchestrator owns it from here and may amend it.

## Findings that shape the plan
- Every prepped `train.parquet` has one column, `item_ids`; `val/test.parquet` have `item_ids` + `targets`. All three ETLs carry a `timestamp` list through `timesplit` (gathers every list column in lockstep) and drop it at the final `select`. Action types are filtered away upstream (Listen+, `is_read`, `is_click`): no dataset exposes one today. KuaiRand train rows are non-overlapping 201-id windows (561,486 rows); 32,038,726 table rows.
- KuaiRand OOM: 32,038,726 x 128 fp32 = 16.4 GB table; dense grad + two AdamW moments = 65.6 GB before activations; `negs_per_pos=256` gathers `[P,256,128]` and OOMs on first backward; 128 peaks at 77.2 GiB. `_resume.pt` was 49 GB and `hub.py` `CKPT_ALWAYS_IGNORE` does not skip it.
- Bench only touches `training.encode.encode_split` (`evaluation/bench/inputs.py:34`): needs `load_model_for_eval`, `get_output_embeddings().weight[1:]`, `predict_last`. `tests/test_dependency_direction.py` allows `training -> eval_datasets.{hub,layout,timesplit}` only.
- Published checkpoint state_dict keys: `item_embedding`, `position_embedding`, `encoder.layers.N.*`, `final_norm`, `output_embedding`; 500M ones ship no `config.json` (`D128_DROP05_DEFAULTS` in encode.py).
- Hub repos hold eval inputs + checkpoints only, not `train/val.parquet`: trainer inputs come from `eval-data <ds> prep`. (Orchestrator session already ran `eval-data yambda prep --variant 500m --output-dir /data/yambda-500m/trainer` on the pod — without timestamps; redo after step 9. Goodreads needs `goodreads all` (download+convert, ~/datasets -> /data/_raw symlink exists) then `prep`; not started.)

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

## 6. Experiment protocol
Per run: test NDCG@10, R@100 (+NDCG@100, R@10, coverage), best epoch, epoch_time_sec, samples/s, peak GPU mem, GPU name, config.json, log under `docs/artifacts/seqrec-encoder/<dataset>-<encoder>-<loss>-d<D>/`.
0. Gate A (no training): evaluate fetched goodreads d64/d128 and yambda-500m d64 checkpoints through the new loader; NDCG@10 and R@100 equal `eval_quality.json` to 4 decimals.
1. Gate B: retrain `sasrec + gbce`, K=256 shared, goodreads d64, published recipe; within ±0.002 NDCG@10; reference throughput/memory row.
2. Ladder on goodreads d64 (shared: B=256, L=200, lr 1e-3, wd 0, warmup 1000, patience 10, cap 100; dropout 0.5 sasrec / 0.1 llama+hstu; `K=8192, M=4096, tau=0.05`, logQ on): L1 sasrec+sampled_softmax; L2 llama H=256 4 blocks 4 heads; L3 hstu same size (+use_time once timestamps exist); L4 winner at H=512, 6 blocks. Ablate rowwise_adagrad on winner (< 1% rel loss allowed).
3. Winner at d128 goodreads, then yambda-500m d64/d128, only epochs/patience changed; success = beat 0.0813 / 0.1489 at d64.
4. Only then KuaiRand d128 (reuse_item_embeddings, rowwise_adagrad, eval_max_users 4096, eval_every > 1), matched to the A100 gSASRec wall clock.
Stop climbing when a step gains < +2% rel NDCG@10 on goodreads or regresses on yambda; 6 h cap per run; one GPU job at a time.

## 7. Ordered steps
1 config (S). 2 model Encoder + blocks, delete GSASRec (M-L). 3 encode/evaluate/test fixture (S). 4 Gate A (S). 5 dataset (S). 6 losses (M). 7 train.py + RowwiseAdagrad + compile + throughput logging + cli + hub ignore list (M). 8 Gate B (~1 h GPU). 9 ETL timestamps + hash check (S code, hours CPU; overlap with L1-L2). 10 ladder runs + docs (L).
Risks: TransformerEncoder vs per-layer numerics (Gate A); fully-masked rows; mask inside compiled body -> recompiles; sparse grads vs clip/autocast; logQ + in-batch is popularity-heavy on yambda (ablate); re-prep must reproduce item_id_map exactly; goodreads prep memory-heavy (129 GB cgroup note in ETL; pod has ~2 TB RAM); KuaiRand eval per epoch stays minutes.
