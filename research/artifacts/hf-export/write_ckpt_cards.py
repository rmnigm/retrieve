"""Neutral README.md and config.json for each shipped checkpoint (files are unlinked first: the
archive holds hardlinks of the Hub download)."""
import json
from pathlib import Path

A = Path("/data/anon-export")
for ck in sorted(A.glob("eval-*/checkpoints/*")):
    ds = ck.parent.parent.name.removeprefix("eval-")
    cfg = json.loads((ck / "config.json").read_text())
    q = json.loads((ck / "eval_quality.json").read_text())
    tm = json.loads((ck / "train_metrics.json").read_text())
    cfg["data_dir"] = f"data/{ds}/trainer"
    cfg["checkpoint_dir"] = f"data/{ds}/checkpoints/{ck.name}"
    cfg["wandb_project"] = "encoder"
    cfg["wandb_run_name"] = ck.name
    for name, body in (("config.json", json.dumps(cfg, indent=2) + "\n"),):
        (ck / name).unlink()
        (ck / name).write_text(body)
    m = q["metrics"]
    card = f"""# {ck.name}

Sequential encoder for `{ds}`, trained with `train run` of this repository's `evaluation/training`:
gSASRec body ({cfg['num_blocks']} blocks, {cfg['num_heads']} heads, ffn {cfg['ffn_hidden_dim']}, dropout {cfg['dropout']}),
sampled softmax (in-batch {cfg['inbatch_negatives']} + uniform {cfg['num_negatives']}) with logQ correction,
L2-normalised output table, D={cfg['embedding_dim']}, max_seq_length {cfg['max_seq_length']}, seed {cfg['seed']},
up to {cfg['num_epochs']} epochs with patience {cfg['patience']}; the weights are those of the best validation ndcg@10 epoch.
The benchmark encodes the first 10,000 users of `test.parquet` with it (`config/{'goodreads' if ds.startswith('goodreads') else ds}.yaml`).

## Test metrics (full catalogue, test split, mask_history={str(q['mask_history']).lower()})

""" + "\n".join(f"- {k}: {v:.4f}" for k, v in m.items()) + f"""

## Files

- `best_model.pt`: the model `state_dict`
- `item_embs.pt`: the scoring table, `[num_items + 1, {cfg['embedding_dim']}]`, row 0 padding
- `config.json`: the training configuration (`TrainConfig`)
- `item_id_map.json`: the dataset's id map, equal to the dataset root's
- `eval_quality.json`: the test evaluation above; `train_metrics.json`: per-epoch loss and validation metrics ({len(tm.get('epoch_losses', []))} epochs)

Loaded by the harness through `training.encode.load_model_for_eval` (docs/system/checkpoints.md).
"""
    (ck / "README.md").unlink()
    (ck / "README.md").write_text(card)
    print(ck, "ok")
