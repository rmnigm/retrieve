from __future__ import annotations

import dataclasses
import json
import random
import shutil
import time
from pathlib import Path

import click
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from training.config import GSASRecConfig
from training.dataset import get_train_dataloader
from training.evaluate import evaluate
from training.losses import gbce_loss
from training.model import GSASRec


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _enable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def _wandb_init(config: GSASRecConfig):
    if not config.wandb_enabled:
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; continuing without it.")
        return None
    run = wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name or Path(config.checkpoint_dir).name,
        config=dataclasses.asdict(config),
    )
    return run


def train(config: GSASRecConfig, resume: bool = False) -> None:
    set_seed(config.seed)
    _enable_tf32()
    device = torch.device(config.device)

    with open(Path(config.data_dir) / "item_id_map.json") as f:
        num_items = len(json.load(f))

    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = GSASRec(
        num_items=num_items,
        max_seq_length=config.max_seq_length,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
        ffn_hidden_dim=config.ffn_hidden_dim,
        dropout=config.dropout,
        reuse_item_embeddings=config.reuse_item_embeddings,
    ).to(device)

    loader = get_train_dataloader(
        parquet_path=str(Path(config.data_dir) / "train.parquet"),
        batch_size=config.batch_size,
        max_length=config.max_seq_length,
        num_items=num_items,
        negs_per_pos=config.negs_per_pos,
    )
    batches_per_epoch = (
        len(loader)
        if config.max_batches_per_epoch is None
        else min(config.max_batches_per_epoch, len(loader))
    )

    use_cuda = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=use_cuda,
    )

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    val_path = Path(config.data_dir) / "val.parquet"
    test_path = Path(config.data_dir) / "test.parquet"

    wandb_run = _wandb_init(config)
    metric = config.early_stop_metric
    eval_ks = tuple(config.eval_ks)

    epoch_losses: list[float] = []
    val_metrics_per_epoch: list[dict[str, float]] = []
    best_metric = -float("inf")
    best_path: Path | None = None
    steps_not_improved = 0
    global_step = 0
    start_epoch = 0
    resume_path = ckpt_dir / "_resume.pt"
    if resume and resume_path.exists():
        logger.info("Resuming from {}", resume_path)
        state = torch.load(resume_path, weights_only=False, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        random.setstate(state["py_rng"])
        np.random.set_state(state["np_rng"])
        torch.set_rng_state(state["torch_rng"])
        if use_cuda and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_epoch = state["next_epoch"]
        best_metric = state["best_metric"]
        steps_not_improved = state["steps_not_improved"]
        global_step = state.get("global_step", 0)
        bp = state.get("best_path")
        best_path = Path(bp) if bp else None
        logger.info(
            "Resumed: start_epoch={} best_metric={:.4f} steps_not_improved={} global_step={}",
            start_epoch,
            best_metric,
            steps_not_improved,
            global_step,
        )
    t0 = time.perf_counter()

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        iterator = iter(loader)
        epoch_loss = 0.0
        ep_t0 = time.perf_counter()
        pbar = tqdm(range(batches_per_epoch), desc=f"Epoch {epoch}")

        for batch_idx in pbar:
            input_seq, target_seq, negatives = next(iterator)
            input_seq = input_seq.to(device, non_blocking=True)
            target_seq = target_seq.to(device, non_blocking=True)
            negatives = negatives.to(device, non_blocking=True)
            mask = target_seq != 0

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_cuda):
                hidden = model(input_seq)
                loss = gbce_loss(
                    hidden,
                    target_seq,
                    mask,
                    model.get_output_embeddings(),
                    uniform_negatives=negatives,
                    gbce_t=config.gbce_t,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            loss_val = loss.item()
            epoch_loss += loss_val
            global_step += 1
            pbar.set_postfix(loss=f"{epoch_loss / (batch_idx + 1):.4f}")

            if wandb_run is not None and global_step % config.log_every == 0:
                wandb_run.log(
                    {
                        "train/loss_step": loss_val,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )

        avg_loss = epoch_loss / batches_per_epoch
        epoch_losses.append(avg_loss)
        epoch_time = time.perf_counter() - ep_t0
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if use_cuda else 0.0

        do_eval = ((epoch + 1) % config.eval_every == 0) or (epoch + 1 == config.num_epochs)
        val_metrics: dict[str, float] = {}
        if do_eval:
            val_metrics = evaluate(
                model,
                str(val_path),
                num_items=num_items,
                max_length=config.max_seq_length,
                batch_size=config.eval_batch_size,
                ks=eval_ks,
                device=device,
                mask_history=config.mask_history,
                max_users=config.eval_max_users,
                score_chunk=config.eval_score_chunk,
            )
            val_metrics_per_epoch.append({"epoch": epoch, **val_metrics})
            logger.info(
                "Epoch {} — loss: {:.4f} | {}: {:.4f} | val: {}",
                epoch,
                avg_loss,
                metric,
                val_metrics.get(metric, float("nan")),
                json.dumps({k: round(v, 4) for k, v in val_metrics.items()}),
            )
        else:
            logger.info("Epoch {} — loss: {:.4f}", epoch, avg_loss)

        if wandb_run is not None:
            log_payload = {
                "train/loss_epoch": avg_loss,
                "train/epoch_time_sec": epoch_time,
                "train/peak_gpu_mem_gb": peak_mem_gb,
                "train/epoch": epoch,
            }
            for k, v in val_metrics.items():
                log_payload[f"val/{k}"] = v
            if val_metrics:
                log_payload[f"val/best_{metric}"] = max(
                    best_metric, val_metrics.get(metric, -float("inf"))
                )
            wandb_run.log(log_payload, step=global_step)

        if val_metrics:
            cur = val_metrics.get(metric, -float("inf"))
            if cur > best_metric:
                best_metric = cur
                steps_not_improved = 0
                if best_path is not None and best_path.exists():
                    best_path.unlink()
                best_path = ckpt_dir / f"gsasrec-ep{epoch}-{metric.replace('@','')}{cur:.4f}.pt"
                torch.save(model.state_dict(), best_path)
            else:
                steps_not_improved += 1

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "next_epoch": epoch + 1,
                "best_metric": best_metric,
                "steps_not_improved": steps_not_improved,
                "best_path": str(best_path) if best_path else None,
                "global_step": global_step,
                "py_rng": random.getstate(),
                "np_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if use_cuda else None,
            },
            resume_path,
        )

        if val_metrics and steps_not_improved >= config.patience:
            logger.info("Early stopping at epoch {}.", epoch)
            break

    total_time = time.perf_counter() - t0
    peak_mem = int(torch.cuda.max_memory_allocated(device)) if use_cuda else 0

    if best_path is not None:
        model.load_state_dict(torch.load(best_path, weights_only=True))

    torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
    config.save(ckpt_dir / "config.json")
    item_embs = model.get_output_embeddings().weight.detach().cpu()
    item_embs[0, :] = 0.0
    torch.save(item_embs, ckpt_dir / "item_embs.pt")

    data_dir = Path(config.data_dir)
    for fname in ("item_id_map.json", "item_attrs.parquet"):
        src = data_dir / fname
        if src.exists():
            shutil.copy2(src, ckpt_dir / fname)

    test_metrics: dict[str, float] = {}
    if test_path.exists():
        test_metrics = evaluate(
            model,
            str(test_path),
            num_items=num_items,
            max_length=config.max_seq_length,
            batch_size=config.eval_batch_size,
            ks=eval_ks,
            device=device,
            mask_history=config.mask_history,
            score_chunk=config.eval_score_chunk,
        )
        logger.info("Test metrics: {}", json.dumps(test_metrics, indent=2))
        with open(ckpt_dir / "eval_quality.json", "w") as f:
            json.dump(
                {
                    "split": "test",
                    "ks": list(eval_ks),
                    "mask_history": config.mask_history,
                    "metrics": test_metrics,
                },
                f,
                indent=2,
            )
        if wandb_run is not None:
            wandb_run.log({f"test/{k}": v for k, v in test_metrics.items()}, step=global_step)

    with open(ckpt_dir / "train_metrics.json", "w") as f:
        json.dump(
            {
                "epoch_losses": epoch_losses,
                "val_metrics_per_epoch": val_metrics_per_epoch,
                "best_val_metric": {metric: best_metric},
                "test_metrics": test_metrics,
                "total_time_sec": total_time,
                "peak_gpu_mem_bytes": peak_mem,
            },
            f,
            indent=2,
        )

    if wandb_run is not None:
        wandb_run.finish()


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="If set, load all hyperparameters from this JSON via GSASRecConfig.load and ignore other flags except --resume.",
)
@click.option("--data-dir", type=str, required=False, default=None)
@click.option("--checkpoint-dir", type=str, required=False, default=None)
@click.option("--embedding-dim", type=int, default=64)
@click.option("--num-blocks", type=int, default=2)
@click.option("--num-heads", type=int, default=2)
@click.option("--dropout", type=float, default=0.0)
@click.option("--batch-size", type=int, default=256)
@click.option("--lr", type=float, default=1e-3)
@click.option("--weight-decay", type=float, default=0.0)
@click.option("--gbce-t", type=float, default=0.75)
@click.option("--max-seq-length", type=int, default=200)
@click.option("--num-epochs", type=int, default=200)
@click.option("--max-batches-per-epoch", type=int, default=None)
@click.option("--patience", type=int, default=20)
@click.option("--negs-per-pos", type=int, default=256)
@click.option("--reuse-item-embeddings", is_flag=True, default=False)
@click.option("--device", type=str, default="cuda")
@click.option("--seed", type=int, default=42)
@click.option("--eval-batch-size", type=int, default=512)
@click.option("--eval-k", "eval_ks", type=int, multiple=True, default=[10, 100])
@click.option("--eval-every", type=int, default=1)
@click.option("--eval-max-users", type=int, default=None)
@click.option("--eval-score-chunk", type=int, default=262144)
@click.option("--mask-history/--no-mask-history", default=False)
@click.option("--early-stop-metric", type=str, default="ndcg@10")
@click.option("--wandb/--no-wandb", "wandb_enabled", default=True)
@click.option("--wandb-project", type=str, default="yambda-gsasrec")
@click.option("--wandb-run-name", type=str, default=None)
@click.option("--log-every", type=int, default=50)
@click.option("--resume", is_flag=True, default=False)
def main(
    config_path: str | None,
    data_dir: str | None,
    checkpoint_dir: str | None,
    embedding_dim: int,
    num_blocks: int,
    num_heads: int,
    dropout: float,
    batch_size: int,
    lr: float,
    weight_decay: float,
    gbce_t: float,
    max_seq_length: int,
    num_epochs: int,
    max_batches_per_epoch: int | None,
    patience: int,
    negs_per_pos: int,
    reuse_item_embeddings: bool,
    device: str,
    seed: int,
    eval_batch_size: int,
    eval_ks: tuple[int, ...],
    eval_every: int,
    eval_max_users: int | None,
    eval_score_chunk: int,
    mask_history: bool,
    early_stop_metric: str,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_run_name: str | None,
    log_every: int,
    resume: bool,
) -> None:
    if config_path is not None:
        config = GSASRecConfig.load(config_path)
    else:
        if not data_dir or not checkpoint_dir:
            raise click.UsageError(
                "Either --config <path> or both --data-dir and --checkpoint-dir must be provided."
            )
        config = GSASRecConfig(
            data_dir=data_dir,
            checkpoint_dir=checkpoint_dir,
            max_seq_length=max_seq_length,
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            ffn_hidden_dim=embedding_dim * 4,
            dropout=dropout,
            reuse_item_embeddings=reuse_item_embeddings,
            negs_per_pos=negs_per_pos,
            gbce_t=gbce_t,
            batch_size=batch_size,
            learning_rate=lr,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            max_batches_per_epoch=max_batches_per_epoch,
            patience=patience,
            device=device,
            seed=seed,
            eval_batch_size=eval_batch_size,
            eval_ks=tuple(sorted(eval_ks)),
            eval_every=eval_every,
            eval_max_users=eval_max_users,
            eval_score_chunk=eval_score_chunk,
            mask_history=mask_history,
            early_stop_metric=early_stop_metric,
            wandb_enabled=wandb_enabled,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
            log_every=log_every,
        )
    train(config, resume=resume)


if __name__ == "__main__":
    main()
