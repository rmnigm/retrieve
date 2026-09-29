"""Encoder trainer — produces the checkpoints the sequential benchmarks encode with.

bf16 autocast, uniform negatives drawn on the GPU (per position for gBCE, one shared vector
plus in-batch positives for sampled softmax), fused AdamW with linear warmup, optional
``torch.compile`` of the dense body, chunked full-catalog eval every ``eval_every`` epochs,
best-metric checkpointing with resumable RNG state. TF32 is *enabled* here, unlike the
benchmark harness.

Usage (``train run …``), the config surface, and what a finished run writes out:
docs/system/datasets.md § Training.
"""

from __future__ import annotations

import dataclasses
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

import click
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from training.config import TrainConfig
from training.dataset import load_sequences, load_val_transitions, target_mask, train_batches
from training.evaluate import evaluate
from training.losses import gbce_loss, sampled_softmax_loss
from training.model import Encoder, build_encoder


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


def _sm_mhz() -> float:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


def _wandb_init(config: TrainConfig):
    if not config.wandb_enabled:
        return None
    import wandb  # noqa: PLC0415 — slow; only when wandb_enabled

    return wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name or Path(config.checkpoint_dir).name,
        config=dataclasses.asdict(config),
    )


def target_frequencies(items: torch.Tensor, first: torch.Tensor, num_items: int) -> torch.Tensor:
    """``p_train [N+1]``: each item's share of the trained target positions (``target_mask``)."""
    counts = torch.bincount(items[:, 1:][target_mask(items, first)], minlength=num_items + 1)
    return counts.double() / counts.sum()


def logq_correction(
    p_train: torch.Tensor, candidates: torch.Tensor, m: int, k: int, n: int
) -> torch.Tensor:
    """``log q_j``, q_j the expected count of item j among M in-batch and K uniform draws."""
    return (m * p_train[candidates] + k / n).log().float()


def step_loss(
    model: Encoder,
    items: torch.Tensor,
    first: torch.Tensor,
    config: TrainConfig,
    num_items: int,
    p_train: torch.Tensor | None,
) -> torch.Tensor:
    inputs, targets = items[:, :-1], items[:, 1:]
    mask = target_mask(items, first)
    queries, pos_ids = model(inputs)[mask], targets[mask]
    table = model.get_output_embeddings().weight
    if config.loss == "gbce":
        shape = (pos_ids.shape[0], config.num_negatives)
        neg_ids = torch.randint(1, num_items + 1, shape, device=items.device)
        return gbce_loss(queries, pos_ids, neg_ids, table, num_items, config.gbce_t)
    neg_ids = torch.randint(1, num_items + 1, (config.num_negatives,), device=items.device)
    perm = torch.randperm(pos_ids.shape[0], device=items.device)[: config.inbatch_negatives]
    candidates = torch.cat([pos_ids[perm], neg_ids])
    log_q = None
    if p_train is not None:
        log_q = logq_correction(p_train, candidates, perm.shape[0], config.num_negatives, num_items)
    return sampled_softmax_loss(
        queries, pos_ids, candidates, table, config.temperature, config.normalize, log_q
    )


def resume_due(epoch: int, config: TrainConfig, stopping: bool) -> bool:
    """Whether ``epoch`` writes ``_resume.pt``: every ``resume_every`` epochs and the last one."""
    return stopping or (epoch + 1) % config.resume_every == 0 or epoch + 1 == config.num_epochs


def train(config: TrainConfig, resume: bool = False) -> None:
    set_seed(config.seed)
    _enable_tf32()
    device = torch.device(config.device)
    use_cuda = device.type == "cuda"
    num_items = config.num_items
    data_dir = Path(config.data_dir)
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = build_encoder(config, num_items).to(device)
    if config.compile:
        model.body = torch.compile(model.body)
    items = load_sequences(str(data_dir / "train.parquet"), config.max_seq_length, device)
    first = torch.zeros(items.shape[0], dtype=torch.long, device=device)
    if config.train_on_val:
        val_items, val_first = load_val_transitions(
            str(data_dir / "val.parquet"), config.max_seq_length, device
        )
        items, first = torch.cat([items, val_items]), torch.cat([first, val_first])
    p_train = target_frequencies(items, first, num_items) if config.logq else None
    n_batches = items.shape[0] // config.batch_size
    batches_per_epoch = min(config.max_batches_per_epoch or n_batches, n_batches)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=use_cuda,
    )
    warmup = max(config.warmup_steps, 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup))

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    gpu_name = torch.cuda.get_device_name(device) if use_cuda else "cpu"

    wandb_run = _wandb_init(config)
    metric = config.early_stop_metric
    eval_kw = {
        "num_items": num_items,
        "max_length": config.max_seq_length,
        "batch_size": config.eval_batch_size,
        "ks": config.eval_ks,
        "device": device,
        "mask_history": config.mask_history,
        "score_chunk": config.eval_score_chunk,
    }

    epoch_losses: list[float] = []
    epoch_times: list[float] = []
    sm_mhz: list[float] = []
    val_metrics_per_epoch: list[dict[str, float]] = []
    best_metric = -float("inf")
    best_path: Path | None = None
    steps_not_improved = 0
    global_step = 0
    start_epoch = 0
    resume_path = ckpt_dir / "_resume.pt"
    if resume and resume_path.exists():
        logger.info("Resuming from {}", resume_path)
        # set_rng_state takes CPU ByteTensors only; load_state_dict moves the rest to the device.
        state = torch.load(resume_path, weights_only=False, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["py_rng"])
        np.random.set_state(state["np_rng"])
        torch.set_rng_state(state["torch_rng"])
        if use_cuda:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_epoch = state["next_epoch"]
        best_metric = state["best_metric"]
        steps_not_improved = state["steps_not_improved"]
        global_step = state["global_step"]
        best_path = ckpt_dir / Path(state["best_path"]).name if state["best_path"] else None
        logger.info(
            "Resumed: start_epoch={} best_metric={:.4f} steps_not_improved={} global_step={}",
            start_epoch,
            best_metric,
            steps_not_improved,
            global_step,
        )
    if resume:
        # Bests saved after _resume.pt, or before the first one, belong to epochs being retrained.
        for snapshot in ckpt_dir.glob("sasrec-ep*.pt"):
            if snapshot != best_path:
                snapshot.unlink()
    resumable_best = best_path
    t0 = time.perf_counter()

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        epoch_loss = torch.zeros((), device=device)
        ep_t0 = time.perf_counter()
        batches = train_batches(items, first, config.batch_size)
        pbar = tqdm(range(batches_per_epoch), desc=f"Epoch {epoch}", mininterval=10)
        for batch_idx in pbar:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_cuda):
                loss = step_loss(model, *next(batches), config, num_items, p_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.detach()
            global_step += 1
            if batch_idx == batches_per_epoch // 2 and use_cuda:
                sm_mhz.append(_sm_mhz())
            if global_step % config.log_every == 0:
                loss_val = loss.item()
                pbar.set_postfix(loss=f"{loss_val:.4f}")
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss_step": loss_val,
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

        avg_loss = epoch_loss.item() / batches_per_epoch
        epoch_time = time.perf_counter() - ep_t0
        epoch_losses.append(avg_loss)
        epoch_times.append(epoch_time)
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if use_cuda else 0.0
        samples_per_sec = batches_per_epoch * config.batch_size / epoch_time

        do_eval = not config.train_on_val and (
            (epoch + 1) % config.eval_every == 0 or epoch + 1 == config.num_epochs
        )
        val_metrics: dict[str, float] = {}
        if do_eval:
            val_metrics = evaluate(
                model,
                str(data_dir / "val.parquet"),
                max_users=config.eval_max_users,
                **eval_kw,
            )
            val_metrics_per_epoch.append({"epoch": epoch, **val_metrics})
        logger.info(
            "Epoch {} — loss {:.4f} | {:.1f} s, {:.0f} seq/s, peak {:.1f} GB | val {}",
            epoch,
            avg_loss,
            epoch_time,
            samples_per_sec,
            peak_mem_gb,
            json.dumps({k: round(v, 4) for k, v in val_metrics.items()}),
        )

        if wandb_run is not None:
            log_payload = {
                "train/loss_epoch": avg_loss,
                "train/epoch_time_sec": epoch_time,
                "train/samples_per_sec": samples_per_sec,
                "train/peak_gpu_mem_gb": peak_mem_gb,
                "train/epoch": epoch,
            }
            for k, v in val_metrics.items():
                log_payload[f"val/{k}"] = v
            wandb_run.log(log_payload, step=global_step)

        if val_metrics:
            cur = val_metrics[metric]
            if cur > best_metric:
                best_metric = cur
                steps_not_improved = 0
                if best_path is not None and best_path != resumable_best:
                    best_path.unlink()
                best_path = ckpt_dir / f"sasrec-ep{epoch}-{metric.replace('@', '')}{cur:.4f}.pt"
                torch.save(model.state_dict(), best_path)
            else:
                steps_not_improved += 1

        stopping = bool(val_metrics) and steps_not_improved >= config.patience
        if resume_due(epoch, config, stopping):
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
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
            # _resume.pt names the best it was written with; that file lives until the next one.
            if resumable_best is not None and resumable_best != best_path:
                resumable_best.unlink()
            resumable_best = best_path

        if stopping:
            logger.info("Early stopping at epoch {}.", epoch)
            break

    total_time = time.perf_counter() - t0
    peak_mem = int(torch.cuda.max_memory_allocated(device)) if use_cuda else 0

    if best_path is not None:
        model.load_state_dict(torch.load(best_path, weights_only=True))

    torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
    config.save(ckpt_dir / "config.json")
    item_embs = model.scoring_table().detach().cpu()
    item_embs[0, :] = 0.0
    torch.save(item_embs, ckpt_dir / "item_embs.pt")

    for fname in ("item_id_map.json", "item_attrs.parquet"):
        src = data_dir / fname
        if src.exists():
            shutil.copy2(src, ckpt_dir / fname)

    test_metrics: dict[str, float] = {}
    test_path = data_dir / "test.parquet"
    if test_path.exists():
        test_metrics = evaluate(model, str(test_path), **eval_kw)
        logger.info("Test metrics: {}", json.dumps(test_metrics, indent=2))
        with open(ckpt_dir / "eval_quality.json", "w") as f:
            json.dump(
                {
                    "split": "test",
                    "ks": list(config.eval_ks),
                    "mask_history": config.mask_history,
                    "metrics": test_metrics,
                },
                f,
                indent=2,
            )
        if wandb_run is not None:
            wandb_run.log({f"test/{k}": v for k, v in test_metrics.items()}, step=global_step)

    train_time = sum(epoch_times)
    with open(ckpt_dir / "train_metrics.json", "w") as f:
        json.dump(
            {
                "epoch_losses": epoch_losses,
                "val_metrics_per_epoch": val_metrics_per_epoch,
                "best_val_metric": {metric: best_metric} if val_metrics_per_epoch else {},
                "test_metrics": test_metrics,
                "total_time_sec": total_time,
                "epoch_time_sec": epoch_times,
                "samples_per_sec": len(epoch_times)
                * batches_per_epoch
                * config.batch_size
                / train_time,
                "peak_gpu_mem_bytes": peak_mem,
                "gpu_name": gpu_name,
                "sm_mhz": sm_mhz,
            },
            f,
            indent=2,
        )

    if wandb_run is not None:
        wandb_run.finish()


def _parse_override(text: str) -> tuple[str, object]:
    key, sep, raw = text.partition("=")
    if not sep or key not in {f.name for f in dataclasses.fields(TrainConfig)}:
        raise click.BadParameter(f"{text!r}: expected FIELD=VALUE with a TrainConfig field")
    try:
        return key, json.loads(raw)
    except json.JSONDecodeError:
        return key, raw


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="start from this JSON (TrainConfig.load) instead of the TrainConfig defaults",
)
@click.option("--resume", is_flag=True, default=False, help="continue from <ckpt>/_resume.pt")
@click.argument("overrides", nargs=-1)
def run(config_path: str | None, resume: bool, overrides: tuple[str, ...]) -> None:
    """Train an Encoder. OVERRIDES are TrainConfig FIELD=VALUE pairs, VALUE parsed as JSON
    when it parses (``loss=gbce num_negatives=256 warmup_steps=0``), else taken as a string."""
    fields = dict(_parse_override(o) for o in overrides)
    config = TrainConfig.load(config_path, **fields) if config_path else TrainConfig(**fields)
    train(config, resume=resume)


if __name__ == "__main__":
    run()
