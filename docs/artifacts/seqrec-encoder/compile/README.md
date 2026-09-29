# torch.compile A/B on the Encoder body (H100, yambda-500m trainer split, sasrec d64 gbce)

`ab.txt`: four runs in order eager, compiled, eager, compiled; each 3 full epochs (358 steps
of B=256), no val eval (`eval_every=100`, test eval on 2000 users). Columns: per-epoch train
seconds, samples/s over the 3 epochs, per-epoch mid-epoch sm_mhz, peak bytes. The first
compiled run includes the cold inductor compile in epoch 0 (13.4 s); the second has a warm
`TORCHINDUCTOR_CACHE_DIR`. Steady state: eager 4.8-5.9 s/epoch, compiled 3.93-4.03 s/epoch.

    for c in false true false true; do
      TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/compile-ab uv run train run \
        data_dir=/data/yambda-500m/trainer checkpoint_dir=/scratch/ckpt/compile-$c \
        encoder=sasrec loss=gbce num_epochs=3 eval_every=100 eval_max_users=2000 \
        wandb_enabled=false dropout=0.5 compile=$c
    done
