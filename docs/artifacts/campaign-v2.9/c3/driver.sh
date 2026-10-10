#!/usr/bin/env bash
# Night queue item 6 (C3 coverage): LiNR V1 (`linr_v1_filter_mask`) torch.compile max-autotune vs Triton, clause, at p 0.01 / 0.1 / 1, bs 1 + 16, k 100,
# seed 0, campaign-v2.9, on every bench staged on pod d (goodreads-synth 0.8 M d128, arxiv-synth 3 M d128, yfcc10m-synth 10 M d192, laion30m-synth
# 30 M d256), one process per dataset, pod d GPU 1. Plain (uncompiled) torch V1 only at 0.8 M, where the pilot's cells exist (controller: skip it at
# 10-30 M; 515-917 s a cell at 0.8 M). Oracle blobs are not
# moved (laion's GPU-0 job reads laion30m's). Hub campaign-v2.9/<dataset>-c3, one upload per dataset.
#   GPU=1 setsid nohup flock -n /scratch/gpu1.lock bash driver.sh > /scratch/v29/c3.driver.log 2>&1 &
export TAG=campaign-v2.9 LEG=c3 REPO=/scratch/wt/v29-axsynth PY=/venvs/d-run-v29-ax/bin/python GPU=${GPU:-1} R=/scratch/campaign-v29/c3-unused
CFG=/scratch/v29/c3-config
. "$(dirname "$(readlink -f "$0")")/../../campaign-v2.5/pod-d/common.sh"
rm -rf "$CFG" && cp -r "$REPO/evaluation/config" "$CFG"
printf 'c3:\n  datasets: [goodreads-synth, arxiv-synth, yfcc10m-synth, laion30m-synth]\n  dims: [128, 192, 256]\n  filter_kinds: [clause]\n  ks: [100]\n  batch_sizes: [1, 16]\n  seeds: [0]\n  sweeps:\n    goodreads-synth: [p001, p01, p1]\n    arxiv-synth: [p001, p01, p1]\n    yfcc10m-synth: [p001, p01, p1]\n    laion30m-synth: [p001, p01, p1]\n  arms:\n    - {algo: linr_v1_filter_mask, backends: [triton]}\n    - {algo: linr_v1_filter_mask, backends: [torch], build: {compile: [max-autotune]}}\n    - {algo: linr_v1_filter_mask, backends: [torch], datasets: {goodreads-synth: {}}}\n' >> "$CFG/suites.yaml"
for DS in goodreads-synth arxiv-synth yfcc10m-synth laion30m-synth; do
  R=/scratch/campaign-v29/$DS-c3
  mkdir -p "$R/logs"
  step check-$DS check --dataset $DS --dim "$(dim $DS)"
  step oracle-$DS oracle --dataset $DS --suite c3 --dim "$(dim $DS)" --config-dir "$CFG"
  NARROW="--config-dir $CFG" stream $DS c3 "linr_v1_filter_mask|triton torch"
  n=$($PY -c 'import json,sys; rs=[json.loads(l) for l in open(sys.argv[1])]; print(sum(r["status"]=="ok" for r in rs), sum(r["status"]!="ok" for r in rs))' "$R/c3/$DS-d$(dim $DS).jsonl")
  echo "$(date -Is) $DS ok / not ok: $n"
  want=6; [ $DS = goodreads-synth ] && want=9
  [ "$n" = "$want 0" ] || { note pod-d "$LEG $DS: ok / not-ok $n; driver stopped"; exit 6; }
  upload campaign-v2.9/$DS-c3
  msg="$LEG $DS ($want cells: V1 triton vs torch.compile, + plain torch at 0.8 M) done on pod d GPU $GPU at $cv; Hub campaign-v2.9/$DS-c3: $UP"
  note pod-d "$msg"; note control "$msg"
done
finish
