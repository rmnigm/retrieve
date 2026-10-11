#!/usr/bin/env bash
# V-YFCC synth (D; controller ruling 2026-10-10): the SYNTH-TRIM `synth` suite on yfcc10m-synth, whole (25 cells: V1 + V2 at
# {0.01, 0.1, 0.2, 0.5, 1.0}, bs 1 + 16; SilverTorch triton n_lists 4096 x {24, 256, 1024}; seed 0, k 100, clause), at the tag in
# common.sh. Needs the 10-rate synth attrs (eval-data synth-filter at staging >= aa7ef59). C1 at 10 M, C6.
LEG=v-yfcc-synth
. "$(dirname "$(readlink -f "$0")")/common.sh"
old_oracles yfcc10m-synth
step oracle oracle --dataset yfcc10m-synth --suite synth
step campaign campaign --suite synth --dataset yfcc10m-synth --resume --interleave --out "$R"
finish
