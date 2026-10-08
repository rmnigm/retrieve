#!/usr/bin/env bash
# Leg 4, V-YFCC at campaign-v2.1: yfcc10m-synth `synth` then yfcc10m `deep` (IVF-TUNE's YFCC
# values), oracles first.
LEG=v-yfcc
. "$(dirname "$(readlink -f "$0")")/common.sh"
step oracle-synth oracle --dataset yfcc10m-synth --suite synth
step oracle-deep oracle --dataset yfcc10m --suite deep
step campaign-synth campaign --suite synth --dataset yfcc10m-synth --resume --interleave --out "$R"
step campaign-deep campaign --suite deep --dataset yfcc10m --resume --interleave --out "$R"
finish
