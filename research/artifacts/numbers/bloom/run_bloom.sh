#!/bin/bash
# Frozen copy per call, as run_leg.sh.
c=$(mktemp /scratch/numbers/p1/bloom/run_bloom.XXXXXX.sh); cp /scratch/numbers/p1/bloom/run_bloom.impl.sh "$c"; exec bash "$c" "$@"
