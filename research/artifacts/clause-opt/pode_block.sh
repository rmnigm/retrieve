#!/bin/bash
# Pod e GPU block (under its /scratch/gpu0.lock): the bit-exact gate and the profile at the
# clause row's large build points (LAION 3M / 10M / 30M d256, PubMed d768; the picks' n_lists /
# n_probe), final-freeze vs CLAUSE-OPT. Points already done (compare PASS in OUT) are skipped.
#   flock /scratch/gpu0.lock bash pode_block.sh
P=/scratch/wt/fr-clause/research/artifacts/clause-opt/point2.sh
export OUT=/scratch/clause-opt/out-pode BASE_TREE=/scratch/wt/clause-base BASE_VENV=/venvs/clause-base
export OPT_TREE=/scratch/wt/fr-clause OPT_VENV=/venvs/fr-clause
mkdir -p $OUT
run() { grep -q "PASS" $OUT/$1.log 2>/dev/null && return; timeout 2700 $P "$@" 2>&1 | tee $OUT/$1.log; }
date -u
run pm16k pubmed 768 16384 '512 2048 4096' 'all5 c0_mesh c3_journal_reverse'
run l10k4 laion10m 256 4096 '24 1024' 'c0_domain tags4'
run l30k4 laion30m 256 4096 '24 1024' 'c0_domain tags4'
run l3k8 laion3m 256 8192 '2048' 'tags4'
run l3k16 laion3m 256 16384 '4096' 'tags4'
run l10k16 laion10m 256 16384 '4096' 'tags4'
run l30k16 laion30m 256 16384 '2048 4096' 'c0_domain tags4'
run pm65k pubmed 768 65536 '1024 8192 16384' 'all5 c0_mesh c3_journal_reverse'
run l10k65 laion10m 256 65536 '16384' 'tags4'
run l30k65 laion30m 256 65536 '8192 16384' 'c0_domain tags4'
date -u
