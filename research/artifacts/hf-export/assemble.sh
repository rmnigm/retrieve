#!/bin/bash
# Hardlink the files the final run read from the Hub download (download.sh) into the archive layout.
set -e; A=/data/anon-export; H=/data/work/hub; mkdir -p $A
G=$A/eval-goodreads-work-id; mkdir -p $G/checkpoints
for f in clause_is_reverse_narrow.pt eval_split.parquet format_vocab.json lang_vocab.json author_vocab.json item_attrs_narrow.pt item_id_map.json test.parquet; do ln $H/eval-goodreads-work-id/$f $G/$f; done
cp -al $H/eval-goodreads-work-id/checkpoints/sasrec-ssm-logq-d128 $G/checkpoints/
X=$A/eval-arxiv-papers; mkdir -p $X
for f in author_vocab.json cat_main_vocab.json clause_is_reverse_narrow.pt eval_split.parquet heldout.parquet item_attrs_narrow.pt item_id_map.json license_vocab.json; do ln $H/eval-arxiv-papers/$f $X/$f; done
cp -al $H/eval-arxiv-papers/content $H/eval-arxiv-papers/content_d128 $X/
P=$A/eval-pubmed; mkdir -p $P
(cd $H/eval-pubmed && find . -type f -not -path './.cache/*' -not -name .gitattributes -not -name README.md) | while read f; do mkdir -p $P/$(dirname $f); ln $H/eval-pubmed/$f $P/$f; done
Y=$A/eval-yambda-500m; mkdir -p $Y/checkpoints
ln $H/eval-yambda-500m/item_id_map.json $Y/item_id_map.json
ln $H/eval-yambda-500m/trainer/test.parquet $Y/test.parquet   # the harness's test.parquet is the trainer split
for d in 64 128 256; do cp -al $H/eval-yambda-500m/checkpoints/sasrec-ssm-logq-d$d $Y/checkpoints/; done
