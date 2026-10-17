#!/bin/bash
set -u
H=/venvs/retrieve/bin/hf
cd /data/work/hub
$H download pinkmeme/eval-results --repo-type dataset --include 'fr/**' --local-dir eval-results > ../logs/dl-results.log 2>&1 &
$H download pinkmeme/eval-goodreads-work-id --repo-type dataset --exclude 'checkpoints/gsasrec-*' --local-dir eval-goodreads-work-id > ../logs/dl-goodreads.log 2>&1 &
$H download pinkmeme/eval-arxiv-papers --repo-type dataset --local-dir eval-arxiv-papers > ../logs/dl-arxiv.log 2>&1 &
$H download pinkmeme/eval-yambda-500m --repo-type dataset --exclude 'checkpoints/gsasrec-*' --local-dir eval-yambda-500m > ../logs/dl-yambda.log 2>&1 &
$H download pinkmeme/eval-pubmed --repo-type dataset --revision 10d48d05 --local-dir eval-pubmed > ../logs/dl-pubmed.log 2>&1 &
wait
echo ALLDONE
