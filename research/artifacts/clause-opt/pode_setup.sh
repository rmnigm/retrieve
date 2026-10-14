#!/bin/bash
# Pod e, once (CPU only, no GPU lock): the CLAUSE-OPT run tree /scratch/wt/fr-clause at COMMIT
# with /venvs/fr-clause, and the final-freeze baseline /scratch/wt/clause-base with
# /venvs/clause-base, each extension built from source (FR runbook recipe), the .so checked.
#   bash pode_setup.sh <run-tree commit>
set -eu
COMMIT=$1
cd /workspace/retrieve
git fetch -q origin "+refs/heads/run/fr-clause:refs/remotes/origin/run/fr-clause" --tags
git worktree add --detach /scratch/wt/fr-clause "$COMMIT"
git worktree add --detach /scratch/wt/clause-base final-freeze
for t in fr-clause clause-base; do
  ln -sfn /data /scratch/wt/$t/evaluation/data
  ln -sfn /workspace/data /scratch/wt/$t/data
  (cd /scratch/wt/$t && UV_PROJECT_ENVIRONMENT=/venvs/$t UV_CACHE_DIR=/scratch/uvcache/$t \
    TORCH_CUDA_ARCH_LIST="8.0" CUDA_HOME=/usr/local/cuda MAX_JOBS=16 \
    uv sync --frozen --all-packages --all-groups --extra official > /scratch/wt/$t.sync.log 2>&1)
done
SO=official-fork/silvertorch_fork/_C.cpython-311-x86_64-linux-gnu.so
grep -aq 'Tensor? clause_slots=None' /scratch/wt/fr-clause/$SO || { echo "fr-clause .so is not C14"; exit 1; }
! grep -aq 'Tensor? clause_slots=None' /scratch/wt/clause-base/$SO || { echo "clause-base .so has C14"; exit 1; }
sha256sum /scratch/wt/fr-clause/$SO /scratch/wt/clause-base/$SO
git -C /scratch/wt/fr-clause log --oneline -1
echo "CLAUSE-OPT pod e setup ok"
