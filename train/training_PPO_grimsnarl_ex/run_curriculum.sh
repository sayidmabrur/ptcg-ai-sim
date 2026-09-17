#!/usr/bin/env bash
#
# Three-stage opponent curriculum for the challenger, run unattended.
#
# The measured problem this exists to solve: the policy beats random 79% and
# plays itself about evenly, but wins only ~15% against the three frozen BC
# agents. At 15% a 48-episode batch contains ~7 wins, so the policy gradient is
# mostly noise — which is why every earlier run plateaued or decayed rather
# than improving. An opponent it can actually beat fixes the arithmetic.
#
# So the ladder is: learn against yourself, then shift weight onto the real
# opponents, then finish on them alone.
#
#   stage 1  60% self-play   entropy 0.03   build general skill, balanced batches
#   stage 2  30% self-play   entropy 0.02   shift onto the real target
#   stage 3   0% self-play   entropy 0.01   polish on the objective the eval measures
#
# Entropy anneals down the ladder: high early keeps exploration open while the
# policy is weak, low late lets it commit once it has something worth
# committing to. The learning rate stays at 1e-4 throughout — raising it to
# 3e-4 was measured to make things actively worse (16.7% -> 6.6% in 31
# updates), because bigger steps along a noisy gradient are just bigger
# mistakes.
#
# --episodes is a CUMULATIVE target, not per-stage: a resumed run restores its
# episode counter, so each stage's number is "train until the total reaches N".
#
# Usage:
#   nohup ./run_curriculum.sh > curriculum.log 2>&1 &
#   tail -f curriculum.log
#
# Stopping: Ctrl-C (or SIGINT to the python process) ends the current stage
# cleanly, writing latest.pt, and `set -e` then halts the chain rather than
# marching on to the next stage. Resume by rerunning the stage you stopped in.

set -euo pipefail
cd "$(dirname "$0")"

# The kaggle-pokemon env is the requirement, not a preference: the base conda
# interpreter segfaults on `import torch` (see train.py). Activate it before
# launching, or pass PY=/path/to/that/env/bin/python — a bare `python` from an
# unactivated shell dies with SIGSEGV, which is not a risk worth taking on a run
# meant to go unattended for a day.
#   conda activate kaggle-pokemon
PY=${PY:-python}
CK=${CK:-checkpoints}
START=${START:-checkpoints/greedy-crustle-selfplay/latest.pt}

"$PY" -c "import torch, wandb" || {
  echo "FATAL: $PY cannot import torch+wandb; set PY to a working interpreter" >&2
  exit 1
}

COMMON="--workers 6 --episodes-per-update 48 --shaping-coef 1.0 --lr 1e-4 \
        --ppo-epochs 2 --target-kl 0.02 --eval-episodes 60 --save-every 10 \
        --baseline-episodes 0 --snapshot-every 25 --snapshot-pool 5"

stage () {  # name, cumulative-episodes, self-play-ratio, entropy, resume-from
  echo "=============================================================="
  echo "stage $1  ->  $2 episodes total, self-play $3, entropy $4"
  echo "resuming from $5"
  echo "=============================================================="
  # --out is passed explicitly so $CK decides where checkpoints land as well as
  # where the next stage resumes from. Without it train.py defaults to
  # checkpoints/<run-name>, which matches $CK only by coincidence — and the
  # next stage would then resume from a path nothing ever wrote.
  "$PY" train.py $COMMON \
      --episodes "$2" \
      --self-play-ratio "$3" \
      --entropy-coef "$4" \
      --resume "$5" \
      --out "$CK/$1" \
      --run-name "$1"
}

# Stage 1 is the run already in flight (greedy-crustle-selfplay, 60k episodes,
# self-play 0.6, entropy 0.03) — so this script picks up where it stops. To run
# the ladder from scratch instead, uncomment the first line and point START at
# whichever checkpoint you want to begin from.
#
# stage crustle-s1-selfplay 45000 0.6 0.03 "$START"

stage crustle-s2-mixed   85000 0.3 0.02 "$START"
stage crustle-s3-target 100000 0.0 0.01 "$CK/crustle-s2-mixed/latest.pt"

echo "=============================================================="
echo "curriculum complete. final weights: $CK/crustle-s3-target/final.pt"
echo "best by eval:                       $CK/crustle-s3-target/best.pt"
echo "=============================================================="
