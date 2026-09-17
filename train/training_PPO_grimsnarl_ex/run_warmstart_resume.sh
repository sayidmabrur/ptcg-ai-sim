#!/usr/bin/env bash
# Resume the warm-started run on the whole box.
#
# Restarted at update 40 to move from 6 rollout workers to 11 once the search
# experiments were done competing for cores: 1,729 ep/h -> expected ~2,800.
# The search line was stopped because it is measurably a dead end here (25-30%
# against a 41.1% BC baseline, and CFR-style selection did not help), so the
# remaining compute is worth more spent on the line that is working.
set -uo pipefail
cd "$(dirname "$0")"
# The kaggle-pokemon env is the requirement (see train.py). Activate it first:
#   conda activate kaggle-pokemon
PY=${PY:-python}
exec "$PY" -u train.py \
    --actor-arch crustle \
    --resume checkpoints/warmstart-crustle/latest.pt \
    --episodes 60000 --episodes-per-update 48 --workers 11 \
    --lr 5e-5 --entropy-coef 0.005 --ppo-epochs 2 --target-kl 0.015 \
    --shaping-coef 1.0 \
    --eval-every 25 --eval-episodes 30 --final-eval-episodes 150 \
    --save-every 10 --baseline-episodes 0 --wandb-mode disabled \
    --out checkpoints/warmstart-crustle --run-name warmstart-crustle
