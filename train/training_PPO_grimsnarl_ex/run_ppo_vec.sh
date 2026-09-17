#!/usr/bin/env bash
#
# The exploit run on pokeka2026's environment: N battles in ONE process, every
# decision batched into one GPU forward.
#
#   conda activate kaggle-pokemon
#   nohup ./run_ppo_vec.sh > vec.log 2>&1 &
#
# Same experiment as run_ppo_exploit.sh — same warm start, same reward shaping,
# same hyperparameters, same checkpoint format — with the environment swapped
# from `cg.game` (one battle per process, hence batch-1 inference and
# `--workers N` to scale) to `ptcg_rl.core.engine.PTCGBattle`, which owns its
# battle pointer and so lets 32 battles run side by side here. See pokeka_env.py
# and vec_rollout.py.
#
# READ THIS BEFORE CHOOSING IT
# ----------------------------
# Measured on this box, 256 episodes at 64 per update:
#
#     --workers 11    ~3120 ep/h
#     --num-envs 32   ~1440 ep/h
#
# This script is 2.2x SLOWER than run_ppo_exploit.sh, and that is not a tuning
# problem. Batching does what it promises — inference drops from 7.6 ms a
# decision to 0.26 ms — but inference was never the cost: `dataset.transform` is
# 6.4 ms a decision and builds ~990 tensors doing it, because a decision's
# `decision_chain` re-featurises up to 60 earlier decisions. That is
# single-threaded Python and 87% of a rollout decision, so throughput tracks
# cores, and eleven one-battle processes beat thirty-two batched battles. See
# FINDINGS.md, "The environment is no longer a singleton".
#
# Run this one when you want:
#   * one process instead of twelve (no worker pool to orphan, no IPC, no
#     parent-side feature rebuild — the transform() output stays in the process
#     that updates on it);
#   * rollout and update numerically identical, because there is no CPU actor
#     replica and so a PPO ratio is no longer computed across two devices;
#   * the GPU to be the machine doing the work, e.g. on a box with few cores.
#
# Otherwise run run_ppo_exploit.sh.

set -uo pipefail
cd "$(dirname "$0")"

# The kaggle-pokemon env is the requirement (see train.py). Activate it first:
#   conda activate kaggle-pokemon
PY=${PY:-python}
NUM_ENVS=${NUM_ENVS:-32}
EPISODES=${EPISODES:-120000}
RESUME=${RESUME:-checkpoints/ppo-vec/latest.pt}
OUT=${OUT:-checkpoints/ppo-vec}

# --resume only if there is something to resume; the first launch warm-starts
# from the BC checkpoint instead (41.1% against the frozen field, vs 0% from
# random init — the single largest lever in this project).
START=(--init-from crustle_frozen/bc_policy.pt)
if [ -f "$RESUME" ]; then
    START=(--resume "$RESUME")
fi

exec "$PY" -u train.py \
    --actor-arch crustle \
    "${START[@]}" \
    --num-envs "$NUM_ENVS" \
    --episodes "$EPISODES" \
    --episodes-per-update 64 \
    --lr 1.5e-4 \
    --entropy-coef 0.001 \
    --ppo-epochs 2 \
    --minibatch-size 128 \
    --target-kl 0.03 \
    --shaping-coef 1.5 \
    --attack-shaping 0.5 \
    --eval-every 25 \
    --eval-episodes 40 \
    --final-eval-episodes 200 \
    --save-every 10 \
    --baseline-episodes 0 \
    --wandb-mode online \
    --wandb-project pokemon-tcg-rl \
    --out "$OUT" \
    --run-name ppo-vec-crustle
