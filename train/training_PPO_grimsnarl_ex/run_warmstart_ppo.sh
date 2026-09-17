#!/usr/bin/env bash
#
# PPO from a behavioural-cloning warm start.
#
#   nohup ./run_warmstart_ppo.sh > warmstart.log 2>&1 &
#
# Why this run exists
# -------------------
# Measured over 1,796 games, crustle's BC checkpoint piloting the challenger
# deck scores 41.1% [38.8, 43.4] against the frozen field. The best PPO run in
# checkpoints/ reached 28.3% after 30k episodes and roughly eight hours — from
# random init, because --init-from has never actually worked: the challenger's
# CardEmbed gained ability fields, widening proj from 151 to 193 inputs, so
# bc_policy.pt fails to load with 19 shape mismatches.
#
# --actor-arch crustle builds the actor from crustle's own PolicyNetwork
# generation instead, which the checkpoint loads into cleanly. Training now
# starts at ~41% rather than 0%.
#
# Hyperparameters, and why they differ from the from-scratch runs
# ---------------------------------------------------------------
# The failure mode here is the opposite of the old one. From random init the
# risk was never learning; from a competent warm start the risk is destroying
# it in the first few updates. Everything below is set to move slowly:
#
#   --lr 5e-5          half the from-scratch 1e-4. 3e-4 was measured actively
#                      harmful even from random init (16.7% -> 6.6% in 31
#                      updates); a good policy is more fragile, not less.
#   --entropy-coef     0.005, well under the 0.01-0.03 used from scratch.
#                      An entropy bonus pulls toward uniform, which is exactly
#                      what a warm start must not do.
#   --target-kl 0.015  tighter than the 0.02 default. Bounds per-update drift.
#   --ppo-epochs 2     fewer passes over the same batch, same reason.
#   --shaping-coef 1.0 potential-based prize shaping. It cannot change which
#                      policy is optimal (Ng, Harada & Russell) and it is what
#                      turns one bit per ~100 decisions into a signal GAE can
#                      actually assign. The best previous run used it.
#
# The critic starts zeroed (see ActorCritic), so advantage = R at init rather
# than R minus whatever a random head emitted.

set -uo pipefail
cd "$(dirname "$0")"

# The kaggle-pokemon env is the requirement (see train.py). Activate it first:
#   conda activate kaggle-pokemon
PY=${PY:-python}
WORKERS=${WORKERS:-6}
EPISODES=${EPISODES:-60000}
OUT=${OUT:-checkpoints/warmstart-crustle}

exec "$PY" -u train.py \
    --actor-arch crustle \
    --init-from crustle_frozen/bc_policy.pt \
    --episodes "$EPISODES" \
    --episodes-per-update 48 \
    --workers "$WORKERS" \
    --lr 5e-5 \
    --entropy-coef 0.005 \
    --ppo-epochs 2 \
    --target-kl 0.015 \
    --shaping-coef 1.0 \
    --eval-every 30 \
    --eval-episodes 20 \
    --final-eval-episodes 100 \
    --save-every 10 \
    --baseline-episodes 0 \
    --wandb-mode disabled \
    --out "$OUT" \
    --run-name warmstart-crustle
