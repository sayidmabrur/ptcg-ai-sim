#!/usr/bin/env bash
#
# Overnight chain for Marnie's Grimmsnarl ex: clone a pilot, then improve it with PPO.
#
#   conda activate kaggle-pokemon
#   nohup ./run_overnight_grimsnarl.sh > overnight_grimsnarl.log 2>&1 &
#
# Why BC first — this is the whole reason the chain has two stages
# ----------------------------------------------------------------
# PPO needs something to improve on, and for this archetype there was nothing. Measured on
# this box against the frozen field (n=72 each):
#
#     random policy                                    5.5%
#     crustle BC checkpoint piloting the GRIMMSNARL deck   12.5%   <- the only "warm start"
#     crustle BC checkpoint piloting its OWN deck          52.8%
#
# A BC policy handed a decklist it never saw the expert play is worth almost nothing, so
# starting PPO here starts it at ~12%. FINDINGS.md records what that costs: from a cold
# start, 30k episodes and ~8 hours reached 28.3%, below the 41.1% BC bar it was chasing.
#
# The corpus removes the excuse. 841,016 frames in
# ``../dataset/2026-08-06-10-marnie-grimmsnarl-ex-cleaned.parquet`` are decisions by real
# ladder pilots of exactly this 60-card list, with ``target_action`` recorded; that pilot won
# 46.0% of 9,784 decided games against the live metagame. Stage 1 clones it. Stage 2 runs PPO
# from that checkpoint, which is the only configuration in which "beat the BC model" is a
# question about PPO rather than about whether PPO can rediscover an archetype unaided.
#
# Stage 2's settings, and what changed from the crustle runs
# ---------------------------------------------------------
#   --gamma 0.997 --gae-lambda 0.95   The crustle runs used 1.0/1.0, which collapses GAE to
#                                     advantage = R - V(s) — pure Monte-Carlo, no
#                                     bootstrapping at all, one bit of signal spread over
#                                     ~100 decisions with every decision in a won game
#                                     credited identically. These are what actually turn the
#                                     critic into a credit-assignment mechanism.
#   --reward winprob                  WIN=1 / LOSS=0, so V(s) regresses onto P(win | s) and
#                                     is readable as a position evaluation (0.85 strong,
#                                     0.17 losing). The value head's bias starts at the 0.5
#                                     prior instead of 0, which a 0/1 label would otherwise
#                                     make confidently wrong on update 1.
#   --shaping-coef 0.5                The crustle run used 1.5. Potential-based shaping
#                                     telescopes to coef * prize_margin, so 1.5 reaches ±1.5
#                                     against a ±1 terminal — shaping *dominated* the win
#                                     signal it is meant to be subordinate to. 0.5 keeps the
#                                     terminal dominant while still grading prize trades.
#   --attack-shaping 0.3              Same potential, kept small for the same reason.
#   30-frame chain/history            observation.LEARNER_SPEC. The learner's temporal window
#                                     halves (the chain is the quadratic term in the 6.4 ms
#                                     transform); the frozen agents keep 60, since they are
#                                     the yardstick and truncating their input would move the
#                                     benchmark rather than the policy.
#
# What to check when you wake up
# ------------------------------
#   grep -E '^\[bc\].*val_exact' overnight_grimsnarl.log | tail    # stage 1 climbed?
#   grep -E '^\[eval'            overnight_grimsnarl.log | tail    # stage 2 vs frozen field
#   grep -E '^\[update'          overnight_grimsnarl.log | tail    # ev / entropy / kl health
#
# The number that matters is ``[eval] ... mean``. Read it against 41.1% (the crustle BC bar
# on its own deck) and against stage 1's own first eval, which is printed before training
# starts. Treat single evals with suspicion: n=120 swings ±10 points here, and a checkpoint
# recorded at 51.7% re-measured at 27.8% on a fresh sample.

set -uo pipefail
cd "$(dirname "$0")"

# The kaggle-pokemon env is the requirement (see train.py). Activate it first:
#   conda activate kaggle-pokemon
PY=${PY:-python}
PARQUET=${PARQUET:-../dataset/2026-08-06-10-marnie-grimmsnarl-ex-cleaned.parquet}
BC_EPOCHS=${BC_EPOCHS:-4}
BC_WORKERS=${BC_WORKERS:-10}
BC_OUT=${BC_OUT:-challenger/bc_policy.pt}
EPISODES=${EPISODES:-120000}
WORKERS=${WORKERS:-11}
OUT=${OUT:-checkpoints/ppo-grimsnarl}

echo "=============================================================="
echo "[stage 1] behavioural cloning the Grimmsnarl pilot"
echo "=============================================================="
"$PY" -u bc_train.py \
    --parquet "$PARQUET" \
    --epochs "$BC_EPOCHS" \
    --batch-size 64 \
    --workers "$BC_WORKERS" \
    --lr 3e-4 \
    --dropout 0.1 \
    --eval-every 2000 \
    --eval-batches 60 \
    --out "$BC_OUT"

if [ ! -f "$BC_OUT" ]; then
    echo "[stage 1] FAILED: no $BC_OUT written; not starting PPO from nothing" >&2
    exit 1
fi
echo "[stage 1] done: $BC_OUT"
cat "$BC_OUT.meta.json" 2>/dev/null | head -8

echo
echo "=============================================================="
echo "[stage 2] PPO from the cloned pilot"
echo "=============================================================="
exec "$PY" -u train.py \
    --actor-arch challenger \
    --init-from "$BC_OUT" \
    --episodes "$EPISODES" \
    --episodes-per-update 64 \
    --workers "$WORKERS" \
    --lr 1.5e-4 \
    --entropy-coef 0.001 \
    --ppo-epochs 2 \
    --minibatch-size 128 \
    --target-kl 0.03 \
    --gamma 0.997 \
    --gae-lambda 0.95 \
    --reward winprob \
    --shaping-coef 0.5 \
    --attack-shaping 0.3 \
    --eval-every 25 \
    --eval-episodes 40 \
    --final-eval-episodes 200 \
    --save-every 10 \
    --baseline-episodes 0 \
    --wandb-mode online \
    --wandb-project pokemon-tcg-rl \
    --out "$OUT" \
    --run-name ppo-grimsnarl-ex
