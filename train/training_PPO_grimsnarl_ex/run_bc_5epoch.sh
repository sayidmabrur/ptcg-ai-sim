#!/usr/bin/env bash
#
# Grimmsnarl ex, five epochs, with the targeting fix enabled.
#
# The previous run covered 70,000 steps = 4.48M frames = 0.97 epoch, which is why its
# val_loss was still falling when the budget ran out. Five epochs of the 4,618,952-frame
# training split at batch 64 is 360,855 steps -- roughly 25 hours at the 257 frames/s this
# pipeline sustains. The bottleneck is Python feature-building in the dataloader, not the
# GPU (which sits near 20%), so the only way to shorten it is fewer frames, not a smaller
# model.
#
# --board-tokens is on. Measured on this corpus, the previous checkpoint attaches a second
# energy to an already-energised Munkidori on 11.2% of the decisions where an empty copy
# was available, against 1.5% for the human pilots -- not a habit it learned, but a
# distinction absent from its input, because the bench is mean-pooled before the option
# scorer sees it. The flag gives each option a cross-attention path to the specific
# Pokemon it names. It costs 2,432 parameters.
#
# wandb runs OFFLINE, deliberately. Nothing about training needs the network -- the corpus
# is a local parquet, the weights are local -- and wandb is the only component that would
# reach out. A transient TLS verification error against api.wandb.ai has already destroyed
# one run of this at step 0, before it trained anything, and a 25-hour job must not hang on
# a link staying up. The identical metric stream lands in wandb/offline-run-*;
# ./sync_wandb.sh uploads it whenever the connection is back. The only thing given up is
# the live view.

set -uo pipefail
cd "$(dirname "$0")"
export WANDB_MODE=offline

PY="/home/steve/miniconda3/envs/kaggle-pokemon/bin/python -u"
CORPUS="../dataset/grimmsnarl_0724_0810.parquet"
STEPS="${STEPS:-360855}"
LOG="logs/bc_grimmsnarl_5epoch.log"
mkdir -p logs

run() {
  $PY bc_train.py \
    --parquet "$CORPUS" \
    --out challenger/bc_policy_xattn_board.pt \
    --run-name bc-grimsnarl-xattn-board-5ep \
    --dim 128 --nhead 8 \
    --chain-layers 12 --hist-layers 12 --ctx-layers 4 --cross-layers 6 \
    --board-tokens \
    --dropout 0.1 \
    --lr 3e-4 --warmup-steps 2000 --lr-schedule cosine --min-lr-frac 0.05 \
    --batch-size 64 --workers 10 \
    --epochs 5 --max-steps "$STEPS" --total-steps "$STEPS" \
    --eval-every 5000 --eval-batches 120 \
    --wandb-mode "$1"
}

echo "[5epoch] starting at $(date -Is), $STEPS steps (wandb offline)"
run offline 2>&1 | tee "$LOG"
echo "[5epoch] finished at $(date -Is)"
