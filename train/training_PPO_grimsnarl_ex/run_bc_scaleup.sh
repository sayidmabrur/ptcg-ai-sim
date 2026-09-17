#!/usr/bin/env bash
#
# Overnight BC scale-up on the 17-day corpus.
#
# Sequence, and why it is a sequence rather than two parallel runs: the pipeline is
# dataloader-bound (277 frames/s at 10% GPU utilisation), so two runs on 12 cores would
# each get roughly half the throughput and neither would finish. One at a time.
#
#   1. wait for the in-flight run on the old 841k-frame corpus to exit
#   2. score that finished checkpoint on the NEW corpus's holdout, so the retrain has a
#      reference measured on the same frames -- its own 74.3% was measured on a
#      different split and is not comparable
#   3. train the scaled-up network from scratch on 4.86M frames
#
# Capacity: dim 64->128, heads 2->8, chain/history 8->12 layers, option self-attention
# 2->4, and OptionCrossAttention 2->6. 2.68M -> 9.59M parameters. Width is what forces
# training from scratch (no loader here tolerates a shape change), which is affordable
# because there are 5.8x more frames than the old checkpoint ever saw.
#
# The schedule is new: there was no warmup or decay at all despite a source comment
# claiming otherwise. At 12 pre-LN layers and double the width, 2000 steps of warmup
# then cosine decay is the difference between training and diverging.

set -uo pipefail
cd "$(dirname "$0")"

# The interpreter directly, not `conda run`: that wrapper buffers child output until the
# process exits, which would leave an 8-hour log empty until it was too late to react.
PY="/home/steve/miniconda3/envs/kaggle-pokemon/bin/python -u"
CORPUS="../dataset/grimmsnarl_0724_0810.parquet"
OLD_CKPT="challenger/bc_policy_xattn.pt"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------- 1. wait
WAIT_PID="${WAIT_PID:-}"
if [[ -n "$WAIT_PID" ]]; then
  echo "[scaleup] waiting for pid $WAIT_PID (run on the old corpus) to finish..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[scaleup] pid $WAIT_PID exited at $(date -Is)"
  sleep 30  # let its wandb process flush and its workers reap
fi

# ---------------------------------------------------------------- 2. reference
echo "[scaleup] benchmarking $OLD_CKPT on the new corpus holdout at $(date -Is)"
$PY eval_checkpoint.py \
  --checkpoint "$OLD_CKPT" \
  --parquet "$CORPUS" \
  --eval-batches 400 \
  --workers 8 \
  2>&1 | tee "$LOG_DIR/baseline_on_new_split.log"

# ---------------------------------------------------------------- 3. scale up
echo "[scaleup] launching the large run at $(date -Is)"
$PY bc_train.py \
  --parquet "$CORPUS" \
  --out challenger/bc_policy_xattn_large.pt \
  --run-name bc-grimsnarl-xattn-large-d128 \
  --dim 128 --nhead 8 \
  --chain-layers 12 --hist-layers 12 --ctx-layers 4 --cross-layers 6 \
  --dropout 0.1 \
  --lr 3e-4 --warmup-steps 2000 --lr-schedule cosine --min-lr-frac 0.1 \
  --batch-size 64 --workers 10 \
  --epochs 3 --max-steps 70000 --total-steps 70000 \
  --eval-every 2500 --eval-batches 120 \
  --wandb-mode online \
  2>&1 | tee "$LOG_DIR/bc_xattn_large.log"

echo "[scaleup] finished at $(date -Is)"
