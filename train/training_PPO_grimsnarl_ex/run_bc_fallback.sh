#!/usr/bin/env bash
#
# Fallback: keep the narrow network, give it the 5.8x larger corpus.
#
# Run only if the dim=128 run does not clear the reference (71.27% exact / 79.69%
# same-play, measured on the same 25,600 holdout frames). The logic: width is the one
# capacity change that forces training from scratch, because nothing in this repo loads a
# state_dict across a shape change. Depth and heads do not -- attention projections are
# (3*dim, dim) whatever the head count, and extra layers only add keys, which
# --init-from tolerates. So this variant keeps dim=64 and buys capacity where it is free:
#
#   heads   2 -> 8        (zero new parameters)
#   chain   8 -> 12 layers
#   history 8 -> 12 layers
#   option self-attn 2 -> 4
#   cross-attention  2 -> 6
#
# 2.68M -> 3.22M parameters, and it warm-starts cleanly from the finished checkpoint
# (0 unexpected keys, 192 missing = the new layers, trained fresh). That makes it a
# strictly better-informed start than the dim=128 run got, and it isolates "more data"
# from "more width" -- the same network that scored 71.27% now sees 5.8x the frames.
#
# Logs offline by default and is synced afterwards. The first attempt at this run died at
# `wandb.init` on a transient TLS verification failure against api.wandb.ai, having
# trained nothing -- an online handshake is a single point of failure in front of a
# three-hour job, and there is no reason to accept one. `wandb sync wandb/offline-run-*`
# puts it in the dashboard afterwards either way.
#
# Usage: STEPS=40000 ./run_bc_fallback.sh

set -uo pipefail
cd "$(dirname "$0")"

PY="/home/steve/miniconda3/envs/kaggle-pokemon/bin/python -u"
CORPUS="../dataset/grimmsnarl_0724_0810.parquet"
STEPS="${STEPS:-40000}"
INIT="${INIT:-challenger/bc_policy_xattn.pt}"
mkdir -p logs

echo "[fallback] $STEPS steps, warm-starting from $INIT, at $(date -Is)"
$PY bc_train.py \
  --parquet "$CORPUS" \
  --out challenger/bc_policy_xattn_deep64.pt \
  --init-from "$INIT" \
  --run-name bc-grimsnarl-xattn-deep64 \
  --dim 64 --nhead 8 \
  --chain-layers 12 --hist-layers 12 --ctx-layers 4 --cross-layers 6 \
  --dropout 0.1 \
  --lr 2e-4 --warmup-steps 1000 --lr-schedule cosine --min-lr-frac 0.1 \
  --batch-size 64 --workers 10 \
  --epochs 3 --max-steps "$STEPS" --total-steps "$STEPS" \
  --eval-every 2500 --eval-batches 120 \
  --wandb-mode "${WANDB_MODE:-offline}" \
  2>&1 | tee "logs/bc_xattn_deep64.log"

echo "[fallback] finished at $(date -Is)"
