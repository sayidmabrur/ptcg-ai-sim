#!/usr/bin/env bash
#
# Upload every offline wandb run to the dashboard. Safe to run at any time, including
# while a run is still training — syncing a live offline directory uploads what has been
# written so far, and running it again later uploads the rest.
#
# Training deliberately never touches the network (see run_bc_5epoch.sh): the corpus is a
# local parquet and the weights are local, so wandb was the only component that could fail
# on a dropped link — and a TLS error at `wandb.init` already destroyed one run before it
# trained a single step. Offline removes that failure mode; this script is how the metrics
# get to the dashboard afterwards.
#
#   ./sync_wandb.sh            # sync everything found
#   ./sync_wandb.sh --list     # just show what is pending

set -uo pipefail
cd "$(dirname "$0")"

WANDB=/home/steve/miniconda3/envs/kaggle-pokemon/bin/wandb
DIRS=$(ls -1d training_PPO_*/wandb/offline-run-* 2>/dev/null)

if [ -z "$DIRS" ]; then
  echo "no offline runs found"
  exit 0
fi

if [ "${1:-}" = "--list" ]; then
  echo "pending offline runs:"
  for d in $DIRS; do
    printf '  %-70s %s\n' "$d" "$(du -sh "$d" | cut -f1)"
  done
  exit 0
fi

for d in $DIRS; do
  echo "=== syncing $d ==="
  # Never abort the loop on one bad run: a partially written directory from an
  # interrupted job should not stop the others from uploading.
  $WANDB sync "$d" || echo "  (failed — safe to retry later)"
done
echo "done at $(date -Is)"
