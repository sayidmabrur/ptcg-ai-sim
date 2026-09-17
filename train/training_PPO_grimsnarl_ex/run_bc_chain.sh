#!/usr/bin/env bash
#
# After the dim=128 run exits, train the narrow-but-deeper variant on the same corpus.
#
# Run unconditionally rather than only on a regression, because the pair is worth more
# than either number alone: both see the identical 4.86M frames and the identical holdout,
# so the difference between them is attributable to width and nothing else. One run tells
# you whether the retrain beat 71.27%; two tell you whether the *width* is what did it.
# The GPU would otherwise sit idle until morning.
#
# Usage: LARGE_PID=68481 STEPS=40000 ./run_bc_chain.sh

set -uo pipefail
cd "$(dirname "$0")"

LARGE_PID="${LARGE_PID:?set LARGE_PID}"
STEPS="${STEPS:-40000}"

echo "[chain] waiting for the dim=128 run (pid $LARGE_PID) at $(date -Is)"
while kill -0 "$LARGE_PID" 2>/dev/null; do sleep 60; done
echo "[chain] dim=128 run exited at $(date -Is)"
sleep 45  # let wandb flush and the dataloader workers reap

STEPS="$STEPS" ./run_bc_fallback.sh
echo "[chain] done at $(date -Is)"
