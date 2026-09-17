#!/usr/bin/env bash
#
# Unattended overnight chain. Waits for the running ablation, then splits the
# box between the two lines of work that are still open.
#
#   nohup ./run_overnight.sh > overnight.log 2>&1 &
#
# 12 cores, allocated 6 + 5 and deliberately not more. Two heavy runs
# oversubscribed this box 3x earlier tonight (a killed Pool orphans its
# children) and everything silently ran at a third speed.
#
#   PPO warm start   6 workers, all night. Starts at ~41% instead of 0%, which
#                    is the single largest lever found today.
#   search sweep     5 workers, ~90 min. Decides whether IS-MCTS can be pushed
#                    past the 41.1% BC bar at all, and whether the regret
#                    (CFR-flavoured) selector beats PUCT.

set -uo pipefail
cd "$(dirname "$0")"
# The kaggle-pokemon env is the requirement (see train.py). Activate it first:
#   conda activate kaggle-pokemon
PY=${PY:-python}

echo "[overnight] waiting for the ablation to finish"
while pgrep -f "ablate_search.p[y]" > /dev/null; do sleep 20; done
echo "[overnight] ablation done:"
tail -8 ablate.log

echo
echo "[overnight] launching PPO warm start (6 workers, background)"
WORKERS=6 nohup ./run_warmstart_ppo.sh > warmstart.log 2>&1 &
sleep 60   # let its workers come up before competing for cores

echo "[overnight] launching focused search sweep (5 workers)"
"$PY" -u sweep_search.py \
    --games 100 --workers 5 --value-net value_mt.pt \
    --axes root_prior,c_puct,simulations,selector,blend \
    --out sweep_search.json > sweep.log 2>&1
echo "[overnight] sweep done:"
tail -40 sweep.log

echo
echo "[overnight] PPO progress so far:"
grep -E "^\[update|^\[eval" warmstart.log | tail -20
echo "[overnight] chain complete; PPO continues in the background"
