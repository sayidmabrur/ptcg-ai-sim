#!/usr/bin/env bash
#
# Unattended experiment chain: value function -> search config -> final numbers.
#
#   nohup ./run_experiments.sh > experiments.log 2>&1 &
#   tail -f experiments.log
#
# Assumes value_data_selfplay.npz already exists (gen_value_data.py).
#
# Reference points, all measured on this box:
#   uniform random vs frozen field    8.3%   (15/180)
#   PPO best after 30k episodes       28.3%  (~8 hours of training)
#   IS-MCTS zero-shot, replay-fit V   21.7%  (60 games)
#
# Deliberately sequential. Every stage saturates all cores through arena.py's
# worker pool, and the last time two heavy runs overlapped on this box they
# oversubscribed it 3x and quietly ran everything at a third speed.

set -uo pipefail
cd "$(dirname "$0")"

# The kaggle-pokemon env is the requirement (see train.py). Activate it first:
#   conda activate kaggle-pokemon
PY=${PY:-python}
WORKERS=${WORKERS:-11}
GAMES=${GAMES:-120}       # per frozen opponent; x3 for an overall rate
FINAL_GAMES=${FINAL_GAMES:-350}

banner () { echo; echo "=============================================================="; echo "$1"; echo "=============================================================="; }

banner "1/4  value function: hyperparameter sweep on on-distribution data"
"$PY" -u train_value.py --sweep --sweep-epochs 20 --epochs 80 --out value_mt.pt || exit 1

banner "2/4  search configuration sweep vs the frozen field"
"$PY" -u sweep_search.py --games "$GAMES" --workers "$WORKERS" \
      --value-net value_mt.pt --out sweep_search.json || exit 1

banner "3/4  head-to-head: replay-fit V vs on-distribution V, matched config"
for net in value_scalar.pt value_mt.pt; do
  echo "--- $net"
  "$PY" -u arena.py --a search --b field --games "$GAMES" --workers "$WORKERS" \
        --simulations 128 --particles 4 --value-net "$net" || true
done

banner "4/4  best-known configuration at high n"
# Filled in from the sweep by hand if it disagrees; these are the defaults the
# sweep starts from, run at a sample size that can actually separate 21% from 28%.
"$PY" -u arena.py --a search --b field --games "$FINAL_GAMES" --workers "$WORKERS" \
      --simulations 256 --particles 8 --value-net value_mt.pt || true

banner "done"
