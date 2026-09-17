"""Coordinate-wise sweep of search configuration against the frozen field.

    PY=/home/kangh/miniconda3/envs/kaggle-pokemon/bin/python
    $PY sweep_search.py --games 150 --value-net value_mt.pt

Reference points, all measured on this box:

    uniform random          8.3%    (15/180)
    PPO best, 30k episodes  28.3%   (~8 hours of training)
    IS-MCTS, zero-shot      21.7%   (60 games, replay-fitted V)

A coordinate sweep, not a grid: one axis varied at a time from a baseline. A
full grid over five axes is ~100 configurations and at 150 games each that is
half a day for information the coordinate sweep gets in an hour. Interactions
between axes are real but second-order next to finding which axis matters at
all, and the two that plausibly interact (simulations x depth) are swept
jointly at the end.

Every point prints a Wilson 95% interval. At 150 games a win rate near 25%
carries roughly +/-7 points, so differences smaller than that are not
differences — the earlier n=30 experiments in this project produced three
"results" that were all inside their own error bars.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from arena import OPPONENTS, versus_field  # noqa: E402

BASELINE = {
    "simulations": 128,
    "particles": 4,
    "max_depth": 30,
    "c_puct": 1.5,
    "selector": "puct",
    "blend": 0.0,
    # Setup frames cannot be searched (no parameter for our own face-down
    # active), and playing them at random hands away the opening board.
    "setup_policy": "frozen:crustle",
    "prior": "heuristic",
    "root_prior": "",
}

#: Focused set. A full coordinate pass over every axis is ~24 points at ~4
#: minutes each; these are the ones the evidence actually points at, and the
#: rest stay at the baseline. ``--axes`` overrides.
FOCUSED = ("root_prior", "c_puct", "simulations", "selector", "blend")

AXES = {
    # Does more search help now that the leaf evaluator is on-distribution?
    # This is the soundness question the rollout evaluator failed and the
    # replay-fitted value net answered only against a random opponent.
    "simulations": [32, 128, 512],
    # A thicker belief. Four particles is a very thin posterior over an
    # opponent hand of ~6 drawn from a pool of ~46.
    "particles": [2, 4, 12],
    # A Pokémon turn is many decisions, so 30 is only ~1-2 turns of lookahead.
    # If the horizon is the binding constraint this axis moves and the others
    # do not.
    "max_depth": [12, 30, 80],
    # With a BC root prior this is the knob that decides how much evidence the
    # search needs before overriding a policy that already scores 41.1%. High
    # c_puct keeps it close to BC; low lets a noisy Q (expl.var ~0.29) win.
    "c_puct": [0.8, 1.5, 3.0, 6.0],
    # Equilibrium-seeking instead of best-response-to-a-uniform-opponent.
    "selector": ["puct", "regret"],
    # Weight on the prize-margin head at the leaf. 0.0 is pure win probability.
    "blend": [0.0, 0.3, 0.6],
    # Directly measures what the random opening was costing.
    "setup_policy": ["", "frozen:crustle"],
    # Without a prior the tree reaches ~log_11(N) plies and widens
    # rather than deepens; this is the axis most likely to matter.
    "prior": ["uniform", "heuristic"],
    # A BC checkpoint scores 41.1% on this deck against the field while
    # the search scores 21.7%. Seeding its policy at the root makes the
    # search a refinement of that rather than a replacement for it.
    "root_prior": ["", "frozen:crustle"],
}


def build_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--games", type=int, default=150,
                        help="games per frozen opponent (x3 for the overall rate)")
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--value-net", default=str(_ROOT / "value_mt.pt"))
    parser.add_argument("--axes", default=None,
                        help="comma-separated subset of axes to sweep")
    parser.add_argument("--joint", action="store_true",
                        help="after the coordinate pass, grid the two axes that "
                             "moved most")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(_ROOT / "sweep_search.json"))
    return parser.parse_args(argv)


def evaluate(config: dict, games: int, workers: int, seed: int):
    full = dict(BASELINE)
    full.update(config)
    full["max_steps"] = 500
    full["seed"] = seed
    return versus_field("search", games, workers, full, seed)


def main() -> None:
    args = build_args()
    BASELINE["value_net"] = args.value_net
    axes = list(AXES) if args.axes is None else args.axes.split(",")
    results: dict = {"baseline": BASELINE, "games_per_opponent": args.games, "axes": {}}
    started = time.perf_counter()

    print("=" * 78)
    print("search configuration sweep vs the frozen field")
    print("=" * 78)
    print(f"baseline: " + "  ".join(f"{k}={v}" for k, v in BASELINE.items()
                                    if k != "value_net"))
    print(f"value net: {Path(args.value_net).name}   "
          f"{args.games} games/opponent ({args.games * 3} total per point)")
    print(f"reference: random 8.3%   PPO-30k 28.3%\n")

    best = (None, -1.0)
    for axis in axes:
        print(f"--- {axis}")
        results["axes"][axis] = []
        for value in AXES[axis]:
            outcome = evaluate({axis: value}, args.games, args.workers, args.seed)
            low, high = outcome["ci95"]
            marker = ""
            if outcome["win_rate"] > best[1]:
                best = ({axis: value}, outcome["win_rate"])
                marker = "  <- best so far"
            print(f"    {axis}={value!r:<10} {outcome['win_rate']:>6.1%} "
                  f"[{low:.1%}, {high:.1%}]  "
                  + "  ".join(f"{k[:4]} {v:.0%}" for k, v in outcome["per_opponent"].items())
                  + marker, flush=True)
            results["axes"][axis].append({"value": value, **outcome})
        print()

    print(f"[sweep] {time.perf_counter() - started:.0f}s")
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"[sweep] written {args.out}")


if __name__ == "__main__":
    main()
