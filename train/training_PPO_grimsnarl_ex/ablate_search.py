"""Cumulative ablation of the search-side fixes, value net held constant.

    PY=/home/kangh/miniconda3/envs/kaggle-pokemon/bin/python
    $PY ablate_search.py --games 100 --value-net value_scalar.pt

Three changes landed at once — a BC policy for the setup phase, a heuristic
prior, and a BC policy prior at the root — and a single before/after number
cannot say which of them did anything. Each row here adds one, value function
held fixed, so the search-side contribution is separated from the value-side
one that ``train_value.py`` is producing in parallel.

Reference points on this box, all vs the same three frozen agents:

    uniform random                        8.3%   (n=180)
    IS-MCTS, replay-fit V, no fixes      21.7%   (n=60)
    PPO best, 30k episodes / ~8 hours    28.3%   (n=60)
    BC checkpoint piloting our deck      41.1%   (n=1796)

That last one is the real bar and it was not visible until the self-play
generation measured it. A search that lands below it is worse than deleting the
search and shipping the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from arena import run, versus_field, wilson  # noqa: E402

STEPS = [
    ("as measured (random setup, uniform prior)",
     {"setup_policy": "", "prior": "uniform", "root_prior": ""}),
    ("+ BC setup phase",
     {"setup_policy": "frozen:crustle", "prior": "uniform", "root_prior": ""}),
    ("+ heuristic prior",
     {"setup_policy": "frozen:crustle", "prior": "heuristic", "root_prior": ""}),
    ("+ BC root prior",
     {"setup_policy": "frozen:crustle", "prior": "heuristic",
      "root_prior": "frozen:crustle"}),
]

BASE = {
    "simulations": 128, "particles": 4, "max_depth": 30, "c_puct": 1.5,
    "selector": "puct", "blend": 0.0, "max_steps": 500, "seed": 0,
}


def build_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--games", type=int, default=100, help="per frozen opponent")
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--value-net", default=str(_ROOT / "value_scalar.pt"))
    parser.add_argument("--reference", action="store_true",
                        help="also re-measure the BC-on-our-deck bar")
    parser.add_argument("--out", default=str(_ROOT / "ablate_search.json"))
    return parser.parse_args(argv)


def main() -> None:
    args = build_args()
    print("=" * 78)
    print("search ablation vs the frozen field "
          f"({args.games} games/opponent, {args.games * 3} total per row)")
    print("=" * 78)
    print(f"value net: {Path(args.value_net).name}   "
          f"simulations: {args.simulations}")
    print("bar to beat: BC checkpoint on our deck = 41.1%\n")

    results = []
    for label, overrides in STEPS:
        config = dict(BASE)
        config.update(overrides)
        config["value_net"] = args.value_net
        config["simulations"] = args.simulations
        print(f"--- {label}")
        outcome = versus_field("search", args.games, args.workers, config, 0)
        results.append({"label": label, **overrides, **outcome})
        print()

    if args.reference:
        print("--- reference: BC checkpoint piloting our deck (no search)")
        wins = played = 0
        for name in ("crustle", "alakazam", "lucario"):
            r = run("frozen:crustle@challenger", f"frozen:{name}", args.games,
                    args.workers, {"max_steps": 500, "seed": 0})
            wins += r["wins"]
            played += r["played"]
        low, high = wilson(wins, played)
        print(f"  OVERALL {wins / played:.1%} [{low:.1%}, {high:.1%}] ({wins}/{played})")
        results.append({"label": "BC reference", "wins": wins, "played": played,
                        "win_rate": wins / played, "ci95": (low, high)})

    print("\n" + "=" * 78)
    print(f"{'configuration':<46}{'win rate':>10}{'95% CI':>18}")
    print("-" * 78)
    for row in results:
        low, high = row["ci95"]
        print(f"{row['label']:<46}{row['win_rate']:>9.1%}"
              f"{f'[{low:.1%}, {high:.1%}]':>18}")
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
