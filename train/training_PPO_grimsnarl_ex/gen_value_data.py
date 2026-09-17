"""Generate on-distribution value-function training data.

    PY=/home/kangh/miniconda3/envs/kaggle-pokemon/bin/python
    $PY gen_value_data.py --scale 1.0 --workers 10

The replay-trained value net reaches explained variance 0.294, but on *other
people's decks* — best Jaccard overlap with the challenger list is 0.16. This
generates the matchups we actually play.

Mixture, and why each part is here
----------------------------------
**Our matchups (50%).** ``frozen:crustle@challenger`` puts a competent BC policy
behind the challenger deck (the two lists share 19 of 25 distinct ids, Jaccard
0.63, so it is a policy that broadly understands these cards) against each of
the three frozen agents. This is the distribution the search is actually
evaluated in.

**Frozen mirror matches (25%).** All three cross-pairings. Strong play, real
decks, and it broadens the board states beyond a single archetype.

**Deliberately weak and mixed play (25%).** Random-vs-frozen and random-vs-random.
This is not filler. A value net is queried at *search leaves*, and MCTS spends
its budget on lines nobody would reach in competent play — including lines it
mistakenly thinks are good. Training only on strong play leaves V undefined
exactly where the search goes looking, and a search maximising an undefined
region is the failure mode that makes more simulations actively harmful. The
earlier corpus had this problem structurally: every game was top-15 play.

Labels are ``+1/-1`` per frame from each seat's point of view (both recorded —
the labels are exact negatives but the feature vectors are not, so it is a real
symmetry augmentation). Draws are dropped rather than labelled 0, since they are
rare enough that the extra class is not worth modelling.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from arena import OPPONENTS, run  # noqa: E402

#: ``(a, b, share)`` — share is a multiplier on ``--scale * base_games``.
MIXTURE = [
    ("frozen:crustle@challenger", "frozen:crustle", 1.0),
    ("frozen:crustle@challenger", "frozen:alakazam", 1.0),
    ("frozen:crustle@challenger", "frozen:lucario", 1.0),
    ("frozen:crustle", "frozen:alakazam", 0.5),
    ("frozen:crustle", "frozen:lucario", 0.5),
    ("frozen:alakazam", "frozen:lucario", 0.5),
    ("random@challenger", "frozen:crustle", 0.5),
    ("random@challenger", "frozen:alakazam", 0.5),
    ("random@challenger", "frozen:lucario", 0.5),
    ("random@challenger", "random", 0.5),
]

BASE_GAMES = 400


def build_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(_ROOT / "value_data_selfplay.npz"))
    return parser.parse_args(argv)


def main() -> None:
    args = build_args()
    config = {"max_steps": args.max_steps, "seed": args.seed}
    rows, labels, turns, sources, games, margins = [], [], [], [], [], []
    started = time.perf_counter()
    print(f"[gen] {args.workers} workers, scale {args.scale}")
    for index, (a, b, share) in enumerate(MIXTURE):
        # Not ``games`` — that name is the accumulator list below, and rebinding
        # it here silently turned it into an int on the first iteration.
        n_games = max(1, int(BASE_GAMES * share * args.scale))
        result = run(a, b, n_games, args.workers, config, collect=True,
                     stride=args.stride, seed=args.seed + index)
        if result.get("rows") is None:
            continue
        rows.append(result["rows"])
        labels.append(result["labels"])
        turns.append(result["turns"])
        sources.append(np.full(len(result["labels"]), index, dtype=np.int16))
        # Offset per matchup so ids stay globally unique across the mixture.
        games.append(result["games"] + index * 1_000_000_000)
        margins.append(result["margins"])

    rows = np.concatenate(rows)
    labels = np.concatenate(labels)
    turns = np.concatenate(turns)
    sources = np.concatenate(sources)
    games = np.concatenate(games)
    margins = np.concatenate(margins)
    np.savez_compressed(args.out, rows=rows, labels=labels, turns=turns,
                        sources=sources, games=games, margins=margins,
                        mixture=np.array([f"{a}|{b}" for a, b, _ in MIXTURE]))
    print(f"\n[gen] {len(labels):,} rows -> {args.out}  "
          f"({time.perf_counter() - started:.0f}s)")
    print(f"[gen] label balance {float((labels > 0).mean()):.1%} positive, "
          f"{len(np.unique(games)):,} distinct games, "
          f"mean |margin| {float(np.abs(margins).mean()):.3f}")


if __name__ == "__main__":
    main()
