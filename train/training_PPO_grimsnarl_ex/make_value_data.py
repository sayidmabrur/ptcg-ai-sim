"""Turn the Marnie's Grimmsnarl ex replay corpus into value-network training data.

    PY=python  # conda activate kaggle-pokemon
    $PY make_value_data.py --parquet ../dataset/2026-08-06-10-marnie-grimmsnarl-ex-cleaned.parquet

Writes the exact ``.npz`` layout ``train_value.py --data`` already reads
(``rows``/``labels``/``turns``/``sources``/``games``/``margins``), so the good trainer —
prize-margin auxiliary head, game-level split, hyperparameter sweep — is reused unchanged
rather than reimplemented for a second corpus.

Why this corpus and not ``gen_value_data.py``'s self-play
--------------------------------------------------------
``gen_value_data.py`` generates on-distribution self-play, which is the right thing when the
policy already plays the deck. Here the deck is *new*: nothing in this repository has ever
piloted Grimmsnarl, so self-play would be sampling from a policy that does not know the
archetype, and the value head would be fitted to positions the trained policy will never
reach. The replay corpus is the opposite trade — real games by a real pilot of exactly this
60-card list, which is the only on-archetype data that exists before the first PPO run.
``train_value.py``'s own docstring records the risk this carries (a replay-fitted net
reached explained variance 0.294 but on other people's decks); that objection does not
apply here, because these replays *are* this deck.

Conventions, copied from ``arena.py`` rather than reinvented — a sign error here trains a
value head that is confidently backwards:

* ``labels`` — ``+1`` if this seat won, ``-1`` otherwise. Draws are dropped, not labelled 0.
  **The outcome comes from ``matches.parquet``, not from the corpus.** The cleaned corpus has
  a ``final_result`` column and it is ``-1`` on all 1,355,855 rows — never populated. Reading
  it as "the winning player index" labels every frame in the dataset a loss, which trains a
  value head that predicts certain defeat from any position and cannot be detected by any
  loss curve, because it fits that constant perfectly. ``matches.parquet`` carries the real
  ±1 per seat (``reward0``/``reward1``), and ``player_index`` maps to ``player0``/``player1``
  exactly — verified on all 8,436 episodes by name, with the crossed pairing matching 0.0%.
  The per-frame ``reward`` column is also not the label: it is 0 everywhere except the
  terminal frame of each game.
* ``margins`` — ``(their prizes remaining - ours) / 6``, POV-relative. Prizes *remaining*,
  so taking prizes makes this positive.
* ``rows`` — ``value_features.scalar_features(state, seat)``, POV-relative.

One approximation, stated because it is not exact: the engine's terminal state is not a
decision frame, so it is not in the corpus. The prize margin is therefore read from each
episode's **last recorded frame** rather than the true terminal. It is off by at most the
prizes taken by the final action, and it feeds the *auxiliary* head — the value head itself
still regresses the exact win/loss label.

Only frames the target archetype was deciding at are kept (``is_target_archetype``), so
every row is a Grimmsnarl-seat decision. The opponent-POV augmentation ``arena.py`` does is
deliberately not reproduced: the opposing decks are a whole metagame, and their positions
are not the distribution this policy will be evaluated in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from value_features import SCALAR_NAMES, scalar_features  # noqa: E402

#: Columns the conversion touches. Naming them keeps the ~90 nested leaves of
#: ``selection``/``options`` off the heap entirely — reading the whole row group expands to
#: several GB of Python objects, which is what makes a naive ``read_table`` unusable here.
COLUMNS = ("episode_id", "frame_index", "player_index", "state", "is_target_archetype")


def load_outcomes(path: Path) -> dict[int, tuple[float, float]]:
    """``episode_id -> (reward for seat 0, reward for seat 1)`` from ``matches.parquet``.

    The corpus's own ``final_result`` cannot be used — see the module docstring.
    """
    table = pq.read_table(path, columns=["episode_id", "reward0", "reward1"]).to_pylist()
    return {
        int(r["episode_id"]): (float(r["reward0"]), float(r["reward1"]))
        for r in table
        if r["reward0"] is not None and r["reward1"] is not None
    }


def prize_counts(state: dict, seat: int) -> tuple[int, int]:
    players = state.get("players") or []
    if len(players) < 2:
        return 0, 0
    mine = players[seat].get("prize") or []
    theirs = players[1 - seat].get("prize") or []
    return len(mine), len(theirs)


def convert(path: Path, matches: Path, out: Path, batch_size: int, limit: int | None) -> None:
    parquet = pq.ParquetFile(path)
    outcomes = load_outcomes(matches)
    print(f"[value-data] {path.name}: {parquet.metadata.num_rows:,} rows, "
          f"{parquet.metadata.num_row_groups} row groups")
    print(f"[value-data] {len(outcomes):,} match outcomes from {matches.name}")

    rows: list[list[float]] = []
    labels: list[float] = []
    turns: list[int] = []
    games: list[int] = []
    #: episode -> (last frame_index seen, margin at that frame). The margin is filled in
    #: per row afterwards, because a frame's own margin is the *episode's* final one.
    last_margin: dict[int, tuple[int, float]] = {}
    row_episode: list[int] = []

    seen = kept = draws = unmatched = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(COLUMNS)):
        table = batch.to_pylist()
        for record in table:
            seen += 1
            if not record.get("is_target_archetype"):
                continue
            state = record.get("state")
            if not state or not state.get("players"):
                continue
            seat = int(record["player_index"])
            episode = int(record["episode_id"])
            outcome = outcomes.get(episode)
            if outcome is None:
                unmatched += 1
                continue
            reward = outcome[seat]

            mine, theirs = prize_counts(state, seat)
            frame = int(record.get("frame_index") or 0)
            previous = last_margin.get(episode)
            if previous is None or frame >= previous[0]:
                last_margin[episode] = (frame, (theirs - mine) / 6.0)

            if reward == 0.0:  # a draw grades neither side
                draws += 1
                continue
            labels.append(1.0 if reward > 0 else -1.0)
            rows.append(scalar_features(state, seat))
            turns.append(int(state.get("turn") or 0))
            games.append(episode)
            row_episode.append(episode)
            kept += 1
            if limit and kept >= limit:
                break
        if limit and kept >= limit:
            break
        print(f"  ... {seen:,} scanned, {kept:,} kept", end="\r", flush=True)

    if not rows:
        raise SystemExit("no usable frames — check --parquet and is_target_archetype")

    margins = np.array([last_margin[e][1] for e in row_episode], dtype=np.float32)
    x = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    np.savez_compressed(
        out,
        rows=x, labels=y, turns=np.asarray(turns, np.int32),
        games=np.asarray(games, np.int64),
        # ``sources`` groups rows by generating distribution; this corpus is one source.
        sources=np.zeros(len(y), np.int16),
        mixture=np.array(["grimmsnarl-replay|marnie|1.0"]),
    )
    print()
    print(f"[value-data] {len(y):,} rows -> {out}")
    print(f"[value-data] {len(set(games)):,} games, {draws:,} draw frames dropped, "
          f"{unmatched:,} frames with no match outcome")
    print(f"[value-data] label balance {float((y > 0).mean()):.1%} positive")
    print(f"[value-data] mean |margin| {float(np.abs(margins).mean()):.3f}, "
          f"margin range [{margins.min():+.2f}, {margins.max():+.2f}]")
    print(f"[value-data] {x.shape[1]} features (SCALAR_NAMES has {len(SCALAR_NAMES)})")
    if x.shape[1] != len(SCALAR_NAMES):
        raise SystemExit("feature width mismatch — value_features.SCALAR_NAMES changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", required=True, help="the *-cleaned.parquet corpus")
    parser.add_argument("--matches", required=True,
                        help="matches.parquet for the same date range — the ONLY source of "
                             "the outcome label (the corpus's final_result is unpopulated)")
    parser.add_argument("--out", default=str(_ROOT / "value_data_grimsnarl.npz"))
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="parquet rows expanded to Python at a time")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many kept rows (smoke tests)")
    args = parser.parse_args()
    convert(Path(args.parquet), Path(args.matches), Path(args.out),
            args.batch_size, args.limit)


if __name__ == "__main__":
    main()
