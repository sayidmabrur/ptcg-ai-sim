from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"


def wilson(wins: float, games: float, z: float = 1.96) -> tuple[float, float]:
    if games == 0:
        return float("nan"), float("nan")
    p = wins / games
    denom = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denom
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denom
    return centre - half, centre + half


def pooled(rows: list[dict], key: str) -> tuple[float, float]:
    wins = games = 0.0
    for row in rows:
        n = float(row[f"n_{key}"])
        if n > 0:
            wins += float(row[f"wr_{key}"]) * n
            games += n
    return wins, games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name")
    parser.add_argument("--last", type=int, default=10, help="iterations to pool (0 = all)")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(RUNS / args.name / "metrics.csv")))
    rows = rows[-args.last:] if args.last else rows
    first, last = rows[0]["iteration"], rows[-1]["iteration"]
    print(f"run {args.name}: iterations {first}-{last} ({len(rows)} iterations)\n")

    columns = [c[3:] for c in rows[0] if c.startswith("wr_") and f"n_{c[3:]}" in rows[0]]
    decks = [c for c in columns if not c.startswith("vs_")]
    kinds = [c for c in columns if c.startswith("vs_")]

    def table(title: str, keys: list[str]) -> None:
        print(f"{title:<22}{'wins/games':>11}{'win rate':>10}   95% interval")
        for key in keys:
            wins, games = pooled(rows, key)
            low, high = wilson(wins, games)
            rate = f"{wins / games:.2f}" if games else "  - "
            print(f"  {key.removeprefix('vs_'):<20}{f'{wins:.0f}/{games:.0f}':>11}{rate:>10}   [{low:.2f}, {high:.2f}]")
        print()

    games_total = sum(pooled(rows, k)[1] for k in decks)
    wins_total = sum(pooled(rows, k)[0] for k in decks)
    low, high = wilson(wins_total, games_total)
    print(f"overall: {wins_total:.0f}/{games_total:.0f} won = {wins_total / games_total:.2f}   [{low:.2f}, {high:.2f}]")
    losses = sum(float(r["loss_rate"]) * sum(float(r[f"n_{k}"]) for k in decks) for r in rows if r["loss_rate"] != "nan")
    print(f"losses {losses:.0f}, draws {games_total - wins_total - losses:.0f}, "
          f"games cut off at the step limit: {sum(float(r['truncated']) for r in rows):.0f}\n")
    table("by opponent deck", decks)
    table("by opponent type", kinds)


if __name__ == "__main__":
    main()
