from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "engine"))

from helpers.helper import build_deck  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start, visualize_data  # noqa: E402

import replay_log  # noqa: E402

MAX_STEPS = 5000


def random_choice(select: dict, rng: random.Random) -> list[int]:
    n_options = len(select["option"])
    count = rng.randint(select["minCount"], min(select["maxCount"], n_options))
    return rng.sample(range(n_options), count)


def play(deck0: list[int], deck1: list[int], names: list[str], rng: random.Random) -> tuple[int, int, Path | None]:
    obs, start = battle_start(deck0, deck1)
    if obs is None:
        raise ValueError(f"player {start.errorPlayer}'s deck is illegal (error {start.errorType})")
    recorder = replay_log.Recorder(names, [deck0, deck1], "random_probe", obs)
    steps = 0
    try:
        while obs["current"]["result"] == -1 and steps < MAX_STEPS:
            seat = obs["current"]["yourIndex"]
            choice = random_choice(obs["select"], rng)
            obs = battle_select(choice)
            recorder.selection(seat, choice, obs)
            steps += 1
        result = obs["current"]["result"]
        visualize = json.loads(visualize_data())
        path = recorder.save(result, visualize)
    finally:
        battle_finish()
    return result, steps, path


def main() -> None:
    parser = argparse.ArgumentParser(description="Play random agent vs random agent and save replays.")
    parser.add_argument("--deck0", default="./decks/main_agent.csv", help="seat 0 decklist (relative to rl-train/ is fine)")
    parser.add_argument("--deck1", default="./decks/opponents/alakazam.csv", help="seat 1 decklist")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    deck0, deck1 = build_deck(args.deck0), build_deck(args.deck1)
    names = [f"random_{Path(args.deck0).stem}", f"random_{Path(args.deck1).stem}"]
    for g in range(args.games):
        result, steps, path = play(deck0, deck1, names, rng)
        verdict = {-1: "unfinished (step cap)", 2: "draw"}.get(result, f"{names[result]} wins")
        print(f"game {g + 1}: {verdict} after {steps} selections -> {path}")


if __name__ == "__main__":
    main()
