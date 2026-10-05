"""Play random vs random and run ``BoardExtractor`` on every decision, to see what it produces.

    python model_v1/feature_extractor/board_extractor_probe.py
    python model_v1/feature_extractor/board_extractor_probe.py --games 3 --deck1 ./decks/opponents/crustle.csv
    python model_v1/feature_extractor/board_extractor_probe.py --show 40     # print decision 40's arrays and table

Every decision becomes 22 rows (one per board slot) in a CSV with the columns
of ``board_state_preprocessed_example.csv``, plus game/step/seat/select context
so rows from the same turn can be told apart. The rows are decoded *from the
arrays* the model will see, so the CSV shows the transformed data, not the raw
observation. Each decision is also checked against the raw observation, and
any mismatch is reported at the end.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_RL_TRAIN = _HERE.parents[1]
sys.path.insert(0, str(_RL_TRAIN))  # helpers.helper
sys.path.insert(0, str(_RL_TRAIN / "engine"))  # the engine as ``cg``
sys.path.insert(0, str(_HERE))

from helpers.helper import build_deck  # noqa: E402
from cg.api import SelectContext  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

from board_extractor import (  # noqa: E402
    CSV_COLUMNS,
    MAX_BENCH,
    MY_ACTIVE,
    MY_BENCH,
    N_SLOTS,
    OPP_ACTIVE,
    OPP_BENCH,
    STADIUM_IN_PLAY,
    BoardExtractor,
)

MAX_STEPS = 5000


def random_choice(select: dict, rng: random.Random) -> list[int]:
    """A legal selection: between minCount and maxCount distinct option indexes."""
    n_options = len(select["option"])
    count = rng.randint(select["minCount"], min(select["maxCount"], n_options))
    return rng.sample(range(n_options), count)


def check(extractor: BoardExtractor, obs: dict, features: dict[str, np.ndarray]) -> list[str]:
    """Compare the arrays with the raw observation; returns a list of problems."""
    problems = []
    for key, (shape, dtype) in BoardExtractor.SPECS.items():
        arr = features[key]
        if arr.shape != shape or arr.dtype != dtype:
            problems.append(f"{key}: got {arr.shape} {arr.dtype}, want {shape} {np.dtype(dtype)}")
    for key in ("card_ids", "tools", "energies"):
        if features[key].min() < 0 or features[key].max() >= extractor.card_vocab_size:
            problems.append(f"{key}: index out of card vocab")

    cur = obs["current"]
    me = cur["yourIndex"]
    decoded = extractor.decode(features)
    for seat, active_slot, bench_slot in ((me, MY_ACTIVE, MY_BENCH), (1 - me, OPP_ACTIVE, OPP_BENCH)):
        player = cur["players"][seat]
        expected = {}
        if player["active"]:
            expected[active_slot] = player["active"][0]
        for i, pokemon in enumerate(player["bench"][:MAX_BENCH]):
            expected[bench_slot + i] = pokemon
        if len(player["bench"]) > MAX_BENCH:
            problems.append(f"seat {seat} bench has {len(player['bench'])} > {MAX_BENCH} Pokémon")
        for slot in [active_slot] + list(range(bench_slot, bench_slot + MAX_BENCH)):
            pokemon = expected.get(slot, "empty")
            got = features["card_ids"][slot]
            if pokemon == "empty":
                if got != 0:
                    problems.append(f"slot {slot}: should be empty, holds {extractor.card_vocab[got]}")
            elif pokemon is not None and got != extractor._card_index(pokemon["id"]):
                problems.append(f"slot {slot}: card {extractor.card_vocab[got]}, obs has id {pokemon['id']}")
            elif pokemon is not None:
                row = decoded[slot]
                if row["hp"] != pokemon["hp"] or row["appear_this_turn"] != pokemon["appearThisTurn"]:
                    problems.append(f"slot {slot}: hp/appear_this_turn mismatch")
        available = [features["slot_mask"][bench_slot + i] for i in range(MAX_BENCH)]
        if available != [i < player["benchMax"] for i in range(MAX_BENCH)]:
            problems.append(f"seat {seat}: bench slot_mask {available} disagrees with benchMax {player['benchMax']}")
        if not features["slot_mask"][active_slot]:
            problems.append(f"slot {active_slot}: Active slot masked")
    if not features["slot_mask"][STADIUM_IN_PLAY:].all():
        problems.append("stadium / per-turn slots must never be masked")
    if bool(cur["stadium"]) != bool(features["card_ids"][STADIUM_IN_PLAY]):
        problems.append("stadium slot does not match obs")
    for slot, flag in ((19, "supporterPlayed"), (20, "energyAttached"), (21, "stadiumPlayed")):
        if cur[flag] and features["card_ids"][slot] == 1:  # <HIDDEN>: flag set, but the card was not found in logs
            problems.append(f"slot {slot}: {flag} is set but the played card was not found in the logs")
        if cur[flag] != bool(features["card_ids"][slot]):
            problems.append(f"slot {slot}: filled disagrees with {flag}")
    return problems


def show(extractor: BoardExtractor, obs: dict, features: dict[str, np.ndarray], step: int) -> None:
    """Print one decision: the raw arrays, then the decoded table."""
    cur, sel = obs["current"], obs["select"]
    print(f"\n=== step {step}: seat {cur['yourIndex']} selecting, turn {cur['turn']}, "
          f"context {SelectContext(sel['context']).name} ===")
    np.set_printoptions(linewidth=200, suppress=True)
    for key, arr in features.items():
        print(f"\n{key} {arr.shape} {arr.dtype}:\n{arr}")
    print()
    rows = extractor.decode(features)
    cols = ["slot", "card_id", "card_type", "zone", "owner", "available", "hp", "max_hp", "stage", "retreat_cost",
            "appear_this_turn", "cond_poisoned", "cond_asleep", "tool_1", "att_e_1", "att_e_2", "att_e_3"]
    print(" | ".join(cols))
    for row in rows:
        if row["card_id"] != "<NONE>" or row["slot"] in (MY_ACTIVE, OPP_ACTIVE):
            print(" | ".join(str(row[c]) for c in cols))
    print({k: rows[0][k] for k in rows[0] if k.startswith(("my_", "opp_", "is_my", "went", "retreated"))})


def play(extractor, deck0, deck1, rng, game, writer, show_step, problems) -> tuple[int, int]:
    """One game; every decision's features go to ``writer``. Returns (result, decisions)."""
    obs, start = battle_start(deck0, deck1)
    if obs is None:
        raise ValueError(f"player {start.errorPlayer}'s deck is illegal (error {start.errorType})")
    extractor.reset()
    steps = 0
    try:
        while obs["current"]["result"] == -1 and steps < MAX_STEPS:
            if obs.get("current") is not None and obs.get("select") is not None:
                features = extractor(obs)
                for p in check(extractor, obs, features):
                    problems.append(f"game {game} step {steps}: {p}")
                if steps == show_step and game == 1:
                    show(extractor, obs, features, steps)
                meta = {
                    "game": game,
                    "step": steps,
                    "seat": obs["current"]["yourIndex"],
                    "select_context": SelectContext(obs["select"]["context"]).name,
                }
                for row in extractor.decode(features):
                    writer.writerow({**meta, **row})
            obs = battle_select(random_choice(obs["select"], rng))
            steps += 1
        return obs["current"]["result"], steps
    finally:
        battle_finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BoardExtractor over random games and dump what it produces.")
    parser.add_argument("--deck0", default="./decks/main_agent.csv")
    parser.add_argument("--deck1", default="./decks/opponents/dragapult_noir.csv")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", type=int, default=30, help="print the arrays of this decision in game 1 (-1: none)")
    parser.add_argument("--raw", action="store_true", help="unscaled numeric/global arrays (the CSV is the same either way)")
    parser.add_argument("--out", default=str(_HERE / "board_extractor_probe_output.csv"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    deck0, deck1 = build_deck(args.deck0), build_deck(args.deck1)
    extractor = BoardExtractor(normalize=not args.raw)
    print(f"card vocab size: {extractor.card_vocab_size}, slots: {N_SLOTS}")

    problems: list[str] = []
    fieldnames = ["game", "step", "seat", "select_context"] + CSV_COLUMNS
    total = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for g in range(1, args.games + 1):
            result, steps = play(extractor, deck0, deck1, rng, g, writer, args.show, problems)
            total += steps
            verdict = {-1: "unfinished (step cap)", 2: "draw"}.get(result, f"seat {result} wins")
            print(f"game {g}: {verdict} after {steps} decisions")

    print(f"\n{total} decisions x {N_SLOTS} slots -> {args.out}")
    if problems:
        print(f"{len(problems)} problems, first 20:")
        for p in problems[:20]:
            print("  " + p)
    else:
        print("all decisions match the raw observations")


if __name__ == "__main__":
    main()
