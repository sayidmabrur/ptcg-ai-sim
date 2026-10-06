from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_RL_TRAIN = _HERE.parents[2]
sys.path.insert(0, str(_RL_TRAIN))
sys.path.insert(0, str(_RL_TRAIN / "engine"))
sys.path.insert(0, str(_HERE.parent))

from helpers.helper import build_deck  # noqa: E402
from cg.api import SelectContext  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

from extractor import OWNERS  # noqa: E402
from card_set_extractor import MAX_HAND, MAX_LOOKING, MAX_PILE, PILES, CardSetExtractor  # noqa: E402

MAX_STEPS = 5000
CSV_COLUMNS = ["game", "step", "seat", "select_context", "zone", "position", "card", "count"]


def random_choice(select: dict, rng: random.Random) -> list[int]:
    n_options = len(select["option"])
    count = rng.randint(select["minCount"], min(select["maxCount"], n_options))
    return rng.sample(range(n_options), count)


def check(extractor: CardSetExtractor, obs: dict, features: dict[str, np.ndarray], memory: dict) -> list[str]:
    problems = []
    cur = obs["current"]
    me = cur["yourIndex"]
    mine, theirs = cur["players"][me], cur["players"][1 - me]
    name = lambda card_id: extractor.vocab[extractor._card_index(card_id)]

    for key, (shape, dtype) in CardSetExtractor.SPECS.items():
        if features[key].shape != shape or features[key].dtype != dtype:
            problems.append(f"{key}: {features[key].shape} {features[key].dtype}, expected {shape} {np.dtype(dtype)}")
    for key in ("hand_ids", "pile_ids"):
        if features[key].min() < 0 or features[key].max() >= extractor.vocab_size:
            problems.append(f"{key} has an index outside the card vocab")

    hand_mask = features["hand_mask"]
    if not (hand_mask[: hand_mask.sum()].all() and not hand_mask[hand_mask.sum():].any()):
        problems.append("hand_mask is not a prefix of Trues")
    if list(features["hand_ids"][hand_mask]) != [extractor._card_index(c["id"]) for c in mine["hand"][:MAX_HAND]]:
        problems.append("hand differs from the observation (content or order)")
    if (features["hand_ids"][~hand_mask] != 0).any():
        problems.append("hand_ids is not 0 where hand_mask is False")

    looking = cur["looking"] or []
    looking_mask = features["looking_mask"]
    if int(looking_mask.sum()) != min(len(looking), MAX_LOOKING) or not looking_mask[: looking_mask.sum()].all():
        problems.append("looking_mask does not cover the looked-at cards")
    for i, card in enumerate(looking[:MAX_LOOKING]):
        want = extractor._card_index(card["id"] if card else None)
        owner = "NEUTRAL" if card is None else ("ME" if card["playerIndex"] == me else "OPP")
        if features["looking_ids"][i] != want or OWNERS[features["looking_owner"][i]] != owner:
            problems.append(f"looking card {i} differs from the observation")
    if (features["looking_ids"][~looking_mask] != 0).any() or (features["looking_owner"][~looking_mask] != 0).any():
        problems.append("looking padding is not zero")

    for p, pile in enumerate(PILES):
        ids, counts, mask = features["pile_ids"][p], features["pile_counts"][p], features["pile_mask"][p]
        if (ids[~mask] != 0).any() or (counts[~mask] != 0).any():
            problems.append(f"{pile}: padding is not zero")
        if (np.diff(ids[mask]) <= 0).any():
            problems.append(f"{pile}: ids are not sorted and unique")

    decoded = extractor.decode(features)
    for pile, discard in (("my_discard", mine["discard"]), ("opp_discard", theirs["discard"])):
        expected = Counter(name(c["id"]) for c in discard)
        if decoded[pile] != dict(expected):
            problems.append(f"{pile}: counts differ from the observation")

    seen = decoded["opp_seen"]
    previous = memory.get(me, {})
    if any(seen.get(card, 0) < count for card, count in previous.items()):
        problems.append("opp_seen lost a card it had already seen")
    memory[me] = dict(seen)
    for card, count in Counter(name(c["id"]) for c in theirs["discard"]).items():
        if seen.get(card, 0) < count:
            problems.append(f"opp_seen holds {seen.get(card, 0)} {card}, their discard has {count}")
    board = Counter(name(p["id"]) for p in theirs["active"] + theirs["bench"] if p)
    for card, count in board.items():
        if seen.get(card, 0) < count:
            problems.append(f"opp_seen misses {card} on their board")
    deck = Counter(name(c) for c in extractor.decklists[1 - me])
    for card, count in seen.items():
        if count > deck[card]:
            problems.append(f"opp_seen has {count} {card}, their decklist only {deck[card]}")

    face_down_active = bool(mine["active"]) and mine["active"][0] is None
    unseen = sum(decoded["my_unseen"].values())
    looking = sum(1 for c in (cur["looking"] or []) if c is not None and c["playerIndex"] == me)
    expected_unseen = mine["deckCount"] + len(mine["prize"]) + looking
    if unseen != expected_unseen and not face_down_active:
        problems.append(f"my_unseen holds {unseen} cards, deck + prizes is {expected_unseen}")
    return problems


def show(extractor: CardSetExtractor, obs: dict, features: dict[str, np.ndarray], step: int) -> None:
    print(f"\n--- decision {step}, context {SelectContext(obs['select']['context']).name} ---")
    np.set_printoptions(linewidth=200, suppress=True, precision=3)
    for key, arr in features.items():
        print(f"\n{key} {arr.shape} {arr.dtype}:\n{arr[..., : 12]}  (first 12 columns)")
    decoded = extractor.decode(features)
    print("\nhand:", decoded["hand"])
    print("looking:", decoded["looking"])
    for pile in PILES:
        print(f"{pile}: {decoded[pile]}")


def play(extractor, deck0, deck1, rng, game, writer, show_step, problems, stats) -> tuple[int, int]:
    memory: dict = {}
    obs, start = battle_start(deck0, deck1)
    if obs is None:
        raise ValueError(f"player {start.errorPlayer}'s deck is illegal (error {start.errorType})")
    extractor.reset()
    steps = 0
    try:
        while obs["current"]["result"] == -1 and steps < MAX_STEPS:
            if obs.get("current") is not None and obs.get("select") is not None:
                features = extractor.encode(obs)
                for p in check(extractor, obs, features, memory):
                    problems.append(f"game {game} step {steps}: {p}")
                stats["hand"] = max(stats["hand"], int(features["hand_mask"].sum()))
                stats["looking"] = max(stats["looking"], int(features["looking_mask"].sum()))
                stats["pile"] = max(stats["pile"], int(features["pile_mask"].sum(axis=1).max()))
                if steps == show_step and game == 1:
                    show(extractor, obs, features, steps)
                meta = {
                    "game": game,
                    "step": steps,
                    "seat": obs["current"]["yourIndex"],
                    "select_context": SelectContext(obs["select"]["context"]).name,
                }
                decoded = extractor.decode(features)
                for position, card in enumerate(decoded["hand"]):
                    writer.writerow({**meta, "zone": "hand", "position": position, "card": card, "count": 1})
                for position, (owner, card) in enumerate(decoded["looking"]):
                    writer.writerow({**meta, "zone": f"looking:{owner}", "position": position, "card": card, "count": 1})
                for pile in PILES:
                    for card, count in decoded[pile].items():
                        writer.writerow({**meta, "zone": pile, "position": "", "card": card, "count": count})
            obs = battle_select(random_choice(obs["select"], rng))
            steps += 1
        return obs["current"]["result"], steps
    finally:
        battle_finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CardSetExtractor over random games and dump what it produces.")
    parser.add_argument("--deck0", default="./decks/main_agent.csv")
    parser.add_argument("--deck1", default="./decks/opponents/dragapult_noir.csv")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", type=int, default=30, help="print the arrays of this decision in game 1 (-1: none)")
    parser.add_argument("--raw", action="store_true", help="raw pile counts instead of log-scaled")
    parser.add_argument("--out", default=str(_HERE / "card_set_extractor_probe_output.csv"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    deck0, deck1 = build_deck(args.deck0), build_deck(args.deck1)
    extractor = CardSetExtractor({0: deck0, 1: deck1}, normalize=not args.raw)
    print(f"card vocab size: {extractor.vocab_size}, MAX_HAND: {MAX_HAND}, MAX_PILE: {MAX_PILE}")

    problems: list[str] = []
    stats = {"hand": 0, "pile": 0, "looking": 0}
    total = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for g in range(1, args.games + 1):
            result, steps = play(extractor, deck0, deck1, rng, g, writer, args.show, problems, stats)
            total += steps
            verdict = {-1: "unfinished (step cap)", 2: "draw"}.get(result, f"seat {result} wins")
            print(f"game {g}: {verdict} after {steps} decisions")

    print(f"\n{total} decisions -> {args.out}")
    print(f"largest hand: {stats['hand']} / {MAX_HAND}, largest pile (unique cards): {stats['pile']} / {MAX_PILE}, "
          f"most looked-at cards: {stats['looking']} / {MAX_LOOKING}")
    if problems:
        print(f"{len(problems)} problems, first 20:")
        for p in problems[:20]:
            print("  " + p)
    else:
        print("all decisions match the raw observations")


if __name__ == "__main__":
    main()
