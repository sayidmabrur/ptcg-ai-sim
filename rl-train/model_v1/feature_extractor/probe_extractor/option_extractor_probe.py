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
from cg.api import AreaType, OptionType, SelectContext  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

from extractor import N_AREAS  # noqa: E402
from option_extractor import (  # noqa: E402
    BOARD_TOKEN_OFFSET,
    HAND_TOKEN_OFFSET,
    MAX_OPTIONS,
    N_OPTION_TYPES,
    N_REF_TOKENS,
    N_SELECT_CONTEXTS,
    N_SELECT_TYPES,
    OptionExtractor,
)

MAX_STEPS = 5000
CSV_COLUMNS = [
    "game", "step", "seat", "select_context", "select_type", "min_count", "max_count", "index", "type", "src_ref",
    "dst_ref", "src_area", "owner", "card", "sub_index", "attack", "number", "count", "attack_damage", "attack_cost",
    "damage_ratio", "is_ko",
]


def random_choice(select: dict, rng: random.Random) -> list[int]:
    n_options = len(select["option"])
    count = rng.randint(select["minCount"], min(select["maxCount"], n_options))
    return rng.sample(range(n_options), count)


def raw_card_id(obs: dict, area, index, player) -> int | None | bool:
    cur, select = obs["current"], obs["select"]
    if area is None or index is None:
        return False
    p = cur["players"][player]
    if area in (AreaType.ACTIVE, AreaType.BENCH):
        zone = p["active"] if area == AreaType.ACTIVE else p["bench"]
        pokemon = zone[index] if index < len(zone) else None
        return pokemon["id"] if pokemon else None
    zone = {
        AreaType.HAND: p["hand"], AreaType.DISCARD: p["discard"], AreaType.PRIZE: p["prize"],
        AreaType.DECK: select["deck"], AreaType.LOOKING: cur["looking"], AreaType.STADIUM: cur["stadium"],
    }.get(area, False)
    if zone is False:
        return False
    index = 0 if area == AreaType.STADIUM else index
    card = zone[index] if zone is not None and index < len(zone) else None
    return card["id"] if card else None


def check(extractor: OptionExtractor, obs: dict, features: dict[str, np.ndarray]) -> list[str]:
    problems = []
    cur, select = obs["current"], obs["select"]
    me = cur["yourIndex"]
    options = select["option"]
    n = min(len(options), MAX_OPTIONS)
    name = lambda card_id: extractor.vocab[extractor._card_index(card_id)]

    for key, (shape, dtype) in OptionExtractor.SPECS.items():
        if features[key].shape != shape or features[key].dtype != dtype:
            problems.append(f"{key}: {features[key].shape} {features[key].dtype}, expected {shape} {np.dtype(dtype)}")
    mask = features["option_mask"]
    if int(mask.sum()) != n or not mask[:n].all():
        problems.append(f"option_mask has {int(mask.sum())} Trues, expected {n} options")
    for key in ("option_type", "src_ref", "dst_ref", "src_area", "owner", "card_id", "sub_index", "attack_id",
                "special_condition"):
        if (features[key][~mask] != 0).any():
            problems.append(f"{key} is not 0 in the padding")
    if features["src_ref"].max() >= N_REF_TOKENS or features["dst_ref"].max() >= N_REF_TOKENS:
        problems.append("a pointer is outside the token sequence")
    if features["option_type"].max() >= N_OPTION_TYPES or features["src_area"].max() >= N_AREAS:
        problems.append("option_type / src_area outside its embedding size")
    if features["select_type"] >= N_SELECT_TYPES or features["select_context"] >= N_SELECT_CONTEXTS:
        problems.append("select_type / select_context outside its embedding size")
    if features["select_context"] != select["context"] + 1:
        problems.append("select_context differs from the observation")

    decoded = extractor.decode(features)
    if decoded["select"]["min_count"] != select["minCount"] or decoded["select"]["max_count"] != select["maxCount"]:
        problems.append("min/max count differ from the observation")

    for i, option in enumerate(options[:n]):
        kind = OptionType(option["type"])
        d = decoded["options"][i]
        where = f"option {i} ({kind.name})"
        if d["type"] != kind.name:
            problems.append(f"{where}: decoded as {d['type']}")
        player = me if option.get("playerIndex") is None else option["playerIndex"]
        area = AreaType.HAND if kind == OptionType.PLAY else option.get("area")

        if kind in (OptionType.PLAY, OptionType.CARD, OptionType.ATTACH, OptionType.EVOLVE, OptionType.ABILITY, OptionType.DISCARD):
            expected = raw_card_id(obs, area, option.get("index"), player)
            if expected is not False and d["card"] != name(expected):
                problems.append(f"{where}: card {d['card']}, observation has {name(expected)}")
        if kind == OptionType.PLAY and d["src_ref"] != f"hand:{option['index']}":
            problems.append(f"{where}: src_ref {d['src_ref']}, expected hand:{option['index']}")
        if kind in (OptionType.ATTACH, OptionType.EVOLVE) and option.get("inPlayArea") in (AreaType.ACTIVE, AreaType.BENCH):
            if not d["dst_ref"].startswith("board:"):
                problems.append(f"{where}: dst_ref {d['dst_ref']!r} does not point at the board")
        if area == AreaType.LOOKING and option.get("index") is not None and d["src_ref"] != f"looking:{option['index']}":
            problems.append(f"{where}: src_ref {d['src_ref']!r}, expected looking:{option['index']}")
        if area in (AreaType.ACTIVE, AreaType.BENCH) and kind != OptionType.PLAY and not d["src_ref"].startswith("board:"):
            problems.append(f"{where}: src_ref {d['src_ref']!r} does not point at the board")
        if kind == OptionType.ATTACK:
            if d["attack"] in ("<NONE>", "<UNK>"):
                problems.append(f"{where}: attack id {option.get('attackId')} not in the attack vocab")
            if not 0.0 <= d["damage_ratio"] <= 2.0:
                problems.append(f"{where}: damage_ratio {d['damage_ratio']} out of range")
    return problems


def show(extractor: OptionExtractor, obs: dict, features: dict[str, np.ndarray], step: int) -> None:
    decoded = extractor.decode(features)
    print(f"\n--- decision {step} ---")
    print({k: v for k, v in decoded["select"].items()})
    cols = ["index", "type", "src_ref", "dst_ref", "src_area", "owner", "card", "sub_index", "attack", "attack_damage",
            "damage_ratio", "is_ko", "number", "count"]
    print(" | ".join(cols))
    for row in decoded["options"]:
        print(" | ".join(str(row[c]) for c in cols))


def play(extractor, deck0, deck1, rng, game, writer, show_step, problems, stats) -> tuple[int, int]:
    obs, start = battle_start(deck0, deck1)
    if obs is None:
        raise ValueError(f"player {start.errorPlayer}'s deck is illegal (error {start.errorType})")
    extractor.reset()
    steps = 0
    try:
        while obs["current"]["result"] == -1 and steps < MAX_STEPS:
            if obs.get("current") is not None and obs.get("select") is not None:
                features = extractor.encode(obs)
                for p in check(extractor, obs, features):
                    problems.append(f"game {game} step {steps}: {p}")
                n = len(obs["select"]["option"])
                stats["max_options"] = max(stats["max_options"], n)
                context = SelectContext(obs["select"]["context"]).name
                stats["contexts"][context] += 1
                if steps == show_step and game == 1:
                    show(extractor, obs, features, steps)
                decoded = extractor.decode(features)
                meta = {
                    "game": game,
                    "step": steps,
                    "seat": obs["current"]["yourIndex"],
                    "select_context": context,
                    "select_type": decoded["select"]["type"],
                    "min_count": decoded["select"]["min_count"],
                    "max_count": decoded["select"]["max_count"],
                }
                for row in decoded["options"]:
                    stats["types"][row["type"]] += 1
                    writer.writerow({**meta, **{c: row[c] for c in CSV_COLUMNS if c in row}})
            obs = battle_select(random_choice(obs["select"], rng))
            steps += 1
        return obs["current"]["result"], steps
    finally:
        battle_finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OptionExtractor over random games and dump what it produces.")
    parser.add_argument("--deck0", default="./decks/main_agent.csv")
    parser.add_argument("--deck1", default="./decks/opponents/dragapult_noir.csv")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", type=int, default=30, help="print the options of this decision in game 1 (-1: none)")
    parser.add_argument("--raw", action="store_true", help="unscaled numeric arrays (the CSV is the same either way)")
    parser.add_argument("--out", default=str(_HERE / "option_extractor_probe_output.csv"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    deck0, deck1 = build_deck(args.deck0), build_deck(args.deck1)
    extractor = OptionExtractor(normalize=not args.raw)
    print(f"card vocab size: {extractor.vocab_size}, attack vocab size: {extractor.attack_vocab_size}, "
          f"MAX_OPTIONS: {MAX_OPTIONS}, pointer tokens: {N_REF_TOKENS} "
          f"(board {BOARD_TOKEN_OFFSET}.., hand {HAND_TOKEN_OFFSET}..)")

    problems: list[str] = []
    stats = {"max_options": 0, "contexts": Counter(), "types": Counter()}
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
    print(f"most options in one decision: {stats['max_options']} / {MAX_OPTIONS}")
    print("option types seen:", dict(stats["types"].most_common()))
    print("select contexts seen:", len(stats["contexts"]), "of", N_SELECT_CONTEXTS - 1)
    if problems:
        print(f"{len(problems)} problems, first 20:")
        for p in problems[:20]:
            print("  " + p)
    else:
        print("all decisions match the raw observations")


if __name__ == "__main__":
    main()
