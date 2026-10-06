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
from cg.api import LogType, SelectContext  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

from extractor import N_AREAS  # noqa: E402
from log_extractor import _SERIAL_FIELDS, MAX_LOGS, N_LOG_TYPES, LogExtractor  # noqa: E402
from option_extractor import HAND_TOKEN_OFFSET, LOOKING_TOKEN_OFFSET, N_REF_TOKENS  # noqa: E402

MAX_STEPS = 5000
CSV_COLUMNS = [
    "game", "step", "seat", "select_context", "index", "type", "owner", "card_1", "card_2", "card_3", "ref_1", "ref_2",
    "from_area", "to_area", "attack", "value", "put_damage_counter", "is_recover", "head", "has_basic_pokemon", "is_new",
]


def random_choice(select: dict, rng: random.Random) -> list[int]:
    n_options = len(select["option"])
    count = rng.randint(select["minCount"], min(select["maxCount"], n_options))
    return rng.sample(range(n_options), count)


def check(extractor: LogExtractor, obs: dict, features: dict[str, np.ndarray], history: list) -> list[str]:
    problems = []
    me = obs["current"]["yourIndex"]
    hand = obs["current"]["players"][me]["hand"]
    looking = obs["current"]["looking"] or []
    history.extend(obs["logs"])
    del history[:-MAX_LOGS]
    kept = history
    n = len(kept)
    name = lambda card_id: extractor.vocab[extractor._card_index(card_id)]

    for key, (shape, dtype) in LogExtractor.SPECS.items():
        if features[key].shape != shape or features[key].dtype != dtype:
            problems.append(f"{key}: {features[key].shape} {features[key].dtype}, expected {shape} {np.dtype(dtype)}")
    mask = features["log_mask"]
    if int(mask.sum()) != n or not mask[:n].all():
        problems.append(f"log_mask has {int(mask.sum())} Trues, expected the last {n} logs")
    for key in ("log_type", "log_owner", "log_cards", "log_refs", "log_areas", "log_attack", "log_numeric"):
        if (features[key][~mask] != 0).any():
            problems.append(f"{key} is not 0 in the padding")
    if features["log_type"].max() >= N_LOG_TYPES or features["log_areas"].max() >= N_AREAS:
        problems.append("log_type / log_areas outside its embedding size")
    if features["log_refs"].max() >= N_REF_TOKENS:
        problems.append("a pointer is outside the token sequence")

    new = min(len(obs["logs"]), len(kept))
    for i, d in enumerate(extractor.decode(features)):
        if d["is_new"] != float(i >= len(kept) - new):
            problems.append(f"log {i}: is_new {d['is_new']}, expected {float(i >= len(kept) - new)}")
    for i, (log, d) in enumerate(zip(kept, extractor.decode(features))):
        kind = LogType(log["type"])
        where = f"log {i} ({kind.name})"
        if d["type"] != kind.name:
            problems.append(f"{where}: decoded as {d['type']}")
        expected_owner = "NEUTRAL" if log.get("playerIndex") is None else ("ME" if log["playerIndex"] == me else "OPP")
        if d["owner"] != expected_owner:
            problems.append(f"{where}: owner {d['owner']}, expected {expected_owner}")
        if kind in (LogType.DRAW, LogType.PLAY, LogType.MOVE_CARD, LogType.ATTACK, LogType.HP_CHANGE):
            if d["card_1"] != name(log["cardId"]):
                problems.append(f"{where}: card {d['card_1']}, observation has {name(log['cardId'])}")
        if kind == LogType.ATTACK and d["attack"] in ("<NONE>", "<UNK>"):
            problems.append(f"{where}: attack id {log.get('attackId')} not in the attack vocab")
        serial_keys = _SERIAL_FIELDS.get(kind, ("serial", None))
        for slot, serial_key in enumerate(serial_keys):
            ref = int(features["log_refs"][i, slot])
            if HAND_TOKEN_OFFSET <= ref < LOOKING_TOKEN_OFFSET:
                if hand[ref - HAND_TOKEN_OFFSET]["serial"] != log.get(serial_key):
                    problems.append(f"{where}: ref_{slot + 1} points at hand:{ref - HAND_TOKEN_OFFSET}, a different card")
            elif ref >= LOOKING_TOKEN_OFFSET:
                if looking[ref - LOOKING_TOKEN_OFFSET]["serial"] != log.get(serial_key):
                    problems.append(f"{where}: ref_{slot + 1} points at looking:{ref - LOOKING_TOKEN_OFFSET}, a different card")
    return problems


def show(extractor: LogExtractor, obs: dict, features: dict[str, np.ndarray], step: int) -> None:
    print(f"\n--- decision {step}, {len(obs['logs'])} logs ---")
    cols = ["index", "type", "owner", "card_1", "card_2", "ref_1", "ref_2", "from_area", "to_area", "attack", "value"]
    print(" | ".join(cols))
    for row in extractor.decode(features):
        print(" | ".join(str(row[c]) for c in cols))


def play(extractor, deck0, deck1, rng, game, writer, show_step, problems, stats) -> tuple[int, int]:
    histories: dict[int, list] = {0: [], 1: []}
    obs, start = battle_start(deck0, deck1)
    if obs is None:
        raise ValueError(f"player {start.errorPlayer}'s deck is illegal (error {start.errorType})")
    extractor.reset()
    steps = 0
    try:
        while obs["current"]["result"] == -1 and steps < MAX_STEPS:
            if obs.get("current") is not None and obs.get("select") is not None:
                features = extractor.encode(obs)
                for p in check(extractor, obs, features, histories[obs["current"]["yourIndex"]]):
                    problems.append(f"game {game} step {steps}: {p}")
                n = len(obs["logs"])
                stats["max_logs"] = max(stats["max_logs"], n)
                stats["truncated"] += len(histories[obs["current"]["yourIndex"]]) >= MAX_LOGS and n > 0
                if steps == show_step and game == 1:
                    show(extractor, obs, features, steps)
                context = SelectContext(obs["select"]["context"]).name
                meta = {"game": game, "step": steps, "seat": obs["current"]["yourIndex"], "select_context": context}
                for row in extractor.decode(features):
                    stats["types"][row["type"]] += 1
                    stats["refs"][row["type"]] += bool(row["ref_1"])
                    writer.writerow({**meta, **{c: row[c] for c in CSV_COLUMNS if c in row}})
            obs = battle_select(random_choice(obs["select"], rng))
            steps += 1
        return obs["current"]["result"], steps
    finally:
        battle_finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LogExtractor over random games and dump what it produces.")
    parser.add_argument("--deck0", default="./decks/main_agent.csv")
    parser.add_argument("--deck1", default="./decks/opponents/dragapult_noir.csv")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", type=int, default=30, help="print the logs of this decision in game 1 (-1: none)")
    parser.add_argument("--raw", action="store_true", help="unscaled numeric array (the CSV is the same either way)")
    parser.add_argument("--out", default=str(_HERE / "log_extractor_probe_output.csv"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    deck0, deck1 = build_deck(args.deck0), build_deck(args.deck1)
    extractor = LogExtractor(normalize=not args.raw)
    print(f"card vocab size: {extractor.vocab_size}, MAX_LOGS: {MAX_LOGS}")

    problems: list[str] = []
    stats = {"max_logs": 0, "truncated": 0, "types": Counter(), "refs": Counter()}
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
    print(f"most logs in one decision: {stats['max_logs']} (MAX_LOGS {MAX_LOGS}); decisions where the history was full (oldest dropped): {stats['truncated']}")
    print("log types seen:", dict(stats["types"].most_common()))
    print("logs whose first pointer resolved to the board/hand:", {t: f"{stats['refs'][t]}/{n}" for t, n in stats["types"].items() if stats["refs"][t]})
    if problems:
        print(f"{len(problems)} problems, first 20:")
        for p in problems[:20]:
            print("  " + p)
    else:
        print("all decisions match the raw observations")


if __name__ == "__main__":
    main()
