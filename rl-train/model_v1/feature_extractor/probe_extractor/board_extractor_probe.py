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

from cg.api import EnergyType, all_attack, all_card_data  # noqa: E402
from board_extractor import (  # noqa: E402
    CSV_COLUMNS,
    ENERGY_COUNT_COLUMNS,
    MATCHUP_COLUMNS,
    MAX_PRE_EVOLUTION,
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
CARDS = {c.cardId: c for c in all_card_data()}
ATTACKS = {a.attackId: a for a in all_attack()}


def matching_shortfall(required: list[int], attached: list[int]) -> int:
    units = []
    for e in attached:
        if e == EnergyType.RAINBOW:
            units.append(None)
        elif e == EnergyType.TEAM_ROCKET:
            units += [int(EnergyType.PSYCHIC), int(EnergyType.DARKNESS)]
        else:
            units.append(int(e))
    owner: dict[int, int] = {}

    def augment(i: int, seen: set[int]) -> bool:
        for j, unit in enumerate(units):
            if j in seen or not (required[i] == EnergyType.COLORLESS or unit is None or unit == required[i]):
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    return len(required) - sum(augment(i, set()) for i in range(len(required)))


def expected_matchup(pokemon: dict, target: dict | None) -> dict[str, float]:
    attacks = [ATTACKS[a] for a in CARDS[pokemon["id"]].attacks if a in ATTACKS]
    out = dict.fromkeys(MATCHUP_COLUMNS, 0.0)
    if not attacks:
        return out
    me, defender = CARDS[pokemon["id"]], CARDS[target["id"]] if target and target["hp"] > 0 else None
    shortfalls, ratios, kos = [], [], []
    for attack in attacks:
        shortfalls.append(matching_shortfall([int(e) for e in attack.energies], [int(e) for e in pokemon["energies"]]))
        damage = attack.damage
        if defender is not None and damage > 0:
            if defender.weakness is not None and me.energyType == defender.weakness:
                damage *= 2
            if defender.resistance is not None and me.energyType == defender.resistance:
                damage = max(0, damage - 30)
        ratios.append(min(damage / target["hp"], 2.0) if defender is not None else 0.0)
        kos.append(defender is not None and damage > 0 and damage >= target["hp"])
    out["has_attack"] = 1.0
    out["can_attack"] = float(0 in shortfalls)
    out["min_shortfall"] = float(min(shortfalls))
    out["best_damage_ratio"] = max(ratios)
    out["ready_damage_ratio"] = max((r for r, s in zip(ratios, shortfalls) if s == 0), default=0.0)
    out["ready_ko"] = float(any(k for k, s in zip(kos, shortfalls) if s == 0))
    out["ko_within_one"] = float(any(k for k, s in zip(kos, shortfalls) if s <= 1))
    return out | {"_best_shortfalls": sorted({s for s in shortfalls})}


def random_choice(select: dict, rng: random.Random) -> list[int]:
    n_options = len(select["option"])
    count = rng.randint(select["minCount"], min(select["maxCount"], n_options))
    return rng.sample(range(n_options), count)


def check(extractor: BoardExtractor, obs: dict, features: dict[str, np.ndarray]) -> list[str]:
    problems = []
    for key, (shape, dtype) in BoardExtractor.SPECS.items():
        arr = features[key]
        if arr.shape != shape or arr.dtype != dtype:
            problems.append(f"{key}: got {arr.shape} {arr.dtype}, want {shape} {np.dtype(dtype)}")
    for key in ("input_ids", "tools", "energies", "pre_evolution"):
        if features[key].min() < 0 or features[key].max() >= extractor.vocab_size:
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
            got = features["input_ids"][slot]
            if pokemon == "empty":
                if got != 0:
                    problems.append(f"slot {slot}: should be empty, holds {extractor.vocab[got]}")
            elif pokemon is not None and got != extractor._card_index(pokemon["id"]):
                problems.append(f"slot {slot}: card {extractor.vocab[got]}, obs has id {pokemon['id']}")
            elif pokemon is not None:
                row = decoded[slot]
                if row["hp"] != pokemon["hp"] or row["appear_this_turn"] != pokemon["appearThisTurn"]:
                    problems.append(f"slot {slot}: hp/appear_this_turn mismatch")
                pre = [extractor._card_index(c["id"]) for c in pokemon["preEvolution"][:MAX_PRE_EVOLUTION]]
                pre += [0] * (MAX_PRE_EVOLUTION - len(pre))
                if list(features["pre_evolution"][slot]) != pre:
                    problems.append(f"slot {slot}: pre_evolution differs from the observation")
                want = expected_matchup(pokemon, (cur["players"][1 - seat]["active"] or [None])[0])
                for name in MATCHUP_COLUMNS:
                    if name == "best_shortfall":
                        if want["has_attack"] and row[name] not in want["_best_shortfalls"]:
                            problems.append(f"slot {slot}: best_shortfall {row[name]} is not any attack's shortfall")
                    elif abs(row[name] - want[name]) > 2e-3:
                        problems.append(f"slot {slot}: {name} is {row[name]}, recomputed {want[name]:.3f}")
                effective = Counter(EnergyType(e).name.lower() for e in pokemon["energies"])
                if {n: row[n] for n in ENERGY_COUNT_COLUMNS} != {f"energy_{t.name.lower()}": effective[t.name.lower()] for t in EnergyType}:
                    problems.append(f"slot {slot}: effective Energy counts differ from the observation")
        for slot in range(active_slot, bench_slot + MAX_BENCH):
            if expected.get(slot) is None and (features["pre_evolution"][slot].any() or features["energy_counts"][slot].any() or features["matchup"][slot].any()):
                problems.append(f"slot {slot}: empty slot has pre_evolution / energy_counts / matchup")
        available = [features["card_masks"][bench_slot + i] for i in range(MAX_BENCH)]
        if available != [i < player["benchMax"] for i in range(MAX_BENCH)]:
            problems.append(f"seat {seat}: bench card_masks {available} disagrees with benchMax {player['benchMax']}")
        if not features["card_masks"][active_slot]:
            problems.append(f"slot {active_slot}: Active slot masked")
    if decoded[0]["turn_action_count"] != cur["turnActionCount"]:
        problems.append(f"turn_action_count {decoded[0]['turn_action_count']}, obs has {cur['turnActionCount']}")
    if not features["card_masks"][STADIUM_IN_PLAY:].all():
        problems.append("stadium / per-turn slots must never be masked")
    if bool(cur["stadium"]) != bool(features["input_ids"][STADIUM_IN_PLAY]):
        problems.append("stadium slot does not match obs")
    for slot, flag in ((19, "supporterPlayed"), (20, "energyAttached"), (21, "stadiumPlayed")):
        if cur[flag] and features["input_ids"][slot] == 1:
            problems.append(f"slot {slot}: {flag} is set but the played card was not found in the logs")
        if cur[flag] != bool(features["input_ids"][slot]):
            problems.append(f"slot {slot}: filled disagrees with {flag}")
    return problems


def show(extractor: BoardExtractor, obs: dict, features: dict[str, np.ndarray], step: int) -> None:
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
    print(f"card vocab size: {extractor.vocab_size}, slots: {N_SLOTS}")

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
