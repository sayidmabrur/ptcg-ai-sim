"""Turn a raw ``cg`` observation into a JSON view model for the browser.

The engine speaks integers (card ids, attack ids, enum ordinals); everything the
UI needs to draw a board and label a choice is resolved here so the client stays
a dumb renderer.
"""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ptcg_engine.api import (
    AreaType, CardType, EnergyType, LogType, OptionType, SelectContext,
    SelectType, SpecialConditionType,
)
from ptcg_engine.sim import lib

ENERGY_CODE = {
    EnergyType.COLORLESS: "C", EnergyType.GRASS: "G", EnergyType.FIRE: "R",
    EnergyType.WATER: "W", EnergyType.LIGHTNING: "L", EnergyType.PSYCHIC: "P",
    EnergyType.FIGHTING: "F", EnergyType.DARKNESS: "D", EnergyType.METAL: "M",
    EnergyType.DRAGON: "N", EnergyType.RAINBOW: "*", EnergyType.TEAM_ROCKET: "TR",
}

CARD_KIND = {
    CardType.POKEMON: "pokemon", CardType.ITEM: "item", CardType.TOOL: "tool",
    CardType.SUPPORTER: "supporter", CardType.STADIUM: "stadium",
    CardType.BASIC_ENERGY: "energy", CardType.SPECIAL_ENERGY: "energy",
}

AREA_LABEL = {
    AreaType.DECK: "deck", AreaType.HAND: "hand", AreaType.DISCARD: "discard",
    AreaType.ACTIVE: "active", AreaType.BENCH: "bench", AreaType.PRIZE: "prize",
    AreaType.STADIUM: "stadium", AreaType.ENERGY: "energy", AreaType.TOOL: "tool",
    AreaType.PRE_EVOLUTION: "pre-evolution", AreaType.PLAYER: "player",
    AreaType.LOOKING: "revealed",
}

PROMPT = {
    SelectContext.MAIN: "Your move",
    SelectContext.SETUP_ACTIVE_POKEMON: "Choose your Active Pokémon",
    SelectContext.SETUP_BENCH_POKEMON: "Choose Bench Pokémon",
    SelectContext.IS_FIRST: "Do you want to go first?",
    SelectContext.MULLIGAN: "Mulligan",
    SelectContext.ATTACK: "Choose an attack",
    SelectContext.SWITCH: "Choose a Pokémon to switch in",
    SelectContext.TO_ACTIVE: "Choose a Pokémon for the Active Spot",
    SelectContext.TO_BENCH: "Choose a Pokémon for the Bench",
    SelectContext.DISCARD: "Choose a card to discard",
    SelectContext.TO_HAND: "Choose a card to put into your hand",
    SelectContext.TO_DECK: "Choose a card to put on your deck",
    SelectContext.ATTACH_FROM: "Choose Energy to attach from",
    SelectContext.ATTACH_TO: "Choose a Pokémon to attach to",
    SelectContext.DETACH_FROM: "Choose Energy to remove",
    SelectContext.EVOLVE: "Choose a Pokémon to evolve",
    SelectContext.EVOLVES_TO: "Choose what to evolve into",
    SelectContext.EFFECT_TARGET: "Choose a target",
    SelectContext.COIN_HEAD: "Call the coin",
    SelectContext.ACTIVATE: "Use this ability?",
    SelectContext.LOOK: "Choose a card",
}


# Card art, one file per card id, served by the app at ``/cards/``. Every id in
# the engine's table has one today; a missing file simply falls back to the
# text-only card in the browser, so this must stay a lookup rather than a
# guessed URL.
def _image_dir() -> Path:
    """Where the card scans live.

    Looked up rather than hard-coded so the art can sit beside the simulator, in
    the repo root as it does today, or anywhere ``PTCG_CARD_IMAGES`` points —
    683 MB of scans is the one part of this that may not travel with the code.
    """
    env = os.environ.get("PTCG_CARD_IMAGES")
    candidates = [Path(env)] if env else []
    candidates += [_HERE / "card_images_en", _HERE.parent / "card_images_en"]
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[-1]


IMAGE_DIR = _image_dir()


@lru_cache(maxsize=1)
def image_index() -> dict[int, str]:
    index: dict[int, str] = {}
    if not IMAGE_DIR.is_dir():
        return index
    for path in IMAGE_DIR.iterdir():
        if path.suffix.lower() not in (".webp", ".png", ".jpg", ".jpeg"):
            continue
        try:
            index[int(path.stem)] = path.name
        except ValueError:
            continue
    return index


def card_image(card_id: int | None) -> str | None:
    name = image_index().get(card_id) if card_id is not None else None
    return f"/cards/{name}" if name else None


@lru_cache(maxsize=1)
def card_db() -> dict[int, dict]:
    return {c["cardId"]: c for c in json.loads(lib.AllCard().decode())}


@lru_cache(maxsize=1)
def attack_db() -> dict[int, dict]:
    return {a["attackId"]: a for a in json.loads(lib.AllAttack().decode())}


def _enum_name(enum_cls, value, default="?"):
    try:
        return enum_cls(value).name.replace("_", " ").title()
    except (ValueError, TypeError):
        return default


def card_info(card_id: int | None) -> dict:
    """Static card data, shaped for the front-end card component."""
    if card_id is None:
        return {"id": None, "name": "Face-down", "kind": "hidden", "facedown": True, "image": None}
    data = card_db().get(card_id)
    if data is None:
        return {"id": card_id, "name": f"Card {card_id}", "kind": "unknown", "image": None}
    attacks = [
        {
            "name": attack_db().get(a, {}).get("name", f"Attack {a}"),
            "damage": attack_db().get(a, {}).get("damage", 0),
            "text": attack_db().get(a, {}).get("text", ""),
            "cost": [ENERGY_CODE.get(e, "?") for e in attack_db().get(a, {}).get("energies", [])],
        }
        for a in data.get("attacks", [])
    ]
    return {
        "id": card_id,
        "name": data["name"],
        "image": card_image(card_id),
        "kind": CARD_KIND.get(data["cardType"], "unknown"),
        "hp": data.get("hp") or 0,
        "energy": ENERGY_CODE.get(data.get("energyType"), ""),
        "weakness": ENERGY_CODE.get(data["weakness"]) if data.get("weakness") is not None else None,
        "resistance": ENERGY_CODE.get(data["resistance"]) if data.get("resistance") is not None else None,
        "retreat": data.get("retreatCost", 0),
        "ex": data.get("ex", False),
        "megaEx": data.get("megaEx", False),
        "tera": data.get("tera", False),
        "aceSpec": data.get("aceSpec", False),
        "stage": ("Basic" if data.get("basic") else "Stage 1" if data.get("stage1")
                  else "Stage 2" if data.get("stage2") else ""),
        "evolvesFrom": data.get("evolvesFrom"),
        "skills": data.get("skills", []),
        "attacks": attacks,
        "facedown": False,
    }


def pokemon_view(mon: dict | None) -> dict | None:
    if mon is None:
        return None
    info = card_info(mon["id"])
    info.update({
        "serial": mon["serial"],
        "curHp": mon["hp"],
        "maxHp": mon["maxHp"],
        "energies": [ENERGY_CODE.get(e, "?") for e in mon.get("energies", [])],
        "tools": [card_info(t["id"])["name"] for t in mon.get("tools", [])],
        "preEvolution": [card_info(c["id"])["name"] for c in mon.get("preEvolution", [])],
        "appearThisTurn": mon.get("appearThisTurn", False),
    })
    return info


def facedown_view() -> dict:
    """A Pokémon that is in play but face-down.

    The engine says these are different things: an *empty* Active Spot is
    ``active == []``, while a face-down Pokémon standing in it is
    ``active == [None]`` — which is where both players are during Set Up, and
    where the opponent stays until the reveal. Collapsing the two made the board
    look empty for the whole of Set Up and the Pokémon appear from nowhere when
    the first turn began.
    """
    return card_info(None)


def player_view(state: dict, reveal_hand: bool) -> dict:
    active = state["active"]
    return {
        "active": (facedown_view() if active[0] is None else pokemon_view(active[0])) if active else None,
        "bench": [pokemon_view(m) if m is not None else facedown_view() for m in state["bench"]],
        "benchMax": state["benchMax"],
        "deckCount": state["deckCount"],
        "discardCount": len(state["discard"]),
        "discard": [card_info(c["id"]) for c in state["discard"]],
        "prizeCount": len(state["prize"]),
        "handCount": state["handCount"],
        "hand": [card_info(c["id"]) for c in (state["hand"] or [])] if reveal_hand else None,
        "conditions": [
            name for name, flag in (
                ("Poisoned", state["poisoned"]), ("Burned", state["burned"]),
                ("Asleep", state["asleep"]), ("Paralyzed", state["paralyzed"]),
                ("Confused", state["confused"]),
            ) if flag
        ],
    }


def _hand_name(state: dict, index: int | None) -> str:
    hand = state.get("hand") or []
    if index is None or index >= len(hand):
        return "a card"
    return card_info(hand[index]["id"])["name"]


def _in_play_target(state: dict, option: dict, seat: int) -> str | None:
    """Name of the Pokémon an ``inPlay*`` option lands on, with its spot."""
    area = option.get("inPlayArea")
    index = option.get("inPlayIndex")
    if area is None or index is None:
        return None
    probe = {"area": area, "index": index, "playerIndex": option.get("playerIndex")}
    card = _board_card(state, probe, seat)
    if card is None:
        return None
    spot = AREA_LABEL.get(area, "")
    return f"{card['name']} ({spot})" if spot else card["name"]


def _board_pokemon(state: dict, option: dict, seat: int) -> dict | None:
    """The raw in-play Pokémon an option points at, attachments included."""
    owner = option.get("playerIndex")
    owner = seat if owner is None else owner
    area = option.get("area") if option.get("area") is not None else option.get("inPlayArea")
    index = option.get("index") if option.get("area") is not None else option.get("inPlayIndex")
    player = state["players"][owner]
    if area == AreaType.ACTIVE:
        return player["active"][0] if player["active"] else None
    if area == AreaType.BENCH:
        bench = player["bench"]
        return bench[index] if index is not None and index < len(bench) else None
    return None


def _board_card(state: dict, option: dict, seat: int) -> dict | None:
    """The card an option points at, resolved through the board it sits on."""
    owner = option.get("playerIndex")
    owner = seat if owner is None else owner
    area = option.get("area") if option.get("area") is not None else option.get("inPlayArea")
    index = option.get("index") if option.get("area") is not None else option.get("inPlayIndex")
    player = state["players"][owner]
    if area == AreaType.ACTIVE:
        mon = player["active"][0] if player["active"] else None
    elif area == AreaType.BENCH:
        bench = player["bench"]
        mon = bench[index] if index is not None and index < len(bench) else None
    elif area == AreaType.STADIUM:
        stadium = state.get("stadium") or []
        return card_info(stadium[0]["id"]) if stadium else None
    elif area == AreaType.HAND and owner == seat:
        hand = player.get("hand") or []
        mon = hand[index] if index is not None and index < len(hand) else None
    elif area == AreaType.DISCARD:
        discard = player["discard"]
        mon = discard[index] if index is not None and index < len(discard) else None
    else:
        mon = None
    return card_info(mon["id"]) if mon else None


def option_label(option: dict, state: dict, seat: int, select: dict) -> str:
    """A human sentence for one option, resolved against the acting seat's board."""
    kind = option.get("type")
    me = state["players"][seat]

    if kind == OptionType.YES:
        return "Yes"
    if kind == OptionType.NO:
        return "No"
    if kind == OptionType.NUMBER:
        return str(option.get("number", "?"))
    if kind == OptionType.END:
        return "End turn"
    if kind == OptionType.RETREAT:
        return "Retreat"
    if kind == OptionType.PLAY:
        return f"Play {_hand_name(me, option.get('index'))}"
    if kind in (OptionType.EVOLVE, OptionType.ATTACH):
        # These come one per (card, target) pair, so without the target every
        # energy in hand produces a row of identical-looking options.
        card = _hand_name(me, option.get("index"))
        target = _in_play_target(state, option, seat)
        verb = "Evolve" if kind == OptionType.EVOLVE else "Attach"
        if kind == OptionType.EVOLVE:
            return f"Evolve {target} into {card}" if target else f"Evolve with {card}"
        return f"Attach {card} to {target}" if target else f"Attach {card}"
    if kind == OptionType.ATTACK:
        atk = attack_db().get(option.get("attackId"), {})
        dmg = atk.get("damage", 0)
        return f"Attack: {atk.get('name', '?')}" + (f" ({dmg})" if dmg else "")
    if kind in (OptionType.ABILITY, OptionType.SKILL):
        # An ABILITY option names a board location, not a card id, so the card
        # has to be looked up where it is in play before its ability can be
        # named -- otherwise every ability on the board reads "Ability".
        card = card_info(option["cardId"]) if option.get("cardId") else _board_card(state, option, seat)
        skills = (card or {}).get("skills") or []
        if len(skills) == 1:
            name = skills[0]["name"]
            return name if name == card["name"] else f"{name} ({card['name']})"
        if skills:
            names = ", ".join(s["name"] for s in skills)
            return f"Ability ({card['name']}: {names})"
        return f"Ability ({card['name']})" if card else "Ability"
    if kind == OptionType.SPECIAL_CONDITION:
        return _enum_name(SpecialConditionType, option.get("specialConditionType"), "Condition")
    if kind in (OptionType.ENERGY, OptionType.ENERGY_CARD, OptionType.TOOL_CARD):
        # These name an attachment by its slot on a Pokémon in play, so both the
        # attachment and the Pokémon carrying it have to be resolved.
        mon = _board_pokemon(state, option, seat)
        holder = card_info(mon["id"])["name"] if mon else "a Pokémon"
        if kind == OptionType.TOOL_CARD:
            tools = (mon or {}).get("tools") or []
            idx = option.get("toolIndex") or 0
            tool = card_info(tools[idx]["id"])["name"] if idx < len(tools) else "Tool"
            return f"{tool} on {holder}"
        idx = option.get("energyIndex") or 0
        if kind == OptionType.ENERGY_CARD:
            cards = (mon or {}).get("energyCards") or []
            name = card_info(cards[idx]["id"])["name"] if idx < len(cards) else "Energy"
            return f"{name} on {holder}"
        energies = (mon or {}).get("energies") or []
        code = ENERGY_CODE.get(energies[idx], "?") if idx < len(energies) else "?"
        count = option.get("count") or 1
        return f"{code} Energy on {holder}" + (f" (worth {count})" if count > 1 else "")

    # CARD / TOOL_CARD / ENERGY_CARD / DISCARD reference a board location.
    if option.get("cardId") is not None:
        name = card_info(option["cardId"])["name"]
    elif option.get("area") == AreaType.HAND:
        name = _hand_name(me, option.get("index"))
    elif (board := _board_card(state, option, seat)) is not None:
        # A Pokémon in play is identified by where it stands, not by a card id;
        # without this every bench target would read "a card (bench)".
        name = board["name"]
    elif select.get("deck") and option.get("index") is not None:
        deck = select["deck"]
        idx = option["index"]
        name = card_info(deck[idx]["id"])["name"] if idx < len(deck) else "a card"
    else:
        name = "a card"

    area = AREA_LABEL.get(option.get("area"), "")
    owner = ""
    if option.get("playerIndex") is not None and option["playerIndex"] != seat:
        owner = "opponent's "
    where = f" ({owner}{area})" if area and area not in ("hand",) else ""
    return f"{name}{where}"


def _ref(area, index, owner, seat, serial=None) -> dict | None:
    if area is None or index is None:
        return None
    label = AREA_LABEL.get(area)
    if label is None:
        return None
    return {
        "area": label,
        "index": index,
        "side": None if owner is None else ("me" if owner == seat else "opp"),
        "serial": serial,
    }


def option_refs(option: dict, seat: int) -> list[dict]:
    """Every board slot this option touches, so the UI can light them up.

    An option can name two places at once -- ``ATTACH``/``EVOLVE`` carry the card
    leaving your hand *and* the Pokémon it lands on -- and ``PLAY`` carries a
    bare hand index with no area field at all, which is why this cannot just
    read ``option["area"]``.
    """
    kind = option.get("type")
    owner = option.get("playerIndex")
    refs = []

    if kind == OptionType.PLAY:
        refs.append(_ref(AreaType.HAND, option.get("index"), seat, seat))
    else:
        refs.append(_ref(option.get("area"), option.get("index"), owner, seat, option.get("serial")))
    refs.append(_ref(option.get("inPlayArea"), option.get("inPlayIndex"), owner, seat))
    return [r for r in refs if r]


RESULT_REASON = {
    1: "all Prize cards taken",
    2: "started a turn with an empty deck",
    3: "no Pokémon left in the Active Spot",
    4: "a card effect",
}


def log_lines(logs: list[dict], seat: int) -> list[str]:
    """Readable one-liners for the event feed.

    The entries come from a *seat-filtered* observation, so anything private to
    the other player has already been replaced by its ``*_REVERSE`` form
    upstream; this only has to name what it is given.
    """
    out = []
    for entry in logs:
        kind = entry.get("type")
        mine = entry.get("playerIndex") == seat
        who = "You" if mine else "Opponent"
        name = card_info(entry["cardId"])["name"] if entry.get("cardId") else ""
        target = card_info(entry["cardIdTarget"])["name"] if entry.get("cardIdTarget") else ""
        if kind == LogType.TURN_START:
            out.append(f"--- {who} turn start ---")
        elif kind == LogType.TURN_END:
            out.append(f"{who} ended the turn")
        elif kind == LogType.DRAW:
            out.append(f"{who} drew {name}" if name else f"{who} drew a card")
        elif kind == LogType.DRAW_REVERSE:
            out.append(f"{who} drew a card")
        elif kind == LogType.SHUFFLE:
            out.append(f"{who} shuffled the deck")
        elif kind == LogType.HAS_BASIC_POKEMON:
            if not entry.get("hasBasicPokemon", True):
                out.append(f"{who} had no Basic Pokémon - mulligan")
        elif kind == LogType.PLAY:
            out.append(f"{who} played {name}")
        elif kind == LogType.MOVE_CARD:
            out.append(
                f"{who}: {name} {AREA_LABEL.get(entry.get('fromArea'), '?')}"
                f" -> {AREA_LABEL.get(entry.get('toArea'), '?')}"
            )
        elif kind == LogType.MOVE_CARD_REVERSE:
            out.append(
                f"{who} moved a face-down card {AREA_LABEL.get(entry.get('fromArea'), '?')}"
                f" -> {AREA_LABEL.get(entry.get('toArea'), '?')}"
            )
        elif kind == LogType.ATTACH:
            out.append(f"{who} attached {name}" + (f" to {target}" if target else ""))
        elif kind == LogType.MOVE_ATTACHED:
            after = card_info(entry["cardIdAfter"])["name"] if entry.get("cardIdAfter") else ""
            out.append(f"{who} moved {name}" + (f" to {after}" if after else ""))
        elif kind == LogType.EVOLVE:
            out.append(f"{who} evolved {target} into {name}" if target else f"{who} evolved into {name}")
        elif kind == LogType.DEVOLVE:
            out.append(f"{who} devolved {target or name}")
        elif kind == LogType.ATTACK:
            atk = attack_db().get(entry.get("attackId"), {})
            out.append(f"{who}: {name or 'Pokémon'} used {atk.get('name', '?')}")
        elif kind == LogType.HP_CHANGE:
            value = entry.get("value", 0)
            if value < 0:
                out.append(f"{name or 'Pokémon'} took {-value} damage")
            elif value > 0:
                out.append(f"{name or 'Pokémon'} healed {value}")
        elif kind == LogType.SWITCH:
            active = card_info(entry["cardIdBench"])["name"] if entry.get("cardIdBench") else "a Pokémon"
            out.append(f"{who} switched {active} into the Active Spot")
        elif kind == LogType.CHANGE:
            after = card_info(entry["cardIdAfter"])["name"] if entry.get("cardIdAfter") else "a Pokémon"
            out.append(f"{who}: Active Pokémon is now {after}")
        elif kind in (LogType.POISONED, LogType.BURNED, LogType.ASLEEP,
                      LogType.PARALYZED, LogType.CONFUSED):
            condition = _enum_name(LogType, kind).lower()
            recovered = entry.get("isRecover")
            out.append(f"{name or 'Pokémon'} " + ("recovered from " + condition if recovered else "is " + condition))
        elif kind == LogType.COIN:
            out.append(f"{who} flipped {'heads' if entry.get('head') else 'tails'}")
        elif kind == LogType.RESULT:
            result = entry.get("result")
            reason = RESULT_REASON.get(entry.get("reason"), "")
            if result == 2:
                verdict = "The match is a draw"
            else:
                verdict = "You win" if result == seat else "You lose"
            out.append(f"=== {verdict}" + (f" ({reason})" if reason else "") + " ===")
    return out


def log_events(logs: list[dict], seat: int) -> list[dict]:
    """The subset of the log worth *showing* on the board, in order.

    The text feed already says what happened; this is what the board should act
    out — a card being played, an attack going off, HP moving. Each event names
    the card by ``serial``, which is what the rendered board carries, so the
    browser can anchor the animation to the right slot without any position
    being sent (positions change between the event and the render anyway).

    These come from the human seat's own observation, so an opponent event is
    here only if that seat was allowed to see it.
    """
    out = []
    for raw_index, entry in enumerate(logs):
        kind = entry.get("type")
        side = "me" if entry.get("playerIndex") == seat else "opp"
        card_id = entry.get("cardId")
        card = card_info(card_id) if card_id else None

        if kind == LogType.HP_CHANGE:
            value = entry.get("value") or 0
            if not value:
                continue
            out.append({
                "kind": "damage" if value < 0 else "heal",
                "side": side, "serial": entry.get("serial"), "value": value,
                "name": (card or {}).get("name"), "counter": bool(entry.get("putDamageCounter")),
            })
        elif kind == LogType.ATTACK:
            attack = attack_db().get(entry.get("attackId"), {})
            out.append({
                "kind": "attack", "side": side, "serial": entry.get("serial"),
                "name": attack.get("name", "Attack"), "value": attack.get("damage") or 0,
                "card": card, "text": attack.get("text", ""),
            })
        elif kind == LogType.PLAY:
            # Trainers and Energy played from hand: the one decision a player
            # makes that leaves no lasting mark on the board.
            out.append({"kind": "play", "side": side, "card": card, "name": (card or {}).get("name")})
        elif kind == LogType.EVOLVE:
            out.append({
                "kind": "evolve", "side": side, "serial": entry.get("serialTarget"),
                "card": card, "name": (card or {}).get("name"),
            })
        elif kind == LogType.ATTACH:
            out.append({
                "kind": "attach", "side": side, "serial": entry.get("serialTarget"),
                "card": card, "name": (card or {}).get("name"),
            })
        elif kind in (LogType.POISONED, LogType.BURNED, LogType.ASLEEP,
                      LogType.PARALYZED, LogType.CONFUSED):
            out.append({
                "kind": "recover" if entry.get("isRecover") else "condition",
                "side": side, "serial": entry.get("serial"),
                "name": _enum_name(LogType, kind),
            })
        elif kind == LogType.COIN:
            out.append({"kind": "coin", "side": side, "name": "Heads" if entry.get("head") else "Tails"})
        elif kind in (LogType.DRAW, LogType.DRAW_REVERSE):
            # A Supporter's whole effect is often "shuffle and draw"; without
            # these the board just changed by itself.
            out.append({"kind": "draw", "side": side, "card": card, "name": (card or {}).get("name")})
        elif kind in (LogType.MOVE_CARD, LogType.MOVE_CARD_REVERSE):
            out.append({
                "kind": "move", "side": side, "card": card, "name": (card or {}).get("name"),
                "from": AREA_LABEL.get(entry.get("fromArea"), "?"),
                "to": AREA_LABEL.get(entry.get("toArea"), "?"),
            })
        elif kind == LogType.HAS_BASIC_POKEMON:
            # A mulligan: no Basic Pokémon in the opening hand, so it goes back
            # and is redrawn. Worth showing — it explains the extra card the
            # other player draws for it.
            if not entry.get("hasBasicPokemon", True):
                out.append({"kind": "mulligan", "side": side, "name": "Mulligan"})
        elif kind == LogType.SHUFFLE:
            out.append({"kind": "shuffle", "side": side, "name": "Shuffled the deck"})
        elif kind == LogType.SWITCH:
            active = card_info(entry["cardIdBench"]) if entry.get("cardIdBench") else None
            out.append({
                "kind": "switch", "side": side, "card": active,
                "name": (active or {}).get("name"), "serial": entry.get("serialBench"),
            })
        elif kind == LogType.RESULT:
            # The end of the match is an event like any other, so it lands as
            # the last beat of the replay rather than as a verdict announced
            # over a turn that is still being played out.
            result = entry.get("result")
            out.append({
                "kind": "result",
                "side": "me" if result == seat else "opp",
                "name": "Draw" if result == 2 else ("You win" if result == seat else "You lose"),
                "text": RESULT_REASON.get(entry.get("reason"), ""),
            })
        elif kind == LogType.MOVE_ATTACHED:
            out.append({
                "kind": "attach", "side": side, "serial": entry.get("serialAfter"),
                "card": card, "name": (card or {}).get("name"),
            })
        if out and "rawIndex" not in out[-1]:
            out[-1]["rawIndex"] = raw_index
    return out


def board_snapshot(obs: dict, seat: int) -> dict:
    """The board as the human may see it, taken from *any* observation.

    Used to capture the position after each of the agent's decisions so the
    replay can move the board one action at a time instead of jumping to the
    end. It is built from the agent's own observation, so it deliberately drops
    every hand: the agent's hand is private, and the human's is not in that
    observation at all (the engine erased it) — the browser keeps its own hand
    from the last view it was given.
    """
    state = obs["current"]
    me = player_view(state["players"][seat], reveal_hand=False)
    opp = player_view(state["players"][1 - seat], reveal_hand=False)
    return {
        "me": me,
        "opp": opp,
        "stadium": card_info(state["stadium"][0]["id"]) if state["stadium"] else None,
        "looking": [card_info(c["id"] if c else None) for c in (state.get("looking") or [])],
        "turn": state["turn"],
    }


def build_view(obs: dict, seat: int, agent_name: str, pending_logs: list[dict],
               your_turn: bool, result: int) -> dict:
    """The full payload the browser renders.

    ``obs`` is always an observation taken from ``seat`` -- the human's -- even
    while the agent is choosing, so the opponent's hand and prizes are absent
    from the payload by construction rather than by anything hidden here. When
    the agent holds the decision, ``your_turn`` is False and the board is simply
    the last position the human legitimately saw.
    """
    state = obs["current"]
    select = obs.get("select")
    your_turn = your_turn and select is not None and state["yourIndex"] == seat

    options = []
    if your_turn:
        for i, opt in enumerate(select["option"]):
            options.append({
                "index": i,
                "label": option_label(opt, state, seat, select),
                "refs": option_refs(opt, seat),
                "type": _enum_name(OptionType, opt.get("type")),
            })

    context = select and _enum_name(SelectContext, select["context"])
    prompt = PROMPT.get(select and select["context"], context or "")

    return {
        "turn": state["turn"],
        "actions": state["turnActionCount"],
        "firstPlayer": state["firstPlayer"],
        # What this turn has already spent: the four once-per-turn rights the
        # rules give you, which the board itself cannot show.
        "flags": {
            "supporter": state["supporterPlayed"],
            "stadium": state["stadiumPlayed"],
            "energy": state["energyAttached"],
            "retreat": state["retreated"],
        },
        "result": result,
        "finished": result != -1,
        "seat": seat,
        "yourTurn": your_turn,
        "waitingOn": "you" if your_turn else agent_name,
        "me": player_view(state["players"][seat], reveal_hand=True),
        "opp": player_view(state["players"][1 - seat], reveal_hand=False),
        "stadium": card_info(state["stadium"][0]["id"]) if state["stadium"] else None,
        "looking": [card_info(c["id"] if c else None) for c in (state.get("looking") or [])],
        "select": None if not your_turn else {
            "type": _enum_name(SelectType, select["type"]),
            "context": context,
            "prompt": prompt,
            "minCount": select["minCount"],
            "maxCount": select["maxCount"],
            "contextCard": card_info(select["contextCard"]["id"]) if select.get("contextCard") else None,
            "effect": card_info(select["effect"]["id"]) if select.get("effect") else None,
        },
        "options": options,
        "log": log_lines(pending_logs, seat),
    }
