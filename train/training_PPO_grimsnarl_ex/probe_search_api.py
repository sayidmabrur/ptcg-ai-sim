"""Measure what ``cg``'s search API can actually do, before any search-based
training is designed around it.

    PY=/home/kangh/miniconda3/envs/kaggle-pokemon/bin/python
    $PY probe_search_api.py

``cg/api.py`` exposes ``search_begin``/``search_step``/``search_release``/
``search_end`` — a determinization interface: you hand it predicted card IDs
for every hidden zone (your deck, your prizes, their deck, their prizes, their
hand, their face-down active) and it instantiates a complete-information world
consistent with the public state. ``cg/game.py`` already stashes the blob it
needs (``obs["search_begin_input"]``) on every observation, and nothing in
``train.py`` reads it.

An IS-MCTS or particle-CFR loop on top of that is a large build, and its shape
depends on facts nobody here has measured. This script measures them, one
probe per open question, and prints a report. Every probe is independently
guarded: a failure is a *result* (it tells us the primitive is unavailable),
not a reason to lose the other twelve answers.

The questions, and why each one changes the design:

P1  search_begin works from a live obs   can we fork a search at all
P2  search_step latency + parse share    sets the per-decision node budget
P3  do stale searchIds stay valid        tree re-descent, or rollouts only
P4  superset / bad-id tolerance          how particles must be constructed
P5  manual_coin surfaces COIN_HEAD       is chance enumerable or sampled
P6  cost of a full playout to terminal   MCTS budget vs value-net leaf eval
P7  memory per retained node             tree size ceiling, release policy
P8  deck-order convention                which end of your_deck is drawn first
P9  consistency validation               must our sampler enforce legality
P10 live battle survives searching       one process, or a separate engine
"""

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import ctypes  # noqa: E402

from cg import api  # noqa: E402
from cg.api import (  # noqa: E402
    AreaType,
    LogType,
    OptionType,
    SelectContext,
    search_begin,
    search_end,
    search_release,
    search_step,
    to_observation_class,
)
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from cg.sim import lib  # noqa: E402

CHALLENGER_DECK = _ROOT / "challenger" / "challenger_deck.csv"
OPPONENT_DECK = _ROOT / "crustle_frozen" / "deck.csv"

#: Built by scanning ``all_attack()`` for coin text: 11 Basic Pokémon with
#: coin-flip attacks (led by Tyrogue #672, whose attack costs no energy and
#: flips until tails) plus 16 basic energy. The real decks flip rarely enough
#: that P5 came back "not observed", which is not the same answer as "no" —
#: this deck exists purely to make chance events frequent enough to probe.
COIN_DECK = _ROOT / "coin_probe_deck.csv"

WARMUP_DECISION_CAP = 4000


# --------------------------------------------------------------------------
# Small helpers


def read_deck(path: Path) -> list[int]:
    lines = [line for line in path.read_text().split("\n") if line.strip()]
    if len(lines) != 60:
        raise SystemExit(f"{path} has {len(lines)} cards; the engine requires exactly 60")
    return [int(line) for line in lines]


def vmrss_kb() -> int:
    """Resident set size in KB. Peak (``ru_maxrss``) is useless here — it never
    falls, so it cannot show whether ``search_release`` gives memory back."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return -1


def random_select(select) -> list[int]:
    """A legal selection for a ``SelectData``, as ``search_step`` wants it."""
    count = min(random.randint(select.minCount, select.maxCount), len(select.option))
    return random.sample(range(len(select.option)), count)


def random_select_dict(select: dict) -> list[int]:
    """Same, for the raw-dict observations the live battle returns."""
    count = min(
        random.randint(select["minCount"], select["maxCount"]), len(select["option"])
    )
    return random.sample(range(len(select["option"])), count)


def end_turn_index(select) -> int | None:
    """Index of the END option, if this decision offers one."""
    for index, option in enumerate(select.option):
        if option.type == OptionType.END:
            return index
    return None


def option_index(select, option_type) -> int | None:
    for index, option in enumerate(select.option):
        if option.type == option_type:
            return index
    return None


def attack_seeking_select(select) -> list[int]:
    """Prefer ATTACK, so coin-flip riders actually fire.

    Uniform random play reaches an attack rarely enough that the coin probes
    time out before observing one — which is exactly how P5 came back
    inconclusive on the real decks. Biasing toward attacks is not a fair
    sample of play and is not meant to be; these probes ask what the *engine*
    does when a coin comes up, not how often one does.
    """
    if select.minCount <= 1 <= select.maxCount:
        index = option_index(select, OptionType.ATTACK)
        if index is not None:
            return [index]
    return random_select(select)


def state_fingerprint(state) -> tuple:
    """Everything about a search state that a re-step could plausibly change.

    Deliberately excludes ``searchId`` (always fresh) and includes the log
    stream, since that is where coin results and drawn card ids surface — the
    two things that would differ if the engine re-randomises per step rather
    than resolving chance once at ``search_begin``.
    """
    observation = state.observation
    current = observation.current
    return (
        current.result, current.turn, current.turnActionCount,
        tuple(
            (p.deckCount, p.handCount, len(p.discard), len(p.prize))
            for p in current.players
        ),
        tuple(
            (log.type, log.head, log.cardId, log.serial, log.value)
            for log in (observation.logs or [])
        ),
        len(observation.select.option) if observation.select else -1,
    )


# --------------------------------------------------------------------------
# Belief construction
#
# This is a miniature of the particle sampler a real search would need, and it
# is here rather than in a helper module because getting it wrong is one of the
# things the probe is checking: if ``search_begin`` accepts an inconsistent
# prediction (P9), the sampler — not the engine — is what has to enforce
# legality.


def _card_ids(node, into: Counter) -> None:
    """Walk a card/Pokémon struct or list, counting every ``id`` seen.

    Recursive because a Pokémon carries ``energyCards``/``tools``/
    ``preEvolution``, each a list of cards that came out of the same 60.
    Face-down entries are ``None`` (every own prize, an unrevealed active) and
    contribute nothing, which is exactly right — they are the hidden part.
    """
    if node is None:
        return
    if isinstance(node, list):
        for item in node:
            _card_ids(item, into)
        return
    if isinstance(node, dict):
        if "id" in node and "serial" in node:
            into[int(node["id"])] += 1
        for key in ("energyCards", "tools", "preEvolution"):
            if key in node:
                _card_ids(node[key], into)


def visible_cards(state: dict, player_index: int) -> Counter:
    """Everything of ``player_index``'s 60 whose identity is public-or-own.

    Deliberately *not* the whole board: ``deckCount`` and face-down prizes are
    counts, and the opponent's hand is ``None``. What is left over after
    subtracting this from the decklist is precisely the pool a belief is a
    distribution over.
    """
    player = state["players"][player_index]
    seen: Counter = Counter()
    for zone in ("hand", "discard", "prize", "active", "bench"):
        _card_ids(player.get(zone), seen)
    # A stadium and any looked-at cards sit outside both player structs but
    # still came out of somebody's deck; ``playerIndex`` says whose.
    for zone in ("stadium", "looking"):
        for card in state.get(zone) or []:
            if isinstance(card, dict) and card.get("playerIndex") == player_index:
                _card_ids(card, seen)
    return seen


def hidden_pool(state: dict, player_index: int, decklist: list[int]) -> list[int]:
    """The multiset of ``player_index``'s cards whose location is unknown."""
    remaining = Counter(decklist) - visible_cards(state, player_index)
    return [card for card, count in remaining.items() for _ in range(count)]


def build_belief(state: dict, seat: int, own_deck: list[int], opp_deck: list[int]) -> dict:
    """A single particle: one concrete assignment of every hidden zone.

    Own side is bookkeeping — the pool is exactly deck + prizes, since our hand
    is visible to us. The opponent's pool is deck + prizes + hand, and the split
    between the three is the part a real belief model would have an opinion
    about. Here it is an arbitrary partition, which is enough to exercise the
    API and is the K=1 degenerate case of the sampler a search would use.

    ``consistent`` reports whether the pool sizes actually add up. If they do
    not, some card left the 60 by a route ``visible_cards`` does not model (a
    lost zone, a card shuffled somewhere unusual) and the caller should say so
    rather than quietly feed the engine a malformed particle.
    """
    own, opponent = state["players"][seat], state["players"][1 - seat]
    own_pool = hidden_pool(state, seat, own_deck)
    opp_pool = hidden_pool(state, 1 - seat, opp_deck)
    random.shuffle(own_pool)
    random.shuffle(opp_pool)

    own_need = own["deckCount"] + len(own["prize"])
    opp_need = opponent["deckCount"] + len(opponent["prize"]) + opponent["handCount"]

    cut = own["deckCount"]
    opp_cut_deck = opponent["deckCount"]
    opp_cut_prize = opp_cut_deck + len(opponent["prize"])

    # A face-down active only happens before it is revealed; the API demands a
    # prediction for it and rejects the call otherwise.
    opponent_active = []
    active = opponent.get("active") or []
    if len(active) > 0 and active[0] is None and opp_pool:
        opponent_active = [opp_pool[0]]

    return {
        "your_deck": own_pool[:cut],
        "your_prize": own_pool[cut:own_need],
        "opponent_deck": opp_pool[:opp_cut_deck],
        "opponent_prize": opp_pool[opp_cut_deck:opp_cut_prize],
        "opponent_hand": opp_pool[opp_cut_prize:opp_need],
        "opponent_active": opponent_active,
        "consistent": len(own_pool) == own_need and len(opp_pool) == opp_need,
        "own_pool_size": len(own_pool),
        "own_need": own_need,
        "opp_pool_size": len(opp_pool),
        "opp_need": opp_need,
    }


def begin(obs: dict, belief: dict, manual_coin: bool = False):
    """``search_begin`` from a raw live observation and a particle."""
    return search_begin(
        to_observation_class(obs),
        belief["your_deck"], belief["your_prize"],
        belief["opponent_deck"], belief["opponent_prize"],
        belief["opponent_hand"], belief["opponent_active"],
        manual_coin=manual_coin,
    )


# --------------------------------------------------------------------------
# Probes


def p1_begin(obs, belief) -> dict:
    """Can we fork a search at all, and what does it cost?"""
    timings = []
    root = None
    for _ in range(20):
        started = time.perf_counter()
        root = begin(obs, belief)
        timings.append((time.perf_counter() - started) * 1000)
    live = obs["current"]
    return {
        "ok": root is not None,
        "search_id": root.searchId,
        "ms_p50": statistics.median(timings),
        "ms_p95": sorted(timings)[int(0.95 * len(timings)) - 1],
        # The forked root must present the same decision as the live frame, or
        # a search rooted here is answering a different question.
        "same_turn": root.observation.current.turn == live["turn"],
        "same_option_count": len(root.observation.select.option) == len(obs["select"]["option"]),
        "sbi_blob_bytes": len(obs["search_begin_input"]),
    }


def p2_step_latency(obs, belief, steps: int = 300) -> dict:
    """Per-step cost, split into engine time and JSON/dataclass time.

    The split matters: engine time is a floor, but if most of the cost is
    ``json.loads`` plus ``to_dataclass`` then a leaner decode (reading only the
    fields a search needs) buys back most of the budget without touching the
    ``.so``.
    """
    root = begin(obs, belief)
    agent_ptr = api.agent_ptr
    state = root
    raw_ms, parse_ms, total_ms = [], [], []
    depth = 0
    for _ in range(steps):
        select = state.observation.select
        if select is None or state.observation.current.result != -1:
            break
        choice = random_select(select)
        arg = (ctypes.c_int * len(choice))(*choice)

        started = time.perf_counter()
        payload = lib.SearchStep(agent_ptr, state.searchId, arg, len(choice))
        mid = time.perf_counter()
        parsed = json.loads(payload.decode())
        done = time.perf_counter()

        raw_ms.append((mid - started) * 1000)
        parse_ms.append((done - mid) * 1000)
        total_ms.append((done - started) * 1000)
        if parsed.get("error", 0) != 0:
            break
        # Re-wrap through the public path so the walk stays on supported ground.
        state = search_step(state.searchId, choice)
        depth += 1
    if not total_ms:
        return {"ok": False, "reason": "no steps taken"}
    return {
        "ok": True,
        "steps": len(total_ms),
        "depth_reached": depth,
        "engine_ms_p50": statistics.median(raw_ms),
        "json_ms_p50": statistics.median(parse_ms),
        "total_ms_p50": statistics.median(total_ms),
        "total_ms_p95": sorted(total_ms)[int(0.95 * len(total_ms)) - 1],
        "json_share": sum(parse_ms) / sum(total_ms),
        "steps_per_second": 1000.0 / statistics.median(total_ms),
    }


def p3_branching(obs, belief) -> dict:
    """Do stale searchIds stay steppable?

    This is the single most consequential answer in the file. If stepping
    consumes its parent, the API supports rollouts but not a tree: MCTS would
    have to re-derive every node from the root, turning an N-node search into
    O(N * depth) steps. If parents persist, IS-MCTS descends normally.
    """
    root = begin(obs, belief)
    select = root.observation.select
    if select is None or len(select.option) < 2 or select.minCount > 1:
        return {"ok": False, "reason": "root decision has no two single-option branches"}

    first = search_step(root.searchId, [0])
    result = {"root_id": root.searchId, "child_a": first.searchId}

    # Same root, different action: is the root still alive after being stepped?
    try:
        second = search_step(root.searchId, [1])
        result["root_reusable"] = True
        result["child_b"] = second.searchId
        result["children_distinct"] = second.searchId != first.searchId
    except ValueError as error:
        result["root_reusable"] = False
        result["root_reuse_error"] = str(error)

    # And is a grandchild's parent still alive after the grandchild exists?
    try:
        if first.observation.select is not None:
            search_step(first.searchId, random_select(first.observation.select))
            search_step(first.searchId, random_select(first.observation.select))
            result["child_reusable"] = True
    except ValueError as error:
        result["child_reusable"] = False
        result["child_reuse_error"] = str(error)

    # Does release actually invalidate, i.e. is it safe to prune a subtree?
    try:
        search_release(first.searchId)
        search_step(first.searchId, [0])
        result["release_invalidates"] = False
    except ValueError:
        result["release_invalidates"] = True
    return result


def p4_tolerance(obs, belief) -> dict:
    """What does ``search_begin`` accept?

    The Python guards in ``cg/api.py`` test ``len(...) < required``, so a
    *longer* list is not rejected there. Whether the engine takes a prefix, the
    whole thing, or misbehaves decides how particles get built — a sampler that
    can pass a superset does not have to compute an exact split.
    """
    out = {}

    padded = dict(belief)
    filler = belief["your_deck"][:1] or belief["opponent_deck"][:1]
    padded["your_deck"] = belief["your_deck"] + filler * 5
    padded["opponent_deck"] = belief["opponent_deck"] + filler * 5
    try:
        state = begin(obs, padded)
        out["superset_accepted"] = state is not None
    except Exception as error:  # noqa: BLE001 — the error type is the finding
        out["superset_accepted"] = False
        out["superset_error"] = f"{type(error).__name__}: {error}"

    short = dict(belief)
    short["opponent_deck"] = belief["opponent_deck"][:-1]
    try:
        begin(obs, short)
        out["undersized_accepted"] = True
    except Exception as error:  # noqa: BLE001
        out["undersized_accepted"] = False
        out["undersized_error"] = f"{type(error).__name__}: {error}"

    bogus = dict(belief)
    bogus["opponent_deck"] = [999_999] * len(belief["opponent_deck"])
    try:
        begin(obs, bogus)
        out["bogus_card_id_accepted"] = True
    except Exception as error:  # noqa: BLE001
        out["bogus_card_id_accepted"] = False
        out["bogus_card_id_error"] = f"{type(error).__name__}: {error}"
    return out


def p5_manual_coin(obs, belief, steps: int = 400) -> dict:
    """Does ``manual_coin=True`` turn chance into a choice?

    ``SelectContext.COIN_HEAD`` exists in the enum, so the intent is clear. If
    it really surfaces, coin flips can be enumerated inside the search instead
    of sampled, which removes a large chunk of leaf-value variance for free.
    """
    found = {"manual": False, "default": False}
    for label, manual in (("manual", True), ("default", False)):
        try:
            state = begin(obs, belief, manual_coin=manual)
        except Exception as error:  # noqa: BLE001
            return {"ok": False, "reason": f"{type(error).__name__}: {error}"}
        for _ in range(steps):
            select = state.observation.select
            if select is None or state.observation.current.result != -1:
                break
            if select.context == SelectContext.COIN_HEAD:
                found[label] = True
                break
            try:
                state = search_step(state.searchId, random_select(select))
            except ValueError:
                break
    return {
        "ok": True,
        "coin_choice_with_manual": found["manual"],
        "coin_choice_by_default": found["default"],
    }


def p6_playout(obs, belief, trials: int = 8, cap: int = 3000) -> dict:
    """Cost of random-playing a search to terminal.

    Sets the trade: if a full playout is cheap, plain MCTS rollouts are an
    option; if it is not, every leaf has to be evaluated by the value network,
    which is the ReBeL/AlphaZero shape anyway.
    """
    lengths, seconds, results = [], [], []
    for _ in range(trials):
        state = begin(obs, belief)
        started = time.perf_counter()
        steps = 0
        while steps < cap:
            select = state.observation.select
            if select is None or state.observation.current.result != -1:
                break
            try:
                state = search_step(state.searchId, random_select(select))
            except ValueError:
                break
            steps += 1
        seconds.append(time.perf_counter() - started)
        lengths.append(steps)
        results.append(state.observation.current.result)
    reached = [r for r in results if r != -1]
    return {
        "trials": trials,
        "mean_steps": statistics.mean(lengths),
        "mean_seconds": statistics.mean(seconds),
        "reached_terminal": len(reached),
        "results": results,
        "playouts_per_second": len(seconds) / sum(seconds) if sum(seconds) else 0.0,
    }


def p7_memory(obs, belief, nodes: int = 2000) -> dict:
    """Bytes per retained node, and whether releasing gives them back."""
    before = vmrss_kb()
    root = begin(obs, belief)
    retained = [root.searchId]
    state = root
    for _ in range(nodes):
        select = state.observation.select
        if select is None or state.observation.current.result != -1:
            state = begin(obs, belief)  # restart rather than stop counting
            retained.append(state.searchId)
            continue
        try:
            state = search_step(state.searchId, random_select(select))
        except ValueError:
            break
        retained.append(state.searchId)
    held = vmrss_kb()
    for search_id in retained:
        try:
            search_release(search_id)
        except Exception:  # noqa: BLE001 — release of a dead id is not the finding here
            pass
    after_release = vmrss_kb()
    search_end()
    after_end = vmrss_kb()
    return {
        "nodes_retained": len(retained),
        "rss_before_kb": before,
        "rss_held_kb": held,
        "kb_per_node": (held - before) / max(1, len(retained)),
        "rss_after_release_kb": after_release,
        "rss_after_end_kb": after_end,
        "release_reclaims": after_release < held,
        "end_reclaims": after_end < held,
    }


def p8_deck_order(obs, belief, seat: int, steps: int = 600) -> dict:
    """Which end of ``your_deck`` is drawn first?

    A particle sampler has to write the deck in the order the engine will deal
    it, and nothing documents that. Rolling forward to our next draw and reading
    the ``DRAW`` log's ``cardId`` settles it — the log carries the drawn card's
    id, so this is a direct observation rather than an inference from a hand
    diff.

    Void if the root decision is a deck search: ``search_begin`` throws
    ``your_deck`` away in that case (it uses the real one) and there is nothing
    to compare against.
    """
    if (obs.get("select") or {}).get("deck") is not None:
        return {"ok": False, "reason": "root is a deck-search decision; your_deck ignored"}
    predicted = belief["your_deck"]
    if len(predicted) < 2:
        return {"ok": False, "reason": "deck too small to distinguish ends"}

    state = begin(obs, belief)
    for _ in range(steps):
        for log in state.observation.logs or []:
            if log.type == LogType.DRAW and log.playerIndex == seat and log.cardId:
                return {
                    "ok": True,
                    "drawn_card_id": log.cardId,
                    "predicted_first": predicted[0],
                    "predicted_last": predicted[-1],
                    "matches_first": log.cardId == predicted[0],
                    "matches_last": log.cardId == predicted[-1],
                    "in_prediction": log.cardId in predicted,
                }
        select = state.observation.select
        if select is None or state.observation.current.result != -1:
            break
        # Pass the turn where possible so a draw arrives in few steps.
        end = end_turn_index(select)
        choice = [end] if end is not None and select.maxCount >= 1 else random_select(select)
        try:
            state = search_step(state.searchId, choice)
        except ValueError:
            break
    return {"ok": False, "reason": f"no own DRAW log within {steps} steps"}


def p9_consistency(obs, belief, seat: int) -> dict:
    """Does the engine reject a belief that contradicts public information?

    The test claims the opponent's whole hand is a card already sitting in their
    discard pile — legal card ids, impossible placement. Acceptance means
    legality is the sampler's job, and a naive particle can hand the search a
    world the real game could never be in.
    """
    opponent = obs["current"]["players"][1 - seat]
    discarded = [int(card["id"]) for card in (opponent.get("discard") or [])]
    if not discarded:
        return {"ok": False, "reason": "opponent discard empty; nothing provably impossible yet"}
    hand_size = opponent["handCount"]
    if hand_size == 0:
        return {"ok": False, "reason": "opponent hand empty"}

    impossible = dict(belief)
    impossible["opponent_hand"] = [discarded[0]] * hand_size
    try:
        state = begin(obs, impossible)
        return {
            "ok": True,
            "impossible_belief_accepted": state is not None,
            "note": "engine does not validate; the particle sampler must",
            "claimed_card_id": discarded[0],
            "copies_claimed": hand_size,
        }
    except Exception as error:  # noqa: BLE001
        return {
            "ok": True,
            "impossible_belief_accepted": False,
            "error": f"{type(error).__name__}: {error}",
        }


def p10_live_battle_survives(obs, belief) -> dict:
    """Can the live battle continue after searches ran in the same process?

    ``agent_ptr`` and ``battle_ptr`` are separate globals, which suggests yes —
    but if searching disturbs the live battle, every rollout worker needs a
    second process just to hold the search context, and the worker layout in
    ``train.py`` changes completely.
    """
    begin(obs, belief)
    state = begin(obs, belief)
    if state.observation.select is not None:
        search_step(state.searchId, random_select(state.observation.select))
    try:
        select = obs["select"]
        next_obs = battle_select(random_select_dict(select))
        return {
            "ok": True,
            "live_step_after_search": True,
            "turn_advanced_or_held": next_obs["current"]["turn"] >= obs["current"]["turn"],
            "result": next_obs["current"]["result"],
        }
    except Exception as error:  # noqa: BLE001
        return {"ok": True, "live_step_after_search": False,
                "error": f"{type(error).__name__}: {error}"}


# --------------------------------------------------------------------------


def build_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--turn", type=int, default=4,
                        help="play the live battle to this turn before forking "
                             "searches. Two probes are only decidable deeper in: "
                             "P9 needs a non-empty opponent discard, P12 needs a "
                             "coin to have come up")
    parser.add_argument("--deck", default=str(CHALLENGER_DECK))
    parser.add_argument("--opponent-deck", default=str(OPPONENT_DECK))
    parser.add_argument("--coin-deck", action="store_true",
                        help=f"run the whole battery on {COIN_DECK.name} for both "
                             f"seats, not just the coin probes — a finding that "
                             f"only holds for one decklist is not a finding")
    parser.add_argument("--out", default=None, help="JSON report path")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def warmup(challenger_deck, opponent_deck, warmup_turn):
    """Play random legal moves until a representative mid-game decision.

    Random rather than the trained policy on purpose: the probe measures the
    engine, and pulling in torch, the four networks and the feature pipeline
    would make a failure here ambiguous between "the search API is broken" and
    "something upstream of it is".
    """
    obs, _ = battle_start(challenger_deck, opponent_deck)
    if obs is None:
        raise SystemExit("battle_start failed")
    decisions = 0
    while decisions < WARMUP_DECISION_CAP:
        if obs["current"]["result"] != -1:
            raise SystemExit("battle ended during warmup; rerun (random play is short sometimes)")
        select = obs.get("select")
        if select is None:
            raise SystemExit("no select during warmup")
        if obs["current"]["turn"] >= warmup_turn and obs["search_begin_input"]:
            return obs, decisions
        obs = battle_select(random_select_dict(select))
        decisions += 1
    raise SystemExit(f"never reached turn {warmup_turn} in {WARMUP_DECISION_CAP} decisions")


def _coin_control_test(state) -> dict:
    """At a ``COIN_HEAD`` decision, does the choice decide the outcome?

    The distinction that matters: a selection that merely *records* a call
    while the engine flips anyway leaves chance stochastic, and the search has
    to average over it. A selection that *determines* the face turns every flip
    into a branch the search can enumerate, which removes the single largest
    source of leaf-value variance in the game.

    Branching both ways from one parent relies on P3's finding that a stepped
    node stays steppable.
    """
    select = state.observation.select
    yes = option_index(select, OptionType.YES)
    no = option_index(select, OptionType.NO)
    if yes is None or no is None:
        return {"ok": False, "reason": f"no YES/NO options (types "
                                       f"{[o.type for o in select.option]})"}

    def outcome(choice: int):
        nxt = search_step(state.searchId, [choice])
        for log in nxt.observation.logs or []:
            if log.type == LogType.COIN:
                return log.head
        return None

    heads_call, tails_call = outcome(yes), outcome(no)
    return {
        "ok": True,
        "choosing_yes_gives_head": heads_call,
        "choosing_no_gives_head": tails_call,
        # Determines only if the two calls actually produce different faces.
        "choice_determines_face": (
            heads_call is not None and tails_call is not None
            and heads_call != tails_call
        ),
    }


def p12_coin(obs, belief, steps: int = 2000) -> dict:
    """Coin mechanics, on a deck built to flip constantly.

    Re-asks P5 with attack-seeking play, and goes past it: how many flips a
    game actually contains, whether ``manual_coin`` surfaces the call as a
    decision, and whether that decision controls the face.
    """
    out: dict = {}
    for label, manual in (("manual_coin_true", True), ("manual_coin_false", False)):
        try:
            state = begin(obs, belief, manual_coin=manual)
        except Exception as error:  # noqa: BLE001
            out[label] = {"FAILED": f"{type(error).__name__}: {error}"}
            continue
        flips = heads = coin_selects = attacks = 0
        control: dict | None = None
        for _ in range(steps):
            observation = state.observation
            for log in observation.logs or []:
                if log.type == LogType.COIN:
                    flips += 1
                    heads += bool(log.head)
                elif log.type == LogType.ATTACK:
                    attacks += 1
            select = observation.select
            if select is None or observation.current.result != -1:
                break
            if select.context == SelectContext.COIN_HEAD:
                coin_selects += 1
                if control is None:
                    try:
                        control = _coin_control_test(state)
                    except Exception as error:  # noqa: BLE001
                        control = {"ok": False, "reason": f"{type(error).__name__}: {error}"}
            try:
                state = search_step(state.searchId, attack_seeking_select(select))
            except ValueError:
                break
        out[label] = {
            "coin_logs": flips,
            "heads_fraction": (heads / flips) if flips else None,
            "attacks_seen": attacks,
            "coin_head_selections": coin_selects,
            "control": control,
        }
    return out


def p13_chance_determinism(
    obs, belief, trials: int = 60, steps: int = 800, manual_coin: bool = False
) -> dict:
    """Is ``search_step`` a function of (state, action), or does it re-randomise?

    This is the question that decides whether the search tree has chance nodes.
    If stepping the same node with the same action twice always lands in the
    same place, the world was fully determinised by ``search_begin`` and the
    particle *is* the randomness — plain IS-MCTS over K particles is then
    correct and no chance handling is needed. If it re-randomises, every
    stochastic edge needs averaging or explicit chance nodes, and a node's
    value estimate is over a distribution rather than a state.

    Run at both ``manual_coin`` settings on purpose. P12 shows the coin call
    becomes a *choice* under ``manual_coin=True``; if that is the only residual
    randomness, determinism should jump to 1.0 and the search gets to enumerate
    chance instead of averaging over it. Anything short of 1.0 there is a
    second source — a shuffle the deck prediction does not pin — and that one
    cannot be turned into a branch.

    Reported separately for frames that contained a coin, since those are the
    ones where a difference would show up first.
    """
    matches = total = coin_matches = coin_total = draw_total = draw_matches = 0
    state = begin(obs, belief, manual_coin=manual_coin)
    for _ in range(steps):
        select = state.observation.select
        if select is None or state.observation.current.result != -1:
            break
        choice = attack_seeking_select(select)
        try:
            first = search_step(state.searchId, choice)
            second = search_step(state.searchId, choice)
        except ValueError:
            break
        same = state_fingerprint(first) == state_fingerprint(second)
        total += 1
        matches += same
        logs = first.observation.logs or []
        if any(log.type == LogType.COIN for log in logs):
            coin_total += 1
            coin_matches += same
        if any(log.type in (LogType.DRAW, LogType.DRAW_REVERSE) for log in logs):
            draw_total += 1
            draw_matches += same
        state = first
        if total >= trials:
            break

    # Separately: does an identical belief replay identically from scratch? If
    # so the particle fully pins the world, which is what lets a search cache
    # results across simulations that share a prefix.
    replay_same = None
    try:
        actions = [[0]] * 12
        prints = []
        for _ in range(2):
            walk = begin(obs, belief, manual_coin=manual_coin)
            trail = []
            for action in actions:
                if walk.observation.select is None or walk.observation.current.result != -1:
                    break
                legal = action if action[0] < len(walk.observation.select.option) else [0]
                walk = search_step(walk.searchId, legal)
                trail.append(state_fingerprint(walk))
            prints.append(tuple(trail))
        replay_same = prints[0] == prints[1]
    except Exception:  # noqa: BLE001
        replay_same = None

    return {
        "manual_coin": manual_coin,
        "frames_tested": total,
        "identical_after_respam": matches,
        "deterministic_fraction": (matches / total) if total else None,
        "coin_frames": coin_total,
        "coin_frames_identical": coin_matches,
        "draw_frames": draw_total,
        "draw_frames_identical": draw_matches,
        "same_belief_replays_identically": replay_same,
    }


def p11_memory_plateau(obs, belief, rounds: int = 12, per_round: int = 400) -> dict:
    """Does RSS plateau across repeated searches, or climb without bound?

    P7 shows neither ``search_release`` nor ``search_end`` returns memory to
    the OS, which ``search_end``'s own docstring predicts ("memory used during
    the search will be reused in the next search") — an internal free-list, not
    a leak, if and only if the high-water mark stops rising. The distinction
    decides whether six long-lived rollout workers each holding a search
    context is fine or whether workers have to be recycled.
    """
    samples = []
    for _ in range(rounds):
        state = begin(obs, belief)
        for _ in range(per_round):
            select = state.observation.select
            if select is None or state.observation.current.result != -1:
                state = begin(obs, belief)
                continue
            try:
                state = search_step(state.searchId, random_select(select))
            except ValueError:
                break
        search_end()
        samples.append(vmrss_kb())
    first_half = statistics.mean(samples[: rounds // 2])
    second_half = statistics.mean(samples[rounds // 2:])
    return {
        "rounds": rounds,
        "nodes_per_round": per_round,
        "rss_kb_samples": samples,
        "rss_kb_first": samples[0],
        "rss_kb_last": samples[-1],
        "growth_kb_per_round_late": (samples[-1] - samples[rounds // 2]) / max(1, rounds - rounds // 2),
        # A free-list plateaus: the back half stops climbing even though it did
        # the same work. A leak keeps the two halves separated.
        "plateaued": (second_half - first_half) < 0.05 * max(1.0, first_half),
    }


PROBES = (
    ("P1  search_begin works + cost", "p1"),
    ("P2  search_step latency", "p2"),
    ("P3  searchId persistence / branching", "p3"),
    ("P4  input tolerance", "p4"),
    ("P5  manual_coin", "p5"),
    ("P6  playout to terminal", "p6"),
    ("P7  memory per node", "p7"),
    ("P8  deck order convention", "p8"),
    ("P9  belief consistency validation", "p9"),
    ("P10 live battle survives search", "p10"),
    ("P11 memory plateau across searches", "p11"),
    ("P12 coin mechanics (deep)", "p12"),
    ("P13 chance determinism", "p13"),
)


def main() -> None:
    args = build_args()
    random.seed(args.seed)
    if args.coin_deck:
        challenger_deck = opponent_deck = read_deck(COIN_DECK)
        label = f"{COIN_DECK.name} (both seats)"
    else:
        challenger_deck = read_deck(Path(args.deck))
        opponent_deck = read_deck(Path(args.opponent_deck))
        label = f"{Path(args.deck).name} vs {Path(args.opponent_deck).name}"

    obs, decisions = warmup(challenger_deck, opponent_deck, args.turn)
    seat = obs["current"]["yourIndex"]
    decks = [None, None]
    decks[0], decks[1] = challenger_deck, opponent_deck
    belief = build_belief(obs["current"], seat, decks[seat], decks[1 - seat])

    print("=" * 74)
    print("cg search API probe")
    print("=" * 74)
    print(f"decks             : {label}")
    print(f"warmup            : {decisions} decisions, turn {obs['current']['turn']}, seat {seat}")
    print(f"options at root   : {len(obs['select']['option'])} "
          f"(min {obs['select']['minCount']}, max {obs['select']['maxCount']}, "
          f"context {obs['select']['context']})")
    print(f"belief consistent : {belief['consistent']} "
          f"(own {belief['own_pool_size']}/{belief['own_need']}, "
          f"opp {belief['opp_pool_size']}/{belief['opp_need']})")
    print(f"sbi blob          : {len(obs['search_begin_input'])} bytes")
    print()
    if not belief["consistent"]:
        print("!! pool sizes do not add up — a card left the 60 by a route")
        print("   visible_cards() does not model. Probes still run, but the")
        print("   particle handed to the engine is malformed.")
        print()

    results = {}
    runners = {
        "p1": lambda: p1_begin(obs, belief),
        "p2": lambda: p2_step_latency(obs, belief),
        "p3": lambda: p3_branching(obs, belief),
        "p4": lambda: p4_tolerance(obs, belief),
        "p5": lambda: p5_manual_coin(obs, belief),
        "p6": lambda: p6_playout(obs, belief),
        "p7": lambda: p7_memory(obs, belief),
        "p8": lambda: p8_deck_order(obs, belief, seat),
        "p9": lambda: p9_consistency(obs, belief, seat),
        "p10": lambda: p10_live_battle_survives(obs, belief),
        "p11": lambda: p11_memory_plateau(obs, belief),
        "p12": lambda: p12_coin(obs, belief),
        "p13": lambda: {
            "engine_flips": p13_chance_determinism(obs, belief, manual_coin=False),
            "we_call_the_coin": p13_chance_determinism(obs, belief, manual_coin=True),
        },
    }
    for title, key in PROBES:
        print(f"--- {title}")
        started = time.perf_counter()
        try:
            value = runners[key]()
        except Exception as error:  # noqa: BLE001 — a failed probe is a result
            value = {"FAILED": f"{type(error).__name__}: {error}"}
        results[key] = value
        for name, item in value.items():
            print(f"      {name:<26} {item}")
        print(f"      {'(probe wall time)':<26} {time.perf_counter() - started:.2f}s")
        print()

    try:
        search_end()
    except Exception:  # noqa: BLE001
        pass
    battle_finish()

    out = Path(args.out) if args.out else _ROOT / (
        "probe_search_api_coin.json" if args.coin_deck else "probe_search_api.json"
    )
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
