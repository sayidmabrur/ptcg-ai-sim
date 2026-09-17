"""The scalar value function: features, model, and checkpoint I/O.

One definition, three consumers — ``probe_value.py`` trains on raw Parquet
``state`` dicts, ``search.py`` evaluates leaves from ``SearchState``
dataclasses, and live play reads the engine's ``obs`` dicts. Those are three
different container types carrying the same fields, which is exactly the
situation ``observation.py`` was written to prevent duplicating: a model fed
features assembled by a second implementation is being fed a different input
than it learned from, silently, and nothing errors.

So ``scalar_features`` reads through ``_get``, which accepts either a mapping
or an attribute-bearing object. The engine names its dataclass fields
identically to its JSON keys, so one function covers all three paths and
``check_parity`` proves it on real data.

Why scalars and not the policy trunk
------------------------------------
Measured on 250 held-out expert-vs-expert games: these 24 features reach
explained variance 0.53 mid-game and 0.29 overall, against 0.12 for prize
differential alone and 0.01 for first-player alone. A single random rollout —
what the search used before — had residual noise of 0.89 against this model's
0.68, while costing 40 ms to this model's tens of microseconds.

The features are also deliberately deck-agnostic. The training corpus is other
players' decks (best Jaccard overlap with the challenger deck is 0.16), so
anything keyed on card identity would be learning a meta we do not play.
Prizes, HP, energy and counts mean the same thing in every matchup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

#: Shared with ``dataset.py``'s normalisation caps so a feature means the same
#: thing to this model as to the policy trunk.
HP_CAP = 400.0
BOARD_HP_CAP = 2000.0

SCALAR_NAMES = (
    "prize_diff", "my_prizes", "opp_prizes",
    "turn", "is_first", "turn_action_count",
    "my_hand", "opp_hand", "my_deck", "opp_deck",
    "my_bench", "opp_bench", "my_discard", "opp_discard",
    "my_active_hp", "opp_active_hp", "my_active_hp_frac", "opp_active_hp_frac",
    "my_board_hp", "opp_board_hp", "my_energy", "opp_energy",
    "my_status", "opp_status",
    # Lethality. Everything above describes the position; none of it answers
    # the question a Pokémon player actually asks first — can I take the
    # knockout this turn, and can they take one on me? Prizes only move when a
    # Pokémon dies, so these are the proximate cause of the thing the value
    # head is predicting, and the model was previously left to infer them from
    # HP and energy counts without knowing any attack's cost or damage.
    "my_best_damage", "opp_best_damage",
    "my_ko_threat", "opp_ko_threat",
    "my_damage_ratio", "opp_damage_ratio",
)

_STATUS = ("poisoned", "burned", "asleep", "paralyzed", "confused")


def _get(node: Any, key: str, default=None):
    """Read ``key`` from a mapping or from an object attribute.

    The single adapter that lets one feature implementation serve the Parquet
    dicts, the engine's live ``obs`` dicts, and ``cg.api``'s dataclasses.
    """
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _board(player) -> tuple[float, float, float, int, int]:
    """``(total board hp, active hp, active max hp, energy count, bench size)``."""
    total = 0.0
    energy = 0
    active_hp = active_max = 0.0
    active = _get(player, "active") or []
    if active and active[0] is not None:
        active_hp = float(_get(active[0], "hp", 0) or 0)
        active_max = float(_get(active[0], "maxHp", 0) or 0) or 1.0
        total += active_hp
        energy += len(_get(active[0], "energies") or [])
    for mon in _get(player, "bench") or []:
        if mon is None:
            continue
        total += float(_get(mon, "hp", 0) or 0)
        energy += len(_get(mon, "energies") or [])
    return total, active_hp, active_max, energy, len(_get(player, "bench") or [])


#: ``cardId -> [attackId]`` and ``attackId -> (damage, cost)``, loaded once from
#: the engine's static tables. Attack cost and damage are not on the board
#: struct, so lethality cannot be computed from the observation alone.
_CARD_ATTACKS: dict[int, list[int]] = {}
_ATTACKS: dict[int, tuple[int, tuple[int, ...]]] = {}

_COLORLESS, _RAINBOW = 0, 10


def _load_tables() -> None:
    if _ATTACKS:
        return
    from cg.api import all_attack, all_card_data

    for attack in all_attack():
        _ATTACKS[attack.attackId] = (
            attack.damage or 0, tuple(int(e) for e in (attack.energies or [])),
        )
    for card in all_card_data():
        _CARD_ATTACKS[card.cardId] = list(card.attacks or [])


def _affordable(cost: tuple[int, ...], attached: list[int]) -> bool:
    """Can this energy pay this cost?

    Typed requirements are met with exact matches first, then Rainbow (which
    counts as any type); whatever is left over pays the colourless portion.
    Greedy rather than a proper matching, which can only ever *under*-report
    affordability — the failure direction that makes the feature conservative
    rather than optimistic.
    """
    from collections import Counter

    have = Counter(attached)
    need = Counter(cost)
    colorless = need.pop(_COLORLESS, 0)
    for energy_type, count in need.items():
        used = min(have.get(energy_type, 0), count)
        have[energy_type] -= used
        count -= used
        if count:
            used = min(have.get(_RAINBOW, 0), count)
            have[_RAINBOW] -= used
            count -= used
        if count:
            return False
    return sum(v for v in have.values() if v > 0) >= colorless


def _best_damage(player) -> float:
    """Highest damage the active Pokémon can pay for right now.

    Active only: a benched Pokémon cannot attack this turn, and counting it
    would describe a position two turns away rather than the immediate threat.
    """
    _load_tables()
    active = _get(player, "active") or []
    if not active or active[0] is None:
        return 0.0
    mon = active[0]
    attached = [int(e) for e in (_get(mon, "energies") or [])]
    best = 0.0
    for attack_id in _CARD_ATTACKS.get(int(_get(mon, "id", 0) or 0), []):
        damage, cost = _ATTACKS.get(attack_id, (0, ()))
        if damage and _affordable(cost, attached):
            best = max(best, float(damage))
    return best


def scalar_features(state, seat: int) -> list[float]:
    """The scoreboard + lethality features, POV-relative to ``seat``.

    Every entry is public or own-side information, so this is computable at a
    search leaf and in live play without any privileged access.
    """
    players = _get(state, "players")
    mine, theirs = players[seat], players[1 - seat]
    my_prizes = len(_get(mine, "prize") or [])
    opp_prizes = len(_get(theirs, "prize") or [])
    my_hp, my_ahp, my_amax, my_en, my_bench = _board(mine)
    op_hp, op_ahp, op_amax, op_en, op_bench = _board(theirs)
    my_damage, op_damage = _best_damage(mine), _best_damage(theirs)
    turn = int(_get(state, "turn", 0) or 0)
    return [
        (opp_prizes - my_prizes) / 6.0,
        my_prizes / 6.0, opp_prizes / 6.0,
        min(turn, 120) / 120.0,
        1.0 if _get(state, "firstPlayer") == seat else 0.0,
        min(int(_get(state, "turnActionCount", 0) or 0), 60) / 60.0,
        min(int(_get(mine, "handCount", 0) or 0), 60) / 60.0,
        min(int(_get(theirs, "handCount", 0) or 0), 60) / 60.0,
        int(_get(mine, "deckCount", 0) or 0) / 60.0,
        int(_get(theirs, "deckCount", 0) or 0) / 60.0,
        my_bench / 8.0, op_bench / 8.0,
        min(len(_get(mine, "discard") or []), 60) / 60.0,
        min(len(_get(theirs, "discard") or []), 60) / 60.0,
        min(my_ahp, HP_CAP) / HP_CAP, min(op_ahp, HP_CAP) / HP_CAP,
        my_ahp / my_amax if my_amax else 0.0,
        op_ahp / op_amax if op_amax else 0.0,
        min(my_hp, BOARD_HP_CAP) / BOARD_HP_CAP,
        min(op_hp, BOARD_HP_CAP) / BOARD_HP_CAP,
        min(my_en, 20) / 20.0, min(op_en, 20) / 20.0,
        float(any(bool(_get(mine, c)) for c in _STATUS)),
        float(any(bool(_get(theirs, c)) for c in _STATUS)),
        min(my_damage, HP_CAP) / HP_CAP, min(op_damage, HP_CAP) / HP_CAP,
        # Clipped ratios rather than raw margins: what matters is whether the
        # knockout is available, and by how much is nearly irrelevant past 1.0.
        1.0 if (op_ahp > 0 and my_damage >= op_ahp) else 0.0,
        1.0 if (my_ahp > 0 and op_damage >= my_ahp) else 0.0,
        min(my_damage / op_ahp, 2.0) / 2.0 if op_ahp > 0 else 0.0,
        min(op_damage / my_ahp, 2.0) / 2.0 if my_ahp > 0 else 0.0,
    ]


class ScalarValue(nn.Module):
    """24 -> 128 -> 128 -> 1, tanh. About 20k parameters.

    Small on purpose. The search calls this once per expanded node against an
    engine step costing ~36 us, so an evaluator in the tens of microseconds
    keeps the search engine-bound. The policy trunk, at ~30 ms per batch-1
    forward, is ~300x an engine step and would make the search unaffordable
    however accurate it turned out to be.
    """

    def __init__(self, width: int = len(SCALAR_NAMES), hidden: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ScalarValueMT(nn.Module):
    """Shared trunk, two heads: game outcome and final prize margin.

    The outcome label is one bit per ~150 frames, and those frames are
    near-duplicates, so the effective supervision is roughly one bit per game.
    The prize margin grades the same game continuously — a 6-0 sweep and a 6-5
    grind are both "+1" to an outcome head but 1.00 and 0.17 here — and it is
    the game's own scoreboard rather than an opinion about how to play, which is
    what makes it safe as an auxiliary. It gives the trunk a dense, correctly
    signed gradient in exactly the early-game region where the outcome head was
    weakest (explained variance 0.088 at turns 0-4).

    Only ``value`` is used at a search leaf; ``margin`` exists to shape the
    representation and, optionally, to blend into the leaf score. Both are
    ``tanh``-bounded so a blend stays inside [-1, 1].
    """

    def __init__(self, width: int = len(SCALAR_NAMES), hidden: int = 256,
                 layers: int = 3, dropout: float = 0.1):
        super().__init__()
        blocks: list[nn.Module] = []
        size = width
        for _ in range(layers):
            blocks += [nn.Linear(size, hidden), nn.ReLU(), nn.Dropout(dropout)]
            size = hidden
        self.trunk = nn.Sequential(*blocks)
        self.value_head = nn.Linear(hidden, 1)
        self.margin_head = nn.Linear(hidden, 1)

    def both(self, x):
        hidden = self.trunk(x)
        return (
            torch.tanh(self.value_head(hidden)).squeeze(-1),
            torch.tanh(self.margin_head(hidden)).squeeze(-1),
        )

    def forward(self, x):
        """Outcome only — the signature ``ValueNetEvaluator`` expects."""
        return torch.tanh(self.value_head(self.trunk(x))).squeeze(-1)


class FeatureSubset(nn.Module):
    """Serve an older checkpoint by slicing the feature vector to its prefix.

    New features are *appended* to ``SCALAR_NAMES``, so a checkpoint trained on
    an earlier set sees exactly the columns it was fitted on and is still
    correct — just blind to whatever came later. Without this, appending a
    feature invalidates every existing checkpoint at once, which is how a
    running configuration sweep got killed mid-flight.

    Strictly a compatibility shim. It never pads or reorders: padding would feed
    a model inputs it never saw, and reordering silently remaps every column.
    """

    def __init__(self, model: nn.Module, width: int):
        super().__init__()
        self.model = model
        self.width = width

    def forward(self, x):
        return self.model(x[..., :self.width])


class BlendedValue(nn.Module):
    """``(1 - w) * outcome + w * margin`` from a ``ScalarValueMT``.

    A search leaf wants the lowest-variance ordering of positions, not
    necessarily a calibrated win probability. The margin head is a denser,
    better-conditioned target, so mixing some of it in can sharpen the ordering
    even though it is not what the game pays out. ``w`` is a knob to be measured,
    not assumed — at ``w = 0`` this is exactly the outcome head.
    """

    def __init__(self, model: ScalarValueMT, weight: float = 0.3):
        super().__init__()
        self.model = model
        self.weight = weight

    def forward(self, x):
        value, margin = self.model.both(x)
        return (1.0 - self.weight) * value + self.weight * margin


def save_value_net(path: Path, model: nn.Module, metrics: dict) -> None:
    """Weights plus the feature names they were fitted against.

    The names are the contract. Reordering or inserting a feature silently
    remaps every input, which produces a model that runs and is wrong — so
    ``load_value_net`` refuses rather than trusting the caller.
    """
    config = {}
    if isinstance(model, ScalarValueMT):
        hidden = model.value_head.in_features
        config = {
            "kind": "mt", "hidden": hidden,
            "layers": sum(1 for m in model.trunk if isinstance(m, nn.Linear)),
        }
    torch.save(
        {"model": model.state_dict(), "features": list(SCALAR_NAMES),
         "metrics": metrics, "config": config},
        path,
    )


def load_value_net(path: Path, device="cpu", blend: float = 0.0):
    """Rebuild whichever architecture the checkpoint holds.

    The feature-name check is the important part: reordering or inserting a
    feature silently remaps every input and produces a model that runs and is
    wrong, which nothing downstream would catch.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    stored = tuple(state["features"])
    prefix = len(stored)
    if stored != SCALAR_NAMES and stored != SCALAR_NAMES[:prefix]:
        raise RuntimeError(
            "value checkpoint was trained on a different feature set:\n"
            f"  checkpoint: {state['features']}\n  current:    {list(SCALAR_NAMES)}"
        )
    config = state.get("config") or {}
    if config.get("kind") == "mt":
        model: nn.Module = ScalarValueMT(
            len(state["features"]), hidden=config.get("hidden", 256),
            layers=config.get("layers", 3),
        )
    else:
        model = ScalarValue(len(state["features"]))
    model.load_state_dict(state["model"])
    model.eval().to(device)
    if blend and isinstance(model, ScalarValueMT):
        model = BlendedValue(model, blend).eval().to(device)
    if prefix != len(SCALAR_NAMES):
        model = FeatureSubset(model, prefix).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, state.get("metrics", {})


def check_parity(obs: dict, search_state) -> list[str]:
    """Confirm a live ``obs`` dict and a forked ``SearchState`` agree.

    ``search_begin`` roots a search at the live frame, so the two must produce
    identical features — if they do not, the search is evaluating a different
    position than the one being played, which no metric downstream would catch.
    """
    seat = obs["current"]["yourIndex"]
    live = scalar_features(obs["current"], seat)
    forked = scalar_features(search_state.observation.current, seat)
    return [
        f"{name}: live {a:.6f} vs search {b:.6f}"
        for name, a, b in zip(SCALAR_NAMES, live, forked)
        if abs(a - b) > 1e-9
    ]
