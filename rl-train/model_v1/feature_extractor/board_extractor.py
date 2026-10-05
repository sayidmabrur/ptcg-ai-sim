"""Board state features: one fixed-size table of 22 slot tokens per decision.

Called once per decision point (every observation that carries a ``select``),
from the perspective of the seat that is selecting (``current.yourIndex``):

    slot  0       my Active             (ACTIVE,  ME)
    slots 1-8     my Bench              (BENCH,   ME)
    slot  9       opponent's Active     (ACTIVE,  OPP)
    slots 10-17   opponent's Bench      (BENCH,   OPP)
    slot  18      Stadium in play       (STADIUM_IN_PLAY, NEUTRAL)
    slot  19      Supporter played this turn   (owner = turn player)
    slot  20      Energy manually attached this turn
    slot  21      Stadium played this turn

The output is plain numpy (no torch), so a TorchRL ``EnvBase`` can wrap it in
a TensorDict as is; ``BoardExtractor.SPECS`` gives the shapes/dtypes for its
observation spec. Every categorical is an index into a vocab below (index 0 is
always the "nothing here" value), and card-like columns (card_id, tools,
attached energy) all index the same card vocab so one ``nn.Embedding`` serves
them all.

The extractor is stateful: slots 19-21 need *which* card was played this turn,
which the state only gives as booleans, so it is read from the logs. The engine
delivers logs per seat (everything since that seat's last selection, the
opponent's turn included), so the tracking is kept per seat. Call ``reset()``
at the start of every game.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The engine is rl-train/engine/cg, imported as ``cg`` like the rest of rl-train
# (importing it as ``engine.cg`` too would load a second copy of the module).
_ENGINE = Path(__file__).resolve().parents[2] / "engine"
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from cg.api import (  # noqa: E402
    AreaType,
    CardData,
    CardType,
    EnergyType,
    LogType,
    Observation,
    Pokemon,
    all_card_data,
    to_observation_class,
)

MAX_BENCH = 8
MAX_TOOLS = 5
MAX_ATTACHED_ENERGY = 10

MY_ACTIVE = 0
MY_BENCH = 1
OPP_ACTIVE = MY_BENCH + MAX_BENCH  # 9
OPP_BENCH = OPP_ACTIVE + 1  # 10
STADIUM_IN_PLAY = OPP_BENCH + MAX_BENCH  # 18
SUPPORTER_PLAYED = STADIUM_IN_PLAY + 1  # 19
ENERGY_PLAYED = SUPPORTER_PLAYED + 1  # 20
STADIUM_PLAYED = ENERGY_PLAYED + 1  # 21
N_SLOTS = STADIUM_PLAYED + 1  # 22

# Vocabularies. Index 0 is the empty / "not applicable" value everywhere.
CARD_SPECIAL_TOKENS = ["<NONE>", "<HIDDEN>", "<UNK>"]  # empty slot, face-down card, id not in the card database
CARD_TYPES = ["SPECIAL", "POKEMON", "TRAINER", "ENERGY"]  # SPECIAL = no card in the slot
ENERGY_TYPES = ["NONE"] + [e.name for e in EnergyType]
ZONES = ["ACTIVE", "BENCH", "STADIUM_IN_PLAY", "SUPPORTER_PLAYED", "ENERGY_PLAYED", "STADIUM_PLAYED"]
OWNERS = ["NEUTRAL", "ME", "OPP"]
STAGES = ["NONE", "BASIC", "STAGE1", "STAGE2"]
SUBTYPES = ["NONE", "ITEM", "TOOL", "SUPPORTER", "STADIUM"]
CONDITIONS = ["POISONED", "BURNED", "ASLEEP", "PARALYZED", "CONFUSED"]

# Column layout of the categorical / numeric / global arrays.
CATEGORICAL_COLUMNS = ["card_type", "pokemon_type", "weakness", "zone", "owner", "stage", "subtype"]
CATEGORICAL_VOCABS = {
    "card_type": CARD_TYPES,
    "pokemon_type": ENERGY_TYPES,
    "weakness": ENERGY_TYPES,
    "zone": ZONES,
    "owner": OWNERS,
    "stage": STAGES,
    "subtype": SUBTYPES,
}
CONDITION_COLUMNS = [f"cond_{c.lower()}" for c in CONDITIONS]
NUMERIC_COLUMNS = [
    "max_hp", "hp", "is_ex", "is_mega_ex", "is_tera", "retreat_cost", "appear_this_turn", "is_special",
] + CONDITION_COLUMNS
_NUM = {name: i for i, name in enumerate(NUMERIC_COLUMNS)}
_CAT = {name: i for i, name in enumerate(CATEGORICAL_COLUMNS)}
GLOBAL_COLUMNS = [
    "turn_number",
    "my_deck", "my_prizes", "my_hand", "my_bench_max",
    "opp_deck", "opp_prizes", "opp_hand", "opp_bench_max",
    "is_my_turn", "went_first", "retreated_this_turn",
]

# Column order of board_state_preprocessed_example.csv; ``decode`` emits rows in it.
CSV_COLUMNS = (
    ["turn_number", "slot", "card_id", "card_type", "pokemon_type", "weakness", "zone", "owner", "available",
     "max_hp", "hp", "stage", "is_ex", "is_mega_ex", "is_tera", "retreat_cost", "appear_this_turn"]
    + CONDITION_COLUMNS
    + ["subtype", "is_special"]
    + [f"tool_{i + 1}" for i in range(MAX_TOOLS)]
    + [f"att_e_{i + 1}" for i in range(MAX_ATTACHED_ENERGY)]
    + GLOBAL_COLUMNS[1:]
)

_SLOT_ZONE = (
    [ZONES.index("ACTIVE")] + [ZONES.index("BENCH")] * MAX_BENCH
    + [ZONES.index("ACTIVE")] + [ZONES.index("BENCH")] * MAX_BENCH
    + [ZONES.index(z) for z in ("STADIUM_IN_PLAY", "SUPPORTER_PLAYED", "ENERGY_PLAYED", "STADIUM_PLAYED")]
)

_COARSE_TYPE = {
    CardType.POKEMON: "POKEMON",
    CardType.ITEM: "TRAINER",
    CardType.TOOL: "TRAINER",
    CardType.SUPPORTER: "TRAINER",
    CardType.STADIUM: "TRAINER",
    CardType.BASIC_ENERGY: "ENERGY",
    CardType.SPECIAL_ENERGY: "ENERGY",
}
_SUBTYPE = {
    CardType.ITEM: "ITEM",
    CardType.TOOL: "TOOL",
    CardType.SUPPORTER: "SUPPORTER",
    CardType.STADIUM: "STADIUM",
}


@dataclass(frozen=True)
class _CardInfo:
    """Static per-card features, precomputed once from the card database."""

    vocab_index: int
    name: str
    card_type: int
    pokemon_type: int
    weakness: int
    stage: int
    subtype: int
    is_ex: int
    is_mega_ex: int
    is_tera: int
    retreat_cost: int
    is_special: int


def _card_info(vocab_index: int, card: CardData) -> _CardInfo:
    ctype = CardType(card.cardType)
    is_pokemon = ctype == CardType.POKEMON
    if not is_pokemon:
        stage = "NONE"
    elif card.stage2:
        stage = "STAGE2"
    elif card.stage1:
        stage = "STAGE1"
    else:
        stage = "BASIC"
    return _CardInfo(
        vocab_index=vocab_index,
        name=card.name,
        card_type=CARD_TYPES.index(_COARSE_TYPE[ctype]),
        pokemon_type=ENERGY_TYPES.index(EnergyType(card.energyType).name) if is_pokemon else 0,
        weakness=ENERGY_TYPES.index(EnergyType(card.weakness).name) if is_pokemon and card.weakness is not None else 0,
        stage=STAGES.index(stage),
        subtype=SUBTYPES.index(_SUBTYPE.get(ctype, "NONE")),
        is_ex=int(card.ex),
        is_mega_ex=int(card.megaEx),
        is_tera=int(card.tera),
        retreat_cost=card.retreatCost if is_pokemon else 0,
        is_special=int(ctype == CardType.SPECIAL_ENERGY),
    )


class _TurnPlays:
    """Which Supporter / Stadium / Energy the turn player has played this turn, as card ids."""

    def __init__(self) -> None:
        self.supporter: int | None = None
        self.stadium: int | None = None
        self.energy: int | None = None
        self.last_energy_attach: int | None = None


class BoardExtractor:
    """Observation (dict or ``Observation``) -> dict of fixed-shape numpy arrays.

    Output keys (S = 22 slots):
        card_ids     int64   [S]       card vocab index of the card in the slot
        tools        int64   [S, 5]    card vocab index of attached tools
        energies     int64   [S, 10]   card vocab index of attached energy cards
        categorical  int64   [S, 7]    CATEGORICAL_COLUMNS, each indexing its own vocab
        numeric      float32 [S, 13]   NUMERIC_COLUMNS (scaled to ~[0, 1] unless normalize=False)
        slot_mask    bool    [S]       True = the slot exists; False = a Bench slot beyond that
                                       player's ``benchMax`` (mask it out of attention)
        globals      float32 [12]      GLOBAL_COLUMNS (scaled to ~[0, 1] unless normalize=False)

    ``retreat_cost`` is the card's printed cost; tools or abilities that change it
    are not visible here (whether a retreat is affordable shows up as the RETREAT
    option being offered or not). ``appear_this_turn`` is the engine's flag for a
    Pokémon put into play or evolved this turn, i.e. it cannot evolve yet.

    ``slot_mask`` is about whether a slot *exists*, not whether it is filled:
    an empty Active or an open Bench spot is a real token ("there is room"), and
    an empty slot 19-21 means "not played yet this turn". Only Bench slots past
    ``benchMax`` (normally 5; 8 with a Stadium that raises it) are masked.
    Whether a slot is filled is in ``card_ids`` (0 = <NONE>).
    """

    SPECS: dict[str, tuple[tuple[int, ...], type]] = {
        "card_ids": ((N_SLOTS,), np.int64),
        "tools": ((N_SLOTS, MAX_TOOLS), np.int64),
        "energies": ((N_SLOTS, MAX_ATTACHED_ENERGY), np.int64),
        "categorical": ((N_SLOTS, len(CATEGORICAL_COLUMNS)), np.int64),
        "numeric": ((N_SLOTS, len(NUMERIC_COLUMNS)), np.float32),
        "slot_mask": ((N_SLOTS,), np.bool_),
        "globals": ((len(GLOBAL_COLUMNS),), np.float32),
    }

    # Divisors used when normalize=True, to keep inputs roughly in [0, 1].
    NUMERIC_SCALE = np.ones(len(NUMERIC_COLUMNS), dtype=np.float32)
    NUMERIC_SCALE[[_NUM["max_hp"], _NUM["hp"]]] = 340.0
    NUMERIC_SCALE[_NUM["retreat_cost"]] = 4.0
    GLOBAL_SCALE = np.array([30.0, 60.0, 6.0, 20.0, 8.0, 60.0, 6.0, 20.0, 8.0, 1.0, 1.0, 1.0], dtype=np.float32)

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize
        self.cards: dict[int, _CardInfo] = {}
        self.card_vocab: list[str] = list(CARD_SPECIAL_TOKENS)
        for card in sorted(all_card_data(), key=lambda c: c.cardId):
            self.cards[card.cardId] = _card_info(len(self.card_vocab), card)
            self.card_vocab.append(card.name)
        self._plays: dict[int, _TurnPlays] = {}

    @property
    def card_vocab_size(self) -> int:
        return len(self.card_vocab)

    def reset(self) -> None:
        """Forget per-turn tracking; call at the start of every game."""
        self._plays = {}

    def __call__(self, obs: dict | Observation) -> dict[str, np.ndarray]:
        if isinstance(obs, dict):
            obs = to_observation_class(obs)
        state = obs.current
        if state is None or obs.select is None:
            raise ValueError("observation has no board (initial deck selection); skip it")

        me = state.yourIndex
        opp = 1 - me
        turn_player = self._turn_player(state.turn, state.firstPlayer)
        plays = self._update_plays(me, obs, turn_player)

        out = {key: np.zeros(shape, dtype) for key, (shape, dtype) in self.SPECS.items()}
        out["categorical"][:, _CAT["zone"]] = _SLOT_ZONE
        out["slot_mask"][:] = True

        for owner_seat, active_slot, bench_slot in ((me, MY_ACTIVE, MY_BENCH), (opp, OPP_ACTIVE, OPP_BENCH)):
            player = state.players[owner_seat]
            owner = OWNERS.index("ME" if owner_seat == me else "OPP")
            out["categorical"][active_slot : bench_slot + MAX_BENCH, _CAT["owner"]] = owner
            if player.active:
                self._fill_pokemon(out, active_slot, player.active[0])
                conditions = (player.poisoned, player.burned, player.asleep, player.paralyzed, player.confused)
                out["numeric"][active_slot, _NUM[CONDITION_COLUMNS[0]] :] = conditions
            for i, pokemon in enumerate(player.bench[:MAX_BENCH]):
                self._fill_pokemon(out, bench_slot + i, pokemon)
            out["slot_mask"][bench_slot + min(player.benchMax, MAX_BENCH) : bench_slot + MAX_BENCH] = False

        if state.stadium:
            self._fill_card(out, STADIUM_IN_PLAY, state.stadium[0].id)

        # The per-turn slots belong to whoever's turn it is, which is not
        # always the selecting seat (e.g. choosing a new Active on their turn).
        if turn_player is not None:
            owner = OWNERS.index("ME" if turn_player == me else "OPP")
            out["categorical"][SUPPORTER_PLAYED:, _CAT["owner"]] = owner
            for slot, played, card_id in (
                (SUPPORTER_PLAYED, state.supporterPlayed, plays.supporter),
                (ENERGY_PLAYED, state.energyAttached, plays.energy),
                (STADIUM_PLAYED, state.stadiumPlayed, plays.stadium),
            ):
                if played:
                    self._fill_card(out, slot, card_id)

        me_state, opp_state = state.players[me], state.players[opp]
        out["globals"][:] = (
            state.turn,
            me_state.deckCount, len(me_state.prize), me_state.handCount, me_state.benchMax,
            opp_state.deckCount, len(opp_state.prize), opp_state.handCount, opp_state.benchMax,
            turn_player == me,
            state.firstPlayer == me,
            state.retreated,
        )

        if self.normalize:
            out["numeric"] /= self.NUMERIC_SCALE
            out["globals"] /= self.GLOBAL_SCALE
        return out

    # -- slot filling -------------------------------------------------------

    def _card_index(self, card_id: int | None) -> int:
        if card_id is None:
            return CARD_SPECIAL_TOKENS.index("<HIDDEN>")
        info = self.cards.get(card_id)
        return info.vocab_index if info else CARD_SPECIAL_TOKENS.index("<UNK>")

    def _fill_card(self, out: dict[str, np.ndarray], slot: int, card_id: int | None) -> None:
        """Static card features. ``card_id=None`` is a card known to be there but not which one."""
        out["card_ids"][slot] = self._card_index(card_id)
        info = self.cards.get(card_id) if card_id is not None else None
        if info is None:
            return
        cat, num = out["categorical"][slot], out["numeric"][slot]
        for name in ("card_type", "pokemon_type", "weakness", "stage", "subtype"):
            cat[_CAT[name]] = getattr(info, name)
        for name in ("is_ex", "is_mega_ex", "is_tera", "retreat_cost", "is_special"):
            num[_NUM[name]] = getattr(info, name)

    def _fill_pokemon(self, out: dict[str, np.ndarray], slot: int, pokemon: Pokemon | None) -> None:
        if pokemon is None:  # face-down Active during setup
            self._fill_card(out, slot, None)
            out["categorical"][slot, _CAT["card_type"]] = CARD_TYPES.index("POKEMON")
            return
        self._fill_card(out, slot, pokemon.id)
        out["numeric"][slot, _NUM["max_hp"]] = pokemon.maxHp
        out["numeric"][slot, _NUM["hp"]] = pokemon.hp
        out["numeric"][slot, _NUM["appear_this_turn"]] = pokemon.appearThisTurn
        for i, tool in enumerate(pokemon.tools[:MAX_TOOLS]):
            out["tools"][slot, i] = self._card_index(tool.id)
        for i, energy in enumerate(pokemon.energyCards[:MAX_ATTACHED_ENERGY]):
            out["energies"][slot, i] = self._card_index(energy.id)

    # -- per-turn tracking --------------------------------------------------

    @staticmethod
    def _turn_player(turn: int, first_player: int) -> int | None:
        """Turn 1 is the first player's, turn 2 the second's, ...; turn 0 is setup."""
        if turn <= 0 or first_player < 0:
            return None
        return first_player if turn % 2 == 1 else 1 - first_player

    def _update_plays(self, seat: int, obs: Observation, turn_player: int | None) -> _TurnPlays:
        """Replay this seat's new logs into its record of the current turn's plays."""
        plays = self._plays.setdefault(seat, _TurnPlays())
        for log in obs.logs:
            if log.type == LogType.TURN_START:
                plays = self._plays[seat] = _TurnPlays()
                continue
            info = self.cards.get(log.cardId) if log.cardId is not None else None
            if info is None or log.playerIndex != turn_player:
                continue
            if log.type == LogType.PLAY:
                if info.subtype == SUBTYPES.index("SUPPORTER"):
                    plays.supporter = log.cardId
                elif info.subtype == SUBTYPES.index("STADIUM"):
                    plays.stadium = log.cardId
            elif log.type == LogType.ATTACH and info.card_type == CARD_TYPES.index("ENERGY"):
                plays.last_energy_attach = log.cardId
        # Abilities can attach Energy too, and the log does not say which attach
        # was the manual one. The flag flips on the manual one, and the turn
        # player gets a selection right after every action of their own, so the
        # most recent Energy attach when it is first seen True is the manual one.
        if obs.current.energyAttached and plays.energy is None:
            plays.energy = plays.last_energy_attach
        return plays

    # -- decoding (for probes / debugging) ----------------------------------

    def decode(self, features: dict[str, np.ndarray]) -> list[dict[str, object]]:
        """Arrays back to one human-readable row per slot, in ``CSV_COLUMNS`` order."""
        numeric = features["numeric"] * (self.NUMERIC_SCALE if self.normalize else 1)
        globals_ = features["globals"] * (self.GLOBAL_SCALE if self.normalize else 1)
        global_values = {name: int(round(v)) for name, v in zip(GLOBAL_COLUMNS, globals_)}
        rows = []
        for slot in range(N_SLOTS):
            row: dict[str, object] = dict(global_values, slot=slot)
            row["card_id"] = self.card_vocab[features["card_ids"][slot]]
            row["available"] = int(features["slot_mask"][slot])
            for name, value in zip(CATEGORICAL_COLUMNS, features["categorical"][slot]):
                row[name] = CATEGORICAL_VOCABS[name][value]
            for name, value in zip(NUMERIC_COLUMNS, numeric[slot]):
                row[name] = int(round(value))
            for i in range(MAX_TOOLS):
                row[f"tool_{i + 1}"] = self.card_vocab[features["tools"][slot, i]]
            for i in range(MAX_ATTACHED_ENERGY):
                row[f"att_e_{i + 1}"] = self.card_vocab[features["energies"][slot, i]]
            rows.append({name: row[name] for name in CSV_COLUMNS})
        return rows


def slot_of(area: int, index: int, player_index: int, your_index: int) -> int | None:
    """Board slot an option's (area, index, playerIndex) points at, or None if not on the board.

    Lets the policy score a card-targeting option by attending to that slot's token.
    """
    if area == AreaType.STADIUM:
        return STADIUM_IN_PLAY
    if area not in (AreaType.ACTIVE, AreaType.BENCH):
        return None
    mine = player_index == your_index
    if area == AreaType.ACTIVE:
        return MY_ACTIVE if mine else OPP_ACTIVE
    if not 0 <= index < MAX_BENCH:
        return None
    return (MY_BENCH if mine else OPP_BENCH) + index
