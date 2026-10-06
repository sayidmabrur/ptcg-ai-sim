from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
    def __init__(self) -> None:
        self.supporter: int | None = None
        self.stadium: int | None = None
        self.energy: int | None = None
        self.last_energy_attach: int | None = None


class Extractor(ABC):
    def __init__(self) -> None:
        self.cards: dict[int, _CardInfo] = {}
        self.vocab: list[str] = list(CARD_SPECIAL_TOKENS)
        for card in sorted(all_card_data(), key=lambda c: c.cardId):
            self.cards[card.cardId] = _card_info(len(self.vocab), card)
            self.vocab.append(card.name)
        self._plays: dict[int, _TurnPlays] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def reset(self) -> None:
        self._plays = {}

    def __call__(self, obs):
        return self.encode(obs)

    @abstractmethod
    def encode(self, obs):
        ...

    @abstractmethod
    def decode(self, features):
        ...

    # -- slot filling -------------------------------------------------------

    def _card_index(self, card_id: int | None) -> int:
        if card_id is None:
            return CARD_SPECIAL_TOKENS.index("<HIDDEN>")
        info = self.cards.get(card_id)
        return info.vocab_index if info else CARD_SPECIAL_TOKENS.index("<UNK>")

    def _fill_card(self, out: dict[str, np.ndarray], slot: int, card_id: int | None) -> None:
        out["input_ids"][slot] = self._card_index(card_id)
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
        if turn <= 0 or first_player < 0:
            return None
        return first_player if turn % 2 == 1 else 1 - first_player

    def _update_plays(self, seat: int, obs: Observation, turn_player: int | None) -> _TurnPlays:
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

    @staticmethod
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
