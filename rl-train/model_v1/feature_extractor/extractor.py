from __future__ import annotations

import re
import sys
from collections import Counter
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
    Attack,
    all_attack,
    all_card_data,
    to_observation_class,
)

MAX_BENCH = 8
MAX_TOOLS = 5
MAX_ATTACHED_ENERGY = 10
MAX_PRE_EVOLUTION = 3

MY_ACTIVE = 0
MY_BENCH = 1
OPP_ACTIVE = MY_BENCH + MAX_BENCH
OPP_BENCH = OPP_ACTIVE + 1
STADIUM_IN_PLAY = OPP_BENCH + MAX_BENCH
SUPPORTER_PLAYED = STADIUM_IN_PLAY + 1
ENERGY_PLAYED = SUPPORTER_PLAYED + 1
STADIUM_PLAYED = ENERGY_PLAYED + 1
N_SLOTS = STADIUM_PLAYED + 1

ATTACK_VOCAB_SPECIAL = ["<NONE>", "<UNK>"]
CARD_SPECIAL_TOKENS = ["<NONE>", "<HIDDEN>", "<UNK>"]
CARD_TYPES = ["SPECIAL", "POKEMON", "TRAINER", "ENERGY"]
ENERGY_TYPES = ["NONE"] + [e.name for e in EnergyType]
ZONES = ["ACTIVE", "BENCH", "STADIUM_IN_PLAY", "SUPPORTER_PLAYED", "ENERGY_PLAYED", "STADIUM_PLAYED"]
OWNERS = ["NEUTRAL", "ME", "OPP"]
STAGES = ["NONE", "BASIC", "STAGE1", "STAGE2"]
SUBTYPES = ["NONE", "ITEM", "TOOL", "SUPPORTER", "STADIUM"]
CONDITIONS = ["POISONED", "BURNED", "ASLEEP", "PARALYZED", "CONFUSED"]

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
MATCHUP_COLUMNS = [
    "has_attack", "can_attack", "min_shortfall", "best_shortfall", "best_damage_ratio", "ready_damage_ratio",
    "ready_ko", "ko_within_one",
]
ENERGY_COUNT_COLUMNS = [f"energy_{t.name.lower()}" for t in EnergyType]
_NUM = {name: i for i, name in enumerate(NUMERIC_COLUMNS)}
_CAT = {name: i for i, name in enumerate(CATEGORICAL_COLUMNS)}
GLOBAL_COLUMNS = [
    "turn_number",
    "my_deck", "my_prizes", "my_hand", "my_bench_max",
    "opp_deck", "opp_prizes", "opp_hand", "opp_bench_max",
    "is_my_turn", "went_first", "retreated_this_turn",
    "turn_action_count",
]

CSV_COLUMNS = (
    ["turn_number", "slot", "card_id", "card_type", "pokemon_type", "weakness", "zone", "owner", "available",
     "max_hp", "hp", "stage", "is_ex", "is_mega_ex", "is_tera", "retreat_cost", "appear_this_turn"]
    + CONDITION_COLUMNS
    + ["subtype", "is_special"]
    + [f"tool_{i + 1}" for i in range(MAX_TOOLS)]
    + [f"att_e_{i + 1}" for i in range(MAX_ATTACHED_ENERGY)]
    + [f"pre_evo_{i + 1}" for i in range(MAX_PRE_EVOLUTION)]
    + ENERGY_COUNT_COLUMNS
    + MATCHUP_COLUMNS
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


N_AREAS = max(AreaType) + 3
UNKNOWN_AREA = N_AREAS - 1


def area_index(area) -> int:
    if area is None:
        return 0
    return int(area) + 1 if int(area) in AreaType._value2member_map_ else UNKNOWN_AREA


def area_name(index: int) -> str:
    if index == 0:
        return ""
    return "<UNK>" if index == UNKNOWN_AREA else AreaType(index - 1).name


_ENERGY_NAMES = ENERGY_TYPES[1:]
STATIC_FEATURE_COLUMNS = (
    ["hp", "retreat_cost", "n_attacks", "max_damage", "min_attack_cost", "max_attack_cost", "has_skill", "ace_spec"]
    + [f"type_{e.lower()}" for e in _ENERGY_NAMES]
    + [f"weak_{e.lower()}" for e in _ENERGY_NAMES]
    + [f"resist_{e.lower()}" for e in _ENERGY_NAMES]
    + [f"best_attack_cost_{e.lower()}" for e in _ENERGY_NAMES]
)
_STATIC = {name: i for i, name in enumerate(STATIC_FEATURE_COLUMNS)}


def _static_features(card: CardData, attacks: dict[int, Attack]) -> np.ndarray:
    row = np.zeros(len(STATIC_FEATURE_COLUMNS), dtype=np.float32)
    owned = [attacks[a] for a in card.attacks if a in attacks]
    row[_STATIC["hp"]] = card.hp / 340.0
    row[_STATIC["retreat_cost"]] = card.retreatCost / 4.0
    row[_STATIC["n_attacks"]] = len(owned) / 3.0
    row[_STATIC["has_skill"]] = bool(card.skills)
    row[_STATIC["ace_spec"]] = card.aceSpec
    if owned:
        costs = [len(a.energies) for a in owned]
        best = max(owned, key=lambda a: a.damage)
        row[_STATIC["max_damage"]] = best.damage / 340.0
        row[_STATIC["min_attack_cost"]] = min(costs) / 5.0
        row[_STATIC["max_attack_cost"]] = max(costs) / 5.0
        for energy in best.energies:
            row[_STATIC[f"best_attack_cost_{EnergyType(energy).name.lower()}"]] += 1 / 5.0
    for prefix, energy in (("type", card.energyType), ("weak", card.weakness), ("resist", card.resistance)):
        if energy is not None and CardType(card.cardType) in (CardType.POKEMON, CardType.BASIC_ENERGY):
            row[_STATIC[f"{prefix}_{EnergyType(energy).name.lower()}"]] = 1.0
    return row


TEXT_VOCAB_SIZE = 96
_TEXT_STOPWORDS = frozenset(
    "the and you your this that with from any for may then its are not has have each all there their them they "
    "can does did too one two three four more less than into onto only also instead which while during until "
    "was were been being would could should whose those these such other another same either both most "
    "pokémon pokemon card cards".split()
)


def _card_text(card: CardData, attacks: dict[int, Attack]) -> str:
    return " ".join([s.text for s in card.skills] + [attacks[a].text for a in card.attacks if a in attacks])


def _text_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[^\W\d_]+", text.lower()) if len(w) > 2 and w not in _TEXT_STOPWORDS}


def _build_text_vocab(texts: list[str], size: int = TEXT_VOCAB_SIZE) -> list[str]:
    texts = [t for t in texts if t]
    document_frequency = Counter(w for t in texts for w in _text_words(t))
    usable = [(n, w) for w, n in document_frequency.items() if 8 <= n <= 0.35 * len(texts)]
    return [w for n, w in sorted(usable, key=lambda x: (-x[0], x[1]))[:size]]


RESISTANCE_REDUCTION = 30


def energy_shortfall(required: list[int], attached: list[int]) -> int:
    units: Counter = Counter()
    wild = 0
    for energy in attached:
        if energy == EnergyType.RAINBOW:
            wild += 1
        elif energy == EnergyType.TEAM_ROCKET:
            units[int(EnergyType.PSYCHIC)] += 1
            units[int(EnergyType.DARKNESS)] += 1
        else:
            units[energy] += 1
    typed = Counter(r for r in required if r != EnergyType.COLORLESS)
    colorless = len(required) - sum(typed.values())
    unmet = 0
    for energy_type, need in typed.items():
        used = min(need, units[energy_type])
        units[energy_type] -= used
        need -= used
        from_wild = min(need, wild)
        wild -= from_wild
        unmet += need - from_wild
    return unmet + max(0, colorless - (sum(units.values()) + wild))


def damage_after_modifiers(damage: int, attacker: CardData | None, defender: CardData | None) -> int:
    if attacker is None or defender is None or damage <= 0:
        return damage
    if defender.weakness is not None and attacker.energyType == defender.weakness:
        damage *= 2
    if defender.resistance is not None and attacker.energyType == defender.resistance:
        damage = max(0, damage - RESISTANCE_REDUCTION)
    return damage


class _TurnPlays:
    def __init__(self) -> None:
        self.supporter: int | None = None
        self.stadium: int | None = None
        self.energy: int | None = None
        self.last_energy_attach: int | None = None


class Extractor(ABC):
    def __init__(self) -> None:
        self.cards: dict[int, _CardInfo] = {}
        self._card_data: dict[int, CardData] = {}
        self.vocab: list[str] = list(CARD_SPECIAL_TOKENS)
        self._attacks = {a.attackId: a for a in all_attack()}
        self.attack_vocab: list[str] = list(ATTACK_VOCAB_SPECIAL)
        self._attack_index: dict[int, int] = {}
        for attack_id in sorted(self._attacks):
            self._attack_index[attack_id] = len(self.attack_vocab)
            self.attack_vocab.append(self._attacks[attack_id].name)
        cards = sorted(all_card_data(), key=lambda c: c.cardId)
        texts = [_card_text(card, self._attacks) for card in cards]
        self.text_vocab = _build_text_vocab(texts)
        self.static_feature_columns = STATIC_FEATURE_COLUMNS + [f"text_{w}" for w in self.text_vocab]
        rows = [np.zeros(len(self.static_feature_columns), dtype=np.float32) for _ in CARD_SPECIAL_TOKENS]
        for card, text in zip(cards, texts):
            self.cards[card.cardId] = _card_info(len(self.vocab), card)
            self._card_data[card.cardId] = card
            self.vocab.append(card.name)
            words = _text_words(text)
            keywords = np.array([w in words for w in self.text_vocab], dtype=np.float32)
            rows.append(np.concatenate([_static_features(card, self._attacks), keywords]))
        self.card_features = np.stack(rows)
        self._plays: dict[int, _TurnPlays] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def attack_vocab_size(self) -> int:
        return len(self.attack_vocab)

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
        if pokemon is None:
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
        for i, card in enumerate(pokemon.preEvolution[:MAX_PRE_EVOLUTION]):
            out["pre_evolution"][slot, i] = self._card_index(card.id)
        for energy_type in pokemon.energies:
            out["energy_counts"][slot, int(energy_type)] += 1


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
        if obs.current.energyAttached and plays.energy is None:
            plays.energy = plays.last_energy_attach
        return plays

    @staticmethod
    def slot_of(area: int, index: int, player_index: int, your_index: int) -> int | None:
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
