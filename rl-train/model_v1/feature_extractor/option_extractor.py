from __future__ import annotations

import numpy as np

from extractor import N_AREAS, N_SLOTS, OWNERS, Extractor, Observation, area_index, area_name, damage_after_modifiers, to_observation_class
from card_set_extractor import MAX_HAND, MAX_LOOKING

from cg.api import (  # noqa: E402
    AreaType,
    OptionType,
    SelectContext,
    SelectType,
    SpecialConditionType,
)

MAX_OPTIONS = 64

BOARD_TOKEN_OFFSET = 1
HAND_TOKEN_OFFSET = BOARD_TOKEN_OFFSET + N_SLOTS
LOOKING_TOKEN_OFFSET = HAND_TOKEN_OFFSET + MAX_HAND
N_REF_TOKENS = LOOKING_TOKEN_OFFSET + MAX_LOOKING

N_OPTION_TYPES = max(OptionType) + 2
N_SELECT_TYPES = max(SelectType) + 2
N_SELECT_CONTEXTS = max(SelectContext) + 2
N_SPECIAL_CONDITIONS = max(SpecialConditionType) + 2

SELECT_NUMERIC_COLUMNS = ["min_count", "max_count", "remain_energy_cost", "remain_damage_counter"]
SELECT_SCALE = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)
OPTION_NUMERIC_COLUMNS = ["number", "count", "attack_damage", "attack_cost", "damage_ratio", "is_ko"]
OPTION_SCALE = np.array([10.0, 5.0, 340.0, 5.0, 1.0, 1.0], dtype=np.float32)
_NUM = {name: i for i, name in enumerate(OPTION_NUMERIC_COLUMNS)}

_NO_CARD = (OptionType.RETREAT, OptionType.END, OptionType.YES, OptionType.NO, OptionType.NUMBER)


def ref_name(ref: int) -> str:
    if ref == 0:
        return ""
    if ref < HAND_TOKEN_OFFSET:
        return f"board:{ref - BOARD_TOKEN_OFFSET}"
    if ref < LOOKING_TOKEN_OFFSET:
        return f"hand:{ref - HAND_TOKEN_OFFSET}"
    return f"looking:{ref - LOOKING_TOKEN_OFFSET}"


def _at(seq, index):
    if seq is None or index is None or not 0 <= index < len(seq):
        return None
    return seq[index]


class OptionExtractor(Extractor):
    SPECS: dict[str, tuple[tuple[int, ...], type]] = {
        "select_type": ((), np.int64),
        "select_context": ((), np.int64),
        "select_numeric": ((len(SELECT_NUMERIC_COLUMNS),), np.float32),
        "select_cards": ((2,), np.int64),
        "option_mask": ((MAX_OPTIONS,), np.bool_),
        "option_type": ((MAX_OPTIONS,), np.int64),
        "src_ref": ((MAX_OPTIONS,), np.int64),
        "dst_ref": ((MAX_OPTIONS,), np.int64),
        "src_area": ((MAX_OPTIONS,), np.int64),
        "owner": ((MAX_OPTIONS,), np.int64),
        "card_id": ((MAX_OPTIONS,), np.int64),
        "sub_index": ((MAX_OPTIONS,), np.int64),
        "attack_id": ((MAX_OPTIONS,), np.int64),
        "special_condition": ((MAX_OPTIONS,), np.int64),
        "option_numeric": ((MAX_OPTIONS, len(OPTION_NUMERIC_COLUMNS)), np.float32),
    }

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize

    def encode(self, obs: dict | Observation) -> dict[str, np.ndarray]:
        if isinstance(obs, dict):
            obs = to_observation_class(obs)
        state, select = obs.current, obs.select
        if state is None or select is None:
            raise ValueError("observation has no board (initial deck selection); skip it")

        me = state.yourIndex
        out = {key: np.zeros(shape, dtype) for key, (shape, dtype) in self.SPECS.items()}
        out["select_type"][()] = int(select.type) + 1
        out["select_context"][()] = int(select.context) + 1
        out["select_numeric"][:] = (
            select.minCount, select.maxCount, select.remainEnergyCost, select.remainDamageCounter,
        )
        out["select_cards"][:] = [self._card_index(c.id) if c is not None else 0 for c in (select.contextCard, select.effect)]

        for i, option in enumerate(select.option[:MAX_OPTIONS]):
            self._encode_option(out, i, option, state, select, me)

        if self.normalize:
            out["select_numeric"] /= SELECT_SCALE
            out["option_numeric"] /= OPTION_SCALE
        return out

    __call__ = encode


    def _encode_option(self, out, i, option, state, select, me) -> None:
        kind = OptionType(option.type)
        player = me if option.playerIndex is None else option.playerIndex
        area = AreaType.HAND if kind == OptionType.PLAY else option.area

        out["option_mask"][i] = True
        out["option_type"][i] = int(kind) + 1
        out["src_area"][i] = area_index(area)
        out["owner"][i] = 0 if option.playerIndex is None else OWNERS.index("ME" if player == me else "OPP")
        out["src_ref"][i] = self._ref(area, option.index, player, me)
        if option.inPlayArea is not None:
            out["dst_ref"][i] = self._ref(option.inPlayArea, option.inPlayIndex, player, me)

        card = self._option_card(kind, option, area, player, state, select)
        if card is not False:
            out["card_id"][i] = self._card_index(card)
        elif option.cardId:
            out["card_id"][i] = self._card_index(option.cardId)

        if option.toolIndex is not None:
            out["sub_index"][i] = option.toolIndex + 1
        elif option.energyIndex is not None:
            out["sub_index"][i] = option.energyIndex + 1
        if option.specialConditionType is not None:
            out["special_condition"][i] = int(option.specialConditionType) + 1

        numeric = out["option_numeric"][i]
        numeric[_NUM["number"]] = option.number or 0
        numeric[_NUM["count"]] = option.count or 0
        if kind == OptionType.ATTACK:
            out["attack_id"][i] = self._attack_index.get(option.attackId, 1)
            self._fill_attack(numeric, option.attackId, state, me)

    @staticmethod
    def _ref(area, index, player, me) -> int:
        if area == AreaType.STADIUM:
            return BOARD_TOKEN_OFFSET + Extractor.slot_of(area, 0, player, me)
        if area is None or index is None:
            return 0
        if area == AreaType.HAND:
            return HAND_TOKEN_OFFSET + index if player == me and 0 <= index < MAX_HAND else 0
        if area == AreaType.LOOKING:
            return LOOKING_TOKEN_OFFSET + index if 0 <= index < MAX_LOOKING else 0
        slot = Extractor.slot_of(area, index, player, me)
        return 0 if slot is None else BOARD_TOKEN_OFFSET + slot

    def _pokemon_at(self, state, area, index, player):
        if area == AreaType.ACTIVE:
            return _at(state.players[player].active, index)
        if area == AreaType.BENCH:
            return _at(state.players[player].bench, index)
        return None

    def _option_card(self, kind, option, area, player, state, select):
        if kind in (OptionType.TOOL_CARD, OptionType.ENERGY_CARD, OptionType.ENERGY):
            pokemon = self._pokemon_at(state, area, option.index, player)
            if pokemon is None:
                return None
            attached = pokemon.tools if kind == OptionType.TOOL_CARD else pokemon.energyCards
            card = _at(attached, option.toolIndex if kind == OptionType.TOOL_CARD else option.energyIndex)
            return None if card is None else card.id
        if kind in _NO_CARD or kind in (OptionType.ATTACK, OptionType.SPECIAL_CONDITION, OptionType.SKILL):
            return False
        if area is None:
            return False
        pokemon = self._pokemon_at(state, area, option.index, player)
        if pokemon is not None:
            return pokemon.id
        zones = {
            AreaType.HAND: state.players[player].hand,
            AreaType.DISCARD: state.players[player].discard,
            AreaType.PRIZE: state.players[player].prize,
            AreaType.DECK: select.deck,
            AreaType.LOOKING: state.looking,
            AreaType.STADIUM: state.stadium,
        }
        if area not in zones:
            return False
        card = _at(zones[area], 0 if area == AreaType.STADIUM else option.index)
        return None if card is None else card.id

    def _fill_attack(self, numeric, attack_id, state, me) -> None:
        attack = self._attacks.get(attack_id)
        if attack is None:
            return
        numeric[_NUM["attack_damage"]] = attack.damage
        numeric[_NUM["attack_cost"]] = len(attack.energies)
        attacker = _at(state.players[me].active, 0)
        defender = _at(state.players[1 - me].active, 0)
        if attacker is None or defender is None or defender.hp <= 0:
            return
        damage = damage_after_modifiers(attack.damage, self._card_data.get(attacker.id), self._card_data.get(defender.id))
        numeric[_NUM["damage_ratio"]] = min(damage / defender.hp, 2.0) / 2.0
        numeric[_NUM["is_ko"]] = damage > 0 and damage >= defender.hp


    def decode(self, features: dict[str, np.ndarray]) -> dict[str, object]:
        numeric = features["option_numeric"] * (OPTION_SCALE if self.normalize else 1)
        select_numeric = features["select_numeric"] * (SELECT_SCALE if self.normalize else 1)
        select = {
            "type": SelectType(int(features["select_type"]) - 1).name,
            "context": SelectContext(int(features["select_context"]) - 1).name,
            **{name: int(round(v)) for name, v in zip(SELECT_NUMERIC_COLUMNS, select_numeric)},
            "context_card": self.vocab[features["select_cards"][0]],
            "effect_card": self.vocab[features["select_cards"][1]],
        }
        options = []
        for i in np.flatnonzero(features["option_mask"]):
            options.append({
                "index": int(i),
                "type": OptionType(int(features["option_type"][i]) - 1).name,
                "src_ref": ref_name(int(features["src_ref"][i])),
                "dst_ref": ref_name(int(features["dst_ref"][i])),
                "src_area": area_name(int(features["src_area"][i])),
                "owner": OWNERS[features["owner"][i]],
                "card": self.vocab[features["card_id"][i]],
                "sub_index": int(features["sub_index"][i]) - 1,
                "attack": self.attack_vocab[features["attack_id"][i]],
                **{name: round(float(v), 3) for name, v in zip(OPTION_NUMERIC_COLUMNS, numeric[i])},
            })
        return {"select": select, "options": options}
