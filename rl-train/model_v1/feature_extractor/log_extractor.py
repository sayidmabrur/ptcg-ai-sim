from __future__ import annotations

import numpy as np

from extractor import N_AREAS, OWNERS, Extractor, Observation, area_index, area_name, to_observation_class
from card_set_extractor import MAX_HAND, MAX_LOOKING
from option_extractor import BOARD_TOKEN_OFFSET, HAND_TOKEN_OFFSET, LOOKING_TOKEN_OFFSET, ref_name

from cg.api import AreaType, LogType  # noqa: E402

MAX_LOGS = 64

N_LOG_TYPES = max(LogType) + 2

LOG_NUMERIC_COLUMNS = ["value", "put_damage_counter", "is_recover", "head", "has_basic_pokemon", "is_new"]
LOG_SCALE = np.array([340.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
_NUM = {name: i for i, name in enumerate(LOG_NUMERIC_COLUMNS)}

_CARD_FIELDS = {
    LogType.SWITCH: ("cardIdActive", "cardIdBench", None),
    LogType.CHANGE: ("cardIdBefore", "cardIdAfter", None),
    LogType.ATTACH: ("cardId", "cardIdTarget", None),
    LogType.EVOLVE: ("cardId", "cardIdTarget", None),
    LogType.DEVOLVE: ("cardId", "cardIdTarget", None),
    LogType.MOVE_ATTACHED: ("cardId", "cardIdBefore", "cardIdAfter"),
}
_SERIAL_FIELDS = {
    LogType.SWITCH: ("serialActive", "serialBench"),
    LogType.CHANGE: ("serialBefore", "serialAfter"),
    LogType.ATTACH: ("serial", "serialTarget"),
    LogType.EVOLVE: ("serial", "serialTarget"),
    LogType.DEVOLVE: ("serial", "serialTarget"),
    LogType.MOVE_ATTACHED: ("serial", "serialAfter"),
}


class LogExtractor(Extractor):
    SPECS: dict[str, tuple[tuple[int, ...], type]] = {
        "log_mask": ((MAX_LOGS,), np.bool_),
        "log_type": ((MAX_LOGS,), np.int64),
        "log_owner": ((MAX_LOGS,), np.int64),
        "log_cards": ((MAX_LOGS, 3), np.int64),
        "log_refs": ((MAX_LOGS, 2), np.int64),
        "log_areas": ((MAX_LOGS, 2), np.int64),
        "log_attack": ((MAX_LOGS,), np.int64),
        "log_numeric": ((MAX_LOGS, len(LOG_NUMERIC_COLUMNS)), np.float32),
    }

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize
        self._history: dict[int, list] = {}

    def reset(self) -> None:
        super().reset()
        self._history = {}

    def encode(self, obs: dict | Observation) -> dict[str, np.ndarray]:
        if isinstance(obs, dict):
            obs = to_observation_class(obs)
        state = obs.current
        if state is None or obs.select is None:
            raise ValueError("observation has no board (initial deck selection); skip it")

        me = state.yourIndex
        out = {key: np.zeros(shape, dtype) for key, (shape, dtype) in self.SPECS.items()}
        serial_token = self._serial_tokens(state, me)

        history = self._history.setdefault(me, [])
        history.extend(obs.logs)
        del history[:-MAX_LOGS]
        first_new = len(history) - min(len(obs.logs), len(history))

        for i, log in enumerate(history):
            kind = LogType(log.type)
            out["log_mask"][i] = True
            out["log_type"][i] = int(kind) + 1
            if log.playerIndex is not None:
                out["log_owner"][i] = OWNERS.index("ME" if log.playerIndex == me else "OPP")

            for slot, field in enumerate(_CARD_FIELDS.get(kind, ("cardId", None, None))):
                card_id = getattr(log, field) if field is not None else None
                if card_id is not None:
                    out["log_cards"][i, slot] = self._card_index(card_id)
            for slot, field in enumerate(_SERIAL_FIELDS.get(kind, ("serial", None))):
                serial = getattr(log, field) if field is not None else None
                if serial is not None:
                    out["log_refs"][i, slot] = serial_token.get(serial, 0)

            out["log_areas"][i] = (area_index(log.fromArea), area_index(log.toArea))
            if log.attackId is not None:
                out["log_attack"][i] = self._attack_index.get(log.attackId, 1)

            numeric = out["log_numeric"][i]
            numeric[_NUM["value"]] = log.value or 0
            numeric[_NUM["put_damage_counter"]] = bool(log.putDamageCounter)
            numeric[_NUM["is_recover"]] = bool(log.isRecover)
            numeric[_NUM["head"]] = bool(log.head)
            numeric[_NUM["has_basic_pokemon"]] = bool(log.hasBasicPokemon)
            numeric[_NUM["is_new"]] = i >= first_new

        if self.normalize:
            out["log_numeric"] /= LOG_SCALE
        return out

    @staticmethod
    def _serial_tokens(state, me: int) -> dict[int, int]:
        tokens: dict[int, int] = {}
        for seat in (me, 1 - me):
            player = state.players[seat]
            for area, pokemons in ((AreaType.ACTIVE, player.active), (AreaType.BENCH, player.bench)):
                for index, pokemon in enumerate(pokemons):
                    if pokemon is None:
                        continue
                    slot = Extractor.slot_of(area, index, seat, me)
                    if slot is None:
                        continue
                    tokens[pokemon.serial] = BOARD_TOKEN_OFFSET + slot
                    for card in pokemon.energyCards + pokemon.tools + pokemon.preEvolution:
                        tokens[card.serial] = BOARD_TOKEN_OFFSET + slot
        for card in state.stadium:
            tokens[card.serial] = BOARD_TOKEN_OFFSET + Extractor.slot_of(AreaType.STADIUM, 0, card.playerIndex, me)
        for index, card in enumerate(state.players[me].hand or []):
            if index < MAX_HAND:
                tokens[card.serial] = HAND_TOKEN_OFFSET + index
        for index, card in enumerate((state.looking or [])[:MAX_LOOKING]):
            if card is not None:
                tokens[card.serial] = LOOKING_TOKEN_OFFSET + index
        return tokens


    def decode(self, features: dict[str, np.ndarray]) -> list[dict[str, object]]:
        numeric = features["log_numeric"] * (LOG_SCALE if self.normalize else 1)
        rows = []
        for i in np.flatnonzero(features["log_mask"]):
            rows.append({
                "index": int(i),
                "type": LogType(int(features["log_type"][i]) - 1).name,
                "owner": OWNERS[features["log_owner"][i]],
                "card_1": self.vocab[features["log_cards"][i, 0]],
                "card_2": self.vocab[features["log_cards"][i, 1]],
                "card_3": self.vocab[features["log_cards"][i, 2]],
                "ref_1": ref_name(int(features["log_refs"][i, 0])),
                "ref_2": ref_name(int(features["log_refs"][i, 1])),
                "from_area": area_name(int(features["log_areas"][i, 0])),
                "to_area": area_name(int(features["log_areas"][i, 1])),
                "attack": self.attack_vocab[features["log_attack"][i]],
                **{name: round(float(v), 3) for name, v in zip(LOG_NUMERIC_COLUMNS, numeric[i])},
            })
        return rows
