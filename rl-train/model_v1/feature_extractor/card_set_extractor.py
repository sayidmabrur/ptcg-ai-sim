from __future__ import annotations

from collections import Counter

import numpy as np

from extractor import OWNERS, Extractor, LogType, Observation, to_observation_class

MAX_HAND = 30
MAX_LOOKING = 16
MAX_PILE = 40

PILES = ["my_discard", "opp_discard", "my_unseen", "opp_seen"]
_LOG_CARDS = (("cardId", "serial"), ("cardIdTarget", "serialTarget"), ("cardIdActive", "serialActive"),
              ("cardIdBench", "serialBench"), ("cardIdBefore", "serialBefore"), ("cardIdAfter", "serialAfter"))
_OWN_CARD_LOGS = frozenset({LogType.PLAY, LogType.ATTACH, LogType.EVOLVE, LogType.DEVOLVE, LogType.MOVE_ATTACHED,
                            LogType.ATTACK, LogType.MOVE_CARD, LogType.SWITCH, LogType.CHANGE})
COUNT_SCALE = float(np.log1p(60))


class CardSetExtractor(Extractor):
    SPECS: dict[str, tuple[tuple[int, ...], type]] = {
        "hand_ids": ((MAX_HAND,), np.int64),
        "hand_mask": ((MAX_HAND,), np.bool_),
        "looking_ids": ((MAX_LOOKING,), np.int64),
        "looking_mask": ((MAX_LOOKING,), np.bool_),
        "looking_owner": ((MAX_LOOKING,), np.int64),
        "pile_ids": ((len(PILES), MAX_PILE), np.int64),
        "pile_counts": ((len(PILES), MAX_PILE), np.float32),
        "pile_mask": ((len(PILES), MAX_PILE), np.bool_),
    }

    def __init__(self, decklists: list[int] | dict[int, list[int]], normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize
        self.decklists = decklists
        self._deck_counts: dict[int, Counter] = {}
        self._opp_seen: dict[int, dict[int, int]] = {}

    def reset(self) -> None:
        super().reset()
        self._opp_seen = {}

    def set_decklists(self, decklists: list[int] | dict[int, list[int]]) -> None:
        self.decklists = decklists
        self._deck_counts = {}

    def _deck_counter(self, seat: int) -> Counter:
        if seat not in self._deck_counts:
            deck = self.decklists[seat] if isinstance(self.decklists, dict) else self.decklists
            self._deck_counts[seat] = Counter(self._card_index(card_id) for card_id in deck)
        return self._deck_counts[seat]

    def encode(self, obs: dict | Observation) -> dict[str, np.ndarray]:
        if isinstance(obs, dict):
            obs = to_observation_class(obs)
        state = obs.current
        if state is None or obs.select is None:
            raise ValueError("observation has no board (initial deck selection); skip it")

        me = state.yourIndex
        me_state, opp_state = state.players[me], state.players[1 - me]
        out = {key: np.zeros(shape, dtype) for key, (shape, dtype) in self.SPECS.items()}

        hand = [self._card_index(card.id) for card in (me_state.hand or [])[:MAX_HAND]]
        out["hand_ids"][: len(hand)] = hand
        out["hand_mask"][: len(hand)] = True

        for i, card in enumerate((state.looking or [])[:MAX_LOOKING]):
            out["looking_ids"][i] = self._card_index(card.id if card is not None else None)
            out["looking_mask"][i] = True
            if card is not None:
                out["looking_owner"][i] = OWNERS.index("ME" if card.playerIndex == me else "OPP")

        piles = (
            Counter(self._card_index(card.id) for card in me_state.discard),
            Counter(self._card_index(card.id) for card in opp_state.discard),
            self._unseen(me, state, obs.select.effect),
            self._opp_seen_pile(me, state, obs.logs),
        )
        for p, counter in enumerate(piles):
            for i, (vocab_index, count) in enumerate(sorted(counter.items())[:MAX_PILE]):
                out["pile_ids"][p, i] = vocab_index
                out["pile_counts"][p, i] = np.log1p(count) / COUNT_SCALE if self.normalize else count
                out["pile_mask"][p, i] = True
        return out

    def _visible_own(self, me: int, state) -> dict[int, int]:
        seen: dict[int, int] = {}
        for card in state.players[me].hand or []:
            seen[card.serial] = card.id
        for card in state.players[me].discard:
            seen[card.serial] = card.id
        for pokemon in state.players[me].active + state.players[me].bench:
            if pokemon is None:
                continue
            seen[pokemon.serial] = pokemon.id
            for card in pokemon.energyCards + pokemon.tools + pokemon.preEvolution:
                seen[card.serial] = card.id
        for card in state.stadium:
            if card.playerIndex == me:
                seen[card.serial] = card.id
        return seen

    def _unseen(self, me: int, state, effect) -> Counter:
        unseen = Counter(self._deck_counter(me))
        seen = self._visible_own(me, state)
        if effect is not None and effect.playerIndex == me:
            seen.setdefault(effect.serial, effect.id)
        for card_id in seen.values():
            vocab_index = self._card_index(card_id)
            if unseen[vocab_index] > 0:
                unseen[vocab_index] -= 1
        return +unseen

    def _opp_seen_pile(self, me: int, state, logs) -> Counter:
        known = self._opp_seen.setdefault(me, {})
        mine = self._visible_own(me, state)
        opponent = state.players[1 - me]
        for card in opponent.discard:
            known[card.serial] = card.id
        for pokemon in opponent.active + opponent.bench:
            if pokemon is None:
                continue
            known[pokemon.serial] = pokemon.id
            for card in pokemon.energyCards + pokemon.tools + pokemon.preEvolution:
                known[card.serial] = card.id
        for card in state.stadium:
            if card.playerIndex != me:
                known[card.serial] = card.id
        for log in logs:
            if log.playerIndex == 1 - me and LogType(log.type) in _OWN_CARD_LOGS:
                for card_field, serial_field in _LOG_CARDS:
                    card_id, serial = getattr(log, card_field), getattr(log, serial_field)
                    if card_id is not None and serial is not None and serial not in mine:
                        known[serial] = card_id
        return Counter(self._card_index(card_id) for card_id in known.values())

    def decode(self, features: dict[str, np.ndarray]) -> dict[str, object]:
        rows: dict[str, object] = {
            "hand": [self.vocab[i] for i, m in zip(features["hand_ids"], features["hand_mask"]) if m],
            "looking": [
                (OWNERS[o], self.vocab[i])
                for i, o, m in zip(features["looking_ids"], features["looking_owner"], features["looking_mask"])
                if m
            ],
        }
        for p, name in enumerate(PILES):
            counts = features["pile_counts"][p]
            if self.normalize:
                counts = np.expm1(counts * COUNT_SCALE)
            rows[name] = {
                self.vocab[i]: int(round(c))
                for i, c, m in zip(features["pile_ids"][p], counts, features["pile_mask"][p])
                if m
            }
        return rows
