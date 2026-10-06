from __future__ import annotations

import numpy as np

from board_extractor import BoardExtractor
from card_set_extractor import CardSetExtractor
from extractor import Observation, to_observation_class
from log_extractor import LogExtractor
from option_extractor import OptionExtractor


class ObservationEncoder:
    def __init__(self, decklists: list[int] | dict[int, list[int]], normalize: bool = True) -> None:
        self.board = BoardExtractor(normalize=normalize)
        self.card_set = CardSetExtractor(decklists, normalize=normalize)
        self.option = OptionExtractor(normalize=normalize)
        self.log = LogExtractor(normalize=normalize)
        self.extractors = (self.board, self.card_set, self.option, self.log)

    @property
    def specs(self) -> dict[str, tuple[tuple[int, ...], type]]:
        merged: dict[str, tuple[tuple[int, ...], type]] = {}
        for extractor in self.extractors:
            clash = merged.keys() & extractor.SPECS.keys()
            assert not clash, f"extractors share keys: {clash}"
            merged.update(extractor.SPECS)
        return merged

    @property
    def vocab_size(self) -> int:
        return self.board.vocab_size

    @property
    def attack_vocab_size(self) -> int:
        return self.board.attack_vocab_size

    @property
    def card_features(self) -> np.ndarray:
        return self.board.card_features

    def set_decklists(self, decklists: list[int] | dict[int, list[int]]) -> None:
        self.card_set.set_decklists(decklists)

    def reset(self) -> None:
        for extractor in self.extractors:
            extractor.reset()

    def encode(self, obs: dict | Observation) -> dict[str, np.ndarray]:
        if isinstance(obs, dict):
            obs = to_observation_class(obs)
        out: dict[str, np.ndarray] = {}
        for extractor in self.extractors:
            out.update(extractor.encode(obs))
        return out

    __call__ = encode

    @staticmethod
    def collate(batch: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {key: np.stack([item[key] for item in batch]) for key in batch[0]}
