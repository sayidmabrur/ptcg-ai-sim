import numpy as np
from extractor import Extractor, observation, to_observation_class


class PlayerExtractor(Extractor):

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize

    def encode(self, obs: dict | observation) -> dict[str, np.ndarray]:
        if isinstance(obs, dict)
            obs = to_observation_class(obs)
        state = obs.current

        if state is None or obs.select is None:
            raise ValueError("observation has no board (initial deck selection); skip it")
