from __future__ import annotations

import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Binary, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE / "feature_extractor"))

from observation_encoder import ObservationEncoder  # noqa: E402
from option_extractor import MAX_OPTIONS  # noqa: E402
from policy import ALL_SPECS, DONE, ENV_SPECS, N_ACTIONS, PolicyNetwork, load_policy  # noqa: E402

from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

_TORCH_DTYPE = {np.int64: torch.int64, np.float32: torch.float32, np.bool_: torch.bool}
INFO_SPECS: dict[str, tuple[tuple[int, ...], type]] = {
    "outcome": ((1,), np.float32),
    "opponent_deck": ((1,), np.int64),
    "opponent_kind": ((1,), np.int64),
}
OPPONENT_RANDOM, OPPONENT_LATEST, OPPONENT_OLDER, OPPONENT_FIXED = range(4)


def selection_state(select: dict, picked: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reachable = min(len(select["option"]), MAX_OPTIONS)
    mask = np.zeros(N_ACTIONS, dtype=np.bool_)
    if len(picked) < select["maxCount"]:
        mask[:reachable] = True
        mask[picked] = False
    mask[DONE] = len(picked) >= select["minCount"]
    option_picked = np.zeros(MAX_OPTIONS, dtype=np.bool_)
    option_picked[picked] = True
    return mask, np.array([len(picked) / 10.0], dtype=np.float32), option_picked


def _random_selection(select: dict, rng: random.Random) -> list[int]:
    n = len(select["option"])
    return rng.sample(range(n), rng.randint(select["minCount"], min(select["maxCount"], n)))


class RandomOpponent:
    kind = OPPONENT_RANDOM

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def select(self, obs: dict, encoder: ObservationEncoder) -> list[int]:
        return _random_selection(obs["select"], self.rng)


class PolicyOpponent:
    def __init__(self, network: PolicyNetwork, greedy: bool = False, kind: int = OPPONENT_FIXED) -> None:
        self.network = network.eval()
        self.greedy = greedy
        self.kind = kind

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def select(self, obs: dict, encoder: ObservationEncoder) -> list[int]:
        select = obs["select"]
        if select["minCount"] > MAX_OPTIONS:
            return _random_selection(select, random.Random())
        features = encoder.encode(obs)
        picked: list[int] = []
        while len(picked) < select["maxCount"]:
            mask, count, option_picked = selection_state(select, picked)
            batch = {key: torch.as_tensor(value)[None] for key, value in {
                **features, "action_mask": mask, "picked_count": count, "option_picked": option_picked}.items()}
            logits, _ = self.network(batch)
            action = int(logits.argmax(-1)) if self.greedy else int(torch.distributions.Categorical(logits=logits).sample())
            if action == DONE:
                break
            picked.append(action)
        return picked


class PoolOpponent:
    def __init__(self, pool_dir: str | Path, encoder: ObservationEncoder, p_random: float = 0.2,
                 latest_prob: float = 0.5, max_cached: int = 3, seed: int | None = None) -> None:
        self.pool_dir = Path(pool_dir)
        self.encoder = encoder
        self.p_random = p_random
        self.latest_prob = latest_prob
        self.max_cached = max_cached
        self.rng = random.Random(seed)
        self._random = RandomOpponent(seed)
        self._cache: OrderedDict[Path, PolicyOpponent] = OrderedDict()
        self.current = self._random
        self.kind = OPPONENT_RANDOM

    def _load(self, path: Path, kind: int) -> PolicyOpponent:
        if path not in self._cache:
            self._cache[path] = PolicyOpponent(load_policy(path, self.encoder), kind=kind)
            while len(self._cache) > self.max_cached:
                self._cache.popitem(last=False)
        self._cache.move_to_end(path)
        self._cache[path].kind = kind
        return self._cache[path]

    def reset(self) -> None:
        snapshots = sorted(self.pool_dir.glob("snapshot_*.pt"))
        if not snapshots or self.rng.random() < self.p_random:
            self.current, self.kind = self._random, OPPONENT_RANDOM
            return
        latest = self.rng.random() < self.latest_prob or len(snapshots) == 1
        path = snapshots[-1] if latest else self.rng.choice(snapshots[:-1])
        kind = OPPONENT_LATEST if latest else OPPONENT_OLDER
        try:
            self.current, self.kind = self._load(path, kind), kind
        except (OSError, RuntimeError, EOFError):
            self.current, self.kind = self._random, OPPONENT_RANDOM

    def select(self, obs: dict, encoder: ObservationEncoder) -> list[int]:
        return self.current.select(obs, encoder)


def make_env_fn(agent_deck: list[int], opponent_decks: list[list[int]], **kwargs):
    from functools import partial

    return partial(PokemonTCGEnv, agent_deck, opponent_decks, **kwargs)


class PokemonTCGEnv(EnvBase):
    def __init__(
        self,
        agent_deck: list[int],
        opponent_decks: list[list[int]],
        opponent=None,
        opponent_pool: str | Path | None = None,
        agent_seat: int | str = "random",
        prize_reward: float = 0.1,
        max_engine_steps: int = 4000,
        seed: int | None = None,
        worker_threads: int | None = None,
        device: str | torch.device = "cpu",
        **pool_kwargs,
    ) -> None:
        super().__init__(device=device, batch_size=torch.Size([]))
        if worker_threads is not None:
            torch.set_num_threads(worker_threads)
        self.agent_deck = agent_deck
        self.opponent_decks = opponent_decks
        self.prize_reward = prize_reward
        self.agent_seat_mode = agent_seat
        self.max_engine_steps = max_engine_steps
        self.rng = random.Random(seed)
        self.encoder = ObservationEncoder(agent_deck)
        if opponent_pool is not None:
            opponent = PoolOpponent(opponent_pool, self.encoder, seed=seed, **pool_kwargs)
        self.opponent = opponent or RandomOpponent(seed)

        self._active = False
        self._over = True
        self._seat = 0
        self._obs: dict | None = None
        self._features: dict[str, np.ndarray] = {}
        self._picked: list[int] = []
        self._engine_steps = 0
        self._outcome = 0.0
        self._truncated = False
        self._opponent_deck = 0
        self._opponent_kind = OPPONENT_RANDOM
        self._prizes: int | None = None
        self._prize_pending = 0.0
        self._make_specs()


    def _make_specs(self) -> None:
        entries = {}
        for key, (shape, dtype) in {**self.encoder.specs, **ENV_SPECS, **INFO_SPECS}.items():
            torch_dtype = _TORCH_DTYPE[dtype]
            if torch_dtype is torch.bool:
                entries[key] = Binary(n=shape[-1], shape=torch.Size(shape), dtype=torch.bool, device=self.device)
            else:
                entries[key] = Unbounded(shape=torch.Size(shape), dtype=torch_dtype, device=self.device)
        assert ALL_SPECS.keys() <= entries.keys(), "the environment's observation does not cover the policy's inputs"
        self.observation_spec = Composite(entries, device=self.device)
        self.action_spec = Categorical(n=N_ACTIONS, shape=torch.Size([]), dtype=torch.int64, device=self.device)
        self.reward_spec = Unbounded(shape=torch.Size([1]), dtype=torch.float32, device=self.device)
        flag = lambda: Categorical(n=2, shape=torch.Size([1]), dtype=torch.bool, device=self.device)
        self.done_spec = Composite(done=flag(), terminated=flag(), truncated=flag(), device=self.device)


    def _finish_game(self) -> None:
        if self._active:
            battle_finish()
            self._active = False

    def _start_game(self) -> None:
        self._finish_game()
        self._seat = self.rng.choice([0, 1]) if self.agent_seat_mode == "random" else int(self.agent_seat_mode)
        self._opponent_deck = self.rng.randrange(len(self.opponent_decks))
        opponent_deck = self.opponent_decks[self._opponent_deck]
        decks = (self.agent_deck, opponent_deck) if self._seat == 0 else (opponent_deck, self.agent_deck)
        obs, start = battle_start(*decks)
        if obs is None:
            raise ValueError(f"player {start.errorPlayer}'s deck is illegal (error {start.errorType})")
        self._active = True
        self._engine_steps = 0
        self._truncated = False
        self._outcome = 0.0
        self._prizes = None
        self._prize_pending = 0.0
        self.encoder.set_decklists({self._seat: self.agent_deck, 1 - self._seat: opponent_deck})
        self.encoder.reset()
        self.opponent.reset()
        self._opponent_kind = getattr(self.opponent, "kind", OPPONENT_RANDOM)
        self._advance(obs)

    def _engine_select(self, picks: list[int]) -> dict:
        self._engine_steps += 1
        return battle_select(picks)

    def _track_prizes(self, obs: dict) -> None:
        count = len(obs["current"]["players"][self._seat]["prize"])
        if self._prizes is not None and count < self._prizes:
            self._prize_pending += (self._prizes - count) * self.prize_reward
        self._prizes = count

    def _advance(self, obs: dict) -> None:
        while True:
            self._track_prizes(obs)
            result = obs["current"]["result"]
            if result != -1:
                self._over = True
                self._outcome = 0.0 if result == 2 else (1.0 if result == self._seat else -1.0)
                self._obs = obs
                self._finish_game()
                return
            if self._engine_steps >= self.max_engine_steps:
                self._over, self._truncated, self._obs = True, True, obs
                self._finish_game()
                return
            select = obs["select"]
            if obs["current"]["yourIndex"] != self._seat:
                obs = self._engine_select(self.opponent.select(obs, self.encoder))
                continue
            if select["minCount"] > MAX_OPTIONS:
                obs = self._engine_select(_random_selection(select, self.rng))
                continue
            self._over = False
            self._obs = obs
            self._features = self.encoder.encode(obs)
            self._picked = []
            return


    def _observation(self) -> TensorDict:
        everything = {**ALL_SPECS, **INFO_SPECS}
        if self._over:
            features = {key: np.zeros(shape, dtype) for key, (shape, dtype) in everything.items()}
        else:
            mask, count, option_picked = selection_state(self._obs["select"], self._picked)
            features = {**self._features, "action_mask": mask, "picked_count": count, "option_picked": option_picked}
        features["outcome"] = np.array([self._outcome if self._over and not self._truncated else 0.0], np.float32)
        features["opponent_deck"] = np.array([self._opponent_deck], np.int64)
        features["opponent_kind"] = np.array([self._opponent_kind], np.int64)
        return TensorDict({key: torch.as_tensor(value, device=self.device) for key, value in features.items()},
                          batch_size=[], device=self.device)

    def _flags(self, td: TensorDict, terminated: bool, truncated: bool) -> TensorDict:
        for key, value in (("terminated", terminated), ("truncated", truncated), ("done", terminated or truncated)):
            td[key] = torch.tensor([value], dtype=torch.bool, device=self.device)
        return td


    def _reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:
        self._start_game()
        return self._flags(self._observation(), False, False)

    def _step(self, tensordict: TensorDict) -> TensorDict:
        if self._over:
            raise RuntimeError("the game is over; call reset()")
        action = int(tensordict["action"])
        select = self._obs["select"]
        mask = selection_state(select, self._picked)[0]
        if not 0 <= action < N_ACTIONS or not mask[action]:
            raise ValueError(f"illegal action {action} (legal: {np.flatnonzero(mask).tolist()})")

        submit = action == DONE
        if not submit:
            self._picked.append(action)
            submit = len(self._picked) >= select["maxCount"]
        if submit:
            self._advance(self._engine_select(list(self._picked)))

        td = self._observation()
        reward = self._prize_pending + (self._outcome if self._over and not self._truncated else 0.0)
        self._prize_pending = 0.0
        td["reward"] = torch.tensor([reward], dtype=torch.float32, device=self.device)
        return self._flags(td, self._over and not self._truncated, self._over and self._truncated)

    def _set_seed(self, seed: int | None) -> None:
        self.rng = random.Random(seed)
        if hasattr(self.opponent, "rng"):
            self.opponent.rng = random.Random(None if seed is None else seed + 1)

    def rand_action(self, tensordict: TensorDict | None = None) -> TensorDict:
        if tensordict is None:
            tensordict = TensorDict({}, batch_size=self.batch_size, device=self.device)
        mask = tensordict["action_mask"].float()
        tensordict["action"] = torch.multinomial(mask, 1).squeeze(-1)
        return tensordict

    def close(self, *, raise_if_closed: bool = True) -> None:
        self._finish_game()
        super().close(raise_if_closed=False)
