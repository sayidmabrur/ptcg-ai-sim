"""``agent()`` entry point for the packaged submission.

A PPO-trained ``ActorCritic``: the behavioural-cloning ``PolicyNetwork`` that
seeded it, plus a learned ``stop`` logit, decoded greedily through the same
sequential rule the policy was trained and evaluated under.

Two properties of the harness shape this file:

*Only our own decisions are visible.* ``agent()`` is invoked only when it is our
turn, so ``opponent_history`` becomes a diff between our successive views of the
opponent's board rather than one entry per opponent decision. That matches the
``own_frames`` spec the policy was trained on, so it is the intended view rather
than a degradation.

*A crash forfeits the match.* Every failure path falls back to a legal random
selection: a weak move scores far better than no move.
"""

import os
import random
import sys
from pathlib import Path

try:
    _ROOT = Path(__file__).resolve().parent
except NameError:
    # kaggle_environments execs this source against a bare globals dict, which
    # has no __file__. Touching it unguarded raises at module level, before any
    # fallback inside agent() could run, so the whole submission fails to load.
    _ROOT = Path("/kaggle_simulations/agent")
    if not _ROOT.is_dir():
        _ROOT = Path.cwd()

for _candidate in (_ROOT, Path("/kaggle_simulations/agent")):
    if not _candidate.is_dir():
        continue
    # The policy modules import each other by bare name, so their directory has
    # to be importable in its own right, not just the bundle root.
    for _entry in (_candidate, _candidate / "policy_network"):
        if _entry.is_dir() and str(_entry) not in sys.path:
            sys.path.insert(0, str(_entry))

from cg.api import Observation, to_observation_class

_NEG_INF = float("-inf")


def _resolve(filename: str) -> str:
    if os.path.exists(filename):
        return filename
    for base in (_ROOT, Path("/kaggle_simulations/agent")):
        candidate = base / filename
        if candidate.exists():
            return str(candidate)
    return filename


def read_deck_csv() -> list[int]:
    """The 60-card challenger decklist the policy was trained to pilot."""
    with open(_resolve("deck.csv"), "r") as file:
        lines = file.read().split("\n")
    return [int(lines[i]) for i in range(60)]


def _load_policy_module():
    """Import THIS bundle's ``policy_experimental`` by path, under a private name.

    A bare ``from policy_experimental import ...`` resolves through
    ``sys.modules``, so if anything else in the process already imported a
    module of that name — a different generation of the same file, with a
    different ``CardEmbed`` width — the bundle silently builds the wrong
    architecture and the weights fail to load. That is not hypothetical: it is
    exactly what happened the first time this bundle was validated in-process
    against an opponent whose loader had imported the challenger's copy first.

    In the deployed submission there is only one copy and the bare import would
    work. Loading by explicit path costs nothing and removes the failure mode.
    """
    import importlib.util
    import sys as _sys

    if "_bundle_policy" in _sys.modules:
        return _sys.modules["_bundle_policy"]
    source = Path(_resolve("policy_network/policy_experimental.py"))
    spec = importlib.util.spec_from_file_location("_bundle_policy", source)
    module = importlib.util.module_from_spec(spec)
    _sys.modules["_bundle_policy"] = module
    spec.loader.exec_module(module)
    return module


def _build_actor_critic():
    """Reconstruct the training-time module tree so the state dict loads.

    Mirrors ``train.ActorCritic``: the policy network, a value head (unused at
    inference but present in the checkpoint), and the ``stop`` logit that the
    decode below depends on.
    """
    import torch.nn as nn

    _policy_module = _load_policy_module()
    D, PolicyNetwork = _policy_module.D, _policy_module.PolicyNetwork

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = PolicyNetwork()
            self.value = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1))
            self.stop = nn.Linear(D, 1)

        def forward(self, features):
            policy = self.policy
            ctx_pooled, option_vecs, options_mask = policy.decision_context(
                features["decision_context"]
            )
            import torch

            # Mirrors ``train.py``'s ``ActorCritic.fused``: the challenger's
            # chain encoder is causal and returns its per-step states so the
            # options can cross-attend over the chain prefix, while the
            # ``*_frozen`` generations mean-pool to one tensor and have no
            # ``option_cross``. Whichever module was bundled, this handles it.
            chain = policy.decision_chain(features["decision_chain"])
            chain_state, chain_steps, chain_mask = (
                chain if isinstance(chain, tuple) else (chain, None, None)
            )
            state = policy.fuse(
                torch.cat(
                    [
                        chain_state,
                        ctx_pooled,
                        policy.global_state(features["global_state"]),
                        policy.opponent_history(features["opponent_history"]),
                        policy.player_state(features["state"]),
                        policy.player_state(features["opponent_state"]),
                    ],
                    dim=-1,
                )
            )
            if chain_steps is not None:
                option_vecs = policy.option_cross(
                    option_vecs, options_mask, state, chain_steps, chain_mask
                )
            query = state.unsqueeze(1).expand(-1, option_vecs.size(1), -1)
            logits = policy.score(torch.cat([option_vecs, query], dim=-1)).squeeze(-1)
            return logits, self.stop(state).squeeze(-1), options_mask

    return ActorCritic()


def _greedy_action(logits, stop_logit, options_mask, min_count, max_count,
                   floor=True):
    """The mode of the sequential action distribution the policy was trained on.

    Picks without replacement, consulting the learned ``stop`` logit once
    ``min_count`` picks have been made and forcing it at ``max_count``. This is
    ``train.rollout_action(greedy=True)``; the BC bundles' ``decode_action``
    reads a sigmoid threshold instead and is a different function.
    """
    import torch

    available = options_mask[0].clone()
    min_c, max_c = int(min_count[0]), int(max_count[0])
    stop_column = options_mask.shape[1]
    action: list[int] = []
    while len(action) < max_c:
        masked = logits.masked_fill(~available.unsqueeze(0), _NEG_INF)
        # ``floor`` forbids stopping before the first pick. The learned stop
        # logit declines 15.7% of optional decisions outright (7 of them Hilda,
        # a Supporter — declining burns the once-per-turn slot for nothing),
        # against an expert decline rate of 4.0%. This is decode_action's
        # min_count.clamp(min=1) applied to the sequential decode.
        stop_open = len(action) >= min_c and not (floor and not action)
        stop = stop_logit if stop_open else torch.full_like(stop_logit, _NEG_INF)
        step = torch.cat([masked, stop.unsqueeze(-1)], dim=-1)[0]
        if not torch.isfinite(step).any():
            break
        choice = int(step.argmax())
        if choice == stop_column:
            break
        action.append(choice)
        available[choice] = False
    return action


class _Policy:
    """Lazily-loaded network plus per-episode feature memory.

    Loading is deferred to the first real decision: the harness's first call
    only asks for a decklist, and paying the torch import plus weight load there
    risks a startup timeout for nothing.
    """

    def __init__(self) -> None:
        self.network = None
        self.extractor = None
        self.episode = 0
        self.failed = False

    def reset(self) -> None:
        self.episode += 1
        if self.extractor is not None:
            self.extractor.reset(episode_id=self.episode)

    def _load(self) -> None:
        import torch

        from live import LiveFeatureExtractor

        self.network = _build_actor_critic()
        state = torch.load(_resolve("actor_critic.pt"), map_location="cpu",
                           weights_only=True)
        self.network.load_state_dict(state)
        self.network.eval()
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)
        if self.extractor is None:
            # The observation spec is baked in at build time, not defaulted. This policy was
            # cloned with a 30-frame decision chain and a 30-frame opponent
            # history; serving it under the 60/60 default would feed it a different input
            # than it learned from, which is precisely the failure observation.py exists to
            # prevent -- and it is silent, costing win rate with nothing in the logs.
            from observation import ObservationSpec

            self.extractor = LiveFeatureExtractor(
                spec=ObservationSpec(opponent_history_size=30,
                                     decision_chain_size=30)
            )
            self.extractor.reset(episode_id=self.episode)

    def act(self, obs_dict: dict) -> list[int]:
        import torch

        from collate import collate_features
        from dataset import transform

        selection_counts = _load_policy_module().selection_counts

        if self.network is None:
            self._load()

        observation = self.extractor(obs_dict)
        features = collate_features([transform(observation)])
        with torch.no_grad():
            logits, stop_logit, options_mask = self.network(features)
        min_count, max_count = selection_counts(features)
        action = _greedy_action(logits, stop_logit, options_mask,
                                min_count, max_count, floor=_FLOOR)
        # The chain the *next* decision reads back only lines up if the move
        # actually submitted is the one recorded.
        self.extractor.record_action(action)
        return action


#: Set False to reproduce the unfloored decode for an A/B.
_FLOOR = os.environ.get("SUBMISSION_DECODE_FLOOR", "1") != "0"

_policy = _Policy()


def _fallback(obs: "Observation") -> list[int]:
    count = random.randint(obs.select.minCount, obs.select.maxCount)
    count = min(count, len(obs.select.option))
    return random.sample(range(len(obs.select.option)), count)


def agent(obs_dict: dict) -> list[int]:
    """Return the option indexes to select for this decision.

    Each element is >= 0 and < ``len(obs.select.option)``; the length is within
    ``[minCount, maxCount]`` with no duplicates.
    """
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        # The harness opens a match by asking for the decklist, which is the
        # one unambiguous signal that carried-over episode memory is stale.
        _policy.reset()
        return read_deck_csv()

    if _policy.failed:
        return _fallback(obs)

    try:
        action = _policy.act(obs_dict)
    except Exception as exc:  # noqa: BLE001 — a live match must not crash
        # Latch it: a broken pipeline is broken every decision, and retrying
        # hundreds of times per match risks a timeout on top of the errors.
        print(f"[agent] policy failed ({exc!r}) — falling back", file=sys.stderr)
        _policy.failed = True
        return _fallback(obs)

    if (not (obs.select.minCount <= len(action) <= obs.select.maxCount)
            or len(set(action)) != len(action)):
        print(
            f"[agent] decoded {len(action)} options outside "
            f"[{obs.select.minCount}, {obs.select.maxCount}] — falling back",
            file=sys.stderr,
        )
        return _fallback(obs)
    return action
