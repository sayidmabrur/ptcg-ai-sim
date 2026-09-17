"""Package a trained PPO checkpoint as a competition submission.

    PY=/home/kangh/miniconda3/envs/kaggle-pokemon/bin/python
    $PY make_submission.py --checkpoint checkpoints/ppo-exploit/best.pt
    $PY make_submission.py --validate          # play the bundle against the field

Produces a directory shaped exactly like ``crustle_frozen/``:

    submission/
      cg/                  the engine bindings
      policy_network/      crustle's generation (see below)
      deck.csv             the CHALLENGER decklist
      actor_critic.pt      the trained weights
      main.py              def agent(obs_dict) -> list[int]

Three things here are easy to get wrong and would each produce a bundle that
loads, runs, and plays measurably worse than the checkpoint it came from.

**The decode must be the trained one.** Training and evaluation both decode with
``train.rollout_action(greedy=True)``, which walks a sequence of picks and
consults a *learned* ``stop`` logit. ``policy_experimental.decode_action`` — what
the BC bundles use — instead reads a sigmoid threshold to choose the selection
size. Those are different functions over different parameters. The 51.7% figure
was measured through the former, so the bundle uses the former; scoring a policy
through a decode it was not trained under measures neither.

**The architecture must be crustle's generation.** The run used
``--actor-arch crustle`` because the challenger's ``CardEmbed`` gained ability
fields that widened ``proj`` from 151 to 193 inputs, which is what stopped
``bc_policy.pt`` loading and forced every earlier run to start from random init.
The weights therefore fit ``crustle_frozen/policy_network``, not the
challenger's.

**The deck must be the challenger's.** The policy was trained piloting
``challenger/challenger_deck.csv``. Bundling crustle's list would hand it a
different 60 cards from the ones it learned on.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

MAIN_PY = '''"""``agent()`` entry point for the packaged submission.

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
        lines = file.read().split("\\n")
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
    PolicyNetwork = _policy_module.PolicyNetwork

    # Baked in at package time from the checkpoint's .meta.json. A state dict records
    # no width or depth, and this bundle used to build ``PolicyNetwork()`` from module
    # defaults -- which silently assumes the 2.68M-parameter generation. A checkpoint
    # trained at any other size then fails to load here, at deploy time, having passed
    # every check up to this point. Empty dict = the defaults, so bundles packaged
    # before the architecture was recorded behave exactly as they did.
    ARCH = __ARCH__
    D = ARCH.get("dim", _policy_module.D)

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = PolicyNetwork(**ARCH) if ARCH else PolicyNetwork()
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
            #
            # Current generations expose the whole pass as ``logits_and_state``, which is
            # the only copy of it — see that method for why re-deriving it here was a
            # train/serve bug rather than duplication. The hand-written path below is
            # kept solely for the ``*_frozen`` bundles, whose modules predate it.
            if hasattr(policy, "logits_and_state"):
                logits, state, options_mask = policy.logits_and_state(features)
                return logits, self.stop(state).squeeze(-1), options_mask

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
            # cloned with a __CHAIN__-frame decision chain and a __HISTORY__-frame opponent
            # history; serving it under the 60/60 default would feed it a different input
            # than it learned from, which is precisely the failure observation.py exists to
            # prevent -- and it is silent, costing win rate with nothing in the logs.
            from observation import ObservationSpec

            self.extractor = LiveFeatureExtractor(
                spec=ObservationSpec(opponent_history_size=__HISTORY__,
                                     decision_chain_size=__CHAIN__)
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
'''


def build(checkpoint: Path, out: Path, arch: str, chain: int, history: int) -> dict:
    import torch

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    shutil.copytree(_ROOT / "cg", out / "cg",
                    ignore=shutil.ignore_patterns("__pycache__", "*Zone.Identifier"))
    # "challenger" is this repo's own generation and lives outside the *_frozen bundles.
    source = (_ROOT / "challenger" / "policy_network" if arch == "challenger"
              else _ROOT / f"{arch}_frozen" / "policy_network")
    shutil.copytree(source, out / "policy_network",
                    ignore=shutil.ignore_patterns("__pycache__", "*Zone.Identifier"))
    shutil.copy(_ROOT / "challenger" / "challenger_deck.csv", out / "deck.csv")

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    bc = "model" not in state
    if bc:
        # A behavioural-cloning checkpoint is a bare PolicyNetwork state dict: no value head
        # and, more importantly, no ``stop`` logit -- that head is an RL-side addition. The
        # bundle's decode consults it on every optional and multi-select decision, so leaving
        # it randomly initialised would corrupt exactly those decisions.
        #
        # -4.0 is not a guess: it is what ``train.py`` sets on any warm start, so it is the
        # configuration the win rate quoted below was actually measured through. Packaging it
        # any other way would ship a policy that was never evaluated.
        weights = {f"policy.{key}": value for key, value in state.items()}
        width = weights["policy.fuse.0.weight"].shape[0]
        weights["stop.weight"] = torch.zeros(1, width)
        weights["stop.bias"] = torch.full((1,), -4.0)
        weights["value.0.weight"] = torch.zeros(width, width)
        weights["value.0.bias"] = torch.zeros(width)
        weights["value.2.weight"] = torch.zeros(1, width)
        weights["value.2.bias"] = torch.zeros(1)
    else:
        weights = state["model"]
    torch.save(weights, out / "actor_critic.pt")

    # The architecture the weights were trained at, from the sidecar bc_train writes.
    # Falling back to {} reproduces the old behaviour (module defaults) for any
    # checkpoint predating that field, which is correct: those are all the 64-wide
    # generation. Sanity-check it against a tensor that is actually in the file, so a
    # stale or hand-edited meta.json cannot ship a bundle that will not load.
    architecture: dict = {}
    meta_path = Path(str(checkpoint) + ".meta.json")
    if meta_path.exists():
        architecture = json.loads(meta_path.read_text()).get("arch") or {}
    if architecture:
        recorded, actual = architecture.get("dim"), weights["policy.fuse.0.weight"].shape[0]
        if recorded != actual:
            raise SystemExit(f"{meta_path.name} says dim={recorded} but the checkpoint's "
                             f"fuse layer is {actual} wide; refusing to package it")

    (out / "main.py").write_text(
        MAIN_PY.replace("__CHAIN__", str(chain))
        .replace("__HISTORY__", str(history))
        .replace("__ARCH__", repr(architecture))
    )

    meta = {
        "checkpoint": str(checkpoint),
        "source": "behavioural cloning" if bc else "PPO",
        "episode": state.get("episode") if not bc else None,
        "update": state.get("update") if not bc else None,
        "best_eval": state.get("best_eval") if not bc else None,
        "actor_arch": arch,
        "policy_arch": architecture or "module defaults (dim=64)",
        "parameters": sum(v.numel() for v in weights.values()),
        "decision_chain_size": chain,
        "opponent_history_size": history,
        "decode": "train.rollout_action(greedy=True) — stop logit"
                  + (" biased to -4.0 (no RL stop head in a BC checkpoint)" if bc else ""),
    }
    (out / "SUBMISSION.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


def validate(out: Path, games: int, workers: int) -> None:
    """Play the *bundle itself* against the frozen field.

    Not the checkpoint — the bundle. Packaging is where a wrong decode, a wrong
    decklist or a wrong architecture silently costs win rate, and every one of
    those produces a submission that loads and runs perfectly well.
    """
    sys.path.insert(0, str(_ROOT))
    from arena import versus_field

    print(f"\n[validate] playing {out.name} against the frozen field")
    versus_field(f"bundle:{out}", games, workers, {"max_steps": 500, "seed": 0}, 0)


def build_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default=str(_ROOT / "checkpoints/ppo-exploit/best.pt"))
    parser.add_argument("--arch", default="crustle",
                        help="policy generation the weights fit: a *_frozen "
                             "bundle name, or 'challenger' for this repo's own")
    parser.add_argument("--chain", type=int, default=60,
                        help="decision_chain_size the policy was TRAINED with; "
                             "baked into the bundle. BC here used 30")
    parser.add_argument("--history", type=int, default=60,
                        help="opponent_history_size the policy was TRAINED with")
    parser.add_argument("--out", default=str(_ROOT / "submission_ppo"))
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--games", type=int, default=120, help="per opponent")
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args(argv)


def main() -> None:
    args = build_args()
    out = Path(args.out)
    meta = build(Path(args.checkpoint), out, args.arch, args.chain, args.history)
    print(f"[submission] {out}")
    for key, value in meta.items():
        print(f"  {key}: {value}")
    print(f"  files: {sorted(p.name for p in out.iterdir())}")
    if args.validate:
        validate(out, args.games, args.workers)


if __name__ == "__main__":
    main()
