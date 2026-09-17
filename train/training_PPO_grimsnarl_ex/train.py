"""PPO for the challenger policy against three frozen behavioral-cloning
opponents — ``greedy-crustle-PPO``.

Run it with the environment that actually has a working torch + wandb::

    PY=/home/kangh/miniconda3/envs/kaggle-pokemon/bin/python
    $PY train.py --episodes 30000 --workers 6 --episodes-per-update 24

The base conda interpreter segfaults on ``import torch`` and has no wandb, so
the ``kaggle-pokemon`` env is not a preference here, it is the requirement.

Wall-clock, measured on this box (12 cores, RTX 5060 Ti)
-------------------------------------------------------
A rollout decision costs ~30 ms and a game runs 50-130 decisions, which is
what sets the schedule for a 30k-episode run:

===========================  ===========  =========================
setting                       ep/hour      30k episodes
===========================  ===========  =========================
``--rollout-device cuda``      ~410        ~73 h
``--workers 1`` (CPU)          ~1230       ~24 h
``--workers 6``                ~1530       ~20 h
===========================  ===========  =========================

The GPU *losing* to the CPU by 2.5x is not a misconfiguration: every rollout
forward is batch 1 over a 2.4M-parameter model, ~390 module calls deep, so it
is bound by Python and kernel-launch latency rather than arithmetic. Hence the
split defaults — rollouts on CPU, the PPO update (batches of 64) on the GPU.

Workers buy less than their count because the rollout stops being the whole
cost once it is parallel. Per batch of 24 episodes: ~13 s of rollout across 6
workers, ~15 s of parent-side feature rebuild (overlapped with the rollout, see
``RolloutPool.collect``), and then a serial PPO update whose two halves measure
~290 ms of feature collation and ~400 ms of forward+backward per minibatch of
64. At ``--ppo-epochs 4`` that update is the largest single term, and it is not
parallelisable against anything. ``--ppo-epochs 2`` roughly halves it if you
want throughput more than sample reuse; caching the collated minibatches across
epochs would cut the other half, but a whole batch collated at once is 2-4 GB,
which is why it is re-collated per minibatch instead.

``--workers 1`` is the simplest path and the one the smoke tests exercise most;
``--workers N`` is the same rollout in N processes.

Two rollout backends
--------------------
``--workers N`` forks N processes each holding one battle, because ``cg.game``
keeps the battle pointer on a class attribute and so a process can hold exactly
one. ``--num-envs V`` instead runs V battles *inside* this process on
pokeka2026's ``PTCGBattle``, which owns its pointer explicitly, and collates the
V decisions waiting at any instant into one forward (see ``pokeka_env`` and
``vec_rollout``). That removes the batch-1 constraint the paragraph above is
about, and with it the worker IPC and the parent-side feature rebuild.

Measured end to end on this box, 256 episodes at 64 per update, ``--ppo-epochs
2 --minibatch-size 128 --target-kl 0.03``:

===========================  ===========  ==============================
setting                       ep/hour      note
===========================  ===========  ==============================
``--workers 11``               ~3120       11 cores of rollout
``--num-envs 32``             ~1440        1.5 cores, GPU-batched
===========================  ===========  ==============================

**The GPU is not the lever, and vectorising is not either.** Batching does what
it promised — inference falls from 7.6 ms a decision to ~0.3 ms — but inference
was never the cost. ``dataset.transform`` is 6.4 ms per decision and builds ~990
individual tensors doing it, because a decision's ``decision_chain`` re-featurises
up to 60 earlier decisions, so the feature build is quadratic in episode length
and single-threaded Python. That is 87% of a rollout decision and it is
unaffected by any amount of batch. Throughput therefore tracks *cores*, which is
why 11 one-battle processes beat 32 batched battles by 2.2x. pokeka's own compute
plan reached the same conclusion from the other side ("GPU is NOT the lever for
this workload"; 48 envs over 8 workers bought 1.63x, Amdahl-capped by the CPU
update).

So ``--workers`` stays the throughput setting. ``--num-envs`` is the right mode
when a single process is wanted rather than twelve, when the GPU is the only
thing available, or to keep rollout and update numerically identical (it drops
the CPU actor replica, so a PPO ratio is no longer computed across two devices).
The two are mutually exclusive, and combining them would not help: at
``--workers 11`` the rollout already finishes in ~13 s of a ~66 s update cycle,
hidden behind the ~38 s the parent spends rebuilding features, so making the
rollout cheaper changes nothing. The wall is that rebuild plus the ~22 s of
collation in the update — both of them the same ``transform`` cost, paid in the
one process that cannot be parallelised away without either shipping 45 MB an
episode between processes (measured: 147 MB/s over shared memory, no faster than
rebuilding) or letting the update run one batch stale.

What this trains
----------------
``challenger/`` holds a ``PolicyNetwork`` with **no checkpoint** — it starts
from random init, deliberately (see ``--init-from`` if that changes). It is the
only thing that learns. ``alakazam_frozen/``, ``crustle_frozen/`` and
``lucario_frozen/`` are BC policies loaded once, put in ``eval()``, kept under
``no_grad``, and never touched by the optimizer. They are the curriculum, not
participants: each episode draws one of them round-robin, and the challenger
alternates seats so a win rate is not partly a measurement of the first-player
advantage.

They play **greedily** — ``decode_action`` is argmax on the 93.6% of decisions
that are single-select — which makes the opponent distribution stationary. That
is what "greedy strategy" means here; the challenger itself *samples* during
rollouts, because a deterministic policy has nothing for a policy-gradient
ratio to push on.

Reward
------
Pure outcome: ``+1`` win, ``-1`` loss, ``0`` draw, delivered once at the
terminal decision. No shaping of any kind — no prize-card bonus, no damage
credit, no step penalty. With ``--gamma 1.0 --gae-lambda 1.0`` (the defaults)
GAE collapses exactly to ``advantage_t = R - V(s_t)``, i.e. every decision in a
won game is reinforced equally. That is the intended experiment, and it is also
the reason 30k episodes may well *not* converge from a random init: the credit
assignment problem is ~100 decisions wide with one bit of signal per episode.
The point of the run is to find that out, so nothing here quietly softens it.

One shared module set
---------------------
``challenger/policy_network`` goes on ``sys.path`` and *all four* policies are
built from it. This is safe because the four copies are byte-identical apart
from docstrings and one thing: ``crustle_frozen``'s ``decode_action`` floors the
selection count at one where the others floor it at ``min_count``. That floor is
reproduced for every opponent by passing ``min_count.clamp(min=1)`` (see
``greedy_action``) — algebraically the same function, and it is the later,
measured-better decode. Importing four copies under bare module names
(``features``, ``vocab``, ...) would collide in ``sys.modules`` anyway; the
vocab sizes are read live from the engine, so the architectures match exactly.

Sequential decisions
--------------------
The engine's action space is not fixed: each decision offers N options and a
``[minCount, maxCount]`` bracket. The policy is factored as a sequence of picks
without replacement, plus a learned ``stop`` logit that becomes available once
``minCount`` picks have been made and is forced at ``maxCount``. For the
single-select majority (``min == max == 1``) this reduces to exactly one
categorical over the options — stop is unavailable at step 0 and forced at step
1, so it contributes no term. For optional and multi-select decisions it means
the *size* of the selection is learned inside the same distribution rather than
read off an uncalibrated probability threshold, so the log-probability PPO
ratios are computed from is the real one.
"""

import argparse
import collections
import importlib.util
import io
import json
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any

# Emitted once per ``TransformerEncoder`` construction — four networks per
# process, times every rollout worker. It reports a fast-path being skipped
# because the frozen architecture uses ``norm_first``, which is not something
# this file can or should change.
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")

import torch  # noqa: E402
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent
# The policy modules import each other by bare name (``from features import
# ...``), so their directory itself has to be importable, not just the repo
# root. One copy for all four policies — see the module docstring.
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "challenger" / "policy_network"))

from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

from collate import collate_features  # noqa: E402
from dataset import transform  # noqa: E402
from live import LiveFeatureExtractor  # noqa: E402
from observation import DEFAULT_SPEC, LEARNER_SPEC, build_observation  # noqa: E402
from policy_experimental import (  # noqa: E402
    D,
    PolicyNetwork,
    decode_action,
    selection_counts,
)


_frozen_modules: dict[str, Any] = {}


def frozen_network(name: str, root: Path) -> "torch.nn.Module":
    """An untrained ``PolicyNetwork`` built from *this* frozen agent's own copy
    of ``policy_experimental``, ready for its ``bc_policy.pt``.

    ``challenger/policy_network`` has diverged: its ``CardEmbed`` now consumes
    the ``skill_count``/``ability_id``/``skill_flags`` fields, which widens
    ``CardEmbed.proj`` and adds an embedding table. The ``bc_policy.pt`` files
    were trained before that and no longer fit the challenger's class, so each
    frozen agent is constructed from its own directory rather than from a
    shared one — the three copies turn out not to be one generation
    (``crustle``/``alakazam`` predate the ``dropout`` parameter that
    ``lucario`` and the challenger have), and a per-directory load is correct
    whichever generation each one is.

    Only the class is taken. Its ``from vocab import ...`` resolves to the
    challenger's already-imported ``vocab``, which is deliberate: the ability
    work only *added* names, so every vocab size the older class reads is
    unchanged. The features need no second pipeline either — the old
    ``CardEmbed`` reads its inputs by name with zero-fallbacks and ignores keys
    it does not know, so it consumes the extended feature tree unchanged while
    simply not seeing the ability fields.

    Constructed with no arguments on purpose: ``dropout`` is not in every
    generation's signature, and it would be inert regardless since these
    networks never leave ``eval()`` and are never trained.
    """
    if name not in _frozen_modules:
        source = root / "policy_network" / "policy_experimental.py"
        spec = importlib.util.spec_from_file_location(f"frozen_{name}_policy", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _frozen_modules[name] = module
    return _frozen_modules[name].PolicyNetwork()

#: Frozen opponents, in the order they are cycled. Each contributes its own
#: decklist as well as its weights: a BC policy piloting a list it never saw
#: the expert play is being handed cards it cannot use, which would make it a
#: weaker opponent for reasons unrelated to the challenger.
OPPONENTS = ("crustle", "alakazam", "lucario")

_NEG_INF = float("-inf")


# --------------------------------------------------------------------------
# Feature-tree plumbing
#
# ``transform()`` returns a nested dict of tensors (dicts, lists of dicts,
# tensors). Both moving it to a device and stacking a batch of them need to
# walk that tree; ``collate_features`` already does the stacking, so only the
# device walk lives here.

def tree_to(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: tree_to(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_to(item, device) for item in value]
    return value


def batch_features(samples, device):
    """Collate a list of ``transform()`` outputs and move it to ``device``."""
    return tree_to(collate_features(samples), device)


# --------------------------------------------------------------------------
# Network


class ActorCritic(nn.Module):
    """The challenger: a ``PolicyNetwork`` actor plus a value head and a stop
    logit, all reading the same fused state vector.

    ``PolicyNetwork.forward`` returns only the option logits, so the fused
    state it computes internally is re-derived here from its own submodules
    rather than by editing the frozen-shared module. The duplication is the
    ``fused`` body below — deliberately kept to that, so the actor's weights
    remain a plain ``PolicyNetwork`` state dict and stay loadable by
    ``load_policy`` (and hence by ``main.py``) once training is done.
    """

    def __init__(self, dropout: float = 0.0, arch: str = "challenger",
                 value_prior: float = 0.0) -> None:
        super().__init__()
        # ``arch="crustle"`` builds the actor from *crustle's* PolicyNetwork
        # instead of the challenger's. The two differ only in that the
        # challenger's ``CardEmbed`` consumes the newer ability/skill fields,
        # which widened ``proj`` and made ``bc_policy.pt`` unloadable here — so
        # ``--init-from`` has never actually worked and every run in this repo
        # started from random init.
        #
        # That mattered more than it looked. Measured over 1,796 games, the
        # crustle BC checkpoint piloting the challenger deck scores 41.1%
        # against the frozen field, while the best PPO run — 30k episodes, ~8
        # hours, from scratch — reached 28.3%. The training was starting from
        # zero and finishing below a checkpoint already in the repo. Giving up
        # the ability features to gain a working warm start is plainly the
        # better trade.
        if arch == "challenger":
            self.policy = PolicyNetwork(dropout=dropout)
        else:
            self.policy = frozen_network(arch, _ROOT / f"{arch}_frozen")
        self.value = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1))
        # Zero the critic's output layer so it predicts exactly 0 at init.
        # ``advantage = R - V(s)`` then starts as the return itself: unbiased,
        # correctly scaled, and independent of whatever a random head would have
        # emitted. It matters most for a warm start, where a random critic's
        # first few updates push a *good* policy in a direction uncorrelated
        # with anything — the smoke test showed value loss 3.5 and explained
        # variance -2.4 on update 1, which is the critic being confidently wrong
        # while the policy is already competent.
        nn.init.zeros_(self.value[-1].weight)
        nn.init.constant_(self.value[-1].bias, value_prior)
        # A single state-conditioned scalar. It does not see the picks made so
        # far, so "stop after k" is modelled with one logit competing against
        # the remaining options at every step rather than a per-step decision.
        # That is enough to learn a selection size and costs nothing on the
        # single-select majority, where stop is never a live choice.
        self.stop = nn.Linear(D, 1)
        if arch != "challenger":
            # A warm-started actor has a trained option head and an untrained
            # stop logit competing inside the same softmax. Left at random
            # init, stop wins often enough to corrupt every optional and
            # multi-select decision from the first rollout. A strongly negative
            # bias makes "keep selecting" the default, which reproduces
            # ``decode_action``'s floor-at-one behaviour that the BC policy was
            # measured under.
            nn.init.zeros_(self.stop.weight)
            nn.init.constant_(self.stop.bias, -4.0)

    def fused(self, features: dict):
        """``(state (B, D), option_vecs (B, N, D), options_mask (B, N))``.

        This mirrors ``PolicyNetwork.forward`` up to the scoring head, and has
        to span both actor architectures ``--actor-arch`` can select. The
        challenger's ``DecisionChainEncoder`` runs causally and returns its
        per-step states alongside the readout, so the current decision's
        options can cross-attend over the chain prefix; the ``*_frozen``
        copies are the older encoder, which mean-pools and returns one tensor
        and has no ``option_cross``. Both are dispatched on below rather than
        forked into two methods, so a warm start from a frozen checkpoint
        (measured 41.1% against the field, vs 28.3% for PPO from scratch —
        see ``__init__``) keeps working.

        Current challenger generations expose this as ``PolicyNetwork.encode``, which is
        the single copy of the pass — this method's hand-written version silently
        omitted ``_address_board``, so PPO on a ``board_tokens`` actor was training with
        the board addressing disabled. The fallback below remains for the ``*_frozen``
        modules, which predate ``encode``.
        """
        policy = self.policy
        if hasattr(policy, "encode"):
            return policy.encode(features)
        ctx_pooled, option_vecs, options_mask = policy.decision_context(
            features["decision_context"]
        )
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
        return state, option_vecs, options_mask

    def forward(self, features: dict):
        """``(option_logits (B, N), stop_logit (B,), value (B,), mask (B, N))``.

        ``option_logits`` keeps masked positions finite here (unlike
        ``PolicyNetwork.forward``, which fills them with ``-inf``): the
        sequence log-probability below masks per step anyway, and ``-inf``
        entries poison the autograd graph the moment a batch row has a step
        where everything is masked out.
        """
        state, option_vecs, options_mask = self.fused(features)
        if hasattr(self.policy, "score_options"):
            # Unmasked by contract, which is exactly what this docstring requires —
            # and it carries the type-conditioned head, which scoring by hand does not.
            logits = self.policy.score_options(option_vecs, state, features)
        else:
            query = state.unsqueeze(1).expand(-1, option_vecs.size(1), -1)
            logits = self.policy.score(
                torch.cat([option_vecs, query], dim=-1)
            ).squeeze(-1)
        return logits, self.stop(state).squeeze(-1), self.value(state).squeeze(-1), options_mask


# --------------------------------------------------------------------------
# The sequential action distribution
#
# An action is an *ordered* list of option indexes. Its probability is the
# product of the per-step categoricals over (still-available options + stop,
# where stop is legal). Treating the order as part of the action — rather than
# marginalising over the orders that produce the same set — is what makes the
# log-probability below exact, which is all PPO needs.


def _step_logits(logits, stop_logit, available, stop_open):
    """``(B, N + 1)`` — this step's choices, stop in the last column.

    ``stop_open`` is per-row: a row that has not yet reached ``min_count``
    cannot stop, and a row that has reached ``max_count`` never gets here (the
    stop is forced, contributing probability 1 and no log-prob term).
    """
    masked = logits.masked_fill(~available, _NEG_INF)
    stop = torch.where(stop_open, stop_logit, torch.full_like(stop_logit, _NEG_INF))
    return torch.cat([masked, stop.unsqueeze(-1)], dim=-1)


def _stop_open(step, min_count, max_count):
    """Per-row: is stopping a legal choice at this step?"""
    return (torch.full_like(min_count, step) >= min_count) & (
        torch.full_like(max_count, step) < max_count
    )


def action_log_prob(logits, stop_logit, options_mask, min_count, max_count, actions):
    """Log-probability and first-step entropy of a batch of stored actions.

    Vectorised across the batch, looping only over *step index* — which is 1
    for the ~94% of decisions that are single-select, so the loop is short in
    practice even though ``max_count`` can reach 60.

    Returns ``(log_prob (B,), entropy (B,))``. The entropy is the first step's
    categorical entropy, used as the exploration bonus: the later steps of a
    multi-select share the same state and add little, and computing the exact
    sequence entropy would need a rollout of its own.
    """
    batch = logits.shape[0]
    device = logits.device
    available = options_mask.clone()
    log_prob = logits.new_zeros(batch)
    entropy = logits.new_zeros(batch)
    lengths = torch.tensor([len(a) for a in actions], device=device)
    row_index = torch.arange(batch, device=device)

    longest = int(lengths.max().item()) if batch else 0
    for step in range(longest):
        taken = lengths > step
        # Stop is offered once ``step`` picks are already made and the bracket
        # still allows more; at ``step == max_count`` the loop is over anyway.
        #
        # ``| ~taken`` is not cosmetic. The loop runs to the batch's longest
        # action, so a short row is still evaluated at steps it never took, and
        # a row that exhausted its options (num_valid == len(action)) with stop
        # already illegal would have *every* column at -inf. ``log_softmax`` of
        # that row is NaN, and while the forward value is discarded by the
        # ``torch.where`` below, its backward still pushes NaN into the shared
        # logits. Leaving stop open for non-participating rows keeps every row
        # finite; their log-prob is masked out either way.
        stop_open = _stop_open(step, min_count, max_count) | ~taken
        step_logits = _step_logits(logits, stop_logit, available, stop_open)
        log_probs = F.log_softmax(step_logits, dim=-1)
        if step == 0:
            probs = log_probs.exp()
            entropy = -(probs * log_probs.masked_fill(probs == 0, 0.0)).sum(-1)
        chosen = torch.tensor(
            [a[step] if len(a) > step else 0 for a in actions], device=device
        )
        log_prob = log_prob + torch.where(
            taken, log_probs[row_index, chosen], torch.zeros_like(log_prob)
        )
        # Sampling is without replacement, and only for rows that actually
        # picked at this step.
        available = available & ~(
            F.one_hot(chosen, options_mask.shape[1]).bool() & taken.unsqueeze(-1)
        )

    # A selection shorter than ``max_count`` ended by *choosing* to stop, and
    # that choice carries probability too. One at ``max_count`` was terminated
    # by the bracket, so it does not.
    explicit_stop = (lengths < max_count) & (lengths >= min_count)
    if explicit_stop.any():
        step_logits = _step_logits(
            logits, stop_logit, available, torch.ones_like(explicit_stop)
        )
        log_probs = F.log_softmax(step_logits, dim=-1)
        log_prob = log_prob + torch.where(
            explicit_stop, log_probs[:, -1], torch.zeros_like(log_prob)
        )
    return log_prob, entropy


@torch.no_grad()
def rollout_action(logits, stop_logit, options_mask, min_count, max_count,
                   greedy=False, floor=False):
    """Draw one action from the same distribution ``action_log_prob`` scores.

    ``greedy=True`` takes the per-step argmax instead — the *mode* of this
    distribution, which is what an evaluation of the trained policy should
    play. Note it is deliberately not ``policy_experimental.decode_action``:
    that decode reads a sigmoid threshold to pick the selection size, a
    quantity this policy learns explicitly through the stop logit. Scoring the
    policy through a different decode than the one it was trained under
    measures neither.

    Single-row only — rollouts are inherently sequential here, because the
    engine keeps one global battle pointer (``cg.sim.Battle``) and so cannot be
    vectorised into parallel environments.
    """
    available = options_mask[0].clone()
    min_c, max_c = int(min_count[0]), int(max_count[0])
    stop_column = options_mask.shape[1]
    action: list[int] = []
    log_prob = 0.0
    while len(action) < max_c:
        # ``floor`` forbids stopping before the first pick, reproducing
        # ``decode_action``'s ``min_count.clamp(min=1)``. Measured on the
        # crustle replays, the unfloored decode declined 29.0% of optional
        # decisions against an expert rate of 4.0%, and flooring lifted
        # exact-match on that subset from 61.0% to 82.0%. The behavioural audit
        # found the learned stop logit reproducing the same fault: 15.7% of
        # ``minCount == 0`` decisions returned an empty selection, 7 of them on
        # Hilda — a Supporter, so declining it burns the once-per-turn slot for
        # nothing.
        stop_open = len(action) >= min_c and not (floor and not action)
        step = _step_logits(
            logits, stop_logit, available.unsqueeze(0),
            torch.tensor([stop_open], device=logits.device),
        )[0]
        if not torch.isfinite(step).any():
            # Every option consumed while the bracket still demands more. The
            # engine cannot offer fewer options than min_count, so this is
            # unreachable — but a rollout that hit it would otherwise sample
            # from an all-NaN distribution rather than fail visibly.
            break
        log_probs = F.log_softmax(step, dim=-1)
        choice = (
            int(log_probs.argmax()) if greedy
            else int(torch.multinomial(log_probs.exp(), 1))
        )
        log_prob += float(log_probs[choice])
        if choice == stop_column:
            break
        action.append(choice)
        available[choice] = False
    return action, log_prob


def greedy_action(logits, options_mask, min_count, max_count):
    """The frozen opponents' decode: ``policy_experimental.decode_action``.

    ``min_count.clamp(min=1)`` reproduces ``crustle_frozen``'s floored variant
    exactly — that copy's only functional difference from the others, and the
    measured-better one (82% vs 61% exact match on optional decisions). Since
    ``decode_action`` uses ``min_count`` for nothing else, passing the clamped
    value through the unfloored implementation *is* the floored implementation.
    """
    return decode_action(
        logits, options_mask, min_count.clamp(min=1), max_count
    )[0]


# --------------------------------------------------------------------------
# Policies


def read_deck(path: Path) -> list[int]:
    lines = [line for line in path.read_text().split("\n") if line.strip()]
    if len(lines) != 60:
        raise SystemExit(f"{path} has {len(lines)} cards; the engine requires exactly 60")
    return [int(line) for line in lines]


class FrozenPolicy:
    """A BC checkpoint playing greedily, with its own per-episode feature
    memory. Never trained, never in the optimizer, never out of ``eval()``."""

    def __init__(self, name: str, root: Path, device) -> None:
        self.name = name
        self.deck = read_deck(root / "deck.csv")
        # This agent's own network class, not the challenger's — see frozen_network.
        self.network = frozen_network(name, root)
        self.network.load_state_dict(
            torch.load(root / "bc_policy.pt", map_location="cpu", weights_only=True)
        )
        self.network.eval().to(device)
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.extractor = LiveFeatureExtractor()

    def reset(self, episode: int) -> None:
        self.extractor.reset(episode_id=episode)

    @torch.no_grad()
    def act(self, obs: dict) -> list[int]:
        observation = self.extractor(obs)
        features = batch_features([transform(observation)], self.device)
        logits = self.network(features)
        min_count, max_count = selection_counts(features)
        options_mask = features["decision_context"]["options"]["options_mask"].squeeze(1)
        action = greedy_action(logits, options_mask, min_count, max_count)
        self.extractor.record_action(action)
        return action


class ChallengerPolicy:
    """The learner. Samples during rollouts and records everything PPO needs
    for each of its own decisions — the opponent's decisions are never stored,
    since they are not this policy's actions."""

    def __init__(
        self, model: ActorCritic, device, deck: list[int], keep_samples: bool = True
    ) -> None:
        self.model = model
        self.device = device
        self.deck = deck
        # The learner sees a 30-frame chain/history; the frozen agents keep the 60 they were
        # fitted at, since they are the yardstick every number in FINDINGS.md is measured
        # against and truncating their input would move the benchmark, not the policy.
        self.extractor = LiveFeatureExtractor(spec=LEARNER_SPEC)
        self.steps: list[dict] = []
        self.greedy = False
        self.attack_shaping = 0.0
        # A rollout worker ships raw decision *rows* back to the parent, which
        # rebuilds the feature trees there — so it has no use for its own copy
        # and holding them would be pure memory. See ``RolloutPool``.
        self.keep_samples = keep_samples

    def reset(self, episode: int) -> None:
        self.extractor.reset(episode_id=episode)

    def act(self, obs: dict) -> list[int]:
        observation = self.extractor(obs)
        sample = transform(observation)
        features = batch_features([sample], self.device)
        with torch.no_grad():
            logits, stop_logit, value, options_mask = self.model(features)
        min_count, max_count = selection_counts(features)
        min_count, max_count = min_count.to(self.device), max_count.to(self.device)

        action, log_prob = rollout_action(
            logits, stop_logit, options_mask, min_count, max_count, greedy=self.greedy
        )

        self.extractor.record_action(action)
        if not self.greedy:
            # ``sample`` is kept on CPU: a rollout is thousands of these and
            # they are only needed again, in batches, at update time.
            self.steps.append(
                {
                    "sample": sample if self.keep_samples else None,
                    "action": action,
                    "log_prob": log_prob,
                    "value": float(value[0]),
                    # ``yourIndex`` is this policy's own seat here, since ``act``
                    # is only ever called when it is our turn to choose.
                    "potential": state_potential(
                        obs, obs["current"]["yourIndex"], self.attack_shaping
                    ),
                }
            )
        return action


class RandomPolicy:
    """Legal-move control. Without it, "the challenger wins 34% of the time"
    has no floor to be read against — and from a random init the challenger
    starts out *as* this, so the two numbers together say whether anything has
    been learned at all."""

    def __init__(self, deck: list[int]) -> None:
        self.deck = deck
        # ``play_episode`` slices this on whichever policy it is handed, so the
        # baseline needs it even though it never records anything.
        self.steps: list[dict] = []
        self.greedy = True

    def reset(self, episode: int) -> None:
        pass

    def act(self, obs: dict) -> list[int]:
        select = obs["select"]
        count = min(
            random.randint(select["minCount"], select["maxCount"]), len(select["option"])
        )
        return random.sample(range(len(select["option"])), count)


# --------------------------------------------------------------------------
# Rollout


def play_episode(challenger, opponent, seat: int, episode: int, max_steps: int,
                 attack_shaping: float = 0.0):
    """One battle. Returns ``(result, steps, challenger_steps, terminal_potential)``.

    ``terminal_potential`` is ``prize_potential`` of the board *after* the game
    ended, which shaping needs as the ``Φ(s')`` of the final transition — it is
    not obtainable from the decision frames, since by definition no decision is
    offered there.

    ``result`` is the engine's winning player index (``2`` for a draw), or
    ``None`` if the episode was abandoned — an engine error or the step cap.
    An abandoned episode's transitions are dropped rather than labelled with a
    guessed outcome, because a fabricated reward is worse than a smaller batch.
    """
    decks = [None, None]
    decks[seat] = challenger.deck
    decks[1 - seat] = opponent.deck
    policies = [None, None]
    policies[seat] = challenger
    policies[1 - seat] = opponent

    before = len(challenger.steps)
    obs, _ = battle_start(decks[0], decks[1])
    if obs is None:
        return None, 0, 0, 0.0
    challenger.reset(episode)
    opponent.reset(episode)
    steps = 0
    terminal_potential = 0.0
    try:
        while obs["current"]["result"] == -1:
            player_index = obs["current"]["yourIndex"]
            obs = battle_select(policies[player_index].act(obs))
            steps += 1
            if steps >= max_steps:
                return None, steps, len(challenger.steps) - before, 0.0
        result = obs["current"]["result"]
        terminal_potential = state_potential(obs, seat, attack_shaping)
    except (ValueError, IndexError, RuntimeError) as error:
        print(f"[episode {episode}] abandoned: {error!r}", file=sys.stderr)
        result = None
    finally:
        battle_finish()
    return result, steps, len(challenger.steps) - before, terminal_potential


# --------------------------------------------------------------------------
# Parallel rollouts
#
# The engine is a singleton per process — ``cg.sim`` keeps one global battle
# pointer — so rollouts cannot be batched into a vector env. They can be
# *processed* in parallel, and measurement says that is where the wall-clock
# is: a rollout forward is batch 1, so it is Python/launch-bound (~30 ms per
# decision, unchanged whether torch gets 1 thread or 4), which means N worker
# processes scale nearly linearly on this 12-core box.
#
# What the workers ship back is the thing that makes this worth doing. The
# obvious payload — the ``transform()`` feature trees the worker already built
# — measures 45 MB and 4.9 s of pickle per episode against 3.3 s to *play* it:
# parallelism would have cost more than it bought. The same episode's raw
# decision rows are 0.10 MB, and the parent rebuilds features from them in
# 0.62 s. So rows go over the wire and the parent rebuilds.
#
# That rebuild is exact, not an approximation: ``opponent_history`` and
# ``decision_chain`` both scan strictly backward from the current row index
# (``cursor = idx - 1``), so a row's features cannot see its own
# ``target_action`` or any later row — which is also why ``dataset.py`` can
# reconstruct live features from a finished replay at all.


def rebuild_steps(payload: dict) -> list[dict]:
    """Turn a worker's shipped rows back into PPO transitions.

    ``log_prob``, ``value`` and ``potential`` come from the worker's own rollout
    rather than being recomputed here — they must be the numbers the *acting*
    policy produced and the states it actually saw, which is what makes them the
    "old" policy in the PPO ratio.
    """
    rows = payload["rows"]
    log_probs, values = payload["log_probs"], payload["values"]
    potentials = payload["potentials"]
    if not (len(rows) == len(log_probs) == len(values) == len(potentials)):
        raise RuntimeError(
            f"worker shipped {len(rows)} rows for {len(log_probs)} decisions — "
            "the challenger's extractor should hold exactly its own frames"
        )
    read_row = rows.__getitem__
    steps = []
    for index, row in enumerate(rows):
        steps.append(
            {
                "sample": transform(build_observation(read_row, index, row, DEFAULT_SPEC)),
                "action": list(row["target_action"]),
                "log_prob": log_probs[index],
                "value": values[index],
                "potential": potentials[index],
            }
        )
    return steps


def _dump_state(state: dict) -> bytes:
    """A state dict as plain bytes, for shipping over a multiprocessing queue.

    ``queue.put`` on a dict of *tensors* goes through torch's shared-memory reducer, which
    turns every storage into a duplicated file descriptor. This model has ~200 of them, so
    one broadcast to 11 workers asks for ~2,200 fds against the default ``ulimit -n`` of
    1024 — and what fails is the queue's *feeder thread*, so the parent does not crash: it
    prints a pile of ``OSError: [Errno 24] Too many open files`` tracebacks and then waits
    the full ``--worker-timeout`` for results from workers that never received weights.
    Measured on this box: soft limit 1024, 1013 fds open at the first broadcast.

    Bytes cost one ~10 MB pipe write per worker per update — a few tens of milliseconds
    against a ~66 s update cycle — and have no such failure mode. ``file_system`` sharing
    would also avoid the fds, but it leaves a shm file per storage behind when a worker is
    killed, which is the orphaned-``Pool`` trap in FINDINGS.md wearing a different hat.
    """
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def _load_state(payload: bytes) -> dict:
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)


def _rollout_worker(rank: int, task_queue, result_queue, args) -> None:
    """One rollout process: own engine, own frozen opponents, own actor copy."""
    torch.set_num_threads(1)  # batch-1 inference gains nothing from more
    seed = args.seed * 10_000 + rank + 1
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")

    deck = read_deck(_ROOT / "challenger" / "challenger_deck.csv")
    opponents = [FrozenPolicy(name, _ROOT / f"{name}_frozen", device) for name in OPPONENTS]
    model = ActorCritic(dropout=args.dropout, arch=args.actor_arch,
                            value_prior=value_prior(args)).to(device).eval()
    challenger = ChallengerPolicy(model, device, deck, keep_samples=False)
    challenger.attack_shaping = args.attack_shaping

    # One reusable network for self-play, with the sampled snapshot's weights
    # loaded into it per episode. Holding every snapshot as its own network
    # would cost 10 MB each per worker; a state-dict load is a few milliseconds
    # against a ~3 s episode, so this trades nothing that matters. The pool is
    # kept in append order so an index chosen by the parent names the same
    # snapshot here.
    snapshots: list[dict] = []
    mirror_model = ActorCritic(dropout=args.dropout, arch=args.actor_arch,
                            value_prior=value_prior(args)).to(device).eval()
    mirror = ChallengerPolicy(mirror_model, device, deck, keep_samples=False)
    mirror.greedy = True  # the frozen agents play greedily; the mirror matches them
    mirror.name = "self"
    result_queue.put(("ready", rank))

    while True:
        task = task_queue.get()
        if task[0] == "stop":
            return
        if task[0] == "weights":
            model.load_state_dict(_load_state(task[1]))
            continue
        if task[0] == "snapshot":
            snapshots.append(_load_state(task[1]))
            del snapshots[: max(0, len(snapshots) - args.snapshot_pool)]
            continue

        _, episode, kind, opponent_index, seat = task
        try:
            if kind == "self":
                mirror_model.load_state_dict(snapshots[opponent_index])
                opponent = mirror
            else:
                opponent = opponents[opponent_index]
            result, steps, _, terminal_potential = play_episode(
                challenger, opponent, seat, episode, args.max_steps,
                args.attack_shaping,
            )
            payload = None
            if result is not None:
                payload = {
                    "rows": challenger.extractor.rows,
                    "log_probs": [step["log_prob"] for step in challenger.steps],
                    "values": [step["value"] for step in challenger.steps],
                    "potentials": [step["potential"] for step in challenger.steps],
                    "terminal_potential": terminal_potential,
                }
        except Exception as error:  # noqa: BLE001 — must reach the parent, not die silently
            result, steps, payload = None, 0, None
            print(f"[worker {rank}] episode {episode} failed: {error!r}", file=sys.stderr)
        challenger.steps.clear()
        mirror.steps.clear()
        result_queue.put(
            ("episode", rank, episode, kind, opponent_index, seat, result, steps, payload)
        )


class RolloutPool:
    """Worker processes playing training episodes, one outstanding each.

    Strictly on-policy: weights are broadcast to every worker *before* the
    batch's first task is dispatched, and a worker holds at most one task, so
    no episode is ever played with weights from before the last update. Tasks
    are handed out as results come back rather than pre-split, which keeps a
    long game from leaving other workers idle.
    """

    def __init__(self, args, num_workers: int) -> None:
        import resource
        import torch.multiprocessing as multiprocessing

        # Raise the fd soft limit to the hard limit before forking, so the children inherit
        # it. ``_dump_state`` removes the reason a broadcast needed thousands of fds, but a
        # pool of N workers still costs pipes per queue and the engine keeps a handle per
        # battle, and this box's login shells default to a soft limit of 1024 against a hard
        # limit of 524288. Cheap insurance against a run that dies an hour in.
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < hard:
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
                print(f"[pool] raised open-file limit {soft} -> {hard}")
            except (ValueError, OSError) as error:
                print(f"[pool] could not raise the open-file limit from {soft}: {error!r}",
                      file=sys.stderr)

        # spawn, not fork: the parent may hold a CUDA context (the update
        # device), and forking one is undefined behaviour.
        context = multiprocessing.get_context("spawn")
        self.result_queue = context.Queue()
        self.task_queues = [context.Queue() for _ in range(num_workers)]
        self.processes = [
            context.Process(
                target=_rollout_worker,
                args=(rank, self.task_queues[rank], self.result_queue, args),
                daemon=True,
            )
            for rank in range(num_workers)
        ]
        self.num_workers = num_workers
        self.timeout = args.worker_timeout
        for process in self.processes:
            process.start()
        for _ in range(num_workers):
            kind, rank = self._get()[:2]
            if kind != "ready":
                raise RuntimeError(f"unexpected first message from worker: {kind}")
        print(f"[pool] {num_workers} rollout workers ready")

    def _get(self):
        try:
            return self.result_queue.get(timeout=self.timeout)
        except Exception as error:
            alive = [p.pid for p in self.processes if p.is_alive()]
            raise RuntimeError(
                f"no rollout result in {self.timeout}s (workers alive: {alive}). "
                f"A worker probably crashed the engine; lower --workers or raise "
                f"--worker-timeout."
            ) from error

    def broadcast(self, model: ActorCritic) -> None:
        state = _dump_state(
            {key: value.detach().cpu() for key, value in model.state_dict().items()}
        )
        for queue in self.task_queues:
            queue.put(("weights", state))

    def broadcast_snapshot(self, state: dict) -> None:
        """Append a self-play snapshot to every worker's pool.

        Sent only when the pool actually gains one (every ``--snapshot-every``
        updates), so this ~10 MB message is rare next to the per-update weight
        broadcast. Every worker receives the same snapshots in the same order,
        which is what lets the parent name one by index alone."""
        payload = _dump_state(state)
        for queue in self.task_queues:
            queue.put(("snapshot", payload))

    def collect(self, count: int, first_episode: int, schedule):
        """Yield ``count`` episodes in completion order, as they arrive.

        ``schedule(episode)`` returns the ``(opponent_index, seat)`` for an
        episode number, so the round-robin stays the parent's decision and is
        identical to the serial path's.

        A generator rather than a list, and the replacement task is dispatched
        *before* each yield, so the caller's per-episode work — rebuilding
        features from rows, which measures about as expensive as playing the
        episode did — overlaps with the workers still playing rather than
        running after all of them have gone idle. Draining this lazily is the
        difference between the parent's rebuild being hidden and it being added
        to every batch's wall-clock.
        """
        dispatched = 0
        for rank in range(min(self.num_workers, count)):
            kind, index, seat = schedule(first_episode + dispatched)
            self.task_queues[rank].put(
                ("play", first_episode + dispatched, kind, index, seat)
            )
            dispatched += 1
        for _ in range(count):
            _, rank, episode, kind, index, seat, result, steps, payload = self._get()
            if dispatched < count:
                next_kind, next_index, next_seat = schedule(first_episode + dispatched)
                self.task_queues[rank].put(
                    ("play", first_episode + dispatched, next_kind, next_index, next_seat)
                )
                dispatched += 1
            yield {
                "episode": episode, "kind": kind, "opponent_index": index, "seat": seat,
                "result": result, "steps": steps, "payload": payload,
            }

    def close(self) -> None:
        for queue in self.task_queues:
            queue.put(("stop",))
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()


def outcome_reward(result, seat: int, mode: str = "signed") -> float:
    """The win/loss term. ``result == 2`` is a draw.

    ``mode="winprob"`` labels a win 1 and a loss 0 (a draw 0.5) instead of ±1, which makes
    the critic's output *literally* a win probability: with ``--gamma`` near 1 and a
    terminal-only outcome, ``V(s)`` regresses onto ``P(win | s)``, so 0.85 means a strong
    position and 0.17 a losing one, and ``explained_variance`` becomes a calibration
    statistic you can read directly. The two labellings are an affine reparametrisation of
    each other, so neither is more expressive — but only one of them is interpretable, and
    an interpretable critic is what lets a plateau be diagnosed instead of guessed at.

    It is not free: the critic's output layer is zero-initialised (see ``ActorCritic``) so
    that ``advantage = R - V`` starts as the return itself, and 0 is a good prior for ±1
    labels averaging ≈0 while being a *terrible* one for 0/1 labels averaging ≈0.46. The
    bias is initialised to the base rate in this mode to keep that property.
    """
    if mode == "winprob":
        if result == 2:
            return 0.5
        return 1.0 if result == seat else 0.0
    if result == 2:
        return 0.0
    return 1.0 if result == seat else -1.0


#: Base rate the ``winprob`` critic is initialised to predict. The Marnie's Grimmsnarl ex
#: pilot won 4,501 of 9,784 decided seat-episodes in the replay corpus (46.0%), and a fresh
#: policy against the frozen field starts well below that — but a base-rate prior is only
#: meant to beat "confidently 0", and 0.5 is the honest starting belief for a game.
WINPROB_PRIOR = 0.5


def value_prior(args) -> float:
    """What the critic's output layer is initialised to predict."""
    return WINPROB_PRIOR if args.reward == "winprob" else 0.0


def prize_potential(obs: dict, seat: int) -> float:
    """``Φ(s)``: the prize-card differential from ``seat``'s point of view.

    This is the game's own scoreboard, not a heuristic — Pokémon TCG is won by
    taking all six of your prizes, so a lead here *is* being ahead. That is the
    whole reason it is the only shaping term: it cannot encode a mistaken
    opinion about how to play the way "put Crustle in front of an ex" could.

    Returns 0 at setup and just after the deal (both players hold 6), so the
    deal itself produces no spurious reward.
    """
    players = obs["current"]["players"]
    mine = len(players[seat]["prize"])
    theirs = len(players[1 - seat]["prize"])
    return (theirs - mine) / 6.0


def attack_potential(obs: dict, seat: int) -> float:
    """Can ``seat``'s active Pokémon pay for an attack right now, and how big?

    Normalised printed damage of the best *affordable* attack, 0 if none is.

    This exists because of a behavioural audit, not a hunch. Watching the agent
    play, the deck's single ``Rock Fighting Energy`` — the only {F} source, and
    what Cornerstone Mask Ogerpon ex's ``Demolish`` ({F}{C}{C}) requires — went
    onto **Crustle** four times in 24 games, a {G} attacker where it counts only
    as colourless, while Ogerpon received ``Mist Energy`` ({C}) three times and
    could therefore never attack at all.

    Nothing here mentions Ogerpon, Crustle or any card. It says only "energy you
    attach should leave your attacker able to attack", which is true of every
    deck. Used as a *potential*, so by Ng, Harada & Russell it cannot change
    which policy is optimal — only how quickly the agent finds it. That
    distinction matters: the agent still has to discover which attacker to build
    and when, it just stops being blind to whether it built one at all.
    """
    from value_features import _best_damage

    players = obs["current"]["players"]
    return min(_best_damage(players[seat]), 300.0) / 300.0


def state_potential(obs: dict, seat: int, attack_coefficient: float) -> float:
    """``Phi(s)``: the prize differential, optionally plus attack readiness.

    A sum of potentials is a potential, so adding the second term keeps the
    policy-invariance guarantee that makes shaping safe here.
    """
    value = prize_potential(obs, seat)
    if attack_coefficient:
        value += attack_coefficient * attack_potential(obs, seat)
    return value


def shaped_rewards(potentials, terminal_potential, outcome, gamma, coefficient):
    """Per-step rewards: the terminal win/loss plus a potential difference.

    ``coefficient == 0`` gives back the untouched outcome-only reward — one
    non-zero entry at the end — so the no-shaping experiment stays exactly what
    it was.

    Otherwise each step earns ``c * (gamma * Φ(s') - Φ(s))``. Two properties
    make this safe, both from Ng, Harada & Russell (1999):

    *It cannot be farmed.* The reward is a **difference** of a function of
    state, so any sequence of moves returning to the same prize count has
    earned exactly zero. Contrast a flat per-step bonus for being in a good
    position, whose optimal play is to sit there collecting it forever.

    *It cannot change the optimal policy.* Only the speed of learning changes.
    A shaping term that turned out to be bad advice would slow training down;
    it could not make a losing policy optimal.

    ``Φ(s')`` for the last step is the *terminal* potential — the board after
    the game ended — which is why ``play_episode`` returns it. Summed over the
    episode the shaping telescopes to ``Φ(s_T) - Φ(s_0) == Φ(s_T)``, so a win
    returns 1 + k/6 and a loss -(1 + k/6) for a k-prize margin: symmetric, and
    bounded in [-2, 2].
    """
    rewards = [0.0] * len(potentials)
    if rewards:
        rewards[-1] = outcome
    if not coefficient:
        return rewards
    for index in range(len(potentials)):
        next_potential = (
            potentials[index + 1] if index + 1 < len(potentials) else terminal_potential
        )
        rewards[index] += coefficient * (gamma * next_potential - potentials[index])
    return rewards


def compute_advantages(values, rewards, gamma, gae_lambda):
    """GAE over one episode's per-step rewards.

    Without shaping every entry of ``rewards`` is 0 but the last, and with
    ``gamma == gae_lambda == 1.0`` this reduces exactly to ``reward - value_t``
    for every step — plain outcome credit with a learned baseline. With shaping
    the intermediate entries are non-zero and GAE does the real work of
    assigning them, which is the whole point: it can finally tell the decision
    that took a prize from the fifty that didn't.
    """
    advantages = [0.0] * len(values)
    running = 0.0
    next_value = 0.0
    for index in reversed(range(len(values))):
        delta = rewards[index] + gamma * next_value - values[index]
        running = delta + gamma * gae_lambda * running
        advantages[index] = running
        next_value = values[index]
    return advantages


# --------------------------------------------------------------------------
# Update


def ppo_update(model, optimizer, batch, args, device):
    """Clipped-surrogate PPO over the collected decisions."""
    samples = [step["sample"] for step in batch]
    actions = [step["action"] for step in batch]
    old_log_probs = torch.tensor(
        [step["log_prob"] for step in batch], dtype=torch.float32, device=device
    )
    returns = torch.tensor(
        [step["return"] for step in batch], dtype=torch.float32, device=device
    )
    advantages = torch.tensor(
        [step["advantage"] for step in batch], dtype=torch.float32, device=device
    )
    old_values = torch.tensor(
        [step["value"] for step in batch], dtype=torch.float32, device=device
    )
    # Did this batch contain more than one outcome? With gamma == 1 and a
    # terminal-only reward, ``returns`` is just each episode's result repeated
    # over its transitions, so zero variance means every episode ended the same
    # way — in practice 24 straight losses, which measured 16% of batches at an
    # 8% win rate.
    #
    # Normalising such a batch is actively harmful, not merely uninformative.
    # Every return is -1, so ``adv_t = -1 - V(s_t)`` and normalising leaves
    # ``-(V(s_t) - mean V) / std V``: the sign is decided entirely by the
    # critic's opinion of the state, not by anything the policy did. It pushes
    # down actions taken where the critic was optimistic and pushes up actions
    # where it was pessimistic — uncorrelated with action quality, so it erodes
    # whatever the policy currently does best. The first 2.4k-episode run
    # peaked at a 12.6% win rate around update 40 and decayed to 6.0% by update
    # 103, with one update in six doing this.
    #
    # Skipping the update entirely is the wrong fix: the value loss still has
    # something real to learn from an all-loss batch (that these states are in
    # fact losing). Only the *policy* term is meaningless, so the advantages are
    # zeroed and the critic keeps training.
    outcome_variance = float(returns.var()) if len(batch) > 1 else 0.0
    degenerate = outcome_variance <= 1e-12
    if degenerate:
        advantages = torch.zeros_like(advantages)
    elif args.normalize_advantages and len(batch) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    stats = collections.defaultdict(list)
    # One epoch on a degenerate batch, not ``--ppo-epochs``. The policy term is
    # zeroed above, so the only gradient left is the value loss — and its target
    # is the *constant* -1 for every transition. Fitting a constant four times
    # over drives the critic toward "always predict a loss", which is the
    # degenerate baseline the first run already had (explained variance averaged
    # +0.11). One pass keeps the real signal, that these states lose, without
    # hammering the shared trunk toward a constant.
    epochs = 1 if degenerate else args.ppo_epochs
    stopped_epoch = epochs
    order = list(range(len(batch)))
    random.shuffle(order)
    # ``[indexes, collated]`` per minibatch. Collating is the update's dominant cost —
    # measured 3.5 ms a transition, i.e. ~22 s of the ~27 s an epoch over 6,400 transitions
    # takes, against ~50 ms of GPU forward+backward per minibatch of 128. Recollating it
    # every epoch was paying that bill ``--ppo-epochs`` times for a *deterministic function
    # of the same samples*, so the cache holds it instead.
    #
    # The price is that a minibatch's membership is now fixed for the whole update; only the
    # order the minibatches are visited in is reshuffled per epoch. That is a real (small)
    # loss of gradient decorrelation, which is why ``--no-cache-collate`` restores the old
    # per-epoch reshuffle. It is kept on CPU and moved per use — ~17 ms a minibatch, against
    # 2-4 GB of resident GPU memory for a whole batch.
    parts = [
        [order[start : start + args.minibatch_size], None]
        for start in range(0, len(order), args.minibatch_size)
    ]
    # Nothing to reuse when only one epoch runs — which is the common case once the KL
    # budget binds — so don't hold 2-4 GB for it.
    cache_collate = args.cache_collate and epochs > 1
    for epoch in range(epochs):
        if cache_collate:
            random.shuffle(parts)
        else:
            random.shuffle(order)
            parts = [
                [order[start : start + args.minibatch_size], None]
                for start in range(0, len(order), args.minibatch_size)
            ]
        for part in parts:
            index = part[0]
            if part[1] is None:
                collated = collate_features([samples[i] for i in index])
                if cache_collate:
                    part[1] = collated
            else:
                collated = part[1]
            features = tree_to(collated, device)
            logits, stop_logit, values, options_mask = model(features)
            min_count, max_count = selection_counts(features)
            log_probs, entropy = action_log_prob(
                logits, stop_logit, options_mask,
                min_count.to(device), max_count.to(device),
                [actions[i] for i in index],
            )

            ratio = (log_probs - old_log_probs[index]).exp()
            mb_advantages = advantages[index]
            surrogate = torch.min(
                ratio * mb_advantages,
                ratio.clamp(1 - args.clip_range, 1 + args.clip_range) * mb_advantages,
            )
            policy_loss = -surrogate.mean()

            if args.clip_value:
                clipped = old_values[index] + (values - old_values[index]).clamp(
                    -args.clip_range, args.clip_range
                )
                value_loss = 0.5 * torch.max(
                    (values - returns[index]) ** 2, (clipped - returns[index]) ** 2
                ).mean()
            else:
                value_loss = 0.5 * F.mse_loss(values, returns[index])

            entropy_loss = -entropy.mean()
            loss = (
                policy_loss
                + args.value_coef * value_loss
                + args.entropy_coef * entropy_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                stats["policy_loss"].append(float(policy_loss))
                stats["value_loss"].append(float(value_loss))
                stats["entropy"].append(float(entropy.mean()))
                stats["approx_kl"].append(float((old_log_probs[index] - log_probs).mean()))
                stats["clip_fraction"].append(
                    float(((ratio - 1).abs() > args.clip_range).float().mean())
                )
                stats["grad_norm"].append(float(grad_norm))

        # A KL budget, checked per epoch. The clip range alone does not bound
        # how far the policy moves across several epochs over the same batch:
        # the first run drifted from a mean approx_kl of 0.013 to 0.044, with
        # 40% of updates past 0.03, and entropy fell 14% as it committed to a
        # narrowing action set. Stopping the remaining epochs once the batch has
        # spent its budget bounds that per update, instead of relying on the
        # learning rate being small enough to never need it.
        if args.target_kl:
            recent = stats["approx_kl"][-max(1, len(order) // args.minibatch_size):]
            if sum(recent) / len(recent) > args.target_kl:
                stopped_epoch = epoch + 1
                break

    with torch.no_grad():
        variance = returns.var()
        explained = 1 - (returns - old_values).var() / variance if variance > 0 else 0.0
    out = {key: sum(value) / len(value) for key, value in stats.items()}
    out["explained_variance"] = float(explained)
    out["mean_return"] = float(returns.mean())
    out["mean_value"] = float(old_values.mean())
    # Both are worth watching, not just internal bookkeeping: a rising
    # ``degenerate_batch`` rate means the policy is losing nearly everything, and
    # ``epochs_run`` below ``--ppo-epochs`` means the KL budget is binding and
    # the learning rate is too high for the current signal.
    out["degenerate_batch"] = float(degenerate)
    out["epochs_run"] = float(stopped_epoch)
    return out


# --------------------------------------------------------------------------
# Evaluation


def evaluate(challenger, opponents, episodes: int, max_steps: int, base_episode: int):
    """Greedy challenger vs each greedy opponent, both seats.

    Both policies are deterministic here, but the *engine* is not — repeated
    greedy games diverge on shuffles and coin flips (checked: six replays of the
    same matchup gave six different lengths), so this is a real sample and not
    one game counted N times. It is the number to watch for convergence; the
    training win rate is measured on a sampling policy and reads far lower.

    Returns its own series names, which the caller only prefixes: one rate per
    opponent, the same split by seat, and ``win_rate`` for the mean. The seat
    split is here because going first is a genuine advantage in this engine, and
    a policy that only ever wins from seat 0 is a different (weaker) result than
    the aggregate suggests.
    """
    challenger.greedy = True
    saved = challenger.steps
    challenger.steps = []
    series: dict[str, float] = {}
    seat_totals = {0: [0, 0], 1: [0, 0]}  # seat -> [wins, played]
    try:
        for opponent in opponents:
            counts = {0: [0, 0], 1: [0, 0]}
            for i in range(episodes):
                seat = i % 2
                result, _, _, _ = play_episode(
                    challenger, opponent, seat, base_episode + i, max_steps
                )
                if result is None:
                    continue
                counts[seat][1] += 1
                seat_totals[seat][1] += 1
                if result == seat:
                    counts[seat][0] += 1
                    seat_totals[seat][0] += 1
            wins = counts[0][0] + counts[1][0]
            played = counts[0][1] + counts[1][1]
            series[opponent.name] = wins / played if played else float("nan")
            for seat in (0, 1):
                won, saw = counts[seat]
                series[f"{opponent.name}_seat{seat}"] = won / saw if saw else float("nan")
    finally:
        challenger.greedy = False
        challenger.steps = saved

    rates = [series[opponent.name] for opponent in opponents
             if series[opponent.name] == series[opponent.name]]
    series["win_rate"] = sum(rates) / len(rates) if rates else float("nan")
    for seat in (0, 1):
        won, saw = seat_totals[seat]
        series[f"win_rate_seat{seat}"] = won / saw if saw else float("nan")
    return series


# --------------------------------------------------------------------------


def build_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--episodes", type=int, default=30000,
                        help="total training battles (the 30k convergence run)")
    parser.add_argument("--episodes-per-update", type=int, default=16,
                        help="battles collected per PPO update; ~100 decisions "
                             "each, so 16 is a batch of ~1600 transitions")
    parser.add_argument("--max-steps", type=int, default=2000,
                        help="abandon a battle after this many decisions "
                             "(both players); a stuck loop otherwise hangs the run")

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.997,
                        help="1.0 was the old default and it disabled bootstrapping "
                             "entirely: with gamma == gae_lambda == 1.0, GAE collapses to "
                             "advantage_t = R - V(s_t), so every decision in a won game got "
                             "identical credit and the critic was never used to propagate "
                             "value backwards. 0.997 over a ~100-decision game keeps ~74%% of "
                             "the terminal signal at the first move while letting each "
                             "decision be graded by the state it produced")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="0.95 is the standard GAE trade: it mixes the one-step TD error "
                             "r_t + gamma*V(s_t+1) - V(s_t) across horizons instead of "
                             "trusting the Monte-Carlo return alone. 1.0 (the old default) "
                             "is pure Monte-Carlo and is why the signal was one bit per "
                             "episode spread over ~100 decisions")
    parser.add_argument("--reward", default="signed", choices=("signed", "winprob"),
                        help="'signed' is +1 win / -1 loss / 0 draw. 'winprob' is 1 / 0 / "
                             "0.5, which makes V(s) literally P(win | s) and the critic "
                             "readable as a position evaluation; the value head's bias is "
                             "initialised to the 0.5 prior in that mode")
    parser.add_argument("--self-play-ratio", type=float, default=0.0,
                        help="fraction of training episodes played against a past "
                             "snapshot of the challenger instead of a frozen BC "
                             "agent. 0 (the default) is the original setup. The "
                             "reason to raise it is measured: the policy beats "
                             "random 79%% and plays itself about evenly, but "
                             "wins only ~15%% against the frozen agents — so at "
                             "24 episodes a batch it sees ~3.6 wins to learn "
                             "from. An equal opponent gives ~12, which is the "
                             "gradient-noise fix. Evaluation stays pinned to the "
                             "frozen agents either way")
    parser.add_argument("--snapshot-every", type=int, default=25,
                        help="PPO updates between adding the current weights to "
                             "the self-play pool")
    parser.add_argument("--snapshot-pool", type=int, default=5,
                        help="how many past snapshots to keep. More than one so "
                             "the opponent is a spread of past strengths rather "
                             "than only the immediately-previous policy, which "
                             "makes chasing a single moving target less likely")
    parser.add_argument("--shaping-coef", type=float, default=0.0,
                        help="weight on potential-based prize shaping. 0 (the "
                             "default) is the pure outcome-only reward, so that "
                             "experiment stays exactly as it was. 1.0 makes the "
                             "shaping the same total magnitude as the win/loss "
                             "term but delivered as prizes are taken -- note "
                             "that PBRS telescopes to coef * prize_margin, so a coef "
                             "ABOVE 1.0 lets shaping outweigh the win/loss term it is "
                             "supposed to be subordinate to (1.5 was used and reached "
                             "+-1.5 against a +-1 terminal). Keep it below 1.0 for the "
                             "terminal to dominate, which is "
                             "what gives the critic something to fit and lets "
                             "GAE tell a prize-taking decision from the fifty "
                             "that were neutral. Being a potential difference it "
                             "cannot be farmed and cannot change which policy "
                             "is optimal — see ``shaped_rewards``")
    parser.add_argument("--attack-shaping", type=float, default=0.0,
                        help="weight on attack-readiness inside the shaping "
                             "potential. A behavioural audit found the deck's "
                             "only {F} energy going onto a {G} attacker while "
                             "Ogerpon ex sat unable to pay for Demolish; this "
                             "penalises attaching energy that leaves your "
                             "attacker unable to attack. Card-agnostic, and "
                             "potential-based so it cannot change the optimal "
                             "policy")
    parser.add_argument("--target-kl", type=float, default=0.02,
                        help="stop an update's remaining epochs once the batch's "
                             "mean approx_kl exceeds this. 0 disables. The clip "
                             "range does not bound multi-epoch drift on its own — "
                             "the first run reached 0.044 with 40%% of updates "
                             "past 0.03, alongside a 14%% entropy collapse")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="0 by default: dropout makes the rollout and update "
                             "policies differ, which shows up as ratio noise PPO "
                             "reads as a real policy change")
    parser.add_argument("--no-cache-collate", dest="cache_collate", action="store_false",
                        help="recollate every minibatch on every PPO epoch instead of "
                             "reusing the first epoch's tensors. Collation is 3.5ms a "
                             "transition and the GPU step is 50ms a minibatch of 128, so "
                             "the cache removes ~40%% of update wall-clock at "
                             "--ppo-epochs 2 and ~75%% at 4. The cost of caching is that "
                             "a minibatch's membership is fixed for the update (only the "
                             "visiting order reshuffles), so pass this to get the old "
                             "per-epoch reshuffle back")
    parser.add_argument("--no-normalize-advantages", dest="normalize_advantages",
                        action="store_false")
    parser.add_argument("--no-clip-value", dest="clip_value", action="store_false")

    parser.add_argument("--actor-arch", default="challenger",
                        choices=("challenger", "crustle", "alakazam", "lucario"),
                        help="which PolicyNetwork generation the actor is built "
                             "from. Anything but 'challenger' gives up the newer "
                             "ability features in exchange for --init-from "
                             "actually loading that agent's bc_policy.pt, which "
                             "starts training at ~41%% instead of 0%%")
    parser.add_argument("--init-from", default=None, metavar="CHECKPOINT",
                        help="warm-start the actor from a BC checkpoint "
                             "(e.g. crustle_frozen/bc_policy.pt). Off by "
                             "default: this run trains from scratch")
    parser.add_argument("--resume", default=None, metavar="CHECKPOINT",
                        help="resume from a train.py checkpoint (model + optimizer)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device for the PPO update, where batches are big "
                             "enough for a GPU to win")
    parser.add_argument("--rollout-device", default="cpu",
                        help="device for rollout inference. CPU by default and "
                             "deliberately: every rollout forward is batch 1, so "
                             "it is bound by Python and kernel-launch overhead "
                             "rather than FLOPs, and measured 28ms/decision on "
                             "CPU against 72ms on this box's GPU. A CPU replica "
                             "of the actor is synced after every update")
    parser.add_argument("--threads", type=int, default=None,
                        help="torch intra-op threads (default: leave torch's own)")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="concurrent battles in THIS process, batched into one "
                             "forward per decision. >1 replaces the environment with "
                             "pokeka2026's per-instance PTCGBattle (see pokeka_env), "
                             "which is what makes a GPU rollout worth anything: batch "
                             "1 measured 7.6ms/decision on CPU and 12.6ms on this "
                             "box's GPU, while batch 64 measured 16ms *total* on the "
                             "GPU and 74ms on the CPU. It also removes the worker "
                             "feature rebuild entirely, since the transform() output "
                             "never leaves the process that updates on it. Mutually "
                             "exclusive with --workers; 24-32 is the measured sweet "
                             "spot (collate cost grows faster than linearly past that)")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel rollout processes. 1 runs everything "
                             "in-process (simplest, and the path the smoke "
                             "tests cover). The engine is a per-process "
                             "singleton, so N>1 means N processes each playing "
                             "whole episodes; measured ~30ms/decision "
                             "single-threaded, so it scales close to linearly "
                             "up to about 6 on a 12-core box before the "
                             "parent's feature rebuild becomes the ceiling")
    parser.add_argument("--worker-timeout", type=float, default=900.0,
                        help="seconds to wait for a rollout result before "
                             "declaring a worker dead (it would otherwise hang "
                             "the run silently)")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--eval-every", type=int, default=100,
                        help="PPO updates between greedy evaluations (0 disables)")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="greedy battles per opponent per evaluation")
    parser.add_argument("--final-eval-episodes", type=int, default=100,
                        help="greedy battles per opponent in the end-of-run "
                             "evaluation, which is the number worth quoting")
    parser.add_argument("--baseline-episodes", type=int, default=60,
                        help="random-policy control battles per opponent, run "
                             "once before training (0 disables)")
    parser.add_argument("--save-every", type=int, default=50,
                        help="PPO updates between checkpoint writes")
    parser.add_argument("--out", default=None,
                        help="checkpoint directory (default checkpoints/<run-name>)")

    parser.add_argument("--run-name", default="greedy-crustle-PPO")
    parser.add_argument("--wandb-project", default="pokemon-tcg-rl")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default="online",
                        choices=("online", "offline", "disabled"))
    parser.add_argument("--window", type=int, default=500,
                        help="episodes in the rolling training win-rate window")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = build_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    # Vectorised rollouts invert the reason --rollout-device defaulted to CPU: the forward
    # is no longer batch 1, so the GPU wins by ~20x per decision and there is nothing to
    # gain from a second, CPU-resident copy of the actor to keep in sync.
    if args.num_envs > 1:
        if args.rollout_device != args.device:
            print(f"[vec] --num-envs {args.num_envs}: rollouts run on {args.device} "
                  f"(--rollout-device {args.rollout_device} ignored — batched inference "
                  f"is the whole point of vectorising)", file=sys.stderr)
        args.rollout_device = args.device
        if args.workers > 1:
            raise SystemExit(
                "--num-envs and --workers are mutually exclusive: one batches N battles "
                "in this process, the other forks N processes each holding one. Pick one."
            )
    rollout_device = torch.device(args.rollout_device)

    out_dir = Path(args.out) if args.out else _ROOT / "checkpoints" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    challenger_deck = read_deck(_ROOT / "challenger" / "challenger_deck.csv")
    opponents = [
        FrozenPolicy(name, _ROOT / f"{name}_frozen", rollout_device) for name in OPPONENTS
    ]

    model = ActorCritic(dropout=args.dropout, arch=args.actor_arch,
                            value_prior=value_prior(args)).to(device)
    if args.init_from:
        # ``strict=False``, deliberately and narrowly. The challenger's PolicyNetwork gained
        # ``option_cross`` after ``bc_policy.pt`` was cloned, so a strict load fails on its 37
        # keys -- which is a checkpoint that predates a *new module*, not a wrong checkpoint.
        # The two cases are distinguished rather than conflated: UNEXPECTED keys mean the
        # weights do not belong to this architecture at all and are still refused, because
        # silently dropping them is how a bundle ends up running a different net than the one
        # that was measured.
        checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=True)
        result = model.policy.load_state_dict(checkpoint, strict=False)
        if result.unexpected_keys:
            raise SystemExit(
                f"{args.init_from} has {len(result.unexpected_keys)} keys that do not map onto "
                f"this PolicyNetwork: {list(result.unexpected_keys)[:5]} -- wrong --actor-arch?"
            )
        print(f"[init] actor warm-started from {args.init_from}")
        if result.missing_keys:
            modules = sorted({key.split(".")[0] for key in result.missing_keys})
            print(f"[init] {len(result.missing_keys)} keys absent from the checkpoint, in: "
                  f"{modules} (modules added after it was trained)")
            # Start those modules as exact identities so the warm start IS the measured policy
            # rather than the measured policy plus random noise. See
            # OptionCrossAttention.zero_residual_branches.
            if any(key.startswith("option_cross.") for key in result.missing_keys):
                model.policy.option_cross.zero_residual_branches()
                print("[init] option_cross zeroed to an exact identity; cross-attention will "
                      "grow in from zero rather than perturbing the clone")
        # The same trap FINDINGS.md records for the crustle warm start, which applies to ANY
        # warm start and so cannot live in the ``arch != challenger`` branch of ActorCritic:
        # a trained option head competing against a randomly-initialised stop logit inside
        # one softmax lets "stop" win often enough to corrupt every optional and
        # multi-select decision from the first rollout. -4.0 makes "keep selecting" the
        # default, reproducing the floor-at-one decode the cloned policy was fitted under.
        nn.init.zeros_(model.stop.weight)
        nn.init.constant_(model.stop.bias, -4.0)
        print("[init] stop logit biased to -4.0 (warm start protects the cloned decode)")
    # Rollouts always run in eval mode; ``ppo_update`` flips to train() and back.
    # With --dropout 0 this is a no-op, but it stops a non-zero --dropout from
    # silently making the pre-first-update rollouts stochastic in a way the
    # update's log-probs would then disagree with.
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_episode = 0
    update = 0
    resumed_best = None
    if args.resume:
        # weights_only=False: the checkpoint carries its ``args`` dict too, and
        # this file is the only thing that writes them.
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        else:
            # A preserved weights-only checkpoint. Adam's moments are a running
            # average that re-adapts within tens of updates, so resuming without
            # them costs a little early instability and nothing lasting — far
            # less than refusing to load a checkpoint worth continuing from.
            print("[resume] no optimizer state; Adam moments start fresh")
        # ``Adam.load_state_dict`` restores ``param_groups``, which carries the
        # *checkpoint's* learning rate — so without this line a resume silently
        # ignores ``--lr`` and keeps training at the old one. Retuning the
        # learning rate is one of the main reasons to resume at all, so the
        # command line has to win. The moment estimates are kept either way,
        # which is the part actually worth resuming.
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        start_episode = state.get("episode", 0)
        update = state.get("update", 0)
        # Carry the historical best forward. Without this every restart begins
        # at -inf, so the first evaluation of the new run overwrites best.pt
        # whatever it scores — which is how a 51.7% checkpoint was replaced by a
        # 36.7% one and survived only because it had already been packaged.
        resumed_best = state.get("best_eval")
        resumed_lr = state.get("args", {}).get("lr")
        print(f"[resume] {args.resume} at episode {start_episode}, update {update}")
        if resumed_lr is not None and resumed_lr != args.lr:
            print(f"[resume] learning rate {resumed_lr:g} -> {args.lr:g} (from --lr)")

    # The rollout actor is a replica on ``rollout_device`` (see --rollout-device),
    # re-synced after every update. When the two devices match it *is* the master
    # model, so there is nothing to sync and no way for the two to drift.
    rollout_model = model
    if rollout_device != device:
        rollout_model = ActorCritic(dropout=args.dropout, arch=args.actor_arch,
                            value_prior=value_prior(args)).to(rollout_device).eval()
        rollout_model.load_state_dict(model.state_dict())
    challenger = ChallengerPolicy(rollout_model, rollout_device, challenger_deck)
    challenger.attack_shaping = args.attack_shaping

    def sync_rollout_model() -> None:
        if rollout_model is not model:
            rollout_model.load_state_dict(model.state_dict())

    import wandb

    if not hasattr(wandb, "init"):
        # This directory holds wandb's own ``wandb/`` run logs, and with no wandb installed
        # that directory imports as an empty namespace package — so the failure surfaces as
        # "module 'wandb' has no attribute 'init'" rather than an ImportError.
        raise SystemExit(
            f"wandb resolved to {getattr(wandb, '__path__', wandb)!r}, which is this run-log "
            f"directory, not the package. Install it: pip install wandb"
        )

    run = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity, name=args.run_name,
        mode=args.wandb_mode, config=vars(args) | {
            "opponents": list(OPPONENTS),
            "reward": "outcome only: +1 win / -1 loss / 0 draw, no shaping",
            "opponent_policy": "greedy (decode_action, count floored at 1)",
            "actor_init": args.init_from or "random",
            "parameters": sum(p.numel() for p in model.parameters()),
        },
    )
    print(f"[wandb] {run.url if args.wandb_mode == 'online' else args.wandb_mode}")
    print(
        f"[setup] {sum(p.numel() for p in model.parameters()):,} parameters, "
        f"rollout on {rollout_device}, updates on {device}, "
        f"{args.episodes} episodes, {args.episodes_per_update} per update"
    )

    if args.baseline_episodes:
        baseline = RandomPolicy(challenger_deck)
        rates = {}
        for opponent in opponents:
            wins = played = 0
            for i in range(args.baseline_episodes):
                result, _, _, _ = play_episode(baseline, opponent, i % 2, -1 - i, args.max_steps)
                if result is None:
                    continue
                played += 1
                wins += result == (i % 2)
            rates[opponent.name] = wins / played if played else float("nan")
            print(f"[baseline] random vs {opponent.name}: {rates[opponent.name]:.1%}")
        wandb.log(
            {f"baseline/random_vs_{name}": rate for name, rate in rates.items()},
            step=start_episode,
        )
        wandb.summary.update({f"baseline/random_vs_{k}": v for k, v in rates.items()})

    window = collections.deque(maxlen=args.window)
    # Frozen-opponent games only. Self-play sits near 50% by construction, so
    # folding it into the headline rate would make that number leap the moment
    # self-play switches on and look like progress it isn't. This is the one the
    # console prints and the one comparable to every earlier run.
    frozen_window = collections.deque(maxlen=args.window)
    labels = (*OPPONENTS, "self")
    per_opponent = {name: collections.deque(maxlen=args.window // len(labels) or 1)
                    for name in labels}

    # Past weights the challenger trains against. Kept on the parent as CPU
    # state dicts and mirrored into every worker; the in-process path loads them
    # into ``mirror`` below.
    snapshots: collections.deque = collections.deque(maxlen=max(1, args.snapshot_pool))
    mirror_model = ActorCritic(dropout=args.dropout, arch=args.actor_arch,
                            value_prior=value_prior(args)).to(rollout_device).eval()
    mirror = ChallengerPolicy(mirror_model, rollout_device, challenger_deck)
    mirror.greedy = True
    mirror.name = "self"

    def schedule(episode_number: int) -> tuple[str, int, int]:
        """``(kind, index, seat)`` for an episode — pure in the episode number
        and the current pool size, so the parallel path draws the identical
        sequence the serial one would.

        Seat alternates every episode regardless of opponent, which is what
        keeps the first-player advantage out of the reported win rate. Frozen
        opponents still cycle by episode number, so all three keep appearing
        even once most episodes are self-play.
        """
        seat = episode_number % 2
        if snapshots and random.Random(episode_number).random() < args.self_play_ratio:
            return "self", episode_number % len(snapshots), seat
        return "frozen", episode_number % len(opponents), seat

    def opponent_for(kind: str, index: int):
        """Resolve a schedule entry to a policy for the in-process path."""
        if kind == "self":
            mirror_model.load_state_dict(snapshots[index])
            mirror.steps.clear()
            return mirror
        return opponents[index]

    # Vectorised path. The self-play mirrors are kept *resident* — one network per snapshot
    # slot, index-aligned with ``snapshots`` — because a state-dict load per tick would cost
    # more than the games it plays, and 10 MB a snapshot is nothing next to a rollout batch.
    runner = None
    mirror_models: list[ActorCritic] = []
    if args.num_envs > 1:
        import vec_rollout  # noqa: PLC0415 -- imports helpers back out of this module

        if args.self_play_ratio > 0:
            mirror_models = [
                ActorCritic(dropout=args.dropout, arch=args.actor_arch,
                            value_prior=value_prior(args)).to(device).eval()
                for _ in range(max(1, args.snapshot_pool))
            ]
        runner = vec_rollout.VecRunner(
            model, opponents, challenger_deck, device, args, args.num_envs,
            mirrors=mirror_models,
        )
        print(f"[vec] {args.num_envs} concurrent battles on {device} "
              f"(pokeka PTCGBattle environment, batched inference)", flush=True)

    def sync_mirrors() -> None:
        """Reload every resident mirror from the snapshot pool it shadows."""
        for network, state in zip(mirror_models, snapshots):
            network.load_state_dict(state)

    if args.workers > 1 and rollout_device.type != "cpu":
        # Workers are CPU-only by construction (one process per core, batch-1
        # inference); say so rather than let --rollout-device look honoured.
        print(f"[warn] --workers {args.workers} runs its rollouts on CPU; "
              f"--rollout-device {args.rollout_device} applies only to the "
              f"parent's evaluation games", file=sys.stderr)
    pool = RolloutPool(args, args.workers) if args.workers > 1 else None
    best_eval = float("-inf")
    if args.resume and resumed_best is not None and resumed_best == resumed_best:
        best_eval = resumed_best
        print(f"[resume] carrying best_eval {best_eval:.4f} forward")
    started = time.time()
    episode = start_episode

    try:
      while episode < args.episodes:
        batch = min(args.episodes_per_update, args.episodes - episode)

        if runner is not None:
            # One tick advances every slot by a decision, so the batch's episodes are played
            # concurrently rather than one after another. Weights are the live ``model``, so
            # this is strictly on-policy by construction — there is no replica to sync and no
            # worker holding a stale copy.
            tasks = [schedule(episode + offset) for offset in range(batch)]
            collected = [
                (item["kind"], item["opponent_index"], item["seat"], item["result"],
                 item["steps"], item["episode_steps"], item["terminal_potential"])
                for item in runner.collect(tasks, episode)
            ]
        elif pool is None:
            collected = []
            for offset in range(batch):
                kind, index, seat = schedule(episode + offset)
                before = len(challenger.steps)
                result, steps, _, terminal = play_episode(
                    challenger, opponent_for(kind, index), seat,
                    episode + offset, args.max_steps, args.attack_shaping,
                )
                episode_steps = challenger.steps[before:]
                del challenger.steps[before:]
                collected.append(
                    (kind, index, seat, result, steps,
                     episode_steps if result is not None else [], terminal)
                )
        else:
            # Weights before tasks: a worker holds one task at a time, so this
            # is what makes every collected episode strictly on-policy.
            pool.broadcast(model)
            collected = []
            for item in pool.collect(batch, episode, schedule):
                if item["result"] is None or item["payload"] is None:
                    episode_steps, terminal = [], 0.0
                else:
                    episode_steps = rebuild_steps(item["payload"])
                    terminal = item["payload"]["terminal_potential"]
                collected.append(
                    (item["kind"], item["opponent_index"], item["seat"],
                     item["result"], item["steps"], episode_steps, terminal)
                )

        pending: list[dict] = []
        abandoned = 0
        shaping_total = 0.0
        self_play_games = 0
        for kind, opponent_index, seat, result, steps, episode_steps, terminal in collected:
            episode += 1
            self_play_games += kind == "self"
            if result is None:
                # Drop an abandoned episode's transitions rather than labelling
                # them with a guessed outcome.
                abandoned += 1
                continue

            reward = outcome_reward(result, seat, args.reward)
            values = [step["value"] for step in episode_steps]
            rewards = shaped_rewards(
                [step["potential"] for step in episode_steps], terminal, reward,
                args.gamma, args.shaping_coef,
            )
            shaping_total += sum(rewards) - reward
            advantages = compute_advantages(values, rewards, args.gamma, args.gae_lambda)
            for step, advantage in zip(episode_steps, advantages):
                step["advantage"] = advantage
                step["return"] = advantage + step["value"]
            pending.extend(episode_steps)

            name = "self" if kind == "self" else opponents[opponent_index].name
            won = float(result == seat)
            window.append(won)
            if kind != "self":
                frozen_window.append(won)
            per_opponent[name].append(won)
            wandb.log(
                {
                    "episode/reward": reward,
                    "episode/win": won,
                    "episode/draw": float(result == 2),
                    "episode/steps": steps,
                    "episode/decisions": len(episode_steps),
                    "episode/seat": seat,
                    "episode/opponent": name,
                    "train/win_rate": sum(window) / len(window),
                    **({"train/win_rate_frozen": sum(frozen_window) / len(frozen_window)}
                       if frozen_window else {}),
                    **{
                        f"train/win_rate_vs_{key}": (sum(deque_) / len(deque_))
                        for key, deque_ in per_opponent.items() if deque_
                    },
                },
                step=episode,
            )

        if abandoned:
            print(f"[warn] {abandoned}/{batch} episodes abandoned in this batch",
                  file=sys.stderr)
        if not pending:
            continue

        model.train()
        update_started = time.time()
        stats = ppo_update(model, optimizer, pending, args, device)
        model.eval()
        sync_rollout_model()
        update_seconds = time.time() - update_started
        update += 1

        # Snapshot for the self-play pool. Taken after the update so the pool
        # holds policies the challenger has actually been, and only when
        # self-play is enabled — otherwise it is 10 MB of IPC for nothing.
        if args.self_play_ratio > 0 and update % max(1, args.snapshot_every) == 0:
            snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            snapshots.append(snapshot)
            sync_mirrors()
            if pool is not None:
                pool.broadcast_snapshot(snapshot)
            print(f"[snapshot] update {update}: pool now {len(snapshots)}", flush=True)
        elapsed = time.time() - started
        wandb.log(
            {
                **{f"loss/{key}": value for key, value in stats.items()},
                "train/update": update,
                "train/transitions": len(pending),
                # Total shaping reward per episode. Potential-based shaping
                # telescopes to Phi(s_T), so this should track the prize margin:
                # near +1 when winning comfortably, near -1 when being swept.
                # If it drifts far outside [-2, 2] something is wrong with Phi.
                "train/shaping_per_episode": shaping_total / max(batch - abandoned, 1),
                "train/self_play_fraction": self_play_games / max(batch, 1),
                "train/snapshot_pool": len(snapshots),
                "train/episodes_per_hour": (episode - start_episode) / elapsed * 3600,
                "train/update_seconds": update_seconds,
                "train/lr": optimizer.param_groups[0]["lr"],
            },
            step=episode,
        )
        print(
            f"[update {update}] ep {episode}/{args.episodes} "
            f"win {sum(frozen_window) / len(frozen_window):.1%} "
            f"(n={len(frozen_window)})  pi {stats['policy_loss']:+.4f} "
            f"v {stats['value_loss']:.4f} H {stats['entropy']:.3f} "
            f"kl {stats['approx_kl']:+.4f} ev {stats['explained_variance']:+.2f} "
            f"ep{stats['epochs_run']:.0f}/{args.ppo_epochs}"
            f"{' DEGEN' if stats['degenerate_batch'] else ''} "
            f"| {(episode - start_episode) / elapsed * 3600:.0f} ep/h",
            flush=True,
        )

        if args.eval_every and update % args.eval_every == 0:
            # Same series, same games; vectorised it costs the batched inference of a couple
            # of dozen games rather than 120 serial ones.
            rates = (
                vec_rollout.evaluate(
                    runner, opponents, args.eval_episodes, args.max_steps,
                    base_episode=10_000_000 + update * 1000,
                )
                if runner is not None
                else evaluate(
                    challenger, opponents, args.eval_episodes, args.max_steps,
                    base_episode=10_000_000 + update * 1000,
                )
            )
            # eval/crustle, eval/alakazam, eval/lucario, the same three split by
            # seat, and eval/win_rate for the mean — ``evaluate`` names them.
            wandb.log({f"eval/{key}": rate for key, rate in rates.items()}, step=episode)
            print(f"[eval {update}] " + "  ".join(
                f"{opponent.name} {rates[opponent.name]:.1%}" for opponent in opponents
            ) + f"  mean {rates['win_rate']:.1%}"
              + f"  (seat0 {rates['win_rate_seat0']:.1%}"
              + f" seat1 {rates['win_rate_seat1']:.1%})", flush=True)
            if rates["win_rate"] == rates["win_rate"] and rates["win_rate"] > best_eval:
                best_eval = rates["win_rate"]
                save(out_dir / "best.pt", model, optimizer, episode, update, args, best_eval)
                wandb.summary.update({"eval/best_win_rate": best_eval, "eval/best_episode": episode})

        if args.save_every and update % args.save_every == 0:
            save(out_dir / "latest.pt", model, optimizer, episode, update, args, best_eval)
    finally:
        # Always write something on the way out — a KeyboardInterrupt 20 hours
        # into a 30k-episode run should not throw the run away — and never
        # leave worker processes behind holding an engine each.
        save(out_dir / "latest.pt", model, optimizer, episode, update, args, best_eval)
        if pool is not None:
            pool.close()

    save(out_dir / "final.pt", model, optimizer, episode, update, args, best_eval)
    if args.final_eval_episodes:
        rates = (
            vec_rollout.evaluate(runner, opponents, args.final_eval_episodes,
                                 args.max_steps, base_episode=20_000_000)
            if runner is not None
            else evaluate(challenger, opponents, args.final_eval_episodes,
                          args.max_steps, base_episode=20_000_000)
        )
        print("[final] " + "  ".join(f"{k} {v:.1%}" for k, v in rates.items()))
        # Same series as the periodic eval, one namespace up: final/crustle, ...
        wandb.summary.update({f"final/{key}": rate for key, rate in rates.items()})
    if runner is not None:
        runner.close()  # free the N native battles the slots still hold
    wandb.finish()


def save(path: Path, model, optimizer, episode, update, args, best_eval) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            # The actor alone, in plain PolicyNetwork shape — so ``load_policy``
            # and therefore the frozen ``main.py`` can read it without knowing
            # anything about the value head.
            "policy": model.policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "episode": episode,
            "update": update,
            "best_eval": best_eval,
            "args": vars(args),
        },
        path,
    )
    (path.parent / f"{path.stem}.meta.json").write_text(
        json.dumps(
            {"episode": episode, "update": update, "best_eval": best_eval,
             "args": vars(args)}, indent=2, default=str
        )
    )


if __name__ == "__main__":
    main()
