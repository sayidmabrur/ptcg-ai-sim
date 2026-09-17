"""Information-Set MCTS over sampled worlds, on ``cg``'s search API.

Built against constants measured in ``probe_search_api.py``, not assumed:

* ``search_step`` costs ~0.1 ms and a stepped node stays steppable, so the tree
  branches freely and descent is cheap (P2, P3).
* With ``manual_coin=True`` the engine is **fully deterministic**: stepping the
  same state with the same action lands in the same place, 60/60 frames, and an
  identical belief replays identically (P13). All randomness is either in the
  particle or in a coin, and coins become explicit decisions.
* The engine validates nothing semantic (P9), so particles come from
  ``belief.py``, which does.
* A network forward is ~30 ms against ~0.1 ms for an engine step — a factor of
  ~300. The leaf evaluator, not the simulator, is the entire cost model, which
  is why ``Evaluator`` takes a *batch* of leaves.

Why IS-MCTS and not plain determinized MCTS
-------------------------------------------
Running one independent tree per particle and voting at the root (PIMC) is
simpler and is what most card-game bots do. It also suffers *strategy fusion*:
each tree plans as though it will know the hidden cards at every future node,
so the root action it likes is one that works only if you can tell the worlds
apart later. A single tree whose statistics are shared across particles cannot
do that — every particle updates the same node, so an action is only good if it
is good on average across the worlds consistent with what is actually known.

That sharing needs nodes to mean the same thing under every particle, which is
why actions are keyed by a **semantic signature** rather than by option index:
option 3 can be Ultra Ball in one world and Boss's Orders in another, and
merging their statistics would be worse than not sharing at all. The signature
mirrors ``policy_experimental._EQUIVALENCE_FIELDS``, which was measured against
replay data for exactly this "are these the same play" question.

The coin trap
-------------
``manual_coin=True`` lets *you* call the flip. It is enabled here for
determinism and for the option to enumerate — **not** because the search may
choose. A ``COIN_HEAD`` node is a chance node: it is sampled, never maximised.
Treating it as a decision produces an agent that plans as though it wins every
50/50, which is both wrong and very hard to notice from win-rate alone.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from cg.api import (
    OptionType,
    SearchState,
    SelectContext,
    search_begin,
    search_end,
    search_step,
    to_observation_class,
)

import torch

from belief import Particle
from value_features import scalar_features

# --------------------------------------------------------------------------
# Actions


#: Fields that decide whether two options are the *same play*. Mirrors
#: ``policy_experimental._EQUIVALENCE_FIELDS``: pointer-ish fields (``index``,
#: ``serial``, ``energyIndex``) are excluded because two options differing only
#: there do the same thing — hand slot 3 and slot 5 both holding an Ultra Ball.
#: ``inPlayIndex`` is kept: it names *which* Pokémon is targeted, and two board
#: Pokémon differ in HP and attached energy even at the same card id.
_SIGNATURE_FIELDS = (
    "type", "area", "cardId", "attackId", "inPlayIndex",
    "specialConditionType", "count", "number", "playerIndex",
)


def option_signature(option) -> tuple:
    return tuple(getattr(option, field, None) for field in _SIGNATURE_FIELDS)


#: An action is an ordered selection of option indexes; its *key* is the
#: multiset of signatures it selects, which is particle-independent.
ActionKey = tuple


def action_key(select, indexes: Sequence[int]) -> ActionKey:
    return tuple(sorted(option_signature(select.option[i]) for i in indexes))


#: Multi-select decisions have a combinatorial action space (choose k of n).
#: ~94% of decisions are single-select, where this never fires; for the rest a
#: bounded sample of legal selections keeps the branching factor finite. It is a
#: real restriction of the action space and is counted in ``SearchStats`` rather
#: than hidden.
MULTISELECT_CANDIDATES = 8


def legal_actions(select, rng: random.Random) -> list[list[int]]:
    """Candidate selections for a decision, as lists of option indexes."""
    n = len(select.option)
    if n == 0:
        return []
    low, high = select.minCount, min(select.maxCount, n)
    if low <= 1 and high == 1:
        return [[i] for i in range(n)]
    actions: list[list[int]] = []
    seen: set[tuple] = set()
    if low == 0:
        actions.append([])
        seen.add(())
    for _ in range(MULTISELECT_CANDIDATES * 3):
        if len(actions) >= MULTISELECT_CANDIDATES:
            break
        count = rng.randint(max(low, 1), high) if high >= max(low, 1) else low
        pick = tuple(sorted(rng.sample(range(n), min(count, n))))
        if pick not in seen:
            seen.add(pick)
            actions.append(list(pick))
    return actions or [list(range(min(low, n)))]


#: Rough prior weight per option type on a MAIN decision, before normalisation.
#:
#: The motivation is depth, not taste. MCTS depth grows like ``log_b(N)`` in the
#: branching factor: at ~11 root options and 128 simulations that is about two
#: plies, so the "search" is barely deeper than a greedy evaluation and extra
#: simulations widen the tree instead of deepening it. A prior concentrates the
#: budget on a few plausible lines, which is the mechanism by which AlphaZero
#: gets depth out of a fixed simulation count — its prior is a trained network,
#: this is a placeholder for one.
#:
#: Deliberately coarse. It encodes only things that are true of the game rather
#: than of a deck: you win by attacking, energy is the scarce resource,
#: evolving is tempo, and passing with plays available is usually wrong.
_PRIOR_WEIGHTS = {
    OptionType.ATTACK: 4.0,
    OptionType.EVOLVE: 3.0,
    OptionType.ATTACH: 3.0,
    OptionType.ABILITY: 2.0,
    OptionType.PLAY: 1.5,
    OptionType.RETREAT: 0.8,
    OptionType.END: 0.4,
}


_ATTACK_DAMAGE: dict[int, int] = {}


def _attack_damage(attack_id) -> int:
    """Printed damage for an attack id.

    An ``Option`` carries ``attackId`` but not damage — ``number`` is the count
    field for NUMBER-type options and is ``None`` on an ATTACK, so reading it
    here (as an earlier version did) silently produced no tilt at all. Damage
    lives in the static ``Attack`` table, loaded once.
    """
    if not _ATTACK_DAMAGE:
        from cg.api import all_attack

        _ATTACK_DAMAGE.update({a.attackId: (a.damage or 0) for a in all_attack()})
    return _ATTACK_DAMAGE.get(attack_id, 0) if attack_id is not None else 0


def heuristic_priors(select) -> list[float] | None:
    """A normalised prior over this decision's options, or ``None`` to stay uniform.

    Only fires on decisions that mix action *types* — a MAIN menu. Sub-decisions
    ("which card to discard", "which energy to attach") offer options of a single
    type, where these weights say nothing and a uniform prior is the honest
    default.
    """
    types = {option.type for option in select.option}
    if len(types) < 2 or not (types & set(_PRIOR_WEIGHTS)):
        return None
    weights = []
    for option in select.option:
        weight = _PRIOR_WEIGHTS.get(option.type, 1.0)
        if option.type == OptionType.ATTACK:
            # Prefer the attack that actually threatens a knockout. Absolute
            # damage is not comparable across decks, so this is a mild tilt
            # rather than a ranking.
            damage = _attack_damage(getattr(option, "attackId", None))
            weight *= 1.0 + min(damage, 300) / 300.0
        weights.append(weight)
    total = sum(weights)
    return [w / total for w in weights] if total > 0 else None


def is_chance(select) -> bool:
    """A coin call. Sampled, never maximised — see the module docstring."""
    return select is not None and select.context == SelectContext.COIN_HEAD


# --------------------------------------------------------------------------
# Leaf evaluation


class Evaluator(Protocol):
    """Scores a batch of leaves from ``seat``'s point of view.

    Batched because of the 300:1 network-to-engine cost ratio: the search is
    affordable only if the frontier is evaluated together. A single-leaf
    evaluator is a valid implementation of this protocol, and
    ``RolloutEvaluator`` is one, but a network one must not be.

    Returns ``(values, priors)``. ``values`` are in ``[-1, 1]``, positive good
    for ``seat``. ``priors`` is optional; when present it is a probability over
    each leaf's option list and is used as the PUCT prior.
    """

    def __call__(
        self, states: Sequence[SearchState], seat: int
    ) -> tuple[list[float], list[list[float]] | None]:
        ...


def prize_differential(state: SearchState, seat: int) -> float:
    """``(their prizes remaining - ours) / 6`` — the game's own scoreboard.

    Not a heuristic about how to play: Pokémon TCG is won by taking six prizes,
    so a lead here *is* being ahead. That is why it is the depth-limit fallback
    rather than something opinionated about board position.
    """
    players = state.observation.current.players
    return (len(players[1 - seat].prize) - len(players[seat].prize)) / 6.0


def terminal_value(state: SearchState, seat: int) -> float:
    result = state.observation.current.result
    if result == 2:
        return 0.0
    return 1.0 if result == seat else -1.0


class ValueNetEvaluator:
    """Leaf evaluation by the scalar value network.

    Replaces ``RolloutEvaluator``, which is why the search did not scale: a
    single random playout's residual noise measured 0.89 against a true
    between-action spread of 0.31, so extra simulations sharpened nothing. This
    model's residual is 0.68 mid-game (explained variance 0.53 on held-out
    expert games) and it costs tens of microseconds instead of 40 ms.

    Terminal states bypass the network entirely — the outcome is known, and
    letting a ``tanh`` head approximate a value it can only get wrong would put
    noise on the one estimate in the tree that is exact.

    Batched in one forward per frontier, per the ``Evaluator`` contract, though
    at this model size the search is engine-bound rather than evaluator-bound
    and the batching is no longer load-bearing.
    """

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(self, states, seat):
        values: list[float] = [0.0] * len(states)
        pending, rows = [], []
        for position, state in enumerate(states):
            if state.observation.current.result != -1:
                values[position] = terminal_value(state, seat)
                continue
            pending.append(position)
            rows.append(scalar_features(state.observation.current, seat))
        if pending:
            batch = torch.tensor(rows, dtype=torch.float32, device=self.device)
            for position, value in zip(pending, self.model(batch).tolist()):
                values[position] = value
        return values, None


#: Rollout action preference, best first. Uniform random play is what made the
#: first scaling sweep flat: measured over 8 root actions x 12 reps, a
#: depth-capped random playout separated actions by 0.047 while a single
#: playout's own noise was 0.085 — the evaluator could not see the thing it was
#: being asked to score. A random policy also *misplays* a good position into a
#: bad one, which biases the estimate downward wherever there was something
#: worth protecting.
#:
#: This is not a strategy, and it is deliberately not tuned: it is an ordering
#: that keeps a playout doing something rather than passing, so games terminate
#: and the prize count actually moves. Anything cleverer belongs in a learned
#: policy prior, not hard-coded here.
_ROLLOUT_PREFERENCE = (
    OptionType.ATTACK, OptionType.EVOLVE, OptionType.ATTACH, OptionType.ABILITY,
    OptionType.PLAY, OptionType.RETREAT,
)


class RolloutEvaluator:
    """Playout to terminal, or prize differential at the depth cap.

    Exists so the search runs with **no network at all** — generation 0 of the
    distillation loop, sidestepping the random-init problem that sank the PPO
    runs. P6 measured a full playout at ~31 ms / 135 steps, so it is affordable.

    It is nonetheless a weak evaluator: at full depth a single playout's noise
    (0.86) dwarfs the true spread between actions (0.31), so ranking five
    options needs roughly eight playouts each. Averaging ``repeats`` playouts
    cuts that as 1/sqrt(n) at linear cost, which is the honest knob. Replacing
    this with a value network is the real fix and the reason ``Evaluator`` is
    batched.
    """

    def __init__(
        self,
        depth: int = 400,
        rng: random.Random | None = None,
        repeats: int = 1,
        greedy_probability: float = 0.75,
    ):
        # Default depth is effectively uncapped: a capped playout that never
        # reaches a result reads the prize differential of a position nothing
        # has resolved, which is where the blindness came from.
        self.depth = depth
        self.rng = rng or random.Random()
        self.repeats = max(1, repeats)
        self.greedy_probability = greedy_probability

    def __call__(self, states, seat):
        values = []
        for state in states:
            samples = [self._one(state, seat) for _ in range(self.repeats)]
            values.append(sum(samples) / len(samples))
        return values, None

    def _choose(self, select) -> list[int]:
        """Preference-ordered when it fires, uniform otherwise.

        Kept stochastic (``greedy_probability``) because a fully deterministic
        rollout policy makes every playout from a state identical — the engine
        is deterministic under ``manual_coin`` — which would collapse
        ``repeats`` to a single sample and make the leaf estimate confidently
        wrong instead of noisily right.
        """
        if self.rng.random() < self.greedy_probability and select.minCount <= 1 <= select.maxCount:
            for wanted in _ROLLOUT_PREFERENCE:
                choices = [
                    index for index, option in enumerate(select.option)
                    if option.type == wanted
                ]
                if choices:
                    return [self.rng.choice(choices)]
        actions = legal_actions(select, self.rng)
        return self.rng.choice(actions) if actions else []

    def _one(self, state: SearchState, seat: int) -> float:
        for _ in range(self.depth):
            observation = state.observation
            if observation.current.result != -1:
                return terminal_value(state, seat)
            select = observation.select
            if select is None:
                break
            action = self._choose(select)
            if not action and select.minCount > 0:
                break
            try:
                state = search_step(state.searchId, action)
            except ValueError:
                break
        return prize_differential(state, seat)


# --------------------------------------------------------------------------
# Tree


@dataclass
class Node:
    """One information set, keyed by the action path that reaches it.

    ``actor`` and ``chance`` are recorded from the first particle to expand the
    node. Both are public facts — whose turn it is, and whether the pending
    decision is a coin — so they cannot differ between particles.
    """

    actor: int = -1
    chance: bool = False
    visits: int = 0
    value_sum: float = 0.0
    children: dict[ActionKey, "Node"] = field(default_factory=dict)
    stats: dict[ActionKey, list] = field(default_factory=dict)  # key -> [N, W, prior]
    expanded: bool = False

    #: Regret-matching state, used only by ``selector="regret"``. Kept off the
    #: hot path for PUCT by lazily creating them.
    regret: dict[ActionKey, float] = field(default_factory=dict)
    strategy_sum: dict[ActionKey, float] = field(default_factory=dict)

    def q(self, key: ActionKey) -> float:
        entry = self.stats.get(key)
        if not entry or entry[0] == 0:
            return 0.0
        return entry[1] / entry[0]


@dataclass
class SearchStats:
    simulations: int = 0
    nodes: int = 0
    engine_steps: int = 0
    leaf_evaluations: int = 0
    max_depth: int = 0
    chance_nodes: int = 0
    signature_collisions: int = 0
    truncated_multiselect: int = 0
    dead_ends: int = 0


@dataclass
class SearchResult:
    """What one search produced, in the form distillation consumes.

    ``visits`` over the root's actions is the improved policy (the AlphaZero
    target); ``value`` is the root's search value (the value target). ``action``
    is what to actually play.
    """

    action: list[int]
    value: float
    visits: dict[ActionKey, int]
    root_actions: list[tuple[ActionKey, list[int], int, float]]
    stats: SearchStats

    def policy_target(self, temperature: float = 1.0) -> list[tuple[list[int], float]]:
        """Visit counts as a distribution over the root's concrete selections."""
        counts = [max(0, n) for _, _, n, _ in self.root_actions]
        if temperature != 1.0:
            counts = [c ** (1.0 / temperature) if c > 0 else 0.0 for c in counts]
        total = sum(counts) or 1.0
        return [
            (indexes, count / total)
            for (_, indexes, _, _), count in zip(self.root_actions, counts)
        ]


@dataclass
class SearchConfig:
    simulations: int = 256
    particles: int = 8
    c_puct: float = 1.5
    #: Decisions deep, not turns. A Pokémon turn is many decisions, so this is
    #: mostly "look a couple of turns ahead".
    max_depth: int = 30
    seed: int = 0
    #: Coins are chance nodes. Exposed only so a test can prove that flipping it
    #: to False makes the agent over-value 50/50 lines; never set it in training.
    coin_as_chance: bool = True
    #: "puct" or "regret". See ``ISMCTS._select_regret`` — PUCT seeks a best
    #: response to the tree's opponent model, regret matching seeks an
    #: equilibrium and needs no such model.
    selector: str = "puct"
    #: "uniform" or "heuristic". See ``heuristic_priors`` — without a prior the
    #: search spreads its budget over every option and reaches ~2 plies.
    prior: str = "uniform"


class ISMCTS:
    """Information-set MCTS with a shared tree over particle-sampled worlds."""

    def __init__(self, evaluator: Evaluator, config: SearchConfig | None = None,
                 opponent_policy=None):
        """``opponent_policy(state, actor) -> option indexes`` replaces PUCT at
        the opponent's nodes.

        Without it the search selects the opponent's moves by PUCT over uniform
        priors and a weak value head, which amounts to modelling them as a
        near-random player. Measured consequence: the search scales beautifully
        against a *genuinely* random opponent (64.0% -> 81.3% over 4 to 64
        simulations) and not at all against the frozen BC agents (21.7% / 21.7%
        / 16.7% over 16 to 256). Both halves follow from the same cause — the
        internal opponent model is accurate in the first case and wrong in the
        second, and extra simulations only converge harder onto the best
        response to whichever model is in use.
        """
        self.evaluator = evaluator
        self.config = config or SearchConfig()
        self.rng = random.Random(self.config.seed)
        self.opponent_policy = opponent_policy
        self._root_prior: list[float] | None = None

    # -- engine plumbing ---------------------------------------------------

    def set_root_prior(self, prior: list[float] | None) -> None:
        """Supply a policy prior over the *root* decision's options.

        The root is where the answer is actually taken, and it is the one
        decision whose option list is public and identical under every particle,
        so a single forward pass covers it. That matters because the obvious
        source of a good prior — a BC checkpoint — costs ~30 ms, which is
        affordable once per decision and not affordable once per node (128
        simulations would be 3.8 s).

        Worth the plumbing because of the measurement that motivated it: a BC
        policy piloting this deck scores 41.1% against the frozen field
        (n=1796), while this search scores 21.7%. A search that ignores a
        much stronger policy and rediscovers the root move from a 2-ply tree is
        throwing that away. Seeding the root prior makes the search a refinement
        of BC rather than a replacement for it.
        """
        self._root_prior = prior

    def _roots(self, obs: dict, particles: Sequence[Particle]) -> list[SearchState]:
        """One determinised root per particle.

        ``manual_coin=True`` throughout: it is what makes ``search_step`` a pure
        function (P13), so a path can be recomputed instead of stored and two
        simulations that share a prefix genuinely share it.
        """
        observation = to_observation_class(obs)
        roots = []
        for particle in particles:
            roots.append(
                search_begin(
                    observation, **particle.search_begin_args(), manual_coin=True
                )
            )
        return roots

    # -- one simulation ----------------------------------------------------

    def _select(self, node: Node, select, keys: list[ActionKey]) -> ActionKey:
        """PUCT over the actions legal in *this* determinisation.

        Restricting to ``keys`` is the information-set part: a node accumulates
        statistics from every particle, but a simulation may only choose among
        the actions its own world actually offers.

        Values are stored from the root seat's point of view throughout, so a
        node where the opponent moves selects on ``-q``. One sign, one place.
        """
        total = max(1, sum(node.stats[k][0] for k in keys if k in node.stats))
        root_sign = 1.0 if node.actor == self._seat else -1.0
        best, best_score = keys[0], -math.inf
        for key in keys:
            entry = node.stats.setdefault(key, [0, 0.0, 1.0 / len(keys)])
            exploit = root_sign * (entry[1] / entry[0]) if entry[0] else 0.0
            explore = self.config.c_puct * entry[2] * math.sqrt(total) / (1 + entry[0])
            score = exploit + explore
            if score > best_score:
                best, best_score = key, score
        return best

    def _regret_strategy(self, node: Node, keys: list[ActionKey]) -> dict[ActionKey, float]:
        """Regret matching: ``sigma(a) = max(R_a, 0) / sum_a' max(R_a', 0)``."""
        positive = {k: max(0.0, node.regret.get(k, 0.0)) for k in keys}
        total = sum(positive.values())
        if total <= 0:
            uniform = 1.0 / len(keys)
            return {k: uniform for k in keys}
        return {k: v / total for k, v in positive.items()}

    def _select_regret(self, node: Node, keys: list[ActionKey]) -> ActionKey:
        """Sample from the regret-matched strategy, then update regrets.

        PUCT converges to a *best response* against whatever the tree believes
        the opponent does; with uniform priors that belief is "plays at random",
        which is why the search scales against a random opponent and not against
        the BC agents. Regret matching instead has both players minimise regret,
        and the average strategy of self-play regret minimisation converges
        toward an equilibrium of the sampled subgame — no opponent model
        required, which is the property CFR is wanted for here.

        This is regret matching over *estimated* action values rather than exact
        counterfactual values: the update uses each action's running mean Q
        against the strategy-weighted node value, in place of the importance-
        weighted terms outcome-sampling MCCFR would compute. That makes it an
        approximation of CFR, not CFR — the sampled-subgame equilibrium
        guarantee does not strictly carry over — but it is a drop-in change to
        one selection rule instead of a separate traversal, and it tests the
        core question (equilibrium-seeking vs best-response) directly.
        """
        strategy = self._regret_strategy(node, keys)
        sign = 1.0 if node.actor == self._seat else -1.0
        # Node value under the current strategy, from the *acting* player's view.
        node_value = sum(strategy[k] * sign * node.q(k) for k in keys)
        for key in keys:
            node.regret[key] = node.regret.get(key, 0.0) + (
                sign * node.q(key) - node_value
            )
            node.strategy_sum[key] = node.strategy_sum.get(key, 0.0) + strategy[key]
        draw = self.rng.random()
        cumulative = 0.0
        for key in keys:
            cumulative += strategy[key]
            if draw <= cumulative:
                return key
        return keys[-1]

    def _simulate(self, root: Node, state: SearchState, stats: SearchStats) -> float:
        path: list[tuple[Node, ActionKey]] = []
        node = root
        depth = 0

        while True:
            observation = state.observation
            if observation.current.result != -1:
                value = terminal_value(state, self._seat)
                break
            select = observation.select
            if select is None:
                value = prize_differential(state, self._seat)
                break
            if depth >= self.config.max_depth:
                values, _ = self.evaluator([state], self._seat)
                stats.leaf_evaluations += 1
                value = values[0]
                break

            chance = self.config.coin_as_chance and is_chance(select)
            if not node.expanded:
                node.actor = observation.current.yourIndex
                node.chance = chance
                node.expanded = True
                stats.nodes += 1
                if chance:
                    stats.chance_nodes += 1
                # Expand and evaluate: the leaf's value comes from the evaluator,
                # and the descent stops here for this simulation.
                values, priors = self.evaluator([state], self._seat)
                stats.leaf_evaluations += 1
                value = values[0]
                actions = legal_actions(select, self.rng)
                if len(actions) < len(select.option) and select.maxCount > 1:
                    stats.truncated_multiselect += 1
                prior = priors[0] if priors else None
                if depth == 0 and self._root_prior is not None:
                    # The root's option list is public and identical under every
                    # particle, so one policy forward covers it.
                    if len(self._root_prior) == len(select.option):
                        prior = self._root_prior
                if prior is None and self.config.prior == "heuristic":
                    prior = heuristic_priors(select)
                for position, indexes in enumerate(actions):
                    key = action_key(select, indexes)
                    if key in node.stats:
                        stats.signature_collisions += 1
                        continue
                    if prior and indexes and all(i < len(prior) for i in indexes):
                        weight = sum(prior[i] for i in indexes) / len(indexes)
                    else:
                        weight = 1.0 / max(1, len(actions))
                    node.stats[key] = [0, 0.0, weight]
                break

            actions = legal_actions(select, self.rng)
            if not actions:
                stats.dead_ends += 1
                value = prize_differential(state, self._seat)
                break
            index_by_key = {action_key(select, a): a for a in actions}
            keys = list(index_by_key)
            for key in keys:
                node.stats.setdefault(key, [0, 0.0, 1.0 / len(keys)])

            if node.chance:
                # Uniform over the coin's faces. Not UCT: the search does not
                # get to call the flip, it only gets to average over it.
                key = self.rng.choice(keys)
            elif self.opponent_policy is not None and node.actor != self._seat:
                indexes = self.opponent_policy(state, node.actor)
                if indexes is None:
                    key = self._select(node, select, keys)
                else:
                    key = action_key(select, indexes)
                    # The modelled move need not be among the sampled
                    # multi-select candidates; it is what the opponent actually
                    # plays, so it is added rather than discarded.
                    if key not in index_by_key:
                        index_by_key[key] = list(indexes)
                        node.stats.setdefault(key, [0, 0.0, 1.0])
            elif self.config.selector == "regret":
                key = self._select_regret(node, keys)
            else:
                key = self._select(node, select, keys)

            try:
                state = search_step(state.searchId, index_by_key[key])
            except ValueError:
                stats.dead_ends += 1
                value = prize_differential(state, self._seat)
                break
            stats.engine_steps += 1
            path.append((node, key))
            node = node.children.setdefault(key, Node())
            depth += 1

        stats.max_depth = max(stats.max_depth, depth)
        # Backup. Values are root-seat-relative everywhere, so no sign flips.
        for parent, key in path:
            parent.visits += 1
            parent.value_sum += value
            entry = parent.stats[key]
            entry[0] += 1
            entry[1] += value
        root.visits += 1
        root.value_sum += value
        return value

    # -- public ------------------------------------------------------------

    def run(self, obs: dict, seat: int, particles: Sequence[Particle]) -> SearchResult:
        """Search the decision in ``obs`` and return the improved policy."""
        if not particles:
            raise ValueError("search needs at least one particle")
        self._seat = seat
        stats = SearchStats()
        root = Node()
        # ``search_end()`` recycles ids: measured, every search restarts them at
        # 0, so id 7 in this decision is an unrelated state to id 7 in the last
        # one. Anything memoising on ``searchId`` is therefore valid only within
        # a single search and must be dropped here — a stale entry does not
        # error, it silently returns another position's answer.
        if hasattr(self.opponent_policy, "clear"):
            self.opponent_policy.clear()
        roots = self._roots(obs, particles)
        try:
            for simulation in range(self.config.simulations):
                # Round-robin rather than random: with a small particle set it
                # guarantees every world is explored evenly, which matters more
                # than the independence a random draw would buy.
                self._simulate(root, roots[simulation % len(roots)], stats)
                stats.simulations += 1

            select = roots[0].observation.select
            actions = legal_actions(select, self.rng)
            rows = []
            for indexes in actions:
                key = action_key(select, indexes)
                entry = root.stats.get(key, [0, 0.0, 0.0])
                # Regret matching's answer is the average strategy, not the
                # visit count: the current strategy oscillates around the
                # equilibrium while the running average is what converges to it.
                weight = (
                    root.strategy_sum.get(key, 0.0)
                    if self.config.selector == "regret" else entry[0]
                )
                rows.append((key, indexes, weight, root.q(key)))
            rows.sort(key=lambda row: row[2], reverse=True)
            best = rows[0][1] if rows else list(range(select.minCount))
            value = root.value_sum / root.visits if root.visits else 0.0
            return SearchResult(
                action=best,
                value=value,
                visits={key: n for key, _, n, _ in rows},
                root_actions=rows,
                stats=stats,
            )
        finally:
            # One agent context per process, so this frees every particle's tree.
            search_end()
