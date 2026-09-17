"""Vectorised rollouts: N concurrent battles in one process, one batched GPU forward.

What changed and why
--------------------
``train.py``'s ``play_episode`` plays one battle at a time because ``cg.game`` can only hold
one (see ``pokeka_env``). Every forward is therefore batch 1, which is why rollouts were
pinned to the CPU and why throughput came from forking ``--workers`` processes — and why the
parent then had to *rebuild* every worker's features from shipped rows, because the feature
trees themselves are 45 MB an episode and 4.9 s to pickle against 3.3 s to play.

Here the battles are all in one process (pokeka's ``PTCGBattle`` owns its pointer), so the
loop is inverted: instead of "play an episode, decision by decision", it is "advance N
battles by one decision each". The N decisions waiting at any instant are collated into a
single forward, and the batch is *free* — it costs the same GPU call as one decision.

Three consequences, all of them the point of the change:

* **The GPU is used for rollouts, not just updates.** Batch-64 forward measured 16 ms on
  this box's GPU against 74 ms on the CPU, and flat in batch size up to 64 — so inference
  falls from 7.6 ms per decision to ~0.3 ms.
* **No worker IPC and no feature rebuild.** The ``transform()`` output is produced once, in
  the process that will run the PPO update on it, so ``rebuild_steps`` (a second full
  ``transform`` of every decision, measured about as expensive as playing the episode) is
  gone rather than merely overlapped.
* **Evaluation gets the same speedup**, which matters more than it sounds: at
  ``--eval-episodes 40`` a greedy evaluation is 120 games of pure batch-1 inference.

What did *not* change is everything the experiment is about: the features, the network, the
sequential stop-logit action distribution, the potential-based shaping, GAE, the PPO update
and the checkpoint format are all imported from ``train`` and called unmodified.

Grouping by actor
-----------------
Each tick the active slots are bucketed by *who* must choose — the learner, one of the three
frozen BC opponents, or a self-play snapshot — and each bucket gets one forward through its
own network. The learner is a little over half the buckets' worth of decisions (it also acts
on both seats), so in practice a tick is 2-4 forwards regardless of ``--num-envs``.

The frozen opponents keep ``decode_action``'s floored decode and the self-play mirror keeps
the argmax of the learner's own distribution, exactly as in the serial path; per-slot feature
memory is what moved, since one ``LiveFeatureExtractor`` per policy would interleave N
episodes' decision chains into one another. Every slot therefore owns two extractors (its
learner-seat memory and its opponent-seat memory) and resets both per episode.
"""

import collections
import sys
from pathlib import Path

import torch

from dataset import transform
from live import LiveFeatureExtractor
from observation import DEFAULT_SPEC, LEARNER_SPEC
from policy_experimental import decode_action, selection_counts

from pokeka_env import EngineError, PTCGBattle


def _train_module():
    """``train.py``'s shared helpers, whether it is running as ``__main__`` or imported.

    ``python train.py`` makes that file the ``__main__`` module, so a plain ``from train
    import ...`` here would execute a *second* copy of it under a different name — a second
    ``_frozen_modules`` cache and a second ``ActorCritic`` class, which is the kind of quiet
    duplication that makes weights load into the wrong class later.
    """
    main = sys.modules.get("__main__")
    if getattr(main, "__file__", None) and Path(main.__file__).name == "train.py":
        return main
    import train  # noqa: PLC0415 -- only reached when train.py is not the entry point

    return train


_train = _train_module()
batch_features = _train.batch_features
rollout_action = _train.rollout_action
state_potential = _train.state_potential


class Slot:
    """One concurrent battle, with its own engine pointer and per-seat feature memory.

    The extractors are created once and ``reset`` per episode (as ``FrozenPolicy`` and
    ``ChallengerPolicy`` do), so a slot is reused across episodes and only the battle is
    torn down.
    """

    def __init__(self) -> None:
        self.battle = PTCGBattle()
        # 30-frame temporal window for the learner, 60 for the frozen opponents (see
        # observation.LEARNER_SPEC).
        self.learner_extractor = LiveFeatureExtractor(spec=LEARNER_SPEC)
        self.opponent_extractor = LiveFeatureExtractor(spec=DEFAULT_SPEC)
        self.kind = "frozen"
        self.opponent_index = 0
        self.seat = 0
        self.episode = -1
        self.steps: list[dict] = []
        self.n_selects = 0
        self.result: int | None = None
        self.terminal_potential = 0.0
        self.finished = False

    def start(self, task, episode: int, learner_deck, opponent_deck) -> None:
        self.kind, self.opponent_index, self.seat = task
        self.episode = episode
        self.steps = []
        self.n_selects = 0
        self.result = None
        self.terminal_potential = 0.0
        self.finished = False
        decks = [None, None]
        decks[self.seat] = learner_deck
        decks[1 - self.seat] = opponent_deck
        # ``PTCGBattle.start`` overwrites ``ptr`` without freeing the previous battle, so the
        # finish has to be explicit or a slot leaks one native battle per episode.
        self.battle.finish()
        self.battle.start(decks[0], decks[1])
        self.learner_extractor.reset(episode)
        self.opponent_extractor.reset(episode)

    @property
    def actor(self) -> tuple[str, int]:
        """``("learner", -1)`` or ``(kind, opponent_index)`` for the seat that must choose."""
        if self.battle.obs["current"]["yourIndex"] == self.seat:
            return ("learner", -1)
        return (self.kind, self.opponent_index)

    def abandon(self) -> None:
        """Drop this episode: ``result is None`` means its transitions are discarded rather
        than labelled with a guessed outcome, matching ``play_episode``."""
        self.result = None
        self.steps = []
        self.finished = True

    def payload(self) -> dict:
        return {
            "episode": self.episode,
            "kind": self.kind,
            "opponent_index": self.opponent_index,
            "seat": self.seat,
            "result": self.result,
            "steps": self.n_selects,
            "episode_steps": self.steps if self.result is not None else [],
            "terminal_potential": self.terminal_potential,
        }


class VecRunner:
    """``num_envs`` battle slots driven in lockstep, with batched inference.

    ``model`` is the learner (an ``ActorCritic``); ``opponents`` are ``FrozenPolicy``
    instances, of which only ``.network`` and ``.deck`` are used here since their extractors
    have moved into the slots. ``mirrors`` are resident ``ActorCritic`` copies of the
    self-play snapshot pool, index-aligned with it — kept resident because loading a state
    dict per tick would cost more than the games it plays, and 10 MB a snapshot is nothing
    next to the batch.
    """

    def __init__(self, model, opponents, learner_deck, device, args, num_envs, mirrors=()):
        self.model = model
        self.opponents = list(opponents)
        self.learner_deck = learner_deck
        self.device = device
        self.args = args
        self.num_envs = max(1, int(num_envs))
        self.mirrors = list(mirrors)
        self.slots = [Slot() for _ in range(self.num_envs)]
        self.greedy = False
        self.record = True
        #: engine failures seen this run, reported by the caller rather than swallowed
        self.abandoned = 0

    # -- one batched decision per actor group ------------------------------

    def _extract(self, group, learner: bool):
        """``(samples, live_slots)`` — the feature tree for each slot's pending decision.

        A slot whose extraction raises is abandoned and dropped from the group rather than
        taking the whole batch down with it.
        """
        samples, live = [], []
        for slot in group:
            extractor = slot.learner_extractor if learner else slot.opponent_extractor
            try:
                samples.append(transform(extractor(slot.battle.obs)))
            except (ValueError, IndexError, KeyError, RuntimeError) as error:
                print(f"[vec] episode {slot.episode} unfeaturisable: {error!r}", flush=True)
                slot.abandon()
                continue
            live.append(slot)
        return samples, live

    def _act_learner(self, group) -> None:
        samples, group = self._extract(group, learner=True)
        if not group:
            return
        features = batch_features(samples, self.device)
        with torch.no_grad():
            logits, stop_logit, value, options_mask = self.model(features)
        min_count, max_count = selection_counts(features)
        # One transfer for the whole group. Sampling walks the option set per row (masking
        # picks out as it goes) and needs no further forward, so doing it on CPU tensors
        # costs nothing and avoids a device sync per row.
        logits = logits.cpu()
        stop_logit = stop_logit.cpu()
        options_mask = options_mask.cpu()
        value = value.cpu()
        min_count, max_count = min_count.cpu(), max_count.cpu()
        for i, slot in enumerate(group):
            # The potential is Phi(s) of the state being acted in, so it is read before the
            # action is submitted. ``yourIndex`` is the learner's own seat here.
            potential = state_potential(
                slot.battle.obs, slot.seat, self.args.attack_shaping
            )
            action, log_prob = rollout_action(
                logits[i : i + 1],
                stop_logit[i : i + 1],
                options_mask[i : i + 1],
                min_count[i : i + 1],
                max_count[i : i + 1],
                greedy=self.greedy,
            )
            slot.learner_extractor.record_action(action)
            if self.record and not self.greedy:
                slot.steps.append(
                    {
                        "sample": samples[i],
                        "action": action,
                        "log_prob": log_prob,
                        "value": float(value[i]),
                        "potential": potential,
                    }
                )
            self._submit(slot, action)

    def _act_frozen(self, group, index: int) -> None:
        samples, group = self._extract(group, learner=False)
        if not group:
            return
        features = batch_features(samples, self.device)
        with torch.no_grad():
            logits = self.opponents[index].network(features)
        min_count, max_count = selection_counts(features)
        options_mask = features["decision_context"]["options"]["options_mask"].squeeze(1)
        # ``train.greedy_action`` for a whole batch: the same ``decode_action`` with the same
        # ``min_count.clamp(min=1)`` floor, reading every row instead of only the first.
        actions = decode_action(
            logits.cpu(), options_mask.cpu(), min_count.cpu().clamp(min=1), max_count.cpu()
        )
        for slot, action in zip(group, actions):
            slot.opponent_extractor.record_action(action)
            self._submit(slot, action)

    def _act_mirror(self, group, index: int) -> None:
        """A self-play snapshot on the opponent seat. Greedy — the mode of the learner's own
        distribution, matching the frozen agents' determinism (``ChallengerPolicy.greedy``)."""
        samples, group = self._extract(group, learner=False)
        if not group:
            return
        features = batch_features(samples, self.device)
        with torch.no_grad():
            logits, stop_logit, _, options_mask = self.mirrors[index](features)
        min_count, max_count = selection_counts(features)
        logits, stop_logit, options_mask = logits.cpu(), stop_logit.cpu(), options_mask.cpu()
        min_count, max_count = min_count.cpu(), max_count.cpu()
        for i, slot in enumerate(group):
            action, _ = rollout_action(
                logits[i : i + 1],
                stop_logit[i : i + 1],
                options_mask[i : i + 1],
                min_count[i : i + 1],
                max_count[i : i + 1],
                greedy=True,
            )
            slot.opponent_extractor.record_action(action)
            self._submit(slot, action)

    def _submit(self, slot: Slot, action) -> None:
        try:
            slot.battle.select(action)
        except (ValueError, IndexError, RuntimeError, EngineError) as error:
            print(f"[vec] episode {slot.episode} abandoned: {error!r}", flush=True)
            slot.abandon()
            return
        slot.n_selects += 1
        if slot.battle.done:
            slot.result = slot.battle.result
            # Phi(s') of the board after the game ended — the last transition's next-state
            # potential, which no decision frame can carry.
            slot.terminal_potential = state_potential(
                slot.battle.obs, slot.seat, self.args.attack_shaping
            )
            slot.finished = True
        elif slot.n_selects >= self.args.max_steps:
            slot.abandon()

    # -- the loop ----------------------------------------------------------

    def collect(self, tasks, first_episode: int, greedy: bool = False, record: bool = True):
        """Play ``tasks`` — a list of ``(kind, index, seat)`` — yielding one payload per
        episode as it finishes.

        Episodes are numbered ``first_episode + position in tasks``, so the sequence a
        ``schedule`` produced is preserved even though completion order is not. ``greedy``
        makes the learner play its argmax (evaluation); ``record`` off skips storing
        transitions, which is what keeps an evaluation from holding 45 MB an episode.
        """
        self.greedy = greedy
        self.record = record
        queue = collections.deque(
            (first_episode + offset, task) for offset, task in enumerate(tasks)
        )
        slots: list[Slot | None] = [None] * min(self.num_envs, len(tasks))
        while True:
            for index in range(len(slots)):
                while slots[index] is None and queue:
                    episode, task = queue.popleft()
                    slot = self.slots[index]
                    try:
                        slot.start(task, episode, self.learner_deck, self._deck_for(task))
                    except (ValueError, RuntimeError, EngineError) as error:
                        print(f"[vec] episode {episode} failed to start: {error!r}", flush=True)
                        self.abandoned += 1
                        yield {
                            "episode": episode, "kind": task[0], "opponent_index": task[1],
                            "seat": task[2], "result": None, "steps": 0,
                            "episode_steps": [], "terminal_potential": 0.0,
                        }
                        continue
                    slots[index] = slot
            active = [slot for slot in slots if slot is not None]
            if not active:
                return
            groups: dict[tuple[str, int], list[Slot]] = collections.defaultdict(list)
            for slot in active:
                groups[slot.actor].append(slot)
            for (kind, index), group in groups.items():
                if kind == "learner":
                    self._act_learner(group)
                elif kind == "self":
                    self._act_mirror(group, index)
                else:
                    self._act_frozen(group, index)
            for index, slot in enumerate(slots):
                if slot is not None and slot.finished:
                    slots[index] = None
                    if slot.result is None:
                        self.abandoned += 1
                    yield slot.payload()

    def _deck_for(self, task) -> list[int]:
        """The opponent's decklist. A frozen agent brings its own (a BC policy piloting a
        list it never saw the expert play is a weaker opponent for reasons unrelated to the
        learner); a self-play snapshot pilots the learner's deck."""
        kind, index, _ = task
        return self.learner_deck if kind == "self" else self.opponents[index].deck

    def close(self) -> None:
        for slot in self.slots:
            slot.battle.finish()


def evaluate(runner: VecRunner, opponents, episodes: int, max_steps: int, base_episode: int):
    """``train.evaluate``'s series, played vectorised.

    Same contract: greedy learner against each greedy opponent, seats alternating, returning
    one rate per opponent, the same split by seat, and ``win_rate`` for the mean. All
    ``len(opponents) * episodes`` games are handed to the runner at once, so a 120-game
    evaluation costs the batched inference of ~24 games rather than 120 serial ones.
    """
    tasks = [
        ("frozen", index, i % 2)
        for index in range(len(opponents))
        for i in range(episodes)
    ]
    counts = {
        index: {0: [0, 0], 1: [0, 0]} for index in range(len(opponents))
    }  # opponent -> seat -> [wins, played]
    for item in runner.collect(tasks, base_episode, greedy=True, record=False):
        if item["result"] is None:
            continue
        seat, index = item["seat"], item["opponent_index"]
        counts[index][seat][1] += 1
        if item["result"] == seat:
            counts[index][seat][0] += 1

    series: dict[str, float] = {}
    seat_totals = {0: [0, 0], 1: [0, 0]}
    for index, opponent in enumerate(opponents):
        wins = played = 0
        for seat in (0, 1):
            won, saw = counts[index][seat]
            series[f"{opponent.name}_seat{seat}"] = won / saw if saw else float("nan")
            wins, played = wins + won, played + saw
            seat_totals[seat][0] += won
            seat_totals[seat][1] += saw
        series[opponent.name] = wins / played if played else float("nan")
    rates = [
        series[opponent.name]
        for opponent in opponents
        if series[opponent.name] == series[opponent.name]
    ]
    series["win_rate"] = sum(rates) / len(rates) if rates else float("nan")
    for seat in (0, 1):
        won, saw = seat_totals[seat]
        series[f"win_rate_seat{seat}"] = won / saw if saw else float("nan")
    return series
