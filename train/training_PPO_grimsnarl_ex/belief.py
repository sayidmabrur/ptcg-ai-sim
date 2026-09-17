"""Particle sampling over the hidden state, for search.

``cg.api.search_begin`` instantiates a complete-information world from
predicted card IDs for every hidden zone. Probe P9 established that it
validates *nothing* semantic: it accepted four copies of a card sitting in the
opponent's discard pile as the opponent's hand. P4 adds that a superset is
accepted too. So the engine will happily plan inside a world the real game
could never be in, and nothing downstream will ever say so — the search just
returns confident nonsense.

Legality is therefore this module's job, and it is the whole reason it exists
separately from the search that consumes it.

What is knowable, and what is guessed
-------------------------------------
The two sides are not symmetric, and conflating them is the easy mistake.

*Your own hidden zones are not a belief at all.* You know your 60-card list.
Subtract your hand, discard, board (including attached energy, tools and
pre-evolutions) and any revealed prize, and what remains is **exactly** the
multiset in deck ∪ prizes. The only thing sampled is the partition between
those two and the deck's order. ``train.py`` gives its network ``deck_count``
as a single scalar and asks it to recover the rest from pooled embeddings of
hand and discard; that is a sparse-recovery problem over a 1267-card vocab
which it cannot solve and does not have to.

*The opponent's zones are a genuine posterior.* Their decklist is unknown a
priori, so this keeps a hypothesis set and hard-filters it: a candidate list is
impossible the moment they reveal a card it does not contain, or a fifth copy
of something it holds four of. Against the three frozen agents that filter
usually collapses to a single list within a few turns, which is real inference
and costs nothing. Their deck/prize/hand split is then partitioned uniformly —
the one place in here that is a genuine approximation, and the designated seam
for a learned model (see ``partition_hidden``).

Conventions the engine imposes
------------------------------
Measured in ``probe_search_api.py``, because none of it is documented:

* **The last element of a deck list is the top.** P8 injected a known order and
  read the ``DRAW`` log: the drawn card was ``your_deck[-1]``. ``prize`` in
  ``cg/api.py`` is documented the same way ("the first element is the bottom").
  Getting this backwards produces particles that are individually legal and
  collectively wrong, which is the worst failure mode available here.
* Lists must be **at least** as long as the zone they fill; longer is accepted,
  shorter raises.
* Every id must exist in the card table, or ``search_begin`` raises
  ``Invalid Card ID``.
* A face-down opponent active requires a predicted **Pokémon** id.
* When ``select.deck`` is populated the engine uses the real deck and ignores
  ``your_deck`` entirely — at those frames your own deck is not hidden.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from cg.api import CardType, all_card_data

# --------------------------------------------------------------------------
# Card table
#
# Loaded once. ``search_begin`` rejects unknown ids outright, and a particle
# that names a non-Pokémon as a face-down active is rejected too, so both
# checks want this.

_CARD_TYPE: dict[int, int] = {}


def _card_table() -> dict[int, int]:
    global _CARD_TYPE
    if not _CARD_TYPE:
        _CARD_TYPE = {card.cardId: int(card.cardType) for card in all_card_data()}
    return _CARD_TYPE


def is_pokemon(card_id: int) -> bool:
    return _card_table().get(card_id) == int(CardType.POKEMON)


def is_known_card(card_id: int) -> bool:
    return card_id in _card_table()


# --------------------------------------------------------------------------
# Reading the public state


def _collect_seen(node: Any, into: dict[int, int]) -> None:
    """Map ``serial -> card id`` for every card reachable from a state fragment.

    Keyed on ``serial``, not counted directly, and that is the point: serial is
    unique per card for the whole match, so a card reachable from two places at
    once — the Supporter mid-resolution that is also named by
    ``selection.effect``, an ability's source Pokémon that is also on the bench —
    is recorded once. Counting occurrences instead double-subtracts it from the
    pool, which shows up as a phantom shortfall.

    Recursive because a Pokémon carries ``energyCards``, ``tools`` and
    ``preEvolution`` — all cards out of the same 60, all of which must come off
    the pool. Missing them inflates it and every particle is then over-full.

    ``None`` entries are face-down (an unrevealed prize, a hidden active) and
    contribute nothing, which is exactly right: they are the hidden part.
    """
    if node is None:
        return
    if isinstance(node, list):
        for item in node:
            _collect_seen(item, into)
        return
    if isinstance(node, dict):
        if "id" in node and node.get("serial") is not None:
            into[int(node["serial"])] = int(node["id"])
        for key in ("energyCards", "tools", "preEvolution"):
            if key in node:
                _collect_seen(node[key], into)


#: Zones inside a ``PlayerState`` whose contents are card-identifiable.
#: ``hand`` is included and is ``None`` for the opponent, which ``_collect_ids``
#: skips — so the same call is correct from either point of view.
_PLAYER_ZONES = ("hand", "discard", "prize", "active", "bench")


def visible_cards(
    state: dict,
    player_index: int,
    selection: dict | None = None,
    count_hand: bool = True,
) -> Counter:
    """Every card of ``player_index``'s 60 whose location is currently known.

    Subtracting this from their decklist leaves precisely the pool a particle
    has to partition. Shared zones (``stadium``, ``looking``) sit outside both
    player structs but still came out of somebody's deck, so they are attributed
    by the ``playerIndex`` the engine stamps on each card.

    ``count_hand=False`` models *the opponent's* knowledge of this player. The
    engine only redacts ``hand`` for the non-owner, so at a frame this player
    owns, their real hand is sitting right there in the struct — reading it
    while claiming to compute what their opponent believes silently subtracts
    the very cards the belief is supposed to be over. That produced a pool the
    true hand was not even drawable from on 71% of frames, which is the sort of
    error that makes a search confidently solve the wrong game.

    ``selection`` matters at deck-search frames: ``select.deck`` lists the deck's
    real contents, so those cards are known rather than hidden.
    """
    seen: dict[int, int] = {}
    player = state["players"][player_index]
    for zone in _PLAYER_ZONES:
        if zone == "hand" and not count_hand:
            continue
        _collect_seen(player.get(zone), seen)
    for zone in ("stadium", "looking"):
        for card in state.get(zone) or []:
            if isinstance(card, dict) and card.get("playerIndex") == player_index:
                _collect_seen(card, seen)
    if selection:
        # A card mid-resolution has left its zone and not yet reached the next
        # one: played from hand, not in the discard, invisible everywhere. It is
        # the surplus-of-one that made 18% of pools fail to close. Attributed by
        # the card's own playerIndex, and deduplicated by serial above so naming
        # a card already on the board costs nothing.
        for key in ("effect", "contextCard"):
            card = selection.get(key)
            if isinstance(card, dict) and card.get("playerIndex") == player_index:
                _collect_seen(card, seen)
        if selection.get("deck") and player_index == state.get("yourIndex"):
            _collect_seen(selection["deck"], seen)
    return Counter(seen.values())


def pinned_prizes(state: dict, player_index: int) -> dict[int, int]:
    """``slot -> card_id`` for prizes whose face is already known.

    Some effects reveal a prize without taking it. Those slots are not free to
    sample: a particle that reshuffles a revealed prize contradicts something
    both players can see.
    """
    prize = state["players"][player_index].get("prize") or []
    return {
        index: int(card["id"])
        for index, card in enumerate(prize)
        if isinstance(card, dict) and card.get("id") is not None
    }


@dataclass
class PoolReport:
    """The hidden pool for one player, plus whether the accounting closed.

    ``shortfall``/``surplus`` are not cosmetic. If a card left the 60 by a route
    ``visible_cards`` does not model — a lost zone, an unusual shuffle-back —
    the pool will not match the counts the engine reports, and every particle
    built from it is malformed in a way nothing downstream will catch. Surfacing
    the mismatch as data is what lets ``probe_belief.py`` find the gap instead of
    the search quietly degrading.
    """

    pool: Counter
    required: int
    pinned: dict[int, int] = field(default_factory=dict)
    #: Face-down Pokémon in play. Out of the deck, but showing as ``None``, so
    #: they consume from the pool without appearing in ``visible_cards``.
    face_down_in_play: int = 0
    #: True at a deck-search frame, where ``select.deck`` reveals the real deck
    #: and ``search_begin`` ignores the prediction entirely.
    deck_revealed: bool = False

    @property
    def available(self) -> int:
        return sum(self.pool.values())

    @property
    def shortfall(self) -> int:
        return max(0, self.required - self.available)

    @property
    def surplus(self) -> int:
        return max(0, self.available - self.required)

    @property
    def closed(self) -> bool:
        return self.available == self.required


def hidden_pool(
    state: dict,
    player_index: int,
    decklist: Sequence[int],
    include_hand: bool,
    selection: dict | None = None,
) -> PoolReport:
    """The multiset of ``player_index``'s cards whose location is unknown.

    ``include_hand`` is the self/opponent asymmetry in one flag: your own hand
    is visible so your pool is deck + prizes, while theirs is deck + prizes +
    hand. It also decides whether the hand counts as *visible*, since a frame's
    owner can read their own hand out of the struct even when the belief being
    computed is the one their opponent holds.

    Two zones make the arithmetic non-obvious, and both were found by
    ``probe_belief.py`` rather than by reading the engine docs:

    *Face-down Pokémon in play.* During setup both players place an active (and
    bench) face-down, and the engine reports those slots as ``None``. Those
    cards have left the deck but appear nowhere, so they have to be charged
    against the pool or it comes out one too large.

    *A revealed deck.* At a deck-search frame ``select.deck`` lists the real
    contents, so the deck stops being hidden at all and drops out of the
    requirement — otherwise the pool looks 46 cards short.
    """
    player = state["players"][player_index]
    pool = Counter(decklist) - visible_cards(
        state, player_index, selection, count_hand=not include_hand
    )
    pinned = pinned_prizes(state, player_index)
    face_down = sum(
        1 for slot in list(player.get("active") or []) + list(player.get("bench") or [])
        if slot is None
    )
    deck_revealed = bool(
        selection and selection.get("deck") is not None
        and player_index == state.get("yourIndex")
    )
    required = (
        (0 if deck_revealed else int(player["deckCount"]))
        + max(0, len(player.get("prize") or []) - len(pinned))
        + (int(player["handCount"]) if include_hand else 0)
        + face_down
    )
    return PoolReport(
        pool=pool, required=required, pinned=pinned,
        face_down_in_play=face_down, deck_revealed=deck_revealed,
    )


# --------------------------------------------------------------------------
# Opponent decklist posterior


class OpponentDeckPosterior:
    """Hard-filtered belief over which decklist the opponent is playing.

    Only exact elimination, no soft scoring: a candidate dies when the opponent
    reveals a card it cannot contain. That is a logical deduction rather than a
    model, so it is always safe, and against a known field of three it usually
    resolves within a few turns. A learned prior over the survivors would go on
    top of this, never underneath it — nothing should ever re-admit a list the
    opponent has already disproved.
    """

    def __init__(self, candidates: Sequence[Sequence[int]], prior: Sequence[float] | None = None):
        if not candidates:
            raise ValueError("need at least one candidate decklist")
        self.candidates = [tuple(deck) for deck in candidates]
        self._counts = [Counter(deck) for deck in self.candidates]
        weight = prior if prior is not None else [1.0] * len(self.candidates)
        if len(weight) != len(self.candidates):
            raise ValueError("prior length must match candidates")
        self.prior = [float(w) for w in weight]
        self.alive = list(range(len(self.candidates)))

    def update(self, state: dict, opponent_index: int) -> list[int]:
        """Eliminate candidates contradicted by what the opponent has shown."""
        seen = visible_cards(state, opponent_index)
        self.alive = [
            index for index in self.alive
            if not (seen - self._counts[index])
        ]
        if not self.alive:
            # Every hypothesis is dead: the opponent is playing something not in
            # the candidate set. Falling back to the full set keeps the sampler
            # producing particles (a wrong list still yields legal-shaped
            # worlds) instead of raising mid-rollout, and the caller can watch
            # ``exhausted`` to know the hypothesis set needs widening.
            self.alive = list(range(len(self.candidates)))
            self.exhausted = True
        return self.alive

    exhausted = False

    def weights(self) -> list[float]:
        total = sum(self.prior[i] for i in self.alive) or 1.0
        return [self.prior[i] / total for i in self.alive]

    def sample(self, rng: random.Random) -> int:
        return rng.choices(self.alive, weights=self.weights(), k=1)[0]

    def resolved(self) -> bool:
        return len(self.alive) == 1


# --------------------------------------------------------------------------
# Particles


@dataclass(frozen=True)
class Particle:
    """One complete assignment of every hidden zone.

    Every list is ordered **bottom first, top last** — the engine draws from the
    end (probe P8). ``search_begin`` consumes these directly.
    """

    your_deck: tuple[int, ...]
    your_prize: tuple[int, ...]
    opponent_deck: tuple[int, ...]
    opponent_prize: tuple[int, ...]
    opponent_hand: tuple[int, ...]
    opponent_active: tuple[int, ...]
    opponent_decklist_index: int

    def search_begin_args(self) -> dict[str, list[int]]:
        return {
            "your_deck": list(self.your_deck),
            "your_prize": list(self.your_prize),
            "opponent_deck": list(self.opponent_deck),
            "opponent_prize": list(self.opponent_prize),
            "opponent_hand": list(self.opponent_hand),
            "opponent_active": list(self.opponent_active),
        }


def _flatten(pool: Counter) -> list[int]:
    return [card for card, count in pool.items() for _ in range(count)]


def partition_hidden(
    pool: Counter,
    deck_count: int,
    prize_slots: int,
    hand_count: int,
    rng: random.Random,
) -> tuple[list[int], list[int], list[int]]:
    """Split an opponent pool into (deck, prize, hand), bottom-first.

    Uniform over the pool, which is the maximum-entropy choice and the one
    honest default given no model of how they play. It is also the single
    approximation in this module and the designated seam: a learned hand model
    would replace this function and nothing else, because every consistency
    guarantee above it is enforced on the *pool*, not on the split.

    A uniform split is a real understatement of what is knowable — a player who
    just searched their deck is holding what they searched for, not a random
    sample. That is exactly the gradient a learned predictor would supply.
    """
    cards = _flatten(pool)
    rng.shuffle(cards)
    hand = cards[:hand_count]
    prize = cards[hand_count:hand_count + prize_slots]
    deck = cards[hand_count + prize_slots:hand_count + prize_slots + deck_count]
    return deck, prize, hand


def _restore_pinned(sampled: list[int], pinned: dict[int, int], total: int) -> list[int]:
    """Rebuild a prize array with revealed faces back in their own slots."""
    out: list[int] = []
    free = iter(sampled)
    for slot in range(total):
        if slot in pinned:
            out.append(pinned[slot])
        else:
            out.append(next(free, sampled[-1] if sampled else 0))
    return out


class BeliefSampler:
    """Draws legal particles from the current public state.

    One instance per episode per seat. ``observe`` folds each frame into the
    opponent-decklist posterior; ``sample`` draws worlds conditioned on it.
    """

    def __init__(
        self,
        own_decklist: Sequence[int],
        opponent_candidates: Sequence[Sequence[int]],
        rng: random.Random | None = None,
    ) -> None:
        self.own_decklist = tuple(own_decklist)
        self.posterior = OpponentDeckPosterior(opponent_candidates)
        self.rng = rng or random.Random()
        self._candidates = [tuple(deck) for deck in opponent_candidates]

    def reset(self) -> None:
        self.posterior.alive = list(range(len(self._candidates)))
        self.posterior.exhausted = False

    def observe(self, obs: dict, seat: int) -> None:
        self.posterior.update(obs["current"], 1 - seat)

    def sample(self, obs: dict, seat: int, count: int = 1) -> list[Particle]:
        state = obs["current"]
        selection = obs.get("select") or {}
        own, opponent = state["players"][seat], state["players"][1 - seat]

        own_report = hidden_pool(state, seat, self.own_decklist, False, selection)
        own_prize_total = len(own.get("prize") or [])
        opp_prize_total = len(opponent.get("prize") or [])
        # A face-down active is only ever the opponent's from our seat; the
        # engine demands a Pokémon id for it and rejects the call otherwise.
        opp_active = opponent.get("active") or []
        needs_active = len(opp_active) > 0 and opp_active[0] is None

        particles: list[Particle] = []
        for _ in range(count):
            deck_index = self.posterior.sample(self.rng)
            opp_report = hidden_pool(
                state, 1 - seat, self._candidates[deck_index], True
            )

            own_cards = _flatten(own_report.pool)
            self.rng.shuffle(own_cards)
            own_free_prizes = own_prize_total - len(own_report.pinned)
            your_prize = _restore_pinned(
                own_cards[:own_free_prizes], own_report.pinned, own_prize_total
            )
            your_deck = own_cards[own_free_prizes:own_free_prizes + int(own["deckCount"])]

            opp_deck, opp_prize_free, opp_hand = partition_hidden(
                opp_report.pool,
                int(opponent["deckCount"]),
                opp_prize_total - len(opp_report.pinned),
                int(opponent["handCount"]),
                self.rng,
            )
            opponent_prize = _restore_pinned(
                opp_prize_free, opp_report.pinned, opp_prize_total
            )

            active: tuple[int, ...] = ()
            if needs_active:
                # Must be a Pokémon, and it must come out of *their* pool — a
                # prediction naming a card they cannot own is exactly the class
                # of thing the engine will accept and should not.
                choices = [c for c in opp_deck + opp_hand if is_pokemon(c)]
                if not choices:
                    choices = [c for c in _flatten(opp_report.pool) if is_pokemon(c)]
                if choices:
                    active = (self.rng.choice(choices),)

            particles.append(
                Particle(
                    your_deck=tuple(your_deck),
                    your_prize=tuple(your_prize),
                    opponent_deck=tuple(opp_deck),
                    opponent_prize=tuple(opponent_prize),
                    opponent_hand=tuple(opp_hand),
                    opponent_active=active,
                    opponent_decklist_index=deck_index,
                )
            )
        return particles


# --------------------------------------------------------------------------
# Validation
#
# This is the check the engine does not do. It is deliberately a separate
# function returning a list of strings rather than assertions inside the
# sampler: the search's hot path should not pay for it, but every test, and any
# run with a new decklist or a new engine build, should.


def validate_particle(
    particle: Particle,
    obs: dict,
    seat: int,
    own_decklist: Sequence[int],
    opponent_decklist: Sequence[int] | None = None,
) -> list[str]:
    """Every way this particle contradicts the public state. Empty is good."""
    state = obs["current"]
    selection = obs.get("select") or {}
    own, opponent = state["players"][seat], state["players"][1 - seat]
    problems: list[str] = []

    def check_ids(name: str, cards: Iterable[int]) -> None:
        bad = sorted({c for c in cards if not is_known_card(c)})
        if bad:
            problems.append(f"{name}: unknown card ids {bad[:5]}")

    for name, cards in (
        ("your_deck", particle.your_deck), ("your_prize", particle.your_prize),
        ("opponent_deck", particle.opponent_deck),
        ("opponent_prize", particle.opponent_prize),
        ("opponent_hand", particle.opponent_hand),
        ("opponent_active", particle.opponent_active),
    ):
        check_ids(name, cards)

    # Lengths: the engine raises on short, silently accepts long.
    deck_search = selection.get("deck") is not None
    expected = [
        ("your_deck", len(particle.your_deck), int(own["deckCount"]), deck_search),
        ("your_prize", len(particle.your_prize), len(own.get("prize") or []), False),
        ("opponent_deck", len(particle.opponent_deck), int(opponent["deckCount"]), False),
        ("opponent_prize", len(particle.opponent_prize), len(opponent.get("prize") or []), False),
        ("opponent_hand", len(particle.opponent_hand), int(opponent["handCount"]), False),
    ]
    for name, got, want, exempt in expected:
        if exempt:
            continue
        if got != want:
            problems.append(f"{name}: {got} cards, state says {want}")

    # Conservation: hidden + visible must not exceed the decklist. This is the
    # check that catches P9's failure — claiming a card that is provably
    # elsewhere shows up here as an over-count against the 60.
    own_total = Counter(particle.your_deck) + Counter(particle.your_prize)
    own_over = own_total + visible_cards(state, seat, selection) - Counter(own_decklist)
    if own_over:
        problems.append(f"own side over-counts vs decklist: {dict(list(own_over.items())[:5])}")

    if opponent_decklist is not None:
        opp_total = (
            Counter(particle.opponent_deck) + Counter(particle.opponent_prize)
            + Counter(particle.opponent_hand)
        )
        opp_over = opp_total + visible_cards(state, 1 - seat) - Counter(opponent_decklist)
        if opp_over:
            problems.append(
                f"opponent side over-counts vs decklist: {dict(list(opp_over.items())[:5])}"
            )

    # Revealed prizes must stay where they were revealed.
    for slot, card_id in pinned_prizes(state, seat).items():
        if slot < len(particle.your_prize) and particle.your_prize[slot] != card_id:
            problems.append(f"your_prize slot {slot} is revealed as {card_id}")
    for slot, card_id in pinned_prizes(state, 1 - seat).items():
        if slot < len(particle.opponent_prize) and particle.opponent_prize[slot] != card_id:
            problems.append(f"opponent_prize slot {slot} is revealed as {card_id}")

    active = opponent.get("active") or []
    if len(active) > 0 and active[0] is None:
        if not particle.opponent_active:
            problems.append("opponent active is face-down but no prediction given")
        elif not is_pokemon(particle.opponent_active[0]):
            problems.append(f"opponent_active {particle.opponent_active[0]} is not a Pokemon")
    elif particle.opponent_active:
        # Harmless (the API zeroes it) but a sign the caller misread the state.
        problems.append("opponent_active given but their active is not face-down")

    return problems
