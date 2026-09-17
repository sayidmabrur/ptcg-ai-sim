"""Belief as *state*: what is known and what is derivable, per seat, per frame.

This is not a history and it is deliberately not a sequence model. ``decision_chain``
and ``opponent_history`` process **past actions** — which selection was offered, which
option was taken — and that is a separate concern from what the game has *revealed*.
Neither of them ever carried a reveal: at a deck-search frame the engine populates
``select.deck`` with the deck's real contents, and ``vocab.from_selection`` reduces it to
``deck_size=len(deck)``. The card ids are dropped. So the single most valuable piece of
information in the game — "these are the cards left, and these are my prizes" — has never
reached the policy in any form, through the history or otherwise.

What is knowable, from ``belief.py``'s own accounting
----------------------------------------------------
*Your own hidden zones are not a belief.* You know your 60-card list. Subtract every card
whose location is visible — hand, discard, board, attached energy, tools, pre-evolutions,
the stadium, cards mid-resolution — and the remainder is **exactly** the multiset in
deck ∪ prizes. No sampling, no posterior, no approximation. ``train.py`` was handing the
network ``deck_count`` as one scalar and asking it to recover this by sparse recovery over
a 1,267-card vocabulary from pooled embeddings of hand and discard.

*Prizes become exact after any deck search, and stay exact.* ``visible_cards`` counts
``select.deck`` as seen at a search frame, so ``remaining`` at that frame is the prize
multiset itself. Prize contents never change afterwards — a prize card only leaves by
being taken — so one Great Ball converts the prize unknown into perfect information for
the rest of the game. That is the mechanic this module exists to expose.

*The opponent's zones are a genuine posterior and are deliberately not modelled here.*
Their decklist is unknown a priori, and the only candidate lists available on this box are
the three frozen agents' own decks — so a posterior over them would let the policy exploit
knowing the exact opposing deck, which is not information it has on the ladder. What is
kept instead is the part that is both exact and available anywhere: the multiset of their
cards this episode has actually shown.
"""

from collections import Counter

from belief import visible_cards


class BeliefTracker:
    """Per-episode belief accumulator for one seat's point of view.

    Two of the three outputs need no memory at all — they are functions of the current
    frame plus the decklist. The two that do need memory are the reasons this is a class:
    a deduced prize multiset (established at a search frame, valid forever after) and the
    opponent's revealed cards (accumulated, since a card seen on turn 3 and discarded on
    turn 5 is still known to be in their 60).
    """

    def __init__(self, decklist: list[int] | None = None) -> None:
        #: Our own 60. ``None`` means "not supplied" — every own-side field then comes back
        #: empty rather than wrong, which is what keeps the offline replay path working
        #: (a parquet row carries no decklist).
        self.decklist = Counter(decklist) if decklist else None
        self.reset()

    def reset(self) -> None:
        self._known_prize: Counter | None = None
        self._opponent_revealed: Counter = Counter()

    # -- the three quantities ---------------------------------------------

    def _remaining(self, state: dict, seat: int, selection: dict | None) -> Counter:
        """The exact multiset in our deck ∪ prizes (∪ nothing else).

        At a deck-search frame the deck is visible, so this collapses to the prizes —
        which is precisely what makes the deduction below exact rather than heuristic.
        """
        if self.decklist is None:
            return Counter()
        # ``+Counter()`` drops the zero and negative entries subtraction can leave. A
        # negative count would mean a card left the 60 by a route visible_cards does not
        # model; belief.py's PoolReport is the place that reports on that, and a feature
        # tree is not the place to raise about it.
        return +(self.decklist - visible_cards(state, seat, selection, count_hand=True))

    def _deduce_prizes(self, state: dict, seat: int, selection: dict, remaining: Counter) -> None:
        """Latch the prize multiset the first time a deck search reveals the deck.

        Only at a frame this seat owns: ``visible_cards`` credits ``select.deck`` to the
        deciding player, so reading it from the other seat's frame would subtract cards
        that seat cannot see.
        """
        if self.decklist is None or state.get("yourIndex") != seat:
            return
        if not selection or selection.get("deck") is None:
            return
        self._known_prize = +remaining

    def _known_prizes_now(self, remaining: Counter) -> Counter:
        """The latched prizes, less any that have since been taken.

        Clamped against ``remaining`` rather than tracked by event: a taken prize turns up
        in hand and so leaves ``remaining`` on its own, and a deck that has been shuffled
        back up only makes ``remaining`` larger. The clamp is therefore correct under both
        without needing to know which happened.
        """
        if self._known_prize is None:
            return Counter()
        return Counter(
            {card: min(count, remaining[card])
             for card, count in self._known_prize.items()
             if remaining[card] > 0}
        )

    def _observe_opponent(self, state: dict, seat: int, selection: dict | None) -> None:
        """Accumulate the opponent's shown cards, as a per-id high-water mark.

        Their hand is redacted by the engine, so this only ever sees public zones. The max
        (not the sum) is the point: the same card moving hand -> board -> discard must not
        count three times, while a genuine second copy appearing alongside the first must
        raise the count to two. ``visible_cards`` is serial-deduplicated within a frame, so
        the max across frames is a sound lower bound on their decklist.
        """
        seen = visible_cards(state, 1 - seat, selection, count_hand=True)
        for card, count in seen.items():
            if count > self._opponent_revealed[card]:
                self._opponent_revealed[card] = count

    # -- frame API ---------------------------------------------------------

    def update(self, obs: dict, seat: int) -> dict:
        """Fold this frame in and return the belief block for both seats.

        Shape mirrors the two player-state dicts it is attached to, so the *shared*
        ``PlayerStateEncoder`` reads one card multiset per seat either way — own side gets
        the exact hidden pool, opponent side gets what they have shown.
        """
        state = obs.get("current") or {}
        selection = obs.get("select")
        if not state.get("players"):
            return {"own": _empty_block(), "opponent": _empty_block()}

        remaining = self._remaining(state, seat, selection)
        self._deduce_prizes(state, seat, selection, remaining)
        self._observe_opponent(state, seat, selection)
        known_prize = self._known_prizes_now(remaining)
        return {
            "own": {
                "hidden": _pairs(remaining),
                "known_prize": _pairs(known_prize),
                "prize_known": float(self._known_prize is not None),
            },
            "opponent": {
                # The opponent's hidden pool is not knowable (see the module docstring);
                # what they have revealed is, and it goes in the same slot so the shared
                # encoder needs no per-seat branch.
                "hidden": _pairs(self._opponent_revealed),
                "known_prize": ([], []),
                "prize_known": 0.0,
            },
        }


def _pairs(counter: Counter) -> tuple[list[int], list[int]]:
    """``(card_ids, counts)`` sorted by id — a multiset in the ragged-card-list shape the
    feature pipeline already collates, with the count carried alongside so a pooled
    embedding is not asked to represent "three of these" by repetition."""
    items = sorted(counter.items())
    return [int(card) for card, _ in items], [int(count) for _, count in items]


def _empty_block() -> dict:
    return {"hidden": ([], []), "known_prize": ([], []), "prize_known": 0.0}
