"""Battle plumbing: one live ``cg`` battle, driven seat by seat.

Two facts about the native engine shape everything here.

*One battle per process.* ``cg.sim.Battle`` parks the battle pointer on a class
attribute, so a process can hold exactly one battle at a time. The server is
therefore single-game and serialises every request through a lock rather than
handing out session objects.

*Observations are already seat-filtered.* ``GetBattleData`` emits JSON for the
player whose turn it is to choose, with the other seat's hand, deck and prizes
erased (``State::erasePlayerData`` / ``ToJson.h``). That is the whole
imperfect-information story: the browser is only ever fed observations taken
from the *human* seat, so the agent's hand, deck order and prize identities
never cross the wire. Nothing here re-adds them.

``simulator/`` is self-contained. The competition engine is one package,
``ptcg_engine``: the C++ sources under ``src/`` and the ctypes bindings that
call into them, with ``build_engine.sh`` compiling the former into the
``libcg.so`` the latter loads. Decklists are in ``decks/``. Nothing here reaches
outside the directory, so the simulator keeps working with the training code in
another repository entirely.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import ctypes  # noqa: E402

from ptcg_engine.game import _get_battle_data, battle_finish, battle_start  # noqa: E402
from ptcg_engine.api import LogType  # noqa: E402
from ptcg_engine.sim import Battle as _CgBattle, lib as _lib  # noqa: E402

import render  # noqa: E402
# ``ptcg_engine.game.battle_select`` collapses every engine rejection into a bare
# ``IndexError``, which reaches the player as an empty error message and looks
# like a button that simply does not work. The same call is made here so the
# engine's own code survives — these are ``State::checkPlayerSelect``'s codes.
SELECT_ERRORS = {
    3: "the match is already over",
    4: "wrong number of options chosen for this selection",
    5: "an option index is out of range",
    6: "the same option was chosen twice",
    30: "the battle pointer is broken",
}


def raw_select(indices: list[int]) -> dict:
    """``lib.Select`` with the error code preserved; returns the next observation."""
    arg = (ctypes.c_int * len(indices))(*indices)
    error = _lib.Select(_CgBattle.battle_ptr, arg, len(indices))
    if error:
        raise ValueError(SELECT_ERRORS.get(error, f"the engine refused that selection (code {error})"))
    return _get_battle_data()

DECK_SIZE = 60

# Decklists that ship with the repo, by the name the UI shows.
DECK_DIR = _HERE / "decks"
DECKS = {
    "Crustle": DECK_DIR / "crustle.csv",
    "Grimmsnarl ex": DECK_DIR / "grimmsnarl_ex.csv",
    "Alakazam": DECK_DIR / "alakazam.csv",
    "Mega Lucario ex": DECK_DIR / "mega_lucario_ex.csv",
}

# The deck-legality rules the engine enforces at ``BattleStart`` (Api.h). They
# are repeated here so a deck can be checked *before* a battle is started and
# the reason shown next to the offending card, rather than as a start failure.
DECK_SAME_CARD_MAX = 4
ACE_SPEC_MAX = 1


def deck_names() -> list[str]:
    return [name for name, path in DECKS.items() if path.is_file()]


def deck_check(cards: list[int], card_db: dict[int, dict]) -> list[str]:
    """Every rule the deck breaks, in the engine's own terms; empty if legal."""
    problems = []
    if len(cards) != DECK_SIZE:
        problems.append(f"{len(cards)}/60 cards")

    by_name: dict[str, int] = {}
    ace_specs = 0
    basics = 0
    for card_id in cards:
        data = card_db.get(card_id)
        if data is None:
            problems.append(f"unknown card id {card_id}")
            continue
        by_name[data["name"]] = by_name.get(data["name"], 0) + 1
        if data.get("aceSpec"):
            ace_specs += 1
        if data.get("basic") and data.get("cardType") == 0:
            basics += 1

    for name, count in sorted(by_name.items()):
        data = next((c for c in card_db.values() if c["name"] == name), None)
        # The four-copy limit is by card *name*, and Basic Energy is exempt.
        if count > DECK_SAME_CARD_MAX and (data or {}).get("cardType") != 5:
            problems.append(f"{count} copies of {name} (max {DECK_SAME_CARD_MAX})")
    if ace_specs > ACE_SPEC_MAX:
        problems.append(f"{ace_specs} ACE SPEC cards (max {ACE_SPEC_MAX})")
    if not basics:
        problems.append("no Basic Pokémon")
    return problems


def read_deck(path: Path) -> list[int]:
    lines = [line for line in Path(path).read_text().split("\n") if line.strip()]
    if len(lines) != DECK_SIZE:
        raise ValueError(f"{path} has {len(lines)} cards, expected {DECK_SIZE}")
    return [int(line) for line in lines]


def deck_by_name(name: str) -> list[int]:
    if name not in DECKS:
        raise KeyError(name)
    return read_deck(DECKS[name])


START_ERRORS = {
    1: "unknown card id",
    2: "more than four copies of a card",
    3: "no Basic Pokémon in the deck",
    4: "more than one ACE SPEC card",
}


def public_logs(logs: list[dict], hidden_seat: int) -> list[dict]:
    """Log entries safe to show a viewer who is *not* ``hidden_seat``.

    Used only for the handful of events that land in the losing seat's final
    observation. An entry attributed to ``hidden_seat`` may name a card that
    seat drew or shuffled away — private information — so those are dropped;
    entries about the viewer, and the match result, are kept.
    """
    keep = []
    for entry in logs:
        if entry.get("type") == LogType.RESULT:
            keep.append(entry)
        elif entry.get("playerIndex") != hidden_seat:
            keep.append(entry)
    return keep


class Battle:
    """A live game between the human seat and one agent.

    ``advance`` runs the engine forward, letting the agent answer every
    selection put to it, and stops the moment the human has a decision to make
    or the match ends. The human's own observation is the only one retained.
    """

    def __init__(self, human_deck: list[int], agent_deck: list[int], human_seat: int, agent) -> None:
        self.human_seat = human_seat
        self.agent = agent
        self.agent_seat = 1 - human_seat
        decks = [None, None]
        decks[human_seat] = human_deck
        decks[self.agent_seat] = agent_deck

        obs, start = battle_start(decks[0], decks[1])
        if obs is None:
            reason = START_ERRORS.get(start.errorType, f"error {start.errorType}")
            raise ValueError(f"player {start.errorPlayer}'s deck is illegal: {reason}")

        self.obs = obs                 # the most recent observation, whoever it belongs to
        self.view_obs: dict | None = None   # the most recent *human* observation
        self.pending_logs: list[dict] = []   # human-visible events not yet drawn
        self.pending_steps: list[dict] = []  # board after each agent decision
        self.finished = False
        self.result = -1
        if hasattr(agent, "reset"):
            agent.reset(0)

    # -- driving --------------------------------------------------------

    @property
    def acting_seat(self) -> int:
        return self.obs["current"]["yourIndex"]

    def _absorb(self) -> None:
        """Fold the current observation into what the human is allowed to see."""
        state = self.obs["current"]
        if state["result"] != -1:
            self.finished = True
            self.result = state["result"]
        if self.acting_seat == self.human_seat:
            self.view_obs = self.obs
            self.pending_logs.extend(self.obs.get("logs") or [])
        elif self.finished:
            # The match ended on the agent's turn: its observation carries the
            # RESULT log the human never otherwise sees. Take only the public part.
            self.pending_logs.extend(public_logs(self.obs.get("logs") or [], self.agent_seat))

    def advance(self) -> None:
        """Let the agent play until the human must choose (or the game ends).

        A snapshot of the board is taken after every one of the agent's
        decisions. Without them the browser only ever receives the position at
        the *end* of the turn, so a narrated replay describes moves against a
        board that has already made all of them — the pop-ups paced correctly
        while the cards jumped to their final places at once.
        """
        self._absorb()
        self._snapshot()
        while not self.finished and self.acting_seat == self.agent_seat:
            select = self.obs["select"]
            if select is None:
                raise RuntimeError("engine asked for a deck mid-battle")
            choice = self.agent.act(self.obs)
            self.obs = raw_select(list(choice))
            self._absorb()
            self._snapshot()

    def _snapshot(self) -> None:
        """Record the board, and how many log entries the step that made it wrote.

        The counts are turned into positions in the *human's* batch when the
        batch is delivered (``take_steps``): both seats see the same entries in
        the same order, redacted per seat, so a count is a position either way.
        """
        board = render.board_snapshot(self.obs, self.human_seat)
        board["_delta"] = len(self.obs.get("logs") or [])
        self.pending_steps.append(board)

    def select(self, choice: list[int]) -> None:
        """Apply the human's selection, then hand control back to the agent."""
        if self.finished:
            raise RuntimeError("the match is over")
        if self.acting_seat != self.human_seat:
            raise RuntimeError("not your decision")
        select = self.obs["select"]
        low, high = select["minCount"], select["maxCount"]
        if not low <= len(choice) <= high:
            raise ValueError(f"choose between {low} and {high} options")
        if len(set(choice)) != len(choice):
            raise ValueError("duplicate selections")
        if any(not 0 <= i < len(select["option"]) for i in choice):
            raise ValueError("option index out of range")
        self.obs = raw_select(list(choice))
        self.advance()

    def take_logs(self) -> list[dict]:
        logs, self.pending_logs = self.pending_logs, []
        return logs

    def take_steps(self, batch_size: int) -> list[dict]:
        """The boards for this batch, each tagged with the log position it holds.

        The first snapshot is the board *before* the agent moved, so what stands
        in front of it is whatever the human's own action wrote; the rest each
        account for one agent decision. Working the offsets out from the batch
        size keeps the two counts anchored to the same starting point, which
        counting forwards from the agent's observations does not — the agent's
        first observation reaches back past the human's last view.
        """
        steps, self.pending_steps = self.pending_steps, []
        agent_deltas = [step.pop("_delta") for step in steps][1:]
        position = max(0, batch_size - sum(agent_deltas))
        for i, step in enumerate(steps):
            step["logsSoFar"] = position
            if i < len(agent_deltas):
                position += agent_deltas[i]
        return steps

    def close(self) -> None:
        battle_finish()
        if hasattr(self.agent, "close"):
            self.agent.close()
