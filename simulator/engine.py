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
    "Dragapult": DECK_DIR / "dragapult.csv",
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


# Zones only their owner can see into. A card whose *destination* is one of
# these keeps its identity secret; a card landing anywhere else is face-up on
# the table and public to both seats.
_PRIVATE_AREAS = {1, 2, 6, 12}          # deck, hand, prize, the cards you are looking at

# Everything the table can see happen, whoever did it.
_PUBLIC_TYPES = {
    LogType.SHUFFLE, LogType.HAS_BASIC_POKEMON, LogType.TURN_START, LogType.TURN_END,
    LogType.DRAW_REVERSE, LogType.MOVE_CARD_REVERSE, LogType.SWITCH, LogType.CHANGE,
    LogType.PLAY, LogType.ATTACH, LogType.EVOLVE, LogType.DEVOLVE, LogType.MOVE_ATTACHED,
    LogType.ATTACK, LogType.HP_CHANGE, LogType.POISONED, LogType.BURNED, LogType.ASLEEP,
    LogType.PARALYZED, LogType.CONFUSED, LogType.COIN,
}

# Log entries that name a card the acting seat alone can see.
_REDACT = {LogType.DRAW: LogType.DRAW_REVERSE, LogType.MOVE_CARD: LogType.MOVE_CARD_REVERSE}


def public_logs(logs: list[dict], hidden_seat: int) -> list[dict]:
    """``logs`` as a viewer who is not ``hidden_seat`` may see them.

    Reached only through the losing seat's final observation, which is the one
    place the human's own view never arrives: when the match ends on the agent's
    turn, this is the only record of the attack that ended it.

    Dropping everything attributed to that seat — which is what this used to do
    — threw the decisive turn away with it, so the game ended in silence with
    the board already rearranged. Public actions are kept instead, and only the
    two entry types that can name a card the other seat cannot see are reduced
    to their face-down forms: a draw, and a move *into* a private zone. An
    unfamiliar entry type is still dropped rather than guessed at.
    """
    keep = []
    for entry in logs:
        kind = entry.get("type")
        if kind == LogType.RESULT or entry.get("playerIndex") != hidden_seat:
            keep.append(entry)
            continue
        if kind == LogType.DRAW:
            keep.append({"type": LogType.DRAW_REVERSE, "playerIndex": entry.get("playerIndex")})
        elif kind == LogType.MOVE_CARD:
            public_destination = entry.get("toArea") not in _PRIVATE_AREAS
            if public_destination:
                keep.append(entry)
            else:
                keep.append({
                    "type": LogType.MOVE_CARD_REVERSE,
                    "playerIndex": entry.get("playerIndex"),
                    "fromArea": entry.get("fromArea"),
                    "toArea": entry.get("toArea"),
                })
        elif kind in _PUBLIC_TYPES:
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
        self._agent_trail: list[dict] = []    # the agent's turn, in case the match ends inside it
        self._trail_open = False             # ...and whether it has started collecting
        self._delivered = 0                  # log entries handed to the browser since the agent moved
        self._shown_before = 0               # ...of those, the ones before the current batch
        self.finished = False
        self.result = -1
        if hasattr(agent, "reset"):
            agent.reset(0)

    # -- driving --------------------------------------------------------

    @property
    def acting_seat(self) -> int:
        return self.obs["current"]["yourIndex"]

    def _absorb(self) -> None:
        """Fold the current observation into what the human is allowed to see.

        Normally the human's own next observation carries the whole of the
        agent's turn, so nothing here has to be kept as it goes. When the match
        *ends* on the agent's turn there is no next observation, and the agent's
        final one holds only the slice since its previous decision — which is
        how a game could end on "You lose" with the attack, the damage and the
        Knock Out that caused it never shown. So the turn is also accumulated as
        it happens, redacted, and used only if the human is never asked again.
        """
        state = self.obs["current"]
        if state["result"] != -1:
            self.finished = True
            self.result = state["result"]
        logs = self.obs.get("logs") or []

        if self.acting_seat == self.human_seat:
            self.view_obs = self.obs
            self.pending_logs.extend(logs)
            # The human's own view supersedes the trail: it covers the same
            # entries, unredacted, and keeping both would show the turn twice.
            self._agent_trail.clear()
            self._trail_open = False
            return

        if not self._trail_open:
            # This first observation of the agent's turn reaches back over what
            # the browser already holds; only what follows is new.
            logs = logs[self._delivered:]
            self._trail_open = True
        self._agent_trail.extend(public_logs(logs, self.agent_seat))
        if self.finished:
            self.pending_logs.extend(self._agent_trail)
            self._agent_trail.clear()
            self._trail_open = False

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
            # From here the agent's next observation starts fresh, so nothing
            # delivered before it has to be subtracted again.
            self._delivered = self._shown_before = 0

    def _snapshot(self, delta: int | None = None) -> None:
        """Record the board, and how many log entries the step that made it wrote.

        The counts are turned into positions in the *human's* batch when the
        batch is delivered (``take_steps``): both seats see the same entries in
        the same order, redacted per seat, so a count is a position either way.

        ``delta`` overrides that count. A step that no action produced — the
        board as it stood before the human chose — wrote nothing, and passing 0
        is what keeps it anchored at the start of the batch.
        """
        board = render.board_snapshot(self.obs, self.human_seat)
        board["_delta"] = len(self.obs.get("logs") or []) if delta is None else delta
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
        # The board as it stands *before* the choice is applied. Without it the
        # batch opens on the position the human's own action already reached —
        # the browser draws that first, so an attack showed its damage and swept
        # the Knocked Out Pokémon into the discard before the attack itself was
        # ever animated. The events then narrate a board that has already moved.
        self._snapshot(delta=0)
        self.obs = raw_select(list(choice))
        self.advance()

    def take_logs(self) -> list[dict]:
        logs, self.pending_logs = self.pending_logs, []
        # What the browser has been given since the agent last moved. The agent's
        # next observation reaches back over all of it — a player often takes
        # several decisions in a row — and subtracting it is what puts this
        # batch's boards among its own events instead of past the last one.
        self._shown_before = self._delivered
        self._delivered += len(logs)
        return logs

    def take_steps(self, batch_size: int) -> list[dict]:
        """The boards for this batch, each tagged with the log position it holds.

        The first snapshot opens the batch and nothing in it has been narrated
        yet: after a human decision that is the board from before the choice was
        applied (``select`` records it with a zero delta), and at the start of a
        match it is the opening position. Each later step accounts for one
        decision. Working the offsets out from the batch size keeps the two
        counts anchored to the same starting point, which counting forwards from
        the agent's observations does not — the agent's first observation
        reaches back past the human's last view.
        """
        steps, self.pending_steps = self.pending_steps, []
        deltas = [step.pop("_delta") for step in steps]
        if not steps:
            return steps

        steps[0]["logsSoFar"] = 0
        if len(steps) > 1:
            # The observation the agent first acts on reaches back past the
            # human's last view: its log holds everything that view already
            # showed, *plus* whatever the human's own action wrote. Only the
            # latter belongs to this batch. Counting it whole is what pushed
            # every board past the last event, so none of them landed until the
            # replay was over and the whole turn arrived at once.
            steps[1]["logsSoFar"] = min(max(0, deltas[1] - self._shown_before), batch_size)
        position = steps[1]["logsSoFar"] if len(steps) > 1 else 0
        for i in range(2, len(steps)):
            position += deltas[i]
            steps[i]["logsSoFar"] = min(position, batch_size)
        # The last snapshot came from the *next* observation, whose log spans the
        # whole batch rather than the decision that produced this board. It is
        # the final position either way, so it belongs at the end.
        steps[-1]["logsSoFar"] = batch_size
        return steps

    def close(self) -> None:
        battle_finish()
        if hasattr(self.agent, "close"):
            self.agent.close()
