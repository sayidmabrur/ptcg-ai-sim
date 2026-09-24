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

import json
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
import replay_log  # noqa: E402
from ptcg_engine.game import visualize_data  # noqa: E402
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
    # 2026 archetypes built on Chaos Rising / Pitch Black cards (CardImplExtra.h).
    "Mega Excadrill ex": DECK_DIR / "mega_excadrill_ex.csv",
    "Bastiodon / Rampardos ex": DECK_DIR / "bastiodon_rampardos_ex.csv",
    "Iron Thorns ex / Bastiodon": DECK_DIR / "iron_thorns_bastiodon.csv",
    "Mega Greninja ex": DECK_DIR / "mega_greninja_ex.csv",
    "Greninja ex": DECK_DIR / "greninja_ex.csv",
    "Dhelmise Hide 'n' Sneak": DECK_DIR / "dhelmise.csv",
    "Mega Chandelure ex": DECK_DIR / "mega_chandelure_ex.csv",
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





def _record(recorder, seat: int, choice: list[int], obs: dict) -> None:
    """Log one selection, and write the replay the moment the game is over."""
    if recorder is None:
        return
    recorder.selection(seat, choice, obs)
    result = obs["current"]["result"]
    if result != -1:
        _save(recorder, result)


def _save(recorder, result: int) -> None:
    """Write the replay; a failure to write must never cost anyone their game."""
    if recorder is None or recorder.saved is not None:
        return
    try:
        visualize = json.loads(visualize_data())
    except Exception:  # noqa: BLE001 -- the full-information frames are a bonus
        visualize = None
    try:
        path = recorder.save(result, visualize)
        if path is not None:
            print(f"replay saved: {path}", file=sys.stderr, flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"replay not saved: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


class Battle:
    """A live game between the human seat and one agent.

    ``advance`` runs the engine forward, letting the agent answer every
    selection put to it, and stops the moment the human has a decision to make
    or the match ends. The human's own observation is the only one retained.
    """

    def __init__(self, human_deck: list[int], agent_deck: list[int], human_seat: int, agent,
                 names: tuple[str, str] = ("Player", "agent")) -> None:
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

        seat_names = [None, None]
        seat_names[human_seat], seat_names[1 - human_seat] = names
        self.recorder = replay_log.Recorder(seat_names, decks, "vs_agent", obs)
        self.obs = obs                 # the most recent observation, whoever it belongs to
        self.view_obs: dict | None = None   # the most recent *human* observation
        self.pending_logs: list[dict] = []   # human-visible events not yet drawn
        self.pending_steps: list[dict] = []  # board after each agent decision
        self._agent_trail: list[dict] = []    # the agent's turn, in case the match ends inside it
        self._trail_open = False             # ...and whether it has started collecting
        self._delivered = 0                  # log entries handed to the browser since the agent moved
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

        One snapshot per decision, each carrying the number of log entries that
        decision wrote. A step is then a whole move — *this* is the board, *this*
        is what was done to reach it — which is what the browser replays, rather
        than two independent streams it has to line up by counting.
        """
        # What the browser already holds of the agent's first observation: that
        # observation reaches back to the agent's previous decision, over
        # everything delivered since.
        shown = self._delivered
        self._absorb()
        first = self.obs.get("logs") or []
        # Whatever is left is what the human's own action just wrote.
        self._snapshot(wrote=max(0, len(first) - shown))
        while not self.finished and self.acting_seat == self.agent_seat:
            select = self.obs["select"]
            if select is None:
                raise RuntimeError("engine asked for a deck mid-battle")
            choice = self.agent.act(self.obs)
            self.obs = raw_select(list(choice))
            _record(self.recorder, self.agent_seat, list(choice), self.obs)
            self._absorb()
            # The observation that follows a decision spans exactly what that
            # decision wrote — unless the agent's turn is over, in which case it
            # is the human's own view and spans the whole batch. That last one
            # is what the remainder is for.
            wrote = len(self.obs.get("logs") or []) if self.acting_seat == self.agent_seat else None
            self._snapshot(wrote=wrote)
            self._delivered = 0

    def _snapshot(self, wrote: int | None = None) -> None:
        """Record the board, and how many log entries produced it.

        ``wrote`` is ``None`` when the count is not knowable from the engine —
        the opening position, and the agent's final decision, whose following
        observation belongs to the human and spans the whole batch. Those are
        filled in at delivery from what is left over.
        """
        self.pending_steps.append({
            "board": render.board_snapshot(self.obs, self.human_seat),
            "_wrote": wrote,
        })

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
        self._snapshot(wrote=0)
        self.obs = raw_select(list(choice))
        _record(self.recorder, self.human_seat, list(choice), self.obs)
        self.advance()

    def take_logs(self) -> list[dict]:
        logs, self.pending_logs = self.pending_logs, []
        # What the browser has been given since the agent last moved: the
        # agent's next observation reaches back over all of it, and ``advance``
        # subtracts it to find what the human's own action wrote.
        self._delivered += len(logs)
        return logs

    def take_steps(self, logs: list[dict]) -> list[dict]:
        """This batch as a list of moves: each board with the events that made it.

        Every step owns its own slice of the log, cut by how many entries each
        decision wrote, so the browser replays *board, what was done, next board*
        instead of trying to line two independent streams up by position. The
        slices come from the human's own batch, so nothing here has to decide
        what may be shown — the engine already redacted it for this seat.

        An unknown count (the opening board, and the agent's last decision,
        whose following observation belongs to the human) takes what is left
        over. If the counts do not add up — a game ending mid-turn hands over a
        reconstructed batch — the remainder simply lands on the last step, which
        is late but never out of order.
        """
        steps, self.pending_steps = self.pending_steps, []
        if not steps:
            return steps

        counts = [step.pop("_wrote") for step in steps]
        known = sum(n for n in counts if n is not None)
        unknown = [i for i, n in enumerate(counts) if n is None]
        # Share whatever the measured decisions did not account for among the
        # steps whose own count the engine could not give us.
        spare = max(0, len(logs) - known)
        for i in unknown:
            counts[i] = spare if i == unknown[-1] else 0

        at = 0
        for step, count in zip(steps, counts):
            slice_ = logs[at:at + count]
            at += count
            step["events"] = render.log_events(slice_, self.human_seat)
        if at < len(logs):                       # anything unaccounted for is still shown
            steps[-1]["events"].extend(render.log_events(logs[at:], self.human_seat))
        return steps

    def close(self) -> None:
        _save(self.recorder, self.result)      # a game left unfinished is kept too
        battle_finish()
        if hasattr(self.agent, "close"):
            self.agent.close()


class PvpBattle:
    """A live game between two people, each seeing only what their seat may see.

    ``Battle`` can keep one observation and discard the other because its second
    seat is an agent that answers inside the request. Here both seats are
    browsers that poll, so each seat keeps its own queue of events and boards,
    and the waiting seat must see the other's turn as it happens rather than all
    at once when its own next decision comes round.

    What makes that safe is a property of the engine checked across thousands of
    decisions: every observation's log reaches back to that seat's previous
    observation, and the two seats' logs describe the same events one for one,
    only redacted differently. So a single running count of events is enough.
    Each new observation carries ``len(logs) - (count - seen[seat])`` entries
    nobody has had yet; the acting seat receives them as the engine wrote them
    for it, and the other seat receives ``public_logs`` of them -- the same
    reduction ``Battle`` already applies to an agent's turn. When the waiting
    seat's own observation arrives later it reaches back over entries it was
    already shown, and those are skipped by the same count.
    """

    def __init__(self, decks: list[list[int]], names: list[str] | None = None) -> None:
        obs, start = battle_start(decks[0], decks[1])
        if obs is None:
            reason = START_ERRORS.get(start.errorType, f"error {start.errorType}")
            raise ValueError(f"player {start.errorPlayer}'s deck is illegal: {reason}")
        self.recorder = replay_log.Recorder(names or ["Player 1", "Player 2"], decks, "pvp", obs)
        self.obs = obs
        self.finished = False
        self.result = -1
        self._count = 0                      # events delivered to anyone so far
        self._seen = [0, 0]                  # ...as of each seat's last own observation
        self.own_obs: list[dict | None] = [None, None]
        self._logs: list[list[dict]] = [[], []]
        self._steps: list[list[dict]] = [[], []]
        self.version = 0                     # bumps on every change either seat can see
        self._absorb()

    @property
    def acting_seat(self) -> int:
        return self.obs["current"]["yourIndex"]

    def _absorb(self) -> None:
        self.version += 1
        state = self.obs["current"]
        if state["result"] != -1:
            self.finished = True
            self.result = state["result"]
        acting = self.acting_seat
        logs = self.obs.get("logs") or []
        fresh = logs[max(0, self._count - self._seen[acting]):]
        self._count += len(fresh)
        self._seen[acting] = self._count
        self.own_obs[acting] = self.obs
        for seat in (0, 1):
            mine = fresh if seat == acting else public_logs(fresh, acting)
            self._logs[seat].extend(mine)
            self._steps[seat].append({
                "board": render.board_snapshot(self.obs, seat),
                "events": render.log_events(mine, seat),
            })

    def select(self, seat: int, choice: list[int]) -> None:
        """Apply ``seat``'s selection; refused unless that seat holds the decision."""
        if self.finished:
            raise RuntimeError("the match is over")
        if self.acting_seat != seat:
            raise RuntimeError("not your decision")
        select = self.obs["select"]
        low, high = select["minCount"], select["maxCount"]
        if not low <= len(choice) <= high:
            raise ValueError(f"choose between {low} and {high} options")
        if len(set(choice)) != len(choice):
            raise ValueError("duplicate selections")
        if any(not 0 <= i < len(select["option"]) for i in choice):
            raise ValueError("option index out of range")
        # Both seats get the board as it stood before the choice, so either one's
        # replay starts from the position the move was made in (see Battle.select).
        for s in (0, 1):
            self._steps[s].append({"board": render.board_snapshot(self.obs, s), "events": []})
        self.obs = raw_select(list(choice))
        _record(self.recorder, seat, list(choice), self.obs)
        self._absorb()

    def take(self, seat: int) -> tuple[list[dict], list[dict]]:
        """The events and boards ``seat`` has not been given yet, consumed."""
        logs, self._logs[seat] = self._logs[seat], []
        steps, self._steps[seat] = self._steps[seat], []
        # A poll with nothing new would otherwise hand back a lone board; the
        # browser draws the current position anyway.
        if not any(step["events"] for step in steps):
            steps = []
        return logs, steps

    def close(self) -> None:
        _save(self.recorder, self.result)
        battle_finish()
