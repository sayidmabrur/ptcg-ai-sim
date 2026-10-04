"""One battle, in its own interpreter, answering the server over a pipe.

The native engine keeps its battle pointer on a class attribute, so a process
holds exactly one game (see ``cg.sim``). That is fine for one player at
a keyboard and wrong for a server several people can open at once: a second
visitor starting a game would stomp the first one's board.

So the battle does not live in the web server any more. It lives here, one
process per session, and the server is a router: it owns no game state, only a
table of these workers (``sessions.py``). What used to be ``server.py``'s
module-level ``_battle`` is this module's — unchanged in substance, because the
isolation is the process boundary rather than any new locking.

The protocol is one JSON object per line, the same shape ``agent_worker.py``
uses for bundles:

    <- {"op": "new", "agent": ..., "humanDeck": ..., "humanCards": [...], "seat": -1}
    <- {"op": "state"}
    <- {"op": "select", "options": [0]}
    -> {"ok": true, "view": {...}}
    -> {"ok": false, "status": 400, "detail": "illegal deck: 54/60"}

``status`` is the HTTP status the server should raise, so the browser sees the
same errors it always did. A worker that dies takes one session's game with it
and nobody else's.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import opponents as agent_lib  # noqa: E402
import engine  # noqa: E402
import render  # noqa: E402

_battle: engine.Battle | None = None
_agent_name = "random"
_history: list[str] = []


class Refused(Exception):
    """A request the browser should see as an HTTP error, with its status."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _view() -> dict:
    """The payload for the browser, drawn from the human's own observation."""
    assert _battle is not None
    fresh = _battle.take_logs()
    seat = _battle.human_seat
    _history.extend(render.log_lines(fresh, seat))
    # This batch as a list of moves: each board with the events that produced
    # it. Consumed once, with the log lines, so a page reload does not replay a
    # turn that already ran.
    steps = _battle.take_steps(fresh)
    # The flat event list the browser narrates, in step order.
    events = [event for step in steps for event in step["events"]]
    obs = _battle.view_obs
    if obs is None:
        # The agent has the first decision and the human has not observed yet.
        return {
            "ready": False,
            "waitingOn": _agent_name,
            "log": _history[-200:],
            "events": events,
            "steps": steps,
            "result": _battle.result,
            "finished": _battle.finished,
        }
    view = render.build_view(
        obs,
        _battle.human_seat,
        _agent_name,
        [],
        your_turn=not _battle.finished and _battle.acting_seat == _battle.human_seat,
        result=_battle.result,
    )
    view["ready"] = True
    view["log"] = _history[-200:]
    view["events"] = events
    view["steps"] = steps
    view["agent"] = _agent_name
    # A bundle that raised is played on with a legal random move rather than
    # forfeiting the human's game; say so instead of hiding it.
    view["agentError"] = getattr(_battle.agent, "last_error", None)
    if _battle.finished:
        view["waitingOn"] = "nobody"
        view["verdict"] = (
            "Draw" if _battle.result == 2
            else "You win" if _battle.result == _battle.human_seat else "You lose"
        )
    return view


def op_new(request: dict) -> dict:
    global _battle, _agent_name, _history
    if _battle is not None:
        _battle.close()
        _battle = None

    human_cards = request.get("humanCards")
    human_name = request.get("humanDeck")
    if human_cards is not None:
        human_deck = [int(c) for c in human_cards]
        problems = engine.deck_check(human_deck, render.card_db())
        if problems:
            raise Refused(400, "illegal deck: " + "; ".join(problems))
    elif human_name is not None:
        try:
            human_deck = engine.deck_by_name(human_name)
        except KeyError as exc:
            raise Refused(400, f"unknown deck {exc}") from exc
    else:
        raise Refused(400, "no deck given")

    # The opponent's deck is not a separate choice: a bundle plays the 60 cards
    # it was trained on.
    agent = request.get("agent", "")
    try:
        agent_deck = engine.read_deck(agent_lib.agent_deck(agent))
    except KeyError as exc:
        raise Refused(400, f"unknown opponent {exc}") from exc

    seat = request.get("seat", -1)
    seat = seat if seat in (0, 1) else random.randint(0, 1)
    try:
        opponent = agent_lib.build_agent(agent)
    except Exception as exc:  # a missing checkpoint, a bundle that will not load
        raise Refused(400, f"cannot load opponent {agent}: {exc}") from exc
    _agent_name = agent

    try:
        _battle = engine.Battle(human_deck, agent_deck, seat, opponent,
                                names=(request.get("playerName") or "Player", agent))
    except ValueError as exc:
        raise Refused(400, str(exc)) from exc
    _history = []
    _battle.advance()
    return _view()


def op_state(request: dict) -> dict:
    if _battle is None:
        raise Refused(409, "no game in progress")
    return _view()


def op_select(request: dict) -> dict:
    if _battle is None:
        raise Refused(409, "no game in progress")
    try:
        _battle.select([int(o) for o in request.get("options", [])])
    except (ValueError, RuntimeError) as exc:
        raise Refused(400, str(exc)) from exc
    return _view()


# -- two players ------------------------------------------------------------
#
# A room's worker holds a PvpBattle instead: one engine battle, two people. Each
# request names the seat it speaks for (the server resolves that from the
# player's token, never from the browser), and each seat's view is built only
# from what that seat may see.

_pvp: engine.PvpBattle | None = None
_pvp_names = ["Player 1", "Player 2"]
_pvp_history: list[list[str]] = [[], []]


def _pvp_view(seat: int) -> dict:
    assert _pvp is not None
    other = _pvp_names[1 - seat]
    logs, steps = _pvp.take(seat)
    _pvp_history[seat].extend(render.log_lines(logs, seat))
    acting = _pvp.acting_seat
    mine = acting == seat and not _pvp.finished
    view = render.build_view(_pvp.obs, seat, other, [], your_turn=mine, result=_pvp.result)
    if not mine:
        # The current observation is the other seat's. Its own hand and whatever
        # it is looking at are private to it; build_view already hides the hand
        # of the seat it is not drawn for, and the looking area is dropped here.
        # This seat's own hand is not in that observation at all, so the one it
        # last held is shown until its next decision brings a fresh one.
        view["looking"] = []
        # The once-per-turn flags are the acting seat's, and the page draws them
        # on this seat's nameplate; during the other player's turn none apply here.
        view["flags"] = {flag: False for flag in view["flags"]}
        own = _pvp.own_obs[seat]
        if own is not None:
            view["me"]["hand"] = render.player_view(own["current"]["players"][seat], True)["hand"]
    view["ready"] = True
    view["log"] = _pvp_history[seat][-200:]
    view["steps"] = steps
    view["events"] = [event for step in steps for event in step["events"]]
    view["agent"] = other
    view["names"] = {"you": _pvp_names[seat], "opponent": other}
    view["pvp"] = True
    view["version"] = _pvp.version
    if _pvp.finished:
        view["waitingOn"] = "nobody"
        view["verdict"] = (
            "Draw" if _pvp.result == 2
            else "You win" if _pvp.result == seat else "You lose"
        )
    return view


def _seat(request: dict) -> int:
    seat = request.get("seat")
    if seat not in (0, 1):
        raise Refused(400, "no seat given")
    return seat


def op_pvp_new(request: dict) -> dict:
    global _pvp, _pvp_names, _pvp_history
    if _pvp is not None:
        _pvp.close()
        _pvp = None
    decks = request.get("decks") or []
    if len(decks) != 2:
        raise Refused(400, "two decks are needed")
    try:
        _pvp_names = list(request.get("names") or ["Player 1", "Player 2"])
        _pvp = engine.PvpBattle([[int(c) for c in d] for d in decks], _pvp_names)
    except ValueError as exc:
        raise Refused(400, str(exc)) from exc
    _pvp_history = [[], []]
    return {"started": True}


def op_pvp_state(request: dict) -> dict:
    if _pvp is None:
        raise Refused(409, "the game has not started")
    return _pvp_view(_seat(request))


def op_pvp_version(request: dict) -> dict:
    """How far the game has moved, without consuming anything -- for long polls."""
    if _pvp is None:
        raise Refused(409, "the game has not started")
    return {"version": _pvp.version}


def op_pvp_select(request: dict) -> dict:
    if _pvp is None:
        raise Refused(409, "the game has not started")
    seat = _seat(request)
    try:
        _pvp.select(seat, [int(o) for o in request.get("options", [])])
    except (ValueError, RuntimeError) as exc:
        raise Refused(400, str(exc)) from exc
    return _pvp_view(seat)


OPS = {"new": op_new, "state": op_state, "select": op_select,
       "pvp_new": op_pvp_new, "pvp_state": op_pvp_state, "pvp_select": op_pvp_select,
       "pvp_version": op_pvp_version}


def _reply(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> None:
    # Say hello once the engine is imported, the way a bundle does: the server
    # then knows a worker is usable before it forwards a request to it.
    _reply({"ready": True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError as exc:
            _reply({"ok": False, "status": 400, "detail": f"bad request: {exc}"})
            continue
        handler = OPS.get(request.get("op"))
        if handler is None:
            _reply({"ok": False, "status": 400, "detail": f"unknown op {request.get('op')!r}"})
            continue
        try:
            _reply({"ok": True, "view": handler(request)})
        except Refused as exc:
            _reply({"ok": False, "status": exc.status, "detail": exc.detail})
        except Exception as exc:  # never take the worker down on one bad request
            print(f"game_worker: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            _reply({"ok": False, "status": 500, "detail": f"{type(exc).__name__}: {exc}"})

    if _battle is not None:
        _battle.close()
    if _pvp is not None:
        _pvp.close()


if __name__ == "__main__":
    main()
