"""Web server for the player-vs-agent simulator.

Run it, open the page, and play a game of the competition's Pokémon TCG against
one of the repo's trained agents:

    PY=/home/capon/miniconda3/envs/kaggle-pokemon/bin/python
    $PY simulator/server.py --port 8000

Design notes
------------
*One game per process.* The native engine keeps its battle pointer on a class
attribute, so a second concurrent battle in the same process would stomp the
first. There is therefore a single module-level battle behind a lock; starting a
new game finishes the old one.

*The browser only ever sees the human seat's observation.* The engine emits
JSON for the seat that owns the current decision, with the other seat's hand,
deck and prizes erased. ``Battle`` keeps the human's observation and discards
the agent's, so the payload the client renders cannot contain the agent's hand,
deck order or prize identities -- there is no client-side hiding to defeat.

*The agent moves inside the request.* A checkpoint forward is milliseconds and
a whole agent turn is well under a second, so ``/api/select`` simply returns the
next position the human has to act on.
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import opponents as agent_lib  # noqa: E402
import engine  # noqa: E402
import render  # noqa: E402

app = FastAPI(title="PTCG player-vs-agent simulator")

_lock = threading.Lock()
_battle: engine.Battle | None = None
_agent_name = "random"
_history: list[str] = []


class NewGame(BaseModel):
    agent: str                        # a registered opponent (simulator/agents/*)
    humanDeck: str | None = None      # a prebuilt decklist by name...
    humanCards: list[int] | None = None  # ...or 60 card ids built in the browser
    seat: int = -1                    # -1 picks at random


class Choice(BaseModel):
    options: list[int]


def _view() -> dict:
    """The payload for the browser, drawn from the human's own observation."""
    assert _battle is not None
    fresh = _battle.take_logs()
    seat = _battle.human_seat
    _history.extend(render.log_lines(fresh, seat))
    # The same events, in the shape the board animates them. Consumed once, with
    # the log lines, so a page reload does not replay a turn that already ran.
    events = render.log_events(fresh, seat)
    # The board after each of the agent's decisions, so the replay can move the
    # cards one action at a time instead of jumping to the end of the turn.
    steps = _battle.take_steps(len(fresh))
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


@app.get("/api/config")
def config() -> dict:
    """Decks you may pilot, and the opponents -- each of which brings its own deck."""
    return {"decks": engine.deck_names(), "agents": agent_lib.available_agents()}


def _deck_entry(card_id: int, count: int) -> dict:
    # The whole card, not a summary: the picker's hover tooltip reads the same
    # fields as the board's, and a trimmed entry showed a name with no rules text.
    return dict(render.card_info(card_id), count=count)


def _deck_listing(cards: list[int]) -> list[dict]:
    """``[image] x N`` rows: one per distinct card, in the order the list has them."""
    counts: dict[int, int] = {}
    for card_id in cards:
        counts[card_id] = counts.get(card_id, 0) + 1
    order = {"pokemon": 0, "item": 1, "tool": 2, "supporter": 3, "stadium": 4, "energy": 5}
    rows = [_deck_entry(card_id, count) for card_id, count in counts.items()]
    rows.sort(key=lambda r: (order.get(r["kind"], 9), r["name"]))
    return rows


@app.get("/api/decks")
def decks() -> dict:
    """The prebuilt decklists, expanded into card rows for the picker."""
    out = []
    for name in engine.deck_names():
        cards = engine.deck_by_name(name)
        out.append({"name": name, "total": len(cards), "cards": _deck_listing(cards)})
    return {"decks": out}


@app.get("/api/cards")
def cards() -> dict:
    """The whole card pool, for the deck builder's search."""
    return {"cards": [render.card_info(card_id) for card_id in sorted(render.card_db())]}


@app.post("/api/deck/check")
def deck_check(cards: list[int]) -> dict:
    """What a built deck still gets wrong, in the engine's own terms."""
    return {"problems": engine.deck_check(cards, render.card_db()),
            "listing": _deck_listing(cards)}


@app.post("/api/new")
def new_game(request: NewGame) -> dict:
    global _battle, _agent_name, _history
    with _lock:
        if _battle is not None:
            _battle.close()
            _battle = None
        if request.humanCards is not None:
            human_deck = list(request.humanCards)
            problems = engine.deck_check(human_deck, render.card_db())
            if problems:
                raise HTTPException(400, "illegal deck: " + "; ".join(problems))
        elif request.humanDeck is not None:
            try:
                human_deck = engine.deck_by_name(request.humanDeck)
            except KeyError as exc:
                raise HTTPException(400, f"unknown deck {exc}") from exc
        else:
            raise HTTPException(400, "no deck given")

        # The opponent's deck is not a separate choice: a bundle plays the 60
        # cards it was trained on.
        try:
            agent_deck = engine.read_deck(agent_lib.agent_deck(request.agent))
        except KeyError as exc:
            raise HTTPException(400, f"unknown opponent {exc}") from exc

        seat = request.seat if request.seat in (0, 1) else random.randint(0, 1)
        try:
            opponent = agent_lib.build_agent(request.agent)
        except Exception as exc:  # a missing checkpoint, a bundle that will not load
            raise HTTPException(400, f"cannot load opponent {request.agent}: {exc}") from exc
        _agent_name = request.agent

        try:
            _battle = engine.Battle(human_deck, agent_deck, seat, opponent)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        _history = []
        _battle.advance()
        return _view()


@app.get("/api/state")
def state() -> dict:
    with _lock:
        if _battle is None:
            raise HTTPException(409, "no game in progress")
        return _view()


@app.post("/api/select")
def select(choice: Choice) -> dict:
    with _lock:
        if _battle is None:
            raise HTTPException(409, "no game in progress")
        try:
            _battle.select(choice.options)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return _view()


def _asset_stamp() -> str:
    """A version string that changes whenever the page assets change."""
    newest = 0.0
    for name in ("app.js", "style.css", "index.html"):
        path = _HERE / "static" / name
        if path.is_file():
            newest = max(newest, path.stat().st_mtime)
    return str(int(newest))


@app.get("/api/version")
def version() -> dict:
    """Which build the server is serving — the page prints this to the console."""
    return {"build": _asset_stamp()}


@app.get("/")
def index() -> HTMLResponse:
    """The page, with its asset URLs versioned by file mtime.

    Header games are not enough on their own: a browser that cached ``app.js``
    *before* the server started sending ``no-cache`` will keep using its copy
    without asking, and the page then runs code from another day — which is
    exactly how a fixed animation can still look broken. A changed URL cannot be
    answered from cache at all, so the version goes in the query string.
    """
    html = (_HERE / "static" / "index.html").read_text()
    stamp = _asset_stamp()
    html = html.replace("/static/app.js", f"/static/app.js?v={stamp}")
    html = html.replace("/static/style.css", f"/static/style.css?v={stamp}")
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


class _FreshStatic(StaticFiles):
    """Serve the page assets with ``no-cache``.

    Not a micro-optimisation in reverse: the browser was holding an old
    ``style.css``/``app.js`` across restarts, so a fixed layout still looked
    broken until a hard reload. ``no-cache`` keeps the file cached but forces a
    revalidation, so an edited asset is picked up on an ordinary reload while an
    unchanged one still answers 304.
    """

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", _FreshStatic(directory=_HERE / "static"), name="static")

# Card art, one file per card id. Served straight off disk: the files are large
# (a few hundred KB each) and never change, so the browser cache does the work.
if render.IMAGE_DIR.is_dir():
    app.mount("/cards", StaticFiles(directory=render.IMAGE_DIR), name="cards")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
