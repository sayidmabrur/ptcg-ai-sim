"""Web server for the player-vs-agent simulator.

Run it, open the page, and play a game of the competition's Pokémon TCG against
one of the repo's trained agents:

    PY=/home/capon/miniconda3/envs/kaggle-pokemon/bin/python
    $PY simulator/server.py --port 8000

Design notes
------------
*One game per process, one process per player.* The native engine keeps its
battle pointer on a class attribute, so a second concurrent battle in the same
process would stomp the first. This server therefore holds no battle at all: it
routes each request to the caller's own ``game_worker.py`` process
(``sessions.py``), which is what lets two people play at once. The read-only
endpoints -- card pool, decklists, opponents -- are answered here, because they
only read the engine's static tables.

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
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import opponents as agent_lib  # noqa: E402
import engine  # noqa: E402
import render  # noqa: E402
import sessions  # noqa: E402

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Take the session workers down with the server.

    A game is two child processes; a server that exits without reaping them
    leaves them holding their memory, which on a container that restarts often
    is how a machine fills up.
    """
    yield
    sessions.registry.close_all()


app = FastAPI(title="PTCG player-vs-agent simulator", lifespan=_lifespan)

# The header the page identifies itself with. Not a cookie: the Space is usually
# viewed in an iframe on huggingface.co, where the Space's own cookies are
# third-party and dropped by default.
SESSION_HEADER = "X-Session"


class NewGame(BaseModel):
    agent: str                        # a registered opponent (simulator/agents/*)
    humanDeck: str | None = None      # a prebuilt decklist by name...
    humanCards: list[int] | None = None  # ...or 60 card ids built in the browser
    seat: int = -1                    # -1 picks at random


class Choice(BaseModel):
    options: list[int]


def _session_reply(response: Response, sid: str, reply: dict) -> dict:
    """A worker's answer, as the browser expects it: a view, or an HTTP error."""
    response.headers[SESSION_HEADER] = sid
    if not reply.get("ok"):
        raise HTTPException(reply.get("status", 500), reply.get("detail", "game error"))
    view = reply["view"]
    view["session"] = sid
    return view


@app.get("/api/config")
def config() -> dict:
    """Decks you may pilot, and the opponents -- each of which brings its own deck."""
    return {"decks": engine.deck_names(), "agents": agent_lib.available_agents(),
            "sessions": sessions.registry.stats()}


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


def _archetype(cards: list[int]) -> dict:
    """The deck's headline Pokémon — what a player would call the deck.

    Picked from the decklist rather than stored: the most numerous Pokémon ex in
    it, and failing that the most numerous Pokémon. That is the card the deck is
    named after in every matchup sheet, and the one worth showing next to the
    agent that pilots it.
    """
    counts: dict[int, int] = {}
    for card_id in cards:
        counts[card_id] = counts.get(card_id, 0) + 1
    best = None
    for card_id, count in counts.items():
        card = render.card_info(card_id)
        if card["kind"] != "pokemon":
            continue
        rank = (bool(card["megaEx"]), bool(card["ex"]), count, -card_id)
        if best is None or rank > best[0]:
            best = (rank, card)
    if best is None:
        return {}
    card = best[1]
    return {"name": card["name"], "image": card["image"], "cardId": card["id"]}


@app.get("/api/agents")
def agents() -> dict:
    """Every registered opponent, with the deck it pilots and what it is."""
    out = []
    for name, bundle in agent_lib.discover_bundles().items():
        cards = engine.read_deck(bundle / "deck.csv")
        out.append({
            "id": name,
            "archetype": _archetype(cards),
            "arch": agent_lib.agent_arch(bundle),
            "deck": _deck_listing(cards),
            "deckTotal": len(cards),
        })
    return {"agents": out}


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
def new_game(request: NewGame, response: Response,
             x_session: str | None = Header(default=None)) -> dict:
    """Start a game in the caller's own worker, making one if they have none."""
    try:
        sid, session = sessions.registry.open(x_session)
    except sessions.SessionBusy as exc:
        raise HTTPException(503, str(exc)) from exc
    except (RuntimeError, OSError) as exc:
        raise HTTPException(500, f"cannot start a game process: {exc}") from exc
    try:
        reply = session.request("new", request.model_dump())
    except sessions.SessionGone as exc:
        sessions.registry.drop(sid)
        raise HTTPException(503, str(exc)) from exc
    return _session_reply(response, sid, reply)


def _forward(op: str, payload: dict, response: Response, sid: str | None) -> dict:
    session = sessions.registry.get(sid)
    if session is None:
        # An unknown session is not an error to explain but a game to restart:
        # the browser turns 409 into the New game dialog, as it always has.
        raise HTTPException(409, "no game in progress")
    try:
        reply = session.request(op, payload)
    except sessions.SessionGone as exc:
        sessions.registry.drop(sid)
        raise HTTPException(409, f"the game ended unexpectedly: {exc}") from exc
    return _session_reply(response, sid, reply)


@app.get("/api/state")
def state(response: Response, x_session: str | None = Header(default=None)) -> dict:
    return _forward("state", {}, response, x_session)


@app.post("/api/select")
def select(choice: Choice, response: Response,
           x_session: str | None = Header(default=None)) -> dict:
    return _forward("select", choice.model_dump(), response, x_session)


@app.post("/api/leave")
def leave(x_session: str | None = Header(default=None)) -> dict:
    """Give the session's processes back — a closed tab should not hold a slot."""
    sessions.registry.drop(x_session)
    return {"ok": True}


@app.get("/api/sessions")
def session_stats() -> dict:
    """How many games this server is running, and how many it will run."""
    return sessions.registry.stats()


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
    # A container publishes a port to the outside world, so the defaults come
    # from the environment: Hugging Face Spaces sets PORT=7860 and expects
    # 0.0.0.0. Locally, neither is set and it stays a loopback server on 8000.
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
