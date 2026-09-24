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

*Two players share a room.* ``/api/room/*`` (``rooms.py``) runs a lobby at
``/?room=<id>``: the host names themselves and gets an invitation link, the
guest names themselves and lands on the same address, and once both have picked
a deck the room starts one worker holding a ``PvpBattle``. Each request is
mapped to its seat from the player's ``X-Player`` token, and each seat is only
ever shown its own observation, or the other seat's with the private parts
removed.

*The agent moves inside the request.* A checkpoint forward is milliseconds and
a whole agent turn is well under a second, so ``/api/select`` simply returns the
next position the human has to act on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
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
import replay_view  # noqa: E402
import rooms  # noqa: E402
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
    rooms.registry.close_all()


app = FastAPI(title="PTCG player-vs-agent simulator", lifespan=_lifespan)

# A replay is a board per decision: ~2 MB of JSON for a long game, ~90 KB
# gzipped. Worth it anywhere, and essential through a metered tunnel.
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402

app.add_middleware(GZipMiddleware, minimum_size=4096)

# The header the page identifies itself with. Not a cookie: the Space is usually
# viewed in an iframe on huggingface.co, where the Space's own cookies are
# third-party and dropped by default.
SESSION_HEADER = "X-Session"


class NewGame(BaseModel):
    agent: str                        # a registered opponent (simulator/agents/*)
    humanDeck: str | None = None      # a prebuilt decklist by name...
    humanCards: list[int] | None = None  # ...or 60 card ids built in the browser
    seat: int = -1                    # -1 picks at random
    playerName: str | None = None     # names the saved replay file


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


# -- two players -------------------------------------------------------------
#
# The room is the page (``/?room=<id>``) and the player is a token the server
# issued when they gave their name, sent back as ``X-Player``. Every request is
# resolved to a seat here, from the token, before it reaches the room's worker.

PLAYER_HEADER = "X-Player"


class RoomName(BaseModel):
    name: str


class RoomJoin(BaseModel):
    room: str
    name: str


class RoomReady(BaseModel):
    room: str
    ready: bool = True
    deck: str | None = None           # a prebuilt decklist by name...
    cards: list[int] | None = None    # ...or 60 card ids built in the browser


class RoomRef(BaseModel):
    room: str


class RoomChoice(BaseModel):
    room: str
    options: list[int]


def _room_call(fn):
    try:
        return fn()
    except rooms.RoomError as exc:
        raise HTTPException(exc.status, exc.detail) from exc
    except sessions.SessionGone as exc:
        raise HTTPException(409, f"the game ended unexpectedly: {exc}") from exc


def _room_game_raw(room: rooms.Room, token: str | None, op: str) -> dict:
    me = room.player(token)
    if room.worker is None or me.seat is None:
        raise rooms.RoomError(409, "the game has not started")
    reply = room.worker.request(op, {"seat": me.seat})
    if not reply.get("ok"):
        raise rooms.RoomError(reply.get("status", 500), reply.get("detail", "game error"))
    return reply["view"]


def _room_game(room: rooms.Room, token: str | None, op: str, payload: dict) -> dict:
    me = room.player(token)
    if room.worker is None or me.seat is None:
        raise rooms.RoomError(409, "the game has not started")
    reply = room.worker.request(op, dict(payload, seat=me.seat))
    if not reply.get("ok"):
        raise rooms.RoomError(reply.get("status", 500), reply.get("detail", "game error"))
    view = reply["view"]
    view["room"] = room.id
    return view


@app.post("/api/room/create")
def room_create(request: RoomName) -> dict:
    """Open a room; the caller is its host and is sent to its lobby."""
    def run():
        room, me = rooms.registry.create(request.name)
        return {"room": room.id, "token": me.token, "lobby": room.lobby(me)}
    return _room_call(run)


@app.post("/api/room/join")
def room_join(request: RoomJoin) -> dict:
    """Take the second seat in a room, from the link its host sent."""
    def run():
        room, me = rooms.registry.join(request.room, request.name)
        return {"room": room.id, "token": me.token, "lobby": room.lobby(me)}
    return _room_call(run)


@app.get("/api/room/info")
def room_info(room: str) -> dict:
    """Before joining: does the room exist, who is hosting, is there a seat."""
    def run():
        r = rooms.registry.get(room)
        return {"room": r.id, "host": r.players[0].name, "full": len(r.players) == 2,
                "status": r.status}
    return _room_call(run)


# How long a poll may be held open waiting for something to change. Long polls
# are what make the game affordable through a metered tunnel (ngrok's free plan
# counts every request): a waiting player costs one request per change, or one
# per LONG_POLL seconds when nothing happens, instead of one a second.
LONG_POLL = float(os.environ.get("PTCG_LONG_POLL", "25"))
POLL_STEP = 0.3


def _lobby_with_tag(r: rooms.Room, me: rooms.Player) -> dict:
    lobby = r.lobby(me)
    lobby["tag"] = hashlib.sha1(json.dumps(lobby, sort_keys=True).encode()).hexdigest()[:12]
    return lobby


@app.get("/api/room/lobby")
def room_lobby(room: str, since: str | None = None,
               x_player: str | None = Header(default=None)) -> dict:
    """The lobby; with ``since``, held until it differs from that tag."""
    def run():
        r = rooms.registry.get(room)
        me = r.player(x_player)
        deadline = time.monotonic() + (LONG_POLL if since else 0)
        while True:
            lobby = _lobby_with_tag(r, me)
            if lobby["tag"] != since or time.monotonic() >= deadline:
                return lobby
            time.sleep(POLL_STEP)
            r = rooms.registry.get(room)   # keeps the room from idling out, and fails if it closed
    return _room_call(run)


@app.post("/api/room/ready")
def room_ready(request: RoomReady, x_player: str | None = Header(default=None)) -> dict:
    """Pick a deck and say you are ready; the game starts once both have."""
    def run():
        r = rooms.registry.get(request.room)
        with r.lock:
            me = r.player(x_player)
            if r.status != "lobby":
                raise rooms.RoomError(409, "the game has already started")
            if request.ready:
                if request.cards is not None:
                    deck = [int(c) for c in request.cards]
                    label = "own deck"
                elif request.deck is not None:
                    try:
                        deck = engine.deck_by_name(request.deck)
                    except KeyError as exc:
                        raise rooms.RoomError(400, f"unknown deck {exc}") from exc
                    label = request.deck
                else:
                    raise rooms.RoomError(400, "choose a deck first")
                problems = engine.deck_check(deck, render.card_db())
                if problems:
                    raise rooms.RoomError(400, "illegal deck: " + "; ".join(problems))
                me.deck, me.deck_label, me.ready = deck, label, True
            else:
                me.ready = False
            r.error = ""
            if len(r.players) == 2 and all(p.ready for p in r.players):
                try:
                    r.start()
                except (rooms.RoomError, RuntimeError, OSError) as exc:
                    for p in r.players:
                        p.ready = False
                    r.error = getattr(exc, "detail", str(exc))
            return _lobby_with_tag(r, me)
    return _room_call(run)


@app.get("/api/room/state")
def room_state(room: str, since: int | None = None,
               x_player: str | None = Header(default=None)) -> dict:
    """This seat's view; with ``since``, held until the game moves past that version."""
    def run():
        r = rooms.registry.get(room)
        if since is not None:
            deadline = time.monotonic() + LONG_POLL
            while time.monotonic() < deadline:
                probe = _room_game_raw(r, x_player, "pvp_version")
                if probe["version"] != since:
                    break
                time.sleep(POLL_STEP)
                r = rooms.registry.get(room)
        return _room_game(r, x_player, "pvp_state", {})
    return _room_call(run)


@app.post("/api/room/select")
def room_select(choice: RoomChoice, x_player: str | None = Header(default=None)) -> dict:
    return _room_call(lambda: _room_game(rooms.registry.get(choice.room), x_player,
                                         "pvp_select", {"options": choice.options}))


@app.post("/api/room/leave")
def room_leave(request: RoomRef, x_player: str | None = Header(default=None)) -> dict:
    """Leave a room. The host leaving, or anyone leaving a game, closes it."""
    def run():
        r = rooms.registry.get(request.room)
        me = r.player(x_player)
        if r.status == "lobby" and me is not r.players[0]:
            with r.lock:
                r.players.remove(me)
                for p in r.players:
                    p.ready = False
        else:
            rooms.registry.drop(r.id)
        return {"ok": True}
    return _room_call(run)


def _ngrok_url() -> str | None:
    """The public https address of a local ngrok agent, if one is running.

    ngrok's agent answers on 127.0.0.1:4040 with the tunnels it holds. Asking it
    means the invitation link is right even when the host opened the page on
    localhost -- a link to localhost would take the guest to their own machine.
    """
    try:
        with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=0.5) as r:
            tunnels = json.loads(r.read()).get("tunnels", [])
    except (OSError, ValueError):
        return None
    for t in tunnels:
        if str(t.get("public_url", "")).startswith("https://"):
            return t["public_url"]
    return None


@app.get("/api/public-url")
def public_url() -> dict:
    """Where a guest can reach this server: PTCG_PUBLIC_URL, else ngrok, else unknown."""
    url = os.environ.get("PTCG_PUBLIC_URL") or _ngrok_url()
    return {"url": url.rstrip("/") if url else None}


@app.get("/api/replays")
def replays() -> dict:
    """Saved games, newest first (``replays/``, see replay_log.py)."""
    return {"replays": replay_view.listing()}


@app.get("/api/replays/view")
def replay(file: str, seat: int = 0) -> dict:
    """One saved game as a board per decision, seen from ``seat``, both hands shown."""
    if seat not in (0, 1):
        raise HTTPException(400, "seat must be 0 or 1")
    try:
        path = replay_view.resolve(file)
    except FileNotFoundError as exc:
        raise HTTPException(404, "no such replay") from exc
    try:
        return replay_view.frames(path, seat)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        raise HTTPException(422, f"cannot read this replay: {type(exc).__name__}: {exc}") from exc


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
#
# PTCG_LIGHT_IMAGES=1 serves lossy copies instead (about a tenth of the size),
# made on first request and kept in .image_cache/. That is for a metered tunnel:
# the lossless scans are ~700 KB each, and ngrok's free plan allows 1 GB a month.
LIGHT_IMAGES = os.environ.get("PTCG_LIGHT_IMAGES", "") not in ("", "0", "false")
LIGHT_DIR = _HERE / ".image_cache"

if LIGHT_IMAGES and render.IMAGE_DIR.is_dir():
    from fastapi.responses import FileResponse

    @app.get("/cards/{name}")
    def light_card(name: str) -> FileResponse:
        source = render.IMAGE_DIR / Path(name).name
        if not source.is_file():
            raise HTTPException(404, "no such card image")
        target = LIGHT_DIR / (source.stem + ".webp")
        if not target.is_file() or target.stat().st_mtime < source.stat().st_mtime:
            from PIL import Image
            LIGHT_DIR.mkdir(exist_ok=True)
            Image.open(source).convert("RGB").save(target, format="WEBP", quality=80, method=4)
        return FileResponse(target, media_type="image/webp",
                            headers={"Cache-Control": "public, max-age=604800"})
elif render.IMAGE_DIR.is_dir():
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
