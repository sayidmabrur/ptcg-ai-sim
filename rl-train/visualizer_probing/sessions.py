"""One game per visitor, as one worker process per session.

The engine holds its battle on a class attribute, so a process plays one game.
The web server therefore keeps no battle of its own: it keeps a table of
``game_worker.py`` processes, one per browser session, and forwards each request
to the worker that belongs to the caller. Two people can play at once because
their games are in different processes, which is the only isolation the native
engine allows.

A session costs real memory: the worker imports the engine, and the bundle it
plays against loads torch and a checkpoint in a third process. ``MAX_SESSIONS``
is therefore a hard cap and idle games are reaped, because a public Space that
runs out of memory kills everybody's game rather than the one that walked away.

Sessions are identified by an id the browser generates and sends as
``X-Session``, not by a cookie: the Space is normally viewed inside an iframe on
huggingface.co, where the Space's own cookies are third-party and quietly
dropped by default in current browsers. A header the page controls works the
same in an iframe as on the direct URL.
"""

from __future__ import annotations

import json
import os
import secrets
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# A worker imports the engine before it answers; a bundle then loads torch and a
# checkpoint on the first "new". Both waits are bounded so a stuck worker fails
# one request with a message instead of hanging a connection forever.
BOOT_TIMEOUT = 60.0
NEW_TIMEOUT = 240.0     # covers the bundle's own START_TIMEOUT (180 s)
REQUEST_TIMEOUT = 90.0

# How many games may run at once, and how long a game may sit untouched before
# its processes are reclaimed. Both are overridable from the environment because
# the right numbers are a property of the machine, not of the code.
MAX_SESSIONS = int(os.environ.get("PTCG_MAX_SESSIONS", "4"))
IDLE_TIMEOUT = float(os.environ.get("PTCG_IDLE_TIMEOUT", str(20 * 60)))


class SessionBusy(Exception):
    """The server is already running as many games as it will run."""


class SessionGone(Exception):
    """This session's worker died; the browser should start a new game."""


def new_session_id() -> str:
    return secrets.token_urlsafe(16)


class Session:
    """One visitor's game: a worker process, and the lock serialising it."""

    def __init__(self, sid: str) -> None:
        self.sid = sid
        self.lock = threading.Lock()
        self.touched = time.monotonic()
        self.started = time.monotonic()
        self.playing = False       # has a game been started in this worker?
        self.process = subprocess.Popen(
            [sys.executable, str(_HERE / "game_worker.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, cwd=str(_HERE),
        )
        hello = json.loads(self._readline(BOOT_TIMEOUT, "starting up"))
        if not hello.get("ready"):
            self.close()
            raise RuntimeError(f"game worker failed to start: {hello.get('error')}")

    def _readline(self, timeout: float, what: str) -> str:
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            self.close()
            raise SessionGone(f"the game stopped responding while {what} "
                              f"(no answer in {timeout:.0f}s)")
        line = self.process.stdout.readline()
        if not line:
            code = self.process.poll()
            self.close()
            raise SessionGone(f"the game exited while {what} (code {code})")
        return line

    def request(self, op: str, payload: dict | None = None) -> dict:
        """Forward one request to this session's worker and return its reply."""
        with self.lock:
            self.touched = time.monotonic()
            if self.process.poll() is not None:
                raise SessionGone(f"the game exited (code {self.process.returncode})")
            body = dict(payload or {}, op=op)
            timeout = NEW_TIMEOUT if op == "new" else REQUEST_TIMEOUT
            try:
                self.process.stdin.write(json.dumps(body) + "\n")
                self.process.stdin.flush()
            except (OSError, ValueError) as exc:
                self.close()
                raise SessionGone(f"the game is no longer reachable: {exc}") from exc
            reply = json.loads(self._readline(timeout, f"handling {op}"))
            if op == "new" and reply.get("ok"):
                self.playing = True
            self.touched = time.monotonic()
            return reply

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


class Registry:
    """The live sessions, and the rules for making and reclaiming them."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def _reap(self) -> None:
        """Drop sessions whose worker died or which nobody has touched."""
        now = time.monotonic()
        for sid, session in list(self._sessions.items()):
            # Never reclaim a session that is mid-request: closing its stdin
            # under the thread reading its stdout would fail a live game. The
            # timeouts already make this arithmetically impossible (a request is
            # bounded well below IDLE_TIMEOUT), which is a reason to check
            # rather than to rely on the arithmetic staying true.
            if session.lock.locked():
                continue
            dead = session.process.poll() is not None
            idle = now - session.touched > IDLE_TIMEOUT
            if dead or idle:
                del self._sessions[sid]
                session.close()

    def get(self, sid: str | None) -> Session | None:
        if not sid:
            return None
        with self._lock:
            self._reap()
            return self._sessions.get(sid)

    def open(self, sid: str | None) -> tuple[str, Session]:
        """The caller's session, started if this is their first game."""
        with self._lock:
            self._reap()
            if sid and sid in self._sessions:
                return sid, self._sessions[sid]
            if len(self._sessions) >= MAX_SESSIONS:
                raise SessionBusy(
                    f"{MAX_SESSIONS} games are already running on this server — "
                    f"try again in a few minutes, or run your own copy"
                )
            sid = sid or new_session_id()
            session = Session(sid)
            self._sessions[sid] = session
            return sid, session

    def drop(self, sid: str | None) -> None:
        if not sid:
            return
        with self._lock:
            session = self._sessions.pop(sid, None)
        if session is not None:
            session.close()

    def stats(self) -> dict:
        with self._lock:
            self._reap()
            return {
                "live": len(self._sessions),
                "max": MAX_SESSIONS,
                "idleTimeout": IDLE_TIMEOUT,
                "playing": sum(1 for s in self._sessions.values() if s.playing),
            }

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()


registry = Registry()
