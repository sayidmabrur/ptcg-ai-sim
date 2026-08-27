"""The opponents you can face, registered as bundles under ``simulator/agents/``.

A checkpoint is not separable from its decklist: every bundle here was trained
piloting the 60 cards sitting next to it, and handing it someone else's list
measures neither the policy nor the deck. So one dropdown entry names both.

A registered opponent is a directory in ``simulator/agents/`` holding

    main.py          an ``agent(obs) -> list[int]`` entry point
    deck.csv         the 60 card ids it was trained on
    *.pt             the weights
    policy_network/  the generation of the policy modules those weights fit

which is the same artefact the competition consumes, so what you play against
here is what would be submitted. To add an opponent, copy a bundle directory in;
nothing else registers it. Bundles elsewhere in the repo are deliberately *not*
discovered — several of them fail to load or hang, and a menu that lists
opponents which cannot play is worse than a short menu.

Each bundle runs in its own process (see ``agent_worker.py``): bundles import
their policy modules by bare name and ship their own ``cg``/``libcg.so``, so
whichever loaded first in a shared interpreter would capture those names and run
the next bundle's weights through the wrong architecture.
"""

from __future__ import annotations

import json
import select
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
AGENT_DIR = _HERE / "agents"

# A bundle loads torch and a ~10M-parameter checkpoint before it can answer, so
# the first response is slow; after that a decision is milliseconds. Both waits
# are bounded, because a bundle that never answers used to hang the whole server
# on "starting..." with nothing to report.
START_TIMEOUT = 180.0
MOVE_TIMEOUT = 60.0


def _deck_ok(path: Path) -> bool:
    try:
        lines = [line for line in path.read_text().split("\n") if line.strip()]
    except OSError:
        return False
    return len(lines) == 60


def discover_bundles() -> dict[str, Path]:
    """Every registered opponent, keyed by its directory name.

    Registered means: a directory under ``simulator/agents/`` with a ``main.py``,
    a 60-card ``deck.csv`` and weights to load.
    """
    found: dict[str, Path] = {}
    if not AGENT_DIR.is_dir():
        return found
    for bundle in sorted(p for p in AGENT_DIR.iterdir() if p.is_dir()):
        if not (bundle / "main.py").is_file():
            continue
        if not _deck_ok(bundle / "deck.csv"):
            continue
        if not any(bundle.glob("*.pt")):
            continue
        found[bundle.name] = bundle
    return found


def available_agents() -> list[str]:
    return list(discover_bundles())


def agent_deck(spec: str) -> Path:
    """The decklist that comes with an opponent."""
    bundles = discover_bundles()
    if spec not in bundles:
        raise KeyError(spec)
    return bundles[spec] / "deck.csv"


class BundleAgent:
    """A registered bundle, driven over a pipe in its own interpreter."""

    def __init__(self, spec: str, bundle: Path) -> None:
        self.name = spec
        self.bundle = bundle
        self.last_error: str | None = None
        self.process = subprocess.Popen(
            [sys.executable, str(_HERE / "agent_worker.py"), str(bundle)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, cwd=str(bundle),
        )
        # The worker says "ready" once its policy is loaded. Waiting for it here
        # means a bundle that cannot load fails when the game is started, with a
        # message, instead of on its first decision — or never.
        line = self._read(START_TIMEOUT, "loading")
        hello = json.loads(line)
        if not hello.get("ready"):
            self.close()
            raise RuntimeError(f"{spec} failed to load: {hello.get('error')}")

    def _read(self, timeout: float, what: str) -> str:
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            self.close()
            raise RuntimeError(f"{self.name} stopped responding while {what} "
                               f"(no answer in {timeout:.0f}s)")
        line = self.process.stdout.readline()
        if not line:
            code = self.process.poll()
            self.close()
            raise RuntimeError(f"{self.name} exited while {what} (code {code}) — "
                               f"its own error is on the server's stderr")
        return line

    def reset(self, episode: int) -> None:
        pass

    def act(self, obs: dict) -> list[int]:
        if self.process.poll() is not None:
            raise RuntimeError(f"{self.name} exited (code {self.process.returncode})")
        self.process.stdin.write(json.dumps(obs) + "\n")
        self.process.stdin.flush()
        reply = json.loads(self._read(MOVE_TIMEOUT, "choosing a move"))
        self.last_error = reply.get("error")
        return reply["action"]

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


def build_agent(spec: str) -> BundleAgent:
    bundles = discover_bundles()
    if spec not in bundles:
        raise KeyError(f"unknown opponent {spec}")
    return BundleAgent(spec, bundles[spec])
