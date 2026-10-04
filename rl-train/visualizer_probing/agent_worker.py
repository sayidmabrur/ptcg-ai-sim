"""Run one packaged submission bundle as a line-oriented JSON service.

Why a subprocess
----------------
A bundle ships its *own* generation of the policy modules, and they import each
other by bare name (``policy_experimental``, ``live``, ``observation`` ...), plus
its own copy of ``cg`` and hence its own ``libcg.so``. Whichever bundle is
imported first in a process wins those names for the life of that process, so
loading a second bundle in the same interpreter would silently run the second
bundle's weights through the first bundle's architecture -- or fail to load at
all. ``arena.py`` sidesteps this by building each bundle inside a fresh worker
process; the simulator does the same, so the agent dropdown can be changed
between games without restarting the server.

Protocol: one JSON observation per line on stdin, one JSON reply per line on
stdout -- ``{"action": [...]}`` or ``{"error": "..."}``. Nothing else may be
written to stdout, so the bundle's own imports are muted onto stderr.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path


def load_agent(bundle: Path):
    """Import ``bundle/main.py`` and return its ``agent`` callable."""
    for entry in (bundle, bundle / "policy_network"):
        if entry.is_dir():
            sys.path.insert(0, str(entry))
    os.chdir(bundle)  # some bundles resolve deck.csv relative to the cwd
    spec = importlib.util.spec_from_file_location("submission_main", bundle / "main.py")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
    return module.agent


def fallback(obs: dict) -> list[int]:
    """A legal selection, for when the bundle raises.

    The competition harness forfeits on a crash and every bundle already carries
    its own fallback; this is the same idea one level out, so a single bad move
    does not end the human's game.
    """
    select = obs["select"]
    count = min(random.randint(select["minCount"], select["maxCount"]), len(select["option"]))
    return random.sample(range(len(select["option"])), count)


def main() -> None:
    bundle = Path(sys.argv[1]).resolve()
    out = sys.stdout
    # The handshake: the parent waits for this before starting a game, so a
    # bundle that cannot load reports it instead of leaving "starting..." hanging.
    try:
        agent = load_agent(bundle)
    except Exception as exc:  # noqa: BLE001
        out.write(json.dumps({"ready": False, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
        out.flush()
        raise
    out.write(json.dumps({"ready": True}) + "\n")
    out.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        obs = json.loads(line)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                action = [int(i) for i in agent(obs)]
            reply = {"action": action}
        except Exception as exc:  # noqa: BLE001 - report and keep the game alive
            reply = {"action": fallback(obs), "error": f"{type(exc).__name__}: {exc}"}
        out.write(json.dumps(reply) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
