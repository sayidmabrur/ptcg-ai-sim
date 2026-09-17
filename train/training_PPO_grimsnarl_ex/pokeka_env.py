"""The environment layer, taken from pokeka2026 (``ptcg_rl.core.engine.PTCGBattle``).

Why this file exists
--------------------
``cg.game`` keeps the battle pointer in a *class attribute* (``cg.sim.Battle.battle_ptr``),
so a process can hold exactly one battle at a time. That single fact is what shaped
``train.py``'s whole rollout design: one battle per process means one decision in flight,
which means every rollout forward is **batch 1** over a 2.4M-parameter transformer — Python
and kernel-launch bound, ~390 module calls deep, measured 7.6 ms on CPU against 12.6 ms on
this box's GPU. Hence "rollouts on CPU, updates on GPU", and hence scaling by forking
processes rather than by batching.

pokeka's ``PTCGBattle`` owns its pointer explicitly (``self.ptr``) instead of parking it on
the class, and its docstring records that concurrent battles in one process are verified
safe. Re-verified here before use (``scripts``-free probe, 8 battles interleaved to
completion, distinct results). That is the whole environment change: with N battles live in
one process, the N decisions waiting on them collate into **one** forward, and the GPU
finally has something to do. Measured on the crustle network, batch-64 forward costs 16 ms
on the GPU against 74 ms on the CPU, and the GPU cost is flat in the batch size up to 64 —
so the per-decision inference cost falls by roughly 20x versus batch 1.

Which native library
--------------------
**This repository's** ``cg/libcg.so``, not pokeka's. ``PTCG_CG_DIR`` is set below to point
pokeka's engine locator at ``training_PPO_crustle/`` so exactly one copy of the library is
loaded and one ``GameInitialize()`` runs. This is not a detail: ``vocab.py`` builds its card
and attack tables live from ``all_card_data``/``all_attack``, so the embedding tables in
every ``bc_policy.pt`` here are sized by *this* library. pokeka ships a newer, different
``libcg.so`` (the md5s differ); resolving to it would change the vocab sizes and make every
checkpoint in this repo — including the crustle warm start the current runs depend on —
unloadable.

Only ``PTCGBattle`` is imported. It reaches ``ptcg_rl.core._cg`` for the ctypes handle and
nothing else, so none of pokeka's feature encoders, deck registry or config models come
along; the features, the policy, the reward shaping and the PPO update all stay this
repository's.
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

#: pokeka2026 checkout providing the environment. Override with ``POKEKA_ROOT``.
POKEKA_ROOT = Path(
    os.environ.get("POKEKA_ROOT", "/home/steve/projects/pokeka2026-main")
).expanduser()


def _install() -> None:
    """Make ``ptcg_rl`` importable and pin its engine to this repo's ``cg``."""
    if not (POKEKA_ROOT / "ptcg_rl" / "core" / "engine.py").is_file():
        raise SystemExit(
            f"pokeka2026 not found at {POKEKA_ROOT} (set POKEKA_ROOT). Its "
            f"ptcg_rl/core/engine.py provides the per-instance battle pointer that "
            f"vectorised rollouts need."
        )
    # ptcg_rl.core._cg reads this at import time and prepends the directory *containing*
    # ``cg`` to sys.path. Set unconditionally: a stale value pointing at another checkout
    # would silently load a different engine build, which is the one failure mode here that
    # produces wrong numbers rather than an error.
    os.environ["PTCG_CG_DIR"] = str(_ROOT)
    if str(POKEKA_ROOT) not in sys.path:
        sys.path.append(str(POKEKA_ROOT))


_install()

from ptcg_rl.core.engine import EngineError, PTCGBattle  # noqa: E402
from ptcg_rl.core._cg import CG_PARENT  # noqa: E402

if Path(CG_PARENT).resolve() != _ROOT:
    raise SystemExit(
        f"pokeka's engine resolved to {CG_PARENT}, not {_ROOT}. The frozen checkpoints' "
        f"vocab tables are sized by this repo's libcg.so; refusing to train against a "
        f"different engine build."
    )

__all__ = ["PTCGBattle", "EngineError", "POKEKA_ROOT"]
