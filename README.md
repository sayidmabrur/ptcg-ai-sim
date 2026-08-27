# Pokémon TCG — play against a trained agent

A browser simulator for the Pokémon Trading Card Game where a human plays a full
match against a reinforcement-learning agent, on the official competition engine
from the [Pokémon TCG AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle).

Built for research and study: the point is to make a trained policy's play
*legible* — to watch an agent take its turn one decision at a time, from the
same seat and with the same hidden information a human opponent would have.

    cd simulator
    ./build_engine.sh                 # compiles the engine from C++ source (~30s)
    python server.py --port 8000      # then open http://127.0.0.1:8000

Python 3.12+, a C++20 compiler, and `fastapi`/`uvicorn`. The agent bundle also
needs `torch`.

## What is here

| | |
| --- | --- |
| `simulator/` | the whole application — engine, agent, web front end |
| `simulator/README.md` | **the detailed write-up**: design decisions and why each one was made |

The training pipeline that produced the agent (behavioural cloning on ~10⁵
replays, then PPO) lives in a separate repository; this one is the simulator and
the packaged inference code it plays against.

## Points of interest

**Imperfect information is structural, not cosmetic.** The engine emits an
observation for the seat that owns the current decision, with the other seat's
hand, deck and prizes erased. The server keeps only the human's observations and
discards the agent's, so the payload the browser receives cannot contain the
opponent's hand — there is nothing hidden client-side to uncover.

**An agent's turn is replayed, not applied.** A turn arrives from the engine as
one batch. The server snapshots the board after each of the agent's decisions,
and the browser rewinds to the position before the agent moved and walks it
forward — one action per second, with the card that moved flying between zones,
so the narration and the board change together.

**The engine is built from source, not shipped as a binary.** `build_engine.sh`
compiles the C++ package into the `libcg.so` the Python bindings load, and
installs the same binary into each agent bundle, so there is one engine
everywhere and no stale binary in version control.

## Attribution and licensing

`simulator/ptcg_engine/` is the competition's engine package, provided for
competition use only under the licence in its `src/LICENSES/` directory; it is
not open-source and is included here only so the simulator runs. Pokémon, card
names, card text and card artwork are trademarks and copyright of
Pokémon/Nintendo/Creatures/GAME FREAK. Card scans are not distributed with this
repository. Everything else — the simulator, the server, the front end — is my
own work.
