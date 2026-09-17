"""Is the search's uniform opponent model what stops it scaling?

    PY=/home/kangh/miniconda3/envs/kaggle-pokemon/bin/python
    $PY probe_opponent_model.py --games 30 --scaling 16,64

Two hypotheses survive the last round of measurement, and they lead to
completely different builds:

  A. The search models the opponent as near-random, so it plans lines that only
     pay off against a blunder. Fix: a policy prior, distilled cheap.
  B. The value function is too inaccurate on our matchups for any amount of
     search to help. Fix: refit V on BC self-play.

The evidence so far fits A neatly — the search scales against a genuinely
random opponent (64.0% -> 81.3% from 4 to 64 simulations) and is flat against
the BC agents (21.7% / 21.7% / 16.7% from 16 to 256) — but "fits neatly" is not
a measurement, and B would produce flatness too.

This separates them directly: put the **real** BC policy at the opponent's
nodes instead of PUCT. If the frozen win rate jumps, it is A; if it stays near
21.7%, it is B. Either answer saves building the wrong thing.

Not a production configuration
------------------------------
A BC forward is ~30 ms against ~0.1 ms for an engine step, so this is far too
slow to ship — the point is to learn which fix to build, not to ship this.

Two fidelity caveats, both of which make this a *lower bound* on how much an
accurate opponent model is worth:

*Truncated history.* The BC policy's ``decision_chain`` and
``opponent_history`` come from a row sequence the search does not have (its
frames are hypothetical continuations, not the real match). Each call gets a
fresh extractor, so the modelled opponent plays with no memory of the game so
far. The real one has it.

*Known identity.* The opponent's decklist posterior resolves to a single
candidate in a median of 5 frames, so naming the right BC agent is legitimate
here. In competition it would not be, and the shippable version is a distilled
prior rather than a specific opponent's weights.
"""

import argparse
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from belief import BeliefSampler  # noqa: E402
from probe_search_play import play_game, read_deck  # noqa: E402
from search import ISMCTS, SearchConfig, ValueNetEvaluator  # noqa: E402
from train import OPPONENTS, FrozenPolicy  # noqa: E402
from value_features import load_value_net  # noqa: E402


class BCOpponentModel:
    """Selects the opponent's moves inside the tree using their real BC policy.

    Cached on ``searchId``. Under ``manual_coin=True`` the engine is a pure
    function of (state, action) — probe P13, 60/60 frames — so a given search
    state always presents the same decision, and the greedy BC policy always
    answers it the same way. One forward per distinct state rather than one per
    visit turns a 30 ms cost into something a 64-simulation search can absorb.

    The cache is valid **within one search only**. ``search_end()`` recycles
    ids: measured, every search restarts them at 0. Carrying entries across
    decisions returns another position's move, which does not error — it
    surfaces as an out-of-range option index only when the two decisions happen
    to differ in width, and silently plays a wrong-but-legal move otherwise.
    ``ISMCTS.run`` calls ``clear()`` at the start of every search.

    Takes its own ``FrozenPolicy`` instance rather than the one playing the
    match: ``act`` mutates a per-episode extractor, so sharing it would have the
    search wipe the live opponent's own history mid-game.
    """

    def __init__(self, policy: FrozenPolicy):
        self.policy = policy
        self.cache: dict[int, list[int]] = {}
        self.calls = 0
        self.hits = 0

    def __call__(self, state, actor: int):
        cached = self.cache.get(state.searchId)
        if cached is not None:
            self.hits += 1
            return cached
        observation = state.observation
        if observation.select is None:
            return None
        obs = {
            "current": asdict(observation.current),
            "select": asdict(observation.select),
        }
        try:
            # Fresh memory per call: the search's frames are hypothetical, so
            # there is no real row sequence to seed the extractor with. See the
            # module docstring — this understates the modelled opponent.
            self.policy.reset(0)
            action = self.policy.act(obs)
        except Exception:  # noqa: BLE001 — fall back to PUCT rather than abort
            return None
        self.calls += 1
        self.cache[state.searchId] = action
        return action

    def clear(self) -> None:
        self.cache.clear()


def evaluate(sims, games, particles, max_steps, seed, decks_all, value_net,
             frozen, models, model_opponent: bool):
    challenger_deck, opponent_decks = decks_all
    rng = random.Random(seed)
    wins = played = 0
    timings: list[float] = []
    nodes: list[int] = []
    for game in range(games):
        seat = game % 2
        index = game % len(OPPONENTS)
        opponent = frozen[index]
        # The tree's model of the opponent is that same agent. Legitimate here
        # because the decklist posterior identifies them within ~5 frames.
        model = BCOpponentModel(models[index]) if model_opponent else None
        searcher = ISMCTS(
            ValueNetEvaluator(value_net),
            SearchConfig(simulations=sims, particles=particles, seed=seed),
            opponent_policy=model,
        )
        decks = [None, None]
        decks[seat] = challenger_deck
        decks[1 - seat] = opponent.deck
        sampler = BeliefSampler(
            challenger_deck, opponent_decks, rng=random.Random(seed * 977 + game)
        )
        result = play_game(
            searcher, sampler, decks, seat, rng, max_steps, timings, nodes,
            opponent, game,
        )
        if result is None:
            continue
        played += 1
        wins += result == seat
    return {
        "sims": sims, "played": played, "wins": wins,
        "win_rate": wins / played if played else float("nan"),
        "seconds_per_decision": statistics.median(timings) if timings else float("nan"),
        "nodes": statistics.median(nodes) if nodes else 0,
    }


def build_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--scaling", default="16,64")
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--value-net", default=str(_ROOT / "value_scalar.pt"))
    return parser.parse_args(argv)


def main() -> None:
    args = build_args()
    challenger_deck = read_deck(_ROOT / "challenger" / "challenger_deck.csv")
    opponent_decks = [read_deck(_ROOT / f"{n}_frozen" / "deck.csv") for n in OPPONENTS]
    value_net, _ = load_value_net(Path(args.value_net))
    frozen = [FrozenPolicy(n, _ROOT / f"{n}_frozen", torch.device("cpu"))
              for n in OPPONENTS]
    # Separate instances for the in-tree model — see BCOpponentModel.
    models = [FrozenPolicy(n, _ROOT / f"{n}_frozen", torch.device("cpu"))
              for n in OPPONENTS]

    print("=" * 78)
    print("opponent model inside the search: PUCT (uniform) vs the real BC policy")
    print("=" * 78)
    print(f"games/point {args.games}   particles {args.particles}   "
          f"vs frozen BC   floor (random) 8.3%   PPO best 28.3%")
    print()
    header = (f"{'sims':>6}  {'opponent model':>16}  {'win rate':>9}  {'w/p':>8}  "
              f"{'s/decision':>11}")
    print(header)
    print("-" * len(header))
    for sims in [int(s) for s in args.scaling.split(",")]:
        for label, modelled in (("PUCT uniform", False), ("real BC policy", True)):
            row = evaluate(
                sims, args.games, args.particles, args.max_steps, args.seed,
                (challenger_deck, opponent_decks), value_net, frozen, models, modelled,
            )
            print(f"{sims:>6}  {label:>16}  {row['win_rate']:>8.1%}  "
                  f"{row['wins']:>3}/{row['played']:<4}  "
                  f"{row['seconds_per_decision']:>10.3f}s", flush=True)
        print()


if __name__ == "__main__":
    main()
