"""Does the search actually play better, and does more of it play better still?

    PY=/home/kangh/miniconda3/envs/kaggle-pokemon/bin/python
    $PY probe_search_play.py --games 20 --sims 32

Two questions, and the second is the real one.

*Does it beat random?* A floor check. If IS-MCTS with rollout leaves cannot
beat uniform legal play, something in the stack is broken and no amount of
network will rescue it.

*Does win rate rise with simulations?* This is the soundness test. A correct
search converges toward better play as its budget grows; a search with a sign
error, a broken backup, or actions keyed so that statistics from different
worlds are being merged incorrectly will plateau or *decay* with more
simulations, because it is converging harder onto a wrong estimate. Scaling is
the property that distinguishes "runs without crashing" from "works", and it is
cheap to measure now and expensive to debug later.

The search here uses ``RolloutEvaluator`` — no network anywhere. That is
deliberate: it is generation 0 of the distillation loop, and its (policy, value)
outputs are what a first network would be trained on.
"""

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from belief import BeliefSampler  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from search import (  # noqa: E402
    ISMCTS,
    RolloutEvaluator,
    SearchConfig,
    ValueNetEvaluator,
)
from value_features import load_value_net  # noqa: E402

OPPONENTS = ("crustle", "alakazam", "lucario")


def read_deck(path: Path) -> list[int]:
    lines = [line for line in path.read_text().split("\n") if line.strip()]
    if len(lines) != 60:
        raise SystemExit(f"{path} has {len(lines)} cards")
    return [int(line) for line in lines]


def random_action(select: dict, rng: random.Random) -> list[int]:
    count = min(
        rng.randint(select["minCount"], select["maxCount"]), len(select["option"])
    )
    return rng.sample(range(len(select["option"])), count)


def play_game(searcher, sampler, decks, seat, rng, max_steps, timings, node_counts,
              opponent=None, episode=0):
    """One battle: the search agent in ``seat``, ``opponent`` in the other.

    ``opponent=None`` is uniform random — a floor check. A ``FrozenPolicy`` is
    the number that actually compares to prior work: ``train.py``'s PPO run
    peaked at 28.3% against these same three BC agents, and beating random is
    not evidence of anything, since that run also beat random 79%.
    """
    obs, _ = battle_start(decks[0], decks[1])
    if obs is None:
        return None
    sampler.reset()
    if opponent is not None:
        opponent.reset(episode)
    steps = 0
    try:
        while obs["current"]["result"] == -1 and steps < max_steps:
            select = obs.get("select")
            if select is None:
                break
            if obs["current"]["yourIndex"] == seat:
                sampler.observe(obs, seat)
                # Setup frames place Pokémon face-down on both sides and
                # search_begin has no parameter for our own hidden active, so
                # the search cannot root there. Random play for those few frames
                # costs little and keeps the harness honest about it.
                if obs["current"]["turn"] < 1:
                    action = random_action(select, rng)
                else:
                    particles = sampler.sample(obs, seat, searcher.config.particles)
                    started = time.perf_counter()
                    result = searcher.run(obs, seat, particles)
                    timings.append(time.perf_counter() - started)
                    node_counts.append(result.stats.nodes)
                    action = result.action
            elif opponent is not None:
                action = opponent.act(obs)
            else:
                action = random_action(select, rng)
            obs = battle_select(action)
            steps += 1
        return obs["current"]["result"]
    except (ValueError, IndexError, RuntimeError):
        return None
    finally:
        battle_finish()


def evaluate(sims, games, particles, rollout_depth, max_steps, seed, decks_all,
             value_net=None, frozen=None):
    challenger_deck, opponent_decks = decks_all
    if value_net is not None:
        evaluator = ValueNetEvaluator(value_net)
    else:
        evaluator = RolloutEvaluator(depth=rollout_depth, rng=random.Random(seed))
    searcher = ISMCTS(
        evaluator, SearchConfig(simulations=sims, particles=particles, seed=seed)
    )
    rng = random.Random(seed)
    wins = played = 0
    timings: list[float] = []
    node_counts: list[int] = []
    for game in range(games):
        seat = game % 2
        opponent_index = game % len(OPPONENTS)
        opponent = frozen[opponent_index] if frozen else None
        decks = [None, None]
        decks[seat] = challenger_deck
        decks[1 - seat] = opponent.deck if opponent else opponent_decks[opponent_index]
        sampler = BeliefSampler(
            challenger_deck, opponent_decks, rng=random.Random(seed * 977 + game)
        )
        result = play_game(
            searcher, sampler, decks, seat, rng, max_steps, timings, node_counts,
            opponent, game,
        )
        if result is None:
            continue
        played += 1
        wins += result == seat
    return {
        "sims": sims,
        "played": played,
        "wins": wins,
        "win_rate": wins / played if played else float("nan"),
        "decisions": len(timings),
        "seconds_per_decision": statistics.median(timings) if timings else float("nan"),
        "nodes_per_search": statistics.median(node_counts) if node_counts else 0,
    }


def build_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--sims", type=int, default=32)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--rollout-depth", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent", default="random", choices=("random", "frozen"),
                        help="'frozen' plays the three BC agents — the only "
                             "result comparable to train.py's 28.3%% eval")
    parser.add_argument("--value-net", default=None,
                        help="checkpoint from probe_value.py; replaces rollout leaves")
    parser.add_argument("--scaling", default=None,
                        help="comma-separated simulation counts to sweep, e.g. 4,16,64. "
                             "Win rate must rise with budget or the search is unsound")
    return parser.parse_args(argv)


def main() -> None:
    args = build_args()
    challenger_deck = read_deck(_ROOT / "challenger" / "challenger_deck.csv")
    opponent_decks = [read_deck(_ROOT / f"{n}_frozen" / "deck.csv") for n in OPPONENTS]
    decks_all = (challenger_deck, opponent_decks)

    frozen = None
    if args.opponent == "frozen":
        import torch

        from train import OPPONENTS as FROZEN_NAMES, FrozenPolicy
        frozen = [FrozenPolicy(n, _ROOT / f"{n}_frozen", torch.device("cpu"))
                  for n in FROZEN_NAMES]
    value_net = None
    if args.value_net:
        value_net, metrics = load_value_net(Path(args.value_net))
        ev = metrics.get("overall", {}).get("explained_variance")
    print("=" * 74)
    print(f"IS-MCTS vs {'frozen BC agents' if frozen else 'uniform random'} "
          f"({'value net, expl.var %.3f' % ev if value_net is not None else 'rollout leaves'})")
    print("=" * 74)
    print(f"games/point {args.games}   particles {args.particles}   "
          f"rollout depth {args.rollout_depth}")
    print()
    budgets = (
        [int(s) for s in args.scaling.split(",")] if args.scaling else [args.sims]
    )
    header = f"{'sims':>6}  {'win rate':>9}  {'w/p':>8}  {'s/decision':>11}  {'nodes':>7}"
    print(header)
    print("-" * len(header))
    rows = []
    for sims in budgets:
        row = evaluate(
            sims, args.games, args.particles, args.rollout_depth,
            args.max_steps, args.seed, decks_all, value_net, frozen,
        )
        rows.append(row)
        print(f"{row['sims']:>6}  {row['win_rate']:>8.1%}  "
              f"{row['wins']:>3}/{row['played']:<4}  "
              f"{row['seconds_per_decision']:>10.3f}s  {row['nodes_per_search']:>7.0f}",
              flush=True)

    if len(rows) > 1:
        first, last = rows[0]["win_rate"], rows[-1]["win_rate"]
        print()
        print(f"scaling: {first:.1%} @ {rows[0]['sims']} sims  ->  "
              f"{last:.1%} @ {rows[-1]['sims']} sims  "
              f"({'RISES' if last > first else 'DOES NOT RISE'})")


if __name__ == "__main__":
    main()
