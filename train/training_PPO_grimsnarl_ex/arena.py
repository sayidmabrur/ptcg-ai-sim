"""Parallel game runner: benchmarking and value-data generation, N processes.

Every question left in this project is a win-rate comparison, and win rates are
expensive to measure. Distinguishing 21.7% from 28.3% at p<0.05 needs ~350 games
a side; at one core and ~4 s a game that is 40 minutes per data point, which is
why the last several experiments came back at n=30 and answered nothing.

The engine is a per-process singleton (one global battle pointer in ``cg.sim``),
so games cannot be batched — but they parallelise across processes almost
perfectly, and this box has 12 cores. That turns a 40-minute question into a
6-minute one, which is the difference between iterating and guessing.

The same fan-out generates value-function training data, because the work is
identical: play games, record what happened. ``--collect`` writes per-frame
scalar features with the game's outcome attached.

Policies are named by string so a worker can build its own — ``FrozenPolicy``
holds a torch module and an engine handle, neither of which pickles usefully:

    random                  uniform legal moves
    frozen:crustle          a BC checkpoint, greedy
    search                  IS-MCTS + value net (needs --value-net)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

OPPONENTS = ("crustle", "alakazam", "lucario")


def read_deck(path: Path) -> list[int]:
    lines = [line for line in path.read_text().split("\n") if line.strip()]
    if len(lines) != 60:
        raise SystemExit(f"{path} has {len(lines)} cards")
    return [int(line) for line in lines]


def split_spec(spec: str) -> tuple[str, str]:
    """``policy@deck`` -> ``(policy, deck)``; the deck defaults to the policy's own.

    The override exists because the challenger deck is never piloted by a BC
    agent otherwise, so value data generated from frozen-vs-frozen games would
    contain no states from the list we actually play. ``frozen:crustle@challenger``
    puts a competent policy behind our cards.
    """
    if spec.startswith("bundle:"):
        return spec, spec
    if "@" in spec:
        policy, deck = spec.split("@", 1)
        return policy, deck
    return spec, spec


def deck_for(spec: str) -> list[int]:
    if spec.startswith("bundle:"):
        return read_deck(Path(spec.split(":", 1)[1]) / "deck.csv")
    _, deck = split_spec(spec)
    if deck.startswith("frozen:"):
        deck = deck.split(":", 1)[1]
    if deck in OPPONENTS:
        return read_deck(_ROOT / f"{deck}_frozen" / "deck.csv")
    return read_deck(_ROOT / "challenger" / "challenger_deck.csv")


# --------------------------------------------------------------------------
# Worker


def _build_policy(spec: str, deck, config: dict):
    """Construct a policy inside the worker. Never crosses a process boundary."""
    import torch

    spec, _ = split_spec(spec)

    if spec == "random":
        class _Random:
            name = "random"

            def reset(self, episode):
                pass

            def act(self, obs):
                select = obs["select"]
                count = min(
                    random.randint(select["minCount"], select["maxCount"]),
                    len(select["option"]),
                )
                return random.sample(range(len(select["option"])), count)

        return _Random()

    if spec.startswith("bundle:"):
        # Load a packaged submission and drive its ``agent(obs_dict)``.
        #
        # Built *first* in ``_worker`` on purpose: the bundle ships its own
        # generation of the policy modules, which import each other by bare
        # name (``policy_experimental``, ``live``, ...). ``train`` inserts the
        # challenger's copies of those same names at import time, so whichever
        # is imported first wins for the whole process. Loading the bundle
        # before ``train`` is what makes this test the bundle rather than a
        # hybrid of the bundle's weights and the repo's architecture.
        import importlib.util

        bundle = Path(spec.split(":", 1)[1]).resolve()
        for entry in (bundle, bundle / "policy_network"):
            if entry.is_dir() and str(entry) not in sys.path:
                sys.path.insert(0, str(entry))
        module_spec = importlib.util.spec_from_file_location(
            "submission_main", bundle / "main.py"
        )
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)

        class _Bundle:
            name = "bundle"

            def reset(self, episode):
                # The harness signals a new match by asking for the decklist,
                # which is what the bundle keys its own reset on.
                # ``Observation.logs`` has no default, so a bare
                # {"select": None} fails to construct. The harness always sends
                # the full shape.
                module.agent({"select": None, "logs": [], "current": None})

            def act(self, obs):
                return module.agent(obs)

        return _Bundle()

    if spec.startswith("frozen:"):
        from train import FrozenPolicy

        name = spec.split(":", 1)[1]
        return FrozenPolicy(name, _ROOT / f"{name}_frozen", torch.device("cpu"))

    if spec == "search":
        from belief import BeliefSampler
        from search import ISMCTS, SearchConfig, ValueNetEvaluator
        from value_features import load_value_net

        def bc_option_probs(policy, obs):
            """Softmax over a BC checkpoint's option logits for this decision.

            Appends the frame to the policy's own extractor, so its
            ``decision_chain`` stays a faithful record of *our* match — the
            caller must follow up with ``record_action`` for whatever the search
            ends up playing, or the chain drifts from reality.
            """
            from collate import collate_features
            from dataset import transform
            from train import tree_to

            observation = policy.extractor(obs)
            features = tree_to(collate_features([transform(observation)]), policy.device)
            with torch.no_grad():
                logits = policy.network(features)
            probs = torch.softmax(logits, dim=-1)[0]
            return [float(p) for p in probs]

        value_net, _ = load_value_net(Path(config["value_net"]),
                                      blend=config.get("blend", 0.0))
        opponent_decks = [read_deck(_ROOT / f"{n}_frozen" / "deck.csv") for n in OPPONENTS]

        setup_spec = config.get("setup_policy", "frozen:crustle")
        setup = _build_policy(setup_spec, deck, config) if setup_spec else None
        # One instance serves both roles so its decision_chain is a continuous
        # record of our match rather than two partial ones.
        prior_spec = config.get("root_prior", "")
        prior_policy = setup if (prior_spec and setup is not None) else None
        if prior_spec and prior_policy is None:
            prior_policy = _build_policy(prior_spec, deck, config)

        class _Search:
            name = "search"

            def __init__(self):
                self.searcher = ISMCTS(
                    ValueNetEvaluator(value_net),
                    SearchConfig(
                        simulations=config["simulations"],
                        particles=config["particles"],
                        max_depth=config.get("max_depth", 30),
                        c_puct=config.get("c_puct", 1.5),
                        selector=config.get("selector", "puct"),
                        prior=config.get("prior", "uniform"),
                        seed=config.get("seed", 0),
                    ),
                )
                self.sampler = BeliefSampler(deck, opponent_decks,
                                             rng=random.Random(config.get("seed", 0)))

            def reset(self, episode):
                self.sampler.reset()
                if setup is not None:
                    setup.reset(episode)
                # Distinct instance when setup is disabled but a root prior is
                # not; without this its extractor carries the previous game.
                if prior_policy is not None and prior_policy is not setup:
                    prior_policy.reset(episode)

            def act(self, obs):
                seat = obs["current"]["yourIndex"]
                self.sampler.observe(obs, seat)
                # Setup places Pokémon face-down on both sides and
                # ``search_begin`` has no parameter for our own hidden active,
                # so the search cannot root here. It used to fall back to
                # uniform random, which is indefensible: the opening board
                # decides which Pokémon is exposed to the first attack, and a
                # random active is a free knockout for a competent opponent. A
                # BC policy is not tailored to this decklist but it is trained
                # on the game, and anything is better than random on ~10 of the
                # most consequential decisions in the match.
                if obs["current"]["turn"] < 1:
                    if setup is not None:
                        return setup.act(obs)
                    select = obs["select"]
                    count = min(
                        random.randint(select["minCount"], select["maxCount"]),
                        len(select["option"]),
                    )
                    return random.sample(range(len(select["option"])), count)
                root_prior = None
                if prior_policy is not None:
                    root_prior = bc_option_probs(prior_policy, obs)
                self.searcher.set_root_prior(root_prior)
                particles = self.sampler.sample(obs, seat, self.searcher.config.particles)
                action = self.searcher.run(obs, seat, particles).action
                if prior_policy is not None:
                    # Record what we actually played, not what BC would have.
                    prior_policy.extractor.record_action(action)
                return action

        return _Search()

    raise ValueError(f"unknown policy spec: {spec!r}")


def _worker(task):
    """Play a block of games and return results (and optionally value rows)."""
    import warnings

    warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
    import torch

    torch.set_num_threads(1)  # batch-1 inference gains nothing from more

    (rank, games, spec_a, spec_b, config, collect, stride, seed) = task
    random.seed(seed * 100_003 + rank)
    torch.manual_seed(seed * 100_003 + rank)

    from cg.game import battle_finish, battle_select, battle_start
    from value_features import scalar_features

    deck_a, deck_b = deck_for(spec_a), deck_for(spec_b)
    policy_a = _build_policy(spec_a, deck_a, config)
    policy_b = _build_policy(spec_b, deck_b, config)

    wins = played = abandoned = 0
    seat_wins = [0, 0]
    seat_played = [0, 0]
    rows, labels, turns, game_ids, margins = [], [], [], [], []
    started = time.perf_counter()

    for game in games:
        # A alternates seats so the first-player advantage cannot leak into the
        # reported rate — measured at 53.2% for seat 0 among expert games.
        seat = game % 2
        decks = [None, None]
        decks[seat], decks[1 - seat] = deck_a, deck_b
        policies = [None, None]
        policies[seat], policies[1 - seat] = policy_a, policy_b

        obs, _ = battle_start(decks[0], decks[1])
        if obs is None:
            abandoned += 1
            continue
        policy_a.reset(game)
        policy_b.reset(game)
        frames, steps = [], 0
        result = None
        try:
            while obs["current"]["result"] == -1 and steps < config.get("max_steps", 500):
                select = obs.get("select")
                if select is None:
                    break
                actor = obs["current"]["yourIndex"]
                if collect and steps % stride == 0 and obs["current"]["turn"] >= 1:
                    # Both points of view: the labels are exact negatives, but
                    # the feature vectors are not, so this is a genuine
                    # symmetry augmentation rather than duplicated rows.
                    frames.append((
                        scalar_features(obs["current"], 0),
                        scalar_features(obs["current"], 1),
                        obs["current"]["turn"],
                    ))
                obs = battle_select(policies[actor].act(obs))
                steps += 1
            result = obs["current"]["result"]
            # The board *after* the game ended. A 6-0 sweep and a 6-5 grind are
            # both "+1" to an outcome-only target, and that is one bit per ~150
            # frames; the prize margin grades the same game continuously and is
            # the game's own scoreboard rather than an opinion about play.
            final = obs["current"]["players"]
            final_prizes = (len(final[0]["prize"]), len(final[1]["prize"]))
        except (ValueError, IndexError, RuntimeError):
            result = None
            final_prizes = (6, 6)
        finally:
            battle_finish()

        if result is None or steps >= config.get("max_steps", 500):
            abandoned += 1
            continue
        played += 1
        seat_played[seat] += 1
        if result == seat:
            wins += 1
            seat_wins[seat] += 1
        if collect and result != 2:
            # A stable global id: frames of one game share a label and are
            # near-duplicates, so a value split must be by game. Splitting by
            # row puts copies of every validation frame into training and the
            # score stops measuring generalisation.
            game_id = rank * 1_000_003 + game
            margin0 = (final_prizes[1] - final_prizes[0]) / 6.0
            for f0, f1, turn in frames:
                rows.append(f0)
                labels.append(1.0 if result == 0 else -1.0)
                margins.append(margin0)
                turns.append(turn)
                game_ids.append(game_id)
                rows.append(f1)
                labels.append(1.0 if result == 1 else -1.0)
                margins.append(-margin0)
                turns.append(turn)
                game_ids.append(game_id)

    return {
        "rank": rank, "wins": wins, "played": played, "abandoned": abandoned,
        "seat_wins": seat_wins, "seat_played": seat_played,
        "seconds": time.perf_counter() - started,
        "rows": np.asarray(rows, dtype=np.float32) if rows else None,
        "labels": np.asarray(labels, dtype=np.float32) if labels else None,
        "turns": np.asarray(turns, dtype=np.int32) if turns else None,
        "games": np.asarray(game_ids, dtype=np.int64) if game_ids else None,
        "margins": np.asarray(margins, dtype=np.float32) if margins else None,
    }


# --------------------------------------------------------------------------


def run(spec_a: str, spec_b: str, games: int, workers: int, config: dict,
        collect: bool = False, stride: int = 3, seed: int = 0, quiet: bool = False):
    import torch.multiprocessing as multiprocessing

    # spawn, not fork: workers build torch modules and hold their own engine, and
    # forking a process that has already initialised the shared library is not
    # something the engine documents as safe.
    context = multiprocessing.get_context("spawn")
    blocks = [list(range(rank, games, workers)) for rank in range(workers)]
    tasks = [
        (rank, block, spec_a, spec_b, config, collect, stride, seed)
        for rank, block in enumerate(blocks) if block
    ]
    started = time.perf_counter()
    with context.Pool(len(tasks)) as pool:
        results = pool.map(_worker, tasks)

    wins = sum(r["wins"] for r in results)
    played = sum(r["played"] for r in results)
    abandoned = sum(r["abandoned"] for r in results)
    seat_wins = [sum(r["seat_wins"][s] for r in results) for s in (0, 1)]
    seat_played = [sum(r["seat_played"][s] for r in results) for s in (0, 1)]
    rate = wins / played if played else float("nan")
    # Wilson interval, not normal-approximation: at the win rates in play here
    # (10-30%) with n in the low hundreds, the normal interval runs off the end
    # of [0, 1] and understates the lower bound.
    low, high = wilson(wins, played)
    out = {
        "a": spec_a, "b": spec_b, "wins": wins, "played": played,
        "abandoned": abandoned, "win_rate": rate, "ci95": (low, high),
        "seat_rates": [
            seat_wins[s] / seat_played[s] if seat_played[s] else float("nan")
            for s in (0, 1)
        ],
        "seconds": time.perf_counter() - started,
        "games_per_second": played / max(1e-9, time.perf_counter() - started),
    }
    if collect:
        out["rows"] = np.concatenate([r["rows"] for r in results if r["rows"] is not None])
        out["labels"] = np.concatenate([r["labels"] for r in results if r["labels"] is not None])
        for key in ("turns", "games", "margins"):
            out[key] = np.concatenate([r[key] for r in results if r[key] is not None])
    if not quiet:
        print(f"  {spec_a} vs {spec_b}: {rate:.1%} [{low:.1%}, {high:.1%}] "
              f"({wins}/{played}, {abandoned} abandoned) "
              f"seat0 {out['seat_rates'][0]:.1%} seat1 {out['seat_rates'][1]:.1%} "
              f"| {out['games_per_second']:.1f} games/s", flush=True)
    return out


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def versus_field(spec: str, games: int, workers: int, config: dict, seed: int = 0):
    """Against all three frozen agents — the number comparable to train.py."""
    total_wins = total_played = 0
    per = {}
    for name in OPPONENTS:
        result = run(spec, f"frozen:{name}", games, workers, config, seed=seed)
        per[name] = result["win_rate"]
        total_wins += result["wins"]
        total_played += result["played"]
    rate = total_wins / total_played if total_played else float("nan")
    low, high = wilson(total_wins, total_played)
    print(f"  OVERALL {rate:.1%} [{low:.1%}, {high:.1%}] ({total_wins}/{total_played})")
    return {"per_opponent": per, "win_rate": rate, "ci95": (low, high),
            "wins": total_wins, "played": total_played}


def build_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", default="search")
    parser.add_argument("--b", default="field", help="a policy spec, or 'field' "
                                                     "for all three frozen agents")
    parser.add_argument("--games", type=int, default=120)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=30)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--value-net", default=str(_ROOT / "value_scalar.pt"))
    parser.add_argument("--collect", default=None, metavar="NPZ",
                        help="write per-frame value training data here")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main() -> None:
    args = build_args()
    config = {
        "simulations": args.simulations, "particles": args.particles,
        "max_depth": args.max_depth, "c_puct": args.c_puct,
        "max_steps": args.max_steps, "value_net": args.value_net, "seed": args.seed,
    }
    print(f"[arena] {args.workers} workers, {args.games} games/matchup")
    started = time.perf_counter()
    if args.b == "field":
        versus_field(args.a, args.games, args.workers, config, args.seed)
    else:
        result = run(args.a, args.b, args.games, args.workers, config,
                     collect=bool(args.collect), stride=args.stride, seed=args.seed)
        if args.collect:
            np.savez_compressed(
                args.collect, rows=result["rows"], labels=result["labels"],
                turns=result["turns"], games=result["games"],
                margins=result["margins"],
            )
            print(f"  collected {len(result['labels']):,} value rows -> {args.collect}")
    print(f"[arena] {time.perf_counter() - started:.0f}s total")


if __name__ == "__main__":
    main()
