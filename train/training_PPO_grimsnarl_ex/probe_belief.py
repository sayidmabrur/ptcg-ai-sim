"""Does the belief sampler produce worlds the real game could be in?

    PY=/home/kangh/miniconda3/envs/kaggle-pokemon/bin/python
    $PY probe_belief.py --games 40

Probe P9 showed ``search_begin`` validates nothing semantic, so a wrong
particle produces a confident search result and no error anywhere. That makes
``belief.py`` untestable by its own outputs — "it returned a particle" says
nothing. It needs ground truth.

Where the ground truth comes from
---------------------------------
Every observation's ``state["players"]`` carries **both** players; only the
non-owner's ``hand`` is redacted to ``None``. So at a frame owned by player P,
``players[P]["hand"]`` is P's real hand, while every other field of
``players[P]`` is exactly what their opponent can see.

That is a complete, same-timestep test, and it needs no instrumentation of the
engine: compute the hidden pool for P *using only the public fields*, then
check the real hand is a sub-multiset of it. If it ever is not, the sampler is
capable of drawing a world that excludes the truth — and every search rooted
there is solving the wrong game.

What is checked
---------------
1. **Accounting closes.** Pool size equals deck + prize + hand exactly. A
   mismatch means a card left the 60 by a route ``visible_cards`` does not
   model, and names the zone to go fix.
2. **Truth is in support.** The real hand is drawable from the pool.
3. **Particles validate.** ``validate_particle`` finds no contradiction.
4. **The engine accepts them.** ``search_begin`` takes the particle and returns
   a root — the end-to-end check that the conventions in ``belief.py`` (top of
   deck last, Pokémon-only active, exact lengths) are right.
5. **The posterior converges.** How many frames until the opponent's decklist
   is uniquely identified, and whether the true list is ever eliminated —
   which would be a logic bug, not an approximation.
"""

import argparse
import collections
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from belief import (  # noqa: E402
    BeliefSampler,
    hidden_pool,
    validate_particle,
)
from cg.api import search_begin, search_end, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

OPPONENTS = ("crustle", "alakazam", "lucario")


def read_deck(path: Path) -> list[int]:
    lines = [line for line in path.read_text().split("\n") if line.strip()]
    if len(lines) != 60:
        raise SystemExit(f"{path} has {len(lines)} cards; the engine requires exactly 60")
    return [int(line) for line in lines]


def random_action(select: dict) -> list[int]:
    count = min(
        random.randint(select["minCount"], select["maxCount"]), len(select["option"])
    )
    return random.sample(range(len(select["option"])), count)


class Tally:
    """Counters plus the first few concrete failures of each kind.

    A bare rate is not actionable — "97% of pools closed" does not say which
    zone leaks. Keeping a handful of examples per failure mode is what turns a
    red number into a fix.
    """

    def __init__(self, keep: int = 5):
        self.counts: collections.Counter = collections.Counter()
        self.examples: dict[str, list] = collections.defaultdict(list)
        self.keep = keep

    def hit(self, key: str, example=None) -> None:
        self.counts[key] += 1
        if example is not None and len(self.examples[key]) < self.keep:
            self.examples[key].append(example)

    def rate(self, numerator: str, denominator: str) -> float:
        total = self.counts[denominator]
        return self.counts[numerator] / total if total else float("nan")


def check_frame(obs: dict, decklists: list[list[int]], tally: Tally) -> None:
    """Ground-truth checks at one frame, for the player who owns it.

    Everything is tallied twice: once overall and once restricted to
    ``turn >= 1``. Setup (``turn == 0``) places Pokémon face-down on both sides,
    so it is a structurally different accounting problem — and it is not a phase
    a search ever runs at, since ``search_begin`` has no way to name your own
    face-down active. Mixing the two would let a setup-only gap hide a real
    mid-game one, or vice versa.
    """
    state = obs["current"]
    owner = state["yourIndex"]
    selection = obs.get("select") or {}
    live = state["turn"] >= 1

    def hit(key: str, example=None) -> None:
        tally.hit(key, example)
        if live:
            tally.hit(f"live_{key}")

    # (1) The owner's own-side accounting: deck + prizes, hand excluded.
    own = hidden_pool(state, owner, decklists[owner], include_hand=False,
                      selection=selection)
    hit("own_pool_checked")
    if own.closed:
        hit("own_pool_closed")
    else:
        hit("own_pool_mismatch", {
            "turn": state["turn"], "available": own.available,
            "required": own.required, "deckCount": state["players"][owner]["deckCount"],
            "face_down": own.face_down_in_play, "deck_revealed": own.deck_revealed,
        })

    # (2) The pool their *opponent* would compute for them, and whether the
    # real hand is drawable from it. This is the load-bearing check.
    theirs = hidden_pool(state, owner, decklists[owner], include_hand=True,
                         selection=selection)
    hit("opp_view_checked")
    if theirs.closed:
        hit("opp_view_closed")
    else:
        hit("opp_view_mismatch", {
            "turn": state["turn"], "available": theirs.available,
            "required": theirs.required, "face_down": theirs.face_down_in_play,
        })

    true_hand = state["players"][owner].get("hand")
    if true_hand is not None:
        hit("hand_checked")
        need = Counter(int(card["id"]) for card in true_hand)
        missing = need - theirs.pool
        if not missing:
            hit("hand_in_support")
        else:
            hit("hand_outside_support", {
                "turn": state["turn"], "missing": dict(list(missing.items())[:5]),
            })


def run_game(
    challenger_deck, opponent_deck, opponent_index, decklists, sampler, tally,
    particles: int, max_steps: int, do_search: bool,
) -> dict:
    obs, _ = battle_start(challenger_deck, opponent_deck)
    if obs is None:
        return {"started": False}
    sampler.reset()
    seat = 0  # challenger is player 0 for this harness
    resolved_at = None
    frames = 0
    try:
        while obs["current"]["result"] == -1 and frames < max_steps:
            select = obs.get("select")
            if select is None:
                break
            check_frame(obs, decklists, tally)

            if obs["current"]["yourIndex"] == seat:
                sampler.observe(obs, seat)
                tally.hit("sampler_frames")
                if sampler.posterior.resolved() and resolved_at is None:
                    resolved_at = frames
                if opponent_index not in sampler.posterior.alive:
                    tally.hit("true_decklist_eliminated", {"turn": obs["current"]["turn"]})

                for particle in sampler.sample(obs, seat, particles):
                    tally.hit("particles")
                    problems = validate_particle(
                        particle, obs, seat, decklists[seat],
                        sampler._candidates[particle.opponent_decklist_index],
                    )
                    if problems:
                        tally.hit("particle_invalid", {
                            "turn": obs["current"]["turn"], "problems": problems[:3],
                        })
                    else:
                        tally.hit("particle_valid")

                    if do_search and not problems:
                        try:
                            root = search_begin(
                                to_observation_class(obs),
                                **particle.search_begin_args(),
                                manual_coin=True,
                            )
                            tally.hit("search_accepted" if root else "search_null")
                        except Exception as error:  # noqa: BLE001 — the message is the finding
                            tally.hit("search_rejected", {
                                "turn": obs["current"]["turn"],
                                "error": f"{type(error).__name__}: {error}",
                            })

            obs = battle_select(random_action(select))
            frames += 1
    except (ValueError, IndexError, RuntimeError) as error:
        tally.hit("engine_error", f"{type(error).__name__}: {error}")
    finally:
        try:
            search_end()
        except Exception:  # noqa: BLE001
            pass
        battle_finish()
    return {"started": True, "frames": frames, "resolved_at": resolved_at,
            "alive_at_end": len(sampler.posterior.alive)}


def build_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--particles", type=int, default=4,
                        help="particles drawn per challenger decision")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--no-search", action="store_true",
                        help="skip the search_begin acceptance check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(_ROOT / "probe_belief.json"))
    return parser.parse_args(argv)


def main() -> None:
    args = build_args()
    random.seed(args.seed)

    challenger_deck = read_deck(_ROOT / "challenger" / "challenger_deck.csv")
    opponent_decks = [read_deck(_ROOT / f"{name}_frozen" / "deck.csv") for name in OPPONENTS]

    print("=" * 74)
    print("belief sampler probe")
    print("=" * 74)
    overlap = [
        (a, b, len(set(opponent_decks[i]) & set(opponent_decks[j])))
        for i, a in enumerate(OPPONENTS) for j, b in enumerate(OPPONENTS) if i < j
    ]
    print("candidate decklists:")
    for name, deck in zip(OPPONENTS, opponent_decks):
        print(f"  {name:<10} {len(set(deck))} distinct card ids")
    for a, b, shared in overlap:
        print(f"  shared({a},{b}) = {shared} distinct ids")
    print()

    tally = Tally()
    resolutions, alive_ends = [], []
    for game in range(args.games):
        opponent_index = game % len(OPPONENTS)
        sampler = BeliefSampler(
            challenger_deck, opponent_decks, rng=random.Random(args.seed * 1000 + game)
        )
        decklists = [challenger_deck, opponent_decks[opponent_index]]
        info = run_game(
            challenger_deck, opponent_decks[opponent_index], opponent_index,
            decklists, sampler, tally, args.particles, args.max_steps,
            not args.no_search,
        )
        if not info.get("started"):
            tally.hit("game_start_failed")
            continue
        tally.hit("games")
        if info["resolved_at"] is not None:
            resolutions.append(info["resolved_at"])
        alive_ends.append(info["alive_at_end"])

    counts = tally.counts
    print(f"games played           : {counts['games']}")
    print(f"frames checked         : {counts['own_pool_checked']}")
    print()
    for scope, prefix in (("all frames", ""), ("turn >= 1 (searchable)", "live_")):
        print(f"--- accounting / ground truth, {scope}")
        for label, hit_key, seen_key in (
            ("own pool closed     ", f"{prefix}own_pool_closed", f"{prefix}own_pool_checked"),
            ("opp-view pool closed", f"{prefix}opp_view_closed", f"{prefix}opp_view_checked"),
            ("true hand in support", f"{prefix}hand_in_support", f"{prefix}hand_checked"),
        ):
            print(f"  {label} : {counts[hit_key]}/{counts[seen_key]} "
                  f"({tally.rate(hit_key, seen_key):.4f})")
        print()
    print("--- particles")
    print(f"  drawn                : {counts['particles']}")
    print(f"  valid                : {counts['particle_valid']} "
          f"({tally.rate('particle_valid', 'particles'):.4f})")
    print(f"  search_begin accepted: {counts['search_accepted']} "
          f"(rejected {counts['search_rejected']}, null {counts['search_null']})")
    print()
    print("--- opponent decklist posterior")
    print(f"  true list eliminated : {counts['true_decklist_eliminated']} frames")
    print(f"  games resolved to 1  : {len(resolutions)}/{counts['games']}")
    if resolutions:
        print(f"  frames to resolve    : median {statistics.median(resolutions):.0f}, "
              f"min {min(resolutions)}, max {max(resolutions)}")
    if alive_ends:
        print(f"  candidates alive @end: mean {statistics.mean(alive_ends):.2f}")
    if counts["engine_error"]:
        print(f"\n  engine errors        : {counts['engine_error']}")

    failures = {k: v for k, v in tally.examples.items()
                if k in ("own_pool_mismatch", "opp_view_mismatch", "hand_outside_support",
                         "particle_invalid", "search_rejected", "engine_error")}
    if failures:
        print("\n--- failure examples")
        for key, items in failures.items():
            print(f"  {key} (x{counts[key]}):")
            for item in items:
                print(f"      {item}")

    Path(args.out).write_text(json.dumps(
        {"counts": dict(counts), "examples": {k: [str(i) for i in v]
                                              for k, v in tally.examples.items()},
         "resolutions": resolutions, "alive_at_end": alive_ends},
        indent=2, default=str,
    ))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
