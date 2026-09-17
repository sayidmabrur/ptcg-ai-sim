"""Behavioural audit: measure the specific faults seen when watching it play.

    PY=/home/kangh/miniconda3/envs/kaggle-pokemon/bin/python
    $PY audit_behavior.py --games 40 --checkpoint checkpoints/ppo-exploit/best.pt

Win rate says the agent is at 40%. It does not say *why*, and watching it play
produced five concrete claims that a number cannot confirm or refute. This turns
each into a measurement:

1. **Empty selections on Hilda.** ``decode_action``'s own docstring records that
   the unfloored decode returned nothing on 29.0% of optional (``minCount == 0``)
   decisions, and names Hilda as the case. The PPO agent decodes through
   ``rollout_action`` with a *learned* stop logit and no floor, so the failure
   may have been reintroduced. Measured as ``empty_optional``.

2. **Rock Fighting Energy misattached.** It is the only {F} source in the deck
   and Cornerstone Mask Ogerpon ex's Demolish costs {F}{C}{C}. Attaching it to
   anything that is not Ogerpon wastes the card and the attacker with it.
   Measured as ``rock_fighting_targets``.

3. **Ogerpon unused against Alakazam.** "Cornerstone Stance" prevents *all*
   damage from attacks by Pokémon that have an Ability, which is a hard counter
   to an Alakazam deck. If the agent never leads it, the matchup is lost for a
   reason that has nothing to do with policy quality. Measured as
   ``ogerpon_active_turns``.

4. **Retreat under Latias.** "Skyliner" gives Basic Pokémon no retreat cost, so
   switching a damaged or energy-starved active out is free. Measured as
   ``retreats_with_latias``.

Everything here counts what the agent *did*, not what it should have done — the
"should" is the user's read of the game, and the point of the audit is to check
whether the behaviour matches the claim before building anything around it.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "crustle_frozen" / "policy_network"))

import torch  # noqa: E402

from cg.api import OptionType  # noqa: E402
from features import resolve_option_card  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

CARD = {
    "hilda": 1225,
    "latias": 184,
    "ogerpon": 117,
    "mist_energy": 11,
    "rock_fighting": 20,
    "switch": 1123,
}


def read_deck(path: Path) -> list[int]:
    return [int(l) for l in path.read_text().split("\n") if l.strip()]


def card_names():
    from cg.api import all_card_data

    return {c.cardId: c.name for c in all_card_data()}


def build_agent(checkpoint: Path, arch: str, deck):
    """The trained policy, decoding exactly as it was evaluated."""
    from collate import collate_features
    from dataset import transform
    from live import LiveFeatureExtractor
    from train import ActorCritic, rollout_action, selection_counts, tree_to

    device = torch.device("cpu")
    model = ActorCritic(dropout=0.0, arch=arch).to(device).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    extractor = LiveFeatureExtractor()

    class _Agent:
        deck = None

        def reset(self, episode):
            extractor.reset(episode_id=episode)

        @torch.no_grad()
        def act(self, obs):
            observation = extractor(obs)
            features = tree_to(collate_features([transform(observation)]), device)
            logits, stop_logit, _value, options_mask = model(features)
            min_count, max_count = selection_counts(features)
            action, _ = rollout_action(
                logits, stop_logit, options_mask, min_count, max_count, greedy=True
            )
            extractor.record_action(action)
            return action

    agent = _Agent()
    agent.deck = deck
    return agent


def _attach_target(state, seat, option) -> str:
    """Name the Pokémon an ATTACH option would put the card on."""
    area, index = option.get("inPlayArea"), option.get("inPlayIndex")
    player = state["players"][seat]
    zone = {4: "active", 5: "bench"}.get(area)
    if zone is None or index is None:
        return "?"
    slots = player.get(zone) or []
    if 0 <= index < len(slots) and slots[index]:
        return _NAMES.get(int(slots[index]["id"]), str(slots[index]["id"]))
    return "?"


_NAMES: dict = {}


def audit(agent, opponent, decks, seat, episode, max_steps, tally, names):
    obs, _ = battle_start(decks[0], decks[1])
    if obs is None:
        return None
    agent.reset(episode)
    opponent.reset(episode)
    steps = 0
    try:
        while obs["current"]["result"] == -1 and steps < max_steps:
            select = obs.get("select")
            if select is None:
                break
            actor = obs["current"]["yourIndex"]
            if actor == seat:
                state = obs["current"]
                mine = state["players"][seat]

                # (3) Is Ogerpon the active Pokémon right now?
                active = mine.get("active") or []
                if active and active[0] is not None:
                    tally["active_turns"] += 1
                    if int(active[0]["id"]) == CARD["ogerpon"]:
                        tally["ogerpon_active_turns"] += 1

                # (4) Latias in play means retreat is free for Basics.
                board = [p for p in ([a for a in active if a] + (mine.get("bench") or []))]
                latias_out = any(int(p["id"]) == CARD["latias"] for p in board)

                action = agent.act(obs)

                # (1) Optional decisions the agent declined outright.
                if select["minCount"] == 0:
                    tally["optional"] += 1
                    if not action:
                        tally["empty_optional"] += 1
                        effect = (select.get("effect") or {}).get("id")
                        tally_ctx = tally.setdefault("empty_by_effect", collections.Counter())
                        tally_ctx[names.get(effect, str(effect))] += 1

                # (2) Where does the single Rock Fighting Energy go?
                # (4) Retreats taken while Latias is out.
                for index in action:
                    if index >= len(select["option"]):
                        continue
                    option = select["option"][index]
                    if option.get("type") == OptionType.RETREAT:
                        tally["retreats"] += 1
                        if latias_out:
                            tally["retreats_with_latias"] += 1
                    # ATTACH/PLAY options name their card positionally, not by
                    # id (features.py: cardId is set on 32 of 22,555 slots), so
                    # the id has to be resolved from the zone index it points at.
                    card_id = option.get("cardId")
                    if card_id is None:
                        card_id = resolve_option_card(state, select, option, seat)
                    if option.get("type") == OptionType.ATTACH and card_id:
                        holder = _attach_target(state, seat, option)
                        tally.setdefault("attach_by_card", collections.Counter())[
                            f"{names.get(card_id, card_id)} -> {holder}"
                        ] += 1
            else:
                action = opponent.act(obs)
            obs = battle_select(action)
            steps += 1
        return obs["current"]["result"]
    except (ValueError, IndexError, RuntimeError) as error:
        tally["errors"] += 1
        return None
    finally:
        battle_finish()


def build_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default=str(_ROOT / "checkpoints/ppo-exploit/best.pt"))
    parser.add_argument("--arch", default="crustle")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--opponent", default="alakazam")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main() -> None:
    args = build_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    names = card_names()
    _NAMES.update(names)

    from train import FrozenPolicy

    deck = read_deck(_ROOT / "challenger" / "challenger_deck.csv")
    opponent = FrozenPolicy(args.opponent, _ROOT / f"{args.opponent}_frozen",
                            torch.device("cpu"))
    agent = build_agent(Path(args.checkpoint), args.arch, deck)

    tally: dict = collections.defaultdict(int)
    wins = played = 0
    for game in range(args.games):
        seat = game % 2
        decks = [None, None]
        decks[seat], decks[1 - seat] = deck, opponent.deck
        result = audit(agent, opponent, decks, seat, game, args.max_steps, tally, names)
        if result is None:
            continue
        played += 1
        wins += result == seat

    print("=" * 74)
    print(f"behavioural audit vs {args.opponent}  ({played} games)")
    print("=" * 74)
    print(f"win rate                     {wins}/{played} = "
          f"{wins / max(played, 1):.1%}")
    print()
    print(f"optional decisions (min=0)   {tally['optional']}")
    empty = tally["empty_optional"]
    print(f"  declined (empty action)    {empty} "
          f"({empty / max(tally['optional'], 1):.1%})   <- Hilda claim")
    for name, count in (tally.get("empty_by_effect") or collections.Counter()).most_common(8):
        print(f"      via {name}: {count}")
    print()
    print(f"own decision frames          {tally['active_turns']}")
    print(f"  Ogerpon was active         {tally['ogerpon_active_turns']} "
          f"({tally['ogerpon_active_turns'] / max(tally['active_turns'], 1):.1%})"
          f"   <- Ogerpon-counters-Alakazam claim")
    print()
    print(f"retreats taken               {tally['retreats']}")
    print(f"  with Latias in play        {tally['retreats_with_latias']}"
          f"   <- free-retreat claim")
    print()
    attaches = tally.get("attach_by_card") or collections.Counter()
    print(f"energy/tool attaches         {sum(attaches.values())}   <- misattach claim")
    for where, count in attaches.most_common(14):
        print(f"      {where}: {count}")
    if tally["errors"]:
        print(f"\nengine errors: {tally['errors']}")

    Path(_ROOT / "audit_behavior.json").write_text(
        json.dumps({k: (dict(v) if isinstance(v, collections.Counter) else v)
                    for k, v in tally.items()}, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
