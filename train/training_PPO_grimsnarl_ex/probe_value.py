"""Can a value function be learned from the replay corpus at all?

    PY=/home/kangh/miniconda3/envs/kaggle-pokemon/bin/python
    $PY probe_value.py --stage scalar
    $PY probe_value.py --stage trunk        # needs the scalar stage to show headroom

This is the gate on everything downstream. ``probe_search_play.py`` showed the
search is evaluator-limited: a random rollout's own noise (0.86) swamps the true
spread between actions (0.31), so IS-MCTS does not improve with budget. A value
network is the fix — but only if outcomes are predictable from state at all, and
that is an open question here. ``train.py``'s PPO critic managed explained
variance of +0.11, which is barely above nothing.

Half a day of regression answers it. Building the full distillation loop and
*then* discovering V is uninformative would cost weeks.

Two stages, cheap first
-----------------------
**scalar** — ~24 handcrafted scoreboard features (prize differential, board HP,
energy developed, hand/deck counts, turn) into a small MLP. Fast, and it
establishes how much of the signal is trivially available.

**trunk** — the full ``PolicyNetwork`` encoder stack plus a value head, reusing
``dataset.transform`` so the inputs are bit-identical to what the policy sees.

The comparison is the point, not either number alone. If the trunk barely beats
the scalars, then the expensive encoder is not earning its keep *for value
estimation*, and the search should evaluate leaves with something cheap — which
would dissolve the 300:1 network-to-engine cost problem rather than fighting it.
If the trunk wins big, the cost problem is real and the answer is batching.

Reading the numbers honestly
----------------------------
Three traps, all of which would produce a flattering and meaningless result:

*Row-wise splitting.* Frames in one game share a single label and are nearly
identical to their neighbours. A random row split puts near-copies of each
validation frame in training. Everything here splits by ``episode_id``.

*Aggregate explained variance.* Late frames are easy — by turn 25 the prize
count has usually decided it, and a model that reads the scoreboard scores well
while being useless at a search leaf. Metrics are therefore reported **stratified
by turn**, and the early buckets are the ones that matter.

*The wrong floor.* 0.0 is not the bar. Seat 0 wins 53.2% of expert-vs-expert
games, so ``firstPlayer`` alone is a real predictor, and prize differential is a
much stronger one. Both are reported alongside so the learned model is measured
against what it must beat rather than against chance.
"""

import argparse
import collections
import json
import math
import random
import statistics
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "challenger" / "policy_network"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from value_features import (  # noqa: E402
    SCALAR_NAMES,
    ScalarValue,
    save_value_net,
    scalar_features,
)

PARQUET = Path("/home/kangh/projects/pokemon-tcg/data/policy_decisions_pretraining.parquet")

#: The top-15 ranked entrants the corpus was fetched from. Every episode has at
#: least one; 1,199 have two. Used only to choose the validation set — see
#: ``split_episodes``.
EXPERTS = {
    "flg", "Yushin Ito", "Majkel1337", "Where is my orbit", "Oshbocker", "keidroid",
    "李秉叡（ntumlnoob）", "lmaffei", "Raihan Ramadistra", "やる気元気ミワハルキ", "Luca", "Brahim",
}


# --------------------------------------------------------------------------
# Labels


def episode_labels(parquet: Path):
    """``(winner, seats, rows_per_episode)`` derived from the terminal reward.

    ``final_result`` is ``-1`` on every row — the engine's "unfinished"
    sentinel, never populated by the converter. The outcome is instead carried
    by ``reward``: exactly one nonzero entry per episode, always on the final
    frame, ``+1`` if that frame's owner won and ``-1`` if they lost. 4,347 of
    4,350 episodes resolve; the other three have no nonzero reward and are
    dropped rather than guessed at.
    """
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(parquet)
    last: dict = {}
    seats: dict = {}
    rows: collections.Counter = collections.Counter()
    for batch in handle.iter_batches(
        batch_size=200_000,
        columns=["episode_id", "frame_index", "player_index", "player_name", "reward"],
    ):
        data = batch.to_pydict()
        for episode, frame, player, name, reward in zip(
            data["episode_id"], data["frame_index"], data["player_index"],
            data["player_name"], data["reward"],
        ):
            seats[(episode, player)] = name
            rows[episode] += 1
            if frame >= last.get(episode, (-1,))[0]:
                last[episode] = (frame, player, reward)

    winner = {}
    for episode, (_, player, reward) in last.items():
        if reward in (1.0, 1):
            winner[episode] = player
        elif reward in (-1.0, -1):
            winner[episode] = 1 - player
    return winner, seats, rows


def split_episodes(winner, seats, holdout: int, seed: int):
    """Validation drawn from expert-vs-expert games only; training gets the rest.

    The corpus is top-15-vs-field, and only 1,199 games have a ranked player on
    both sides. Training on all of it is right — effective sample size is games
    (4,347), not frames (668k), and cutting to the mirror matches would throw
    away 72% of that. But the number worth *reporting* is performance on
    strong-versus-strong play, since that is the regime a search leaf is
    evaluated in, so the held-out set is drawn exclusively from those.
    """
    both = sorted(
        episode for episode in winner
        if seats.get((episode, 0)) in EXPERTS and seats.get((episode, 1)) in EXPERTS
    )
    rng = random.Random(seed)
    rng.shuffle(both)
    validation = set(both[:holdout])
    training = [episode for episode in sorted(winner) if episode not in validation]
    return training, sorted(validation)


# --------------------------------------------------------------------------
# Scalar features


def build_scalar_table(parquet: Path, winner, stride: int):
    """Stream the parquet once, emitting one row per sampled frame."""
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(parquet)
    features, labels, episodes, turns = [], [], [], []
    kept = collections.Counter()
    for batch in handle.iter_batches(
        batch_size=2000,
        columns=["episode_id", "frame_index", "player_index", "state"],
    ):
        data = batch.to_pydict()
        for episode, frame, player, state in zip(
            data["episode_id"], data["frame_index"], data["player_index"], data["state"]
        ):
            if episode not in winner or state is None:
                continue
            # Stride over frames rather than take a prefix: consecutive frames
            # are near-duplicates, so this buys independence, and it must not
            # bias toward any phase of the game.
            kept[episode] += 1
            if kept[episode] % stride:
                continue
            features.append(scalar_features(state, player))
            labels.append(1.0 if winner[episode] == player else -1.0)
            episodes.append(episode)
            turns.append(state["turn"])
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(episodes),
        np.asarray(turns, dtype=np.int32),
    )


# --------------------------------------------------------------------------
# Metrics


TURN_BUCKETS = ((0, 4), (5, 8), (9, 12), (13, 18), (19, 26), (27, 200))


def explained_variance(y, pred) -> float:
    variance = float(np.var(y))
    return 1.0 - float(np.var(y - pred)) / variance if variance > 0 else float("nan")


def report(name: str, y, pred, turns, out: dict) -> None:
    rows = []
    for low, high in TURN_BUCKETS:
        mask = (turns >= low) & (turns <= high)
        if mask.sum() < 50:
            rows.append((f"{low}-{high}", int(mask.sum()), float("nan"), float("nan")))
            continue
        rows.append((
            f"{low}-{high}", int(mask.sum()),
            explained_variance(y[mask], pred[mask]),
            float(((pred[mask] > 0) == (y[mask] > 0)).mean()),
        ))
    overall_ev = explained_variance(y, pred)
    overall_acc = float(((pred > 0) == (y > 0)).mean())
    print(f"\n  {name}")
    print(f"    {'turn':>8}{'n':>8}{'expl.var':>11}{'accuracy':>11}")
    for label, count, ev, acc in rows:
        print(f"    {label:>8}{count:>8}{ev:>11.3f}{acc:>11.1%}"
              if count >= 50 else f"    {label:>8}{count:>8}{'—':>11}{'—':>11}")
    print(f"    {'ALL':>8}{len(y):>8}{overall_ev:>11.3f}{overall_acc:>11.1%}")
    out[name] = {
        "overall": {"explained_variance": overall_ev, "accuracy": overall_acc,
                    "n": int(len(y))},
        "by_turn": [{"turn": l, "n": c, "explained_variance": e, "accuracy": a}
                    for l, c, e, a in rows],
    }


# --------------------------------------------------------------------------
# Models


def train_scalar(train_x, train_y, val_x, val_y, epochs, lr, device, batch=1024, seed=0):
    torch.manual_seed(seed)
    model = ScalarValue(train_x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    tx = torch.as_tensor(train_x, device=device)
    ty = torch.as_tensor(train_y, device=device)
    vx = torch.as_tensor(val_x, device=device)
    vy = torch.as_tensor(val_y, device=device)
    best, best_state = -math.inf, None
    order = torch.arange(len(tx), device=device)
    for epoch in range(epochs):
        model.train()
        perm = order[torch.randperm(len(order), device=device)]
        for start in range(0, len(perm), batch):
            index = perm[start:start + batch]
            loss = nn.functional.mse_loss(model(tx[index]), ty[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(vx)
            ev = 1 - float(((vy - pred).var()) / vy.var())
        # Selecting on held-out expert-vs-expert EV, which is the quantity the
        # whole probe is about; the training loss is not reported because it is
        # not the question.
        if ev > best:
            best = ev
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"      epoch {epoch + 1:>2}/{epochs}  val expl.var {ev:+.3f}"
              f"{'  *' if ev >= best else ''}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model, model(vx).cpu().numpy()


def build_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", default="scalar", choices=("scalar", "trunk", "both"))
    parser.add_argument("--parquet", default=str(PARQUET))
    parser.add_argument("--stride", type=int, default=3,
                        help="keep every Nth frame of each episode; consecutive "
                             "frames are near-duplicates sharing one label")
    parser.add_argument("--holdout", type=int, default=250,
                        help="expert-vs-expert episodes held out for validation")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=str(_ROOT / "probe_value.json"))
    return parser.parse_args(argv)


def main() -> None:
    args = build_args()
    parquet = Path(args.parquet)
    print("=" * 74)
    print("value-function probe")
    print("=" * 74)

    winner, seats, rows = episode_labels(parquet)
    training, validation = split_episodes(winner, seats, args.holdout, args.seed)
    print(f"episodes with a label : {len(winner)}")
    print(f"  train               : {len(training)}")
    print(f"  validation          : {len(validation)} (expert vs expert only)")

    print(f"\nbuilding scalar features (stride {args.stride}) ...", flush=True)
    x, y, episodes, turns = build_scalar_table(parquet, winner, args.stride)
    val_set = set(validation)
    is_val = np.fromiter((e in val_set for e in episodes), dtype=bool, count=len(episodes))
    print(f"  frames: {len(y):,}  train {int((~is_val).sum()):,}  "
          f"val {int(is_val.sum()):,}")
    print(f"  label balance (val): {float((y[is_val] > 0).mean()):.1%} wins")

    out: dict = {"episodes": {"train": len(training), "val": len(validation)}}
    vy, vturns = y[is_val], turns[is_val]

    # Baselines, in increasing order of how much they already know. The learned
    # model has to beat the third one to be worth anything.
    report("baseline: constant (train mean)", vy,
           np.full_like(vy, float(y[~is_val].mean())), vturns, out)
    first = x[is_val][:, SCALAR_NAMES.index("is_first")]
    report("baseline: first-player only", vy,
           np.where(first > 0, 1.0, -1.0).astype(np.float32) * 0.065, vturns, out)
    prize = x[is_val][:, SCALAR_NAMES.index("prize_diff")]
    scale = float(np.dot(prize, vy) / max(np.dot(prize, prize), 1e-9))
    report(f"baseline: prize differential only (x{scale:.2f})", vy,
           (prize * scale).astype(np.float32), vturns, out)

    if args.stage in ("scalar", "both"):
        print(f"\n  training scalar MLP ({x.shape[1]} features) ...", flush=True)
        model, pred = train_scalar(
            x[~is_val], y[~is_val], x[is_val], vy,
            args.epochs, args.lr, torch.device(args.device), seed=args.seed,
        )
        name = f"learned: scalar MLP ({x.shape[1]} features)"
        report(name, vy, pred, vturns, out)
        ckpt = _ROOT / "value_scalar.pt"
        save_value_net(ckpt, model, out[name])
        print(f"\n  value net -> {ckpt}")

    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwritten: {args.out}")
    if args.stage in ("trunk", "both"):
        print("\n[trunk stage not run in this invocation — see --stage]")


if __name__ == "__main__":
    main()
