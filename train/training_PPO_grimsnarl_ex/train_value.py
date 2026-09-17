"""Train the value function on on-distribution self-play, with a sweep.

    PY=/home/kangh/miniconda3/envs/kaggle-pokemon/bin/python
    $PY train_value.py --sweep

Supersedes the ``--stage scalar`` path in ``probe_value.py``, which fitted the
replay corpus only. Three changes, each measured rather than assumed:

**On-distribution data.** The replay net reached explained variance 0.294 — but
on other people's decks, best Jaccard overlap 0.16 with the challenger list.
``gen_value_data.py`` produces the matchups actually played, including
deliberately weak play so V is defined where a search probes.

**A prize-margin auxiliary.** The outcome label is one bit per game, and the
frames of a game are near-duplicates, so effective supervision is ~1 bit per
game however many rows are collected. The final prize margin grades the same
game continuously and is the game's own scoreboard, not a heuristic. It is a
second head, not a modified label: the leaf still scores win probability.

**A game-level split.** Frames of a game share a label; a row-wise split leaks
it and reports a flattering number that means nothing.

Both corpora are reported separately. The self-play holdout is the number that
matters — it is the distribution the search runs in — while the replay holdout
says whether the model still generalises to decks it was not fitted on, which
is a proxy for how badly it will break in competition.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from value_features import (  # noqa: E402
    SCALAR_NAMES,
    ScalarValueMT,
    save_value_net,
)

TURN_BUCKETS = ((1, 4), (5, 8), (9, 12), (13, 18), (19, 26), (27, 200))


def explained_variance(y, pred) -> float:
    variance = float(np.var(y))
    return 1.0 - float(np.var(y - pred)) / variance if variance > 0 else float("nan")


def split_by_game(games: np.ndarray, holdout: float, seed: int):
    unique = np.unique(games)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    cut = max(1, int(len(unique) * holdout))
    validation = set(unique[:cut].tolist())
    mask = np.fromiter((g in validation for g in games), dtype=bool, count=len(games))
    return ~mask, mask, len(unique) - cut, cut


def load_selfplay(path: Path):
    data = np.load(path, allow_pickle=True)
    return {
        "x": data["rows"].astype(np.float32),
        "y": data["labels"].astype(np.float32),
        "margin": data["margins"].astype(np.float32),
        "turn": data["turns"].astype(np.int32),
        "game": data["games"],
        "source": data["sources"],
    }


def train_once(train, val, hp, device, epochs, seed, verbose=False):
    torch.manual_seed(seed)
    model = ScalarValueMT(
        len(SCALAR_NAMES), hidden=hp["hidden"], layers=hp["layers"],
        dropout=hp["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"],
                                  weight_decay=hp["weight_decay"])
    tx = torch.as_tensor(train["x"], device=device)
    ty = torch.as_tensor(train["y"], device=device)
    tm = torch.as_tensor(train["margin"], device=device)
    vx = torch.as_tensor(val["x"], device=device)
    vy = torch.as_tensor(val["y"], device=device)
    batch = hp["batch"]
    steps = max(1, len(tx) // batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=hp["lr"], total_steps=epochs * steps, pct_start=0.2
    )
    best, best_state, patience = -math.inf, None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(tx), device=device)
        for start in range(0, steps * batch, batch):
            index = perm[start:start + batch]
            value, margin = model.both(tx[index])
            loss = (
                nn.functional.mse_loss(value, ty[index])
                + hp["margin_weight"] * nn.functional.mse_loss(margin, tm[index])
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
        model.eval()
        with torch.no_grad():
            pred = model(vx)
            ev = 1 - float((vy - pred).var() / vy.var())
        if ev > best:
            best, patience = ev, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if verbose:
            print(f"      epoch {epoch + 1:>3}/{epochs}  val EV {ev:+.4f}"
                  f"{'  *' if ev >= best else ''}", flush=True)
        # Early stop on the metric being selected on, not on training loss:
        # with ~1 bit of real signal per game this overfits long before the
        # training loss stops falling.
        if patience >= hp.get("patience", 12):
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, best


def report(name: str, model, x, y, turns, device, out: dict) -> None:
    with torch.no_grad():
        pred = model(torch.as_tensor(x, device=device)).cpu().numpy()
    rows = []
    for low, high in TURN_BUCKETS:
        mask = (turns >= low) & (turns <= high)
        if mask.sum() < 50:
            rows.append((f"{low}-{high}", int(mask.sum()), float("nan"), float("nan")))
            continue
        rows.append((f"{low}-{high}", int(mask.sum()),
                     explained_variance(y[mask], pred[mask]),
                     float(((pred[mask] > 0) == (y[mask] > 0)).mean())))
    overall = explained_variance(y, pred)
    accuracy = float(((pred > 0) == (y > 0)).mean())
    print(f"\n  {name}")
    print(f"    {'turn':>8}{'n':>9}{'expl.var':>11}{'accuracy':>11}")
    for label, count, ev, acc in rows:
        if count >= 50:
            print(f"    {label:>8}{count:>9}{ev:>11.3f}{acc:>11.1%}")
        else:
            print(f"    {label:>8}{count:>9}{'—':>11}{'—':>11}")
    print(f"    {'ALL':>8}{len(y):>9}{overall:>11.3f}{accuracy:>11.1%}")
    out[name] = {
        "overall": {"explained_variance": overall, "accuracy": accuracy, "n": int(len(y))},
        "by_turn": [{"turn": l, "n": c, "explained_variance": e, "accuracy": a}
                    for l, c, e, a in rows],
    }


DEFAULT_HP = {
    "hidden": 256, "layers": 3, "dropout": 0.1, "lr": 2e-3,
    "weight_decay": 1e-4, "batch": 2048, "margin_weight": 0.5, "patience": 12,
}

SWEEP = {
    "hidden": [128, 256, 512],
    "layers": [2, 3],
    "dropout": [0.05, 0.15],
    "lr": [1e-3, 3e-3],
    "margin_weight": [0.0, 0.5, 1.0],
}


def build_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default=str(_ROOT / "value_data_selfplay.npz"))
    parser.add_argument("--holdout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=str(_ROOT / "value_mt.pt"))
    return parser.parse_args(argv)


def main() -> None:
    args = build_args()
    device = torch.device(args.device)
    data = load_selfplay(Path(args.data))
    train_mask, val_mask, n_train, n_val = split_by_game(
        data["game"], args.holdout, args.seed
    )
    train = {k: v[train_mask] for k, v in data.items() if k != "game"}
    val = {k: v[val_mask] for k, v in data.items() if k != "game"}

    print("=" * 74)
    print("value function — on-distribution self-play")
    print("=" * 74)
    print(f"rows   {len(data['y']):,}   train {int(train_mask.sum()):,}  "
          f"val {int(val_mask.sum()):,}")
    print(f"games  {n_train + n_val:,}   train {n_train:,}  val {n_val:,}")
    print(f"label balance {float((data['y'] > 0).mean()):.1%} positive   "
          f"mean |margin| {float(np.abs(data['margin']).mean()):.3f}")

    hp = dict(DEFAULT_HP)
    results: dict = {}
    if args.sweep:
        keys = list(SWEEP)
        combos = list(itertools.product(*(SWEEP[k] for k in keys)))
        print(f"\nsweeping {len(combos)} configurations "
              f"({args.sweep_epochs} epochs each)")
        scored = []
        started = time.perf_counter()
        for i, combo in enumerate(combos):
            candidate = dict(DEFAULT_HP)
            candidate.update(dict(zip(keys, combo)))
            _, ev = train_once(train, val, candidate, device,
                               args.sweep_epochs, args.seed)
            scored.append((ev, candidate))
            print(f"  [{i + 1:>2}/{len(combos)}] EV {ev:+.4f}  "
                  + "  ".join(f"{k}={candidate[k]}" for k in keys), flush=True)
        scored.sort(key=lambda item: -item[0])
        hp = scored[0][1]
        print(f"\nbest sweep EV {scored[0][0]:+.4f} after "
              f"{time.perf_counter() - started:.0f}s")
        print("  " + "  ".join(f"{k}={hp[k]}" for k in keys))
        results["sweep"] = [{"ev": ev, **{k: c[k] for k in keys}} for ev, c in scored]

    print(f"\ntraining final model ({args.epochs} epochs)")
    model, best_ev = train_once(train, val, hp, device, args.epochs, args.seed)
    report("held-out self-play (our matchups)", model, val["x"], val["y"],
           val["turn"], device, results)

    # Our three real matchups alone — sources 0-2 in the mixture. This is the
    # slice the search is actually evaluated on; the rest is coverage.
    ours = val["source"] <= 2
    if ours.sum() > 200:
        report("held-out, our matchups only", model, val["x"][ours], val["y"][ours],
               val["turn"][ours], device, results)

    baseline = val["x"][:, SCALAR_NAMES.index("prize_diff")]
    scale = float(np.dot(baseline, val["y"]) / max(np.dot(baseline, baseline), 1e-9))
    print(f"\n  baseline: prize differential only (x{scale:.2f})")
    print(f"    {'ALL':>8}{len(val['y']):>9}"
          f"{explained_variance(val['y'], baseline * scale):>11.3f}"
          f"{float(((baseline > 0) == (val['y'] > 0)).mean()):>11.1%}")

    results["hyperparameters"] = hp
    save_value_net(Path(args.out), model, results)
    Path(args.out).with_suffix(".json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nsaved {args.out}  (val EV {best_ev:+.4f})")


if __name__ == "__main__":
    main()
