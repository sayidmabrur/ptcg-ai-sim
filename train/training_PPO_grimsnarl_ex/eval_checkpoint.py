"""Score any BC checkpoint on a fixed held-out split, so two runs are comparable.

    conda activate kaggle-pokemon
    python eval_checkpoint.py --checkpoint challenger/bc_policy_xattn.pt \
        --parquet ../dataset/grimmsnarl_0724_0810.parquet

Why this is needed to judge the retrain
---------------------------------------
``bc_train.py`` reports ``val_exact`` on a 5% episode holdout of *whatever corpus it was
given*. The current checkpoint's 74.3% was measured on a holdout of the 841k-frame
corpus; a run on the 4.86M-frame corpus holds out different games. Comparing those two
numbers directly would confound capacity with a change of test set, and the two corpora
are not equally hard -- the new one spans 17 days and a metagame that shifted underneath
them. So the honest comparison evaluates every checkpoint on the *same* frames.

The split is reproduced rather than stored: ``bc_train`` derives it from the episode list
and ``--seed`` alone, so passing the same corpus, seed and holdout fraction here yields
the identical validation episodes.

Architecture comes from the checkpoint's ``.meta.json`` when present, since a state_dict
records no width or depth. Checkpoints predating that field are assumed to be the
2.68M-parameter default, which is what they are.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "challenger" / "policy_network"))

from bc_train import (  # noqa: E402
    ArchetypeDataset,
    Subset,
    collate,
    evaluate,
    format_categories,
)
from dataset import transform  # noqa: E402
from observation import LEARNER_SPEC  # noqa: E402
from policy_experimental import PolicyNetwork  # noqa: E402

DEFAULT_ARCH = {"dim": 64, "nhead": 2, "chain_layers": 8, "hist_layers": 8,
                "ctx_layers": 2, "cross_layers": 2, "ff_mult": 2, "dropout": 0.1}


def arch_for(checkpoint: Path, override: str | None) -> dict:
    if override:
        return DEFAULT_ARCH | json.loads(override)
    meta = Path(str(checkpoint) + ".meta.json")
    if meta.exists():
        recorded = json.loads(meta.read_text()).get("arch")
        if recorded:
            return DEFAULT_ARCH | recorded
        print(f"[eval] {meta.name} has no 'arch' field; assuming the 2.68M default")
    return dict(DEFAULT_ARCH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--holdout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=0,
                        help="0 evaluates the whole holdout")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cached-row-groups", type=int, default=24)
    parser.add_argument("--arch", default=None,
                        help="JSON overriding the recorded architecture")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    architecture = arch_for(args.checkpoint, args.arch)
    print(f"[eval] arch {architecture}")

    base = ArchetypeDataset(args.parquet, transform=transform, spec=LEARNER_SPEC,
                            cached_row_groups=args.cached_row_groups)
    episodes = base.episode_ids()
    # Reproduce bc_train's split exactly: same rng consumption order, same fraction.
    rng = random.Random(args.seed)
    unique = list(np.unique(episodes))
    rng.shuffle(unique)
    holdout = set(unique[: max(1, int(len(unique) * args.holdout))])
    val_positions = [i for i, e in enumerate(episodes) if e in holdout]
    print(f"[eval] {len(val_positions):,} val frames / {len(holdout):,} games "
          f"(of {len(unique):,})")

    model = PolicyNetwork(**architecture).to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"[eval] {sum(p.numel() for p in model.parameters()):,} parameters")

    loader = DataLoader(Subset(base, val_positions), batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate, num_workers=args.workers)
    started = time.time()
    val_loss, exact, same_play, categories = evaluate(
        model, loader, device, args.eval_batches or None)
    frames = (args.eval_batches * args.batch_size) if args.eval_batches else len(val_positions)
    print(f"\n{args.checkpoint.name}  on {Path(args.parquet).name} "
          f"(seed {args.seed}, holdout {args.holdout})")
    print(f"  val_loss       {val_loss:.4f}")
    print(f"  val_exact      {exact:.4%}   over ~{frames:,} frames "
          f"({time.time() - started:.0f}s)")
    print(f"  val_same_play  {same_play:.4%}   tie-aware; the metric's own ceiling "
          f"is ~83.8%")
    # Per-category, because the aggregate cannot distinguish "picks the wrong Pokemon to
    # attach to" from "attaches when the expert attacked" -- see ``evaluate``.
    print("\n  by the category the expert chose:")
    print(format_categories(categories))


if __name__ == "__main__":
    main()
