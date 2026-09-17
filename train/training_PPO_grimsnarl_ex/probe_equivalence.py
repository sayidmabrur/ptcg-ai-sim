"""How much of the BC objective is fitting coin flips?

    conda activate kaggle-pokemon
    python probe_equivalence.py --parquet ../dataset/grimmsnarl_0724_0810.parquet \
        --checkpoint challenger/bc_policy_xattn.pt

``equivalence_mask`` in policy_experimental.py identifies every option that is the *same
play* as the one the expert clicked (same type, card, target, count -- differing only in
which hand slot the card sat in). Its docstring reports 50.5% of single-select decisions
carrying such a tie on the *crustle* corpus, and argues the index objective therefore
"caps exact-index accuracy at ~63.7%".

But ``bc_train.py`` never passes it. Both the loss and ``val_exact`` score against the one
index the expert happened to click, so a model that plays an identical alternative is
marked wrong. Two things follow, and this measures both on the Grimmsnarl corpus:

1. **How much of the reported error is illusory** -- predictions that differ from the
   expert's index but are behaviourally the same move.
2. **What the honest metric is** -- exact match under tie-awareness, i.e. the number to
   optimise if the loss is switched over.

The distinction matters more than a point of val_exact: capacity spent memorising which
duplicate slot a player clicked is capacity not spent on play strength, and it is the one
lever available that costs no extra data and no extra parameters.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "challenger" / "policy_network"))

from bc_train import ArchetypeDataset, Subset, collate, tree_to  # noqa: E402
from dataset import transform  # noqa: E402
from observation import LEARNER_SPEC  # noqa: E402
from policy_experimental import (  # noqa: E402
    PolicyNetwork,
    decode_action,
    equivalence_mask,
    selection_counts,
)

DEFAULT_ARCH = {"dim": 64, "nhead": 2, "chain_layers": 8, "hist_layers": 8,
                "ctx_layers": 2, "cross_layers": 2, "ff_mult": 2, "dropout": 0.1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="omit to measure only the tie structure of the data")
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    base = ArchetypeDataset(args.parquet, transform=transform, spec=LEARNER_SPEC,
                            cached_row_groups=12)
    total = len(base)
    stride = max(1, total // (args.batches * args.batch_size))
    positions = list(range(0, total, stride))[: args.batches * args.batch_size]
    print(f"[eq] sampling {len(positions):,} of {total:,} frames (stride {stride})")
    loader = DataLoader(Subset(base, positions), batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=args.workers)

    model = None
    if args.checkpoint:
        meta = Path(str(args.checkpoint) + ".meta.json")
        arch = DEFAULT_ARCH | (json.loads(meta.read_text()).get("arch") or {}
                               if meta.exists() else {})
        model = PolicyNetwork(**arch).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu",
                                         weights_only=True))
        model.eval()
        print(f"[eq] {args.checkpoint.name}: {sum(p.numel() for p in model.parameters()):,} "
              f"parameters, arch dim={arch['dim']}")

    tie_widths: Counter = Counter()
    single = ties = 0
    strict_hit = tie_hit = 0
    multi = multi_hit = 0
    scored = 0

    with torch.no_grad():
        for features, targets, options_mask in loader:
            features = tree_to(features, device)
            targets, options_mask = targets.to(device), options_mask.to(device)
            counts = targets.sum(-1)
            is_single = counts == 1

            # --- tie structure of the data, model-independent ---
            if is_single.any():
                equiv = equivalence_mask(features["decision_context"]["options"],
                                         targets, options_mask)
                width = equiv.sum(-1)[is_single]
                for value in width.tolist():
                    tie_widths[int(value)] += 1
                single += int(is_single.sum())
                ties += int((width > 1).sum())

            if model is None:
                continue

            # --- what the model actually gets right, both ways ---
            logits = model(features)
            min_count, max_count = selection_counts(features)
            decoded = decode_action(logits, options_mask, min_count.clamp(min=1), max_count)
            equiv_all = equivalence_mask(features["decision_context"]["options"],
                                         targets, options_mask)
            for row, picks in enumerate(decoded):
                expert = set(torch.nonzero(targets[row]).flatten().tolist())
                predicted = set(picks)
                scored += 1
                if bool(is_single[row]):
                    hit = predicted == expert
                    strict_hit += hit
                    # tie-aware: one pick, and that pick is the same play
                    lenient = len(picks) == 1 and bool(equiv_all[row, picks[0]])
                    tie_hit += lenient
                else:
                    multi += 1
                    multi_hit += predicted == expert

    print("\n" + "=" * 72)
    print("TIE STRUCTURE  (single-select decisions)")
    print("=" * 72)
    print(f"single-select frames sampled: {single:,}")
    if single:
        print(f"with >=1 identical alternative: {ties:,} ({ties/single:.1%})")
        print("\ntie-group width -> frames:")
        for width in sorted(tie_widths):
            share = tie_widths[width] / single
            print(f"  {width:>2} identical option(s): {tie_widths[width]:>7,} ({share:>6.1%})")
        expected = sum(count / width for width, count in tie_widths.items() if width)
        print(f"\nCeiling on exact-INDEX accuracy if the policy knew the right *play* "
              f"perfectly\nand broke ties uniformly at random: {expected/single:.1%}")

    if model is not None and scored:
        print("\n" + "=" * 72)
        print(f"CHECKPOINT ACCURACY  ({scored:,} frames)")
        print("=" * 72)
        single_scored = scored - multi
        print(f"single-select frames: {single_scored:,}")
        print(f"  exact index match      {strict_hit/max(single_scored,1):.2%}   "
              f"(what val_exact reports)")
        print(f"  same-play match        {tie_hit/max(single_scored,1):.2%}   "
              f"(tie-aware)")
        print(f"  illusory errors        "
              f"{(tie_hit-strict_hit)/max(single_scored,1):.2%} of frames were marked wrong "
              f"for clicking an identical option")
        if multi:
            print(f"multi-select frames:  {multi:,}")
            print(f"  exact set match        {multi_hit/multi:.2%}")
        overall_strict = (strict_hit + multi_hit) / scored
        overall_lenient = (tie_hit + multi_hit) / scored
        print(f"\noverall exact match     {overall_strict:.2%}")
        print(f"overall same-play match {overall_lenient:.2%}  "
              f"(+{(overall_lenient-overall_strict)*100:.2f} points)")


if __name__ == "__main__":
    main()
