"""Does the checkpoint pick the right Pokemon when several share a card id?

    conda activate kaggle-pokemon
    python probe_targeting.py --checkpoint challenger/bc_policy_xattn_large.pt \
        --parquet ../dataset/grimmsnarl_0724_0810.parquet

Background. ``probe_munkidori.py`` measured the ladder pilots spreading energy across
their Munkidori correctly 99.4% of the time, so the corpus does not teach the mistake.
``probe_option_blindness.py`` then showed that options aimed at different board Pokemon
differ only in pointer fields -- ``inPlayArea`` and ``inPlayIndex`` -- while
``PlayerStateEncoder`` mean+sum pools the bench, so no per-slot state survives to be
attended to. The prediction is that the network cannot condition on the target's own
energy and must fall back on a positional prior.

This tests that prediction directly. It finds decisions where one action could be aimed
at two or more Pokemon *of the same card* holding **different** amounts of energy, runs
the checkpoint, and compares its target against the expert's on the same frames.

The headline number is ``stacked`` -- how often a chooser attaches to a Pokemon that
already has energy while an identical, empty one is available. For Munkidori that is
strictly worse: the ability needs one energy and fires once per copy, so a second energy
on the same copy buys nothing and costs an activation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "challenger" / "policy_network"))

from bc_train import ArchetypeDataset, Subset, collate, tree_to  # noqa: E402
from dataset import transform  # noqa: E402
from observation import LEARNER_SPEC  # noqa: E402
from policy_experimental import PolicyNetwork, decode_action, selection_counts  # noqa: E402

DEFAULT_ARCH = {"dim": 64, "nhead": 2, "chain_layers": 8, "hist_layers": 8,
                "ctx_layers": 2, "cross_layers": 2, "ff_mult": 2, "dropout": 0.1}
#: ``inPlayArea`` values seen in the corpus: 4 addresses the active slot, 5 the bench.
ACTIVE_AREA, BENCH_AREA = 4, 5


def board_pokemon(state: dict, player_index: int):
    board = state["players"][player_index]
    active = [m for m in (board["active"] or []) if m is not None]
    bench = [m for m in (board["bench"] or []) if m is not None]
    return active, bench


def resolve(option: dict, active: list, bench: list):
    """The Pokemon an option is aimed at, or None if it does not name one."""
    area, index = option["inPlayArea"], option["inPlayIndex"]
    if area == ACTIVE_AREA and active:
        return active[0] if (index or 0) == 0 else None
    if area == BENCH_AREA and index is not None and index < len(bench):
        return bench[index]
    return None


def find_frames(parquet_path: str, card_id: int | None, every: int, limit: int):
    """Raw row indexes of decisions that aim one action at several same-card Pokemon
    holding different amounts of energy, with the option group and each target's energy."""
    parquet = pq.ParquetFile(parquet_path)
    starts, offset = [], 0
    for g in range(parquet.metadata.num_row_groups):
        starts.append(offset)
        offset += parquet.metadata.row_group(g).num_rows

    found = []
    for g in range(0, parquet.metadata.num_row_groups, every):
        table = parquet.read_row_group(g, columns=["state", "player_index", "options",
                                                   "target_action", "is_target_archetype"])
        for local, row in enumerate(table.to_pylist()):
            if not row["is_target_archetype"]:
                continue
            active, bench = board_pokemon(row["state"], row["player_index"])
            options = row["options"] or []
            # One "action" = same verb and same card; the variants differ only in target.
            buckets = defaultdict(list)
            for idx, o in enumerate(options):
                if o["inPlayArea"] not in (ACTIVE_AREA, BENCH_AREA):
                    continue
                buckets[(o["type"], o["area"], o["index"], o["cardId"])].append((idx, o))
            for entries in buckets.values():
                targets = {}
                for idx, o in entries:
                    mon = resolve(o, active, bench)
                    if mon is None:
                        continue
                    if card_id is not None and mon["id"] != card_id:
                        continue
                    targets[idx] = (mon["id"], len(mon["energies"] or []))
                if len(targets) < 2:
                    continue
                by_card = defaultdict(set)
                for cid, energy in targets.values():
                    by_card[cid].add(energy)
                # only interesting when identical cards hold different amounts
                if not any(len(v) > 1 for v in by_card.values()):
                    continue
                found.append({
                    "raw": starts[g] + local,
                    "targets": targets,
                    "expert": list(row["target_action"] or []),
                })
                break
            if len(found) >= limit:
                return found
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--card-id", type=int, default=112, help="112 = Munkidori; omit with -1 for any")
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    card_id = None if args.card_id < 0 else args.card_id
    print(f"[targeting] scanning for decisions aiming at several "
          f"{'card ' + str(card_id) if card_id else 'same-card'} Pokemon...")
    frames = find_frames(args.parquet, card_id, args.every, args.limit)
    print(f"[targeting] {len(frames):,} qualifying decisions")
    if not frames:
        return

    meta = Path(str(args.checkpoint) + ".meta.json")
    arch = DEFAULT_ARCH | ((json.loads(meta.read_text()).get("arch") or {}) if meta.exists() else {})
    device = torch.device(args.device)
    model = PolicyNetwork(**arch).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    print(f"[targeting] {args.checkpoint.name}: dim={arch['dim']}, "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")

    base = ArchetypeDataset(args.parquet, transform=transform, spec=LEARNER_SPEC,
                            cached_row_groups=16)
    raw_to_sample = {raw: i for i, raw in enumerate(base._row_indexes)}
    usable = [f for f in frames if f["raw"] in raw_to_sample]
    print(f"[targeting] {len(usable):,} of them are trainable samples")

    positions = [raw_to_sample[f["raw"]] for f in usable]
    loader = DataLoader(Subset(base, positions), batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=2)

    picks: list[list[int]] = []
    with torch.no_grad():
        for features, targets, options_mask in loader:
            features = tree_to(features, device)
            options_mask = options_mask.to(device)
            logits = model(features)
            lo, hi = selection_counts(features)
            picks.extend(decode_action(logits, options_mask, lo.clamp(min=1), hi))

    stats = {"model": Counter(), "expert": Counter()}
    for frame, chosen in zip(usable, picks):
        targets = frame["targets"]
        available = [e for _, e in targets.values()]
        empty_available = min(available) == 0
        for who, selection in (("expert", frame["expert"]), ("model", chosen)):
            aimed = [targets[i][1] for i in selection if i in targets]
            if not aimed:
                stats[who]["off_target"] += 1
                continue
            stats[who]["decisions"] += 1
            energy = aimed[0]
            stats[who]["attached_energy_total"] += energy
            if empty_available and energy >= 1:
                stats[who]["stacked"] += 1

    print("\n" + "=" * 72)
    print("WHERE THE ENERGY WENT  (an empty identical copy was available)")
    print("=" * 72)
    for who in ("expert", "model"):
        s = stats[who]
        n = s["decisions"]
        if not n:
            continue
        print(f"  {who:<7} decisions {n:>6}   "
              f"stacked onto an already-energised copy {s['stacked']:>5} "
              f"({s['stacked']/n:>6.2%})   "
              f"mean energy already on the chosen target {s['attached_energy_total']/n:.2f}")
        if s["off_target"]:
            print(f"          ({s['off_target']} picks landed outside the target group)")


if __name__ == "__main__":
    main()
