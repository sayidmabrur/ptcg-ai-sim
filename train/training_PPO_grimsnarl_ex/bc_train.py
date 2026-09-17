"""Behavioural cloning the Marnie's Grimmsnarl ex pilot ('flg') from that pilot's replays.

    conda activate kaggle-pokemon
    python build_corpus.py --replays dataset --agent flg --out grimmsnarl_flg.parquet
    python bc_train.py --parquet grimmsnarl_flg.parquet

Three things were added to the original file and nothing else changed: gradient
accumulation, so the recipe's effective batch of 64 survives on a 16GB/12GB box at widths
where a batch of 64 will not fit; ``--log-every``, because 200 steps of silence at this
box's throughput is indistinguishable from a hung run and has already cost a healthy run;
and a last-step checkpoint alongside the best one. The network, the loss, the split, the
sampler, the metrics and the schedule are untouched, so every number in FINDINGS.md remains
a yardstick. The previous version is kept as ``bc_train_pre512.py.bak``.

What the corpus is now, and why it is not what FINDINGS.md was measured on
-------------------------------------------------------------------------
``dataset/`` holds one pilot's own games, so ``build_corpus.py`` marks the frames where
*that pilot* was deciding. The corpus FINDINGS.md quotes was built differently — a scan of
the whole ladder for one 60-card *list*, piloted by hundreds of accounts, 4.86M frames.
This one is far smaller. Both give the guarantee that matters for cloning (every sampled
target is a decision one consistent expert actually made) and both leave every row
readable, but **val_exact from this corpus is not comparable with 71.27% or 78.35%**: those
were measured on different frames, from a different and much larger pool of pilots.

The compensating advantage is expert quality. This pilot wins ~70% of decided games,
against the ~46% of the mixed-account pool the old corpus averaged over. Cloning a stronger
player is worth more per frame, and BC's ceiling is the pilot it imitates.

Two contract details that decide whether the result is usable at all
--------------------------------------------------------------------
*The spec must be the one PPO serves.* ``LEARNER_SPEC`` (30-frame decision chain and
opponent history) is passed here, not ``DEFAULT_SPEC``. A policy cloned at 60 and rolled out
at 30 is being fed a different input than it learned from — the exact failure
``observation.py`` exists to prevent.

*The split must be by game.* Frames of one game share a label and are near-duplicates, so a
row-wise split leaks across it and reports a validation number that means nothing. Episodes
are partitioned, then frames follow their episode.

Sampling is block-shuffled, not row-shuffled, on purpose: the dataset is random-access over
Parquet row groups and a backward history scan walks off the front of its own group, so a
uniformly shuffled sampler evicts and refills the row-group cache on nearly every item.
Shuffling the order of blocks and reading sequentially inside them keeps the cache warm and
still decorrelates the gradient across games.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import resource
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# A sample here is not a tensor, it is a deep dict of them — one collated batch
# of 64 carries **653 separate tensors** (the feature tree: every card field of
# every option of every chain step, and so on). Under torch's default
# ``file_descriptor`` sharing strategy each of those becomes its own
# ``/dev/shm`` object whose file descriptor is held open in *both* the worker
# and this process for as long as the batch is alive. With ``--workers 10``,
# the default ``prefetch_factor`` of 2, and a train and a val loader, that is
# ~13,000 live descriptors before anything has been learned — which is what
# raised ``RuntimeError: unable to open shared memory object ... Too many open
# files``. (The ``(24)`` in that message is ``errno.EMFILE``, not a limit of
# 24; the real ceiling is ``ulimit -n``.)
#
# ``file_system`` shares one named region per storage instead of passing
# descriptors, so the FD count stops scaling with tensors-per-batch. This must
# be set in the parent before any worker is forked. The tradeoff is that a
# ``SIGKILL``ed run can leak files in ``/dev/shm`` (a clean exit or Ctrl-C
# still cleans up); the alternative is a ~13k FD ceiling on a knife edge.
torch.multiprocessing.set_sharing_strategy("file_system")

# ...and raise the soft descriptor limit to the hard one anyway, since the
# batches themselves are the parquet row caches' problem too and a login shell
# often defaults the soft limit to 1024 while permitting far more.
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
if _soft < _hard:
    resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "challenger" / "policy_network"))

from collate import collate_features, pad_stack  # noqa: E402
from dataset import PolicyFeatureDataset, transform  # noqa: E402
from observation import LEARNER_SPEC  # noqa: E402
from policy_experimental import (  # noqa: E402
    PolicyNetwork,
    decode_action,
    equivalence_mask,
    masked_selection_loss,
    selection_counts,
)
from vocab import OptionType  # noqa: E402


class ArchetypeDataset(PolicyFeatureDataset):
    """``PolicyFeatureDataset`` restricted to the frames one archetype was deciding at.

    The base class filters samples by ``player_name``, which cannot express this corpus: the
    deck is piloted by hundreds of different ladder accounts, and the same account plays
    other archetypes on other days. ``is_target_archetype`` is the per-row flag that does
    express it. Only the *sample* set is restricted — every row stays readable, because the
    backward history scans must still see the opponent's frames.
    """

    def __init__(self, parquet_path, **kwargs) -> None:
        super().__init__(parquet_path, **kwargs)
        flags = self._parquet.read(columns=["is_target_archetype"]).to_pandas()
        column = flags["is_target_archetype"].fillna(False).astype(bool)
        self._row_indexes = column.index[column].tolist()
        episodes = self._parquet.read(columns=["episode_id"]).to_pandas()["episode_id"]
        self.episode_of = episodes.to_numpy()

    def episode_ids(self) -> np.ndarray:
        """The episode each *sample* (not row) belongs to, for the game-level split."""
        return self.episode_of[np.asarray(self._row_indexes)]


class Subset(Dataset):
    def __init__(self, base: ArchetypeDataset, positions: list[int]) -> None:
        self.base, self.positions = base, positions

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index):
        return self.base[self.positions[index]]


def block_shuffled(count: int, block: int, rng: random.Random) -> list[int]:
    """Sequential inside blocks, blocks in random order (see the module docstring)."""
    blocks = [list(range(start, min(start + block, count))) for start in range(0, count, block)]
    rng.shuffle(blocks)
    return [index for chunk in blocks for index in chunk]


def collate(batch):
    """``(features, targets, options_mask)`` for one minibatch.

    The target is a multi-hot over the *options offered at that frame*, which is why it is
    built after collation: the option count is ragged and ``collate_features`` is what pads
    it to the batch maximum.
    """
    features = collate_features([observation["features"] for observation, _ in batch])
    width = features["decision_context"]["options"]["options_mask"].shape[-1]
    targets = pad_stack(
        [
            torch.zeros(width).index_fill_(
                0, torch.as_tensor([i for i in action if i < width], dtype=torch.long), 1.0
            )
            if len(action)
            else torch.zeros(width)
            for _, action in batch
        ],
        0.0,
    )
    options_mask = features["decision_context"]["options"]["options_mask"].squeeze(1)
    return features, targets, options_mask


def tree_to(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: tree_to(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_to(item, device) for item in value]
    return value


#: Reported per-category only when a category is offered at least this many times in the
#: evaluated slice. Below it the rate is noise — ENERGY_CARD and TOOL_CARD are 0.0% of
#: expert picks in the corpus scan, so a handful of frames would swing their number by
#: tens of points and read as a regression.
MIN_CATEGORY_SUPPORT = 20


@torch.no_grad()
def evaluate(model, loader, device, limit_batches: int | None = None):
    """``(loss, exact_match, same_play_match, per_category)``.

    ``per_category`` breaks the single-select decisions down by the ``OptionType`` the
    expert actually chose, as ``{name: {"n", "same_play", "chose_this_type"}}``. It is
    the diagnostic the aggregate numbers cannot give: one pointer softmax ranks every
    category against every other, so a policy can hold a perfectly good opinion about
    *which* attach while being systematically wrong about attaching versus ending the
    turn, and a single ``val_exact`` averages the two into one number that says which
    neither. ``chose_this_type`` is the confusion axis — how often the model picked an
    option of the expert's category at all, independent of picking the right one within
    it — so a category error and a targeting error are separable.

    ``exact_match`` is the historical number: the decoded selection equals the expert's
    index-for-index. ``same_play_match`` credits a pick that is *behaviourally identical*
    to the expert's — same option type, card, target and count, differing only in which
    duplicate slot was clicked.

    Both are reported because the strict one is misleading on its own. Measured on the
    Grimmsnarl corpus, 27.8% of single-select decisions offered at least one identical
    alternative, so a policy that reproduced the expert's *play* perfectly and broke ties
    at random would still score only **83.8%** exact index match. That ceiling has not
    been re-measured on this corpus, so treat 83.8% as an estimate here rather than a
    number: what is certain is that ``exact_match``'s ceiling is below 100% while
    ``same_play_match``'s is 100%, so each must be read against its own.
    """
    model.eval()
    total_loss = matched = same_play = seen = batches = 0.0
    per_category: dict[int, Counter] = defaultdict(Counter)
    for features, targets, options_mask in loader:
        features = tree_to(features, device)
        targets, options_mask = targets.to(device), options_mask.to(device)
        logits = model(features)
        total_loss += float(masked_selection_loss(logits, targets, options_mask))
        min_count, max_count = selection_counts(features)
        decoded = decode_action(logits, options_mask, min_count.clamp(min=1), max_count)
        equivalent = equivalence_mask(features["decision_context"]["options"], targets,
                                     options_mask)
        singles = targets.sum(-1) == 1
        option_types = features["decision_context"]["options"]["type"].squeeze(1)
        for row, picks in enumerate(decoded):
            expert = set(torch.nonzero(targets[row]).flatten().tolist())
            hit = set(picks) == expert
            matched += hit
            # The tie expansion only describes single-select decisions (only the first
            # target is expanded), so multi-select falls back to the strict result.
            if bool(singles[row]):
                row_same_play = len(picks) == 1 and bool(equivalent[row, picks[0]])
                same_play += row_same_play
                # Keyed on the expert's category, so each row lands in exactly one
                # bucket and the buckets partition the single-select decisions.
                expert_type = int(option_types[row, next(iter(expert))])
                bucket = per_category[expert_type]
                bucket["n"] += 1
                bucket["same_play"] += row_same_play
                bucket["chose_this_type"] += bool(picks) and all(
                    int(option_types[row, pick]) == expert_type for pick in picks
                )
            else:
                same_play += hit
            seen += 1
        batches += 1
        if limit_batches and batches >= limit_batches:
            break
    model.train()
    breakdown = {
        OptionType(option_type).name: dict(counts)
        for option_type, counts in sorted(per_category.items())
    }
    return (total_loss / max(batches, 1), matched / max(seen, 1),
            same_play / max(seen, 1), breakdown)


def format_categories(breakdown: dict, min_support: int = MIN_CATEGORY_SUPPORT) -> str:
    """One line per category with enough support to mean anything — see
    ``MIN_CATEGORY_SUPPORT``."""
    rows = [
        (name, c["n"], c["same_play"] / c["n"], c["chose_this_type"] / c["n"])
        for name, c in breakdown.items()
        if c["n"] >= min_support
    ]
    if not rows:
        return "    (no category reached minimum support)"
    width = max(len(name) for name, *_ in rows)
    return "\n".join(
        f"    {name:<{width}}  n={n:<6} same_play={sp:6.1%}  right_category={rc:6.1%}"
        for name, n, sp, rc in sorted(rows, key=lambda row: -row[1])
    )


def category_metrics(breakdown: dict, min_support: int = MIN_CATEGORY_SUPPORT) -> dict:
    """Flatten the breakdown into scalar wandb series, one pair per category."""
    metrics = {}
    for name, counts in breakdown.items():
        if counts["n"] < min_support:
            continue
        key = name.lower()
        metrics[f"bc/val_same_play/{key}"] = counts["same_play"] / counts["n"]
        metrics[f"bc/val_right_category/{key}"] = counts["chose_this_type"] / counts["n"]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="the MICRO-batch: how many samples are resident at once. "
                             "Multiply by --grad-accum for the batch the optimizer "
                             "actually sees")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="micro-batches accumulated per optimizer step. This box has "
                             "16GB of RAM and a 12GB GPU, where the reference recipe's "
                             "batch of 64 does not fit -- 64 collated samples is ~653 "
                             "tensors each, and the DataLoader holds "
                             "workers*prefetch_factor of them live. --batch-size 16 "
                             "--grad-accum 4 is the same 64 samples per update and the "
                             "same gradient, a quarter of the peak memory, and it keeps "
                             "every LR/step number below comparable with the reference "
                             "runs, since a 'step' stays one optimizer step")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    # --- capacity -----------------------------------------------------------------
    # Every default reproduces the 2,681,153-parameter network the current checkpoint
    # holds, so omitting all of these changes nothing. Raising --dim is the only one
    # that makes --init-from impossible (no loader here tolerates a shape change);
    # --nhead is free and extra layers only add keys.
    parser.add_argument("--dim", type=int, default=64,
                        help="hidden width. NOT checkpoint-compatible: changing it "
                             "forces training from scratch")
    parser.add_argument("--nhead", type=int, default=2,
                        help="attention heads; free to change, shapes do not depend on it")
    parser.add_argument("--chain-layers", type=int, default=8)
    parser.add_argument("--hist-layers", type=int, default=8)
    parser.add_argument("--ctx-layers", type=int, default=2,
                        help="self-attention layers across the options of one decision")
    parser.add_argument("--cross-layers", type=int, default=2,
                        help="OptionCrossAttention depth: each option querying the "
                             "fused state and every step of the decision prefix")
    parser.add_argument("--ff-mult", type=int, default=2,
                        help="feedforward width as a multiple of --dim")
    parser.add_argument("--board-tokens", action="store_true",
                        help="let each option cross-attend to the specific Pokemon it "
                             "targets, addressed by a shared slot embedding. Without it "
                             "the bench is mean-pooled before the option scorer sees it, "
                             "so 'attach to this Munkidori' and 'attach to that one' "
                             "differ only by a slot number and the target's own energy "
                             "is not an input. NOT checkpoint-compatible")
    parser.add_argument("--reset-slot-embed", action="store_true",
                        help="with --init-from, re-initialise slot_embed from scratch. "
                             "The board addressing now covers the CARD/ABILITY/ENERGY "
                             "options that name a Pokemon through area/index, not just "
                             "ATTACH and EVOLVE -- 2.7x the addressed option slots, and "
                             "the opponent-side slot rows were never trained before, "
                             "because no option addressed under the old scheme ever "
                             "pointed at the opponent's board. Loading those rows into "
                             "the wider scheme starts the run with a garbage address on "
                             "every newly-addressed option: measured on "
                             "bc_policy_grimmsnarl_d128_L54_3ep, evaluating it under the "
                             "wider addressing drops val_exact 61.7%% -> 60.1%% and CARD "
                             "same_play 60.9%% -> 58.5%%, entirely off-distribution. This "
                             "makes the addresses regrow from random instead, keeping "
                             "the rest of the warm start")
    parser.add_argument("--type-conditioned", action="store_true",
                        help="give the scorer an explicit per-OptionType logit bias and "
                             "gate its query on select_type/select_context. One pointer "
                             "softmax ranks every category on one scale, so 'should I "
                             "retreat at all' competes for the same weights that pick "
                             "which Pokemon to attach to -- measured on "
                             "bc_policy_grimmsnarl_d128_L54_3ep, every RETREAT/END/"
                             "ATTACK/ATTACH error is a category error, and RETREAT is "
                             "picked on 3.3%% of the frames the pilot retreated. "
                             "Checkpoint-compatible: both additions are zero-init, so "
                             "--init-from reproduces the source exactly")
    parser.add_argument("--tie-aware-loss", action="store_true",
                        help="credit any behaviourally identical option instead of the "
                             "one index the expert clicked. Changes the objective, not "
                             "the reported val_exact, so runs stay comparable")
    # --- schedule -----------------------------------------------------------------
    # There was no scheduler at all, despite a comment in policy_experimental.py
    # claiming bc_train adds warmup. At 8 layers and pre-LN that was survivable; at
    # 12 layers and 2x the width it is not, so warmup is now real.
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="linear LR warmup; 0 keeps the old constant-LR behaviour")
    parser.add_argument("--lr-schedule", default="constant", choices=("constant", "cosine"),
                        help="cosine decays to --min-lr-frac of --lr over --total-steps")
    parser.add_argument("--min-lr-frac", type=float, default=0.1)
    parser.add_argument("--total-steps", type=int, default=0,
                        help="horizon for the cosine schedule; defaults to --max-steps "
                             "or one epoch's worth of batches")
    parser.add_argument("--holdout", type=float, default=0.05,
                        help="fraction of EPISODES (not rows) held out")
    parser.add_argument("--block", type=int, default=512,
                        help="block-shuffle window; keeps the row-group cache warm")
    parser.add_argument("--cached-row-groups", type=int, default=24)
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers. 0 keeps the row caches in one process, "
                             "which is what makes the block shuffle pay")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="stop after this many optimizer steps (0 = full epochs)")
    parser.add_argument("--log-every", type=int, default=200,
                        help="optimizer steps between progress lines. The reference "
                             "recipe's 200 is ~3 minutes of total silence at this box's "
                             "throughput, right after startup -- which is "
                             "indistinguishable from a hung run and has already caused a "
                             "healthy run to be killed. 25 costs nothing and shows the "
                             "loop is alive within ~20s")
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--eval-batches", type=int, default=60)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(_ROOT / "challenger" / "bc_policy_grimmsnarl.pt"),
                        help="where the BEST val_exact checkpoint is written")
    parser.add_argument("--last-out", default=None, metavar="CHECKPOINT",
                        help="where the LATEST weights are written, refreshed at every "
                             "eval and at the end regardless of the metric. Defaults to "
                             "'<--out>.last.pt'; pass '' to switch it off")
    parser.add_argument("--init-from", default=None, metavar="CHECKPOINT",
                        help="warm-start from an earlier clone. Useful across an architecture "
                             "change: the pre-cross-attention checkpoint still agrees with the "
                             "new network on 94.8%% of decoded actions, so it is a far better "
                             "start than random even though the chain readout changed from "
                             "mean-pooling to a causal last-step and its weights are therefore "
                             "being used in a different computation")
    parser.add_argument("--wandb-project", default="pokemon-tcg-rl")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default="online",
                        choices=("online", "offline", "disabled"))
    parser.add_argument("--run-name", default="bc-grimsnarl-ex")
    args = parser.parse_args()
    if args.last_out is None:
        args.last_out = str(args.out) + ".last.pt"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    started = time.time()
    base = ArchetypeDataset(
        args.parquet, transform=transform, spec=LEARNER_SPEC,
        cached_row_groups=args.cached_row_groups,
    )
    episodes = base.episode_ids()
    unique = np.unique(episodes)
    rng.shuffle(unique := list(unique))
    holdout = set(unique[: max(1, int(len(unique) * args.holdout))])
    val_positions = [i for i, e in enumerate(episodes) if e in holdout]
    train_positions = [i for i, e in enumerate(episodes) if e not in holdout]
    print(f"[bc] {len(episodes):,} archetype frames over {len(unique):,} episodes "
          f"({time.time() - started:.0f}s to index)")
    print(f"[bc] train {len(train_positions):,} frames / {len(unique) - len(holdout):,} games,"
          f"  val {len(val_positions):,} frames / {len(holdout):,} games")

    import wandb

    if not hasattr(wandb, "init"):
        # This directory holds wandb's own run logs, so with no wandb installed the
        # directory imports as an empty namespace package and the failure looks like a
        # missing attribute rather than a missing package.
        raise SystemExit("wandb resolved to the run-log directory, not the package. "
                         "Install it: pip install wandb")
    run = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity, name=args.run_name,
        mode=args.wandb_mode,
        config=vars(args) | {
            "stage": "behavioural-cloning",
            "archetype": "marnie-grimmsnarl-ex",
            "effective_batch": args.batch_size * args.grad_accum,
            "frames": len(episodes),
            "episodes": len(unique),
            "train_frames": len(train_positions),
            "val_frames": len(val_positions),
            "val_games": len(holdout),
            "decision_chain_size": LEARNER_SPEC.decision_chain_size,
            "opponent_history_size": LEARNER_SPEC.opponent_history_size,
        },
    )
    print(f"[wandb] {run.url if args.wandb_mode == 'online' else args.wandb_mode}", flush=True)

    train_set, val_set = Subset(base, train_positions), Subset(base, val_positions)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.workers)

    model = PolicyNetwork(
        dim=args.dim, dropout=args.dropout, nhead=args.nhead,
        chain_layers=args.chain_layers, hist_layers=args.hist_layers,
        ctx_layers=args.ctx_layers, cross_layers=args.cross_layers,
        ff_mult=args.ff_mult, board_tokens=args.board_tokens,
        type_conditioned=args.type_conditioned,
    ).to(device)
    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=True)
        result = model.load_state_dict(checkpoint, strict=False)
        if result.unexpected_keys:
            raise SystemExit(f"{args.init_from} has keys this network does not have: "
                             f"{list(result.unexpected_keys)[:5]}")
        modules = sorted({key.split(".")[0] for key in result.missing_keys})
        print(f"[bc] warm-started from {args.init_from}"
              + (f"; {len(result.missing_keys)} keys absent, in {modules} (trained fresh)"
                 if result.missing_keys else ""))
        if args.reset_slot_embed and hasattr(model, "slot_embed"):
            model.slot_embed.reset_parameters()
            print("[bc] slot_embed re-initialised; board addresses regrow from scratch")
        # NOT zeroed here, unlike the PPO warm start: BC has 800k frames of dense
        # supervision, which is the one place cross-attention can actually be *fitted*
        # rather than nudged, so it starts at its normal random init and trains.
    parameters = sum(p.numel() for p in model.parameters())
    print(f"[bc] {parameters:,} parameters on {device}  arch={model.arch}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    steps_per_epoch = max(1, len(train_positions) // (args.batch_size * args.grad_accum))
    horizon = args.total_steps or args.max_steps or steps_per_epoch * args.epochs

    def lr_scale(current: int) -> float:
        """Warmup then optional cosine decay, as a multiplier on ``--lr``."""
        if args.warmup_steps and current < args.warmup_steps:
            return (current + 1) / args.warmup_steps
        if args.lr_schedule == "cosine":
            progress = min(1.0, max(0.0, (current - args.warmup_steps)
                                    / max(1, horizon - args.warmup_steps)))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return args.min_lr_frac + (1.0 - args.min_lr_frac) * cosine
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    wandb.summary.update({"bc/parameters": parameters, "bc/steps_per_epoch": steps_per_epoch,
                          "bc/lr_horizon": horizon})
    print(f"[bc] {steps_per_epoch:,} steps/epoch; schedule={args.lr_schedule} "
          f"warmup={args.warmup_steps} horizon={horizon:,}")

    best = -1.0
    step = 0
    micro = 0  # micro-batches since the last optimizer step
    history = []

    def write_checkpoint(path, at_step: int, val_loss: float, exact: float,
                         same_play: float) -> None:
        """Weights plus a sidecar recording what they are.

        The ``.meta.json`` is not optional bookkeeping: a ``state_dict`` carries no width
        or depth, so without it a checkpoint is identifiable only by tensor shape -- and a
        mismatched module has already silently broken a warm start in this repo once.
        ``eval_checkpoint.py`` reads ``arch`` from here to rebuild the network.
        """
        torch.save(model.state_dict(), path)
        Path(str(path) + ".meta.json").write_text(json.dumps(
            {"step": at_step, "val_loss": val_loss, "val_exact": exact,
             "val_same_play": same_play, "arch": model.arch, "parameters": parameters,
             "spec": {"decision_chain_size": LEARNER_SPEC.decision_chain_size,
                      "opponent_history_size": LEARNER_SPEC.opponent_history_size},
             "args": vars(args), "history": history}, indent=2, default=str))
    print(f"[bc] starting epoch 0; the first batch waits on {args.workers} dataloader "
          f"workers building features from scratch, so expect ~30s of silence before the "
          f"first progress line", flush=True)
    for epoch in range(args.epochs):
        order = block_shuffled(len(train_set), args.block, rng)
        loader = DataLoader(
            Subset(train_set, order), batch_size=args.batch_size, shuffle=False,
            collate_fn=collate, num_workers=args.workers,
        )
        for features, targets, options_mask in loader:
            features = tree_to(features, device)
            targets, options_mask = targets.to(device), options_mask.to(device)
            equivalent = None
            if args.tie_aware_loss:
                # Stop charging the model for picking a duplicate slot: the objective
                # becomes -logsumexp over the tie group, so any identical play is a
                # correct answer. 27.8% of single-select frames here have such a group,
                # and fitting which slot the human clicked is fitting a coin flip.
                equivalent = equivalence_mask(features["decision_context"]["options"],
                                              targets, options_mask)
            loss = masked_selection_loss(model(features), targets, options_mask,
                                         equivalent=equivalent)
            # Divided by the accumulation count so the accumulated gradient is the MEAN
            # over the whole effective batch, matching what one big batch would produce
            # rather than a gradient --grad-accum times larger.
            (loss / args.grad_accum).backward()
            micro += 1
            if micro < args.grad_accum:
                continue
            micro = 0
            # Clipping after the last micro-batch, so the norm being clipped is the
            # effective batch's -- clipping each micro-batch separately would be a
            # different (and stricter) objective than the reference runs used.
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if args.log_every and step % args.log_every == 0:
                rate = step * args.batch_size * args.grad_accum / (time.time() - started)
                wandb.log({"bc/train_loss": float(loss.detach()), "bc/frames_per_second": rate,
                           "bc/epoch": epoch, "bc/lr": optimizer.param_groups[0]["lr"]},
                          step=step)
                # Progress and ETA on the line itself: this run is hours long, and
                # "step 4200 loss 0.9" alone gives no way to tell whether that is on
                # schedule without doing arithmetic against the horizon.
                remaining = (horizon - step) * (time.time() - started) / max(step, 1)
                print(f"[bc] epoch {epoch} step {step:,}/{horizon:,} "
                      f"({step / max(horizon, 1):.0%}) loss {float(loss.detach()):.4f} "
                      f"({rate:.0f} frames/s, ~{remaining / 3600:.1f}h left)", flush=True)
            if args.eval_every and step % args.eval_every == 0:
                val_loss, exact, same_play, categories = evaluate(
                    model, val_loader, device, args.eval_batches)
                history.append({"step": step, "val_loss": val_loss, "val_exact": exact,
                                "val_same_play": same_play, "categories": categories})
                marker = ""
                if exact > best:
                    best, marker = exact, "  <- best, saved"
                    write_checkpoint(args.out, step, val_loss, exact, same_play)
                # The latest weights, overwritten every eval and kept whatever the metric
                # did. Two reasons it is not redundant with --out. First, a kill (OOM,
                # power, Ctrl-C) otherwise loses everything since the last *improvement*,
                # which late in a cosine schedule can be thousands of steps. Second, best
                # and last answer different questions: --out is the checkpoint to play,
                # --last-out is where training actually ended, which is what a warm start
                # or a resume wants -- and it is the only one that shows what the run did
                # after val_exact turned over, which on this pipeline is the point where
                # win rate and val_exact stop agreeing (see FINDINGS.md).
                if args.last_out:
                    write_checkpoint(args.last_out, step, val_loss, exact, same_play)
                wandb.log({"bc/val_loss": val_loss, "bc/val_exact_match": exact,
                           "bc/val_same_play_match": same_play,
                           "bc/best_val_exact_match": best,
                           **category_metrics(categories)}, step=step)
                print(f"[bc] step {step}  val_loss {val_loss:.4f}  "
                      f"val_exact {exact:.1%}  same_play {same_play:.1%}{marker}\n"
                      f"{format_categories(categories)}",
                      flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        # Drop a partial accumulation group rather than letting its gradient survive into
        # the next epoch's first update, where it would be mixed with samples the block
        # shuffle has already reordered.
        optimizer.zero_grad(set_to_none=True)
        micro = 0
        if args.max_steps and step >= args.max_steps:
            break

    # The final step almost never lands on an --eval-every boundary, so evaluate once more
    # here: without it the "last" checkpoint is really "last eval", up to --eval-every
    # steps stale, and the run would end without a scored record of where it finished.
    # This also updates --out if those trailing steps happened to be the best.
    final_loss, final_exact, final_same, final_categories = evaluate(
        model, val_loader, device, args.eval_batches)
    history.append({"step": step, "val_loss": final_loss, "val_exact": final_exact,
                    "val_same_play": final_same, "categories": final_categories,
                    "final": True})
    if args.last_out:
        write_checkpoint(args.last_out, step, final_loss, final_exact, final_same)
    if final_exact > best:
        best = final_exact
        write_checkpoint(args.out, step, final_loss, final_exact, final_same)
    print(f"[bc] final   val_loss {final_loss:.4f}  val_exact {final_exact:.1%}  "
          f"same_play {final_same:.1%}\n{format_categories(final_categories)}", flush=True)
    wandb.summary.update({"bc/best_val_exact_match": best, "bc/steps": step,
                          "bc/final_val_exact_match": final_exact,
                          "bc/final_val_loss": final_loss,
                          "bc/minutes": (time.time() - started) / 60,
                          **{f"final_{k}": v
                             for k, v in category_metrics(final_categories).items()}})
    wandb.finish()
    print(f"[bc] done in {(time.time() - started) / 60:.1f} min, best val exact {best:.1%}")
    print(f"[bc] best -> {args.out}")
    if args.last_out:
        print(f"[bc] last -> {args.last_out}")


if __name__ == "__main__":
    main()
