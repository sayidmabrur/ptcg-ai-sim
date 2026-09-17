#!/usr/bin/env bash
#
# Grimmsnarl ex at dim=512, on the pilot corpus in dataset/.
#
#   ./run_bc_512.sh                  # build the corpus if absent, then train
#   DIM=256 ./run_bc_512.sh          # a rung lower on the width ladder
#   EPOCHS=6 ./run_bc_512.sh
#   SKIP_BUILD=1 ./run_bc_512.sh
#
# Why 512 is affordable, measured on this box rather than assumed
# ---------------------------------------------------------------
# Benchmarked forward+backward on real batches of 16, dataloader cost excluded:
#
#     dim 128    9,590,017 params   0.26 GB peak   162 ms/step    99 frames/s
#     dim 256   31,795,713 params   0.66 GB peak   166 ms/step    96 frames/s
#     dim 384   66,617,089 params   1.40 GB peak   181 ms/step    89 frames/s
#     dim 512  114,054,145 params   2.36 GB peak   185 ms/step    87 frames/s
#
# 11.9x the parameters of dim 128 for 14% more time per step and 2.36GB of a 12GB card.
# That is not a bargain the model found -- it is what happens when the GPU was never the
# bottleneck. This pipeline is dataloader-bound: Python feature-building in the workers ran
# at ~83 frames/s while the GPU sat at ~25%, so most of the extra width is paid for out of
# idle GPU time. Expect roughly 65-75 frames/s in practice, i.e. 10-25% slower wall clock,
# not 12x.
#
# Nothing in the network blocks the width: 512 is divisible by --nhead, and
# ReversePositionalEmbedding is indexed by recency with max_len=64, independent of dim.
#
# The real risk here is statistical, not computational
# ----------------------------------------------------
# 114M parameters against this corpus's ~180k trainable frames is ~600 parameters per
# frame. The dim=128 evidence for underfitting is real -- bc_policy_board_snap's val_loss
# was still falling at step 150,000 (0.6433 -> 0.6262 -> 0.6185) -- but that was measured on
# the 4.86M-frame ladder corpus, which is ~27x this one. The Alakazam run is the closer
# analogue: 259k frames, and dim 128 -> 82.94% vs dim 64 -> 81.97%, i.e. +0.97pp for 3x the
# parameters, with no overfitting at 10 epochs. Width helped there, but with sharply
# diminishing returns.
#
# So this script defaults to dim 512 as asked, and the guards that make an overfit visible
# rather than silent are all on: the split is by game, the saved checkpoint is the best
# val_exact rather than the last, a --last-out checkpoint records where training actually
# ended, and dropout is raised to 0.15 (from 0.1) because 114M parameters on 180k frames is
# where regularisation starts to matter. If val_loss turns up while val_exact keeps
# climbing, the width is too large for this corpus and DIM=256 is the rung to try -- the
# ladder above is exactly why the same script takes a DIM.
#
# And read the win rate before believing any of it. FINDINGS.md records a run reaching
# 78.4% val_exact while its win rate against the frozen field FELL to 19.8%. A better
# imitation of a pool that loses half its games is a worse player -- though note this pilot
# wins ~70%, so that trap is shallower here than it was for the mixed-account corpus.

set -uo pipefail
cd "$(dirname "$0")"

# The interpreter directly, not `conda run`: that wrapper buffers child output until the
# process exits, which would leave a multi-hour log empty until it was too late to react.
PY="/home/capon/miniconda3/envs/kaggle-pokemon/bin/python -u"

CORPUS="${CORPUS:-grimmsnarl_all.parquet}"
REPLAYS="${REPLAYS:-dataset}"
# Every replay lives flat in dataset/ now, one uniform <date>_<id>_<slug>.json convention.
# Seats are selected by DECK, not by pilot name: card 647 is Marnie's Grimmsnarl ex, so any
# seat that opened with it is on the archetype -- which is what makes this correct across
# pilots who also play other decks, replays whose filename carries no pilot at all, and
# lists that differ from challenger_deck.csv by a card or two.
DECK_CARD="${DECK_CARD:-647}"
DIM="${DIM:-512}"
NHEAD="${NHEAD:-8}"
# Depth, configurable because it is the cheapest capacity on this box: doubling the two
# temporal stacks costs ~3M parameters against the ~22M that going dim 128->256 costs, and
# the pipeline is dataloader-bound so neither shows up much in wall clock. Note the two
# sequence stacks only ever see 30 elements (LEARNER_SPEC), where every position already
# reaches every other in one layer -- so extra depth here refines, it does not extend reach.
CHAIN_LAYERS="${CHAIN_LAYERS:-12}"
HIST_LAYERS="${HIST_LAYERS:-12}"
CTX_LAYERS="${CTX_LAYERS:-4}"
CROSS_LAYERS="${CROSS_LAYERS:-6}"
DROPOUT="${DROPOUT:-0.15}"
EPOCHS="${EPOCHS:-8}"
BATCH="${BATCH:-16}"
ACCUM="${ACCUM:-4}"
WORKERS="${WORKERS:-6}"
LR="${LR:-2e-4}"
WANDB="${WANDB:-online}"
# Fraction of EPISODES held out. 0.10 (9:1) rather than bc_train's 0.05 default: at 5% this
# corpus leaves ~100 validation games, which is few enough that the val curve carries real
# noise -- and the split is by game, so a small holdout is small in games, not just rows.
HOLDOUT="${HOLDOUT:-0.10}"
# The checkpoint path must encode every dimension the RUN_NAME does -- EPOCHS included.
# It did not, and two runs differing only in --epochs wrote the same file and overwrote each
# other's best checkpoint for 55 minutes. The logs were distinguishable (RUN_NAME carries the
# epoch count) but the weights were not, so the .pt could have come from either run.
OUT="${OUT:-challenger/bc_policy_grimmsnarl_d${DIM}_L$((CHAIN_LAYERS+HIST_LAYERS+CTX_LAYERS+CROSS_LAYERS))_${EPOCHS}ep.pt}"
RUN_NAME="${RUN_NAME:-bc-grimsnarl-d${DIM}-L$((CHAIN_LAYERS+HIST_LAYERS+CTX_LAYERS+CROSS_LAYERS))-${EPOCHS}ep}"
mkdir -p logs

# ------------------------------------------------------------------ 1. corpus
if [[ -z "${SKIP_BUILD:-}" && ! -f "$CORPUS" ]]; then
  echo "[512] building $CORPUS from $REPLAYS at $(date -Is)"
  $PY build_corpus.py --replays "$REPLAYS" --deck-contains "$DECK_CARD" --out "$CORPUS" \
    2>&1 | tee logs/build_corpus.log
else
  echo "[512] using existing $CORPUS ($(du -h "$CORPUS" | cut -f1))"
fi

# ------------------------------------------------------------------ 2. steps
# Derived, not hardcoded: --total-steps is the cosine horizon, and a horizon that disagrees
# with the run leaves the LR either still high at the end or at its floor halfway through.
# Both are silent failures.
STEPS="${STEPS:-$($PY - "$CORPUS" "$EPOCHS" "$BATCH" "$ACCUM" "$HOLDOUT" <<'EOF'
import sys
import pyarrow.parquet as pq
corpus, epochs, batch, accum = sys.argv[1], *map(int, sys.argv[2:5])
flags = pq.read_table(corpus, columns=["is_target_archetype"])["is_target_archetype"]
samples = sum(1 for v in flags.to_pylist() if v)
train = int(samples * (1.0 - float(sys.argv[5])))   # --holdout, split by episode
print(max(1, train // (batch * accum)) * epochs)
EOF
)}"

echo "[512] dim $DIM, $EPOCHS epochs = $STEPS optimizer steps"
echo "[512] effective batch $((BATCH * ACCUM)) (micro $BATCH x accum $ACCUM), $WORKERS workers"

# ------------------------------------------------------------------ 3. train
# --lr 2e-4 rather than the 3e-4 the dim=128 recipe used. Not a preference: the update a
# pre-LN transformer takes at a fixed LR grows with width, and 2000 steps of warmup is
# already carrying a 12-layer stack. If it diverges anyway, halve it again -- a diverged run
# shows up as val_loss flat or rising from the very first eval, not as noise.
$PY bc_train.py \
  --parquet "$CORPUS" \
  --out "$OUT" \
  --run-name "$RUN_NAME" \
  --dim "$DIM" --nhead "$NHEAD" \
  --chain-layers "$CHAIN_LAYERS" --hist-layers "$HIST_LAYERS" \
  --ctx-layers "$CTX_LAYERS" --cross-layers "$CROSS_LAYERS" \
  --board-tokens \
  --dropout "$DROPOUT" \
  --holdout "$HOLDOUT" \
  --lr "$LR" --warmup-steps 2000 --lr-schedule cosine --min-lr-frac 0.05 \
  --batch-size "$BATCH" --grad-accum "$ACCUM" \
  --workers "$WORKERS" --cached-row-groups 6 --block 512 \
  --epochs "$EPOCHS" --max-steps "$STEPS" --total-steps "$STEPS" \
  --log-every 25 --eval-every 1000 --eval-batches 200 \
  --wandb-project pokemon-tcg-rl \
  --wandb-mode "$WANDB" \
  2>&1 | tee "logs/${RUN_NAME}.log"

# ------------------------------------------------------------------ 4. score
# On the whole holdout, not the 200-batch sample used for checkpoint selection.
echo "[512] scoring $OUT on the full holdout at $(date -Is)"
$PY eval_checkpoint.py --checkpoint "$OUT" --parquet "$CORPUS" --holdout "$HOLDOUT" --eval-batches 0 \
  2>&1 | tee "logs/${RUN_NAME}.eval.log"

echo "[512] finished at $(date -Is)"
