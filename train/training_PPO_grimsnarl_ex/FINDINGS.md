# Search + value function for the challenger agent — findings

Everything here is measured on this box against the three frozen BC agents
(`crustle`, `alakazam`, `lucario`), seats alternated, Wilson 95% intervals.

## The reference ladder

The single most important result of the session, because it moved the target:

| agent | vs frozen field | n |
|---|---|---|
| uniform random | 8.3% | 180 |
| IS-MCTS, replay-fit V, no fixes | 21.3% [17.1, 26.3] | 300 |
| PPO best checkpoint, 30k episodes / ~8 h | 28.3% | 60 |
| **BC checkpoint (`crustle_frozen/bc_policy.pt`) piloting the challenger deck** | **41.1% [38.8, 43.4]** | **1796** |

**The PPO run spent eight hours training from random init and finished well
below a checkpoint that was already in the repository.** Nothing in this project
had measured that baseline, so every comparison up to this point was against
28.3% when the real bar was 41.1%.

The cause is mechanical and was visible from the first read of `train.py`:
`--init-from` is documented but has never worked. The challenger's `CardEmbed`
gained ability/skill fields, widening `proj` from 151 to 193 inputs, so
`bc_policy.pt` fails to load with 19 shape mismatches and 19 missing keys. Every
run in `checkpoints/` has `init_from: None` — not by choice.

**Fix shipped:** `train.py --actor-arch crustle --init-from
crustle_frozen/bc_policy.pt`. Builds the actor from crustle's own
`PolicyNetwork` generation, which the checkpoint loads into cleanly. Gives up
the newer ability features in exchange for starting training at ~41% instead of
0%.

Two initialisation details that matter for a warm start, both fixed:

- **Stop logit.** A warm-started actor has a trained option head competing
  against a random `stop` logit inside the same softmax; left alone it corrupts
  every optional and multi-select decision from the first rollout. Now biased to
  -4.0, reproducing `decode_action`'s measured floor-at-one behaviour.
- **Critic output layer zeroed.** `advantage = R - V(s)` then starts as the
  return itself. The smoke test showed value loss 3.5 and explained variance
  -2.4 on update 1 — a confidently wrong critic pushing an already-competent
  policy in an uncorrelated direction.

## RL is beating BC — current status

Target: **>=60%** against the frozen field, achieved through RL. BC is not a
deliverable; it is the starting point.

| stage | greedy eval | n |
|---|---|---|
| BC checkpoint (warm start) | 41.1% | 1796 |
| PPO exploit run, eval 150 | 44.2% | 120 |
| **PPO exploit run, eval 175** | **51.7%** | 120 |

Per-opponent at eval 175: crustle 57.5%, alakazam 40.0%, lucario 57.5%. Crustle
went from 16.7% to 57.5%, which is the matchup that was worst.

### What unblocked it

The conservative settings that protected the warm start were also preventing it
from moving — the KL budget bound on essentially every update, truncating each
to a single epoch. The opponents are *fixed and deterministic*, so the target is
a best response, not an equilibrium, and best responses are found by exploiting
specific weaknesses. That needs step size and exploration:

| | before | after |
|---|---|---|
| lr | 5e-5 | 1.5e-4 |
| entropy | 0.005 | 0.01, then 0.005 |
| target KL | 0.015 | 0.03 |
| PPO epochs run | 1 of 2 | 3-4 of 4 |
| prize shaping | 1.0 | 1.5 |

### An intervention worth recording

At entropy 0.01, greedy eval improved (44.2% -> 51.7%) while entropy climbed
0.68 -> 0.89 and the **sampled** win rate fell 43% -> 35%. The bonus was
flattening the distribution: the argmax kept improving while the policy it is
drawn from dissolved. Rollouts come from the sampled policy, so left alone this
ends with the gains lost. Halved to 0.005 rather than removed, since the
exploration is what found the improvement in the first place.

## PPO from a BC warm start — the earlier, superseded result

`./run_warmstart_ppo.sh`. Actor built from crustle's `PolicyNetwork`
generation so `bc_policy.pt` loads, critic zeroed, stop logit biased to -4.0,
lr 5e-5, entropy 0.005, target KL 0.015, prize shaping on.

| | greedy eval vs field |
|---|---|
| PPO from scratch, 30k episodes / ~8 hours | 28.3% |
| BC checkpoint, no training | 41.1% |
| **warm-started PPO, 1,440 episodes / ~45 minutes** | **53.3%** |

Sampled training win rate over the run so far: 29.2% → 42.8%. Entropy flat at
~0.68, so the warm start is not being flattened by the entropy bonus. Critic
explained variance 0.00 → 0.16, which is the zeroed-init critic learning from a
sensible starting point rather than fighting its own random output.

**Caveat:** the eval is 20 episodes per opponent, n=60, so 53.3% carries roughly
±12.6 points and overlaps the 41.1% baseline. The single number is not yet
conclusive; the trajectory is.

This is the answer to "a training run that can learn". It was blocked by a
mechanical bug — a checkpoint that would not load — not by anything about PPO,
the reward, or the game.

## The engine has a search API, and nothing was using it

`cg/api.py` exposes `search_begin` / `search_step` / `search_release` /
`search_end` — a determinisation interface that instantiates a
complete-information world from predicted card ids for every hidden zone.
`cg/game.py` already stashes the blob it needs (`obs["search_begin_input"]`) on
every observation. `train.py` imports none of it.

Thirteen probes (`probe_search_api.py`), all measured:

| | result | consequence |
|---|---|---|
| `search_step` | 0.098 ms (engine 0.036, JSON 0.056) | ~10k steps/s |
| stale `searchId`s | remain steppable; `release` invalidates | full tree branching works |
| **determinism** | **1.000 with `manual_coin=True`** (60/60 frames) | no chance nodes; a path can be recomputed |
| coin control | choosing YES/NO **decides the face** | chance is enumerable |
| memory | 30 KB/node, free-listed, plateaus ~135 MB | not a leak |
| semantic validation | **none** — accepted 4 copies of a discarded card as a hand | legality is the sampler's job |
| deck order | `your_deck[-1]` is the top | particles written bottom-first |
| live battle | survives searching in-process | worker layout unaffected |

`searchId`s **restart at 0 after every `search_end()`**. Anything memoising on
them across decisions silently returns another position's answer. This cost a
real debugging cycle and is a landmine for any future transposition table.

**The coin trap:** `manual_coin=True` lets *you* call the flip. It is on for
determinism, not because the search may choose. A `COIN_HEAD` node is sampled,
never maximised — treating it as a decision produces an agent that plans as
though it wins every 50/50.

## Belief sampler

`belief.py` + `probe_belief.py`. Ground truth comes from the fact that every
observation carries *both* players and only the non-owner's `hand` is redacted,
so at any frame the owner's real hand validates the belief an opponent would
have formed. Exact, same-timestep, no engine instrumentation.

| check | result |
|---|---|
| own-side pool closes | 22,443 / 22,443 |
| opponent-view pool closes | 22,443 / 22,443 |
| **true hand in support** | **22,443 / 22,443** |
| particles accepted by `search_begin` | 88,592 / 88,592 |
| true decklist wrongly eliminated | 0 frames |
| posterior resolves to one candidate | 120/120 games, median 5 frames |

Throughput ~50 µs/particle. Three bugs the ground-truth check caught, none of
which the engine would have reported:

1. `visible_cards` read the `hand` zone unconditionally, so "what the opponent
   believes about me" subtracted my actual hand — the true hand was not even
   *drawable* from the pool on **71% of frames**.
2. Face-down in-play Pokémon are out of the deck but invisible (surplus of 1).
3. A card mid-resolution belongs to no zone; it appears only as
   `selection.effect` (surplus of 1 on 18% of frames). Fixed by keying the walk
   on `serial` rather than counting occurrences.

**Known limitation:** `partition_hidden` splits the opponent pool uniformly.
This is **bias, not variance** — it does not shrink with more particles, it is
concentrated on exactly the decision-relevant cards, and search amplifies it
rather than averaging it out. Because the support is provably exact, a learned
hand model can be retrofitted as importance weights over the same particles
without rewriting the sampler, and `probe_belief.py` shows the labels are free
to harvest.

## Value function

Rollout evaluation was measured and rejected: at full depth a single random
playout's residual noise is **0.89** against a true between-action spread of
**0.31** (SNR 0.36), and no rollout policy changes that — averaging three
playouts gave exactly 0.89/√3, i.e. pure Monte Carlo variance. Reaching SNR 2
would need ~40 playouts per action per node at 40 ms each. **A value network is
a precondition for the search, not an optimisation of it.**

`value_mt.pt`, trained on 331,702 frames from 3,894 on-distribution self-play
games (`gen_value_data.py`), split by game:

| turn | expl. var | accuracy |
|---|---|---|
| 1–4 | 0.122 | 64.9% |
| 5–8 | 0.211 | 68.8% |
| 9–12 | 0.323 | 74.9% |
| 13–18 | 0.384 | 76.8% |
| 19–26 | 0.501 | 80.9% |
| 27+ | 0.528 | 82.0% |
| **ALL** | **0.294** | **72.6%** |

Prize-differential-only baseline on the same holdout: **0.060**. The model beats
it roughly 5×, so it is not merely reading the scoreboard.

**Prize-margin auxiliary** (the second head, predicting final prize margin
rather than just win/loss) — mean explained variance across 24 configurations
each:

| margin weight | mean EV | best EV |
|---|---|---|
| 0.0 | 0.2865 | 0.2951 |
| 0.5 | 0.2890 | 0.2957 |
| 1.0 | **0.2896** | **0.2973** |

Monotone and consistently positive, but small: **+0.003 EV (+1.1% relative)**.
It is real and the best configuration uses it, but it is not the lever that
changes the agent's strength.

### Lethality features

Added six features answering the question a Pokémon player asks first and which
nothing in the original 24 captured: *can I take the knockout this turn, and can
they take one on me?* Attack cost and damage are not on the board struct, so
this needs a join against the engine's static `Attack` table plus an energy
affordability check (greedy, which can only ever under-report — the conservative
direction).

Measured on the replay holdout, 24 features vs 30:

| turn | 24 feat | 30 feat |
|---|---|---|
| 0–4 | 0.088 | 0.100 |
| 5–8 | 0.249 | 0.258 |
| 9–12 | 0.474 | 0.485 |
| 13–18 | 0.534 | 0.525 |
| 19–26 | 0.534 | 0.565 |
| 27+ | 0.840 | 0.867 |
| **ALL** | **0.294** | **0.304** |

**+0.010 EV, +0.7 pt accuracy.** Positive in five of six buckets, but far smaller
than expected for what should be the most decision-relevant fact in the game.
The likely reason is that HP and attached-energy counts already proxy it well
enough for a model with this much data.

> **Incident worth recording.** Appending these features to `SCALAR_NAMES` while
> a configuration sweep was running invalidated `value_mt.pt` and killed the
> sweep, costing ~40 minutes. The feature-name guard in `load_value_net` did its
> job — the alternative is silently misaligned columns — but the mistake was
> mutating a shared contract while a job depended on it. `FeatureSubset` now
> serves older checkpoints by slicing to their prefix, since new features are
> appended rather than inserted. It never pads or reorders: padding feeds a
> model inputs it never saw, reordering silently remaps every column.

Three methodological traps, all avoided and all worth keeping avoided: split by
episode not row (frames of one game share a label and are near-duplicates);
report stratified by turn (late frames are easy and dominate an aggregate); and
measure against the prize-differential baseline, not against zero. The trainer
passes both a positive and a negative control.

## Infrastructure

`arena.py` — parallel game runner, 11 processes, **10× throughput** (2.4 games/s
vs 0.25). Wilson intervals, seat-balanced, doubles as the value-data collector.
Every earlier experiment in this thread ran at n=30, where a win rate near 0.7
carries a ±17-point interval; three "results" were entirely inside their own
error bars.

A trap worth recording: killing a `Pool` parent orphans its children. Two dead
runs left 23 zombie workers and the box ran 3× oversubscribed, silently, at a
third speed.

A second one, same family, found the hard way at `--workers 11`: **never put a
state dict on a multiprocessing queue.** `torch`'s reducer turns every storage
into a duplicated file descriptor, so one weight broadcast to 11 workers asks for
~2,200 fds against a login shell's default `ulimit -n` of 1024. What fails is the
queue's *feeder thread*, not the parent — so the run prints a wall of
`OSError: [Errno 24] Too many open files` and then sits there, waiting the full
`--worker-timeout` for results from workers that never received weights, while
whatever workers did get through play with an **untrained** actor (`--init-from`
is applied in the parent, so a worker's fallback is random init). A crash would
have been kinder than 3 updates of quietly off-policy data. `broadcast` now ships
`torch.save` bytes — one ~10 MB pipe write per worker per update against a ~66 s
cycle — and `RolloutPool` raises the fd soft limit to the hard limit before
forking as well. Verified with `soft = hard = 1024` so only the bytes path can
save it.

### The environment is no longer a singleton — and it did not make training faster

`cg.game` keeps the battle pointer on a class attribute, so one process holds one
battle and every rollout forward is batch 1. pokeka2026's `PTCGBattle` owns its
pointer explicitly, so N battles now run in one process and the N pending
decisions collate into one forward: `pokeka_env.py` (engine, pinned to *this*
repo's `libcg.so` — pokeka ships a newer one and the vocab tables that size every
`bc_policy.pt` are read live from the engine) and `vec_rollout.py`
(`train.py --num-envs V`).

Batching delivered exactly what it should and it did not matter:

| | per rollout decision |
|---|---|
| network forward, batch 1, CPU | 7.6 ms |
| network forward, batch 64, CPU | 1.2 ms |
| network forward, batch 64, GPU | **0.26 ms** (16 ms/batch, flat to B=64) |
| `dataset.transform` | **6.4 ms** |
| `collate_features` | 4.4 ms at B=1, 3.1 ms at B=64 |
| engine `select` | 0.2 ms |

End to end, 256 episodes at 64 per update, `--ppo-epochs 2 --minibatch-size 128`:

| setting | ep/hour |
|---|---|
| `--workers 11` (11 one-battle processes) | **3120** |
| `--num-envs 32` (32 batched battles, GPU) | 1440 |

**Throughput tracks cores, not batch.** `transform` builds ~990 individual
tensors per decision because a decision's `decision_chain` re-featurises up to 60
earlier decisions — the feature build is quadratic in episode length and
single-threaded Python, and it is 87% of a rollout decision. No amount of
batching touches it, so 11 processes beat 32 batched battles by 2.2×. pokeka's own
compute plan reached the same conclusion from the other side: "GPU is NOT the
lever for this workload", 48 envs over 8 workers bought 1.63×.

`--num-envs` is still the better mode when one process is wanted rather than
twelve, when only a GPU is available, or to make rollout and update numerically
identical (it drops the CPU actor replica, so the PPO ratio stops being computed
across two devices). It is not the mode to run a 120k-episode job in.

### Where the wall actually is

A `--workers 11` update cycle at 64 episodes is ~66 s: ~13 s of rollout (hidden),
**~38 s of parent-side feature rebuild**, ~22 s of collation inside the update.
Both of the last two are the same `transform` cost, paid in the one process that
cannot be parallelised away. Two things were measured and one was shipped:

- **Shipped: `card_type_flags` looks its tables up lazily.** It built all sixteen
  per-card lookups and then discarded the ones a restricted `fields` had not asked
  for — 41 µs a call, 32 calls a decision. Verified identical against the frozen
  generation's own copy.
- **Shipped: the PPO update caches its collated minibatches** across epochs
  (`--no-cache-collate` restores the old behaviour). Collation is deterministic in
  the samples, so recollating per epoch was paying 22 s repeatedly for the same
  tensors; a reused epoch measured **7.3 s against 22.5 s**. The price is that a
  minibatch's membership is fixed for the update (only the visiting order
  reshuffles). It only engages when more than one epoch actually runs — with
  `--target-kl 0.03` binding, most updates stop after one.
- **Measured and rejected: shipping built features instead of rebuilding them.**
  A worker can hand collated trees to the parent through `torch.multiprocessing`
  shared memory, but it moves at 147 MB/s (1.2 ms per tensor, and a 1536-transition
  chunk is 7,836 tensors) — 9.4 s where rebuilding the same chunk costs 9.2 s of
  transform plus 5.4 s of collate. Not worth the shared-memory failure modes for
  a third off one term.

The two levers left, neither taken: **memoise a row's chain-entry features within
an episode** (row *j* is re-featurised in every later decision's chain, which is
the whole quadratic term — the largest win available anywhere in this pipeline,
and the riskiest, since it must reproduce the features bit-for-bit or every
checkpoint is invalidated), or **let the update run one batch stale** so the
rebuild overlaps it (max(38, 22) instead of 38+22, ~1.6×, at the cost of the
strict on-policy property the rollout pool is built around).

## Search ablation — what actually helped

Value net held constant (`value_scalar.pt`), 128 simulations, 300 games per row
against the frozen field:

| configuration | win rate | 95% CI |
|---|---|---|
| as measured (random setup, uniform prior) | 21.3% | [17.1, 26.3] |
| + BC setup phase | 19.0% | [15.0, 23.8] |
| + heuristic prior | 20.0% | [15.9, 24.9] |
| **+ BC root prior** | **29.8%** | **[24.9, 35.2]** |

Two of the three fixes I predicted would help did nothing measurable, and the
setup fix was mildly negative. Only the BC root prior moved: **+8.5 points**
over baseline (z = 2.38, p ≈ 0.017), enough to clear the old PPO number of
28.3%.

**But it is still 11 points below BC alone at 41.1%.** That is the conclusion
that matters: the search, seeded with BC's policy, then *degrades* it. With
explained variance of 0.29 and roughly two plies of lookahead, the search's Q
estimates are worse than the BC policy's own judgement, so every decision where
the search overrides BC costs more than it gains.

This reframes the remaining options. The search does not need more tuning of the
kind tried here; it needs either a leaf evaluator good enough to be worth
listening to, or a rule that only lets it deviate from the prior on strong
evidence (the `c_puct` axis of the sweep is a direct test — high `c_puct` keeps
the search near BC). If neither closes an 11-point gap, the honest answer is
that IS-MCTS is not the right tool at this value-function quality, and the
compute is better spent fine-tuning BC directly.

## CFR vs PUCT

The selection rule swapped head-to-head, everything else held constant, 300
games each:

| selector | win rate | 95% CI |
|---|---|---|
| PUCT | 23.0% | [18.6, 28.1] |
| regret matching | 20.3% | [16.2, 25.2] |

**Regret matching is not better.** Slightly worse, well inside overlapping
intervals.

The limit of this test, stated precisely: it is regret matching over *estimated*
action values, not outcome-sampling MCCFR, so it is not a verdict on CFR proper.
But it is consistent with the diagnosis — CFR changes *how the search selects*,
not *how well it evaluates*, and evaluation quality is what binds here. A
faithful MCCFR would meet the same wall.

## Conclusion on search

Three independent experiments at n=300 each put IS-MCTS at **25–30%** against a
**41.1%** BC baseline. The search, seeded with BC's own policy, then degrades
it. The mechanism is established rather than guessed:

- **Evaluation.** Explained variance 0.29 means the leaf estimates are worse
  than the BC policy's judgement, so every decision where the search overrides
  BC loses more than it gains.
- **Depth.** MCTS depth grows like log_b(N). At ~11 root options, 128
  simulations buys about two plies. Extra simulations widen the tree rather than
  deepen it — which is exactly why the search scaled cleanly against a random
  opponent (64.0% → 81.3%) and not at all against real ones.

Neither a policy prior, a better-conditioned value function, a setup-phase fix,
nor an equilibrium-seeking selector closed the gap. **Recommendation: stop
investing in search at this value-function quality.** It would need a leaf
evaluator good enough to be worth listening to — realistically a value head on
the full policy trunk, not 30 scalars — before the search is worth its compute.

## Status of the search itself

Against a *random* opponent the search scales cleanly: 64.0% → 80.0% → 81.3%
over 4/16/64 simulations (n=150 each, p≈0.002 for the first step). Against the
frozen BC agents it is flat: 21.7% / 21.7% / 16.7% over 16/64/256 (n=60).

That dissociation is the central open problem. Two contributing causes are
identified and fixed; results pending:

- **Depth.** MCTS depth grows like log_b(N); at ~11 root options and 128
  simulations that is ~2 plies. Extra simulations were *widening* the tree, not
  deepening it. A heuristic prior (ATTACK 47%, ATTACH 33%, PLAY 16%, END 4%) and
  a BC root prior both concentrate the budget.
- **Setup played at random.** The agent was choosing its opening active and
  bench — roughly ten of the most consequential decisions in the game —
  uniformly at random, because `search_begin` has no parameter for your own
  face-down active. Now uses a BC policy.

A regret-matching selector is implemented as the CFR alternative. It is regret
matching over *estimated* action values rather than outcome-sampling MCCFR, so
the sampled-subgame equilibrium guarantee does not strictly carry over; it is
swept head-to-head against PUCT rather than adopted on faith.

---

# Recommendation

**Ship the warm-started PPO line. Stop the search line.**

## What to run

```bash
./run_warmstart_resume.sh     # or run_warmstart_ppo.sh from scratch
```

The single change that mattered was mechanical: `--init-from` never worked
because the challenger's `CardEmbed` gained ability fields and widened `proj`
from 151 to 193 inputs, so `bc_policy.pt` could not load. Every run in
`checkpoints/` started from random init by accident. `--actor-arch crustle`
builds the actor from the generation the checkpoint fits, and training starts at
~41% instead of 0%.

Supporting fixes, all necessary for the warm start to survive PPO:

- critic output layer zeroed, so `advantage = R` at init rather than `R` minus a
  random head's output
- stop logit biased to -4.0, so a trained option head is not fighting an
  untrained stop inside the same softmax
- lr 5e-5, entropy 0.005, target KL 0.015 — all lower than the from-scratch
  settings. From random init the risk is never learning; from a good warm start
  the risk is destroying it.

## What not to bother with

| tried | result |
|---|---|
| IS-MCTS as the primary policy | 25–30% vs BC's 41.1% — actively harmful |
| CFR-style regret matching selector | 20.3% vs PUCT's 23.0% — no better |
| BC policy for the setup phase | 19.0% vs 21.3% — no help |
| Heuristic action prior | 20.0% vs 21.3% — no help |
| Rollout leaf evaluation | SNR 0.36; a value net is a precondition, not an optimisation |

Both suggestions from the user measured **positive but small**: prize-margin
auxiliary head +1.1% relative explained variance, lethality features +0.010 EV.
Both are kept in the code; neither changes the agent's strength.

## If the search is revisited

The belief sampler (`belief.py`) is fully validated and reusable — 22,443/22,443
frames with the true hand in support, 88,592/88,592 particles accepted by the
engine. The engine probes (`probe_search_api.py`) are a permanent reference. The
blocker is leaf-evaluation quality, so the first thing to try is a value head on
the full `PolicyNetwork` trunk rather than 30 scalars, accepting the ~600x cost
increase and compensating with far fewer simulations.

## Open / unfinished

- The 53.3% eval is n=60 (±12.6 points) and overlaps the 41.1% baseline. The
  overnight run produces more evals plus a 150-episode final measurement.
- `value_data_selfplay.npz` holds the 24-feature vectors; the 30-feature set was
  only validated on the replay corpus. Regenerating would let the lethality
  features into the on-distribution model.
- `partition_hidden` is still uniform. This is bias, not variance — see above.

## The policy head ranks every action category in one softmax (2026-08-31)

The head is a pointer network over the legal option list, not a fixed action space, so
there are no per-category heads to fix — the "categories" (attach / play a trainer /
attack / retreat / end turn) are just `OptionType`s of options competing in one masked
softmax. That is the right shape for a dynamic action space. What was missing was any way
for the model to hold an opinion about a *category* separately from an opinion about which
option within it, and any grounding for half the options that name a Pokemon.

### The measurement that separates the two failures

`bc_train.evaluate` now returns a per-`OptionType` breakdown, keyed on the category the
expert chose, and reports two rates: `same_play` (the usual tie-aware match) and
`right_category` (did the model pick an option of the expert's category at all).
`eval_checkpoint.py` prints it; `bc_train.py` prints and logs it every eval.

`bc_policy_grimmsnarl_d128_L54_3ep.pt`, `grimmsnarl_all.parquet`, holdout 0.1 seed 0,
7,680 frames — val_exact 61.7%, val_same_play 67.8% in aggregate:

| expert chose | n | same_play | right_category |
|---|---|---|---|
| CARD | 2945 | 60.9% | **100.0%** |
| PLAY | 1432 | 67.0% | 80.1% |
| ABILITY | 879 | 80.0% | 83.6% |
| NUMBER | 464 | 100.0% | 100.0% |
| EVOLVE | 402 | 71.4% | 75.6% |
| ATTACK | 367 | 59.4% | **59.4%** |
| ATTACH | 333 | 56.2% | **60.7%** |
| YES | 265 | 100.0% | 100.0% |
| RETREAT | 90 | 3.3% | **3.3%** |
| ENERGY | 80 | 100.0% | 100.0% |
| END | 80 | 55.0% | **55.0%** |

The aggregate hides two unrelated problems that need different fixes:

**Category errors.** For ATTACK, RETREAT and END, `same_play == right_category` exactly —
*every* error is a category error, not a targeting one. RETREAT is the extreme: the pilot
retreats on 90 of these frames and the clone ranks a RETREAT option top on 3 of them. A
single shared `score` MLP had to express "how attractive is retreating at all" as an
emergent property of the same weights that pick which Pokemon to attach to.

**Targeting errors.** CARD is the mirror image: `right_category` 100% (a CARD
sub-selection usually offers nothing but CARD options, so there is no category to get
wrong) and `same_play` 60.9%. Pure target confusion, in the largest bucket — 45.4% of
everything the expert picks.

### Fix 1: the board addressing missed the options that name a Pokemon by `area`/`index`

`--board-tokens` addressed only `in_play_area`/`in_play_index`, which is ATTACH and EVOLVE.
An option can also name a Pokemon through `area`/`index` when that area is ACTIVE or
BENCH — that is how CARD, ABILITY, ENERGY, ENERGY_CARD and TOOL_CARD point at a target.
Over 29,021 corpus decisions:

| naming | option slots |
|---|---|
| `in_play_*` (addressed before) | 35,740 — ATTACH 28,844, EVOLVE 6,896 |
| `area`/`index` (**not** addressed) | 31,830 — CARD 26,538, ABILITY 4,445, ENERGY 779, other 68 |

47% of Pokemon-naming option slots were collapsed onto the single `NO_SLOT` address —
exactly the blindness `board_tokens` exists to remove, for nearly half the cases, and
concentrated in the category with the worst targeting rate. Side matters too: CARD options
name the opponent's board on 14,366 of 26,538 such slots (54%), and no option addressed
under the old scheme ever pointed at the opponent, so half of `slot_embed` was dead.

`PolicyNetwork._named_slot` now reads both pointer pairs (`in_play_*` wins where both are
set, since that pair is the target while `area`/`index` is then the source card). On real
batches this takes addressed slots from 191 to 514 per 1,319, and every address the old
scheme assigned is unchanged.

**This is not backward-compatible in effect, only in shape.** Evaluating the existing
checkpoint under the wider addressing *drops* it to val_exact 60.1% / CARD same_play 58.5%,
because its `slot_embed` rows for the newly-addressed slots and for the whole opponent side
were never trained. Hence `--reset-slot-embed`, which re-initialises that one table on a
warm start and keeps the rest.

### Fix 2: `--type-conditioned` gives the scorer a category axis

Two additions, both zero-initialised so a warm start reproduces its source exactly
(verified: max |logit diff| 0.0 on valid options at init, and all new params receive
gradient on the first backward):

- `type_bias`: one free logit level per `OptionType`, 17 numbers, added to the logit. The
  quantity RETREAT's 3.3% needs and that nothing in the network held explicitly.
- `select_film`: FiLM the scoring query on `select_type` + `select_context`. These reached
  the scorer only as one of seven concatenated blocks diluted into the fused state, so a
  MAIN menu and a "discard exactly two" prompt were ranked by the same function.

If category errors survive both, the next step is a factorised head —
`p(category | state) x p(option | category, state)` as two masked softmaxes, which
decomposes the log-likelihood exactly and keeps the pointer dynamic. Not worth its
complexity until the cheap version is measured.

### Fix 3: three hand-written copies of the forward pass (a live train/serve bug)

`main.py`, `make_submission.py` and `train.py` (PPO's `ActorCritic.fused`) each re-derived
the forward pass by hand to reach the fused state for their `value`/`stop` heads. All three
predate `_address_board`, so **a `board_tokens` checkpoint was served, and PPO-trained,
with its board addressing silently disabled** — scoring options against a mean-pooled bench
as if the flag were off. Nothing errors; the weights load and the agent is quietly weaker
than the one that was measured. `--type-conditioned` would have broken identically.

The pass now lives once, as `PolicyNetwork.encode` + `score_options` (composed by
`logits_and_state`, which is what `forward` calls). All three callers use them, keeping
their old hand-written path only as a fallback for the `*_frozen` modules that predate it.
Verified: serving and PPO logits match training logits to 3.6e-7, and PPO's stay finite on
masked positions as its docstring requires. `main.py` also reads `arch` from the
checkpoint's `.meta.json` sidecar now, like the packaged bundle does — a flag-only arch
difference does not fail a load, it just serves the wrong network.

### Retraining

Both fixes are aimed at measured failures, so run them together and read the per-category
table, not just `val_exact` — the two fixes move different rows and Fix 1 costs the
aggregate before it pays.

    conda activate kaggle-pokemon

    # warm start (cheap; slot_embed regrows, everything else inherited)
    python bc_train.py --parquet grimmsnarl_all.parquet \
        --init-from challenger/bc_policy_grimmsnarl_d128_L54_3ep.pt --reset-slot-embed \
        --board-tokens --type-conditioned \
        --dim 128 --nhead 8 --chain-layers 20 --hist-layers 20 --ctx-layers 6 \
        --cross-layers 8 --batch-size 16 --grad-accum 4 --lr 3e-4 \
        --warmup-steps 2000 --lr-schedule cosine --min-lr-frac 0.05 --epochs 3 \
        --holdout 0.1 --workers 6 --cached-row-groups 6 \
        --out challenger/bc_policy_grimmsnarl_typed.pt \
        --run-name bc-grimsnarl-typed-warm

    # from scratch (the clean measurement; drop --init-from/--reset-slot-embed)

Then compare per-category against the table above:

    python eval_checkpoint.py --checkpoint challenger/bc_policy_grimmsnarl_typed.pt \
        --parquet grimmsnarl_all.parquet --holdout 0.1 --seed 0

What each fix should move: Fix 1 -> CARD/ABILITY `same_play`. Fix 2 -> RETREAT, END,
ATTACK, ATTACH `right_category`. Neither should touch NUMBER/YES/ENERGY, already at 100%.
`probe_targeting.py` remains the direct test of the stacked-energy failure.
