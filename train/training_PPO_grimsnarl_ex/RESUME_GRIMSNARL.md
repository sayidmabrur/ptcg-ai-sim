# Marnie's Grimmsnarl ex — what is running and how to read it

Launched overnight 2026-08-10 ~22:0x by `./run_overnight_grimsnarl.sh`, log
`overnight_grimsnarl.log`. Two stages, chained: clone a pilot, then PPO from it.

```bash
grep -E '^\[bc\].*val_exact' overnight_grimsnarl.log | tail    # stage 1 climbing?
grep -E '^\[eval'            overnight_grimsnarl.log | tail    # stage 2 vs frozen field
grep -E '^\[update'          overnight_grimsnarl.log | tail    # ev / entropy / kl health
grep -E '^\[wandb'           overnight_grimsnarl.log           # both run URLs
```

## wandb

Both stages log to project `pokemon-tcg-rl`, as two runs (separate because the metrics have
nothing in common; the URL of each is printed to the log at startup):

| run | series |
|---|---|
`bc-grimsnarl-ex` | `bc/train_loss`, `bc/val_loss`, **`bc/val_exact_match`**, `bc/best_val_exact_match`, `bc/frames_per_second`, `bc/epoch`, `bc/lr`; summary carries the best exact match, step count and wall-clock |
`ppo-grimsnarl-ex` | per episode `episode/{reward,win,draw,steps,decisions,seat,opponent}` and `train/win_rate*`; per update `loss/*` (policy, value, entropy, approx_kl, clip_fraction, grad_norm, explained_variance, degenerate_batch, epochs_run) plus `train/{update,transitions,shaping_per_episode,episodes_per_hour,update_seconds,lr}`; **`eval/*` every 25 updates** (per opponent, split by seat, and `eval/win_rate`), `eval/best_win_rate` in the summary, and `final/*` at the end |

`bc/val_exact_match` is the stage-1 number to watch (fraction of held-out frames where the
decoded selection matches the expert exactly) and `eval/win_rate` is the stage-2 one.

Stage 1 is ~4.3 h (4 epochs × 799k frames at ~208 frames/s); stage 2 takes the rest at
~2500–2800 ep/h. If stage 1 produced no checkpoint the script refuses to start stage 2
rather than run PPO from nothing.

## ARCHITECTURE A/B (2026-08-12) — cross-attention is win-rate neutral

Both clones packaged as bundles and played against the frozen field at **n=300 per opponent,
900 per policy** (`arena.versus_field`, 11 workers, seed 1):

| bundle | exact match | crustle | alakazam | lucario | overall |
|---|---|---|---|---|---|
| `submission_bc_grimsnarl` (mean-pooled chain) | 73.4% | 46.3% | 63.7% | 68.7% | **59.6%** [56.3, 62.7] |
| `submission_bc_xattn` (causal chain + option cross-attention) | **75.0%** | 45.7% | 69.0% | 65.0% | **59.9%** [56.7, 63.0] |

**Indistinguishable** — 3 games in 900. They trade matchups and land in the same place.

Two conclusions worth keeping, because both cost real time to learn:

*Imitation accuracy is not the objective.* Cross-attention fits the expert **better** by 1.6
points on 41,693 held-out frames — a difference far outside noise — and converts none of it
into win rate. Optimising `val_exact` further is not obviously progress.

*n=120 cannot resolve 5 points, and it fooled us.* The old clone measured 62.5% at n=120 and
**59.6% at n=900**; the cross-attention clone measured 57.5% at n=120 and 59.9% at n=900. The
apparent 5-point regression was entirely sampling error. The n=120 *bundle* validation (59.2%)
happened to be the accurate one. Any future architecture or reward comparison needs n>=900,
which costs ~14 minutes at 1.1 games/s over 11 workers — cheap enough that there is no excuse.

So the honest headline for this archetype: **the clone plays at ~59.7% [56-63] against the
frozen field**, from either architecture, and PPO has not yet improved on it.

## STAGE 1 RESULT (2026-08-11 02:48) — the clone already clears the target

`challenger/bc_policy.pt`, step 42,000, **73.4% held-out exact match** (converged: 72.9 /
73.4 / 72.9 / 73.3 / 73.0 over the last five evals; 3.5 h; beats the crustle clone's ~71%).

Played greedily against the frozen field at n=120 — the same harness and sample size stage 2's
own `eval/win_rate` uses, so the numbers are directly comparable:

| opponent | seat 0 | seat 1 | mean |
|---|---|---|---|
| crustle | 50.0% | 45.0% | 47.5% |
| alakazam | 80.0% | 50.0% | 65.0% |
| lucario | 80.0% | 70.0% | 75.0% |
| **overall** | 70.0% | 55.0% | **62.5%** |

**This is the bar PPO has to beat, and it is high.** Two things to keep in view when reading
it. First, a large part of it is the *deck*: this is a top-meta list against three older
archetypes piloted by their own clones, so 62.5% is "Grimmsnarl + competent pilot" versus
"crustle/alakazam/lucario + competent pilot", not a pure policy comparison — the crustle clone
scores 52.8% on its own deck by the same measurement. Second, n=120 carries roughly ±9 points,
and this repository has already seen a 51.7% eval re-measure at 27.8%.

The consequence for stage 2 is that the risk is no longer "PPO fails to reach the target" but
**"PPO destroys a policy that already clears it"** — which is the documented failure mode
here: the crustle run went 52.8% → 41.7% by update 200. `--save-every 10` and `best.pt`
selection protect the clone, but the run should be stopped if the evals trend below 62.5%.

## Earlier read (2026-08-11 00:30, stage 1 mid-flight)

Stage 1 is doing its job. Held-out exact match climbed monotonically with the loss falling
alongside it — no overfitting yet at epoch 1 of 4:

| step | 2000 | 4000 | 6000 | 8000 | 10000 | 12000 | 14000 |
|---|---|---|---|---|---|---|---|
| `val_exact` | 62.7% | 63.5% | 65.6% | 66.5% | 67.4% | 67.8% | **68.5%** |
| `val_loss` | 1.059 | 1.011 | 0.954 | 0.946 | 0.911 | 0.901 | **0.879** |

And the number that matters — the step-14,000 checkpoint played greedily against the frozen
field scores **36.1%** (n=36, so ±14: a rough read, taken early on purpose to see whether
cloning was working at all). Against the table below, that is the whole point of the stage:
the archetype went from having *no* usable start (12.5%) to a competent one, with three
epochs left to run. For reference the crustle clone plateaued near 71% exact match, so this
one is not yet converged.

## The measurement that set this up

Everything against the three frozen BC agents, greedy, seats alternated, n=72 each:

| policy | vs frozen field |
|---|---|
| random | 5.5% |
| crustle BC checkpoint piloting the **Grimmsnarl** deck | **12.5%** |
| crustle BC checkpoint piloting **its own** deck | 52.8% |

That 12.5% is why the chain has a BC stage. There was no warm start for this archetype — a
cloned policy handed a decklist it never saw the expert play is barely above random, and
FINDINGS.md records what PPO from a cold start does here: 30k episodes, ~8 h, 28.3%, below
the 41.1% BC bar it was chasing. Stage 1 removes that handicap using the 841,016 frames in
`../dataset/2026-08-06-10-marnie-grimmsnarl-ex-cleaned.parquet` where a real ladder pilot of
this exact 60-card list chose an action. First smoke run reached 54.8% exact match on
held-out *games* in 100 steps.

## What changed in the algorithm

| | before (crustle runs) | now | why |
|---|---|---|---|
`--gamma` / `--gae-lambda` | 1.0 / 1.0 | **0.997 / 0.95** | 1.0/1.0 collapses GAE to `advantage = R - V(s)` — pure Monte-Carlo, **no bootstrapping at all**, one bit of signal spread over ~100 decisions with every decision in a won game credited identically |
reward | ±1 | **`--reward winprob`** (1/0) | `V(s)` now regresses onto `P(win \| s)`, so it reads as a position evaluation; the value head's bias starts at the 0.5 prior instead of 0, which a 0/1 label would otherwise make confidently wrong on update 1 |
`--shaping-coef` | 1.5 | **0.5** | PBRS telescopes to `coef × prize_margin`, so 1.5 reaches ±1.5 against a ±1 terminal — shaping *dominated* the win signal it is meant to be subordinate to |
`--attack-shaping` | 0.5 | 0.3 | same reason, kept small |
chain / history | 60 | **30 for the learner**, 60 for the frozen agents | `observation.LEARNER_SPEC`. Halves the quadratic term in `transform`. The frozen agents keep 60 because they are the yardstick every number in FINDINGS.md is measured against — truncating their input moves the benchmark, not the policy |
stop-logit bias on warm start | only for `arch != challenger` | **any `--init-from`** | a trained option head against a random stop logit corrupts every optional/multi-select decision from the first rollout; this warm start is the challenger arch, so the old guard did not fire |

## Reading the result honestly

**The number that matters is `[eval] ... mean`**, and the first thing to compare it against
is stage 2's own first eval — i.e. did PPO improve on the policy it started from. That is the
question "beat the behaviour-cloned model" actually asks.

**Do not trust a single eval.** At `--eval-episodes 40` (120 games) the sampling error here
is ±10 points, and this is not theoretical: `checkpoints/preserved/eval0.517-ep9232.pt` was
recorded at 51.7% and re-measures at **27.8%** on a fresh n=72. A checkpoint was selected on
that noise once already. Two consecutive evals agreeing is the weakest evidence worth acting
on; if a number needs to be trusted, raise `--eval-episodes`.

**A 55–60% target is not forecastable from here.** The frozen field is a different opposition
from the live ladder the corpus pilot won 46% against, so those two numbers are not
comparable, and no run in this repository has yet cleared 52.8% on a re-measured sample. What
this chain establishes is whether PPO improves on a *competent* start — which has never been
tested here, because there has never been a competent start for an archetype PPO was training
on.

## Still open, deliberately not in this run

* **Belief features** (`belief_features.py`, verified: accounting closes on 983/983 frames,
  and at all 99 deck-search frames the hidden pool equals the prize multiset exactly). The
  policy still gets `deck_count` and `prize_count` as two scalars and nothing else about
  hidden information — `belief.py` calls recovering the rest "a sparse-recovery problem over
  a 1267-card vocab which it cannot solve and does not have to". The module computes it;
  wiring it into `transform_player_state` and widening `PlayerStateEncoder` is not done.
  One loose end when it is: the latched prize multiset exceeded the remaining prize slots on
  8 of 865 frames (0.9%) — the clamp is against `remaining`, which grows again when the deck
  is shuffled back; clamping the total to `prize_count` as well would close it.
* **The corpus's `final_result` column is unpopulated** (`-1` on all 1,355,855 rows). Any
  label must come from `matches.parquet` (`reward0`/`reward1`; `player_index` maps to
  `player0`/`player1`, verified on all 8,436 episodes by name). `make_value_data.py` does
  this correctly — reading `final_result` instead labels every frame a loss and fits
  perfectly, which no loss curve reveals.
* `make_value_data.py` + `train_value.py` are ready if the separate scalar value net is
  wanted later; the run above uses PPO's own critic with GAE bootstrapping instead.
