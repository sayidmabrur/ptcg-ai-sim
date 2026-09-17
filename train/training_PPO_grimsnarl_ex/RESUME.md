# Resume here

Everything is stopped and saved. Target: **>=60%** against the three frozen BC
agents, achieved through RL.

## State

| agent | vs frozen field | n | how measured |
|---|---|---|---|
| uniform random | 8.3% | 180 | arena |
| BC checkpoint on our deck | 41.1% | 1796 | arena |
| PPO, no decode floor | 40.0% [34.0, 46.3] | 240 | arena |
| **PPO, decode floor (current bundle)** | **42.7% [36.6, 49.0]** | 239 | arena |
| old from-scratch PPO (superseded) | 28.3% | 60 | train.py eval |

**Use `arena.py` for every number.** `train.py`'s in-training eval overstated by
~12 points at n=120 and was acted on once; per-opponent rates need n>=300 (the
same weights produced alakazam at 21.2% and 32.5% on two n=80 samples).

## Checkpoints

```
checkpoints/preserved/eval0.517-ep9232.pt   best known weights (train.py eval 51.7%,
                                            arena 42.7% floored) - weights only,
                                            --resume works, Adam moments restart
checkpoints/ppo-exploit/latest.pt           ep 11792, first run with attack shaping
checkpoints/ppo-exploit/best.pt             ep 10896 (36.7%) - NOT the best; see below
checkpoints/warmstart-crustle/latest.pt     ep 6096, the pre-exploit warm start
```

The preserved file exists because `best.pt` got clobbered: every restart reset
`best_eval` to `-inf`, so the first eval of a new run overwrote `best.pt`
whatever it scored. Those weights survived only inside the packaged submission.
**Fixed** — `train.py` now carries `best_eval` across `--resume`, and tolerates a
checkpoint with no optimizer state.

## Submission

`submission_ppo/` — shaped like `crustle_frozen/`: `cg/`, `policy_network/`
(crustle's generation), `deck.csv` (the **challenger** list), `actor_critic.pt`,
`main.py` with `def agent(obs_dict) -> list[int]`. Validated end-to-end through
the arena at n=239, zero fallbacks.

```bash
PY=/home/kangh/miniconda3/envs/kaggle-pokemon/bin/python
$PY make_submission.py --checkpoint checkpoints/preserved/eval0.517-ep9232.pt
```

Three things the packager gets right that are easy to break: the decode must be
`rollout_action(greedy=True)` with the learned stop logit (not
`decode_action`), the architecture must be crustle's generation, and the deck
must be the challenger's. A fourth was found by validating: the bundle loads its
own `policy_experimental` by explicit path, because a bare import resolved to
the challenger's copy and the agent silently played **random moves** while
reporting no error.

## Restart training

```bash
nohup ./run_ppo_exploit.sh > exploit5.log 2>&1 &
```

Edit `RESUME=` in that script to pick the starting checkpoint. Current settings:
lr 1.5e-4, entropy 0.001, target-kl 0.03, ppo-epochs 4, episodes-per-update 64,
shaping-coef 1.5, **attack-shaping 0.5**, wandb online.

## Open work, in priority order

1. **Re-audit behaviour after attack shaping.** `audit_behavior.py --games 30
   --opponent alakazam`. The check is whether `Rock Fighting Energy -> Crustle`
   disappears, not whether win rate moved — behaviour is measurable at n=24,
   win rate is not. The shaping had only ~900 episodes before shutdown, so it
   has not been evaluated at all yet.
2. **Re-measure through arena at n>=300/opponent** once the shaping has run.
3. **Is the challenger deck simply weaker?** `frozen:crustle@challenger` scores
   34.7% against crustle's own deck with the *same policy* on both sides. That
   is confounded (the policy was cloned on crustle's list) but it is a real hint
   that part of the 41% -> 60% gap is not the policy's fault. Isolating it
   properly would say whether 60% is reachable with this decklist.
4. **Ogerpon / Alakazam.** Cornerstone Stance prevents *all* damage from Pokémon
   with an Ability — a hard counter — but Demolish needs the deck's single
   `{F}`. That conjunction (draw Ogerpon, draw Rock Fighting Energy, attach it
   right, stay active) is too rare for uniform exploration to find. If the
   shaping does not surface it, weight the opponent schedule toward alakazam.

## What was tried and did not work

Detailed in `FINDINGS.md`. Briefly: IS-MCTS reached 25-30% and *degrades* the
policy it is seeded with; a CFR-style regret-matching selector scored 20.3%
against PUCT's 23.0%. Both are blocked on the same thing — value estimation caps
near explained variance 0.30 whether the estimator is 30 hand-crafted scalars
(0.294) or a 2.4M-parameter full-board trunk (0.279) — so a search cannot beat
the policy's own judgement. The belief sampler (`belief.py`, validated
22,443/22,443 frames) and the engine probes (`probe_search_api.py`) are sound and
reusable if a better value function ever appears.
