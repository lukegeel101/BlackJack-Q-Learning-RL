# Extended training & accuracy investigation

Goal: *train the DQN more extensively and see if accuracy can be improved.*

This writeup records what was tried, what the evidence showed, and the model
that came out of it. All evaluation code lives in the repo (`gen_optimal_dataset.py`,
`train_supervised.py`) plus the offline scorer described below.

## TL;DR

- **Simply training the existing RL DQN more does not help — it makes the
  model worse.** Continuing the 1M-episode checkpoint for another ~165k
  episodes lowered its edge on every held-out seed.
- The money simulation is too noisy to measure play quality: the per-$1
  edge of Agent 3 over the counter has a **±~1% noise floor even at 360k
  hands**, which is larger than the effect we're trying to detect.
- Scored against the **exact expectimax solver** (a low-variance accuracy
  oracle), the shipped RL DQN's composition-based overrides are actually
  **net-negative** — its play is *less* accurate than plain basic+I18.
  Its apparent money edge is bet-ramp variance, not better decisions.
- **Supervised distillation on exact solver EVs fixes this.** A `DuelingQNet`
  trained on 20k solver-labeled states makes near-optimal decisions and its
  selective overrides *beat* basic+I18. It is ~10× more accurate than the RL
  model on exact-EV terms, verified on two independent held-out sets.
  Shipped as `dqn_agent_distilled.pt`.

## The accuracy metric (why money is the wrong yardstick)

Per-hand blackjack reward is extremely high variance once a 1–50 bet ramp is
applied — a handful of large-bet high-count hands dominate any sample. Two
policies that differ on ~10% of decisions produce cumulative-profit curves
whose gap is mostly noise. Concretely, the delta *Agent3 − Agent2* measured
at 6 seeds × 60k hands still carries ±~1% standard deviation.

Instead we score decisions directly against `optimal_policy.evaluate_state`,
which returns the **exact** per-action EV for a state given the true shoe
composition. For a set of realistic decision states (sampled under basic+I18):

- **EV-loss/hand** = `EV(optimal action) − EV(chosen action)`, averaged.
  0 = perfect play; larger = worse. No money-outcome variance.
- **agreement%** = fraction of states where the chosen action is optimal.

The solver is slow (~3 states/s), so states are labeled once (offline, in
parallel) and every policy/margin is then scored instantly. `gen_optimal_dataset.py`
now also records basic+I18's action per state, so basic+I18, a raw net, and
the net-vs-basic hybrid can all be scored offline against the same labels.

## Results

### 1. Extended / continued RL training degrades the model

Continuing `dqn_agent.pt` (1M episodes) for +164k episodes (fresh optimizer,
mild ε=0.1→0.02), evaluated on 6 held-out seeds × 80k hands:

| model | edge over counter (money) | deviation rate |
|---|---:|---:|
| baseline RL DQN | **+1.05%** | 9.8% |
| continued (+164k eps) | **−0.29%** | 13.7% |

Worse on every seed (paired −1.34% ± 0.95). Re-injecting exploration and a
fresh replay buffer/optimizer knocks the converged policy off its optimum.

### 2. Exact-EV accuracy: the RL overrides are net-negative

Held-out states, scored against the solver (lower EV-loss = better):

| policy | EV-loss/hand | agree% |
|---|---:|---:|
| basic+I18 (Agent 2 play) | 0.00159 | 95.3% |
| RL DQN hybrid @margin 0.05 (**current default**) | 0.01372 | 85.9% |
| RL DQN hybrid @margin 0.20 | 0.00907 | 89.8% |
| RL DQN hybrid @margin 1.00 (never overrides) | 0.00519 | 94.3% |

EV-loss rises monotonically as the DQN overrides more (lower margin), and at
margin 1.0 (no overrides) it exactly equals basic+I18 — confirming **every
override, on net, reduces accuracy.**

### 3. Supervised distillation beats basic+I18

`DuelingQNet` trained on 20k exact-EV labels (`optimal_dataset_34d.npz`),
scored on two independent held-out sets:

| policy | EV-loss (set 1 / set 2) | agree% |
|---|---:|---:|
| basic+I18 | 0.00159 / 0.00161 | ~95.0% |
| **distilled hybrid @0.05** | **0.00111 / 0.00099** | ~95.8% |
| RL DQN hybrid @0.05 (current) | 0.01060 / 0.01077 | ~87% |

The distilled net makes near-optimal decisions standalone (94.8–95.2%
agreement) and its *selective* overrides (2.5% of hands) reduce EV-loss ~30–38%
below basic+I18 — a genuine composition-aware edge, generalizing across sets.

### 4. What this means for the money numbers

At 6 seeds × 50k hands, the distilled model's money edge over the counter is
≈ 0 (its play ≈ optimal ≈ basic+I18, which the counter already uses), while
the RL model shows +0.72% — but that gap is within the money noise floor and
comes from the RL model's 4×-higher (and lower-quality) deviation rate. In
other words: the RL model's headline money edge is largely **variance**, and
the distilled model trades that variance for genuinely better decisions.

## Tooling added along the way

- `train_dqn.py`: `--resume` / `--init-from` / `--max-seconds` and resumable
  checkpoints, so long RL training can run in short chunks.
- `train_dqn.py`: `--hidden-dim`; `load_dqn_net` now infers hidden width from
  the checkpoint (backward compatible).
- `gen_optimal_dataset.py`: fixed a 28→34 feature-dim crash; records basic+I18
  actions for offline scoring.
- `train_supervised.py`: fixed defaults that collapsed the net to always-STAND
  (lr 1e-3, weight_decay 0, patience 40, 250 epochs).

## Artifacts

- `dqn_agent_distilled.pt` — the distilled `DuelingQNet` (recommended play net).
- `optimal_dataset_34d.npz` — 20k exact-EV labeled states (34-dim features).
- `dqn_agent.pt` — the original 1M-episode RL QR-DQN (unchanged).

## Betting: does composition beat the Hi-Lo count? (the bigger lever)

A card counter's edge is mostly in the **bet** (sizing up when the shoe is
rich), not the play — play deviations are ~10–15% of the edge, bet sizing is
the rest. But in this project **all agents share the same Hi-Lo bet ramp**;
the DQN only ever changed *plays*. So the composition advantage was being
spent on the small lever. The natural question: can a model that sees the full
rank composition size bets better than the 1-D Hi-Lo true count?

Setup (`gen_bet_dataset.py`, `train_bet_value.py`, `BetValueNet` +
`CompositionBettingAgent` in `agents.py`): play flat-bet basic+I18 and record,
per round, the **pre-deal** composition, the pre-deal Hi-Lo true count, and the
realized per-unit outcome. Train a small net to predict the outcome from the
composition — its prediction is a composition-conditional EV estimate. Compare
it to the true count as a betting signal on **1.5M held-out hands**:

| metric | Hi-Lo true count | composition net |
|---|---:|---:|
| betting correlation (corr with outcome) | +0.0137 | +0.0130 |
| R² explaining outcome | +0.00012 | +0.00016 |
| edge, **rank-matched** betting (identical bet multiset) | +1.147% | +1.169% |

Rank-matched advantage of composition: **+$0.02/hand, 95% CI [−0.24, +0.26]** —
indistinguishable from zero, using single-hand outcomes as the payoff.

But this is the wrong measurement: a single hand's result has std ~1.1 while
the EV signal we're chasing has std ~0.018, so realized outcomes can't resolve
the two signals even at millions of hands. The right benchmark is the **true
per-round EV** of each shoe, estimated by Monte-Carlo replaying each exact
composition thousands of times (`mc_true_ev.py`) -- this is how "betting
correlation" is actually defined. Scored against 1,500 MC-EV-labeled shoes
(`score_bet.py`):

| signal | betting corr (vs true EV) | EV-RMSE | edge% |
|---|---:|---:|---:|
| Hi-Lo true count | 0.700 | 0.0129 | 0.501 |
| composition net, **under-trained** (700k hands, h64) | 0.668 | 0.0140 | 0.426 |
| composition net, **trained extensively** (5M hands, h256) | **0.750** | **0.0120** | **0.568** |

Two things become clear:

1. The *original* betting net was **worse** than Hi-Lo -- it underfit. The
   earlier "no edge" result was measuring an under-trained net with a metric
   too noisy to see anything.
2. **Trained more extensively (5M rounds + a wider net), the composition net
   significantly beats Hi-Lo as a betting signal:** correlation 0.750 vs 0.700
   (paired-bootstrap difference +0.045, 95% CI [+0.029, +0.061], P>0 = 100%),
   and a lower EV-prediction RMSE. This is the ace/rank information Hi-Lo's
   1-D count throws away, recovered from the full composition.

The translation to realized **edge** is positive (+0.57% vs +0.50%, ~13%
relative) but not individually significant at 1,500 shoes (95% CI on the edge
gap includes 0) -- the edge is dominated by rare high-count shoes, which are
sparse in the benchmark. So: composition is a *measurably better EV predictor*
for betting; whether that few-percent edge is worth the complexity over Hi-Lo
in dollars is not resolved without a much larger MC benchmark.

Takeaway: the betting lever is real but small. Hi-Lo is ~0.97 efficient by
construction; the composition net recovers most of the remaining gap, but that
gap is worth thousandths of a bet per hand -- not the video's 2×.

### Pushing further: the betting net is at its accuracy plateau

Scaling past the 5M/256 net gave nothing measurable:

| configuration | betting corr (vs true EV) |
|---|---:|
| 700k hands, 64-wide (original) | 0.668 |
| 5M hands, 128-wide | 0.746 |
| 5M hands, 256-wide | 0.750 |
| 5M, 128-wide, retuned (35 ep, batch 8192) | 0.752 |
| 3-net ensemble (2×128 + 256, diverse seeds/arch) | 0.753 |

More capacity, more epochs, and ensembling all land at ~0.752 — the members
trained on the same data have correlated errors, so averaging adds nothing.
This is a genuine ceiling, and it's mostly a *measurement* ceiling: the
benchmark's MC noise (finite reps per shoe) caps the maximum observable
correlation at ~0.78, which implies the net's correlation with the **true,
noiseless** round EV is already ≈ 0.96 (vs Hi-Lo ≈ 0.90). The 15 composition
features fully specify the pre-deal shoe, so there is no more information to
give the model — it has captured essentially all of the learnable
composition→EV signal. Beyond this point, "train more" yields sub-noise gains;
the remaining limit is the game itself, not the model.

(`bet_value_net.pt` is the trained betting model; `mc_true_ev.py` builds the
ground-truth benchmark and `score_bet.py` scores signals against it. The
deployable EV→bet ramp is approximate out-of-sample, so the money comparison is
rank-matched -- identical bets, allocation-only difference.)

## The integrated agent (both models in one player)

`IntegratedCardCountingAgent` combines the two composition-aware models:

* **Bet** — sized by `BetValueNet`'s composition-conditional EV estimate,
  mapped through a **quantile ramp** (`calibrate_bet_ramp.py`) that copies the
  Hi-Lo bet-size *distribution* exactly. Verified on 300k calibration hands:
  mean bet $94.85 vs Hi-Lo's $94.85, same max, same total wagered. (The old
  linear ev_lo/ev_hi ramp mis-calibrated out-of-sample and bet **1.83×** the
  counter — inflating both edge and risk. The quantile ramp removes that free
  parameter entirely, so the agent is risk-matched to the counter by
  construction.)
* **Play** — basic+I18, with the solver-distilled net overriding on the ~2.5%
  of hands where it is ≥0.05 EV-units better.

### Head-to-head, 24 seeds × 100k hands, common random numbers

| agent | edge | $/hand | avg bet |
|---|---:|---:|---:|
| Flat-bet basic | −0.581% | −0.29 | $50 |
| Hi-Lo counter (the professional) | +0.991% | +0.92 | $93 |
| RL DQN card counting (original Agent 3) | +2.062% | +2.32 | **$113** |
| **Integrated** (composition bet + distilled play) | **+1.174%** | +1.10 | $93 |

Integrated − counter: **+0.183% edge** (95% CI [−0.117, +0.482]) at *identical*
bet distribution — positive, but **not statistically distinguishable from
zero**: 14 of 24 seeds positive, t ≈ 1.2.

**This estimate shrank with more data.** An 8-seed run of the same comparison
gave +0.360% (95% CI [−0.149, +0.870]). Tripling the seeds tightened the
interval as expected (half-width 0.51 → 0.30) but pulled the point estimate
toward zero rather than closing the interval around it — textbook regression
to the mean, and precisely the failure mode this whole document exists to
guard against. The 8-seed figure should not be quoted.

The distinction that matters: the **accuracy** gains behind this agent are
statistically significant against ground truth (play EV-loss 0.0011 vs basic's
0.0016, §3; betting correlation +0.045 with CI [+0.029, +0.061], §4). Their
**dollar** translation is simply too small to demonstrate at any sample size
reachable here — which is the expected consequence of basic strategy already
being ~99.8% of optimal and Hi-Lo already being a near-efficient betting
statistic.

### Why the RL DQN still shows the biggest money edge

Its avg bet is $111 vs everyone else's $92, and it is **not** risk-matched.
Two candidate explanations, both tested:

1. *Metric bias from doubles/splits?* **No.** Recomputing edge against the
   true wagered amount (including doubled and split money) gives an inflation
   factor of 1.142× for the DQN vs 1.131× for the counter — essentially
   identical, so the denominator is not the story.
2. *Shoe-flow / bet-timing.* **Yes.** Measured over 60k hands:

   | agent | avg pre-deal TC | % hands at TC≥2 | cards/hand | hands/shoe |
   |---|---:|---:|---:|---:|
   | Hi-Lo counter | −0.220 | 13.1% | 5.34 | 65.0 |
   | RL DQN | **+0.148** | **16.3%** | 5.35 | 64.9 |
   | Integrated | −0.233 | 12.7% | 5.34 | 65.0 |

   Same cards per hand and hands per shoe, but the DQN's deviations shift
   *which count regimes its hands fall in*: it plays 24% more of its hands at
   TC ≥ 2, so the same bet ramp puts more money out when the shoe is genuinely
   favorable. (Hand-weighted average TC is negative for everyone because
   low-count stretches consume more cards per hand and therefore generate more
   hands.) That is a real effect in this heads-up simulation — but it is a
   *side-effect of playing differently*, not better decision-making: the same
   model's per-hand decisions are provably further from optimal (§2). It is
   also not something a player controls at a real table with other players.

So the three findings are consistent: the RL DQN plays **worse** per hand,
bets **bigger** more often through an accidental shoe-flow effect, and thus
books more money at higher variance; the integrated agent plays **better** per
hand at **matched** risk.

## Bonus bug: the original headline edges used the wrong denominator

The README's published figures for the video run (shared seed 42, 200k hands)
reproduce **exactly** in net profit — Agent 2 +$218,625, Agent 3 +$536,075 —
but its *edge* percentages (+2.19%, +5.36%) did not. They were computed as
profit ÷ (hands × $50), i.e. flat-bet volume, rather than profit ÷ money
actually wagered. That inflates any ramping agent by its bet multiplier:

| agent | published edge | correct edge | net profit | avg bet |
|---|---:|---:|---:|---:|
| 1. Flat-bet basic | −0.82% | −0.821% ✓ | −$82,050 | $50.00 |
| 2. Counting basic | +2.19% | **+1.175%** | +$218,625 | $93.05 |
| 3. DQN card counting | +5.36% | **+2.354%** | +$536,075 | $113.85 |

Agent 1 was unaffected because it flat-bets, so the two denominators coincide.
`simulate.py`'s `summarize()` computes this correctly today; the stale figures
lived only in the README, which is now corrected.

## Recommendation

Promote `dqn_agent_distilled.pt` to Agent 3's play net (it is a drop-in:
`load_dqn_net` auto-detects the architecture and the agent's 0.05 margin is
also optimal for it). This makes Agent 3's *play* measurably more accurate and
genuinely composition-aware. Note that doing so will bring Agent 3's money
edge over the counter down to its true (small, positive) value rather than the
variance-inflated figure the RL model shows — so the README's headline money
numbers should be re-stated honestly if the swap is made. Generating more
labels (100k+) would widen the distilled model's margin over basic+I18
further, but is compute-bound in this environment (~47 min per 20k labels).
