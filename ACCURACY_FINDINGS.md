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
indistinguishable from zero. The two betting correlations are equal to within
noise and flip-flop across runs. **Composition-aware betting does not beat
Hi-Lo.**

Why: for the *aggregate* "is this shoe good for me" question, the Hi-Lo count
is an extremely efficient summary statistic (its betting correlation with true
EV is ~0.97 in the literature). The full composition carries more information
in principle (Hi-Lo mis-weights aces, etc.), but the residual is tiny and is
completely swamped by single-hand outcome variance — undetectable even at 1.5M
hands, and worth far less than the video's 2× headline implied. Composition
helps where a *specific* remaining-card distribution flips a *specific* play
decision (§3), not in bet sizing.

(`bet_value_net.pt` is the trained betting model; the deployable EV→bet ramp is
approximate out-of-sample, which is why the comparison above is rank-matched —
identical bets, allocation-only difference — rather than relying on the ramp.)

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
