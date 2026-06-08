"""
Three blackjack agents for comparison.

  Agent 1: FlatBetBasicStrategyAgent
    Hardcoded multi-deck H17 basic strategy.
    Does NOT use the count. Flat bet every hand.
    Expected outcome: slow loss (~0.5%/hand house edge).

  Agent 2: CountingBasicStrategyAgent
    Hardcoded basic strategy + Illustrious-18 count deviations.
    Hi-Lo true count from the env; bet scales with true count.
    Expected outcome: slow profit.

  Agent 3: DQNCardCountingAgent
    Same bet ramp as Agent 2.
    A trained DQN picks the action from features that include the EXACT
    remaining-shoe distribution (one feature per card value), so it can
    deviate from basic strategy based on composition, not just running count.
    Expected outcome: more profit than Agent 2.

All three agents share the same env (BlackjackEnv from cardCounting.py).
The harness calls agent.get_bet(state) once per hand and overrides the
env's bet so each agent's betting policy is honoured.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Action codes -- match cardCounting.BlackjackEnv.step()
HIT, STAND, DOUBLE, SPLIT, SURRENDER = 0, 1, 2, 3, 4

# Surrender-with-fallback markers used in the basic-strategy tables. The
# chart distinguishes between "surrender if allowed, else <X>" cells. When
# the lookup returns one of these tuples, basic_strategy_action emits
# SURRENDER if the hand is still surrender-eligible (2 cards, un-split,
# only hand in play); otherwise it emits the fallback.
SR_H = (SURRENDER, HIT)     # "Uh" -- surrender if allowed else hit
SR_S = (SURRENDER, STAND)   # "Us" -- surrender if allowed else stand
SR_P = (SURRENDER, SPLIT)   # "Usp" -- surrender if allowed else split

# All 13 ranks exposed to the DQN separately. 10/J/Q/K are mechanically
# identical in blackjack but we expose them as four distinct features so
# the network can observe the exact composition of the remaining shoe.
FEATURE_CARD_VALUES = ['2', '3', '4', '5', '6', '7', '8', '9',
                       '10', 'J', 'Q', 'K', 'A']


# ---------------------------------------------------------------------------
# Basic strategy tables -- multi-deck, dealer hits soft 17, double-after-split
# ---------------------------------------------------------------------------

HARD_STRATEGY = {
    21: {d: STAND for d in range(2, 12)},
    20: {d: STAND for d in range(2, 12)},
    19: {d: STAND for d in range(2, 12)},
    18: {d: STAND for d in range(2, 12)},
    17: {2: STAND, 3: STAND, 4: STAND, 5: STAND, 6: STAND, 7: STAND, 8: STAND, 9: STAND, 10: STAND, 11: SR_S}, # 17 vs A: Us (surrender else stand)
    16: {2: STAND, 3: STAND, 4: STAND, 5: STAND, 6: STAND, 7: HIT,   8: HIT,   9: SR_H,  10: SR_H, 11: SR_H}, # 16 vs 9, 10, A: Uh (surrender else hit)
    15: {2: STAND, 3: STAND, 4: STAND, 5: STAND, 6: STAND, 7: HIT,   8: HIT,   9: HIT,   10: SR_H, 11: SR_H}, # 15 vs 10, A: Uh
    14: {2: STAND, 3: STAND, 4: STAND, 5: STAND, 6: STAND, 7: HIT,   8: HIT,   9: HIT,   10: HIT,  11: HIT},
    13: {2: STAND, 3: STAND, 4: STAND, 5: STAND, 6: STAND, 7: HIT,   8: HIT,   9: HIT,   10: HIT,  11: HIT},
    12: {2: HIT,   3: HIT,   4: STAND, 5: STAND, 6: STAND, 7: HIT,   8: HIT,   9: HIT,   10: HIT,  11: HIT},
    11: {d: DOUBLE for d in range(2, 12)},
    10: {2: DOUBLE, 3: DOUBLE, 4: DOUBLE, 5: DOUBLE, 6: DOUBLE, 7: DOUBLE, 8: DOUBLE, 9: DOUBLE, 10: HIT,    11: HIT},
    9:  {2: HIT,    3: DOUBLE, 4: DOUBLE, 5: DOUBLE, 6: DOUBLE, 7: HIT,    8: HIT,    9: HIT,    10: HIT,   11: HIT},
    8:  {d: HIT for d in range(2, 12)},
    7:  {d: HIT for d in range(2, 12)},
    6:  {d: HIT for d in range(2, 12)},
    5:  {d: HIT for d in range(2, 12)},
    4:  {d: HIT for d in range(2, 12)},
}

SOFT_STRATEGY = {
    21: {d: STAND for d in range(2, 12)},
    20: {d: STAND for d in range(2, 12)},
    # H17: soft 19 vs 6 is "Ds" -- double if allowed else stand.
    # The DOUBLE here falls through to STAND if we're past the first 2 cards.
    19: {2: STAND, 3: STAND, 4: STAND, 5: STAND, 6: DOUBLE, 7: STAND, 8: STAND, 9: STAND, 10: STAND, 11: STAND},
    18: {2: DOUBLE, 3: DOUBLE, 4: DOUBLE, 5: DOUBLE, 6: DOUBLE, 7: STAND,  8: STAND,  9: HIT,    10: HIT,   11: HIT},
    17: {2: HIT,    3: DOUBLE, 4: DOUBLE, 5: DOUBLE, 6: DOUBLE, 7: HIT,    8: HIT,    9: HIT,    10: HIT,   11: HIT},
    16: {2: HIT,    3: HIT,    4: DOUBLE, 5: DOUBLE, 6: DOUBLE, 7: HIT,    8: HIT,    9: HIT,    10: HIT,   11: HIT},
    15: {2: HIT,    3: HIT,    4: DOUBLE, 5: DOUBLE, 6: DOUBLE, 7: HIT,    8: HIT,    9: HIT,    10: HIT,   11: HIT},
    14: {2: HIT,    3: HIT,    4: HIT,    5: DOUBLE, 6: DOUBLE, 7: HIT,    8: HIT,    9: HIT,    10: HIT,   11: HIT},
    13: {2: HIT,    3: HIT,    4: HIT,    5: DOUBLE, 6: DOUBLE, 7: HIT,    8: HIT,    9: HIT,    10: HIT,   11: HIT},
}

# Keyed by the pair's card.value string. 10/J/Q/K all stand (don't split 10s).
PAIR_STRATEGY = {
    'A':  {d: SPLIT for d in range(2, 12)},
    '10': {d: STAND for d in range(2, 12)},
    'K':  {d: STAND for d in range(2, 12)},
    'Q':  {d: STAND for d in range(2, 12)},
    'J':  {d: STAND for d in range(2, 12)},
    '9':  {2: SPLIT, 3: SPLIT, 4: SPLIT, 5: SPLIT, 6: SPLIT, 7: STAND, 8: SPLIT, 9: SPLIT, 10: STAND, 11: STAND},
    # 8,8 vs A: Usp (surrender else split)
    '8':  {2: SPLIT, 3: SPLIT, 4: SPLIT, 5: SPLIT, 6: SPLIT, 7: SPLIT, 8: SPLIT, 9: SPLIT, 10: SPLIT, 11: SR_P},
    '7':  {2: SPLIT, 3: SPLIT, 4: SPLIT, 5: SPLIT, 6: SPLIT, 7: SPLIT, 8: HIT,   9: HIT,   10: HIT,  11: HIT},
    '6':  {2: SPLIT, 3: SPLIT, 4: SPLIT, 5: SPLIT, 6: SPLIT, 7: HIT,   8: HIT,   9: HIT,   10: HIT,  11: HIT},
    '5':  {2: DOUBLE, 3: DOUBLE, 4: DOUBLE, 5: DOUBLE, 6: DOUBLE, 7: DOUBLE, 8: DOUBLE, 9: DOUBLE, 10: HIT,    11: HIT},
    '4':  {2: HIT,   3: HIT,   4: HIT,   5: SPLIT, 6: SPLIT, 7: HIT,   8: HIT,   9: HIT,   10: HIT,  11: HIT},
    '3':  {2: SPLIT, 3: SPLIT, 4: SPLIT, 5: SPLIT, 6: SPLIT, 7: SPLIT, 8: HIT,   9: HIT,   10: HIT,  11: HIT},
    '2':  {2: SPLIT, 3: SPLIT, 4: SPLIT, 5: SPLIT, 6: SPLIT, 7: SPLIT, 8: HIT,   9: HIT,   10: HIT,  11: HIT},
}


def _resolve_surrender(action, state) -> int:
    """If `action` is a SR_* tuple, return SURRENDER when surrender is legal
    on this state; otherwise return the fallback. Pass-through for ints."""
    if not isinstance(action, tuple):
        return action
    primary, fallback = action
    # Late surrender: only the very first action, on the original 2-card,
    # un-split hand. The env enforces this; the chart assumes it too.
    player_hand = state['player_hand']
    num_cards = len(player_hand.cards)
    can_surrender = (num_cards == 2 and not player_hand.is_splitted)
    return primary if can_surrender else fallback


def basic_strategy_action(state) -> int:
    """Return the basic-strategy action for `state`.

    Handles legality: if the chart says DOUBLE/SPLIT but they aren't allowed
    here (already hit, or not a pair), falls back to the standard alternative
    (HIT for hard hands, STAND for soft hands when "Ds" is not allowed).
    Surrender cells in the chart are stored as (SURRENDER, fallback) tuples
    and get resolved against the current hand's surrender eligibility.
    """
    player_hand = state['player_hand']
    player_value = state['player_value']
    dealer_up_card = state['dealer_up_card']
    is_soft = state['is_soft']
    can_split = state['can_split']
    num_cards = len(player_hand.cards)

    if can_split and num_cards == 2:
        pair_key = player_hand.cards[0].value
        action = PAIR_STRATEGY.get(pair_key, {}).get(dealer_up_card, HIT)
    elif is_soft:
        action = SOFT_STRATEGY.get(player_value, {}).get(dealer_up_card, HIT)
    else:
        action = HARD_STRATEGY.get(player_value, {}).get(dealer_up_card, HIT)

    action = _resolve_surrender(action, state)

    if action == DOUBLE and num_cards != 2:
        # Soft "Ds" = double if allowed else stand. Hard fallback is hit.
        return STAND if is_soft else HIT
    if action == SPLIT and not (can_split and num_cards == 2):
        return HIT
    return action


# ---------------------------------------------------------------------------
# Illustrious-18 count deviations (subset that applies inside our env)
# ---------------------------------------------------------------------------
#
# Each entry: (player_value, dealer_up_card, is_soft) ->
#     (true_count_threshold, action_when_ge, action_when_lt)
#
# i.e. take `action_when_ge` if true_count >= threshold, else `action_when_lt`.
# The basic-strategy default is encoded inside this same tuple, so this list
# fully describes the cell.

ILLUSTRIOUS_18 = {
    # Each rule: (threshold, deviation_action, comparison)
    #   comparison='ge' -> take deviation when true_count >= threshold
    #   comparison='lt' -> take deviation when true_count <  threshold
    # Otherwise, fall back to the (possibly surrender-including) basic action.
    # This composes correctly with surrender: at e.g. 16 vs 10, basic is
    # SURRENDER; the deviation only triggers above the threshold and leaves
    # SURRENDER intact at low counts.
    #
    # Entries assume H17 multi-deck. "11 vs A double at TC>=1" is omitted
    # because H17 basic strategy already doubles.
    #
    # ---- Don Schlesinger's Illustrious 18 (the famous 18) ----
    (16, 10, False): (0,  STAND,  'ge'),    # stand 16 vs 10 at TC >= 0
    (15, 10, False): (4,  STAND,  'ge'),
    (10, 10, False): (4,  DOUBLE, 'ge'),
    (12, 3,  False): (2,  STAND,  'ge'),
    (12, 2,  False): (3,  STAND,  'ge'),
    (9,  2,  False): (1,  DOUBLE, 'ge'),
    (10, 11, False): (4,  DOUBLE, 'ge'),
    (9,  7,  False): (3,  DOUBLE, 'ge'),
    (16, 9,  False): (5,  STAND,  'ge'),
    (13, 2,  False): (-1, HIT,    'lt'),    # hit 13 vs 2 at TC < -1
    (12, 4,  False): (0,  HIT,    'lt'),
    (12, 5,  False): (-2, HIT,    'lt'),
    (12, 6,  False): (-1, HIT,    'lt'),
    (13, 3,  False): (-2, HIT,    'lt'),

    # ---- Fab 4 surrender indices ----
    # These trigger surrender at high counts where the dealer's
    # 10-or-A upcard is even MORE likely to be paired with a 10 hole.
    (14, 10, False): (3,  SURRENDER, 'ge'),   # 14 vs 10: surrender at TC >= +3
    (15, 9,  False): (2,  SURRENDER, 'ge'),   # 15 vs 9:  surrender at TC >= +2
    (15, 11, False): (-1, SURRENDER, 'ge'),   # 15 vs A:  surrender at TC >= -1 (H17)
    # 8,8 vs A: Fab-4 says surrender at TC >= -2; basic strategy already
    # surrenders (the chart's "Usp"), so no extra index needed here.

    # ---- Additional hard-hand indices (Sweet 16 / Catch 22 family) ----
    (16, 8,  False): (4,  STAND,  'ge'),    # stand 16 vs 8 at TC >= +4
    (13, 4,  False): (-2, HIT,    'lt'),    # hit 13 vs 4 at TC <= -3
    (14, 5,  False): (-3, HIT,    'lt'),    # hit 14 vs 5 at TC <= -4
    (14, 6,  False): (-2, HIT,    'lt'),    # hit 14 vs 6 at TC <= -3
    (14, 4,  False): (-1, HIT,    'lt'),    # hit 14 vs 4 at TC <= -2
}


# Soft-hand deviations. Indexed by (player_value, dealer_up_card, True).
# Returning DOUBLE for these falls back to STAND if doubling isn't legal
# (3+ cards), matching the chart's "Ds" semantics.
SOFT_DEVIATIONS = {
    # A,8 (soft 19) deviations -- basic is STAND vs 2-5, Ds vs 6, S vs 7-A.
    (19, 4, True):  (3, DOUBLE, 'ge'),       # soft 19 vs 4: Ds at TC >= +3
    (19, 5, True):  (1, DOUBLE, 'ge'),       # soft 19 vs 5: Ds at TC >= +1
    # A,7 (soft 18) deviations -- basic stand vs 2/7/8.
    (18, 2, True):  (1, DOUBLE, 'ge'),       # soft 18 vs 2: Ds at TC >= +1 (extra push past basic Ds)
}


PAIR_DEVIATIONS = {
    # (pair_card_value, dealer_up_card): (threshold, deviation_action, comparison)
    # Split 10s deviations from Illustrious-18.
    ('10', 5): (5, SPLIT, 'ge'),
    ('10', 6): (4, SPLIT, 'ge'),
    ('J',  5): (5, SPLIT, 'ge'),
    ('J',  6): (4, SPLIT, 'ge'),
    ('Q',  5): (5, SPLIT, 'ge'),
    ('Q',  6): (4, SPLIT, 'ge'),
    ('K',  5): (5, SPLIT, 'ge'),
    ('K',  6): (4, SPLIT, 'ge'),
    # 10,10 vs 4 -- split at deep TC, only happens at very high counts.
    ('10', 4): (6, SPLIT, 'ge'),
    ('J',  4): (6, SPLIT, 'ge'),
    ('Q',  4): (6, SPLIT, 'ge'),
    ('K',  4): (6, SPLIT, 'ge'),

    # 9,9 vs A -- basic is STAND in H17. Split at higher counts (more 10s
    # left -> two hands of 9 + 10 = 19, each beats most dealer no-BJ totals).
    ('9', 11): (3, SPLIT, 'ge'),
    # 9,9 vs 7 -- basic is STAND. Split when the count is positive.
    ('9',  7): (3, SPLIT, 'ge'),

    # 7,7 vs 8 -- basic is HIT. Split at moderate counts (each 7 + 10 = 17
    # is OK against dealer 8).
    ('7',  8): (4, SPLIT, 'ge'),

    # 6,6 vs 7 -- basic is HIT. Split at higher counts.
    ('6',  7): (5, SPLIT, 'ge'),

    # 2,2 / 3,3 vs 8 -- basic is HIT. Split when count is high enough.
    ('2',  8): (4, SPLIT, 'ge'),
    ('3',  8): (4, SPLIT, 'ge'),
}


def _deviation_triggers(threshold, comparison, true_count) -> bool:
    if comparison == 'ge':
        return true_count >= threshold
    if comparison == 'lt':
        return true_count < threshold
    return False


def basic_strategy_with_deviations(state) -> int:
    """Basic strategy + index plays.

    Each deviation is unidirectional -- it overrides the basic action only
    when its TC threshold is met. Below the threshold, the basic action
    (which may itself be SURRENDER) is preserved.

    Three deviation tables:
      - PAIR_DEVIATIONS for pair states (keyed by pair card value)
      - SOFT_DEVIATIONS for soft hands (Aces still counted as 11)
      - ILLUSTRIOUS_18 for everything else (hard non-pair hands), including
        Fab-4 surrender indices and a handful of Sweet-16/Catch-22 plays
    """
    action = basic_strategy_action(state)
    tc = state['true_count']

    num_cards = len(state['player_hand'].cards)
    is_pair = state['can_split'] and num_cards == 2

    if is_pair:
        pair_value = state['player_hand'].cards[0].value
        pkey = (pair_value, state['dealer_up_card'])
        if pkey in PAIR_DEVIATIONS:
            threshold, dev_action, comparison = PAIR_DEVIATIONS[pkey]
            if _deviation_triggers(threshold, comparison, tc):
                action = dev_action
    elif state['is_soft']:
        key = (state['player_value'], state['dealer_up_card'], True)
        if key in SOFT_DEVIATIONS:
            threshold, dev_action, comparison = SOFT_DEVIATIONS[key]
            if _deviation_triggers(threshold, comparison, tc):
                action = dev_action
    else:
        key = (state['player_value'], state['dealer_up_card'], False)
        if key in ILLUSTRIOUS_18:
            threshold, dev_action, comparison = ILLUSTRIOUS_18[key]
            if _deviation_triggers(threshold, comparison, tc):
                action = dev_action

    # Legality fallbacks.
    if action == DOUBLE and num_cards != 2:
        return STAND if state['is_soft'] else HIT
    if action == SPLIT and not is_pair:
        return HIT
    if action == SURRENDER and (num_cards != 2
                                or state['player_hand'].is_splitted):
        # Surrender no longer legal -- fall back to a reasonable alternative.
        # For 14/15 vs strong dealer, the fallback should match basic.
        # Re-run the lookup without the SURRENDER deviation by stripping
        # this particular cell.
        return basic_strategy_action(state)
    return action


# ---------------------------------------------------------------------------
# Bet ramp shared by Agent 2 and Agent 3
# ---------------------------------------------------------------------------

def count_based_bet(true_count: float, unit: int = 50, max_units: int = 50) -> int:
    """Spread the bet by true count.

    Default ramp is a 1-50 spread (min = 1 unit, max = 50 units) -- matches
    a typical high-limit table where the minimum is $50 and the max is
    $2,500. The ramp doubles each unit of TC until 5, then climbs more
    slowly, and is capped at `max_units` no matter how high the count goes.

      TC <= 1: 1 unit       TC = 5: 16 units      TC = 8: 40 units
      TC =  2: 2 units      TC = 6: 24 units      TC >= 9: max_units (50)
      TC =  3: 4 units      TC = 7: 32 units
      TC =  4: 8 units

    We FLOOR the true count first, so TC = 1.99 still bets 1 unit while
    TC = 2.0 bets 2 units.
    """
    tc = int(np.floor(true_count))
    if tc <= 1:
        units = 1
    elif tc == 2:
        units = 2
    elif tc == 3:
        units = 4
    elif tc == 4:
        units = 8
    elif tc == 5:
        units = 16
    elif tc == 6:
        units = 24
    elif tc == 7:
        units = 32
    elif tc == 8:
        units = 40
    else:
        units = max_units
    return unit * max(1, min(units, max_units))


# ---------------------------------------------------------------------------
# Agent 1: hardcoded basic strategy, flat bet, ignores the count
# ---------------------------------------------------------------------------

class FlatBetBasicStrategyAgent:
    name = "Flat-bet Basic Strategy"

    def __init__(self, flat_bet: int = 50):
        self.flat_bet = flat_bet

    def get_bet(self, state) -> int:
        return self.flat_bet

    def get_action(self, state) -> int:
        return basic_strategy_action(state)


# ---------------------------------------------------------------------------
# Agent 2: basic strategy + Hi-Lo count + Illustrious-18 deviations
# ---------------------------------------------------------------------------

class CountingBasicStrategyAgent:
    name = "Counting Basic Strategy"

    def __init__(self, unit: int = 50, max_units: int = 50):
        self.unit = unit
        self.max_units = max_units

    def get_bet(self, state) -> int:
        # IMPORTANT: bet on the PRE-deal count. state['true_count'] is the
        # POST-deal count (after the env dealt 3 cards), which is
        # anti-correlated with this hand's outcome -- low cards being
        # dealt makes the count tick up but also leaves the player with
        # a small total. The harness injects 'pre_true_count'.
        tc = state.get('pre_true_count', state['true_count'])
        return count_based_bet(tc, self.unit, self.max_units)

    def get_action(self, state) -> int:
        return basic_strategy_with_deviations(state)


# ---------------------------------------------------------------------------
# Agent 3: DQN that sees the full remaining-shoe composition
# ---------------------------------------------------------------------------

INPUT_DIM = 8 + 2 * len(FEATURE_CARD_VALUES)  # = 34 (8 game + 13 counts + 13 ratios)


class DQNNet(nn.Module):
    """Legacy scalar-Q DQN. Kept for backward compatibility with old saved
    checkpoints; new training uses DuelingQRDQN (QR-DQN RL training) or
    DuelingQNet (supervised distillation experiment)."""

    INPUT_DIM = INPUT_DIM
    OUTPUT_DIM = 5

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(self.INPUT_DIM, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.OUTPUT_DIM)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def q_values(self, x):
        """For interface compatibility with DuelingQRDQN."""
        return self.forward(x)


class DuelingQRDQN(nn.Module):
    """Dueling Quantile-Regression DQN.

    Three improvements over the plain DQN, addressing the three failure
    modes we saw with scalar-Q:

    1. **Distributional (QR-DQN)**: instead of predicting a single
       Q-value per action, predict N quantiles of the RETURN DISTRIBUTION
       per action. Argmax for action selection still uses the mean of the
       quantiles, but the gradient updates now respect the full spread.
       This is the fix for vanilla DQN's "variance-aversion on doubles"
       failure -- a doubled $50 bet on 11 vs 10 has high reward variance,
       and a scalar-Q network downweights it. A distributional network
       can see "high mean despite high variance" and still pick it.

    2. **Dueling**: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a)). The value
       and advantage streams are learned separately, which gives the
       network a much easier time on state-only features (penetration,
       count, etc. mostly affect V) versus action-relative features
       (player vs dealer card relative to action). This is the
       "advantage framing" of #3: A(s,a) is now an explicit output the
       network can learn to nudge separately from V(s).

    3. (Tied to training, not architecture: n-step returns and Polyak
       target updates -- see train_dqn.py.)

    Input: 34-dim state (8 game + 13 per-rank counts + 13 per-rank ratios).
    Output: quantile tensor shape (batch, n_actions, n_quantiles).
    """

    INPUT_DIM = INPUT_DIM
    OUTPUT_DIM = 5         # n_actions
    # 8 quantiles is plenty for blackjack -- per-hand rewards take only a
    # handful of distinct values (lose bet, push, win, win-BJ, lose-bust-
    # double, etc.), so the quantile distribution doesn't need many bins.
    # The QR-Huber loss is O(N²), so going from 16 -> 8 cuts loss compute
    # 4x and lets us push more episodes through training.
    N_QUANTILES = 8

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(self.INPUT_DIM, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Dueling: separate value and advantage heads.
        self.value_head = nn.Linear(hidden_dim, self.N_QUANTILES)
        self.adv_head = nn.Linear(hidden_dim,
                                  self.OUTPUT_DIM * self.N_QUANTILES)

    def forward(self, x):
        """Returns quantile tensor: (batch, n_actions, n_quantiles)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        v = self.value_head(h).unsqueeze(1)                    # (B, 1, Q)
        a = self.adv_head(h).view(-1, self.OUTPUT_DIM,
                                  self.N_QUANTILES)            # (B, A, Q)
        a_mean = a.mean(dim=1, keepdim=True)                   # (B, 1, Q)
        # Standard dueling combination: Q = V + (A - mean_a A).
        return v + a - a_mean

    def q_values(self, x):
        """Mean Q-value per action: (batch, n_actions). The mean of the
        quantile distribution is the standard scalar-Q estimate."""
        return self.forward(x).mean(dim=2)


class DuelingQNet(nn.Module):
    """Dueling scalar-Q net used by the supervised-distillation trainer.

    The expectimax solver computes exact per-action EXPECTED VALUES, so
    the training signal is a vector of scalars (one EV per action), not
    a distribution over returns. There's no reason to spend network
    capacity on N quantile heads when every label is a delta -- a plain
    scalar Dueling Q-net is a better fit and trains a few times faster.

    Architecture:
        fc1: INPUT_DIM -> hidden_dim   (relu)
        fc2: hidden_dim -> hidden_dim  (relu)
        value_head: hidden_dim -> 1
        adv_head:   hidden_dim -> OUTPUT_DIM
        Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))

    The dueling split is still useful here: the V stream can specialize
    in penetration/count/composition (purely state-dependent factors that
    shift every action's EV together), and the A stream specializes in
    the action-relative shifts (which action wins for THIS hand).
    """

    INPUT_DIM = INPUT_DIM
    OUTPUT_DIM = 5

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(self.INPUT_DIM, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.adv_head = nn.Linear(hidden_dim, self.OUTPUT_DIM)

    def forward(self, x):
        """Returns Q-values: (batch, n_actions)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        v = self.value_head(h)                                   # (B, 1)
        a = self.adv_head(h)                                     # (B, A)
        a_mean = a.mean(dim=1, keepdim=True)                     # (B, 1)
        return v + a - a_mean                                    # (B, A)

    def q_values(self, x):
        """Same as forward (kept for interface compat with QR-DQN)."""
        return self.forward(x)


def load_dqn_net(checkpoint_path: str, device='cpu'):
    """Load a DQN checkpoint, auto-detecting which architecture it was
    trained under.

    Three supported architectures:
      - Scalar-Q dueling Q (DuelingQNet) -- supervised-distillation default.
      - Distributional QR-DQN dueling (DuelingQRDQN) -- legacy RL trainer.
      - Plain scalar-Q (DQNNet) -- earliest checkpoints.

    Raises a clear error if the checkpoint's input dimension doesn't match
    the current 28-dim features (e.g. it was trained before the
    10/J/Q/K collapse) -- you'll need to retrain.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('net', checkpoint) if isinstance(checkpoint, dict) else checkpoint

    # First-layer weight tells us the saved input dim; bail loudly if it
    # doesn't match the current feature spec.
    fc1_w = state_dict.get('fc1.weight')
    if fc1_w is not None and fc1_w.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} was trained with input dim "
            f"{fc1_w.shape[1]} but the current feature spec is {INPUT_DIM}. "
            f"This usually means the checkpoint predates the 10/J/Q/K "
            f"collapse. Retrain with train_dqn.py and re-save.")

    has_value_head = any(k.startswith('value_head.') for k in state_dict)
    has_adv_head = any(k.startswith('adv_head.') for k in state_dict)
    if has_value_head and has_adv_head:
        # Dueling. Disambiguate quantile vs scalar by the adv_head out_features.
        # Scalar dueling: adv_head out = OUTPUT_DIM (5).
        # QR-DQN dueling:  adv_head out = OUTPUT_DIM * N_QUANTILES.
        adv_w = state_dict['adv_head.weight']
        if adv_w.shape[0] == DuelingQNet.OUTPUT_DIM:
            net = DuelingQNet().to(device)
        else:
            net = DuelingQRDQN().to(device)
    else:
        net = DQNNet().to(device)

    net.load_state_dict(state_dict)
    return net


_INITIAL_CARDS_PER_RANK = 32  # 8 decks * 4 suits
_INITIAL_TOTAL_CARDS = 8 * 52  # = 416


def state_to_features(state, device='cpu') -> torch.Tensor:
    """Map an env state dict -> 34-dim feature tensor consumed by the DQN.

    Features (34 total):
      Game state (8):
        0  player_value / 21
        1  dealer_up_card / 11
        2  is_soft (0/1)
        3  can_split (0/1)
        4  num_cards / 10
        5  true_count, tanh-normalized (range roughly [-1, +1])
        6  penetration (0 = fresh shoe, ~0.84 = at the cut card)
        7  total_cards_remaining / 416

      Per-rank counts (13), one per rank in [2,3,4,5,6,7,8,9,10,J,Q,K,A]:
        8..20  count_remaining / 32   (= fraction of original still in shoe)

      Per-rank ratios (13), same order:
        21..33  count_remaining / total_remaining
                (= probability the next dealt card is this rank)

    Note: 10 / J / Q / K are mechanically identical (all count as 10) but
    are exposed as four distinct features so the network can see the full
    composition. The network can learn they're equivalent.
    """
    player_hand = state['player_hand']
    num_cards = len(player_hand.cards) if player_hand else 0

    dist = state.get('card_distribution') or {}
    total = sum(dist.values()) or 1
    true_count = state.get('true_count', 0.0)

    feats = [
        state.get('player_value', 0) / 21.0,
        state.get('dealer_up_card', 0) / 11.0,
        1.0 if state.get('is_soft', False) else 0.0,
        1.0 if state.get('can_split', False) else 0.0,
        num_cards / 10.0,
        np.tanh(true_count / 5.0),
        1.0 - total / _INITIAL_TOTAL_CARDS,
        total / _INITIAL_TOTAL_CARDS,
    ]

    # 13 per-rank counts (normalized by initial 32 per rank).
    for value in FEATURE_CARD_VALUES:
        feats.append(dist.get(value, 0) / _INITIAL_CARDS_PER_RANK)

    # 13 per-rank ratios (count / total_remaining).
    for value in FEATURE_CARD_VALUES:
        feats.append(dist.get(value, 0) / total)

    return torch.tensor(feats, dtype=torch.float32, device=device)


def pick_device(prefer: Optional[str] = None) -> torch.device:
    """Pick a torch device. `prefer` overrides if given.

    Note: for INFERENCE with our tiny 16->128->128->4 net, CPU is usually
    faster than MPS because each MPS kernel launch has ~ms-scale overhead
    and we forward one example at a time. We default to CPU for inference.
    Training (where we forward 256-example batches) might do better on MPS
    but in practice CPU is competitive for this network size.
    """
    if prefer is not None:
        return torch.device(prefer)
    return torch.device('cpu')


def _valid_actions(state) -> list:
    # Split-aces hands: env forces STAND regardless of the action passed
    # in (no hit/double/resplit allowed by casino rule). So the only
    # action that maps to a meaningful play is STAND.
    hand = state.get('player_hand')
    if hand and getattr(hand, 'is_split_aces', False):
        return [STAND]

    valid = [HIT, STAND]
    if hand and len(hand.cards) == 2:
        valid.append(DOUBLE)
    if state.get('can_split', False):
        valid.append(SPLIT)
    # Late surrender: only on the very first action -- 2 cards, single
    # un-split hand. Matches the env's enforcement.
    if (hand
            and len(hand.cards) == 2
            and not hand.is_splitted):
        valid.append(SURRENDER)
    return valid


class DQNCardCountingAgent:
    """Agent 3. Hybrid: basic strategy + Illustrious-18 by default; the DQN
    can override an action when it is sufficiently confident the deviation
    is worth it.

    The DQN sees the FULL remaining-shoe distribution as features, so its
    deviations are composition-aware -- they go beyond what running-count
    alone can express.

    Why hybrid instead of letting the DQN play standalone? Q-learning with
    a sparse per-hand reward and only ~10^5 episodes does not reliably
    re-discover all of basic strategy from scratch -- some rare states
    stay under-trained. Keeping basic strategy as the default guarantees
    the agent never loses EV by under-training, and the DQN can ONLY
    improve on it.

    If `model_path` is given and the file exists the network's weights are
    loaded; otherwise the agent is just Agent 2.
    """

    name = "DQN Card Counting"

    def __init__(self,
                 model_path: Optional[str] = None,
                 device=None,
                 unit: int = 50,
                 max_units: int = 50,
                 deviation_margin: float = 0.05):
        self.device = pick_device(device)
        self.unit = unit
        self.max_units = max_units
        # Minimum Q-value advantage the DQN must show for an alternative
        # action before we deviate from basic+I18. For Dueling QR-DQN the
        # Q-values represent expected return in normalized $50-units, so
        # 0.05 means "the deviation must be at least 5% of a unit bet
        # better in expected return". Lower this to express more
        # deviations, raise it to be more conservative.
        self.deviation_margin = deviation_margin
        self.loaded_from = None
        self.decision_count = 0
        self.deviation_count = 0

        if model_path and os.path.exists(model_path):
            self.net = load_dqn_net(model_path, device=self.device)
            self.loaded_from = model_path
        else:
            # No model: fall back to the new architecture with random weights
            # so the agent runs (and at margin>0 it will degrade to basic+I18).
            self.net = DuelingQRDQN().to(self.device)
        self.net.eval()

    def get_bet(self, state) -> int:
        tc = state.get('pre_true_count', state['true_count'])
        return count_based_bet(tc, self.unit, self.max_units)

    def get_action(self, state) -> int:
        self.decision_count += 1
        default = basic_strategy_with_deviations(state)
        valid = _valid_actions(state)

        with torch.no_grad():
            feats = state_to_features(state, device=self.device)
            # q_values() gives the mean of the quantile distribution for
            # QR-DQN, or the raw Q-values for legacy scalar-Q. Same shape
            # either way: (1, n_actions).
            q = self.net.q_values(feats).squeeze(0).cpu().numpy()

        masked = q.copy()
        # Mask invalid actions across both architectures (both have the
        # same OUTPUT_DIM=5).
        for a in range(masked.shape[0]):
            if a not in valid:
                masked[a] = -np.inf

        dqn_choice = int(np.argmax(masked))
        if dqn_choice == default:
            return default

        # The DQN disagrees. Only deviate if it is meaningfully more
        # confident than the default action.
        if masked[dqn_choice] - masked[default] >= self.deviation_margin:
            self.deviation_count += 1
            return dqn_choice
        return default


__all__ = [
    'FlatBetBasicStrategyAgent',
    'CountingBasicStrategyAgent',
    'DQNCardCountingAgent',
    'DQNNet',
    'DuelingQRDQN',
    'DuelingQNet',
    'load_dqn_net',
    'basic_strategy_action',
    'basic_strategy_with_deviations',
    'count_based_bet',
    'state_to_features',
    'pick_device',
    'HIT', 'STAND', 'DOUBLE', 'SPLIT', 'SURRENDER',
    'ILLUSTRIOUS_18',
    'FEATURE_CARD_VALUES',
    'INPUT_DIM',
]
