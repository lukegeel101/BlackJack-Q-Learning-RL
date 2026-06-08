"""
Exact expectimax solver for blackjack: given an env decision point (player
hand value, dealer upcard, exact remaining-shoe composition), compute the
expected value of each legal action AND the optimal action.

This is the offline "teacher" for Option 2 -- supervised distillation of
the DQN. The expectimax labels are the strongest possible signal we can
give the network: not "basic strategy says STAND", but "STAND's exact EV
given this shoe is +0.07 while HIT's is -0.12, with this much margin."

Game rules (must match cardCounting.BlackjackEnv):
    8-deck shoe, dealer hits soft 17, blackjack pays 3:2, DAS allowed,
    late surrender allowed (only on first action of un-split hand),
    split aces get one card each with no further play and no resplits,
    max 4 hands from splits, US peek for dealer BJ on T/A upcards.

Composition representation
--------------------------
We work in the 10-rank space already used by state_to_features:
    ('2', '3', '4', '5', '6', '7', '8', '9', 'T', 'A')

'T' aggregates 10/J/Q/K (mechanically identical, all worth 10). The env's
card_distribution dict still uses 13 separate keys; comp_from_state below
collapses to the 10-rank representation.

The composition is carried around as a tuple of 10 ints (counts in rank
order) so that it's hashable for LRU caching. Card removal returns a
fresh tuple via tuple slicing -- cheap at n=10.

Tractability
------------
Per-state solve cost is dominated by:
  - dealer_play_dist: deep recursion bounded by "dealer at 17+". Hits an
    LRU cache aggressively because dealer paths repeat across player
    actions.
  - ev_hit recursion: at most ~5 player draws (4 + 5 + ... <= 21).
  - ev_split: ONE level of subhand expectimax. NRSA is approximated as
    "no resplit at all" -- we let each split sub-hand run as a normal
    2-card decision with DAS, but we don't recurse on another split
    inside it. Resplit EV impact is small (< 0.05% of total game EV)
    and trying to model it exactly explodes runtime.

Typical solve cost: ~1-5 ms per (hand, composition). Dataset generation
of 200k states should take ~5-15 minutes single-threaded.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Tuple

from agents import HIT, STAND, DOUBLE, SPLIT, SURRENDER

# --------------------------------------------------------------------------
# Rank conventions
# --------------------------------------------------------------------------

RANKS: Tuple[str, ...] = ('2', '3', '4', '5', '6', '7', '8', '9', 'T', 'A')
RANK_TO_IDX: Dict[str, int] = {r: i for i, r in enumerate(RANKS)}
# Hard value of each rank treated as a single card (A = 11 by default;
# downgraded to 1 inside the hand logic if needed).
RANK_HARD_VALUE: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
T_IDX = RANK_TO_IDX['T']
A_IDX = RANK_TO_IDX['A']

# Initial counts in an 8-deck shoe per feature rank. 'T' is 4 ranks
# bundled together so it gets 4x the count.
INITIAL_COMP: Tuple[int, ...] = tuple(128 if r == 'T' else 32 for r in RANKS)

# Dealer up-card pre-image: the env emits dealer_up_card in {2..11}. We
# need to map back to the rank for composition arithmetic. 10 -> 'T'.
_UPCARD_TO_RANK = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6',
                   7: '7', 8: '8', 9: '9', 10: 'T', 11: 'A'}


# --------------------------------------------------------------------------
# Composition helpers
# --------------------------------------------------------------------------

def comp_from_state(state) -> Tuple[int, ...]:
    """Extract the 10-rank composition tuple from an env state dict.

    The env's card_distribution uses 13 keys ('10','J','Q','K' separate);
    we sum them into 'T'.
    """
    dist = state.get('card_distribution') or {}
    out = [0] * 10
    for i, r in enumerate(RANKS):
        if r == 'T':
            out[i] = (dist.get('10', 0) + dist.get('J', 0)
                      + dist.get('Q', 0) + dist.get('K', 0))
        else:
            out[i] = dist.get(r, 0)
    return tuple(out)


def remove_rank(comp: Tuple[int, ...], idx: int) -> Tuple[int, ...]:
    """Return a new tuple with comp[idx] decremented by 1."""
    if comp[idx] <= 0:
        return comp
    return comp[:idx] + (comp[idx] - 1,) + comp[idx + 1:]


def card_total(comp: Tuple[int, ...]) -> int:
    return sum(comp)


def upcard_to_rank(upcard_val: int) -> str:
    return _UPCARD_TO_RANK[upcard_val]


# --------------------------------------------------------------------------
# Hand-value helpers
# --------------------------------------------------------------------------

def apply_card(value: int, soft_aces: int, rank_idx: int) -> Tuple[int, int]:
    """Add a card of the given rank to a hand with `value` total and
    `soft_aces` aces still counted as 11. Returns (new_value, new_soft_aces)
    after downgrading aces as needed to keep the value <= 21 when possible.
    """
    new_value = value + RANK_HARD_VALUE[rank_idx]
    new_soft = soft_aces + (1 if rank_idx == A_IDX else 0)
    while new_value > 21 and new_soft > 0:
        new_value -= 10
        new_soft -= 1
    return new_value, new_soft


def two_card_total(rank_idx_a: int, rank_idx_b: int) -> Tuple[int, int]:
    """Compute (value, soft_aces) for a 2-card hand."""
    v, s = 0, 0
    for r in (rank_idx_a, rank_idx_b):
        v, s = apply_card(v, s, r)
    return v, s


# --------------------------------------------------------------------------
# Dealer recursion (with caching)
# --------------------------------------------------------------------------
#
# Returns the distribution of dealer final values. Possible outcomes:
#   17, 18, 19, 20, 21, -1 (busted).
# Encoding as -1 saves us a string key.
#
# H17: dealer must hit soft 17 (stands on hard 17, 18+).

_BUST = -1


@lru_cache(maxsize=500_000)
def _dealer_play_dist(value: int, soft_aces: int,
                      comp: Tuple[int, ...]) -> Tuple[Tuple[int, float], ...]:
    """Distribution over dealer's final value given the current dealer
    hand value, soft-ace count, and remaining shoe composition.

    Returns a tuple of (final_value, prob) pairs so the result is hashable
    and immutable. The cache hit rate is high because dealer paths from
    different player decisions converge on the same (value, soft, comp)
    intermediate states.
    """
    # Bust.
    if value > 21:
        return ((_BUST, 1.0),)
    # Stand: hard 17+ or soft 18+.
    is_soft = soft_aces > 0
    if value >= 18 or (value == 17 and not is_soft):
        return ((value, 1.0),)
    # H17: at value 17 with soft, must hit.

    total = sum(comp)
    if total == 0:
        # Out of cards (deck exhaustion edge case). Treat as stand.
        return ((value, 1.0),)

    bucket: Dict[int, float] = defaultdict(float)
    for r_idx in range(10):
        c = comp[r_idx]
        if c == 0:
            continue
        p = c / total
        new_value, new_soft = apply_card(value, soft_aces, r_idx)
        new_comp = remove_rank(comp, r_idx)
        for outcome, p2 in _dealer_play_dist(new_value, new_soft, new_comp):
            bucket[outcome] += p * p2
    return tuple(sorted(bucket.items()))


@lru_cache(maxsize=200_000)
def dealer_outcome_dist(upcard_rank_idx: int,
                        comp: Tuple[int, ...]) -> Tuple[Tuple[int, float], ...]:
    """Distribution over the dealer's final value, integrating over the
    hidden hole card. Conditions on "dealer doesn't have natural BJ" when
    the upcard is T or A (US peek would otherwise have already ended the
    round before the player acted).

    Cached aggressively -- within a single solve, the player's HIT
    recursion reaches many (upcard, comp) combinations that share the
    same dealer dist (since order of player draws doesn't matter for
    the dealer's remaining composition).

    `comp` should be the composition AS SEEN BY THE PLAYER at decision
    time -- i.e. after the player's cards, the dealer upcard, AND the
    dealer hole (if dealt) have already been removed.

    Subtle point about the hole: when upcard is T/A the env's peek already
    dealt the hole and removed it from the deck pool, so the hole is
    NOT in `comp`. We model the hole's distribution by adding 1 to each
    possible hole rank (the actual hole is some specific card we don't
    know the value of; the posterior over its rank is proportional to
    rank counts in the *original* composition before hole removal, which
    equals comp + (1 in hole's slot)).

    For non-peek upcards (2-9), no hole has been pre-dealt, so we simply
    draw the hole from `comp`.
    """
    if upcard_rank_idx in (T_IDX, A_IDX):
        # Peek case: hole was dealt during reset and removed from comp.
        # Posterior P(hole = r) ∝ (comp[r] + 1) restricted to non-BJ ranks.
        # BJ-forbidden rank: T if upcard=A, A if upcard=T.
        forbidden = A_IDX if upcard_rank_idx == T_IDX else T_IDX
        # Total mass of allowed ranks (with +1 for the missing hole card).
        denom = 0
        for r in range(10):
            if r == forbidden:
                continue
            denom += comp[r] + 1
        if denom == 0:
            # Pathological -- treat as stand on upcard value.
            v, s = apply_card(0, 0, upcard_rank_idx)
            return ((v, 1.0),)
        bucket: Dict[int, float] = defaultdict(float)
        for r in range(10):
            if r == forbidden:
                continue
            mass = comp[r] + 1
            if mass <= 0:
                continue
            p_hole = mass / denom
            # The actual unseen hole really WAS this rank: comp + (hole=r).
            # But since the hole is hidden, dealer's onward draws come from
            # `comp` directly (the hole is set aside, then revealed and
            # incorporated into the dealer total before further hits).
            # For dealer's continuation, the composition is `comp`.
            v, s = apply_card(0, 0, upcard_rank_idx)
            v, s = apply_card(v, s, r)
            for outcome, p2 in _dealer_play_dist(v, s, comp):
                bucket[outcome] += p_hole * p2
        return tuple(sorted(bucket.items()))
    else:
        # Non-peek upcard: hole hasn't been dealt yet. Draw the hole from
        # comp, then play out.
        total = sum(comp)
        if total == 0:
            v, s = apply_card(0, 0, upcard_rank_idx)
            return ((v, 1.0),)
        bucket = defaultdict(float)
        for r in range(10):
            c = comp[r]
            if c == 0:
                continue
            p = c / total
            new_comp = remove_rank(comp, r)
            v, s = apply_card(0, 0, upcard_rank_idx)
            v, s = apply_card(v, s, r)
            for outcome, p2 in _dealer_play_dist(v, s, new_comp):
                bucket[outcome] += p * p2
        return tuple(sorted(bucket.items()))


# --------------------------------------------------------------------------
# Player decision EV
# --------------------------------------------------------------------------
#
# Convention: EVs are normalized to "per unit bet". A win returns +1, a
# loss returns -1, a push returns 0, a blackjack returns +1.5. DOUBLE'd
# wins/losses return ±2 (we report the EV of the full doubled wager).
# SURRENDER returns -0.5.

@lru_cache(maxsize=400_000)
def _ev_stand_inner(player_value: int, upcard_rank_idx: int,
                    comp: Tuple[int, ...]) -> float:
    """Cached core: EV of standing on a non-busted, non-blackjack hand.
    Wrapper `ev_stand` handles the BJ and bust special cases so this
    cache key stays minimal."""
    dist = dealer_outcome_dist(upcard_rank_idx, comp)
    ev = 0.0
    for outcome, p in dist:
        if outcome == _BUST:
            ev += p
        elif outcome > player_value:
            ev -= p
        elif outcome < player_value:
            ev += p
        # outcome == player_value: push, += 0.
    return ev


def ev_stand(player_value: int, is_blackjack: bool,
             upcard_rank_idx: int, comp: Tuple[int, ...]) -> float:
    """EV of standing. is_blackjack short-circuits to +1.5 (3:2 payout);
    dealer-BJ cases are already ruled out by US peek before this is called.
    """
    if player_value > 21:
        return -1.0
    if is_blackjack:
        return 1.5
    return _ev_stand_inner(player_value, upcard_rank_idx, comp)


@lru_cache(maxsize=400_000)
def ev_hit_then_optimal(player_value: int, soft_aces: int,
                        upcard_rank_idx: int,
                        comp: Tuple[int, ...]) -> float:
    """EV of HIT followed by optimal continuation (HIT or STAND only --
    no DOUBLE/SPLIT/SURRENDER are legal after the first hit in our env).
    Cached because the same continuation state is reached from many
    different player draw orderings within one solve.
    """
    total = sum(comp)
    if total == 0:
        return _ev_stand_inner(player_value, upcard_rank_idx, comp)

    ev = 0.0
    for r_idx in range(10):
        c = comp[r_idx]
        if c == 0:
            continue
        p = c / total
        new_value, new_soft = apply_card(player_value, soft_aces, r_idx)
        new_comp = remove_rank(comp, r_idx)

        if new_value > 21:
            ev -= p
        else:
            stand_ev = _ev_stand_inner(new_value, upcard_rank_idx, new_comp)
            if new_value < 21:
                hit_ev = ev_hit_then_optimal(new_value, new_soft,
                                             upcard_rank_idx, new_comp)
                ev += p * (stand_ev if stand_ev >= hit_ev else hit_ev)
            else:
                ev += p * stand_ev
    return ev


@lru_cache(maxsize=200_000)
def ev_double(player_value: int, soft_aces: int,
              upcard_rank_idx: int, comp: Tuple[int, ...]) -> float:
    """EV of doubling: one card, forced stand, doubled wager. Cached."""
    total = sum(comp)
    if total == 0:
        return 2.0 * _ev_stand_inner(player_value, upcard_rank_idx, comp)

    ev = 0.0
    for r_idx in range(10):
        c = comp[r_idx]
        if c == 0:
            continue
        p = c / total
        new_value, _new_soft = apply_card(player_value, soft_aces, r_idx)
        new_comp = remove_rank(comp, r_idx)
        if new_value > 21:
            ev -= 2.0 * p
        else:
            ev += p * 2.0 * _ev_stand_inner(new_value, upcard_rank_idx,
                                            new_comp)
    return ev


def _ev_split_aces(comp: Tuple[int, ...],
                   upcard_rank_idx: int) -> float:
    """EV of splitting aces. Each hand gets exactly one card, then is
    forced to stand. Resulting hand of A+T pays NOT as natural BJ but
    as a plain 21 (env enforces is_splitted -> is_blackjack=False).

    We approximate the two sub-hands as independent draws from `comp`,
    which has had the two original Aces already removed (since the
    env's deck.cards no longer contains them).
    """
    total = sum(comp)
    if total == 0:
        return 0.0

    sub_ev = 0.0
    for r_idx in range(10):
        c = comp[r_idx]
        if c == 0:
            continue
        p = c / total
        # Each sub-hand: A + this rank. A starts as soft 11.
        v, s = 11, 1
        v, s = apply_card(v, s, r_idx)
        new_comp = remove_rank(comp, r_idx)
        # Forced stand. A+T is 21 but NOT natural BJ post-split.
        sub_ev += p * ev_stand(v, is_blackjack=False,
                               upcard_rank_idx=upcard_rank_idx,
                               comp=new_comp)
    # Two sub-hands, each contributes its own EV.
    return 2.0 * sub_ev


def _ev_split_nonaces(pair_rank_idx: int, upcard_rank_idx: int,
                      comp: Tuple[int, ...]) -> float:
    """EV of splitting a non-Ace pair into 2 sub-hands. Approximations:

      - Treat the two sub-hands as independent draws from `comp` (the
        env's deck has both halves of the pair already removed, so
        sampling from `comp` is correct in expectation; the dependence
        between the two halves' card sequences is negligible at 416-card
        scale).

      - NO RESPLITS in the sub-hand decision tree. If the sub-hand draws
        another card matching `pair_rank_idx`, we just treat it as a normal
        2-card hand and play optimally without SPLIT. Real env allows up
        to 4 hands; the EV impact of this approximation is small.
    """
    total = sum(comp)
    if total == 0:
        return 0.0

    sub_ev = 0.0
    for r_idx in range(10):
        c = comp[r_idx]
        if c == 0:
            continue
        p = c / total
        # Sub-hand: pair_rank + this rank.
        v, s = two_card_total(pair_rank_idx, r_idx)
        new_comp = remove_rank(comp, r_idx)
        # Sub-hand play: HIT/STAND/DOUBLE (DAS). NO further split,
        # no surrender post-split.
        sub_ev += p * _optimal_ev_no_split_no_surrender(
            v, s, num_cards=2,
            is_split=True, is_split_aces=False,
            upcard_rank_idx=upcard_rank_idx, comp=new_comp)
    return 2.0 * sub_ev


def _optimal_ev_no_split_no_surrender(player_value: int, soft_aces: int,
                                      num_cards: int, is_split: bool,
                                      is_split_aces: bool,
                                      upcard_rank_idx: int,
                                      comp: Tuple[int, ...]) -> float:
    """EV of playing optimally from a sub-hand, without SPLIT or
    SURRENDER available. DOUBLE is allowed iff num_cards == 2 and not
    split_aces. STAND and HIT always allowed.

    Used inside split sub-trees.
    """
    if is_split_aces:
        # Forced stand after the one allowed card.
        return ev_stand(player_value, False, upcard_rank_idx, comp)

    # STAND.
    stand_ev = ev_stand(player_value, False, upcard_rank_idx, comp)
    best = stand_ev

    # HIT.
    if player_value < 21:
        hit_ev = ev_hit_then_optimal(player_value, soft_aces,
                                     upcard_rank_idx, comp)
        if hit_ev > best:
            best = hit_ev

    # DOUBLE.
    if num_cards == 2:
        d_ev = ev_double(player_value, soft_aces, upcard_rank_idx, comp)
        if d_ev > best:
            best = d_ev

    return best


def evaluate_actions(player_value: int, soft_aces: int, num_cards: int,
                     is_blackjack: bool,
                     can_split: bool, pair_rank_idx: int,
                     is_splitted: bool, is_split_aces: bool,
                     can_surrender: bool,
                     upcard_rank_idx: int,
                     comp: Tuple[int, ...]) -> Dict[int, float]:
    """Compute EV per legal action. Returns {action: EV}.

    Legality follows BlackjackEnv.step():
      - HIT always legal (unless split_aces -- env forces STAND).
      - STAND always legal.
      - DOUBLE: only if num_cards == 2 AND not split_aces.
      - SPLIT: only if can_split (pair) AND num_cards == 2 (DAS implicit
        in pair_rank's sub-hand recursion).
      - SURRENDER: only if num_cards == 2 AND NOT splitted AND first
        decision (caller supplies `can_surrender`).

    If `is_blackjack` is True the only realistic option is STAND -- the
    env doesn't even prompt for action on a natural BJ -- but we still
    return STAND's EV for completeness.
    """
    actions: Dict[int, float] = {}

    # Split-aces: env forces STAND. Only STAND is reachable.
    if is_split_aces:
        actions[STAND] = ev_stand(player_value, False,
                                  upcard_rank_idx, comp)
        return actions

    # STAND.
    actions[STAND] = ev_stand(player_value, is_blackjack,
                              upcard_rank_idx, comp)

    # HIT (un-available if already at 21? Still legal; just bad. Include
    # it so the network learns it's bad).
    actions[HIT] = ev_hit_then_optimal(player_value, soft_aces,
                                       upcard_rank_idx, comp)

    # DOUBLE.
    if num_cards == 2:
        actions[DOUBLE] = ev_double(player_value, soft_aces,
                                    upcard_rank_idx, comp)

    # SPLIT.
    if can_split and num_cards == 2:
        if pair_rank_idx == A_IDX:
            actions[SPLIT] = _ev_split_aces(comp, upcard_rank_idx)
        else:
            actions[SPLIT] = _ev_split_nonaces(pair_rank_idx,
                                               upcard_rank_idx, comp)

    # SURRENDER.
    if can_surrender and num_cards == 2 and not is_splitted:
        actions[SURRENDER] = -0.5

    return actions


def optimal_action_and_ev(*args, **kwargs) -> Tuple[int, float]:
    """Wrapper: return (best_action, best_ev) for the state."""
    actions = evaluate_actions(*args, **kwargs)
    best_action = max(actions, key=actions.get)
    return best_action, actions[best_action]


# --------------------------------------------------------------------------
# Entry point used by gen_optimal_dataset.py
# --------------------------------------------------------------------------

def evaluate_state(state) -> Tuple[Dict[int, float], int]:
    """Evaluate an env state dict and return (action -> EV, optimal_action).

    Reads everything from the state dict so callers can just hand over
    whatever env.reset()/env.step() returned.
    """
    player_hand = state['player_hand']
    player_value = state['player_value']
    is_soft = state['is_soft']
    can_split = state['can_split']
    upcard_val = state['dealer_up_card']

    num_cards = len(player_hand.cards)
    is_blackjack = player_hand.is_blackjack
    is_splitted = player_hand.is_splitted
    is_split_aces = player_hand.is_split_aces

    # Soft-aces count: how many aces are currently counted as 11.
    soft_aces = player_hand.aces if is_soft else 0

    # Pair rank for SPLIT branch.
    pair_rank_idx = -1
    if can_split and num_cards == 2:
        pair_value = player_hand.cards[0].value
        # Map env card.value strings to our 10-rank space.
        if pair_value in ('10', 'J', 'Q', 'K'):
            pair_rank_idx = T_IDX
        else:
            pair_rank_idx = RANK_TO_IDX[pair_value]

    # SURRENDER is legal only on the FIRST action of an un-split hand,
    # and only if there's exactly one hand in play. The env enforces this;
    # the state dict doesn't expose "first action", but we conservatively
    # allow SURRENDER iff num_cards == 2 and not is_splitted. That's
    # exactly what the env's eligibility check tests (the env's additional
    # current_hand_index == 0 / single hand check is encoded by
    # is_splitted -- if there are split hands, is_splitted is True).
    can_surrender = (num_cards == 2 and not is_splitted)

    upcard_rank_idx = RANK_TO_IDX[upcard_to_rank(upcard_val)]
    comp = comp_from_state(state)

    evs = evaluate_actions(
        player_value=player_value,
        soft_aces=soft_aces,
        num_cards=num_cards,
        is_blackjack=is_blackjack,
        can_split=can_split,
        pair_rank_idx=pair_rank_idx,
        is_splitted=is_splitted,
        is_split_aces=is_split_aces,
        can_surrender=can_surrender,
        upcard_rank_idx=upcard_rank_idx,
        comp=comp,
    )
    best_action = max(evs, key=evs.get)
    return evs, best_action


def clear_caches():
    """Free memoization storage. Call between long-running solve loops
    if memory becomes a concern."""
    _dealer_play_dist.cache_clear()
    dealer_outcome_dist.cache_clear()
    _ev_stand_inner.cache_clear()
    ev_hit_then_optimal.cache_clear()
    ev_double.cache_clear()


def cache_info():
    """Return a dict of cache_info() readouts for the memoized functions."""
    return {
        '_dealer_play_dist': _dealer_play_dist.cache_info(),
        'dealer_outcome_dist': dealer_outcome_dist.cache_info(),
        '_ev_stand_inner': _ev_stand_inner.cache_info(),
        'ev_hit_then_optimal': ev_hit_then_optimal.cache_info(),
        'ev_double': ev_double.cache_info(),
    }


__all__ = [
    'RANKS', 'RANK_TO_IDX', 'INITIAL_COMP',
    'comp_from_state', 'remove_rank', 'apply_card',
    'dealer_outcome_dist',
    'ev_stand', 'ev_hit_then_optimal', 'ev_double',
    'evaluate_actions', 'optimal_action_and_ev',
    'evaluate_state', 'clear_caches',
]
