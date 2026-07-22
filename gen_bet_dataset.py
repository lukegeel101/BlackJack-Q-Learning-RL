"""Build a dataset for the composition-aware BETTING model.

Plays flat-bet (1 unit) blackjack under basic strategy + Illustrious-18 and,
for every round, records:

  * bet_features : the PRE-DEAL shoe composition features (13 rank fractions
                   + penetration + fraction remaining) -- what you know when
                   you place the bet.
  * true_count   : the PRE-DEAL Hi-Lo true count (the classic betting signal,
                   kept for a head-to-head comparison).
  * outcome      : the realized result of the round per unit of initial bet
                   (win +1, blackjack +1.5, lost double -2, ...). Its
                   expectation given the pre-deal composition IS the round's
                   EV -- the quantity a bettor wants to predict.

Because the bet is flat and the play policy is fixed, `outcome` is an
unbiased, noisy sample of the composition-conditional EV. Averaged over many
rounds a network recovers that EV from the composition -- a strictly richer
signal than the 1-D true count.

Run:
    python gen_bet_dataset.py --hands 800000 --out bet_dataset.npz
"""
import argparse
import random
import time

import numpy as np

from cardCounting import BlackjackEnv
from agents import (basic_strategy_with_deviations, bet_features_from_composition,
                    BET_INPUT_DIM, FEATURE_CARD_VALUES)

_FULL_SHOE = None


def _fresh_composition(num_decks):
    return {v: 4 * num_decks for v in FEATURE_CARD_VALUES}


def collect(hands, num_decks=8, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    env = BlackjackEnv(num_decks=num_decks, min_bet=1, max_bet=1)

    feats = np.zeros((hands, BET_INPUT_DIM), dtype=np.float32)
    tcs = np.zeros(hands, dtype=np.float32)
    out = np.zeros(hands, dtype=np.float32)

    total_cards = num_decks * 52
    t0 = time.time()
    for i in range(hands):
        # PRE-DEAL snapshot: if the shoe is about to be shuffled (or this is
        # the very first hand), the pre-deal shoe is a fresh full shoe.
        pre_shuffled = (env.deck.needs_shuffle()
                        or len(env.deck.dealt_cards) == 0)
        if pre_shuffled:
            comp = _fresh_composition(num_decks)
            remaining = total_cards
            pre_tc = 0.0
        else:
            comp = env.deck.get_card_count()
            remaining = env.deck.cards_remaining()
            pre_running = env.counter.running_count
            pre_decks = max(env.counter.decks_remaining, 0.5)
            pre_tc = pre_running / pre_decks

        feats[i] = bet_features_from_composition(comp, remaining, total_cards)
        tcs[i] = pre_tc

        state = env.reset()
        env.bets[0] = 1  # flat 1-unit bet; label is per-unit result
        done = False
        r_sum = 0.0
        while not done:
            a = basic_strategy_with_deviations(state)
            state, r, done = env.step(a)
            r_sum += r
        out[i] = r_sum

        if (i + 1) % 200000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1:,}/{hands:,} hands ({rate:.0f}/s)", flush=True)

    return feats, tcs, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hands', type=int, default=800000)
    p.add_argument('--num-decks', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, default='bet_dataset.npz')
    a = p.parse_args()

    print(f"Collecting {a.hands:,} flat-bet basic+I18 rounds "
          f"(seed {a.seed})...")
    feats, tcs, out = collect(a.hands, a.num_decks, a.seed)
    print(f"  mean outcome/unit = {out.mean():+.4f} "
          f"(flat-bet basic+I18 house edge)")
    print(f"  corr(true_count, outcome) = {np.corrcoef(tcs, out)[0,1]:+.4f}")
    np.savez(a.out, bet_features=feats, true_count=tcs, outcome=out)
    print(f"Saved -> {a.out}")


if __name__ == '__main__':
    main()
