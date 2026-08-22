"""Calibrate the betting net's quantile-mapped bet ramp.

The linear ev_lo/ev_hi ramp fitted during training mis-calibrates out of
sample (its search hit the boundary, and the deployed agent bet ~2x the
Hi-Lo counter's average -- inflating both edge and risk, and making the
money comparison apples-to-oranges).

This fits the ramp the robust way: play a stretch of hands, record the
pre-deal composition and pre-deal true count for each, then map the net's
predicted-EV RANKING onto the exact multiset of bet sizes the Hi-Lo ramp
would have used. The result has, by construction, the same min / max /
mean bet and the same total wagered as the Hi-Lo counter -- only the
allocation differs.

Run:
    python calibrate_bet_ramp.py --model bet_value_net.pt --hands 400000
"""
import argparse
import random

import numpy as np
import torch

from cardCounting import BlackjackEnv
from agents import (basic_strategy_with_deviations, bet_features_from_composition,
                    calibrate_quantile_ramp, count_based_bet, load_bet_value_net,
                    composition_based_bet, FEATURE_CARD_VALUES, pick_device)


def collect(hands, num_decks=8, seed=0):
    random.seed(seed); np.random.seed(seed)
    env = BlackjackEnv(num_decks=num_decks, min_bet=1, max_bet=1)
    total_cards = num_decks * 52
    feats, tcs = [], []
    for _ in range(hands):
        pre_shuffled = (env.deck.needs_shuffle()
                        or len(env.deck.dealt_cards) == 0)
        if pre_shuffled:
            comp = {v: 4 * num_decks for v in FEATURE_CARD_VALUES}
            remaining, pre_tc = total_cards, 0.0
        else:
            comp = env.deck.get_card_count()
            remaining = env.deck.cards_remaining()
            pre_tc = env.counter.running_count / max(env.counter.decks_remaining, 0.5)
        feats.append(bet_features_from_composition(comp, remaining, total_cards))
        tcs.append(pre_tc)
        state = env.reset(); env.bets[0] = 1
        done = False
        while not done:
            state, _, done = env.step(basic_strategy_with_deviations(state))
    return np.array(feats, dtype=np.float32), np.array(tcs, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='bet_value_net.pt')
    p.add_argument('--hands', type=int, default=400000)
    p.add_argument('--unit', type=int, default=50)
    p.add_argument('--max-units', type=int, default=50)
    p.add_argument('--seed', type=int, default=31337)
    a = p.parse_args()

    device = pick_device(None)
    net, old_ramp = load_bet_value_net(a.model, device=device)

    print(f"Collecting {a.hands:,} calibration hands...")
    feats, tcs = collect(a.hands, seed=a.seed)
    with torch.no_grad():
        pred = net(torch.from_numpy(feats).to(device)).cpu().numpy()

    hilo = np.array([count_based_bet(t, a.unit, a.max_units) for t in tcs],
                    dtype=np.float64)
    ramp = calibrate_quantile_ramp(pred, hilo, unit=a.unit)

    new_bets = np.array([composition_based_bet(e, ramp, a.unit, a.max_units)
                         for e in pred], dtype=np.float64)
    print(f"\n  Hi-Lo  bets: mean ${hilo.mean():.2f}  max ${hilo.max():.0f}  "
          f"total ${hilo.sum():,.0f}")
    print(f"  ramped bets: mean ${new_bets.mean():.2f}  max ${new_bets.max():.0f}  "
          f"total ${new_bets.sum():,.0f}")
    if 'ev_lo' in old_ramp:
        old = np.array([composition_based_bet(e, old_ramp, a.unit, a.max_units)
                        for e in pred], dtype=np.float64)
        print(f"  (old linear ramp mean was ${old.mean():.2f} -- "
              f"{old.mean()/hilo.mean():.2f}x the counter)")

    ckpt = torch.load(a.model, map_location='cpu')
    ckpt['ramp'] = ramp
    torch.save(ckpt, a.model)
    print(f"\nSaved quantile ramp into {a.model} "
          f"({len(ramp['ev_units'])} bet levels)")


if __name__ == '__main__':
    main()
