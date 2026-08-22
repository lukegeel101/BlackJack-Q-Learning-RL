"""Monte-Carlo ground-truth EV for pre-deal shoe compositions.

The single-hand outcome is far too noisy to measure whether a betting signal
predicts EV well -- so we estimate the TRUE per-round EV of a composition by
replaying that exact shoe thousands of times under basic strategy + I18 and
averaging. This is how "betting correlation" (the classic ~0.97 for Hi-Lo) is
actually defined: correlation of the signal with the true EV, not with noisy
realized results.

Output: an .npz of (bet_features, true_count, true_ev, cards_remaining) for a
set of compositions sampled from natural play -- a low-noise benchmark on
which any betting signal can be scored.

Run:
    python mc_true_ev.py --num-comps 1200 --reps 8000 --out mc_ev.npz
"""
import argparse
import multiprocessing as mp
import random
import time

import numpy as np

HILO = {'2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 0, '8': 0, '9': 0,
        '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1}


def _setup_deck(env, comp, Card):
    cards = []
    for v, c in comp.items():
        for _ in range(int(c)):
            cards.append(Card('S', v))
    random.shuffle(cards)
    env.deck.cards = cards
    env.deck.dealt_cards = []
    total = len(cards)
    rc = -sum(HILO[v] * int(c) for v, c in comp.items())
    decks = max(total / 52.0, 0.5)
    env.counter.running_count = rc
    env.counter.decks_remaining = decks
    env.counter.true_count = rc / decks


def _play_round(env, comp, Card, basic_fn):
    _setup_deck(env, comp, Card)
    state = env.reset()
    env.bets[0] = 1
    done = False
    r = 0.0
    while not done:
        a = basic_fn(state)
        state, rr, done = env.step(a)
        r += rr
    return r


def _sample_compositions(n, gap, seed):
    """Sample n pre-deal compositions from natural flat-bet basic+I18 play."""
    from cardCounting import BlackjackEnv
    from agents import basic_strategy_with_deviations, FEATURE_CARD_VALUES
    random.seed(seed); np.random.seed(seed)
    env = BlackjackEnv(num_decks=8, min_bet=1, max_bet=1)
    comps = []
    hands_since = 0
    while len(comps) < n:
        pre_shuffled = env.deck.needs_shuffle() or len(env.deck.dealt_cards) == 0
        if not pre_shuffled and hands_since >= gap:
            comp = dict(env.deck.get_card_count())
            comp = {v: comp.get(v, 0) for v in FEATURE_CARD_VALUES}
            rc = env.counter.running_count
            decks = max(env.counter.decks_remaining, 0.5)
            comps.append((comp, rc / decks, sum(comp.values())))
            hands_since = 0
        state = env.reset()
        env.bets[0] = 1
        done = False
        while not done:
            state, _, done = env.step(basic_strategy_with_deviations(state))
        hands_since += 1
    return comps


def _worker(args):
    wid, comps, reps, seed = args
    import random as _r
    _r.seed(seed)
    from cardCounting import BlackjackEnv, Card
    from agents import (basic_strategy_with_deviations,
                        bet_features_from_composition, BET_INPUT_DIM)
    env = BlackjackEnv(num_decks=8, min_bet=1, max_bet=1)
    feats = np.zeros((len(comps), BET_INPUT_DIM), dtype=np.float32)
    tcs = np.zeros(len(comps), dtype=np.float32)
    evs = np.zeros(len(comps), dtype=np.float32)
    rem = np.zeros(len(comps), dtype=np.float32)
    t0 = time.time()
    for i, (comp, tc, total) in enumerate(comps):
        s = 0.0
        for _ in range(reps):
            s += _play_round(env, comp, Card, basic_strategy_with_deviations)
        evs[i] = s / reps
        feats[i] = bet_features_from_composition(comp, total, 416)
        tcs[i] = tc
        rem[i] = total
        if wid == 0 and (i + 1) % 25 == 0:
            rate = (i + 1) * reps / (time.time() - t0)
            print(f"  [w0] {i+1}/{len(comps)} comps ({rate:.0f} rounds/s)",
                  flush=True)
    return feats, tcs, evs, rem


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num-comps', type=int, default=1200)
    p.add_argument('--reps', type=int, default=8000)
    p.add_argument('--gap', type=int, default=15,
                   help='hands to skip between sampled compositions')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--out', default='mc_ev.npz')
    a = p.parse_args()

    print(f"Sampling {a.num_comps:,} compositions...")
    comps = _sample_compositions(a.num_comps, a.gap, a.seed)
    print(f"MC EV: {a.reps:,} reps each across {a.workers} workers "
          f"({a.num_comps * a.reps / 1e6:.1f}M rounds total)")

    chunks = [comps[i::a.workers] for i in range(a.workers)]
    wargs = [(w, chunks[w], a.reps, a.seed + 100 * w) for w in range(a.workers)]
    t0 = time.time()
    if a.workers == 1:
        res = [_worker(wargs[0])]
    else:
        ctx = mp.get_context('spawn')
        with ctx.Pool(a.workers) as pool:
            res = pool.map(_worker, wargs)
    feats = np.concatenate([r[0] for r in res])
    tcs = np.concatenate([r[1] for r in res])
    evs = np.concatenate([r[2] for r in res])
    rem = np.concatenate([r[3] for r in res])
    print(f"done in {(time.time()-t0)/60:.1f} min")
    print(f"  EV range [{evs.min():+.4f}, {evs.max():+.4f}]  "
          f"mean {evs.mean():+.4f}  std {evs.std():.4f}")
    print(f"  corr(true_count, true_ev) = {np.corrcoef(tcs, evs)[0,1]:.4f}")
    np.savez(a.out, bet_features=feats, true_count=tcs, true_ev=evs,
             cards_remaining=rem)
    print(f"Saved -> {a.out}")


if __name__ == '__main__':
    main()
