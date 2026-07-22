"""Score betting signals against the MC ground-truth EV benchmark (mc_ev.npz).

Because the payoff here is the MC-estimated TRUE per-round EV (not a noisy
single-hand result), both the betting correlation and the realized edge are
low-variance -- so a real difference between signals, if it exists, shows up.

For each model we report:
  * betting correlation: corr(signal, true_ev)
  * EV-prediction RMSE  : how close the model's EV estimate is to true EV
  * edge, rank-matched  : sum(bet*true_ev)/sum(bet) with the composition
                          bettor forced onto the Hi-Lo bet multiset.
"""
import argparse
import numpy as np
import torch
from agents import load_bet_value_net, count_based_bet, BetValueNet


def tc_curve_fit(tc, ev, lo=-12, hi=16):
    tcf = np.clip(np.floor(tc).astype(int), lo, hi)
    curve = {b: ev[tcf == b].mean() if (tcf == b).any() else ev.mean()
             for b in range(lo, hi + 1)}
    glob = ev.mean()
    return lambda t: np.array([curve.get(int(np.clip(np.floor(x), lo, hi)), glob)
                               for x in t], dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mc', default='mc_ev.npz')
    p.add_argument('--models', nargs='+', default=[])
    p.add_argument('--unit', type=int, default=50)
    p.add_argument('--max-units', type=int, default=50)
    a = p.parse_args()

    d = np.load(a.mc)
    feats, tc, true_ev = d['bet_features'], d['true_count'], d['true_ev']
    n = len(true_ev)
    print(f"benchmark: {n:,} compositions   true-EV std {true_ev.std():.4f}\n")

    def corr(x, y): return float(np.corrcoef(x, y)[0, 1])

    # Hi-Lo reference
    hilo_bets = np.array([count_based_bet(t, a.unit, a.max_units) for t in tc],
                         dtype=np.float64)
    hilo_edge = (hilo_bets * true_ev).sum() / hilo_bets.sum() * 100
    tc_pred = tc_curve_fit(tc, true_ev)(tc)
    print(f"{'signal':<30}{'bet-corr':>10}{'EV-RMSE':>10}{'edge%':>9}")
    print(f"{'Hi-Lo true count':<30}{corr(tc, true_ev):>10.4f}"
          f"{np.sqrt(((tc_pred-true_ev)**2).mean()):>10.4f}{hilo_edge:>9.3f}")

    for m in a.models:
        net, _ = load_bet_value_net(m)
        with torch.no_grad():
            pred = net(torch.from_numpy(feats.astype(np.float32))).numpy()
        # rank-match onto the Hi-Lo bet multiset
        order = np.argsort(pred, kind='stable')
        cb = np.empty_like(hilo_bets); cb[order] = np.sort(hilo_bets)
        edge = (cb * true_ev).sum() / cb.sum() * 100
        rmse = np.sqrt(((pred - true_ev) ** 2).mean())
        # paired bootstrap CI of (composition edge - Hi-Lo edge) over comps
        rng = np.random.default_rng(0)
        diffs = np.empty(2000)
        for k in range(2000):
            idx = rng.integers(0, n, n)
            ce = (cb[idx] * true_ev[idx]).sum() / cb[idx].sum() * 100
            he = (hilo_bets[idx] * true_ev[idx]).sum() / hilo_bets[idx].sum() * 100
            diffs[k] = ce - he
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f"{m.split('/')[-1]:<30}{corr(pred, true_ev):>10.4f}"
              f"{rmse:>10.4f}{edge:>9.3f}   vs Hi-Lo {edge-hilo_edge:+.3f}% "
              f"[{lo:+.3f},{hi:+.3f}]")


if __name__ == '__main__':
    main()
