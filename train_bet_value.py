"""Train the composition-aware betting model and compare it head-to-head
against the Hi-Lo true count as a betting signal.

The question: does the full shoe composition predict the upcoming round's EV
better than the 1-D Hi-Lo true count -- and does that translate into a real
betting edge?

Two comparisons, both on held-out data:

1. SIGNAL QUALITY (low variance):
   - betting correlation: corr(signal, realized outcome) for each signal.
   - held-out R^2: how much of the outcome variance each signal explains.
     The Hi-Lo signal is given its BEST possible use -- a nonparametric
     (binned) mean-outcome-vs-true-count curve fit on the training set --
     so this is count-vs-composition, not linear-count-vs-composition.

2. REALIZED EDGE, risk-matched (rank-matched betting):
   The composition bettor is forced to use the EXACT SAME multiset of bet
   sizes as the Hi-Lo ramp -- identical min, max, mean, and total wagered --
   but allocated to hands by its own signal ranking instead of the count's.
   Any profit difference is purely better ALLOCATION of the same bets.

Run:
    python train_bet_value.py --train bet_train.npz --eval bet_eval.npz \
        --save-path bet_value_net.pt
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents import BetValueNet, count_based_bet, pick_device


def bootstrap_diff(a, b, weights_a, weights_b, n_boot=1000, seed=0):
    """Bootstrap 95% CI for mean(weights_a*a) - mean(weights_b*b) over hands
    (paired: same resampled indices)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(n_boot)
    pa, pb = weights_a * a, weights_b * b
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = pa[idx].mean() - pb[idx].mean()
    return np.percentile(diffs, [2.5, 97.5])


def fit_tc_curve(tc, out, lo=-10, hi=15):
    """Best nonparametric use of the count: mean outcome per integer TC bin."""
    tcf = np.clip(np.floor(tc).astype(int), lo, hi)
    curve = {}
    for b in range(lo, hi + 1):
        m = tcf == b
        curve[b] = out[m].mean() if m.any() else out.mean()
    glob = out.mean()
    return lambda t: np.array([curve.get(int(np.clip(np.floor(x), lo, hi)), glob)
                               for x in t], dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='bet_train.npz')
    p.add_argument('--eval', default='bet_eval.npz')
    p.add_argument('--save-path', default='bet_value_net.pt')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--unit', type=int, default=50)
    p.add_argument('--max-units', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = pick_device(None)

    tr = np.load(a.train); ev = np.load(a.eval)
    Xtr, ytr = tr['bet_features'].astype(np.float32), tr['outcome'].astype(np.float32)
    Xev, yev = ev['bet_features'].astype(np.float32), ev['outcome'].astype(np.float32)
    tc_ev = ev['true_count'].astype(np.float32)
    tc_tr = tr['true_count'].astype(np.float32)
    print(f"train {len(Xtr):,}  eval {len(Xev):,}  feat_dim {Xtr.shape[1]}")

    # ---- train BetValueNet to regress outcome from composition ----
    net = BetValueNet(hidden_dim=a.hidden).to(device)
    opt = optim.Adam(net.parameters(), lr=a.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    Xt = torch.from_numpy(Xtr).to(device); yt = torch.from_numpy(ytr).to(device)
    n = len(Xt)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        net.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, a.batch_size):
            idx = perm[i:i + a.batch_size]
            loss = ((net(Xt[idx]) - yt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    print(f"trained in {time.time()-t0:.0f}s")

    net.eval()
    with torch.no_grad():
        pred_ev = net(torch.from_numpy(Xev).to(device)).cpu().numpy()

    # ---- 1. signal quality ----
    def corr(x, y): return float(np.corrcoef(x, y)[0, 1])
    tc_curve = fit_tc_curve(tc_tr, ytr)
    tc_pred = tc_curve(tc_ev)               # count's best EV estimate
    mse0 = ((yev - yev.mean()) ** 2).mean()
    mse_tc = ((yev - tc_pred) ** 2).mean()
    mse_net = ((yev - pred_ev) ** 2).mean()
    print("\n=== 1. SIGNAL QUALITY (held-out) ===")
    print(f"  betting corr  Hi-Lo TC : {corr(tc_ev, yev):+.4f}")
    print(f"  betting corr  composition net : {corr(pred_ev, yev):+.4f}")
    print(f"  R^2 explaining outcome  Hi-Lo : {1-mse_tc/mse0:+.5f}   "
          f"composition : {1-mse_net/mse0:+.5f}")
    # correlation between the two EV estimates (how much composition departs
    # from what the count can express)
    print(f"  corr(Hi-Lo EV-estimate, composition EV-estimate) = "
          f"{corr(tc_pred, pred_ev):.4f}")

    # ---- 2. risk-matched (rank-matched) money comparison ----
    hilo_bets = np.array([count_based_bet(t, a.unit, a.max_units) for t in tc_ev],
                         dtype=np.float64)
    # rank-match: same bet multiset, allocated by composition signal ranking
    order = np.argsort(pred_ev, kind='stable')
    comp_bets = np.empty_like(hilo_bets)
    comp_bets[order] = np.sort(hilo_bets)
    prof_hilo = (hilo_bets * yev).sum()
    prof_comp = (comp_bets * yev).sum()
    wag = hilo_bets.sum()
    ci = bootstrap_diff(yev, yev, comp_bets, hilo_bets, seed=a.seed)
    print("\n=== 2. RISK-MATCHED MONEY (identical bet multiset) ===")
    print(f"  hands {len(yev):,}   avg bet ${hilo_bets.mean():.0f}   "
          f"total wagered ${wag:,.0f}")
    print(f"  Hi-Lo betting        edge {prof_hilo/wag*100:+.3f}%   "
          f"profit/hand ${prof_hilo/len(yev):+.4f}")
    print(f"  composition betting  edge {prof_comp/wag*100:+.3f}%   "
          f"profit/hand ${prof_comp/len(yev):+.4f}")
    print(f"  advantage per hand   ${(prof_comp-prof_hilo)/len(yev):+.4f}  "
          f"(95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}])")

    # ---- calibrate a deployable ramp (match Hi-Lo average bet) ----
    ev_lo = float(np.percentile(pred_ev, 55))
    target = hilo_bets.mean()
    best = None
    for ev_hi in np.linspace(ev_lo + 0.005, ev_lo + 0.10, 60):
        frac = np.clip((pred_ev - ev_lo) / (ev_hi - ev_lo), 0, 1)
        units = np.clip(np.round(1 + frac * (a.max_units - 1)), 1, a.max_units)
        mb = (units * a.unit).mean()
        if best is None or abs(mb - target) < best[0]:
            best = (abs(mb - target), ev_hi)
    ramp = {'ev_lo': ev_lo, 'ev_hi': float(best[1])}

    torch.save({'net': net.state_dict(), 'ramp': ramp,
                'input_dim': Xtr.shape[1]}, a.save_path)
    print(f"\nSaved -> {a.save_path}  ramp={ramp}")


if __name__ == '__main__':
    main()
