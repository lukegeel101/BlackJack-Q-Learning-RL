"""
Supervised distillation trainer for Option 2.

Trains a Dueling scalar-Q net to regress the EXPECTIMAX-derived per-action
EVs from `optimal_dataset.npz` (built by gen_optimal_dataset.py). The
trained network's Q-values represent the *true* expected return per
action under composition-aware optimal play, so:

  - argmax gives the true optimal action,
  - the spread between argmax and second-best is the *real* EV margin
    (no longer an artifact of how many episodes hit each state).

This sidesteps the variance-aversion failures of scalar-Q DQN trained
with sparse reward, and the convergence pains of QR-DQN training over
22M+ episodes -- supervised regression on exact labels converges in a
few minutes on a 100k-row dataset.

Run:
    python train_supervised.py                                  # use defaults
    python train_supervised.py --dataset optimal_dataset.npz \
        --epochs 80 --save-path dqn_agent.pt
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Match train_dqn.py's M2 perf tuning.
torch.set_num_threads(6)

from agents import (
    DuelingQNet, INPUT_DIM, pick_device,
    HIT, STAND, DOUBLE, SPLIT, SURRENDER,
)

ACTION_NAME = {HIT: 'HIT', STAND: 'STAND', DOUBLE: 'DOUBLE',
               SPLIT: 'SPLIT', SURRENDER: 'SURR'}


def load_dataset(path: str):
    """Load and lightly validate the .npz produced by gen_optimal_dataset.py.

    Drops any row that has a NaN in a valid action's EV -- those are rare
    edge cases the offline solver missed (~0.5-1% of states) and would
    poison the MSE loss.
    """
    npz = np.load(path)
    feats = npz['features']
    evs = npz['ev_actions']
    mask = npz['valid_mask']
    opt = npz['optimal']
    assert feats.shape[1] == INPUT_DIM, (
        f"Feature dim mismatch: dataset has {feats.shape[1]}, "
        f"agents.INPUT_DIM = {INPUT_DIM}. Regenerate the dataset.")
    assert evs.shape == (feats.shape[0], 5)
    assert mask.shape == evs.shape

    # Identify and drop rows where any valid action has NaN EV.
    masked_evs = np.where(mask, evs, 0.0)
    nan_rows = np.isnan(masked_evs).any(axis=1)
    n_bad = int(nan_rows.sum())
    if n_bad:
        print(f"  Dropping {n_bad:,} rows with NaN EVs "
              f"({n_bad / feats.shape[0] * 100:.2f}% of dataset)")
        keep = ~nan_rows
        feats = feats[keep]
        evs = evs[keep]
        mask = mask[keep]
        opt = opt[keep]

    # NaN EVs on INVALID actions: replace with 0 (mask zeroes them anyway).
    evs = np.where(mask, evs, 0.0).astype(np.float32)
    return feats, evs, mask.astype(np.float32), opt.astype(np.int64)


def train_one_epoch(net, opt_, feats, evs, mask, batch_size, device):
    """One pass over the (shuffled) training set. Returns mean loss."""
    n = feats.shape[0]
    perm = np.random.permutation(n)

    total_loss = 0.0
    total_n = 0
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        bf = torch.from_numpy(feats[idx]).to(device)
        be = torch.from_numpy(evs[idx]).to(device)
        bm = torch.from_numpy(mask[idx]).to(device)

        q = net(bf)                                # (B, 5)
        # Masked MSE: only count valid actions. Sum over valid actions in
        # each row, divide by the number of valid actions for that row,
        # then mean over the batch. Equivalent to per-element MSE weighted
        # by the validity mask.
        sq_err = (q - be) ** 2 * bm                # (B, 5)
        per_row_loss = sq_err.sum(dim=1) / bm.sum(dim=1).clamp(min=1.0)
        loss = per_row_loss.mean()

        opt_.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt_.step()

        total_loss += loss.item() * len(idx)
        total_n += len(idx)
    return total_loss / total_n


@torch.no_grad()
def evaluate(net, feats, evs, mask, opt_labels, batch_size, device):
    """Validation metrics:
      - mean MSE on valid actions.
      - argmax accuracy: net's argmax over valid actions == optimal label.
      - mean EV gap: actual EV of the picked action minus EV of the
        labeled optimal action. 0 means perfectly matched, negative
        means we're picking sub-optimal actions.
    """
    n = feats.shape[0]
    total_loss = 0.0
    total_correct = 0
    total_ev_gap = 0.0
    total_n = 0
    for i in range(0, n, batch_size):
        bf = torch.from_numpy(feats[i:i + batch_size]).to(device)
        be = torch.from_numpy(evs[i:i + batch_size]).to(device)
        bm = torch.from_numpy(mask[i:i + batch_size]).to(device)
        bo = torch.from_numpy(opt_labels[i:i + batch_size]).to(device)

        q = net(bf)
        sq_err = (q - be) ** 2 * bm
        per_row_loss = sq_err.sum(dim=1) / bm.sum(dim=1).clamp(min=1.0)
        total_loss += per_row_loss.sum().item()

        # Masked argmax: set invalid actions to -inf.
        q_masked = q.masked_fill(bm == 0, float('-inf'))
        pred = q_masked.argmax(dim=1)
        total_correct += (pred == bo).sum().item()

        # EV gap: true EV of net's pick minus true EV of optimal pick.
        true_picked_ev = be.gather(1, pred.unsqueeze(1)).squeeze(1)
        true_best_ev = be.gather(1, bo.unsqueeze(1)).squeeze(1)
        total_ev_gap += (true_picked_ev - true_best_ev).sum().item()

        total_n += bf.size(0)

    return (total_loss / total_n,
            total_correct / total_n,
            total_ev_gap / total_n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='optimal_dataset.npz')
    parser.add_argument('--save-path', type=str, default='dqn_agent.pt')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--val-frac', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--patience', type=int, default=15,
                        help='Stop early if val EV gap fails to improve '
                             'for this many epochs (-1 = disabled).')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    print(f"Device: {device}  threads={torch.get_num_threads()}")

    print(f"Loading dataset: {args.dataset}")
    feats, evs, mask, opt_labels = load_dataset(args.dataset)
    n = feats.shape[0]
    print(f"  {n:,} states, {INPUT_DIM}-dim features")

    # Print optimal action distribution.
    counts = np.bincount(opt_labels, minlength=5)
    print("  Optimal-action distribution:")
    for a, c in enumerate(counts):
        print(f"    {ACTION_NAME[a]:8s} {c:>7,} ({c/n*100:5.2f}%)")

    # Train/val split. Deterministic given args.seed.
    perm = np.random.permutation(n)
    val_size = int(n * args.val_frac)
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    f_tr, f_va = feats[train_idx], feats[val_idx]
    e_tr, e_va = evs[train_idx], evs[val_idx]
    m_tr, m_va = mask[train_idx], mask[val_idx]
    o_tr, o_va = opt_labels[train_idx], opt_labels[val_idx]
    print(f"  train: {len(f_tr):,}  val: {len(f_va):,}")

    net = DuelingQNet().to(device)
    opt_ = optim.Adam(net.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt_, T_max=args.epochs)

    best_gap = float('-inf')   # closer to 0 (or more positive) is better
    best_state = None
    epochs_since_improve = 0

    print(f"\nTraining DuelingQNet ({sum(p.numel() for p in net.parameters()):,} params) "
          f"for up to {args.epochs} epochs")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        net.train()
        train_loss = train_one_epoch(net, opt_, f_tr, e_tr, m_tr,
                                     args.batch_size, device)
        net.eval()
        val_loss, val_acc, val_gap = evaluate(
            net, f_va, e_va, m_va, o_va, args.batch_size, device)
        sched.step()

        # Track best-on-val by EV gap (closest to 0 from below).
        if val_gap > best_gap:
            best_gap = val_gap
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            epochs_since_improve = 0
            marker = '*'
        else:
            epochs_since_improve += 1
            marker = ' '

        elapsed = time.time() - t0
        print(f"  ep {epoch:>3}/{args.epochs}  "
              f"train_loss={train_loss:.5f}  "
              f"val_loss={val_loss:.5f}  "
              f"val_acc={val_acc*100:5.2f}%  "
              f"val_ev_gap={val_gap:+.5f} {marker}  "
              f"lr={sched.get_last_lr()[0]:.2e}  "
              f"[{elapsed:.0f}s]")

        if args.patience > 0 and epochs_since_improve >= args.patience:
            print(f"  (early stop -- no improvement for {args.patience} epochs)")
            break

    # Save best-val checkpoint.
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    torch.save({'net': net.state_dict(),
                'architecture': 'DuelingQNet',
                'input_dim': INPUT_DIM,
                'dataset': args.dataset,
                'best_val_ev_gap': best_gap},
               args.save_path)
    print(f"\nSaved -> {args.save_path}")
    print(f"  best val EV gap: {best_gap:+.5f}  "
          f"(0 = always picks optimal; negative = sometimes sub-optimal)")


if __name__ == '__main__':
    main()
