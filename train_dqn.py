"""
Train Agent 3's DQN -- Dueling QR-DQN with n-step returns and Polyak
target updates.

This is a substantial upgrade over the previous scalar-Q trainer, putting
the three improvements I owed you into one place:

  1. **Variance reduction** -- n-step returns (n=3) accumulate reward
     across multiple steps before bootstrapping, and a Polyak-averaged
     target network smooths the target values across training steps
     instead of hard-copying every N steps. Both reduce the target
     variance that pulled scalar-Q's DOUBLE estimates around.

  2. **Distributional DQN (QR-DQN)** -- the network predicts 32 quantiles
     of the RETURN DISTRIBUTION per action, not a scalar mean. The loss
     is the quantile-Huber loss against the target distribution. Argmax
     still uses the mean of the quantiles, but gradient updates respect
     the full spread. This directly fixes scalar-Q's variance-aversion
     on doubles -- a doubled-bet hand's reward distribution has a wide
     spread but a clearly positive mean, and QR-DQN captures both.

  3. **Dueling architecture** -- network has separate value V(s) and
     advantage A(s,a) heads, recombined as Q = V + (A - mean_a A). The
     value stream sees state-only features (penetration, count, etc.),
     the advantage stream sees action-relative features. This is the
     "advantage framing" of #3 -- A(s,a) is now an explicit output that
     the network learns to nudge separately from V(s), so it can't
     accidentally drag the basic-strategy action's Q-value down.

Same env, same features, same teacher (basic + Illustrious-18) as before.

Run:
    python train_dqn.py                           # 5M episodes default
    python train_dqn.py --episodes 1000000        # shorter
"""

import argparse
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# M2 perf tuning -- see prior benchmark in this file's git history.
torch.set_num_threads(6)

from cardCounting import BlackjackEnv
from agents import (
    DuelingQRDQN, state_to_features,
    basic_strategy_action, basic_strategy_with_deviations,
    pick_device, _valid_actions,
)


BASE_BET = 50
N_QUANTILES = DuelingQRDQN.N_QUANTILES


# ----------------------------------------------------------------------
# Replay buffer
# ----------------------------------------------------------------------

class ReplayBuffer:
    """Pre-allocated numpy buffer. Sampling is uniform with replacement
    via np.random.randint -- the collision rate at 200k buffer / 512
    batch is 0.13%, not worth the cost of replace=False (60x slower)."""

    def __init__(self, max_size: int, input_dim: int, n_actions: int):
        self.max_size = max_size
        self.size = 0
        self.ptr = 0
        self.feats      = np.zeros((max_size, input_dim), dtype=np.float32)
        self.next_feats = np.zeros((max_size, input_dim), dtype=np.float32)
        self.actions    = np.zeros(max_size, dtype=np.int64)
        self.rewards    = np.zeros(max_size, dtype=np.float32)
        self.dones      = np.zeros(max_size, dtype=np.float32)
        # We store n-step returns and a discount factor for the bootstrap.
        # Bootstrap target uses `next_feats` and `gammas`:
        #     q_target = reward_nstep + gammas * (1 - done) * Q_target(s_n)
        # `gammas` will be γ^n for full n-step rollouts and lower powers
        # of γ when the episode terminates inside the window.
        self.gammas     = np.zeros(max_size, dtype=np.float32)
        self.next_masks = np.zeros((max_size, n_actions), dtype=np.float32)

    def add(self, feats, action, reward, next_feats, done, gamma_eff, next_mask):
        i = self.ptr
        self.feats[i] = feats
        self.next_feats[i] = next_feats
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.gammas[i] = gamma_eff
        self.next_masks[i] = next_mask
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, batch_size)
        return (self.feats[idx], self.actions[idx], self.rewards[idx],
                self.next_feats[idx], self.dones[idx], self.gammas[idx],
                self.next_masks[idx])


# ----------------------------------------------------------------------
# N-step return accumulator
# ----------------------------------------------------------------------

class NStepAccumulator:
    """Build n-step transitions from per-step transitions.

    For a sequence (s0,a0,r0), (s1,a1,r1), ..., (sn,...) the n-step
    transition for index 0 is:
        s = s0
        a = a0
        n-step reward = r0 + γ r1 + γ² r2 + ... + γ^(n-1) r_{n-1}
        next_s = s_n
        done = (episode ended by step n-1)
        gamma_eff = γ^n  (or γ^k if episode ended at step k < n)
    """

    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self.buf = deque(maxlen=n)

    def add(self, feats, action, reward, next_feats, done, next_mask):
        """Add one step. Returns a list of n-step transitions that are
        now complete (0 or 1 transitions per call, plus a flush at end
        of episode)."""
        self.buf.append((feats, action, reward, next_feats, done, next_mask))
        out = []
        if len(self.buf) == self.n:
            out.append(self._build())
            self.buf.popleft()
        if done:
            # Episode ended: flush remaining transitions with truncated
            # n-step returns.
            while self.buf:
                out.append(self._build())
                self.buf.popleft()
        return out

    def _build(self):
        """Build the n-step transition rooted at self.buf[0]."""
        s, a, _, _, _, _ = self.buf[0]
        r_total = 0.0
        gamma_pow = 1.0
        next_feats, next_mask, done = None, None, 0.0
        for (_, _, r, nf, d, nm) in self.buf:
            r_total += gamma_pow * r
            next_feats, next_mask = nf, nm
            if d:
                done = 1.0
                gamma_pow *= self.gamma
                break
            gamma_pow *= self.gamma
        # gamma_pow at this point = γ^k where k is the number of rewards
        # accumulated (== n for a full window, < n if we broke on done).
        gamma_eff = gamma_pow
        return (s, a, r_total, next_feats, done, gamma_eff, next_mask)


# ----------------------------------------------------------------------
# Quantile-Huber loss for QR-DQN
# ----------------------------------------------------------------------

def quantile_huber_loss(predicted_q: torch.Tensor,
                        target_q: torch.Tensor,
                        kappa: float = 1.0) -> torch.Tensor:
    """Standard QR-DQN loss.

    Args:
        predicted_q: (B, N) quantile predictions.
        target_q:    (B, N) target quantile values.
        kappa: Huber threshold.

    Returns: scalar loss.
    """
    B, N = predicted_q.shape
    device = predicted_q.device
    # tau_i = (i + 0.5) / N -- the quantile fractions our network predicts.
    tau = (torch.arange(N, device=device).float() + 0.5) / N      # (N,)

    # Pairwise differences: u[b, i, j] = target_q[b, j] - predicted_q[b, i].
    diff = target_q.unsqueeze(1) - predicted_q.unsqueeze(2)        # (B, N, N)

    # Huber.
    abs_d = diff.abs()
    huber = torch.where(abs_d <= kappa,
                        0.5 * diff.pow(2),
                        kappa * (abs_d - 0.5 * kappa))             # (B, N, N)

    # Asymmetric quantile weight: |tau_i - 1{u < 0}|.
    # tau is indexed by i (the PREDICTED quantile), so its dim is the i-axis.
    tau_view = tau.view(1, N, 1)
    weight = (tau_view - (diff < 0).float()).abs()                 # (B, N, N)

    # Average over j (target quantile index), sum over i (predicted),
    # mean over batch.
    loss = (weight * huber).mean(dim=2).sum(dim=1).mean()
    return loss


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _features_np(state) -> np.ndarray:
    return state_to_features(state).cpu().numpy()


def _valid_action_mask(state) -> np.ndarray:
    mask = np.zeros(DuelingQRDQN.OUTPUT_DIM, dtype=np.float32)
    for a in _valid_actions(state):
        mask[a] = 1.0
    return mask


# ----------------------------------------------------------------------
# Warm-start
# ----------------------------------------------------------------------

def warm_start(net: DuelingQRDQN,
               opt: optim.Optimizer,
               device: torch.device,
               num_batches: int,
               batch_size: int = 256,
               teacher_uses_deviations: bool = True):
    """Pre-train the network so its mean-Q output matches a one-hot
    target on the teacher's action (basic + Illustrious-18). Each
    quantile is pulled toward the same target value so the predicted
    return distribution starts tight around 1.0 (basic) or 0.0 (others),
    and RL then spreads the quantiles to capture real return variance."""
    print(f"Warm-start: imitating teacher for {num_batches:,} batches "
          f"(teacher = basic{'+I18' if teacher_uses_deviations else ''})")
    teacher_fn = (basic_strategy_with_deviations
                  if teacher_uses_deviations else basic_strategy_action)
    net.train()
    env = BlackjackEnv(num_decks=8, min_bet=BASE_BET, max_bet=BASE_BET)
    state = env.reset()

    for b in range(num_batches):
        feat_buf = np.zeros((batch_size, DuelingQRDQN.INPUT_DIM), dtype=np.float32)
        action_buf = np.zeros(batch_size, dtype=np.int64)
        i = 0
        while i < batch_size:
            if env.dealer_pre_bj:
                env.step(0)
                state = env.reset()
                continue
            feat_buf[i] = _features_np(state)
            action_buf[i] = teacher_fn(state)
            next_state, _, done = env.step(action_buf[i])
            if done or next_state is None:
                state = env.reset()
            else:
                state = next_state
            i += 1

        bf = torch.from_numpy(feat_buf).to(device)
        ba = torch.from_numpy(action_buf).to(device)
        # Quantile-level one-hot target: teacher action's quantiles = 1.0,
        # others = 0.0. All quantiles share the same target so the
        # predicted distribution starts as a delta at the basic-action
        # value.
        q = net.forward(bf)                                       # (B, A, Q)
        target = torch.zeros_like(q)
        target.scatter_(1, ba.view(-1, 1, 1).expand(-1, 1, N_QUANTILES), 1.0)

        loss = F.mse_loss(q, target)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if (b + 1) % 100 == 0:
            with torch.no_grad():
                acc = (net.q_values(bf).argmax(dim=1) == ba).float().mean().item()
            print(f"  warm-start batch {b+1}/{num_batches} | "
                  f"loss={loss.item():.4f} | match-rate={acc*100:.1f}%")


# ----------------------------------------------------------------------
# RL training
# ----------------------------------------------------------------------

def polyak_update(target_net: nn.Module, online_net: nn.Module, tau: float):
    """Soft target update: target <- tau * online + (1 - tau) * target."""
    with torch.no_grad():
        for t, o in zip(target_net.parameters(), online_net.parameters()):
            t.data.mul_(1.0 - tau).add_(o.data, alpha=tau)


def rl_train(net: DuelingQRDQN,
             target_net: DuelingQRDQN,
             opt: optim.Optimizer,
             device: torch.device,
             args):
    env = BlackjackEnv(num_decks=8, min_bet=BASE_BET, max_bet=BASE_BET)
    buf = ReplayBuffer(args.buffer_size, DuelingQRDQN.INPUT_DIM,
                       DuelingQRDQN.OUTPUT_DIM)
    accumulator = NStepAccumulator(args.n_step, args.gamma)

    epsilon = args.epsilon_start
    eps_per_episode = (args.epsilon_start - args.epsilon_end) / max(
        args.epsilon_decay_episodes, 1)

    step_count = 0
    rewards_window = np.zeros(2000, dtype=np.float32)
    rew_ptr = 0; rew_filled = 0
    skipped_peek_bj = 0
    print_every = max(1, args.episodes // 200)
    t0 = time.time()

    print(f"\nRL training: {args.episodes:,} episodes "
          f"(ε {args.epsilon_start} -> {args.epsilon_end} "
          f"over {args.epsilon_decay_episodes:,})")
    print(f"  Dueling QR-DQN, N_QUANTILES={N_QUANTILES}, n-step={args.n_step}, "
          f"polyak_tau={args.polyak_tau}")
    print(f"  batch={args.batch_size}  buffer={args.buffer_size:,}  "
          f"lr={args.lr}  train_every={args.train_every}")

    for episode in range(args.episodes):
        state = env.reset()

        if env.dealer_pre_bj:
            env.step(0)
            skipped_peek_bj += 1
            continue

        ep_reward = 0.0
        cur_feats = _features_np(state)
        # Reset the accumulator for each episode (n-step transitions
        # don't cross episode boundaries).
        accumulator.buf.clear()

        while state is not None:
            valid = _valid_actions(state)

            if random.random() < epsilon:
                action = random.choice(valid)
            else:
                with torch.no_grad():
                    cur_t = torch.from_numpy(cur_feats).to(device)
                    qm = net.q_values(cur_t).squeeze(0).cpu().numpy()
                masked = qm.copy()
                for a in range(DuelingQRDQN.OUTPUT_DIM):
                    if a not in valid:
                        masked[a] = -np.inf
                action = int(np.argmax(masked))

            next_state, reward, done = env.step(action)
            ep_reward += reward

            if next_state is not None and not done:
                next_feats = _features_np(next_state)
                next_mask = _valid_action_mask(next_state)
            else:
                next_feats = np.zeros(DuelingQRDQN.INPUT_DIM, dtype=np.float32)
                next_mask = np.zeros(DuelingQRDQN.OUTPUT_DIM, dtype=np.float32)

            terminal = float(done or next_state is None)
            # Push to the n-step accumulator; it emits 0+ completed
            # n-step transitions per call.
            new_transitions = accumulator.add(
                cur_feats, action, reward / BASE_BET, next_feats,
                terminal, next_mask)
            for (s_, a_, r_, ns_, d_, gamma_eff_, nm_) in new_transitions:
                buf.add(s_, a_, r_, ns_, d_, gamma_eff_, nm_)

            step_count += 1

            should_train = (buf.size >= args.batch_size
                            and step_count % args.train_every == 0)
            if should_train:
                (bf_np, ba_np, br_np, bns_np, bd_np, bg_np, bmask_np
                 ) = buf.sample(args.batch_size)
                bf  = torch.from_numpy(bf_np).to(device)
                ba  = torch.from_numpy(ba_np).to(device)
                br  = torch.from_numpy(br_np).to(device)
                bns = torch.from_numpy(bns_np).to(device)
                bd  = torch.from_numpy(bd_np).to(device)
                bg  = torch.from_numpy(bg_np).to(device)
                bmask = torch.from_numpy(bmask_np).to(device)

                # Current quantile prediction for the taken action.
                # forward(bf) -> (B, A, N_QUANTILES)
                # gather along A using ba -> (B, 1, N_QUANTILES)
                pred_quantiles = net.forward(bf).gather(
                    1, ba.view(-1, 1, 1).expand(-1, 1, N_QUANTILES)
                ).squeeze(1)                                       # (B, N_QUANTILES)

                with torch.no_grad():
                    # Double-DQN style: pick next action with online net's
                    # mean Q, then read target net's quantiles for that
                    # action.
                    online_next_means = net.q_values(bns).masked_fill(
                        bmask == 0, -1e9)
                    next_actions = online_next_means.argmax(dim=1)
                    target_quantiles = target_net.forward(bns).gather(
                        1, next_actions.view(-1, 1, 1).expand(-1, 1, N_QUANTILES)
                    ).squeeze(1)                                   # (B, N_QUANTILES)
                    # N-step bootstrap with effective gamma = gamma**n
                    # (or smaller if the episode ended within the window).
                    target_q = (br.unsqueeze(1)
                                + (1.0 - bd.unsqueeze(1))
                                * bg.unsqueeze(1)
                                * target_quantiles)                # (B, N_QUANTILES)

                loss = quantile_huber_loss(pred_quantiles, target_q)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step()

                # Polyak (soft) target update every training step.
                polyak_update(target_net, net, args.polyak_tau)

            if done or next_state is None:
                break
            state = next_state
            cur_feats = next_feats

        rewards_window[rew_ptr] = ep_reward
        rew_ptr = (rew_ptr + 1) % rewards_window.size
        rew_filled = min(rew_filled + 1, rewards_window.size)
        if eps_per_episode > 0:
            epsilon = max(args.epsilon_end, epsilon - eps_per_episode)

        if (episode + 1) % print_every == 0:
            window_slice = (rewards_window[:rew_filled]
                            if rew_filled < rewards_window.size else rewards_window)
            avg = float(window_slice.mean())
            elapsed = time.time() - t0
            rate = (episode + 1) / max(elapsed, 1e-6)
            eta = (args.episodes - (episode + 1)) / max(rate, 1)
            print(f"  ep {episode+1:>9,}/{args.episodes:,} | "
                  f"avg reward(last 2k): ${avg:+.2f} | "
                  f"ε={epsilon:.3f} | "
                  f"buf={buf.size:>7,} | "
                  f"{rate:.0f} eps/s | "
                  f"ETA {eta/60:>4.1f}min | "
                  f"skip-peek-BJ={skipped_peek_bj:,}")

        # Periodic checkpoint -- protects against the training process
        # being killed before the final save.
        if (args.save_every > 0
                and (episode + 1) % args.save_every == 0
                and args.save_path):
            net.eval()
            torch.save({'net': net.state_dict()}, args.save_path)
            net.train()
            print(f"    [checkpoint saved at ep {episode+1:,} to {args.save_path}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5_000_000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--polyak-tau', type=float, default=0.005,
                        help='soft target update rate per training step')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--buffer-size', type=int, default=200_000)
    parser.add_argument('--epsilon-start', type=float, default=0.4)
    parser.add_argument('--epsilon-end', type=float, default=0.02)
    parser.add_argument('--epsilon-decay-episodes', type=int, default=2_000_000)
    parser.add_argument('--train-every', type=int, default=4)
    parser.add_argument('--warm-start-batches', type=int, default=3000)
    parser.add_argument('--save-path', type=str, default='dqn_agent.pt')
    parser.add_argument('--save-every', type=int, default=500_000,
                        help='checkpoint every N episodes (0 = only at end)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=None,
                        help='override device (cpu/mps/cuda)')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    print(f"Training on {device} (threads={torch.get_num_threads()})")
    print(f"Architecture: Dueling QR-DQN, "
          f"input={DuelingQRDQN.INPUT_DIM}, n_actions={DuelingQRDQN.OUTPUT_DIM}, "
          f"n_quantiles={DuelingQRDQN.N_QUANTILES}")

    net = DuelingQRDQN().to(device)
    target_net = DuelingQRDQN().to(device)
    target_net.load_state_dict(net.state_dict())
    opt = optim.Adam(net.parameters(), lr=args.lr)

    if args.warm_start_batches > 0:
        warm_start(net, opt, device, num_batches=args.warm_start_batches,
                   batch_size=args.batch_size)
        target_net.load_state_dict(net.state_dict())

    if args.episodes > 0:
        rl_train(net, target_net, opt, device, args)

    net.eval()
    torch.save({'net': net.state_dict()}, args.save_path)
    print(f"\nSaved DQN to {args.save_path}")


if __name__ == '__main__':
    main()
