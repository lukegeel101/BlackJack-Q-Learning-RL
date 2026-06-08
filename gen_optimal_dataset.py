"""
Offline dataset generator for Option-2 supervised distillation.

Plays the env, snapshots states at every player decision point, calls the
expectimax solver from optimal_policy.evaluate_state on each one, and
saves (features_28d, ev_per_action_5d, valid_mask_5d, optimal_action) to
a .npz file. Multiprocessing across CPU cores because the solver is
single-threaded pure Python and slow enough that we need the parallelism.

Run:
    python gen_optimal_dataset.py --num-states 100000 \
        --out optimal_dataset.npz

Sampling strategy
-----------------
Each worker plays an independent env and snapshots EVERY decision point.
We don't filter by hand type or count -- the network needs to see the
state distribution it will actually face at inference time, which is
exactly what env play emits. The play policy IS basic strategy + I18
(via basic_strategy_with_deviations) so that downstream states reflect
realistic post-decision compositions.

Output format
-------------
A single .npz file with arrays of length N (one row per labeled state):
    features      (N, INPUT_DIM)   float32
    ev_actions    (N, 5)           float32  -- EV per action; invalid = NaN
    valid_mask    (N, 5)           bool     -- True for legal actions
    optimal       (N,)             int8     -- argmax of EV over valid actions
"""

import argparse
import multiprocessing as mp
import os
import random
import time
from typing import List, Tuple

import numpy as np

# Importing torch transitively via agents is expensive in workers, so we
# defer it to inside the worker function where the import gets pickled
# fresh per process anyway.

ACTION_COUNT = 5
NUM_FEATURES = 28


def _worker(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One worker process produces `n_states` labeled examples.

    Args is a tuple (worker_id, n_states, seed). Returns (features,
    ev_actions, valid_mask, optimal) numpy arrays.
    """
    worker_id, n_states, seed = args

    # Imports inside worker so each process gets its own torch state etc.
    import torch  # noqa: F401 -- imported for tensor type used by state_to_features
    from cardCounting import BlackjackEnv
    from agents import (basic_strategy_with_deviations, state_to_features,
                        _valid_actions)
    from optimal_policy import evaluate_state, clear_caches

    random.seed(seed)
    np.random.seed(seed)

    env = BlackjackEnv(num_decks=8, min_bet=50, max_bet=50)

    feats = np.zeros((n_states, NUM_FEATURES), dtype=np.float32)
    evs = np.full((n_states, ACTION_COUNT), np.nan, dtype=np.float32)
    mask = np.zeros((n_states, ACTION_COUNT), dtype=bool)
    opt = np.zeros(n_states, dtype=np.int8)

    i = 0
    state = env.reset()
    last_report = time.time()
    while i < n_states:
        # Skip dealer-pre-BJ rounds (no player action) and any None state.
        if state is None or env.dealer_pre_bj:
            if env.dealer_pre_bj:
                env.step(0)
            state = env.reset()
            continue

        # Label THIS decision point with the optimal action and per-action EVs.
        action_evs, optimal_action = evaluate_state(state)
        valid = _valid_actions(state)

        # NaN guard: if the solver returned a NaN for any valid action,
        # skip this state rather than poison the training set. Also
        # dump the offending state so we can diagnose offline.
        bad = False
        for a in valid:
            ev = action_evs.get(a)
            if ev is None or ev != ev:   # NaN check (NaN != NaN)
                bad = True
                break
        if bad:
            ph = state['player_hand']
            print(f"  [worker {worker_id}] WARN skip NaN-state: "
                  f"cards={[str(c) for c in ph.cards]} value={state['player_value']} "
                  f"soft={state['is_soft']} splitted={ph.is_splitted} "
                  f"split_aces={ph.is_split_aces} can_split={state['can_split']} "
                  f"upcard={state['dealer_up_card']} evs={action_evs}",
                  flush=True)
            action_to_step = basic_strategy_with_deviations(state)
            next_state, _, done = env.step(action_to_step)
            if done or next_state is None:
                state = env.reset()
            else:
                state = next_state
            continue

        feats[i] = state_to_features(state).numpy()
        for a in valid:
            mask[i, a] = True
        for a, ev in action_evs.items():
            evs[i, a] = ev
        opt[i] = optimal_action
        i += 1

        # Step the env with the basic+I18 default action so the next state
        # distribution reflects realistic play (rather than e.g. always
        # hitting). Could also step with optimal_action -- we use the
        # default to stay aligned with how Agent 2 plays in deployment.
        action_to_step = basic_strategy_with_deviations(state)
        # The basic-strategy fallback can return SURRENDER on an illegal
        # state if our predicate is too generous; the env will silently
        # fall back to STAND for invalid surrenders, so just pass through.
        next_state, _, done = env.step(action_to_step)
        if done or next_state is None:
            state = env.reset()
        else:
            state = next_state

        # Periodically clear caches to keep per-worker memory in check.
        # The dealer cache grows to ~1M entries when full -- with 4 or
        # more workers that's 4+ GB total. Less-frequent clearing
        # (every 5k states) keeps the in-recursion cache hits while
        # bounding total memory.
        if i % 5000 == 0 and i > 0:
            clear_caches()

        if worker_id == 0 and i % 500 == 0:
            now = time.time()
            rate = 500 / max(now - last_report, 1e-6)
            print(f"  [worker 0] {i:>7,}/{n_states:,} states "
                  f"({rate:.0f}/s)", flush=True)
            last_report = now

    return feats, evs, mask, opt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-states', type=int, default=100_000,
                        help='Total labeled examples to generate.')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Worker processes. 0 = use all CPU cores '
                             '(reduced by 1 for the parent).')
    parser.add_argument('--out', type=str, default='optimal_dataset.npz',
                        help='Output .npz path.')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.num_workers == 0:
        args.num_workers = max(1, mp.cpu_count() - 1)

    # Divide work as evenly as possible.
    per_worker = [args.num_states // args.num_workers] * args.num_workers
    for k in range(args.num_states % args.num_workers):
        per_worker[k] += 1

    print(f"Generating {args.num_states:,} labeled states across "
          f"{args.num_workers} workers ({per_worker[0]:,} each, with "
          f"remainder spread across the first {args.num_states % args.num_workers} "
          f"workers).")
    print(f"Output: {args.out}")

    worker_args = [
        (wid, per_worker[wid], args.seed + wid * 10_000)
        for wid in range(args.num_workers)
    ]

    t0 = time.time()
    if args.num_workers == 1:
        results = [_worker(worker_args[0])]
    else:
        # 'spawn' avoids inheriting torch state from the parent (faster
        # worker startup and lower memory).
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.num_workers) as pool:
            results = pool.map(_worker, worker_args)
    elapsed = time.time() - t0

    feats = np.concatenate([r[0] for r in results], axis=0)
    evs = np.concatenate([r[1] for r in results], axis=0)
    mask = np.concatenate([r[2] for r in results], axis=0)
    opt = np.concatenate([r[3] for r in results], axis=0)

    print(f"\nDone in {elapsed/60:.1f} min "
          f"({args.num_states / max(elapsed, 1):.0f} states/s overall).")
    print(f"Dataset shape: features {feats.shape}, evs {evs.shape}, "
          f"mask {mask.shape}, opt {opt.shape}")

    # Quick sanity: distribution of optimal actions.
    from agents import HIT, STAND, DOUBLE, SPLIT, SURRENDER
    ACTION_NAME = {HIT: 'HIT', STAND: 'STAND', DOUBLE: 'DOUBLE',
                   SPLIT: 'SPLIT', SURRENDER: 'SURR'}
    print("\nOptimal action distribution:")
    counts = np.bincount(opt, minlength=5)
    for a, n in enumerate(counts):
        pct = n / len(opt) * 100
        print(f"  {ACTION_NAME[a]:8s} {n:>8,}  ({pct:5.2f}%)")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(args.out, features=feats, ev_actions=evs,
             valid_mask=mask, optimal=opt)
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"\nSaved -> {args.out} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
