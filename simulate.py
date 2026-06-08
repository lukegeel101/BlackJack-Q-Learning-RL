"""
Comparison harness for the three blackjack agents.

Each agent is run for the same number of hands on its own shoe, then the
results are summarized side-by-side.

Run directly:

    python simulate.py                     # quick: 50k hands per agent
    python simulate.py --hands 200000      # longer run
    python simulate.py --model dqn_agent.pt --no-plot

Expected ordering of net profit:
    Agent 3 (DQN)  >  Agent 2 (counting)  >  0  >  Agent 1 (flat bet)
"""

import argparse
import os
import random
import time
from typing import Sequence

import numpy as np

from cardCounting import BlackjackEnv
from agents import (
    FlatBetBasicStrategyAgent,
    CountingBasicStrategyAgent,
    DQNCardCountingAgent,
)


def run_agent(agent,
              num_hands: int,
              num_decks: int = 8,
              seed: int = None,
              show_progress: bool = True):
    """Play `num_hands` hands with `agent` on a fresh env.

    Returns (rewards, bets, true_counts) as numpy arrays of length num_hands.

    Notes
    -----
    * Each agent gets its OWN BlackjackEnv (and shoe), seeded independently.
      We could share shoes between agents for variance reduction, but the
      different bet sizes complicate that. Independent shoes are simpler
      and the per-hand expected value is the right comparison anyway.
    * The env's CardCounter sets a bet on reset() automatically. We override
      it with `agent.get_bet(state)` so each agent's own bet policy is used.
      Splits/doubles inside the env scale from that base bet correctly.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # min_bet=max_bet=50 makes the env's auto-bet a no-op; the agent's
    # get_bet decides the actual bet.
    env = BlackjackEnv(num_decks=num_decks, min_bet=50, max_bet=50)

    rewards = np.zeros(num_hands, dtype=np.float64)
    bets = np.zeros(num_hands, dtype=np.float64)
    true_counts = np.zeros(num_hands, dtype=np.float64)

    iterator = range(num_hands)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=agent.name, leave=False)
        except ImportError:
            pass

    for i in iterator:
        # Snapshot PRE-deal count BEFORE reset deals new cards.
        # If the env is about to reshuffle (or this is the first hand),
        # treat the pre-deal count as 0.
        pre_was_shuffled = (env.deck.needs_shuffle()
                            or len(env.deck.dealt_cards) == 0)
        if pre_was_shuffled:
            pre_running = 0
            pre_decks = float(env.deck.num_decks)
        else:
            pre_running = env.counter.running_count
            pre_decks = max(env.counter.decks_remaining, 0.5)
        pre_true_count = pre_running / pre_decks

        state = env.reset()
        # Inject the pre-deal count so agents bet on it.
        state['pre_true_count'] = pre_true_count

        bet = int(agent.get_bet(state))
        env.bets[0] = bet
        true_counts[i] = pre_true_count
        bets[i] = bet

        done = False
        ep_reward = 0.0
        while not done:
            action = agent.get_action(state)
            state, r, done = env.step(action)
            ep_reward += r
        rewards[i] = ep_reward

    return rewards, bets, true_counts


def summarize(name: str,
              rewards: np.ndarray,
              bets: np.ndarray) -> dict:
    """Print and return summary stats for an agent."""
    total_profit = float(rewards.sum())
    total_wagered = float(bets.sum())
    avg_bet = float(bets.mean())
    avg_profit = float(rewards.mean())
    edge = (total_profit / total_wagered * 100.0) if total_wagered > 0 else 0.0
    win_rate = float((rewards > 0).mean()) * 100.0
    loss_rate = float((rewards < 0).mean()) * 100.0
    push_rate = float((rewards == 0).mean()) * 100.0

    print(f"  {name}")
    print(f"    hands           : {len(rewards):,}")
    print(f"    avg bet         : ${avg_bet:,.2f}")
    print(f"    total wagered   : ${total_wagered:,.0f}")
    print(f"    net profit      : ${total_profit:+,.2f}")
    print(f"    profit / hand   : ${avg_profit:+.4f}")
    print(f"    edge (per $1)   : {edge:+.3f}%")
    print(f"    win/loss/push   : {win_rate:.2f}% / {loss_rate:.2f}% / {push_rate:.2f}%")

    return {
        'name': name,
        'hands': len(rewards),
        'total_profit': total_profit,
        'total_wagered': total_wagered,
        'avg_bet': avg_bet,
        'edge_pct': edge,
        'win_rate_pct': win_rate,
    }


def plot_cumulative(results: Sequence[tuple], save_path: str = None):
    """results: sequence of (name, rewards_array). Plots cumulative profit."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    for name, rewards in results:
        ax.plot(np.cumsum(rewards), label=name, linewidth=1.2)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.6, alpha=0.6)
    ax.set_xlabel('Hand number')
    ax.set_ylabel('Cumulative profit ($)')
    ax.set_title('Blackjack agent comparison -- cumulative profit')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"saved plot -> {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hands', type=int, default=50_000,
                        help='hands to play per agent (default 50,000)')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--model', type=str, default='dqn_agent.pt',
                        help='DQN weights for Agent 3 (default dqn_agent.pt)')
    parser.add_argument('--flat-bet', type=int, default=50,
                        help='flat bet for Agent 1 (default 50)')
    parser.add_argument('--unit', type=int, default=50,
                        help='betting unit for Agents 2 and 3 (default 50)')
    parser.add_argument('--max-units', type=int, default=50,
                        help='max bet for Agents 2 and 3 in units (default 50 = $2500)')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--plot-path', default='comparison.png')
    parser.add_argument('--skip-dqn', action='store_true',
                        help='skip Agent 3 (useful if you have not trained yet)')
    args = parser.parse_args()

    agents = [
        FlatBetBasicStrategyAgent(flat_bet=args.flat_bet),
        CountingBasicStrategyAgent(unit=args.unit, max_units=args.max_units),
    ]
    if not args.skip_dqn:
        if not os.path.exists(args.model):
            print(f"!! DQN weights not found at {args.model}.")
            print(f"!! Agent 3 will use an UNTRAINED net (essentially random).")
            print(f"!! Train with:  python train_dqn.py --save-path {args.model}")
        agents.append(DQNCardCountingAgent(
            model_path=args.model,
            unit=args.unit,
            max_units=args.max_units,
        ))

    print(f"Running {len(agents)} agent(s) for {args.hands:,} hands each.\n")

    raw_results = []
    summaries = []
    for i, agent in enumerate(agents):
        t0 = time.time()
        # Seed each agent's shoe a little differently if a seed was given.
        seed = (args.seed + i) if args.seed is not None else None
        rewards, bets, _ = run_agent(agent, args.hands, seed=seed)
        elapsed = time.time() - t0

        print(f"\n[{elapsed:.1f}s] {agent.name}:")
        summaries.append(summarize(agent.name, rewards, bets))
        raw_results.append((agent.name, rewards))

    # Sanity-check ordering of net profits
    print("\n--- ordering check ---")
    ordered = sorted(summaries, key=lambda s: s['total_profit'], reverse=True)
    for s in ordered:
        print(f"  ${s['total_profit']:+,.2f}   {s['name']}")

    if not args.no_plot:
        plot_cumulative(raw_results, save_path=args.plot_path)


if __name__ == '__main__':
    main()
