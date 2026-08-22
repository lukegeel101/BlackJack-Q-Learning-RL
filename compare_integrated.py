"""Head-to-head: the integrated agent vs the counter and the RL DQN.

All agents are run on the SAME seeds (common random numbers), so the shoes
they face are identical hand-for-hand except where their own play decisions
diverge -- a large variance reduction versus independent shoes.

Agents compared:
  1. Flat-bet basic strategy            (baseline house edge)
  2. Hi-Lo counter: basic+I18 play, Hi-Lo bet ramp   (the professional)
  3. RL DQN card counting               (the original Agent 3)
  4. Integrated: composition bet + solver-distilled play

Agents 2 and 4 wager the SAME total by construction (the integrated agent's
quantile ramp copies the Hi-Lo bet distribution), so their edge difference
is a like-for-like comparison at equal risk.

Run:
    python compare_integrated.py --hands 100000 --seeds 1 2 3 4 5 6
"""
import argparse
import os

import numpy as np

from agents import (FlatBetBasicStrategyAgent, CountingBasicStrategyAgent,
                    DQNCardCountingAgent, IntegratedCardCountingAgent)
from simulate import run_agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hands', type=int, default=100000)
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    p.add_argument('--bet-model', default='bet_value_net.pt')
    p.add_argument('--play-model', default='dqn_agent_distilled.pt')
    p.add_argument('--rl-model', default='dqn_agent.pt')
    p.add_argument('--unit', type=int, default=50)
    p.add_argument('--max-units', type=int, default=50)
    p.add_argument('--margin', type=float, default=0.05)
    a = p.parse_args()

    def build():
        agents = [
            FlatBetBasicStrategyAgent(flat_bet=a.unit),
            CountingBasicStrategyAgent(unit=a.unit, max_units=a.max_units),
        ]
        if os.path.exists(a.rl_model):
            agents.append(DQNCardCountingAgent(model_path=a.rl_model,
                                               unit=a.unit,
                                               max_units=a.max_units))
        agents.append(IntegratedCardCountingAgent(
            bet_model_path=a.bet_model, play_model_path=a.play_model,
            unit=a.unit, max_units=a.max_units, deviation_margin=a.margin))
        return agents

    agents = build()
    names = [ag.name for ag in agents]
    edges = {n: [] for n in names}
    perhand = {n: [] for n in names}
    avgbet = {n: [] for n in names}
    profits = {n: [] for n in names}

    print(f"{len(agents)} agents x {len(a.seeds)} seeds x {a.hands:,} hands "
          f"(common random numbers)\n")
    for s in a.seeds:
        for ag in agents:
            if hasattr(ag, 'decision_count'):
                try:
                    ag._play.decision_count = 0; ag._play.deviation_count = 0
                except AttributeError:
                    ag.decision_count = 0; ag.deviation_count = 0
            r, b, _ = run_agent(ag, a.hands, seed=s, show_progress=False)
            edges[ag.name].append(r.sum() / b.sum() * 100)
            perhand[ag.name].append(r.mean())
            avgbet[ag.name].append(b.mean())
            profits[ag.name].append(r.sum())
        print(f"  seed {s}: " + "  ".join(
            f"{n.split()[0]} {edges[n][-1]:+.2f}%" for n in names))

    print(f"\n{'agent':<46}{'edge':>9}{'$/hand':>10}{'avg bet':>10}")
    for n in names:
        e = np.array(edges[n])
        print(f"{n:<46}{e.mean():>+8.3f}%{np.mean(perhand[n]):>+10.4f}"
              f"{np.mean(avgbet[n]):>10.0f}")

    counter = names[1]
    integ = names[-1]
    d = np.array(edges[integ]) - np.array(edges[counter])
    dph = np.array(perhand[integ]) - np.array(perhand[counter])
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"\nIntegrated - Hi-Lo counter (paired, same shoes):")
    print(f"  edge   {d.mean():+.3f}%  (per-seed " +
          " ".join(f"{x:+.2f}" for x in d) + ")")
    print(f"  $/hand {dph.mean():+.4f}   SE {se*np.mean(avgbet[counter])/100:.4f}")
    print(f"  95% CI on edge diff: [{d.mean()-1.96*se:+.3f}, "
          f"{d.mean()+1.96*se:+.3f}]%")


if __name__ == '__main__':
    main()
