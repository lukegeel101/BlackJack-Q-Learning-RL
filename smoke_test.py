"""Fast smoke tests for CI.

Deliberately checks invariants rather than profit: money outcomes in
blackjack are far too noisy to assert on (see ACCURACY_FINDINGS.md), so the
bounds here are wide and everything is seeded. What this catches is the class
of bug that actually bit this project -- feature-dim drift between the env
and the nets, checkpoints that no longer load, a solver returning NaN, an
agent emitting an illegal action, or a bet ramp that quietly changes size.

Run:  python smoke_test.py
Exit code 0 = pass, 1 = failure (with the failing check named).
"""
import sys
import traceback

FAILURES = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:
        FAILURES.append(name)
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()


# ----------------------------------------------------------------------
# 1. Imports
# ----------------------------------------------------------------------

def t_imports():
    import cardCounting, agents, simulate, optimal_policy          # noqa: F401
    import train_dqn, train_supervised, gen_optimal_dataset        # noqa: F401
    import gen_bet_dataset, train_bet_value, calibrate_bet_ramp    # noqa: F401
    import mc_true_ev, score_bet, compare_integrated               # noqa: F401


# ----------------------------------------------------------------------
# 2. Feature spec agrees with the nets
# ----------------------------------------------------------------------

def t_feature_dims():
    from agents import (INPUT_DIM, BET_INPUT_DIM, state_to_features,
                        DuelingQRDQN, DuelingQNet, BetValueNet)
    from cardCounting import BlackjackEnv
    assert INPUT_DIM == 34, f"INPUT_DIM drifted to {INPUT_DIM}"
    assert BET_INPUT_DIM == 15, f"BET_INPUT_DIM drifted to {BET_INPUT_DIM}"
    for net in (DuelingQRDQN, DuelingQNet):
        assert net.INPUT_DIM == INPUT_DIM
    assert BetValueNet.INPUT_DIM == BET_INPUT_DIM

    env = BlackjackEnv(num_decks=8, min_bet=50, max_bet=50)
    state = env.reset()
    feats = state_to_features(state)
    assert tuple(feats.shape) == (INPUT_DIM,), f"got {tuple(feats.shape)}"


# ----------------------------------------------------------------------
# 3. Checkpoints still load and score
# ----------------------------------------------------------------------

def t_checkpoints():
    import os
    import torch
    from agents import (load_dqn_net, load_bet_value_net, state_to_features,
                        INPUT_DIM)
    from cardCounting import BlackjackEnv

    env = BlackjackEnv(num_decks=8, min_bet=50, max_bet=50)
    feats = state_to_features(env.reset())

    for path in ('dqn_agent.pt', 'dqn_agent_distilled.pt'):
        if not os.path.exists(path):
            continue
        net = load_dqn_net(path)
        with torch.no_grad():
            q = net.q_values(feats)
        assert tuple(q.shape) == (1, 5), f"{path}: q shape {tuple(q.shape)}"
        assert torch.isfinite(q).all(), f"{path}: non-finite Q values"

    if os.path.exists('bet_value_net.pt'):
        net, ramp = load_bet_value_net('bet_value_net.pt')
        assert 'ev_edges' in ramp or 'ev_lo' in ramp, "ramp missing"
        with torch.no_grad():
            out = net(torch.zeros(1, net.fc1.weight.shape[1]))
        assert torch.isfinite(out).all(), "bet net produced non-finite EV"


# ----------------------------------------------------------------------
# 4. Agents play legally and produce finite results
# ----------------------------------------------------------------------

def t_agents_play():
    import os
    import numpy as np
    from agents import (FlatBetBasicStrategyAgent, CountingBasicStrategyAgent,
                        DQNCardCountingAgent, IntegratedCardCountingAgent)
    from simulate import run_agent

    agents = [FlatBetBasicStrategyAgent(flat_bet=50),
              CountingBasicStrategyAgent(unit=50, max_units=50)]
    if os.path.exists('dqn_agent.pt'):
        agents.append(DQNCardCountingAgent(model_path='dqn_agent.pt',
                                           unit=50, max_units=50))
    if os.path.exists('bet_value_net.pt') and os.path.exists('dqn_agent_distilled.pt'):
        agents.append(IntegratedCardCountingAgent(
            bet_model_path='bet_value_net.pt',
            play_model_path='dqn_agent_distilled.pt',
            unit=50, max_units=50))

    for ag in agents:
        rewards, bets, _ = run_agent(ag, 4000, seed=123, show_progress=False)
        assert np.isfinite(rewards).all(), f"{ag.name}: non-finite rewards"
        assert (bets > 0).all(), f"{ag.name}: non-positive bet"
        assert bets.max() <= 2500, f"{ag.name}: bet {bets.max()} over table max"
        # Wide sanity band -- this is a "did the agent break" check, not a
        # performance assertion.
        edge = rewards.sum() / bets.sum() * 100
        assert -15.0 < edge < 15.0, f"{ag.name}: implausible edge {edge:.2f}%"


# ----------------------------------------------------------------------
# 5. Bet ramps stay on their unit ladder
# ----------------------------------------------------------------------

def t_bet_ramp():
    import os
    import numpy as np
    from agents import CountingBasicStrategyAgent, IntegratedCardCountingAgent
    from simulate import run_agent

    ladder = {50 * u for u in (1, 2, 4, 8, 16, 24, 32, 40, 50)}
    _, bets, _ = run_agent(CountingBasicStrategyAgent(unit=50, max_units=50),
                           3000, seed=7, show_progress=False)
    assert set(np.unique(bets)) <= ladder, "Hi-Lo ramp left its unit ladder"

    if os.path.exists('bet_value_net.pt') and os.path.exists('dqn_agent_distilled.pt'):
        ag = IntegratedCardCountingAgent(
            bet_model_path='bet_value_net.pt',
            play_model_path='dqn_agent_distilled.pt', unit=50, max_units=50)
        _, bets, _ = run_agent(ag, 3000, seed=7, show_progress=False)
        # The quantile ramp copies the Hi-Lo bet distribution, so it must use
        # the same discrete bet sizes.
        assert set(np.unique(bets)) <= ladder, (
            "integrated ramp left the Hi-Lo unit ladder")


# ----------------------------------------------------------------------
# 6. Solver returns finite EVs and a legal optimal action
# ----------------------------------------------------------------------

def t_solver():
    import math
    from cardCounting import BlackjackEnv
    from agents import basic_strategy_with_deviations, _valid_actions
    from optimal_policy import evaluate_state

    env = BlackjackEnv(num_decks=8, min_bet=50, max_bet=50)
    state = env.reset()
    checked = 0
    while checked < 12:
        if state is None or env.dealer_pre_bj:
            if env.dealer_pre_bj:
                env.step(0)
            state = env.reset()
            continue
        evs, best = evaluate_state(state)
        valid = _valid_actions(state)
        assert best in valid, f"solver picked illegal action {best}"
        for a in valid:
            assert a in evs and math.isfinite(evs[a]), f"non-finite EV for {a}"
            assert -3.0 <= evs[a] <= 3.0, f"EV out of range: {evs[a]}"
        checked += 1
        nxt, _, done = env.step(basic_strategy_with_deviations(state))
        state = env.reset() if (done or nxt is None) else nxt


# ----------------------------------------------------------------------
# 7. Strategy tables only emit legal actions
# ----------------------------------------------------------------------

def t_legal_actions():
    from cardCounting import BlackjackEnv
    from agents import (basic_strategy_action, basic_strategy_with_deviations,
                        HIT, STAND, DOUBLE, SPLIT, SURRENDER)

    env = BlackjackEnv(num_decks=8, min_bet=50, max_bet=50)
    state = env.reset()
    for _ in range(1500):
        if state is None or env.dealer_pre_bj:
            if env.dealer_pre_bj:
                env.step(0)
            state = env.reset()
            continue
        for fn in (basic_strategy_action, basic_strategy_with_deviations):
            a = fn(state)
            assert a in (HIT, STAND, DOUBLE, SPLIT, SURRENDER), \
                f"{fn.__name__} returned {a}"
        nxt, _, done = env.step(basic_strategy_with_deviations(state))
        state = env.reset() if (done or nxt is None) else nxt


def main():
    print("Blackjack RL smoke tests\n")
    check("imports", t_imports)
    check("feature dims", t_feature_dims)
    check("checkpoints load", t_checkpoints)
    check("agents play legally", t_agents_play)
    check("bet ramps on ladder", t_bet_ramp)
    check("solver sane", t_solver)
    check("strategy tables legal", t_legal_actions)

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES)}")
        return 1
    print("All smoke tests passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
