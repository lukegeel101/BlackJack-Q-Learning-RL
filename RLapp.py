from train import train_agent, run_reinforcement_learning, visualize_count_based_strategy, analyze_results
from cardCounting import BlackjackEnv, BasicStrategyAgent, CardCounter

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
from cardCounting import BlackjackEnv
import time

# Train a basic strategy agent
print("Training basic strategy agent...")
env, agent, results_df, deviations_df, bankroll_history, count_history, bet_history = train_agent(num_episodes=1000)

# Plot bankroll progression during training
plt.figure(figsize=(12, 6))
plt.plot(bankroll_history)
plt.title('Bankroll Progression During Training')
plt.xlabel('Hand Number')
plt.ylabel('Bankroll ($)')
plt.grid(True)
plt.savefig('bankroll_progression_training.png')
plt.show()

# Analyze results
print("\nAnalyzing results...")
analyze_results(results_df, deviations_df, bankroll_history, count_history, bet_history)

# Visualize count-based strategy
print("\nVisualizing count-based strategy...")
visualize_count_based_strategy()

# Run reinforcement learning to discover optimal strategy
print("\nRunning reinforcement learning to discover optimal strategy...")
Q, learned_strategy = run_reinforcement_learning()

# Now let's run a simulation with the learned Q-values to see bankroll progression
print("\nRunning simulation with learned strategy...")

# Function to select action based on Q-values
def select_q_action(state, Q):
    player_val = state['player_value']
    dealer_val = state['dealer_up_card']
    is_soft = 1 if state['is_soft'] else 0
    count = max(-6, min(6, int(round(state['true_count']))))
    
    # Get valid actions
    valid_actions = [0, 1]  # Hit and stand are always valid
    
    # Double is valid only on first two cards
    if len(state['player_hand'].cards) == 2:
        valid_actions.append(2)  # Double
    
    # Split is valid only for pairs
    if state['can_split']:
        valid_actions.append(3)  # Split
    
    # Get Q-values for valid actions
    q_values = [Q.get(((player_val, dealer_val, is_soft, count), a), float('-inf')) 
               for a in valid_actions]
    
    # Select best action
    return valid_actions[np.argmax(q_values)] if q_values else 1  # Default to stand if no Q-values

# Run simulation with learned Q-values
env = BlackjackEnv(num_decks=8, min_bet=50, max_bet=5000)
env.bankroll = 10000  # Start with $10,000
q_bankroll_history = [env.bankroll]

num_hands = 5000
for _ in tqdm(range(num_hands)):
    state = env.reset()
    done = False
    
    while not done:
        action = select_q_action(state, Q)
        state, reward, done = env.step(action)
        
    q_bankroll_history.append(env.bankroll)

# Plot bankroll progression with learned strategy
plt.figure(figsize=(12, 6))
plt.plot(q_bankroll_history)
plt.title('Bankroll Progression with Learned Q-Strategy')
plt.xlabel('Hand Number')
plt.ylabel('Bankroll ($)')
plt.grid(True)
plt.savefig('bankroll_progression_q_strategy.png')
plt.show()

# Additional visualization for learned strategy
print("\nVisualizing learned strategy deviations...")

# Convert learned strategy to DataFrame for better visualization
learned_df = []
for (player_val, dealer_val, count_bin), data in learned_strategy.items():
    action_map = {0: 'Hit', 1: 'Stand', 2: 'Double', 3: 'Split'}
    learned_df.append({
        'player_value': player_val,
        'dealer_value': dealer_val,
        'count': count_bin,
        'basic_action': action_map[data['basic_action']],
        'rl_action': action_map[data['rl_action']],
        'q_diff': data['q_value_diff']
    })

learned_df = pd.DataFrame(learned_df)

# Plot the heatmap of q-value differences
if not learned_df.empty:
    pivot = learned_df.pivot_table(
        index='player_value', 
        columns='dealer_value', 
        values='q_diff',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn')
    plt.title('Average Q-Value Difference by Player Hand vs Dealer Card')
    plt.xlabel('Dealer Up Card')
    plt.ylabel('Player Total')
    plt.tight_layout()
    plt.savefig('learned_strategy_heatmap.png')
    plt.show()

    # Plot the strongest deviations
    top_deviations = learned_df.sort_values('q_diff', ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    bars = plt.barh(
        range(len(top_deviations)),
        top_deviations['q_diff'],
        color='skyblue'
    )
    plt.yticks(
        range(len(top_deviations)),
        [f"{row['player_value']} vs {row['dealer_value']} (count={row['count']}): {row['basic_action']} → {row['rl_action']}" 
         for _, row in top_deviations.iterrows()]
    )
    plt.xlabel('Q-Value Difference')
    plt.title('Top 10 Strategy Deviations by Q-Value Difference')
    plt.tight_layout()
    plt.savefig('top_deviations.png')
    plt.show()

# Compare performance: basic strategy vs Q-learning
print("\nComparing Basic Strategy vs Q-Learning Strategy...")

# Run simulation with basic strategy
env = BlackjackEnv(num_decks=8, min_bet=50, max_bet=5000)
agent = BasicStrategyAgent()  # Make sure to import this
env.bankroll = 10000
basic_bankroll_history = [env.bankroll]

for _ in tqdm(range(num_hands)):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.get_action(state)
        state, reward, done = env.step(action)
        
    basic_bankroll_history.append(env.bankroll)

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(basic_bankroll_history, label='Basic Strategy')
plt.plot(q_bankroll_history, label='Q-Learning Strategy')
plt.title('Bankroll Comparison: Basic Strategy vs Q-Learning')
plt.xlabel('Hand Number')
plt.ylabel('Bankroll ($)')
plt.legend()
plt.grid(True)
plt.savefig('strategy_comparison.png')
plt.show()

print("\nAll visualizations have been saved to the current directory.")