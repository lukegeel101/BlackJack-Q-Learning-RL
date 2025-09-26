import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Import the classes from our implementation
from cardCounting import BlackjackEnv, BasicStrategyAgent, CardCounter
from train import train_agent, analyze_results, visualize_count_based_strategy
import train_q_agent

def simulate_game(num_hands=1000, initial_bankroll=10000, show_progress=True, min_bet=50, max_bet=5000):
    """
    Simulate a blackjack game with the trained agent
    """
    env = BlackjackEnv(num_decks=8, min_bet=min_bet, max_bet=max_bet)
    agent = BasicStrategyAgent()
    
    env.bankroll = initial_bankroll
    
    # For tracking results
    hand_results = []
    count_tracking = []
    
    iterator = tqdm(range(num_hands)) if show_progress else range(num_hands)
    
    for hand_num in iterator:
        state = env.reset()
        done = False
        
        # Track the starting true count for this hand
        starting_true_count = state['true_count']
        
        # Track the initial bet
        initial_bet = state['bet']
        
        # Track the decisions and results for this hand
        hand_actions = []
        
        while not done:
            action = agent.get_action(state)
            hand_actions.append(action)
            state, reward, done = env.step(action)
        
        # Record the result of this hand
        final_result = env.results_history[-1] if env.results_history else {'result': 'Unknown', 'reward': 0}
        
        hand_results.append({
            'hand_num': hand_num,
            'result': final_result['result'],
            'reward': final_result['reward'],
            'starting_true_count': starting_true_count,
            'bet': initial_bet,
            'bankroll': env.bankroll,
            'actions': hand_actions
        })
        
        # Track the count after every hand
        count_tracking.append({
            'hand_num': hand_num,
            'running_count': env.counter.running_count,
            'true_count': env.counter.true_count,
            'decks_remaining': env.counter.decks_remaining
        })
    
    # Convert to DataFrames for analysis
    results_df = pd.DataFrame(hand_results)
    count_df = pd.DataFrame(count_tracking)
    
    return env, agent, results_df, count_df

def analyze_simulation(results_df, count_df):
    """
    Analyze the results of a simulation run
    """
    # Overall performance
    total_hands = len(results_df)
    win_count = results_df[results_df['reward'] > 0].shape[0]
    loss_count = results_df[results_df['reward'] < 0].shape[0]
    push_count = results_df[results_df['reward'] == 0].shape[0]
    
    win_percentage = win_count / total_hands * 100
    loss_percentage = loss_count / total_hands * 100
    push_percentage = push_count / total_hands * 100
    
    profit = results_df['reward'].sum()
    initial_bankroll = results_df['bankroll'].iloc[0] - results_df['reward'].iloc[0]
    final_bankroll = results_df['bankroll'].iloc[-1]
    
    roi = (profit / initial_bankroll) * 100
    
    print("=== Simulation Results ===")
    print(f"Total hands played: {total_hands}")
    print(f"Wins: {win_count} ({win_percentage:.2f}%)")
    print(f"Losses: {loss_count} ({loss_percentage:.2f}%)")
    print(f"Pushes: {push_count} ({push_percentage:.2f}%)")
    print(f"Initial bankroll: ${initial_bankroll:.2f}")
    print(f"Final bankroll: ${final_bankroll:.2f}")
    print(f"Total profit: ${profit:.2f}")
    print(f"ROI: {roi:.2f}%")
    print(f"Profit per hand: ${profit / total_hands:.2f}")
    
    # Plot bankroll over time
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['hand_num'], results_df['bankroll'])
    plt.title('Bankroll Progression')
    plt.xlabel('Hand Number')
    plt.ylabel('Bankroll ($)')
    plt.grid(True)
    plt.savefig('simulation_bankroll.png')
    plt.show()
    
    # Plot true count distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(results_df['starting_true_count'], kde=True, bins=25)
    plt.title('True Count Distribution')
    plt.xlabel('True Count')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('true_count_distribution.png')
    plt.show()
    
    # Analyze win rate by true count
    results_df['count_bin'] = pd.cut(results_df['starting_true_count'], 
                                    bins=np.arange(-6, 7, 1), 
                                    labels=[f"{i}" for i in range(-6, 6)])
    
    count_win_rates = results_df.groupby('count_bin').apply(
        lambda x: pd.Series({
            'win_rate': (x['reward'] > 0).mean() * 100,
            'profit_per_hand': x['reward'].mean(),
            'hands': len(x)
        })
    ).reset_index()
    
    # Only show bins with sufficient data
    count_win_rates = count_win_rates[count_win_rates['hands'] >= 10]
    
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    sns.barplot(x='count_bin', y='win_rate', data=count_win_rates)
    plt.title('Win Rate by True Count')
    plt.xlabel('True Count')
    plt.ylabel('Win Rate (%)')
    plt.grid(axis='y')
    
    plt.subplot(2, 1, 2)
    sns.barplot(x='count_bin', y='profit_per_hand', data=count_win_rates)
    plt.title('Average Profit per Hand by True Count')
    plt.xlabel('True Count')
    plt.ylabel('Profit per Hand ($)')
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig('win_rate_by_count.png')
    plt.show()
    
    # Analyze bet sizing
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='starting_true_count', y='bet', data=results_df, alpha=0.5)
    plt.title('Bet Size vs. True Count')
    plt.xlabel('True Count')
    plt.ylabel('Bet Size ($)')
    plt.grid(True)
    plt.savefig('bet_size_vs_count.png')
    plt.show()
    
    # Analyze win rate by bet size
    results_df['bet_bin'] = pd.cut(results_df['bet'], 
                                  bins=[0, 100, 500, 1000, 2000, 5000, float('inf')],
                                  labels=['$50-100', '$100-500', '$500-1000', 
                                          '$1000-2000', '$2000-5000', '$5000+'])
    
    bet_win_rates = results_df.groupby('bet_bin').apply(
        lambda x: pd.Series({
            'win_rate': (x['reward'] > 0).mean() * 100,
            'profit_per_hand': x['reward'].mean(),
            'hands': len(x)
        })
    ).reset_index()
    
    bet_win_rates = bet_win_rates[bet_win_rates['hands'] >= 5]
    
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    sns.barplot(x='bet_bin', y='win_rate', data=bet_win_rates)
    plt.title('Win Rate by Bet Size')
    plt.xlabel('Bet Size')
    plt.ylabel('Win Rate (%)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    plt.subplot(2, 1, 2)
    sns.barplot(x='bet_bin', y='profit_per_hand', data=bet_win_rates)
    plt.title('Average Profit per Hand by Bet Size')
    plt.xlabel('Bet Size')
    plt.ylabel('Profit per Hand ($)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig('win_rate_by_bet.png')
    plt.show()
    
    # Analyzing true count effects on action choices
    plt.figure(figsize=(14, 8))
    
    # Group all the count-based decisions
    action_map = {0: 'Hit', 1: 'Stand', 2: 'Double', 3: 'Split'}
    action_df = pd.DataFrame([
        {'true_count': row['starting_true_count'], 'action': action_map[action]}
        for _, row in results_df.iterrows()
        for action in row['actions']
    ])
    
    action_counts = action_df.groupby(['true_count', 'action']).size().reset_index(name='count')
    pivot_actions = action_counts.pivot_table(index='true_count', columns='action', values='count', fill_value=0)
    
    # Normalize by row
    pivot_actions_pct = pivot_actions.div(pivot_actions.sum(axis=1), axis=0) * 100
    
    # Plot the percentage of each action by true count
    pivot_actions_pct.plot(kind='bar', stacked=True, figsize=(14, 8))
    plt.title('Action Distribution by True Count')
    plt.xlabel('True Count')
    plt.ylabel('Percentage of Actions')
    plt.grid(axis='y')
    plt.legend(title='Action')
    plt.savefig('action_distribution_by_count.png')
    plt.show()
    
    return {
        'total_hands': total_hands,
        'win_percentage': win_percentage,
        'loss_percentage': loss_percentage,
        'push_percentage': push_percentage,
        'initial_bankroll': initial_bankroll,
        'final_bankroll': final_bankroll,
        'profit': profit,
        'roi': roi
    }

def main():
    """
    Main function to run the simulation and display results
    """
    print("Starting Blackjack Simulation with Card Counting")
    
    # Run simulation with 5000 hands
    env, agent, results_df, count_df = simulate_game(num_hands=5000, initial_bankroll=10000)
    
    # Analyze the results
    stats = analyze_simulation(results_df, count_df)
    
    # Print key strategy deviations
    print("\nKey Strategy Deviations:")
    for i, row in enumerate(agent.deviation_history[:20]):  # Top 20 deviations
        player_val = row['player_value']
        dealer_card = row['dealer_up_card']
        true_count = row['true_count']
        default = row['default_action']
        chosen = row['chosen_action']
        
        action_map = {0: 'Hit', 1: 'Stand', 2: 'Double', 3: 'Split'}
        default_action = action_map.get(default, 'Unknown')
        chosen_action = action_map.get(chosen, 'Unknown')
        
        print(f"{i+1}. Player {player_val} vs Dealer {dealer_card}, True Count {true_count:.1f}: {default_action} → {chosen_action}")
    
    # Compare different betting strategies
    print("\nComparing Betting Strategies:")
    
    # Traditional flat betting
    _, _, flat_results, _ = simulate_game(
        num_hands=1000, 
        initial_bankroll=10000,
        min_bet=100,
        max_bet=100,  # Flat bet of $100
        show_progress=False
    )
    
    # Conservative card counting (smaller bet spread)
    _, _, conservative_results, _ = simulate_game(
        num_hands=1000, 
        initial_bankroll=10000,
        min_bet=50,
        max_bet=500,  # Max 10x spread
        show_progress=False
    )
    
    # Aggressive card counting (larger bet spread)
    _, _, aggressive_results, _ = simulate_game(
        num_hands=1000, 
        initial_bankroll=10000,
        min_bet=50,
        max_bet=5000,  # Max 100x spread
        show_progress=False
    )
    
    print(f"Flat betting ($100): ${flat_results['reward'].sum():.2f} profit")
    print(f"Conservative counting ($50-$500): ${conservative_results['reward'].sum():.2f} profit")
    print(f"Aggressive counting ($50-$5000): ${aggressive_results['reward'].sum():.2f} profit")
    
    # Compare profit over time
    plt.figure(figsize=(12, 6))
    plt.plot(flat_results.index, flat_results['reward'].cumsum(), label='Flat $100')
    plt.plot(conservative_results.index, conservative_results['reward'].cumsum(), label='Conservative $50-$500')
    plt.plot(aggressive_results.index, aggressive_results['reward'].cumsum(), label='Aggressive $50-$5000')
    plt.title('Profit Comparison: Different Betting Strategies')
    plt.xlabel('Hand Number')
    plt.ylabel('Cumulative Profit ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('betting_strategy_comparison.png')
    plt.show()
    
    print("\nSimulation complete!")

    # Add Q-learning training
    print("\nTraining Q-Learning Agent...")
    q_agent, q_rewards = train_q_agent(num_episodes=500000)

if __name__ == "__main__":
    main()