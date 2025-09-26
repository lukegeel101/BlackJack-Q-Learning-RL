import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Import the classes from our implementation
from cardCounting import BlackjackEnv, BasicStrategyAgent, CardCounter

def simulate_game(num_hands=1000, initial_bankroll=10000, show_progress=True, min_bet=50, max_bet=5000):
    """
    Simulate a blackjack game with basic strategy only (no card counting)
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
        
        # Track the starting count (will always be 0 now)
        starting_true_count = state['true_count']
        
        # Track the initial bet (will always be min_bet now)
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
            'starting_true_count': starting_true_count,  # Always 0
            'bet': initial_bet,  # Always min_bet
            'bankroll': env.bankroll,
            'actions': hand_actions
        })
        
        # Track the count after every hand (will always be 0)
        count_tracking.append({
            'hand_num': hand_num,
            'running_count': env.counter.running_count,  # Always 0
            'true_count': env.counter.true_count,        # Always 0
            'decks_remaining': env.counter.decks_remaining
        })
    
    # Convert to DataFrames for analysis
    results_df = pd.DataFrame(hand_results)
    count_df = pd.DataFrame(count_tracking)
    
    return env, agent, results_df, count_df

def analyze_simulation(results_df, count_df):
    """
    Analyze the results of a basic strategy simulation run
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
    
    print("=== Basic Strategy Simulation Results ===")
    print(f"Total hands played: {total_hands}")
    print(f"Wins: {win_count} ({win_percentage:.2f}%)")
    print(f"Losses: {loss_count} ({loss_percentage:.2f}%)")
    print(f"Pushes: {push_count} ({push_percentage:.2f}%)")
    print(f"Initial bankroll: ${initial_bankroll:.2f}")
    print(f"Final bankroll: ${final_bankroll:.2f}")
    print(f"Total profit: ${profit:.2f}")
    print(f"ROI: {roi:.2f}%")
    print(f"Profit per hand: ${profit / total_hands:.2f}")
    print(f"Flat bet amount: ${results_df['bet'].iloc[0]:.2f}")
    
    # Plot bankroll over time
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['hand_num'], results_df['bankroll'])
    plt.title('Bankroll Progression (Basic Strategy)')
    plt.xlabel('Hand Number')
    plt.ylabel('Bankroll ($)')
    plt.grid(True)
    plt.savefig('basic_strategy_bankroll.png')
    plt.show()
    
    # Analyze win rate by result type
    result_counts = results_df['result'].value_counts()
    
    plt.figure(figsize=(10, 6))
    result_counts.plot(kind='bar')
    plt.title('Outcome Distribution (Basic Strategy)')
    plt.xlabel('Result Type')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('basic_strategy_outcomes.png')
    plt.show()
    
    # Analyze action distribution
    action_map = {0: 'Hit', 1: 'Stand', 2: 'Double', 3: 'Split'}
    all_actions = []
    for actions_list in results_df['actions']:
        all_actions.extend(actions_list)
    
    action_df = pd.DataFrame([{'action': action_map[action]} for action in all_actions])
    action_counts = action_df['action'].value_counts()
    
    plt.figure(figsize=(10, 6))
    action_counts.plot(kind='bar')
    plt.title('Action Distribution (Basic Strategy)')
    plt.xlabel('Action')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('basic_strategy_actions.png')
    plt.show()
    
    # Show cumulative profit progression
    plt.figure(figsize=(12, 6))
    cumulative_profit = results_df['reward'].cumsum()
    plt.plot(results_df['hand_num'], cumulative_profit)
    plt.title('Cumulative Profit Progression (Basic Strategy)')
    plt.xlabel('Hand Number')
    plt.ylabel('Cumulative Profit ($)')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.grid(True)
    plt.savefig('basic_strategy_cumulative_profit.png')
    plt.show()
    
    # Analyze streaks (winning/losing runs)
    streaks = []
    current_streak = 0
    current_streak_type = None
    
    for _, row in results_df.iterrows():
        if row['reward'] > 0:  # Win
            if current_streak_type == 'win':
                current_streak += 1
            else:
                if current_streak_type is not None:
                    streaks.append((current_streak_type, current_streak))
                current_streak = 1
                current_streak_type = 'win'
        elif row['reward'] < 0:  # Loss
            if current_streak_type == 'loss':
                current_streak += 1
            else:
                if current_streak_type is not None:
                    streaks.append((current_streak_type, current_streak))
                current_streak = 1
                current_streak_type = 'loss'
        else:  # Push - doesn't break streaks
            continue
    
    # Add the final streak
    if current_streak_type is not None:
        streaks.append((current_streak_type, current_streak))
    
    if streaks:
        win_streaks = [length for streak_type, length in streaks if streak_type == 'win']
        loss_streaks = [length for streak_type, length in streaks if streak_type == 'loss']
        
        print(f"\n=== Streak Analysis ===")
        if win_streaks:
            print(f"Longest winning streak: {max(win_streaks)} hands")
            print(f"Average winning streak: {np.mean(win_streaks):.1f} hands")
        if loss_streaks:
            print(f"Longest losing streak: {max(loss_streaks)} hands")
            print(f"Average losing streak: {np.mean(loss_streaks):.1f} hands")
    
    # Calculate house edge
    house_edge = -profit / (total_hands * results_df['bet'].iloc[0]) * 100
    print(f"\nHouse edge experienced: {house_edge:.3f}%")
    print(f"Expected house edge (basic strategy): ~0.5%")
    
    # Show some basic strategy statistics
    blackjack_count = results_df[results_df['result'] == 'Blackjack'].shape[0]
    blackjack_rate = blackjack_count / total_hands * 100
    bust_count = results_df[results_df['result'].str.contains('Bust', na=False)].shape[0]
    bust_rate = bust_count / total_hands * 100
    
    print(f"\n=== Basic Strategy Statistics ===")
    print(f"Blackjack rate: {blackjack_rate:.2f}% (expected ~4.8%)")
    print(f"Bust rate: {bust_rate:.2f}%")
    
    return {
        'total_hands': total_hands,
        'win_percentage': win_percentage,
        'loss_percentage': loss_percentage,
        'push_percentage': push_percentage,
        'initial_bankroll': initial_bankroll,
        'final_bankroll': final_bankroll,
        'profit': profit,
        'roi': roi,
        'house_edge': house_edge
    }

def main():
    """
    Main function to run the card counting simulation and display results
    """
    print("Starting Blackjack Simulation with Card Counting")
    print("(Deep Q-learning disabled, using basic strategy with count deviations)")
    
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

if __name__ == "__main__":
    main()