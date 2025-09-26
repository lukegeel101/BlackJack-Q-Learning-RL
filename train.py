import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import the Blackjack environment and agents
from cardCounting import BlackjackEnv, BasicStrategyAgent, CardCounter
import train_q_agent

def train_agent(num_episodes=10000):
    """
    Train a blackjack agent and collect results
    """
    env = BlackjackEnv(num_decks=8, min_bet=50, max_bet=5000)
    agent = BasicStrategyAgent()
    
    bankroll_history = []
    count_history = []
    bet_history = []
    
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.get_action(state)
            state, reward, done = env.step(action)
            
            # Record data
            if state:
                bankroll_history.append(state['bankroll'])
                count_history.append(state['true_count'])
                bet_history.append(state['bet'])
    
    results_df = pd.DataFrame(env.results_history)
    deviations_df = pd.DataFrame(agent.deviation_history)
    
    return env, agent, results_df, deviations_df, bankroll_history, count_history, bet_history

def analyze_results(results_df, deviations_df, bankroll_history, count_history, bet_history):
    """
    Analyze and visualize training results
    """
    # Overall statistics
    total_hands = len(results_df)
    win_count = results_df[results_df['reward'] > 0].shape[0]
    loss_count = results_df[results_df['reward'] < 0].shape[0]
    push_count = results_df[results_df['reward'] == 0].shape[0]
    
    win_percentage = win_count / total_hands * 100
    loss_percentage = loss_count / total_hands * 100
    push_percentage = push_count / total_hands * 100
    
    print(f"Total hands played: {total_hands}")
    print(f"Wins: {win_count} ({win_percentage:.2f}%)")
    print(f"Losses: {loss_count} ({loss_percentage:.2f}%)")
    print(f"Pushes: {push_count} ({push_percentage:.2f}%)")
    
    # Bankroll progression
    plt.figure(figsize=(12, 6))
    plt.plot(bankroll_history)
    plt.title('Bankroll Progression')
    plt.xlabel('Hand Number')
    plt.ylabel('Bankroll ($)')
    plt.grid(True)
    plt.savefig('bankroll_progression.png')
    plt.close()
    
    # Bet size vs. True Count
    plt.figure(figsize=(12, 6))
    plt.scatter(count_history, bet_history, alpha=0.1)
    plt.title('Bet Size vs. True Count')
    plt.xlabel('True Count')
    plt.ylabel('Bet Size ($)')
    plt.grid(True)
    plt.savefig('bet_size_vs_true_count.png')
    plt.close()
    
    # Win rate vs. True Count
    count_bins = np.arange(-6, 7, 1)
    count_labels = [f"{i}" for i in range(-6, 6)]
    
    results_df['count_bin'] = pd.cut(results_df['true_count'], bins=count_bins, labels=count_labels)
    
    win_rates = []
    count_groups = []
    
    for group, data in results_df.groupby('count_bin'):
        if len(data) > 10:  # Only consider bins with enough data
            win_rate = data[data['reward'] > 0].shape[0] / len(data) * 100
            win_rates.append(win_rate)
            count_groups.append(group)
    
    plt.figure(figsize=(12, 6))
    plt.bar(count_groups, win_rates)
    plt.title('Win Rate vs. True Count')
    plt.xlabel('True Count')
    plt.ylabel('Win Rate (%)')
    plt.grid(True)
    plt.savefig('win_rate_vs_true_count.png')
    plt.close()
    
    # Strategy deviations analysis
    if not deviations_df.empty:
        deviation_counts = deviations_df.groupby(['player_value', 'dealer_up_card', 'default_action', 'chosen_action']).size().reset_index(name='count')
        deviation_counts = deviation_counts.sort_values('count', ascending=False)
        
        print("\nTop 10 Strategy Deviations:")
        for _, row in deviation_counts.head(10).iterrows():
            player = row['player_value']
            dealer = row['dealer_up_card']
            default = row['default_action']
            chosen = row['chosen_action']
            count = row['count']
            
            action_map = {0: 'Hit', 1: 'Stand', 2: 'Double', 3: 'Split'}
            default_action = action_map.get(default, 'Unknown')
            chosen_action = action_map.get(chosen, 'Unknown')
            
            print(f"Player: {player} vs Dealer: {dealer} - Changed from {default_action} to {chosen_action} ({count} times)")
    
    # Create a heatmap showing win rate by player hand vs dealer up card
    pivot_data = pd.pivot_table(
        results_df,
        values='reward',
        index='player_hand',
        columns='dealer_hand',
        aggfunc=lambda x: (np.sum(x > 0) / len(x) * 100) if len(x) > 0 else 0
    )
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt='.1f')
    plt.title('Win Rate (%) by Player Hand vs Dealer Hand')
    plt.tight_layout()
    plt.savefig('win_rate_heatmap.png')
    plt.close()
    
    return {
        'total_hands': total_hands,
        'win_percentage': win_percentage,
        'loss_percentage': loss_percentage,
        'push_percentage': push_percentage,
        'final_bankroll': bankroll_history[-1] if bankroll_history else 0
    }

def visualize_count_based_strategy():
    """
    Create visualizations showing how strategy changes with count
    """
    # Initialize the agent
    agent = BasicStrategyAgent()
    
    # Define the count ranges to visualize
    count_ranges = [-5, 0, 5]
    
    # 1. Hard totals strategy matrices
    for true_count in count_ranges:
        matrix = np.zeros((10, 10))  # Player values 12-21 vs Dealer 2-A
        
        for player_val in range(12, 22):
            for dealer_val in range(2, 12):
                # Create a dummy state
                state = {
                    'player_hand': None,
                    'player_value': player_val,
                    'dealer_up_card': dealer_val,
                    'is_soft': False,
                    'can_split': False,
                    'true_count': true_count
                }
                
                action = agent.get_action(state)
                matrix[player_val - 12, dealer_val - 2] = action
        
        plt.figure(figsize=(12, 10))
        # Change fmt='d' to fmt='.0f' to format as floats with no decimal places
        sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt='.0f',
                    xticklabels=['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A'],
                    yticklabels=[str(i) for i in range(12, 22)])
        plt.title(f'Hard Totals Strategy Matrix (True Count = {true_count})')
        plt.xlabel('Dealer Up Card')
        plt.ylabel('Player Hand Value')
        plt.savefig(f'hard_strategy_count_{true_count}.png')
        plt.close()
    
    # 2. Visualize key decision points with count ranges
    key_decisions = [
        {'player_value': 16, 'dealer_up_card': 10, 'is_soft': False, 'can_split': False, 'name': '16_vs_10'},
        {'player_value': 15, 'dealer_up_card': 10, 'is_soft': False, 'can_split': False, 'name': '15_vs_10'},
        {'player_value': 12, 'dealer_up_card': 3, 'is_soft': False, 'can_split': False, 'name': '12_vs_3'},
        {'player_value': 12, 'dealer_up_card': 2, 'is_soft': False, 'can_split': False, 'name': '12_vs_2'},
        {'player_value': 11, 'dealer_up_card': 11, 'is_soft': False, 'can_split': False, 'name': '11_vs_A'},
        {'player_value': 10, 'dealer_up_card': 10, 'is_soft': False, 'can_split': False, 'name': '10_vs_10'},
    ]
    
    count_range = list(range(-6, 7))
    
    for decision in key_decisions:
        actions = []
        for true_count in count_range:
            state = {
                'player_hand': None,
                'player_value': decision['player_value'],
                'dealer_up_card': decision['dealer_up_card'],
                'is_soft': decision['is_soft'],
                'can_split': decision['can_split'],
                'true_count': true_count
            }
            
            action = agent.get_action(state)
            actions.append(action)
        
        # Convert action codes to names
        action_names = ['Hit', 'Stand', 'Double', 'Split']
        action_labels = [action_names[a] for a in actions]
        
        plt.figure(figsize=(12, 6))
        # Use different colors for different actions
        colors = {'Hit': 'red', 'Stand': 'blue', 'Double': 'green', 'Split': 'purple'}
        for i, (tc, action) in enumerate(zip(count_range, action_labels)):
            plt.bar(tc, 1, color=colors[action], label=action if i == 0 or action_labels[i-1] != action else "")
        
        # Remove duplicate legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.title(f'Strategy for {decision["player_value"]} vs {decision["dealer_up_card"]} by True Count')
        plt.xlabel('True Count')
        plt.ylabel('Action')
        plt.yticks([])
        plt.grid(axis='x')
        plt.savefig(f'strategy_by_count_{decision["name"]}.png')
        plt.close()

def run_reinforcement_learning():
    """
    Run a Q-learning reinforcement learning algorithm on the blackjack environment
    to discover optimal strategy deviations based on count
    """
    # Initialize the environment and parameters
    env = BlackjackEnv(num_decks=8, min_bet=50, max_bet=5000)
    
    # Define the state space (simplified)
    # player_values: 4-21, dealer_upcard: 2-11, is_soft: 0-1, count_bin: -6 to 6 (binned)
    # This creates a discrete state space for Q-learning
    player_values = list(range(4, 22))
    dealer_values = list(range(2, 12))
    is_soft_values = [0, 1]
    count_bins = list(range(-6, 7))
    
    # Initialize Q-table
    Q = {}
    basic_agent = BasicStrategyAgent()
    
    # Rough expected values for different outcomes
    # These values are approximations and can be tuned
    win_value = 1.0      # Expected value for winning
    loss_value = -1.0    # Expected value for losing
    push_value = 0.0     # Expected value for a push
    double_win = 2.0     # Expected value for winning after doubling
    double_loss = -2.0   # Expected value for losing after doubling
    
    # Probability adjustments based on player advantage
    advantage_by_count = {
        -6: -0.015, -5: -0.012, -4: -0.01, -3: -0.007, -2: -0.005, 
        -1: -0.002, 0: 0.0, 
        1: 0.002, 2: 0.005, 3: 0.007, 4: 0.01, 5: 0.012, 6: 0.015
    }
    
    # Training parameters
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon_start = 0.5  # Initial exploration rate (higher)
    epsilon_end = 0.01  # Final exploration rate
    num_episodes = 100000
    
    # Create Q-table keys for all states and actions
    for player_val in player_values:
        for dealer_val in dealer_values:
            for is_soft in is_soft_values:
                for count_bin in count_bins:


                    # Create a test state to get basic strategy action
                    test_state = {
                        'player_hand': None,
                        'player_value': player_val,
                        'dealer_up_card': dealer_val,
                        'is_soft': is_soft == 1,
                        'can_split': False,
                        'true_count': count_bin
                    }
                    
                    # Get the basic strategy action for this state
                    basic_action = basic_agent.get_action(test_state)
                    
                    # Initialize all actions to slightly below expected value of basic strategy
                    # This encourages exploration of all actions
                    base_value = -0.2
                    
                    # Initialize Q-values for all possible actions for all states
                    # This ensures all state-action pairs exist in the Q-table
                    Q[((player_val, dealer_val, is_soft, count_bin), 0)] = base_value  # Hit
                    Q[((player_val, dealer_val, is_soft, count_bin), 1)] = base_value  # Stand
                    Q[((player_val, dealer_val, is_soft, count_bin), 2)] = base_value  # Double
                    Q[((player_val, dealer_val, is_soft, count_bin), 3)] = base_value  # Split

                    # Adjust based on count advantage
                    count_adjustment = advantage_by_count[count_bin]
                    
                    # Assign higher initial value to the basic strategy action
                    # and adjust based on the count
                    if basic_action == 0:  # Hit
                        # Hitting is more valuable with more low cards left (negative count)
                        hit_adjustment = -count_adjustment * 0.5  # Reverse the effect slightly
                        Q[((player_val, dealer_val, is_soft, count_bin), 0)] = 0.0 + hit_adjustment
                    
                    elif basic_action == 1:  # Stand
                        # Standing is more valuable with more high cards left (positive count)
                        stand_adjustment = count_adjustment 
                        Q[((player_val, dealer_val, is_soft, count_bin), 1)] = 0.0 + stand_adjustment
                    
                    elif basic_action == 2:  # Double
                        # Doubling is more valuable with more high cards left (positive count)
                        double_adjustment = count_adjustment * 2.0  # Double the effect for doubling
                        Q[((player_val, dealer_val, is_soft, count_bin), 2)] = 0.0 + double_adjustment
                    
                    elif basic_action == 3:  # Split
                        # Splitting pairs like A,8 is better with more high cards
                        # Splitting pairs like 2,3,4,6,7 is better with more low cards
                        split_adjustment = count_adjustment if player_val >= 16 else -count_adjustment
                        Q[((player_val, dealer_val, is_soft, count_bin), 3)] = 0.0 + split_adjustment
    
    # Training loop
    for episode in tqdm(range(num_episodes)):
        # Decay epsilon linearly from epsilon_start to epsilon_end
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (episode / num_episodes)
        
        state = env.reset()
        done = False
        
        while not done:
            # Convert state to discrete form for Q-table
            player_val = state['player_value']
            dealer_val = state['dealer_up_card']
            is_soft = 1 if state['is_soft'] else 0
            
            # Bin the true count
            count = state['true_count']
            count_bin = max(-6, min(6, int(round(count))))
            
            # Get valid actions
            valid_actions = [0, 1]  # Hit and stand are always valid
            
            # Double is valid only on first two cards
            if len(state['player_hand'].cards) == 2:
                valid_actions.append(2)  # Double
            
            # Split is valid only for pairs
            if state['can_split']:
                valid_actions.append(3)  # Split
            
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                # Explore: choose a random valid action
                action = np.random.choice(valid_actions)
            else:
                # Exploit: choose the best action from Q-table
                q_values = [Q.get(((player_val, dealer_val, is_soft, count_bin), a), float('-inf')) 
                           for a in valid_actions]
                action = valid_actions[np.argmax(q_values)]
            
            # Take action
            next_state, reward, done = env.step(action)
            
            # Update Q-value
            if not done:
                # Convert next state to discrete form
                next_player_val = next_state['player_value']
                next_dealer_val = next_state['dealer_up_card']
                next_is_soft = 1 if next_state['is_soft'] else 0
                next_count = next_state['true_count']
                next_count_bin = max(-6, min(6, int(round(next_count))))
                
                # Get next valid actions
                next_valid_actions = [0, 1]  # Hit and stand are always valid
                
                if len(next_state['player_hand'].cards) == 2:
                    next_valid_actions.append(2)  # Double
                
                if next_state['can_split']:
                    next_valid_actions.append(3)  # Split
                
                # Max Q-value for next state
                next_q_values = [Q.get(((next_player_val, next_dealer_val, next_is_soft, next_count_bin), a), 0.0) 
                                for a in next_valid_actions]
                max_next_q = max(next_q_values)
                
                # Q-learning update
                Q[((player_val, dealer_val, is_soft, count_bin), action)] += alpha * (
                    reward + gamma * max_next_q - Q[((player_val, dealer_val, is_soft, count_bin), action)]
                )
            else:
                # Terminal state update
                Q[((player_val, dealer_val, is_soft, count_bin), action)] += alpha * (
                    reward - Q[((player_val, dealer_val, is_soft, count_bin), action)]
                )
    
    # Extract learned strategy and compare with basic strategy
    learned_strategy = {}
    basic_agent = BasicStrategyAgent()
    
    for player_val in range(4, 22):  # Look at all player values
        for dealer_val in range(2, 12):
            for count_bin in range(-6, 9):  # Examine all count bins
                # Create a test state
                test_state = {
                    'player_hand': None,
                    'player_value': player_val,
                    'dealer_up_card': dealer_val,
                    'is_soft': False,
                    'can_split': False,
                    'true_count': count_bin
                }
                
                # Get basic strategy action
                basic_action = basic_agent.get_action(test_state)
                
                valid_actions = [0, 1]  # Hit and stand are always valid
                if player_val <= 11:  # Only allow double on 11 or less (conservative constraint)
                    valid_actions.append(2)  # Double

                q_values = [Q.get(((player_val, dealer_val, 0, count_bin), a), float('-inf')) 
                        for a in valid_actions]
                rl_action = valid_actions[np.argmax(q_values)] if q_values else 0  # Default to hit if no valid actions
                
                # If different, record the deviation
                if basic_action != rl_action:
                    key = (player_val, dealer_val, count_bin)
                    learned_strategy[key] = {
                        'basic_action': basic_action,
                        'rl_action': rl_action,
                        'q_value_diff': q_values[rl_action] - q_values[basic_action]
                    }
    
    # Print learned deviations
    print("\nLearned Strategy Deviations:")
    
    action_map = {0: 'Hit', 1: 'Stand', 2: 'Double', 3: 'Split'}
    
    for (player_val, dealer_val, count_bin), data in sorted(
        learned_strategy.items(), 
        key=lambda x: abs(x[1]['q_value_diff']), 
        reverse=True
    )[:20]:  # Show top 20 strongest deviations
        print(f"Player: {player_val} vs Dealer: {dealer_val} at count {count_bin}: " +
              f"Basic: {action_map[data['basic_action']]} → RL: {action_map[data['rl_action']]} " +
              f"(Q-diff: {data['q_value_diff']:.2f})")
    
    return Q, learned_strategy

def train_q_learning_agent():
    """
    Train Q-learning agent with proper settings
    """
    # Your improved Q-learning training code here
    agent, rewards = train_q_agent(num_episodes=500000)
    return agent, rewards

def main():
    # Run the training
    print("Training basic strategy agent...")
    env, agent, results_df, deviations_df, bankroll_history, count_history, bet_history = train_agent(num_episodes=10000)
    
    # Analyze results
    print("\nAnalyzing results...")
    stats = analyze_results(results_df, deviations_df, bankroll_history, count_history, bet_history)
    
    # Visualize count-based strategy
    print("\nVisualizing count-based strategy...")
    visualize_count_based_strategy()
    
    # Optional: Run reinforcement learning
    print("\nRunning reinforcement learning to discover optimal count-based strategy...")
    Q, learned_strategy = run_reinforcement_learning()
    
    # Display summary
    print("\nSummary:")
    print(f"Total hands played: {stats['total_hands']}")
    print(f"Win rate: {stats['win_percentage']:.2f}%")
    print(f"Final bankroll: ${stats['final_bankroll']:.2f}")
    print(f"Strategy deviations recorded: {len(deviations_df)}")

if __name__ == "__main__":
    main()