import numpy as np
import random
from collections import defaultdict
import pickle

class ImprovedQAgent:
    """
    Improved Q-Learning agent with better state representation and training
    """
    def __init__(self, learning_rate=0.1, discount_factor=0.95, 
                 epsilon_start=0.3, epsilon_end=0.01, epsilon_decay=0.995):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Use defaultdict for automatic initialization
        self.Q = defaultdict(lambda: defaultdict(float))
        
        # Initialize with basic strategy values to speed up learning
        self.initialize_with_basic_strategy()
        
        # Track learning progress
        self.training_rewards = []
        self.training_mode = True
        
    def initialize_with_basic_strategy(self):
        """
        Initialize Q-values with basic strategy to give Q-learning a head start
        """
        # Basic strategy initialization - gives positive values to known good moves
        basic_strategy_bonus = 0.5
        
        # Some key basic strategy rules
        basic_rules = [
            # (player_value, dealer_upcard, is_soft, action, bonus)
            # Always stand on 21 and 20
            *[(21, dealer, False, 1, basic_strategy_bonus) for dealer in range(2, 12)],
            *[(20, dealer, False, 1, basic_strategy_bonus) for dealer in range(2, 12)],
        
            *[(11, dealer, False, 2, basic_strategy_bonus) for dealer in range(2, 12)],
            (10, 9, False, 2, basic_strategy_bonus),   # Double on 10 vs 9
            (16, 10, False, 0, -basic_strategy_bonus), # Hit 16 vs 10 (default)
            (12, 6, False, 1, basic_strategy_bonus),   # Stand 12 vs 6
        ]
        
        for player_val, dealer_val, is_soft, action, bonus in basic_rules:
            for count in range(-6, 7):
                state = self._get_state_key(player_val, dealer_val, is_soft, False, count)
                self.Q[state][action] = bonus
    
    def _get_state_key(self, player_value, dealer_upcard, is_soft, can_split, true_count):
        """
        Create a more efficient state representation
        """
        # Bin the true count to reduce state space
        count_bin = max(-3, min(3, int(true_count)))  # Reduce count bins to -3 to +3
        
        # Simplify player values - group similar hands
        if player_value >= 17:
            player_bin = 17  # 17-21 all play similarly
        elif player_value <= 8:
            player_bin = 8   # Always hit low hands
        else:
            player_bin = player_value
            
        return (player_bin, dealer_upcard, is_soft, can_split, count_bin)
    
    def get_action(self, state):
        """
        Get action using epsilon-greedy policy
        """
        if not state:
            return 1
            
        state_key = self._get_state_key(
            state['player_value'],
            state['dealer_up_card'], 
            state['is_soft'],
            state['can_split'],
            state['true_count']
        )
        
        # Get valid actions
        valid_actions = [0, 1]  # Hit, Stand always valid
        if len(state['player_hand'].cards) == 2:
            valid_actions.append(2)  # Double
        if state['can_split']:
            valid_actions.append(3)  # Split
            
        # Exploration vs exploitation
        if self.training_mode and random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            # Choose best action
            q_values = [self.Q[state_key][a] for a in valid_actions]
            best_action_idx = np.argmax(q_values)
            return valid_actions[best_action_idx]
    
    def update(self, state, action, reward, next_state):
        """
        Update Q-values using Q-learning update rule
        """
        if not self.training_mode:
            return
            
        state_key = self._get_state_key(
            state['player_value'],
            state['dealer_up_card'],
            state['is_soft'], 
            state['can_split'],
            state['true_count']
        )
        
        if next_state:
            next_state_key = self._get_state_key(
                next_state['player_value'],
                next_state['dealer_up_card'],
                next_state['is_soft'],
                next_state['can_split'], 
                next_state['true_count']
            )
            
            # Get valid actions for next state
            valid_next_actions = [0, 1]
            if len(next_state['player_hand'].cards) == 2:
                valid_next_actions.append(2)
            if next_state['can_split']:
                valid_next_actions.append(3)
                
            max_next_q = max([self.Q[next_state_key][a] for a in valid_next_actions])
        else:
            max_next_q = 0
            
        # Q-learning update
        old_q = self.Q[state_key][action]
        self.Q[state_key][action] = old_q + self.lr * (reward + self.gamma * max_next_q - old_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def set_training_mode(self, training=True):
        """
        Switch between training and evaluation mode
        """
        self.training_mode = training
        if not training:
            self.epsilon = 0  # No exploration during evaluation
    
    def save_model(self, filename):
        """
        Save the trained Q-table
        """
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.Q), f)
    
    def load_model(self, filename):
        """
        Load a trained Q-table
        """
        with open(filename, 'rb') as f:
            loaded_q = pickle.load(f)
            self.Q = defaultdict(lambda: defaultdict(float), loaded_q)

def train_q_agent(num_episodes=500000, save_interval=50000):
    """
    Train the Q-learning agent with proper training protocol
    """
    from cardCounting import BlackjackEnv  # Import your environment
    
    env = BlackjackEnv(num_decks=8, min_bet=50, max_bet=5000)
    agent = ImprovedQAgent()
    
    episode_rewards = []
    
    print("Training Q-Learning Agent...")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            
            # Update Q-values
            agent.update(state, action, reward, next_state)
            
            episode_reward += reward
            state = next_state
        
        episode_rewards.append(episode_reward)
        
        # Print progress
        if episode % 10000 == 0:
            avg_reward = np.mean(episode_rewards[-1000:]) if len(episode_rewards) >= 1000 else np.mean(episode_rewards)
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        # Save intermediate models
        if episode % save_interval == 0 and episode > 0:
            agent.save_model(f'q_model_episode_{episode}.pkl')
    
    # Final save
    agent.save_model('q_model_final.pkl')
    return agent, episode_rewards

def compare_agents(num_hands=10000):
    """
    Compare Q-learning agent vs card counting agent
    """
    from cardCounting import BlackjackEnv, BasicStrategyAgent
    
    # Test environments
    env_q = BlackjackEnv(num_decks=8, min_bet=50, max_bet=5000)
    env_cc = BlackjackEnv(num_decks=8, min_bet=50, max_bet=5000)
    
    # Agents
    q_agent = ImprovedQAgent()
    q_agent.load_model('q_model_final.pkl')  # Load trained model
    q_agent.set_training_mode(False)  # Set to evaluation mode
    
    cc_agent = BasicStrategyAgent()
    
    # Test Q-learning agent
    q_rewards = []
    for _ in range(num_hands):
        state = env_q.reset()
        done = False
        while not done:
            action = q_agent.get_action(state)
            state, reward, done = env_q.step(action)
        q_rewards.append(env_q.results_history[-1]['reward'])
    
    # Test card counting agent  
    cc_rewards = []
    for _ in range(num_hands):
        state = env_cc.reset()
        done = False
        while not done:
            action = cc_agent.get_action(state)
            state, reward, done = env_cc.step(action)
        cc_rewards.append(env_cc.results_history[-1]['reward'])
    
    # Results
    print(f"\n=== Performance Comparison ({num_hands} hands) ===")
    print(f"Q-Learning - Total: ${sum(q_rewards):.2f}, Avg: ${np.mean(q_rewards):.2f}")
    print(f"Card Counting - Total: ${sum(cc_rewards):.2f}, Avg: ${np.mean(cc_rewards):.2f}")
    
    return q_rewards, cc_rewards

# Example usage:
if __name__ == "__main__":
    # Train the agent
    trained_agent, rewards = train_q_agent(num_episodes=200000)
    
    # Compare performance
    q_rewards, cc_rewards = compare_agents(num_hands=5000)