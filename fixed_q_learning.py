import numpy as np
import random
from collections import defaultdict, deque
import pickle
from tqdm import tqdm
import multiprocessing as mp

# Define this function at module level so it can be pickled
def make_default_dict():
    return defaultdict(float)

class FixedQAgent:
    """
    Fixed Q-Learning implementation with proper debugging and optimizations
    """
    def __init__(self, learning_rate=0.05, discount_factor=0.9, 
                 epsilon_start=0.1, epsilon_end=0.001, epsilon_decay=0.9999):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Use smaller state space
        self.Q = defaultdict(lambda: defaultdict(float))
        
        # Debug tracking
        self.debug_mode = True
        self.action_counts = defaultdict(int)
        self.q_value_history = deque(maxlen=1000)
        
        # Initialize with basic strategy
        self.initialize_basic_strategy()
        
        self.training_mode = True
        
    def initialize_basic_strategy(self):
        """
        Initialize Q-values with basic strategy knowledge
        """
        # Simplified basic strategy rules with proper rewards
        rules = [
            # Stand on high values
            (21, 10, False, False, 0, 1, 1.0),   # Always stand on 21
            (20, 10, False, False, 0, 1, 1.0),   # Always stand on 20
            (19, 10, False, False, 0, 1, 0.8),   # Stand on 19
            
            # Hit on low values  
            (8, 10, False, False, 0, 0, 0.5),    # Always hit on 8
            (9, 10, False, False, 0, 0, 0.5),    # Hit on 9 vs 10
            (10, 10, False, False, 0, 0, 0.3),   # Hit on 10 vs 10 (usually)
            
            # Double on 11
            (11, 6, False, False, 0, 2, 0.8),    # Double 11 vs 6
            (11, 5, False, False, 0, 2, 0.8),    # Double 11 vs 5
            
            # Stand on stiffs vs dealer bust cards
            (12, 6, False, False, 0, 1, 0.4),    # Stand 12 vs 6
            (13, 6, False, False, 0, 1, 0.6),    # Stand 13 vs 6
            (14, 6, False, False, 0, 1, 0.6),    # Stand 14 vs 6
            (15, 6, False, False, 0, 1, 0.6),    # Stand 15 vs 6
            (16, 6, False, False, 0, 1, 0.6),    # Stand 16 vs 6
            
            # Hit stiffs vs dealer strong cards
            (12, 10, False, False, 0, 0, 0.2),   # Hit 12 vs 10
            (13, 10, False, False, 0, 0, 0.1),   # Hit 13 vs 10
            (14, 10, False, False, 0, 0, 0.1),   # Hit 14 vs 10
            (15, 10, False, False, 0, 0, 0.1),   # Hit 15 vs 10
            (16, 10, False, False, 0, 0, 0.1),   # Hit 16 vs 10
        ]
        
        for player_val, dealer_val, is_soft, can_split, count, action, q_val in rules:
            state_key = self._get_state_key(player_val, dealer_val, is_soft, can_split, count)
            self.Q[state_key][action] = q_val
    
    def _get_state_key(self, player_value, dealer_upcard, is_soft, can_split, true_count):
        """
        Ultra-simplified state representation to reduce complexity
        """
        # Even more aggressive binning
        if player_value <= 11:
            player_bin = 11
        elif player_value <= 16:
            player_bin = 16  
        elif player_value <= 17:
            player_bin = 17
        else:
            player_bin = 20  # 18-21
            
        # Bin dealer upcard
        if dealer_upcard <= 6:
            dealer_bin = 6   # Bust cards
        elif dealer_upcard <= 9:
            dealer_bin = 9   # Medium cards
        else:
            dealer_bin = 10  # Strong cards (10, A)
            
        # Simplified count (just negative, neutral, positive)
        if true_count <= -1:
            count_bin = -1
        elif true_count >= 1:
            count_bin = 1
        else:
            count_bin = 0
            
        return (player_bin, dealer_bin, is_soft, can_split, count_bin)
    
    def get_action(self, state):
        """
        Get action with better exploration strategy
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
        valid_actions = [0, 1]  # Hit, Stand
        if len(state['player_hand'].cards) == 2:
            valid_actions.append(2)  # Double
        if state['can_split']:
            valid_actions.append(3)  # Split
            
        # Epsilon-greedy with valid action restriction
        if self.training_mode and random.random() < self.epsilon:
            action = random.choice(valid_actions)
        else:
            # Vectorized Q-value lookup
            q_values = np.array([self.Q[state_key][a] for a in valid_actions])
            
            if np.all(q_values == q_values[0]):  # All equal, use basic strategy
                if state['player_value'] <= 11:
                    action = 0  # Hit
                elif state['player_value'] >= 17:
                    action = 1  # Stand
                elif state['dealer_up_card'] <= 6:
                    action = 1  # Stand vs bust cards
                else:
                    action = 0  # Hit vs strong cards
            else:
                best_action_idx = np.argmax(q_values)
                action = valid_actions[best_action_idx]
        
        if self.debug_mode:
            self.action_counts[action] += 1
            
        return action
    
    def update(self, state, action, reward, next_state):
        """
        Fixed Q-learning update with proper reward scaling
        """
        if not self.training_mode:
            return
            
        # Scale reward to reasonable range
        scaled_reward = reward / 100.0  # Scale $100 to 1.0
        
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
            
            # Get valid next actions
            valid_next_actions = [0, 1]
            if len(next_state['player_hand'].cards) == 2:
                valid_next_actions.append(2)
            if next_state['can_split']:
                valid_next_actions.append(3)
                
            max_next_q = max([self.Q[next_state_key][a] for a in valid_next_actions])
        else:
            max_next_q = 0
            
        # Q-learning update with clipping
        old_q = self.Q[state_key][action]
        target = scaled_reward + self.gamma * max_next_q
        self.Q[state_key][action] = old_q + self.lr * (target - old_q)
        
        # Clip Q-values to reasonable range
        self.Q[state_key][action] = max(-5.0, min(5.0, self.Q[state_key][action]))
        
        if self.debug_mode:
            self.q_value_history.append(self.Q[state_key][action])
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def set_training_mode(self, training=True):
        self.training_mode = training
        if not training:
            self.epsilon = 0
    
    def get_debug_info(self):
        """
        Return debugging information
        """
        if self.q_value_history:
            avg_q = np.mean(self.q_value_history)
            q_std = np.std(self.q_value_history)
        else:
            avg_q = q_std = 0
            
        return {
            'action_distribution': dict(self.action_counts),
            'avg_q_value': avg_q,
            'q_value_std': q_std,
            'num_states': len(self.Q),
            'epsilon': self.epsilon
        }

def fast_train_q_agent(num_episodes=1000000, eval_interval=50000):
    """
    Optimized training with debugging and evaluation
    """
    try:
        from cardCounting import BlackjackEnv
    except ImportError:
        print("Error: Cannot import BlackjackEnv from cardCounting")
        return None, []
    
    env = BlackjackEnv(num_decks=8, min_bet=50, max_bet=5000)
    agent = FixedQAgent()
    
    episode_rewards = []
    eval_scores = []
    
    print(f"Training Q-Learning Agent for {num_episodes:,} episodes...")
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        episode_reward = 0
        done = False
        
        prev_state = None
        prev_action = None
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            
            # Update previous state-action pair if it exists
            if prev_state is not None:
                agent.update(prev_state, prev_action, 0, state)  # Intermediate reward = 0
            
            # Store current state-action for next update
            prev_state = state
            prev_action = action
            episode_reward += reward
            state = next_state
        
        # Final update with actual reward
        if prev_state is not None:
            agent.update(prev_state, prev_action, episode_reward, None)
        
        episode_rewards.append(episode_reward)
        
        # Evaluation and progress
        if episode % eval_interval == 0 and episode > 0:
            # Quick evaluation
            agent.set_training_mode(False)
            eval_rewards = []
            for _ in range(1000):  # Quick eval
                eval_state = env.reset()
                eval_done = False
                while not eval_done:
                    eval_action = agent.get_action(eval_state)
                    eval_state, eval_reward, eval_done = env.step(eval_action)
                eval_rewards.append(env.results_history[-1]['reward'])
            
            eval_avg = np.mean(eval_rewards)
            eval_scores.append(eval_avg)
            
            # Debug info
            debug_info = agent.get_debug_info()
            
            print(f"\nEpisode {episode:,}")
            print(f"Training avg (last 1000): ${np.mean(episode_rewards[-1000:]):.2f}")
            print(f"Evaluation avg: ${eval_avg:.2f}")
            print(f"Epsilon: {agent.epsilon:.4f}")
            print(f"States learned: {debug_info['num_states']:,}")
            print(f"Action distribution: {debug_info['action_distribution']}")
            print(f"Avg Q-value: {debug_info['avg_q_value']:.3f}")
            
            agent.set_training_mode(True)
    
    return agent, episode_rewards, eval_scores

def compare_optimized_agents(num_hands=5000):
    """
    Compare with proper evaluation mode
    """
    try:
        from cardCounting import BlackjackEnv, BasicStrategyAgent
    except ImportError:
        print("Error: Cannot import from cardCounting")
        return [], []
    
    print(f"\nComparing agents over {num_hands} hands...")
    
    # Q-learning evaluation
    env_q = BlackjackEnv(num_decks=8, min_bet=50, max_bet=5000)
    q_agent = FixedQAgent()
    
    try:
        q_agent.load_model('q_model_final.pkl')
        print("Loaded trained Q-model")
    except:
        print("No trained model found, using randomly initialized agent")
    
    q_agent.set_training_mode(False)  # CRITICAL: No exploration
    
    q_rewards = []
    for _ in tqdm(range(num_hands), desc="Testing Q-Learning"):
        state = env_q.reset()
        done = False
        while not done:
            action = q_agent.get_action(state)
            state, reward, done = env_q.step(action)
        q_rewards.append(env_q.results_history[-1]['reward'])
    
    # Card counting evaluation
    env_cc = BlackjackEnv(num_decks=8, min_bet=50, max_bet=5000)
    cc_agent = BasicStrategyAgent()
    
    cc_rewards = []
    for _ in tqdm(range(num_hands), desc="Testing Card Counting"):
        state = env_cc.reset()
        done = False
        while not done:
            action = cc_agent.get_action(state)
            state, reward, done = env_cc.step(action)
        cc_rewards.append(env_cc.results_history[-1]['reward'])
    
    # Results
    print(f"\n=== Performance Comparison ({num_hands:,} hands) ===")
    print(f"Q-Learning - Total: ${sum(q_rewards):,.2f}, Avg: ${np.mean(q_rewards):.2f}")
    print(f"Card Counting - Total: ${sum(cc_rewards):,.2f}, Avg: ${np.mean(cc_rewards):.2f}")
    print(f"Difference: ${np.mean(q_rewards) - np.mean(cc_rewards):.2f} per hand")
    
    return q_rewards, cc_rewards

# Example usage for M2 MacBook optimization
if __name__ == "__main__":
    # Train with reasonable episode count
    print("Starting optimized Q-Learning training...")
    agent, rewards, eval_scores = fast_train_q_agent(num_episodes=2000000)
    
    if agent:
        agent.save_model('q_model_optimized.pkl')
        
        # Compare performance
        q_rewards, cc_rewards = compare_optimized_agents(num_hands=10000)