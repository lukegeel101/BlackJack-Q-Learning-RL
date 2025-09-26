import numpy as np
import random
from collections import defaultdict, deque
import pickle
from tqdm import tqdm

# Define this function at module level so it can be pickled
def make_default_dict():
    return defaultdict(float)

class PickleCompatibleQAgent:
    """
    Q-Learning agent that can be pickled for multiprocessing
    """
    def __init__(self, learning_rate=0.1, discount_factor=0.95, 
                 epsilon_start=0.2, epsilon_end=0.001, epsilon_decay=0.99998):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Use regular dict instead of defaultdict with lambda
        self.Q = {}
        
        # Debug tracking
        self.debug_mode = True
        self.action_counts = defaultdict(int)
        self.q_value_history = deque(maxlen=1000)
        
        # Initialize with basic strategy
        self.initialize_basic_strategy()
        
        self.training_mode = True
    
    def _get_q_value(self, state_key, action):
        """
        Get Q-value with default initialization
        """
        if state_key not in self.Q:
            self.Q[state_key] = {}
        if action not in self.Q[state_key]:
            self.Q[state_key][action] = 0.0
        return self.Q[state_key][action]
    
    def _set_q_value(self, state_key, action, value):
        """
        Set Q-value with automatic initialization
        """
        if state_key not in self.Q:
            self.Q[state_key] = {}
        self.Q[state_key][action] = value
        
    def initialize_basic_strategy(self):
        """
        Comprehensive basic strategy initialization with properly scaled Q-values
        """
        # Hard totals strategy (hands without usable aces)
        hard_totals = {
            # Player total: {dealer_upcard: (action, q_value)}
            # Actions: 0=Hit, 1=Stand, 2=Double, 3=Split
            21: {d: (1, 10.0) for d in range(2, 12)},  # Always stand on 21
            20: {d: (1, 10.0) for d in range(2, 12)},  # Always stand on 20
            19: {d: (1, 9.0) for d in range(2, 12)},   # Always stand on 19
            18: {d: (1, 8.0) for d in range(2, 12)},   # Always stand on 18
            17: {d: (1, 7.0) for d in range(2, 12)},   # Always stand on 17
            
            16: {2: (1, 3.0), 3: (1, 3.0), 4: (1, 4.0), 5: (1, 4.0), 6: (1, 4.0),
                7: (0, 2.0), 8: (0, 1.0), 9: (0, 1.0), 10: (0, 1.0), 11: (0, 1.0)},
            
            15: {2: (1, 2.0), 3: (1, 3.0), 4: (1, 3.0), 5: (1, 3.0), 6: (1, 3.0),
                7: (0, 2.0), 8: (0, 1.0), 9: (0, 1.0), 10: (0, 1.0), 11: (0, 1.0)},
            
            14: {2: (1, 2.0), 3: (1, 2.0), 4: (1, 3.0), 5: (1, 3.0), 6: (1, 3.0),
                7: (0, 2.0), 8: (0, 1.0), 9: (0, 1.0), 10: (0, 1.0), 11: (0, 1.0)},
            
            13: {2: (1, 1.0), 3: (1, 2.0), 4: (1, 3.0), 5: (1, 3.0), 6: (1, 3.0),
                7: (0, 2.0), 8: (0, 1.0), 9: (0, 1.0), 10: (0, 1.0), 11: (0, 1.0)},
            
            12: {2: (0, 1.0), 3: (0, 1.0), 4: (1, 2.0), 5: (1, 3.0), 6: (1, 3.0),
                7: (0, 2.0), 8: (0, 1.0), 9: (0, 1.0), 10: (0, 1.0), 11: (0, 1.0)},
            
            11: {d: (2, 8.0) for d in range(2, 12)},  # Always double on 11
            
            10: {2: (2, 6.0), 3: (2, 6.0), 4: (2, 7.0), 5: (2, 7.0), 6: (2, 7.0),
                7: (2, 6.0), 8: (2, 6.0), 9: (2, 5.0), 10: (0, 3.0), 11: (0, 2.0)},
            
            9: {2: (0, 3.0), 3: (2, 4.0), 4: (2, 5.0), 5: (2, 5.0), 6: (2, 5.0),
                7: (0, 3.0), 8: (0, 2.0), 9: (0, 2.0), 10: (0, 2.0), 11: (0, 1.0)},
            
            8: {d: (0, 4.0) for d in range(2, 12)},   # Always hit on 8 or less
            7: {d: (0, 4.0) for d in range(2, 12)},
            6: {d: (0, 4.0) for d in range(2, 12)},
            5: {d: (0, 4.0) for d in range(2, 12)},
            4: {d: (0, 4.0) for d in range(2, 12)},
        }
        
        # Soft totals strategy (hands with usable ace)
        soft_totals = {
            21: {d: (1, 10.0) for d in range(2, 12)},  # Soft 21 (A,10) - blackjack
            20: {d: (1, 9.0) for d in range(2, 12)},   # Soft 20 (A,9)
            
            19: {2: (1, 7.0), 3: (1, 7.0), 4: (1, 7.0), 5: (1, 7.0), 6: (2, 6.0),
                7: (1, 7.0), 8: (1, 7.0), 9: (1, 7.0), 10: (1, 7.0), 11: (1, 7.0)},  # A,8
            
            18: {2: (2, 4.0), 3: (2, 5.0), 4: (2, 6.0), 5: (2, 6.0), 6: (2, 6.0),
                7: (1, 5.0), 8: (1, 4.0), 9: (0, 2.0), 10: (0, 1.0), 11: (0, 1.0)},  # A,7
            
            17: {2: (0, 3.0), 3: (2, 4.0), 4: (2, 5.0), 5: (2, 5.0), 6: (2, 5.0),
                7: (0, 3.0), 8: (0, 2.0), 9: (0, 2.0), 10: (0, 1.0), 11: (0, 1.0)},  # A,6
            
            16: {2: (0, 3.0), 3: (0, 3.0), 4: (2, 4.0), 5: (2, 5.0), 6: (2, 5.0),
                7: (0, 3.0), 8: (0, 2.0), 9: (0, 2.0), 10: (0, 1.0), 11: (0, 1.0)},  # A,5
            
            15: {2: (0, 3.0), 3: (0, 3.0), 4: (2, 4.0), 5: (2, 5.0), 6: (2, 5.0),
                7: (0, 3.0), 8: (0, 2.0), 9: (0, 2.0), 10: (0, 1.0), 11: (0, 1.0)},  # A,4
            
            14: {2: (0, 3.0), 3: (0, 3.0), 4: (0, 3.0), 5: (2, 4.0), 6: (2, 4.0),
                7: (0, 3.0), 8: (0, 2.0), 9: (0, 2.0), 10: (0, 1.0), 11: (0, 1.0)},  # A,3
            
            13: {2: (0, 3.0), 3: (0, 3.0), 4: (0, 3.0), 5: (2, 4.0), 6: (2, 4.0),
                7: (0, 3.0), 8: (0, 2.0), 9: (0, 2.0), 10: (0, 1.0), 11: (0, 1.0)},  # A,2
        }
        
        # Pair splitting strategy
        pair_splits = {
            # Pair value: {dealer_upcard: (action, q_value)}
            11: {d: (3, 9.0) for d in range(2, 12)},  # Always split Aces
            10: {d: (1, 8.0) for d in range(2, 12)},  # Never split 10s
            
            9: {2: (3, 6.0), 3: (3, 6.0), 4: (3, 6.0), 5: (3, 6.0), 6: (3, 6.0),
                7: (1, 5.0), 8: (3, 5.0), 9: (3, 5.0), 10: (1, 4.0), 11: (1, 4.0)},
            
            8: {d: (3, 7.0) for d in range(2, 12)},   # Always split 8s
            
            7: {2: (3, 5.0), 3: (3, 5.0), 4: (3, 5.0), 5: (3, 5.0), 6: (3, 5.0),
                7: (3, 5.0), 8: (0, 3.0), 9: (0, 2.0), 10: (0, 2.0), 11: (0, 2.0)},
            
            6: {2: (3, 4.0), 3: (3, 4.0), 4: (3, 5.0), 5: (3, 5.0), 6: (3, 5.0),
                7: (0, 3.0), 8: (0, 2.0), 9: (0, 2.0), 10: (0, 2.0), 11: (0, 2.0)},
            
            5: {2: (2, 6.0), 3: (2, 6.0), 4: (2, 7.0), 5: (2, 7.0), 6: (2, 7.0),
                7: (2, 6.0), 8: (2, 6.0), 9: (2, 5.0), 10: (0, 3.0), 11: (0, 2.0)},  # Treat like 10
            
            4: {2: (0, 3.0), 3: (0, 3.0), 4: (0, 3.0), 5: (3, 4.0), 6: (3, 4.0),
                7: (0, 3.0), 8: (0, 2.0), 9: (0, 2.0), 10: (0, 2.0), 11: (0, 2.0)},
            
            3: {2: (3, 4.0), 3: (3, 4.0), 4: (3, 5.0), 5: (3, 5.0), 6: (3, 5.0),
                7: (3, 4.0), 8: (0, 3.0), 9: (0, 2.0), 10: (0, 2.0), 11: (0, 2.0)},
            
            2: {2: (3, 4.0), 3: (3, 4.0), 4: (3, 5.0), 5: (3, 5.0), 6: (3, 5.0),
                7: (3, 4.0), 8: (0, 3.0), 9: (0, 2.0), 10: (0, 2.0), 11: (0, 2.0)},
        }
        
        # Initialize all hard totals
        for player_val, dealer_dict in hard_totals.items():
            for dealer_val, (action, q_val) in dealer_dict.items():
                for count in range(-10, 11):
                    state_key = self._get_state_key(player_val, dealer_val, False, False, count)
                    self._set_q_value(state_key, action, q_val)
        
        # Initialize all soft totals
        for player_val, dealer_dict in soft_totals.items():
            for dealer_val, (action, q_val) in dealer_dict.items():
                for count in range(-10, 11):
                    state_key = self._get_state_key(player_val, dealer_val, True, False, count)
                    self._set_q_value(state_key, action, q_val)
        
        # Initialize all pair splits
        for pair_val, dealer_dict in pair_splits.items():
            for dealer_val, (action, q_val) in dealer_dict.items():
                for count in range(-10, 11):
                    # Pairs can be soft (A,A) or hard (2,2 through 10,10)
                    is_soft = (pair_val == 11)  # Ace pairs are soft
                    state_key = self._get_state_key(pair_val * 2 if pair_val != 11 else 12, 
                                                dealer_val, is_soft, True, count)
                    self._set_q_value(state_key, action, q_val)
    
    def _get_state_key(self, player_value, dealer_upcard, is_soft, can_split, true_count):
        """
        No artificial binning - use actual game values
        """
        return (
            player_value,    # 4-21: Actual hand value
            dealer_upcard,   # 2-11: Actual dealer upcard  
            is_soft,         # True/False: Soft hand?
            can_split,       # True/False: Can split?
            min(10, max(-10, int(true_count)))  # Count: -10 to +10
        )
    
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
        valid_actions = [0, 1]  # Hit, Stand
        if len(state['player_hand'].cards) == 2:
            valid_actions.append(2)  # Double
        if state['can_split']:
            valid_actions.append(3)  # Split
            
        # Epsilon-greedy with vectorized Q-values
        if self.training_mode and random.random() < self.epsilon:
            action = random.choice(valid_actions)
        else:
            q_values = np.array([self._get_q_value(state_key, a) for a in valid_actions])
            
            # If Q-values are very similar, use basic strategy
            if np.std(q_values) < 0.1:  # Very similar values
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
        Q-learning update with proper reward scaling
        """
        if not self.training_mode:
            return
            
        # Scale reward
        scaled_reward = reward
        
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
            
            valid_next_actions = [0, 1]
            if len(next_state['player_hand'].cards) == 2:
                valid_next_actions.append(2)
            if next_state['can_split']:
                valid_next_actions.append(3)
                
            max_next_q = max([self._get_q_value(next_state_key, a) for a in valid_next_actions])
        else:
            max_next_q = 0
            
        # Q-learning update
        old_q = self._get_q_value(state_key, action)
        target = scaled_reward + self.gamma * max_next_q
        new_q = old_q + self.lr * (target - old_q)
        
        # Clip Q-values
        new_q = max(-5.0, min(5.0, new_q))
        self._set_q_value(state_key, action, new_q)
        
        if self.debug_mode:
            self.q_value_history.append(new_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def set_training_mode(self, training=True):
        self.training_mode = training
        if not training:
            self.epsilon = 0
    
    def save_model(self, filename):
        """
        Save the trained Q-table
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.Q, f)
    
    def load_model(self, filename):
        """
        Load a trained Q-table
        """
        with open(filename, 'rb') as f:
            self.Q = pickle.load(f)
    
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

def single_core_train(num_episodes):
    """
    Corrected single core training function 
    """
    try:
        from cardCounting import BlackjackEnv
    except ImportError:
        print("Error: Cannot import BlackjackEnv")
        return None, []
    
    env = BlackjackEnv(num_decks=8, min_bet=50, max_bet=5000)
    agent = PickleCompatibleQAgent(learning_rate=0.1)  # Higher learning rate
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        # Store only the FIRST state-action pair per episode
        first_state = state
        first_action = None
        
        while not done:
            action = agent.get_action(state)
            
            # Record first action only
            if first_action is None:
                first_action = action
            
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state
        
        # CORRECT: Single update per episode with final outcome
        # Scale reward to match Q-value range
        scaled_reward = episode_reward / 1000.0  # Scale to reasonable range
        agent.update(first_state, first_action, scaled_reward, None)
        
        episode_rewards.append(episode_reward)
    
    return agent, episode_rewards