import numpy as np
import random
from collections import defaultdict, deque
import pickle
from tqdm import tqdm

class EnhancedQAgent:
    """
    Enhanced Q-Learning agent with full card counting capabilities
    Tracks both true count and detailed shoe composition
    """
    def __init__(self, learning_rate=0.1, discount_factor=0.95, 
                 epsilon_start=0.5, epsilon_end=0.01, epsilon_decay=0.9999):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Use regular dict for pickling
        self.Q = {}
        
        # Card counting parameters
        self.hi_lo_values = {
            2: 1, 3: 1, 4: 1, 5: 1, 6: 1,  # Low cards: +1
            7: 0, 8: 0, 9: 0,                # Neutral: 0
            10: -1, 11: -1                   # High cards (10,J,Q,K,A): -1
        }
        
        # Tracking
        self.training_mode = True
        self.action_counts = defaultdict(int)
        self.q_value_history = deque(maxlen=1000)
        self.win_rate_history = deque(maxlen=10000)
        
        # Initialize with card-counting-aware basic strategy
        self.initialize_counting_strategy()
    
    def _get_shoe_composition_features(self, shoe_composition):
        """
        Extract key features from shoe composition for state representation
        Returns normalized features about remaining cards
        """
        if not shoe_composition:
            # Default for 8-deck shoe (416 cards total)
            total_cards = 416
            return (
                0.5,  # Ratio of high cards (10,A)
                0.5,  # Ratio of low cards (2-6)
                0.5,  # Ratio of aces
                1.0,  # Deck penetration (cards dealt / total)
                0.0,  # True count
                0.5,  # 10s ratio
            )
        
        total_cards = sum(shoe_composition.values())
        if total_cards == 0:
            return (0.5, 0.5, 0.5, 1.0, 0.0, 0.5)
        
        # Calculate key ratios
        high_cards = sum(shoe_composition.get(c, 0) for c in [10, 11])  # 10s and Aces
        low_cards = sum(shoe_composition.get(c, 0) for c in range(2, 7))  # 2-6
        aces = shoe_composition.get(11, 0)
        tens = shoe_composition.get(10, 0)
        
        # Normalize by remaining cards
        high_ratio = high_cards / max(1, total_cards)
        low_ratio = low_cards / max(1, total_cards)
        ace_ratio = aces / max(1, total_cards)
        tens_ratio = tens / max(1, total_cards)
        
        # Deck penetration (how deep into shoe)
        initial_cards = 416  # 8 decks
        penetration = 1.0 - (total_cards / initial_cards)
        
        # Calculate true count from composition
        running_count = 0
        for value, count in shoe_composition.items():
            expected_count = 32 if value != 10 else 128  # Expected in 8 decks
            actual_count = count
            if value in self.hi_lo_values:
                # More low cards remaining = negative count
                if value in [2, 3, 4, 5, 6]:
                    running_count -= (actual_count - expected_count * (total_cards/416))
                elif value in [10, 11]:
                    running_count += (expected_count * (total_cards/416) - actual_count)
        
        decks_remaining = total_cards / 52
        true_count = running_count / max(1, decks_remaining)
        
        return (
            round(high_ratio, 2),
            round(low_ratio, 2),
            round(ace_ratio, 3),  # More precision for aces
            round(penetration, 2),
            round(true_count / 10, 2),  # Normalize true count
            round(tens_ratio, 2)
        )
    
    def _num_cards_in_hand(self, hand):
        """Return number of cards in a hand whether it's a dict or a Hand object."""
        if hand is None:
            return 0
        # Case 1: dict-like
        if isinstance(hand, dict):
            return len(hand.get('cards', []))
        # Case 2: object with .cards (list) or .get_cards()
        if hasattr(hand, 'cards'):
            try:
                return len(hand.cards)
            except TypeError:
                pass
        if hasattr(hand, 'get_cards'):
            try:
                return len(hand.get_cards())
            except TypeError:
                pass
        # Fallback: try len()
        try:
            return len(hand)
        except TypeError:
            return 0

    def _get_enhanced_state_key(self, state):
        """
        Create state key that includes both basic info and shoe composition
        """
        # Basic state info
        player_value = state.get('player_value', 0)
        dealer_upcard = state.get('dealer_up_card', 0)
        is_soft = state.get('is_soft', False)
        can_split = state.get('can_split', False)
        true_count = state.get('true_count', 0)
        
        # Get shoe composition features
        shoe_features = self._get_shoe_composition_features(
            state.get('shoe_composition', {})
        )
        
        # Combine into comprehensive state
        return (
            player_value,           # 4-21
            dealer_upcard,         # 2-11
            is_soft,              # True/False
            can_split,            # True/False
            min(10, max(-10, int(true_count))),  # Clamped count
            shoe_features         # Tuple of shoe metrics
        )
    
    def initialize_counting_strategy(self):
        """
        Initialize Q-values with card-counting-aware strategy
        Adjusts basic strategy based on count
        """
        # Count-based adjustments to basic strategy
        # Format: (player_value, dealer_up, is_soft) -> {count_threshold: action}
        count_deviations = {
            # Insurance: Take at high counts
            'insurance': {3: True, -3: False},  # Take insurance at TC >= 3
            
            # Stand on 16 vs 10 at negative counts (more 10s coming)
            (16, 10, False): {0: 1, -1: 0},  # Stand at TC >= 0
            (16, 11, False): {0: 1, -1: 0},
            
            # Double 9 vs 2 at high counts
            (9, 2, False): {1: 2, -1: 0},  # Double at TC >= 1
            
            # Double 10 vs 10/A at high counts
            (10, 10, False): {4: 2, -1: 0},  # Double at TC >= 4
            (10, 11, False): {4: 2, -1: 0},
            
            # Stand on 15 vs 10 at high counts (dealer likely to bust)
            (15, 10, False): {4: 1, -1: 0},  # Stand at TC >= 4
            
            # Double soft 19 vs weak dealers at high counts
            (19, 5, True): {1: 2, -1: 1},   # Double at TC >= 1
            (19, 6, True): {1: 2, -1: 1},
            
            # Split 10s at very high counts (controversial but mathematically correct)
            (20, 5, False): {5: 3, -1: 1},  # Split at TC >= 5 vs 5
            (20, 6, False): {5: 3, -1: 1},  # Split at TC >= 5 vs 6
        }
        
        # Basic strategy with count adjustments
        for player_val in range(4, 22):
            for dealer_val in range(2, 12):
                for is_soft in [True, False]:
                    for can_split in [True, False]:
                        for count_cat in range(-10, 11):
                            # Create various shoe compositions
                            for high_ratio in [0.3, 0.5, 0.7]:
                                for low_ratio in [0.3, 0.5, 0.7]:
                                    for ace_ratio in [0.02, 0.025, 0.03]:
                                        
                                        shoe_features = (
                                            high_ratio,
                                            low_ratio, 
                                            ace_ratio,
                                            0.5,  # penetration
                                            count_cat / 10,  # normalized count
                                            high_ratio * 0.8  # tens ratio
                                        )
                                        
                                        state_key = (
                                            player_val,
                                            dealer_val,
                                            is_soft,
                                            can_split,
                                            count_cat,
                                            shoe_features
                                        )
                                        
                                        # Determine optimal action based on count
                                        action, q_value = self._get_count_adjusted_action(
                                            player_val, dealer_val, is_soft, can_split, count_cat,
                                            high_ratio, ace_ratio
                                        )
                                        
                                        self._set_q_value(state_key, action, q_value)
    
    def _get_count_adjusted_action(self, player_val, dealer_val, is_soft, 
                                   can_split, true_count, high_ratio, ace_ratio):
        """
        Get action adjusted for count and shoe composition
        """
        # High count favors player (more 10s gone, small cards remain)
        # Low count favors dealer (more 10s remain)
        
        # Basic strategy baseline
        if player_val >= 17:
            action = 1  # Stand
            q_value = 5.0
        elif player_val <= 11:
            if player_val == 11:
                action = 2  # Double
                q_value = 8.0 + true_count * 0.5  # Better at high counts
            else:
                action = 0  # Hit
                q_value = 3.0
        else:
            # 12-16: Count-dependent
            if dealer_val <= 6:  # Dealer bust cards
                if true_count >= 0:
                    action = 1  # Stand (dealer more likely to bust)
                    q_value = 4.0 + true_count * 0.3
                else:
                    action = 0  # Hit (dealer less likely to bust)
                    q_value = 2.0
            else:  # Dealer strong cards
                if true_count >= 3 and player_val == 16:
                    action = 1  # Stand on 16 vs 10 at high count
                    q_value = 1.0 + true_count * 0.2
                else:
                    action = 0  # Hit
                    q_value = 2.0
        
        # Soft hand adjustments
        if is_soft:
            if player_val >= 19:
                action = 1  # Stand
                q_value = 7.0
            elif player_val == 18:
                if dealer_val in [9, 10, 11]:
                    action = 0  # Hit soft 18 vs strong
                    q_value = 3.0
                else:
                    action = 1  # Stand
                    q_value = 5.0
            else:
                # Double soft hands vs weak dealers
                if dealer_val <= 6 and player_val >= 13:
                    action = 2  # Double
                    q_value = 6.0 + true_count * 0.4
                else:
                    action = 0  # Hit
                    q_value = 3.0
        
        # Pair splitting adjustments
        if can_split:
            pair_val = player_val // 2
            if pair_val == 11 or pair_val == 8:  # Always split Aces and 8s
                action = 3
                q_value = 9.0
            elif pair_val == 10:
                # Split 10s only at very high counts vs weak dealers
                if true_count >= 5 and dealer_val in [5, 6]:
                    action = 3
                    q_value = 6.0 + true_count * 0.3
                else:
                    action = 1  # Stand
                    q_value = 8.0
            elif pair_val == 9:
                if dealer_val not in [7, 10, 11]:
                    action = 3  # Split 9s
                    q_value = 6.0
                else:
                    action = 1
                    q_value = 5.0
        
        # Shoe composition adjustments
        if high_ratio > 0.6:  # Lots of high cards remaining
            # Favor doubling and standing
            if action == 0 and player_val >= 9 and player_val <= 11:
                action = 2  # Double more aggressively
                q_value += 1.0
        elif high_ratio < 0.4:  # Few high cards remaining
            # Avoid doubling
            if action == 2 and player_val < 11:
                action = 0  # Hit instead of double
                q_value -= 1.0
        
        # Ace ratio adjustments
        if ace_ratio > 0.03:  # Many aces remaining
            # Slightly favor hitting soft hands
            if is_soft and action == 1 and player_val <= 18:
                q_value -= 0.5
        
        return action, q_value
    
    def _set_q_value(self, state_key, action, value):
        """Set Q-value with automatic initialization"""
        if state_key not in self.Q:
            self.Q[state_key] = {}
        self.Q[state_key][action] = value
    
    def _get_q_value(self, state_key, action):
        """Get Q-value with default initialization"""
        if state_key not in self.Q:
            self.Q[state_key] = {}
        if action not in self.Q[state_key]:
            # Initialize with small random value to encourage exploration
            self.Q[state_key][action] = np.random.randn() * 0.01
        return self.Q[state_key][action]
    
    def get_action(self, state):
        """
        Get action using epsilon-greedy policy with count-based adjustments
        """
        if not state:
            return 1
        
        state_key = self._get_enhanced_state_key(state)
        
        # Get valid actions
        valid_actions = [0, 1]
        if self._num_cards_in_hand(state.get('player_hand')) == 2:
            valid_actions.append(2)
        if state.get('can_split', False):
            valid_actions.append(3)  # Split
        
        # Epsilon-greedy with exploration
        if self.training_mode and random.random() < self.epsilon:
            # Weighted random based on count
            true_count = state.get('true_count', 0)
            
            # At high counts, bias toward aggressive plays
            if true_count >= 3:
                weights = [0.2, 0.2, 0.4, 0.2]  # Favor doubling
            # At low counts, bias toward conservative plays  
            elif true_count <= -3:
                weights = [0.4, 0.4, 0.1, 0.1]  # Favor hit/stand
            else:
                weights = [0.25, 0.25, 0.25, 0.25]  # Equal
            
            # Adjust weights for valid actions
            action_weights = []
            for i, action in enumerate([0, 1, 2, 3]):
                if action in valid_actions:
                    action_weights.append(weights[i])
                else:
                    action_weights.append(0)
            
            # Normalize
            total_weight = sum(action_weights)
            if total_weight > 0:
                action_weights = [w/total_weight for w in action_weights]
                action = np.random.choice([0, 1, 2, 3], p=action_weights)
                if action not in valid_actions:
                    action = random.choice(valid_actions)
            else:
                action = random.choice(valid_actions)
        else:
            # Exploitation: Choose best Q-value
            q_values = {a: self._get_q_value(state_key, a) for a in valid_actions}
            
            # Add small noise to break ties
            for a in q_values:
                q_values[a] += np.random.randn() * 0.001
            
            action = max(q_values.keys(), key=lambda a: q_values[a])
            
            # Count-based overrides for critical decisions
            true_count = state.get('true_count', 0)
            player_value = state.get('player_value', 0)
            dealer_up = state.get('dealer_up_card', 0)
            
            # Insurance decision (if available)
            if 'insurance' in valid_actions and true_count >= 3:
                action = 'insurance'
            
            # Override: Always stand on 16 vs 10 at TC >= 0
            if player_value == 16 and dealer_up == 10 and true_count >= 0:
                if 1 in valid_actions:
                    action = 1
        
        self.action_counts[action] += 1
        return action
    
    def update(self, state, action, reward, next_state):
        """
        Q-learning update with enhanced state representation
        """
        if not self.training_mode:
            return
        
        state_key = self._get_enhanced_state_key(state)
        
        # Scale reward based on bet size and count
        true_count = state.get('true_count', 0)
        
        # Adjust reward scaling based on count (higher bets at high counts)
        if true_count >= 3:
            reward_scale = 1.5  # Emphasize wins at high counts
        elif true_count <= -3:
            reward_scale = 0.5  # De-emphasize at low counts
        else:
            reward_scale = 1.0
        
        scaled_reward = reward * reward_scale / 100.0  # Normalize
        
        if next_state:
            next_state_key = self._get_enhanced_state_key(next_state)
            
            # Get valid next actions
            valid_next_actions = [0, 1]
            if self._num_cards_in_hand(next_state.get('player_hand')) == 2:
                valid_next_actions.append(2)
            if next_state.get('can_split', False):
                valid_next_actions.append(3)
            
            # Get max Q-value for next state
            next_q_values = [self._get_q_value(next_state_key, a) 
                           for a in valid_next_actions]
            max_next_q = max(next_q_values) if next_q_values else 0
        else:
            max_next_q = 0
        
        # Q-learning update with momentum
        old_q = self._get_q_value(state_key, action)
        target = scaled_reward + self.gamma * max_next_q
        
        # Adaptive learning rate based on visit count
        if not hasattr(self, 'visit_counts'):
            self.visit_counts = defaultdict(int)
        
        self.visit_counts[state_key] += 1
        adaptive_lr = self.lr / (1 + self.visit_counts[state_key] * 0.01)
        
        new_q = old_q + adaptive_lr * (target - old_q)
        
        # Clip Q-values to reasonable range
        new_q = np.clip(new_q, -10.0, 10.0)
        
        self._set_q_value(state_key, action, new_q)
        
        # Track for debugging
        self.q_value_history.append(new_q)
        if reward > 0:
            self.win_rate_history.append(1)
        elif reward < 0:
            self.win_rate_history.append(0)
        else:
            self.win_rate_history.append(0.5)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def set_training_mode(self, training=True):
        """Set training mode"""
        self.training_mode = training
        if not training:
            self.epsilon = 0
    
    def save_model(self, filename):
        """Save the trained Q-table"""
        model_data = {
            'Q': self.Q,
            'epsilon': self.epsilon,
            'visit_counts': getattr(self, 'visit_counts', {})
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filename):
        """Load a trained Q-table"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
            
        # Handle both old and new format
        if isinstance(model_data, dict) and 'Q' in model_data:
            self.Q = model_data['Q']
            self.epsilon = model_data.get('epsilon', 0)
            self.visit_counts = model_data.get('visit_counts', defaultdict(int))
        else:
            # Old format - just Q table
            self.Q = model_data
            self.visit_counts = defaultdict(int)
    
    def get_debug_info(self):
        """Return debugging information"""
        win_rate = np.mean(self.win_rate_history) if self.win_rate_history else 0
        
        return {
            'action_distribution': dict(self.action_counts),
            'avg_q_value': np.mean(self.q_value_history) if self.q_value_history else 0,
            'q_value_std': np.std(self.q_value_history) if self.q_value_history else 0,
            'num_states': len(self.Q),
            'epsilon': self.epsilon,
            'win_rate': win_rate,
            'unique_states_visited': len(getattr(self, 'visit_counts', {}))
        }

# Backward compatibility
PickleCompatibleQAgent = EnhancedQAgent

def enhanced_single_core_train(episodes, core_id):
    """
    Training function with enhanced card counting
    """
    try:
        from cardCounting import BlackjackEnv
    except ImportError:
        print(f"Core {core_id}: Cannot import BlackjackEnv")
        return None, []
    
    env = BlackjackEnv(num_decks=8, min_bet=50, max_bet=5000)
    agent = EnhancedQAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=0.5,
        epsilon_end=0.01,
        epsilon_decay=0.99998
    )
    
    episode_rewards = []
    wins = losses = draws = 0
    
    print(f"Core {core_id}: Training {episodes:,} episodes with enhanced counting...")
    
    # Bet sizing based on count (Kelly Criterion approximation)
    def get_bet_size(true_count, min_bet=50, max_bet=5000):
        """Vary bet size based on true count"""
        if true_count <= -2:
            return min_bet  # Minimum bet at very negative counts
        elif true_count <= 0:
            return min_bet * 2
        elif true_count <= 2:
            return min_bet * 4
        elif true_count <= 4:
            return min_bet * 8
        else:
            return min(max_bet, min_bet * (2 ** true_count))
    
    print(f"Core {core_id}: Training {episodes:,} episodes with full card counting...")
    
    # Batch processing
    batch_size = 1000
    num_batches = episodes // batch_size
    
    for batch in range(num_batches):
        batch_rewards = []
        
        for episode_in_batch in range(batch_size):
            episode_num = batch * batch_size + episode_in_batch
            
            # Reset environment
            raw_state = env.reset()
            
            # Get shoe composition from environment
            shoe_comp = {}
            if hasattr(env, 'shoe'):
                # Get detailed shoe composition
                if hasattr(env.shoe, 'get_composition'):
                    shoe_comp = env.shoe.get_composition()
                elif hasattr(env.shoe, 'cards'):
                    # Count cards manually if needed
                    shoe_comp = defaultdict(int)
                    for card in env.shoe.cards:
                        if hasattr(card, 'value'):
                            shoe_comp[card.value] += 1
                        elif isinstance(card, int):
                            shoe_comp[card] += 1
            
            # Get true count from environment
            true_count = 0
            if hasattr(env, 'get_true_count'):
                true_count = env.get_true_count()
            elif 'true_count' in raw_state:
                true_count = raw_state['true_count']
            elif hasattr(env, 'running_count') and hasattr(env, 'cards_dealt'):
                # Calculate true count manually
                cards_dealt = getattr(env, 'cards_dealt', 0)
                running_count = getattr(env, 'running_count', 0)
                decks_remaining = (416 - cards_dealt) / 52  # 8 deck shoe
                true_count = running_count / max(1, decks_remaining)
            
            # Create enhanced state with all information
            state = {
                'player_value': raw_state.get('player_value', 0),
                'dealer_up_card': raw_state.get('dealer_up_card', 0),
                'is_soft': raw_state.get('is_soft', False),
                'can_split': raw_state.get('can_split', False),
                'true_count': true_count,
                'player_hand': raw_state.get('player_hand'),  # Pass the actual Hand object
                'shoe_composition': shoe_comp
            }
            
            # Adjust bet size based on count
            if hasattr(env, 'set_bet'):
                bet_size = get_bet_size(true_count)
                env.set_bet(bet_size)
            
            episode_reward = 0
            done = False
            transitions = []
            
            while not done:
                # Get action with full state information
                action = agent.get_action(state)
                
                # Step environment
                next_raw_state, reward, done = env.step(action)
                episode_reward += reward
                
                # Create next enhanced state
                if next_raw_state and not done:
                    # Update shoe composition
                    next_shoe_comp = {}
                    if hasattr(env, 'shoe'):
                        if hasattr(env.shoe, 'get_composition'):
                            next_shoe_comp = env.shoe.get_composition()
                        elif hasattr(env.shoe, 'cards'):
                            next_shoe_comp = defaultdict(int)
                            for card in env.shoe.cards:
                                if hasattr(card, 'value'):
                                    next_shoe_comp[card.value] += 1
                                elif isinstance(card, int):
                                    next_shoe_comp[card] += 1
                    
                    # Update true count
                    next_true_count = 0
                    if hasattr(env, 'get_true_count'):
                        next_true_count = env.get_true_count()
                    elif 'true_count' in next_raw_state:
                        next_true_count = next_raw_state['true_count']
                    
                    next_state = {
                        'player_value': next_raw_state.get('player_value', 0),
                        'dealer_up_card': next_raw_state.get('dealer_up_card', 0),
                        'is_soft': next_raw_state.get('is_soft', False),
                        'can_split': next_raw_state.get('can_split', False),
                        'true_count': next_true_count,
                        'player_hand': next_raw_state.get('player_hand'),  # Pass the actual Hand object
                        'shoe_composition': next_shoe_comp
                    }
                else:
                    next_state = None
                
                # Store transition for learning
                transitions.append((state, action, reward, next_state))
                
                # Update state
                if next_state:
                    state = next_state
            
            # Update Q-values with all transitions from episode
            for state, action, reward, next_state in transitions:
                agent.update(state, action, reward, next_state)
            
            # Track results
            if episode_reward > 0:
                wins += 1
            elif episode_reward < 0:
                losses += 1
            else:
                draws += 1
            
            batch_rewards.append(episode_reward)
            episode_rewards.append(episode_reward)
        
        # Batch statistics
        if batch % 100 == 0:
            win_rate = wins / max(1, wins + losses + draws)
            recent_avg = np.mean(batch_rewards) if batch_rewards else 0
            debug_info = agent.get_debug_info()
            
            print(f"Core {core_id} - Batch {batch}/{num_batches}: "
                  f"Episodes: {(batch+1)*batch_size:,}, "
                  f"Win rate: {win_rate:.2%}, "
                  f"Recent avg: ${recent_avg:.2f}, "
                  f"States: {debug_info['num_states']:,}, "
                  f"ε: {debug_info['epsilon']:.4f}, "
                  f"Unique visited: {debug_info.get('unique_states_visited', 0):,}")
    
    # Train remaining episodes
    remaining = episodes % batch_size
    for _ in range(remaining):
        # Same training logic as above
        raw_state = env.reset()
        
        # Get full state information
        shoe_comp = {}
        if hasattr(env, 'shoe') and hasattr(env.shoe, 'get_composition'):
            shoe_comp = env.shoe.get_composition()
        
        true_count = raw_state.get('true_count', 0)
        
        state = {
            'player_value': raw_state.get('player_value', 0),
            'dealer_up_card': raw_state.get('dealer_up_card', 0),
            'is_soft': raw_state.get('is_soft', False),
            'can_split': raw_state.get('can_split', False),
            'true_count': true_count,
            'player_hand': raw_state.get('player_hand'),  # Pass the actual Hand object
            'shoe_composition': shoe_comp
        }
        
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.get_action(state)
            next_raw_state, reward, done = env.step(action)
            episode_reward += reward
            
            if next_raw_state and not done:
                next_shoe_comp = {}
                if hasattr(env, 'shoe') and hasattr(env.shoe, 'get_composition'):
                    next_shoe_comp = env.shoe.get_composition()
                
                next_state = {
                    'player_value': next_raw_state.get('player_value', 0),
                    'dealer_up_card': next_raw_state.get('dealer_up_card', 0),
                    'is_soft': next_raw_state.get('is_soft', False),
                    'can_split': next_raw_state.get('can_split', False),
                    'true_count': next_raw_state.get('true_count', 0),
                    'player_hand': next_raw_state.get('player_hand'),  # Pass the actual Hand object
                    'shoe_composition': next_shoe_comp
                }
            else:
                next_state = None
            
            agent.update(state, action, reward, next_state)
            
            if next_state:
                state = next_state
        
        episode_rewards.append(episode_reward)
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        else:
            draws += 1
    
    # Set to evaluation mode
    agent.set_training_mode(False)
    
    # Final statistics
    final_win_rate = wins / max(1, wins + losses + draws)
    final_avg = np.mean(episode_rewards) if episode_rewards else 0
    
    print(f"Core {core_id} completed: "
          f"Episodes: {episodes:,}, "
          f"Win rate: {final_win_rate:.2%}, "
          f"Avg reward: ${final_avg:.2f}, "
          f"Q-table size: {len(agent.Q):,}")
    
    return agent, episode_rewards

# For backward compatibility
single_core_train = enhanced_single_core_train