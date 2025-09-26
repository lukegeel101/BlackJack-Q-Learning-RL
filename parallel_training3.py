import multiprocessing as mp
import numpy as np
#from pickle_compatible_q_learning import single_core_train, PickleCompatibleQAgent
from enhanced_q_learning_agent import EnhancedQAgent as PickleCompatibleQAgent
from enhanced_q_learning_agent import enhanced_single_core_train as single_core_train
from functools import partial
import pickle
from collections import defaultdict, deque
import os
import cProfile
import time
import csv

def _extract_true_count(env, state, default_decks=8):
    """
    Try to obtain the true count from env or state and return an int bucket.
    Falls back to computing TC from running_count/cards_dealt if needed.
    """
    tc = None

    # Preferred: environment API
    if hasattr(env, 'get_true_count'):
        try:
            tc = env.get_true_count()
        except Exception:
            tc = None

    # Next: state dict
    if tc is None and isinstance(state, dict) and ('true_count' in state):
        tc = state['true_count']

    # Fallback: compute from running_count / decks_remaining if available
    if tc is None and hasattr(env, 'running_count') and hasattr(env, 'cards_dealt'):
        cards_dealt = getattr(env, 'cards_dealt', 0)
        total_cards = default_decks * 52
        decks_remaining = max(1e-9, (total_cards - cards_dealt) / 52.0)
        tc = getattr(env, 'running_count', 0) / decks_remaining

    if tc is None:
        tc = 0.0

    # Bucket to integer and clamp to a reasonable range for histograms
    tc_int = int(round(tc))
    return max(-20, min(20, tc_int))

def _write_count_histogram_to_csv(counter_dict, filename):
    """
    Write 'count, hands' CSV sorted by count.
    """
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['count', 'hands'])
            for count in sorted(counter_dict.keys()):
                writer.writerow([count, counter_dict[count]])
        print(f"Saved count histogram to {filename}")
    except Exception as e:
        print(f"Failed to write {filename}: {e}")

# Set environment variables for M2 optimization
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'  # Prevent thread oversubscription
os.environ['OMP_NUM_THREADS'] = '1'

def compare_parallel_agents(num_hands=10000000):
    """
    Compare merged agent vs card counting
    """
    try:
        from cardCounting import BlackjackEnv, BasicStrategyAgent
    except ImportError:
        print("Error: Cannot import from cardCounting")
        return [], []
    
    print(f"\nComparing agents over {num_hands} hands...")

    q_count_hist = defaultdict(int)
    cc_count_hist = defaultdict(int)
    
    # Load merged Q-agent
    env_q = BlackjackEnv(num_decks=8, min_bet=50, max_bet=5000)
    q_agent = PickleCompatibleQAgent()
    
    try:
        q_agent.load_model('q_model_parallel.pkl')
        print("Loaded parallel-trained Q-model")
    except Exception as e:
        print(f"No parallel model found: {e}")
        return [], []
    
    q_agent.set_training_mode(False)  # No exploration
    
    # Test Q-learning
    q_rewards = []
    for i in range(num_hands):
        state = env_q.reset()
        # Record starting-hand true count bucket for histogram
        q_tc_bucket = _extract_true_count(env_q, state, default_decks=8)
        q_count_hist[q_tc_bucket] += 1
        done = False
        episode_reward = 0
        
        while not done:
            action = q_agent.get_action(state)
            state, reward, done = env_q.step(action)
            episode_reward += reward
        
        q_rewards.append(episode_reward)
        
        # Progress indicator
        if (i + 1) % 10000 == 0:
            print(f"Q-Learning progress: {i+1}/{num_hands} hands")
    
    # Test card counting
    env_cc = BlackjackEnv(num_decks=8, min_bet=50, max_bet=5000)
    cc_agent = BasicStrategyAgent()
    
    cc_rewards = []
    for i in range(num_hands):
        state = env_cc.reset()
        cc_tc_bucket = _extract_true_count(env_cc, state, default_decks=8)
        cc_count_hist[cc_tc_bucket] += 1
        done = False
        episode_reward = 0
        
        while not done:
            action = cc_agent.get_action(state)
            state, reward, done = env_cc.step(action)
            episode_reward += reward
        
        cc_rewards.append(episode_reward)
        
        # Progress indicator
        if (i + 1) % 10000 == 0:
            print(f"Card Counting progress: {i+1}/{num_hands} hands")
    
    # Results
    print(f"\n=== Performance Comparison ({num_hands:,} hands) ===")
    print(f"Q-Learning - Total: ${sum(q_rewards):,.2f}, Avg: ${np.mean(q_rewards):.2f}")
    print(f"Card Counting - Total: ${sum(cc_rewards):,.2f}, Avg: ${np.mean(cc_rewards):.2f}")
    print(f"Difference: ${np.mean(q_rewards) - np.mean(cc_rewards):.2f} per hand")

    _write_count_histogram_to_csv(q_count_hist,  '/Users/lukegeel/Desktop/blackjack project/q_hand_counts.csv')
    _write_count_histogram_to_csv(cc_count_hist, '/Users/lukegeel/Desktop/blackjack project/cc_hand_counts.csv')
    
    # IMPORTANT: Actually return the rewards!
    return q_rewards, cc_rewards

def optimized_single_core_train(episodes, core_id, shared_dict=None):
    """
    Optimized training function with batch processing and performance enhancements
    """
    # Import here to avoid multiprocessing issues
    try:
        from pickle_compatible_q_learning import PickleCompatibleQAgent
        from cardCounting import BlackjackEnv
    except ImportError:
        # Fallback if imports fail - define minimal versions
        print(f"Core {core_id}: Using fallback implementations")
        return None, None
    
    # Use process-specific random seed for diversity
    np.random.seed(core_id * 12345 + int(time.time()))
    
    # Initialize environment and agent
    env = BlackjackEnv(num_decks=8, min_bet=50, max_bet=5000)
    agent = PickleCompatibleQAgent()
    
    # Performance optimizations
    reward_history = deque(maxlen=10000)  # Use deque for memory efficiency
    
    # Cache for frequently accessed states (LRU-like)
    state_cache = {}
    max_cache_size = 50000
    
    # Batch processing parameters
    batch_size = 1000
    num_batches = episodes // batch_size
    
    # Training hyperparameters with decay schedules
    initial_epsilon = 1.0
    final_epsilon = 0.01
    epsilon_decay = (initial_epsilon - final_epsilon) / episodes
    
    initial_alpha = 0.15
    final_alpha = 0.01
    alpha_decay = (initial_alpha - final_alpha) / episodes
    
    gamma = 0.95  # Discount factor
    
    # Statistics tracking
    total_episodes = 0
    wins = 0
    losses = 0
    draws = 0
    
    # Batch-level experience replay buffer
    experience_buffer = []
    replay_frequency = 100  # Replay every N episodes
    replay_batch_size = 32
    
    print(f"Core {core_id}: Starting training with {episodes:,} episodes")
    
    for batch in range(num_batches):
        batch_rewards = []
        
        # Process batch of episodes
        for episode_in_batch in range(batch_size):
            total_episodes += 1
            
            # Calculate current hyperparameters
            current_epsilon = max(final_epsilon, 
                                 initial_epsilon - epsilon_decay * total_episodes)
            current_alpha = max(final_alpha,
                               initial_alpha - alpha_decay * total_episodes)
            
            # Set exploration rate
            agent.epsilon = current_epsilon
            agent.alpha = current_alpha
            
            # Reset environment
            state = env.reset()
            
            # Episode trajectory for experience replay
            episode_trajectory = []
            episode_reward = 0
            done = False
            
            # Run episode
            while not done:
                # Cache state encoding for faster lookup
                state_key = state
                if isinstance(state, (list, np.ndarray)):
                    state_key = tuple(state)
                
                # Get action using epsilon-greedy
                if np.random.random() < current_epsilon:
                    action = np.random.choice([0, 1, 2, 3])  # Random action
                else:
                    # Get Q-values for state
                    if state_key in agent.Q:
                        q_values = agent.Q[state_key]
                        # Choose action with highest Q-value
                        valid_actions = list(q_values.keys())
                        if valid_actions:
                            action = max(valid_actions, key=lambda a: q_values[a])
                        else:
                            action = np.random.choice([0, 1, 2, 3])
                    else:
                        action = np.random.choice([0, 1, 2, 3])
                
                # Take action
                next_state, reward, done = env.step(action)
                episode_reward += reward
                
                # Store transition
                episode_trajectory.append((state_key, action, reward, next_state, done))
                
                # Online Q-value update
                if not done:
                    next_state_key = next_state
                    if isinstance(next_state, (list, np.ndarray)):
                        next_state_key = tuple(next_state)
                    
                    # Get max Q-value for next state
                    max_next_q = 0
                    if next_state_key in agent.Q:
                        next_q_values = agent.Q[next_state_key]
                        if next_q_values:
                            max_next_q = max(next_q_values.values())
                    
                    # Q-learning update
                    old_q = agent.get_q_value(state_key, action)
                    new_q = old_q + current_alpha * (reward + gamma * max_next_q - old_q)
                    agent._set_q_value(state_key, action, new_q)
                else:
                    # Terminal state update
                    old_q = agent.get_q_value(state_key, action)
                    new_q = old_q + current_alpha * (reward - old_q)
                    agent._set_q_value(state_key, action, new_q)
                
                state = next_state
            
            # Track game outcome
            if episode_reward > 0:
                wins += 1
            elif episode_reward < 0:
                losses += 1
            else:
                draws += 1
            
            # Add to experience buffer for replay
            experience_buffer.extend(episode_trajectory)
            if len(experience_buffer) > 100000:  # Limit buffer size
                experience_buffer = experience_buffer[-50000:]
            
            # Store reward
            batch_rewards.append(episode_reward)
            reward_history.append(episode_reward)
            
            # Experience replay every N episodes
            if total_episodes % replay_frequency == 0 and len(experience_buffer) >= replay_batch_size:
                # Sample random batch from experience buffer
                replay_indices = np.random.choice(len(experience_buffer), 
                                                size=min(replay_batch_size, len(experience_buffer)),
                                                replace=False)
                
                for idx in replay_indices:
                    s, a, r, s_next, is_done = experience_buffer[idx]
                    
                    if not is_done:
                        # Non-terminal state
                        next_state_key = s_next
                        if isinstance(s_next, (list, np.ndarray)):
                            next_state_key = tuple(s_next)
                        
                        max_next_q = 0
                        if next_state_key in agent.Q:
                            next_q_values = agent.Q[next_state_key]
                            if next_q_values:
                                max_next_q = max(next_q_values.values())
                        
                        old_q = agent.get_q_value(s, a)
                        new_q = old_q + current_alpha * (r + gamma * max_next_q - old_q)
                        agent._set_q_value(s, a, new_q)
                    else:
                        # Terminal state
                        old_q = agent.get_q_value(s, a)
                        new_q = old_q + current_alpha * (r - old_q)
                        agent._set_q_value(s, a, new_q)
            
            # Periodic cache cleanup
            if len(state_cache) > max_cache_size:
                # Remove oldest half of cache
                keys = list(state_cache.keys())
                for key in keys[:len(keys)//2]:
                    del state_cache[key]
        
        # Batch statistics
        if batch % 100 == 0:  # Log every 100 batches
            avg_reward = np.mean(batch_rewards) if batch_rewards else 0
            win_rate = wins / max(1, wins + losses + draws)
            recent_avg = np.mean(list(reward_history)) if reward_history else 0
            
            print(f"Core {core_id} - Batch {batch}/{num_batches}: "
                  f"Episodes: {total_episodes:,}, "
                  f"Win rate: {win_rate:.2%}, "
                  f"Avg reward: ${avg_reward:.2f}, "
                  f"Recent avg: ${recent_avg:.2f}, "
                  f"ε: {current_epsilon:.3f}, "
                  f"α: {current_alpha:.3f}, "
                  f"Q-states: {len(agent.Q):,}")
    
    # Train remaining episodes
    remaining = episodes % batch_size
    if remaining > 0:
        for _ in range(remaining):
            total_episodes += 1
            
            # Calculate current hyperparameters
            current_epsilon = max(final_epsilon,
                                 initial_epsilon - epsilon_decay * total_episodes)
            current_alpha = max(final_alpha,
                               initial_alpha - alpha_decay * total_episodes)
            
            agent.epsilon = current_epsilon
            agent.alpha = current_alpha
            
            # Run episode (same logic as above, simplified)
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
                
                # Get action
                if np.random.random() < current_epsilon:
                    action = np.random.choice([0, 1, 2, 3])
                else:
                    if state_key in agent.Q:
                        q_values = agent.Q[state_key]
                        valid_actions = list(q_values.keys())
                        if valid_actions:
                            action = max(valid_actions, key=lambda a: q_values[a])
                        else:
                            action = np.random.choice([0, 1, 2, 3])
                    else:
                        action = np.random.choice([0, 1, 2, 3])
                
                next_state, reward, done = env.step(action)
                episode_reward += reward
                
                # Q-update
                if not done:
                    next_state_key = tuple(next_state) if isinstance(next_state, (list, np.ndarray)) else next_state
                    max_next_q = 0
                    if next_state_key in agent.Q:
                        next_q_values = agent.Q[next_state_key]
                        if next_q_values:
                            max_next_q = max(next_q_values.values())
                    
                    old_q = agent.get_q_value(state_key, action)
                    new_q = old_q + current_alpha * (reward + gamma * max_next_q - old_q)
                    agent._set_q_value(state_key, action, new_q)
                else:
                    old_q = agent.get_q_value(state_key, action)
                    new_q = old_q + current_alpha * (reward - old_q)
                    agent._set_q_value(state_key, action, new_q)
                
                state = next_state
            
            reward_history.append(episode_reward)
            if episode_reward > 0:
                wins += 1
            elif episode_reward < 0:
                losses += 1
            else:
                draws += 1
    
    # Final statistics
    final_win_rate = wins / max(1, wins + losses + draws)
    final_avg_reward = np.mean(list(reward_history)) if reward_history else 0
    
    print(f"Core {core_id} completed: "
          f"Total episodes: {total_episodes:,}, "
          f"Final win rate: {final_win_rate:.2%}, "
          f"Final avg reward: ${final_avg_reward:.2f}, "
          f"Q-table size: {len(agent.Q):,} states")
    
    # Set agent to evaluation mode (no exploration)
    agent.epsilon = 0.0
    agent.set_training_mode(False)
    
    return agent, list(reward_history)

def parallel_train_episodes_optimized(episodes_per_core=32000000):
    import multiprocessing as mp
    from functools import partial

    cores = min(mp.cpu_count() - 2, 6)
    print(f"Using {cores} cores for parallel training")
    print(f"Total episodes: {cores * episodes_per_core:,}")

    # Important on macOS: do this under __main__ guard in the caller file
    mp.set_start_method('spawn', force=True)

    with mp.Pool(cores, maxtasksperchild=1) as pool:
        # Build argument tuples: (episodes, core_id)
        args = [(episodes_per_core, i) for i in range(cores)]
        # Send both args so each worker gets a unique core_id
        results = pool.starmap(single_core_train, args)

    valid_results = [r for r in results if r and r[0] is not None]
    print(f"Successfully trained {len(valid_results)} agents")
    return valid_results

def vectorized_merge_q_tables(results):
    """
    Optimized Q-table merging using NumPy vectorization
    """
    if not results:
        print("No valid results to merge")
        return None
    
    agents, _ = zip(*results)
    merged_agent = PickleCompatibleQAgent()
    
    # Collect all unique states efficiently
    state_action_values = defaultdict(lambda: defaultdict(list))
    
    for agent in agents:
        for state, actions in agent.Q.items():
            for action, value in actions.items():
                state_action_values[state][action].append(value)
    
    print(f"Merging {len(state_action_values)} unique states from {len(agents)} agents")
    
    # Vectorized averaging using NumPy
    for state, actions in state_action_values.items():
        for action, values in actions.items():
            # Use NumPy for faster averaging
            merged_agent._set_q_value(state, action, np.mean(values))
    
    return merged_agent

def parallel_merge_q_tables(results, num_workers=4):
    """
    Parallel merging of Q-tables for very large state spaces
    """
    if not results:
        return None
    
    agents, _ = zip(*results)
    
    # Collect all states
    all_states = set()
    for agent in agents:
        all_states.update(agent.Q.keys())
    
    # Split states for parallel processing
    states_list = list(all_states)
    chunk_size = len(states_list) // num_workers + 1
    state_chunks = [states_list[i:i+chunk_size] 
                   for i in range(0, len(states_list), chunk_size)]
    
    # Parallel merge function
    def merge_chunk(states_chunk):
        chunk_q = {}
        for state in states_chunk:
            chunk_q[state] = {}
            for action in [0, 1, 2, 3]:
                q_values = []
                for agent in agents:
                    if state in agent.Q and action in agent.Q[state]:
                        q_values.append(agent.Q[state][action])
                if q_values:
                    chunk_q[state][action] = np.mean(q_values)
        return chunk_q
    
    # Use threads for I/O bound merging
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        chunk_results = list(executor.map(merge_chunk, state_chunks))
    
    # Combine chunks
    merged_agent = PickleCompatibleQAgent()
    for chunk_q in chunk_results:
        for state, actions in chunk_q.items():
            for action, value in actions.items():
                merged_agent._set_q_value(state, action, value)
    
    return merged_agent

def save_checkpoint(agent, filename='checkpoint.pkl'):
    """
    Save intermediate checkpoints to avoid losing progress
    """
    with open(filename, 'wb') as f:
        pickle.dump(agent, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_and_resume_training(checkpoint_file, additional_episodes):
    """
    Load checkpoint and continue training
    """
    with open(checkpoint_file, 'rb') as f:
        agent = pickle.load(f)
    
    # Continue training from checkpoint
    # Implementation depends on your agent structure
    return agent

def adaptive_episodes_per_core(target_time_minutes=60):
    """
    Automatically determine optimal episodes per core based on timing
    """
    import time
    
    # Quick benchmark
    test_episodes = 10000
    start = time.time()
    single_core_train(test_episodes)
    elapsed = time.time() - start
    
    # Calculate episodes for target time
    episodes_per_second = test_episodes / elapsed
    target_episodes = int(episodes_per_second * target_time_minutes * 60)
    
    # Round to nearest million
    return (target_episodes // 1000000) * 1000000

def memory_efficient_training(episodes_per_core=32000000, checkpoint_interval=8000000):
    """
    Training with periodic checkpointing and memory management
    """
    cores = min(mp.cpu_count() - 2, 6)
    
    # Train in chunks to manage memory
    chunks = episodes_per_core // checkpoint_interval
    all_agents = []
    
    for chunk in range(chunks):
        print(f"Training chunk {chunk+1}/{chunks}")
        
        with mp.Pool(cores) as pool:
            results = pool.map(single_core_train, 
                             [checkpoint_interval] * cores)
        
        valid_results = [r for r in results if r[0] is not None]
        
        if valid_results:
            # Merge this chunk's agents
            chunk_agent = vectorized_merge_q_tables(valid_results)
            all_agents.append(chunk_agent)
            
            # Save checkpoint
            save_checkpoint(chunk_agent, f'checkpoint_chunk_{chunk}.pkl')
            
            # Clear memory
            del valid_results
    
    # Final merge of all chunks
    if all_agents:
        final_results = [(agent, None) for agent in all_agents]
        final_agent = vectorized_merge_q_tables(final_results)
        return final_agent
    
    return None

if __name__ == "__main__":
    print("Starting optimized parallel Q-Learning training for M2...")
    
    # Option 1: Standard optimized training
    results = parallel_train_episodes_optimized(episodes_per_core=64000000)
    
    # Option 2: Memory-efficient training with checkpoints
    # final_agent = memory_efficient_training(episodes_per_core=16000000)
    
    # Option 3: Adaptive training based on time budget
    # optimal_episodes = adaptive_episodes_per_core(target_time_minutes=30)
    # print(f"Using {optimal_episodes:,} episodes per core")
    # results = parallel_train_episodes_optimized(episodes_per_core=optimal_episodes)
    
    if results:
        print("Using vectorized merge...")
        # Use vectorized merge for speed
        merged_agent = vectorized_merge_q_tables(results)
        
        # For very large state spaces, use parallel merge
        # merged_agent = parallel_merge_q_tables(results, num_workers=4)
        
        if merged_agent:
            merged_agent.save_model('q_model_parallel_optimized.pkl')
            print("Saved optimized model")
            
            # Compare performance
            from cardCounting import BlackjackEnv, BasicStrategyAgent
            q_rewards, cc_rewards = compare_parallel_agents(num_hands=1000000)
        else:
            print("Failed to merge agents")

    cProfile.run('single_core_train(10000, core_id=0)', sort='cumtime')
