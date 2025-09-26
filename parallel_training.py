import multiprocessing as mp
import numpy as np
from pickle_compatible_q_learning import single_core_train, PickleCompatibleQAgent

def parallel_train_episodes(episodes_per_core=8000000):
    cores = mp.cpu_count()  # M2 has 8 cores
    print(f"Using {cores} cores for parallel training")
    print(f"Total episodes: {cores * episodes_per_core:,}")
    
    with mp.Pool(cores) as pool:
        results = pool.map(single_core_train, [episodes_per_core] * cores)
    
    # Filter out None results
    valid_results = [r for r in results if r[0] is not None]
    print(f"Successfully trained {len(valid_results)} agents")
    
    return valid_results

def merge_q_tables(results):
    """
    Merge Q-tables from multiple agents by averaging
    """
    if not results:
        print("No valid results to merge")
        return None
        
    agents, reward_histories = zip(*results)
    merged_agent = PickleCompatibleQAgent()
    all_states = set()
    
    # Collect all states from all agents
    for agent in agents:
        all_states.update(agent.Q.keys())
    
    print(f"Merging {len(all_states)} unique states from {len(agents)} agents")
    
    # Average Q-values across agents
    for state in all_states:
        for action in [0, 1, 2, 3]:
            q_values = []
            for agent in agents:
                if state in agent.Q and action in agent.Q[state]:
                    q_values.append(agent.Q[state][action])
            
            if q_values:
                merged_agent._set_q_value(state, action, np.mean(q_values))
    
    return merged_agent

def compare_parallel_agents(num_hands=100000):
    """
    Compare merged agent vs card counting
    """
    try:
        from cardCounting import BlackjackEnv, BasicStrategyAgent
    except ImportError:
        print("Error: Cannot import from cardCounting")
        return [], []
    
    print(f"\nComparing agents over {num_hands} hands...")
    
    # Load merged Q-agent
    env_q = BlackjackEnv(num_decks=8, min_bet=50, max_bet=5000)
    q_agent = PickleCompatibleQAgent()
    
    try:
        q_agent.load_model('q_model_parallel.pkl')
        print("Loaded parallel-trained Q-model")
    except:
        print("No parallel model found")
        return [], []
    
    q_agent.set_training_mode(False)  # No exploration
    
    # Test Q-learning
    q_rewards = []
    for _ in range(num_hands):
        state = env_q.reset()
        done = False
        while not done:
            action = q_agent.get_action(state)
            state, reward, done = env_q.step(action)
        q_rewards.append(env_q.results_history[-1]['reward'])
    
    # Test card counting
    env_cc = BlackjackEnv(num_decks=8, min_bet=50, max_bet=5000)
    cc_agent = BasicStrategyAgent()
    
    cc_rewards = []
    for _ in range(num_hands):
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

if __name__ == "__main__":
    print("Starting parallel Q-Learning training...")
    
    # Parallel training
    results = parallel_train_episodes(episodes_per_core=16000000)
    
    if results:
        # Merge results
        print("Merging Q-tables from parallel training...")
        merged_agent = merge_q_tables(results)
        
        if merged_agent:
            merged_agent.save_model('q_model_parallel.pkl')
            print("Saved merged model as 'q_model_parallel.pkl'")
            
            # Compare performance
            q_rewards, cc_rewards = compare_parallel_agents(num_hands=100000)
        else:
            print("Failed to merge agents")
    else:
        print("Parallel training failed")