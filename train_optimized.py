from fixed_q_learning import fast_train_q_agent, compare_optimized_agents
from parallel_training import parallel_train_episodes, merge_q_tables

def main():
    print("Choose training method:")
    print("1. Single-core training (2M episodes)")
    print("2. Multi-core training (4M episodes total)")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        # Single-core training
        agent, rewards, eval_scores = fast_train_q_agent(num_episodes=2000000)
        if agent:
            agent.save_model('q_model_single.pkl')
    
    elif choice == "2":
        # Multi-core training
        print("Starting parallel training...")
        results = parallel_train_episodes(episodes_per_core=500000)  # 8 cores × 500k = 4M total
        
        # Merge results
        merged_agent = merge_q_tables(results)
        merged_agent.save_model('q_model_parallel.pkl')
        
        print("Parallel training complete!")
    
    # Compare performance
    print("\nTesting performance...")
    q_rewards, cc_rewards = compare_optimized_agents(num_hands=10000)

if __name__ == "__main__":
    main()