from fixed_q_learning import fast_train_q_agent, compare_optimized_agents

if __name__ == "__main__":
    # Train the Q-learning agent
    agent, rewards, eval_scores = fast_train_q_agent(num_episodes=200000)
    
    # Compare performance against card counting
    q_rewards, cc_rewards = compare_optimized_agents(num_hands=500000)