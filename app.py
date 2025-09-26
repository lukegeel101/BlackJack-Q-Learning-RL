from mainCounting import simulate_game, analyze_simulation

# Run simulation with 1000 hands
env, agent, results_df, count_df = simulate_game(num_hands=10000000, initial_bankroll=1000000)

# Analyze the results
stats = analyze_simulation(results_df, count_df)