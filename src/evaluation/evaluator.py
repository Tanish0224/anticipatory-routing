import numpy as np

def evaluate_agents(env, agents_dict, eval_seeds, max_steps=50):
    """
    Evaluates multiple agents (or baselines) on a set of fixed seeds.
    Returns a dictionary of metrics for each agent.
    """
    results = {name: {'rewards': [], 'successes': [], 'steps': []} for name in agents_dict.keys()}
    
    for seed in eval_seeds:
        # Pre-generate start/dest and noise for fairness
        # We can just use the gym seed, but need to be careful that different 
        # agents taking different path lengths might shift the random state.
        # Actually, TrafficGenerator relies on the env's np_random.
        # Pre-roll the environment for the Oracle if it exists in the test pool
        future_weights = []
        state, _ = env.reset(seed=seed)
        future_weights.append(env.traffic_gen._get_current_weights())
        # Save env state so we can restore it? Actually, TrafficGenerator relies on np_random.
        # We can just re-seed to get the exact same traffic sequence for the actual agent runs!
        for _ in range(max_steps):
            env.traffic_gen.step()
            future_weights.append(env.traffic_gen._get_current_weights())
            
        for name, agent in agents_dict.items():
            state, _ = env.reset(seed=seed)
            if hasattr(agent, 'reset'):
                import inspect
                # If reset accepts future_weights (like Oracle does)
                if 'future_weights' in inspect.signature(agent.reset).parameters:
                    agent.reset(future_weights=future_weights)
                else:
                    agent.reset()
                
            episode_reward = 0
            steps = 0
            success = False
            
            for step in range(max_steps):
                if hasattr(agent, 'act') and 'obs' not in agent.act.__code__.co_varnames:
                    # RL agent
                    action = agent.act(state, eval_mode=True)
                else:
                    # Baseline
                    action = agent.act(state)
                    
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                steps += 1
                state = next_state
                
                if terminated:
                    # Reward contains +100 bonus, check if we actually reached dest
                    if env.current_node == env.dest_node:
                        success = True
                    break
                elif truncated:
                    break
                    
            results[name]['rewards'].append(episode_reward)
            results[name]['successes'].append(1.0 if success else 0.0)
            results[name]['steps'].append(steps)
            
    # Aggregate
    summary = {}
    for name in results:
        summary[name] = {
            'mean_reward': np.mean(results[name]['rewards']),
            'std_reward': np.std(results[name]['rewards']),
            'success_rate': np.mean(results[name]['successes']),
            'mean_steps': np.mean(results[name]['steps'])
        }
        
    return summary, results
