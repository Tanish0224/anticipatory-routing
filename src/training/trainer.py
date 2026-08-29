import numpy as np

def train_dqn(env, agent, num_episodes=2000, max_steps=50):
    rewards_history = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            agent.step(state, action, reward, next_state, terminated or truncated)
            
            state = next_state
            episode_reward += reward
            
            if terminated or truncated:
                break
                
        agent.update_epsilon()
        rewards_history.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode+1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f}")
            
    return rewards_history
