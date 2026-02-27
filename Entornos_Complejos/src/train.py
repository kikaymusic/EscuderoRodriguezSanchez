from .utils import SEMILLA
import numpy as np

N_EPISODES = 80000
MAX_STEPS = 10

def train_agent(env, agent):
    episode_returns = []
    episode_lengths = []

    for ep in range(N_EPISODES):
        state, info = env.reset(seed=SEMILLA + ep)

        done = False
        total_reward = 0.0
        steps = 0
        next_action = None  # Used only by SARSA-like agents

        while not done and steps < MAX_STEPS:
            if next_action is None:
                action = agent.get_action(state)
            else:
                action = next_action
                next_action = None

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # For MC / Q-Learning this will usually return None.
            # For SARSA it may return the next action to preserve on-policy behavior.
            maybe_next_action = agent.update(state, action, reward, next_state, done)

            if (maybe_next_action is not None) and (not done):
                next_action = maybe_next_action

            state = next_state
            total_reward += reward
            steps += 1

        episode_returns.append(total_reward)
        episode_lengths.append(steps)

        if (ep + 1) % 5000 == 0:
            avg_last = float(np.mean(episode_returns[-5000:]))
            print(f"Episode {ep+1}/{N_EPISODES} - avg_return_last_5000={avg_last:.4f}")

    return episode_returns, episode_lengths