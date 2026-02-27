from .utils import SEMILLA
import numpy as np

N_EPISODES = 80000
MAX_STEPS = 10

def train_agent(env, agent, n_episodes=N_EPISODES, max_steps=MAX_STEPS, print_freq=5000):
    """
    Bucle de entrenamiento generalizado para agentes de RL (MC, Q-Learning, SARSA).
    """
    episode_returns = []
    episode_lengths = []

    for ep in range(n_episodes):
        # Mantenemos la semilla incremental para reproducibilidad en la exploración
        state, info = env.reset(seed=SEMILLA + ep)

        done = False
        total_reward = 0.0
        steps = 0
        next_action = None  # Usado por agentes tipo SARSA (on-policy)

        while not done and steps < max_steps:
            # 1. Selección de acción
            if next_action is None:
                action = agent.get_action(state)
            else:
                action = next_action
                next_action = None

            # 2. Interacción con el entorno
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 3. Actualización del agente
            # - MC: Solo guarda la tupla (s, a, r). Si done=True, procesa el episodio. Retorna None.
            # - SARSA: Actualiza pesos con (s, a, r, s') y retorna la SIGUIENTE acción a(t+1).
            maybe_next_action = agent.update(state, action, reward, next_state, done)

            if (maybe_next_action is not None) and (not done):
                next_action = maybe_next_action

            # 4. Preparar siguiente paso
            state = next_state
            total_reward += reward
            steps += 1

        # 5. Registro de métricas
        episode_returns.append(total_reward)
        episode_lengths.append(steps)

        # 6. Progreso por pantalla
        if (ep + 1) % print_freq == 0:
            avg_last = float(np.mean(episode_returns[-print_freq:]))
            print(f"Episode {ep+1}/{n_episodes} - avg_return_last_{print_freq}={avg_last:.4f}")

    return episode_returns, episode_lengths