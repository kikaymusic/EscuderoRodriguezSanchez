import numpy as np

SEMILLA = 42


def q_to_v_and_policy(Q):
    """
    Convierte una tabla de valores de acción Q(s,a) en:
    - una función de valor de estado V(s)
    - una política greedy pi(s)

    Para cada estado s se calcula:
    - V(s) = max_a Q(s,a)
    - pi(s) = argmax_a Q(s,a)

    Parámetros
    Q : dict
        Diccionario que mapea estado -> array/lista de valores de acción.
        Ejemplo:
        Q[(suma_jugador, carta_visible_dealer, usable_ace)] = [Q_stick, Q_hit]

    Devuelve
    V : dict
        Diccionario estado -> valor de estado, calculado como el máximo valor Q.
    policy : dict
        Diccionario estado -> acción greedy (índice de la acción con mayor valor).
        En Blackjack:
        0 = STICK, 1 = HIT.
    """
    V = {}
    policy = {}
    for s, qvals in Q.items():
        qvals = np.asarray(qvals, dtype=np.float32)
        best_a = int(np.argmax(qvals))
        policy[s] = best_a
        V[s] = float(np.max(qvals))
    return V, policy


import numpy as np

def evaluate_agent(env, agent, n_episodes=1000, seed_base=0, greedy_eval=True):
    """
    Evaluate an agent without rendering over many episodes.

    Parameters
    ----------
    env : gymnasium.Env
        Blackjack environment.
    agent : AgentMonteCarlo
        Trained or untrained agent.
    n_episodes : int
        Number of evaluation episodes.
    seed_base : int
        Base seed to make evaluation reproducible.
    greedy_eval : bool
        If True, force greedy action selection from Q-table.
        If False, use the agent policy as-is.

    Returns
    -------
    dict
        Aggregated evaluation metrics.
    """

    def select_action(state):
        # Greedy evaluation directly from Q if available
        if greedy_eval and hasattr(agent, "q_table"):
            q_values = agent.q_table.get(state, None)
            if q_values is not None:
                return int(np.argmax(q_values))

        # Fallbacks for policy-based evaluation
        if hasattr(agent, "get_action"):
            try:
                return agent.get_action(state)
            except TypeError:
                pass

        if hasattr(agent, "target_policy") and hasattr(agent, "q_table"):
            try:
                return agent.target_policy.get_action(state, agent.q_table)
            except TypeError:
                pass

        raise AttributeError("No valid action selection method found for evaluation.")

    wins, losses, draws = 0, 0, 0
    total_reward = 0.0
    episode_lengths = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed_base + ep)
        done = False
        steps = 0

        while not done:
            action = select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        total_reward += reward
        episode_lengths.append(steps)

        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(100 * wins / n_episodes, 2),
        "avg_reward": round(total_reward / n_episodes, 4),
        "avg_episode_length": round(float(np.mean(episode_lengths)), 4),
        "episodes": n_episodes
    }