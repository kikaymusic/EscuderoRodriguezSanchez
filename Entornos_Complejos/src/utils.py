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


def extract_v_and_policy_continuous(env, agent, resolution=50):
    """
    Equivalente a q_to_v_and_policy para entornos continuos con aproximación de funciones.
    Muestrea el espacio de estados en una cuadrícula (grid) regular.

    :param env: Entorno de Gymnasium (necesario para conocer los límites).
    :param agent: Agente entrenado (puede ser SARSA semi-gradient o Deep Q-Learning).
    :param resolution: Número de puntos a muestrear por cada dimensión del estado.
    :return: pos_grid, vel_grid, V_grid, Policy_grid
    """
    import torch

    # 1. Obtener los límites del entorno
    min_pos, min_vel = env.observation_space.low
    max_pos, max_vel = env.observation_space.high

    # 2. Crear los ejes de la cuadrícula
    pos_space = np.linspace(min_pos, max_pos, resolution)
    vel_space = np.linspace(min_vel, max_vel, resolution)

    # 3. Inicializar las matrices de Valor y Política
    V_grid = np.zeros((resolution, resolution))
    policy_grid = np.zeros((resolution, resolution))

    n_actions = env.action_space.n

    # Detectar si es un agente Deep Q-Learning o uno basado en features
    is_deep_q = hasattr(agent, 'model') and hasattr(agent, 'device')

    # 4. Evaluar la aproximación de la función en cada punto
    for i, p in enumerate(pos_space):
        for j, v in enumerate(vel_space):
            state = np.array([p, v])

            if is_deep_q:
                # Para Deep Q-Learning, usar la red neuronal
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    q_values = agent.model(state_tensor).cpu().numpy().squeeze()
            else:
                # Para agentes basados en features (SARSA semi-gradient)
                q_values = np.zeros(n_actions)
                for a in range(n_actions):
                    features = agent.feature_extractor(state, a)
                    q_values[a] = np.dot(agent.weights, features)

            # V(s) es el valor máximo entre las acciones disponibles
            V_grid[i, j] = np.max(q_values)
            # La política determinista (greedy) es la acción con mayor valor
            policy_grid[i, j] = np.argmax(q_values)

    return pos_space, vel_space, V_grid, policy_grid