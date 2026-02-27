import numpy as np


def simple_feature_extractor(state, action, env):
    """
    Extractor de características simple para espacios discretos.

    Crea un vector one-hot encoding para el par estado-acción.
    Útil para problemas pequeños o como baseline.

    :param state: Estado del entorno (debe ser hashable).
    :param action: Acción a tomar.
    :param env: Entorno de Gymnasium.
    :return: Vector de características one-hot.
    """
    # Para espacios discretos, podemos usar un enfoque tabular
    # Asumimos que el estado es un entero o tupla de enteros

    # Si el espacio de estados es discreto
    if hasattr(env.observation_space, 'n'):
        n_states = env.observation_space.n
        n_actions = env.action_space.n

        # Creamos un vector one-hot
        features = np.zeros(n_states * n_actions)
        idx = state * n_actions + action
        features[idx] = 1.0

        return features
    else:
        # Para espacios continuos, usamos las características del estado directamente
        state = np.array(state)
        n_actions = env.action_space.n
        n_dims = len(state)

        # Agregamos un término de sesgo
        base_features = np.concatenate([[1.0], state])
        n_base_features = len(base_features)

        # Creamos un vector para cada acción
        features = np.zeros(n_base_features * n_actions)
        start_idx = action * n_base_features
        end_idx = start_idx + n_base_features
        features[start_idx:end_idx] = base_features

        return features