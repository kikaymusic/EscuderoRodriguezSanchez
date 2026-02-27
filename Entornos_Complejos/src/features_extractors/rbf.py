import numpy as np


def rbf_feature_extractor(state, action, env, n_centers=10, sigma=1.0):
    """
    Extractor de características usando Funciones de Base Radial (RBF).

    Las RBF son útiles para aproximar funciones suaves y capturar
    similitudes locales en el espacio de estados.

    :param state: Estado del entorno.
    :param action: Acción a tomar.
    :param env: Entorno de Gymnasium.
    :param n_centers: Número de centros RBF por dimensión.
    :param sigma: Ancho de las funciones gaussianas.
    :return: Vector de características RBF.
    """
    state = np.array(state)
    state_low = env.observation_space.low
    state_high = env.observation_space.high
    n_actions = env.action_space.n
    n_dims = len(state)

    # Creamos una grilla de centros en el espacio de estados
    centers = []
    for dim in range(n_dims):
        dim_centers = np.linspace(state_low[dim], state_high[dim], n_centers)
        centers.append(dim_centers)

    # Calculamos las características RBF
    rbf_features = []

    # Para cada combinación de centros
    import itertools
    for center_combo in itertools.product(*centers):
        center = np.array(center_combo)
        # Calculamos la distancia al centro
        distance = np.linalg.norm(state - center)
        # Aplicamos la función gaussiana
        rbf_value = np.exp(-distance ** 2 / (2 * sigma ** 2))
        rbf_features.append(rbf_value)

    # Convertimos a numpy array
    base_features = np.array(rbf_features)

    # Creamos un vector de características para cada acción
    n_base_features = len(base_features)
    features = np.zeros(n_base_features * n_actions)

    # Activamos las características correspondientes a la acción
    start_idx = action * n_base_features
    end_idx = start_idx + n_base_features
    features[start_idx:end_idx] = base_features

    return features