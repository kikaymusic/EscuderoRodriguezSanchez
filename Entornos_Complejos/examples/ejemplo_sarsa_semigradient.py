"""
Ejemplo de uso del agente SARSA Semi-Gradiente con diferentes extractores de características.

Este ejemplo muestra cómo usar el AgentSarsaSemiGradient con aproximación de funciones
para resolver problemas de aprendizaje por refuerzo con espacios de estados continuos
o muy grandes.
"""

import gymnasium as gym
import numpy as np
from Entornos_Complejos.src.agents import AgentSarsaSemiGradient
from Entornos_Complejos.src.policies import EpsilonGreedyPolicy


# ============================================================================
# EXTRACTORES DE CARACTERÍSTICAS (FEATURE EXTRACTORS)
# ============================================================================

def tile_coding_feature_extractor(state, action, env, n_tilings=8, n_tiles_per_dim=8):
    """
    Extractor de características usando Tile Coding.

    Tile Coding es una técnica popular para discretizar espacios continuos
    creando múltiples "grillas" (tilings) desplazadas entre sí.

    :param state: Estado del entorno (puede ser continuo).
    :param action: Acción a tomar.
    :param env: Entorno de Gymnasium.
    :param n_tilings: Número de tilings (grillas superpuestas).
    :param n_tiles_per_dim: Número de tiles por dimensión en cada tiling.
    :return: Vector de características binario (one-hot encoding).
    """
    # Obtenemos los límites del espacio de estados
    state_low = env.observation_space.low
    state_high = env.observation_space.high
    n_actions = env.action_space.n

    # Convertimos el estado a numpy array si no lo es
    state = np.array(state)
    n_dims = len(state)

    # Calculamos el tamaño total del vector de características
    tiles_per_tiling = n_tiles_per_dim ** n_dims
    total_features = n_tilings * tiles_per_tiling * n_actions

    # Inicializamos el vector de características
    features = np.zeros(total_features)

    # Para cada tiling
    for tiling_idx in range(n_tilings):
        # Calculamos el desplazamiento (offset) para este tiling
        offset = (state_high - state_low) / (n_tiles_per_dim * n_tilings) * tiling_idx

        # Normalizamos y desplazamos el estado
        normalized_state = (state - state_low + offset) / (state_high - state_low)

        # Calculamos el índice del tile en cada dimensión
        tile_indices = np.floor(normalized_state * n_tiles_per_dim).astype(int)
        tile_indices = np.clip(tile_indices, 0, n_tiles_per_dim - 1)

        # Convertimos los índices multidimensionales a un índice único
        tile_idx = 0
        for dim in range(n_dims):
            tile_idx = tile_idx * n_tiles_per_dim + tile_indices[dim]

        # Calculamos el índice en el vector de características
        # Incluimos la acción en el índice
        feature_idx = tiling_idx * tiles_per_tiling * n_actions + tile_idx * n_actions + action

        # Activamos esta característica
        features[feature_idx] = 1.0

    return features


def polynomial_feature_extractor(state, action, env, degree=2):
    """
    Extractor de características polinómicas.

    Crea características polinómicas del estado hasta un grado especificado.
    Útil para aproximar funciones no lineales.

    :param state: Estado del entorno.
    :param action: Acción a tomar.
    :param env: Entorno de Gymnasium.
    :param degree: Grado del polinomio.
    :return: Vector de características polinómicas.
    """
    state = np.array(state)
    n_actions = env.action_space.n
    n_dims = len(state)

    # Creamos características polinómicas
    features_list = [1.0]  # Término de sesgo (bias)

    # Términos lineales
    for i in range(n_dims):
        features_list.append(state[i])

    # Términos de grado superior
    if degree >= 2:
        # Términos cuadráticos
        for i in range(n_dims):
            features_list.append(state[i] ** 2)

        # Términos cruzados
        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                features_list.append(state[i] * state[j])

    if degree >= 3:
        # Términos cúbicos
        for i in range(n_dims):
            features_list.append(state[i] ** 3)

    # Convertimos a numpy array
    base_features = np.array(features_list)

    # Creamos un vector de características para cada acción (one-hot encoding)
    n_base_features = len(base_features)
    features = np.zeros(n_base_features * n_actions)

    # Activamos las características correspondientes a la acción
    start_idx = action * n_base_features
    end_idx = start_idx + n_base_features
    features[start_idx:end_idx] = base_features

    return features


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


# ============================================================================
# FUNCIÓN DE ENTRENAMIENTO
# ============================================================================

def train_sarsa_semigradient(env_name='CartPole-v1', feature_extractor=None,
                             n_features=None, n_episodes=500, alpha=0.01,
                             gamma=0.99, epsilon=0.1):
    """
    Entrena un agente SARSA Semi-Gradiente en un entorno.

    :param env_name: Nombre del entorno de Gymnasium.
    :param feature_extractor: Función extractora de características.
    :param n_features: Número de características.
    :param n_episodes: Número de episodios de entrenamiento.
    :param alpha: Tasa de aprendizaje.
    :param gamma: Factor de descuento.
    :param epsilon: Parámetro epsilon para la política epsilon-greedy.
    :return: Agente entrenado y lista de recompensas por episodio.
    """
    # Creamos el entorno
    env = gym.make(env_name)

    # Si no se proporciona un extractor de características, usamos el simple
    if feature_extractor is None:
        feature_extractor = simple_feature_extractor

    # Si no se proporciona el número de características, lo calculamos
    if n_features is None:
        # Hacemos una llamada de prueba para obtener el tamaño
        test_state, _ = env.reset()
        test_action = 0
        test_features = feature_extractor(test_state, test_action, env)
        n_features = len(test_features)
        print(f"Número de características detectado: {n_features}")

    # Creamos la política
    n_actions = env.action_space.n
    policy = EpsilonGreedyPolicy(epsilon=epsilon, n_actions=n_actions)

    # Creamos el agente
    agent = AgentSarsaSemiGradient(
        env=env,
        policy=policy,
        feature_extractor=feature_extractor,
        n_features=n_features,
        alpha=alpha,
        gamma=gamma
    )

    # Lista para almacenar las recompensas
    episode_rewards = []

    # Entrenamiento
    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0

        # Seleccionamos la primera acción
        action = agent.get_action(state)

        while not (done or truncated):
            # Ejecutamos la acción
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            # Actualizamos el agente y obtenemos la siguiente acción
            next_action = agent.update(state, action, reward, next_state, done or truncated)

            # Preparamos para el siguiente paso
            state = next_state
            action = next_action if next_action is not None else agent.get_action(state)

        episode_rewards.append(total_reward)

        # Mostramos el progreso cada 50 episodios
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episodio {episode + 1}/{n_episodes}, Recompensa promedio (últimos 50): {avg_reward:.2f}")

    env.close()
    return agent, episode_rewards


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Ejemplo: SARSA Semi-Gradiente con características polinómicas")
    print("=" * 70)

    # Definimos el extractor de características
    def my_feature_extractor(state, action, env):
        return polynomial_feature_extractor(state, action, env, degree=2)

    # Calculamos el número de características
    # Para CartPole: 4 dimensiones de estado, 2 acciones
    # Características: 1 (bias) + 4 (lineales) + 4 (cuadráticas) + 6 (cruzadas) = 15
    # Por 2 acciones = 30 características totales

    # Entrenamos el agente
    agent, rewards = train_sarsa_semigradient(
        env_name='CartPole-v1',
        feature_extractor=my_feature_extractor,
        n_features=None,  # Se calculará automáticamente
        n_episodes=500,
        alpha=0.001,  # Tasa de aprendizaje más baja para características polinómicas
        gamma=0.99,
        epsilon=0.1
    )

    print("\n" + "=" * 70)
    print("Entrenamiento completado!")
    print(f"Recompensa promedio (últimos 100 episodios): {np.mean(rewards[-100:]):.2f}")
    print("=" * 70)

    # Mostramos los pesos aprendidos
    print(f"\nNorma del vector de pesos: {np.linalg.norm(agent.get_weights()):.4f}")
