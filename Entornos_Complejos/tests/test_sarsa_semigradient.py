"""
Tests para el agente SARSA Semi-Gradiente.
"""

import numpy as np
import gymnasium as gym
from Entornos_Complejos.src.agents import AgentSarsaSemiGradient
from Entornos_Complejos.src.features_extractors.simple import simple_feature_extractor
from Entornos_Complejos.src.policies import EpsilonGreedyPolicy




def test_agent_initialization():
    """
    Test: Verificar que el agente se inicializa correctamente.
    """
    print("Test 1: Inicialización del agente...")

    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    policy = EpsilonGreedyPolicy(epsilon=0.1, n_actions=n_actions)

    # Calcular número de características
    test_state, _ = env.reset()
    test_features = simple_feature_extractor(test_state, 0, env)
    n_features = len(test_features)

    agent = AgentSarsaSemiGradient(
        env=env,
        policy=policy,
        feature_extractor=simple_feature_extractor,
        n_features=n_features,
        alpha=0.01,
        gamma=0.99
    )

    # Verificaciones
    assert agent.weights.shape == (n_features,), "El vector de pesos tiene el tamaño incorrecto"
    assert np.all(agent.weights == 0), "Los pesos deberían inicializarse en cero"
    assert agent.alpha == 0.01, "Alpha no se estableció correctamente"
    assert agent.gamma == 0.99, "Gamma no se estableció correctamente"

    env.close()
    print("✓ Test 1 pasado: Inicialización correcta")


def test_get_action():
    """
    Test: Verificar que get_action devuelve una acción válida.
    """
    print("\nTest 2: Selección de acciones...")

    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    policy = EpsilonGreedyPolicy(epsilon=0.1, n_actions=n_actions)

    test_state, _ = env.reset()
    test_features = simple_feature_extractor(test_state, 0, env)
    n_features = len(test_features)

    agent = AgentSarsaSemiGradient(
        env=env,
        policy=policy,
        feature_extractor=simple_feature_extractor,
        n_features=n_features
    )

    state, _ = env.reset()
    action = agent.get_action(state)

    # Verificaciones
    assert isinstance(action, (int, np.integer)), "La acción debe ser un entero"
    assert 0 <= action < n_actions, f"La acción {action} está fuera del rango válido [0, {n_actions})"

    env.close()
    print("✓ Test 2 pasado: Selección de acciones válida")


def test_update():
    """
    Test: Verificar que update modifica los pesos correctamente.
    """
    print("\nTest 3: Actualización de pesos...")

    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    policy = EpsilonGreedyPolicy(epsilon=0.1, n_actions=n_actions)

    test_state, _ = env.reset()
    test_features = simple_feature_extractor(test_state, 0, env)
    n_features = len(test_features)

    agent = AgentSarsaSemiGradient(
        env=env,
        policy=policy,
        feature_extractor=simple_feature_extractor,
        n_features=n_features,
        alpha=0.1
    )

    # Guardamos los pesos iniciales
    initial_weights = agent.get_weights()

    # Realizamos una transición
    state, _ = env.reset()
    action = agent.get_action(state)
    next_state, reward, done, truncated, _ = env.step(action)

    # Actualizamos
    agent.update(state, action, reward, next_state, done or truncated)

    # Verificamos que los pesos cambiaron
    updated_weights = agent.get_weights()
    assert not np.allclose(initial_weights, updated_weights), "Los pesos deberían haber cambiado después de update"

    env.close()
    print("✓ Test 3 pasado: Actualización de pesos correcta")


def test_q_value_calculation():
    """
    Test: Verificar que los valores Q se calculan correctamente.
    """
    print("\nTest 4: Cálculo de valores Q...")

    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    policy = EpsilonGreedyPolicy(epsilon=0.1, n_actions=n_actions)

    test_state, _ = env.reset()
    test_features = simple_feature_extractor(test_state, 0, env)
    n_features = len(test_features)

    agent = AgentSarsaSemiGradient(
        env=env,
        policy=policy,
        feature_extractor=simple_feature_extractor,
        n_features=n_features
    )

    # Establecemos pesos conocidos
    agent.weights = np.ones(n_features)

    state, _ = env.reset()

    # Calculamos valores Q para todas las acciones
    q_values = agent._get_all_q_values(state)

    # Verificaciones
    assert q_values.shape == (n_actions,), "El array de valores Q tiene el tamaño incorrecto"
    assert np.all(np.isfinite(q_values)), "Los valores Q deben ser finitos"

    env.close()
    print("✓ Test 4 pasado: Cálculo de valores Q correcto")


def test_episode_training():
    """
    Test: Verificar que el agente puede completar un episodio de entrenamiento.
    """
    print("\nTest 5: Entrenamiento de un episodio completo...")

    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    policy = EpsilonGreedyPolicy(epsilon=0.1, n_actions=n_actions)

    test_state, _ = env.reset()
    test_features = simple_feature_extractor(test_state, 0, env)
    n_features = len(test_features)

    agent = AgentSarsaSemiGradient(
        env=env,
        policy=policy,
        feature_extractor=simple_feature_extractor,
        n_features=n_features,
        alpha=0.01,
        gamma=0.99
    )

    # Ejecutamos un episodio completo
    state, _ = env.reset()
    action = agent.get_action(state)
    done = False
    truncated = False
    total_reward = 0
    steps = 0

    while not (done or truncated) and steps < 500:
        next_state, reward, done, truncated, _ = env.step(action)
        total_reward += reward

        next_action = agent.update(state, action, reward, next_state, done or truncated)

        state = next_state
        action = next_action if next_action is not None else agent.get_action(state)
        steps += 1

    # Verificaciones
    assert steps > 0, "El episodio debería tener al menos un paso"
    assert total_reward > 0, "La recompensa total debería ser positiva"

    env.close()
    print(f"✓ Test 5 pasado: Episodio completado con {steps} pasos y recompensa {total_reward}")


def test_weight_management():
    """
    Test: Verificar las funciones de gestión de pesos.
    """
    print("\nTest 6: Gestión de pesos...")

    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    policy = EpsilonGreedyPolicy(epsilon=0.1, n_actions=n_actions)

    test_state, _ = env.reset()
    test_features = simple_feature_extractor(test_state, 0, env)
    n_features = len(test_features)

    agent = AgentSarsaSemiGradient(
        env=env,
        policy=policy,
        feature_extractor=simple_feature_extractor,
        n_features=n_features
    )

    # Test get_weights
    weights = agent.get_weights()
    assert isinstance(weights, np.ndarray), "get_weights debe devolver un numpy array"
    assert weights.shape == (n_features,), "El tamaño de los pesos es incorrecto"

    # Test set_weights
    new_weights = np.random.randn(n_features)
    agent.set_weights(new_weights)
    assert np.allclose(agent.weights, new_weights), "set_weights no estableció los pesos correctamente"

    # Test reset_weights
    agent.reset_weights()
    assert np.all(agent.weights == 0), "reset_weights debería establecer todos los pesos a cero"

    env.close()
    print("✓ Test 6 pasado: Gestión de pesos correcta")


def run_all_tests():
    """
    Ejecuta todos los tests.
    """
    print("=" * 70)
    print("EJECUTANDO TESTS PARA AGENTE SARSA SEMI-GRADIENTE")
    print("=" * 70)

    try:
        test_agent_initialization()
        test_get_action()
        test_update()
        test_q_value_calculation()
        test_episode_training()
        test_weight_management()

        print("\n" + "=" * 70)
        print("✓ TODOS LOS TESTS PASARON EXITOSAMENTE")
        print("=" * 70)

    except AssertionError as e:
        print(f"\n✗ TEST FALLIDO: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR INESPERADO: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
