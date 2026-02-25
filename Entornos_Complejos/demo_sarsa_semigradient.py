"""
Demo rápida del agente SARSA Semi-Gradiente.
Ejecuta este archivo para ver el agente en acción.
"""

import gymnasium as gym
import numpy as np
from Entornos_Complejos.src.agents import AgentSarsaSemiGradient
from Entornos_Complejos.src.policies import EpsilonGreedyPolicy


def simple_feature_extractor(state, action, env):
    """
    Extractor de características simple para espacios continuos.
    Usa características polinómicas de grado 2.
    """
    state = np.array(state).flatten()  # Asegurar que sea 1D
    n_actions = env.action_space.n

    # Características base: [1, s1, s2, s1^2, s2^2, s1*s2]
    # Para MountainCar: posición y velocidad
    features_list = [1.0]  # Bias

    # Términos lineales
    for s in state:
        features_list.append(s)

    # Términos cuadráticos
    for s in state:
        features_list.append(s ** 2)

    # Términos cruzados (si hay más de una dimensión)
    if len(state) > 1:
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                features_list.append(state[i] * state[j])

    base_features = np.array(features_list)
    n_base_features = len(base_features)

    # Crear vector de características con one-hot encoding para la acción
    features = np.zeros(n_base_features * n_actions)
    start_idx = action * n_base_features
    end_idx = start_idx + n_base_features
    features[start_idx:end_idx] = base_features

    return features


def main():
    print("=" * 70)
    print("DEMO: SARSA Semi-Gradiente en MountainCar-v0")
    print("=" * 70)

    # Configuración
    env_name = 'MountainCar-v0'
    n_episodes = 5000
    alpha = 0.2  # Tasa de aprendizaje más alta para MountainCar
    gamma = 0.99
    epsilon = 0.1

    # Crear entorno
    env = gym.make(env_name)
    n_actions = env.action_space.n

    # Calcular número de características
    test_state, _ = env.reset()
    test_features = simple_feature_extractor(test_state, 0, env)
    n_features = len(test_features)

    print(f"\nConfiguración:")
    print(f"  - Entorno: {env_name}")
    print(f"  - Número de acciones: {n_actions}")
    print(f"  - Número de características: {n_features}")
    print(f"  - Alpha (tasa de aprendizaje): {alpha}")
    print(f"  - Gamma (factor de descuento): {gamma}")
    print(f"  - Epsilon (exploración): {epsilon}")
    print(f"  - Episodios de entrenamiento: {n_episodes}")

    # Crear política
    policy = EpsilonGreedyPolicy(epsilon=epsilon, n_actions=n_actions)

    # Crear agente
    agent = AgentSarsaSemiGradient(
        env=env,
        policy=policy,
        feature_extractor=simple_feature_extractor,
        n_features=n_features,
        alpha=alpha,
        gamma=gamma
    )

    print("\n" + "=" * 70)
    print("INICIANDO ENTRENAMIENTO...")
    print("=" * 70)

    # Entrenamiento
    episode_rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        action = agent.get_action(state)
        done = False
        truncated = False
        total_reward = 0

        while not (done or truncated):
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            # Actualizar agente
            next_action = agent.update(state, action, reward, next_state, done or truncated)

            state = next_state
            action = next_action if next_action is not None else agent.get_action(state)

        episode_rewards.append(total_reward)

        # Mostrar progreso cada 50 episodios
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            weight_norm = np.linalg.norm(agent.get_weights())
            print(f"Episodio {episode + 1:3d}/{n_episodes} | "
                  f"Recompensa promedio (últimos 50): {avg_reward:6.2f} | "
                  f"Norma de pesos: {weight_norm:8.4f}")

    env.close()

    # Resultados finales
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 70)

    avg_last_100 = np.mean(episode_rewards[-100:])
    avg_first_100 = np.mean(episode_rewards[:100])
    max_reward = np.max(episode_rewards)

    print(f"\nEstadísticas:")
    print(f"  - Recompensa promedio (primeros 100 episodios): {avg_first_100:.2f}")
    print(f"  - Recompensa promedio (últimos 100 episodios): {avg_last_100:.2f}")
    print(f"  - Mejora: {avg_last_100 - avg_first_100:+.2f}")
    print(f"  - Recompensa máxima alcanzada: {max_reward:.2f}")
    print(f"  - Norma final del vector de pesos: {np.linalg.norm(agent.get_weights()):.4f}")

    # Evaluación (sin exploración)
    print("\n" + "=" * 70)
    print("EVALUACIÓN (sin exploración, epsilon=0)")
    print("=" * 70)

    # Crear política greedy para evaluación
    eval_policy = EpsilonGreedyPolicy(epsilon=0.0, n_actions=n_actions)
    agent.policy = eval_policy

    eval_rewards = []
    n_eval_episodes = 10

    for episode in range(n_eval_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0

        while not (done or truncated):
            action = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state

        eval_rewards.append(total_reward)
        print(f"  Episodio de evaluación {episode + 1}: {total_reward:.0f}")

    env.close()

    print(f"\nRecompensa promedio en evaluación: {np.mean(eval_rewards):.2f}")
    print("\n" + "=" * 70)
    print("DEMO COMPLETADA")
    print("=" * 70)

    # Información adicional
    print("\n💡 Consejos:")
    print("  - MountainCar es un problema difícil (recompensa -1 por paso)")
    print("  - El objetivo es llegar a la bandera en la cima de la montaña")
    print("  - Si el agente no aprende, prueba ajustar alpha (0.05 - 0.2)")
    print("  - Para mejor rendimiento, prueba tile coding (ver ejemplos)")
    print("  - Ver 'examples/ejemplo_sarsa_semigradient.py' para más opciones")
    print("  - Ver 'docs/SARSA_SEMIGRADIENT.md' para documentación completa")


if __name__ == "__main__":
    main()
