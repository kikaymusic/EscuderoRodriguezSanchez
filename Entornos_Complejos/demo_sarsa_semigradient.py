"""
Demo rápida del agente SARSA Semi-Gradiente.
Ejecuta este archivo para ver el agente en acción.
"""

import gymnasium as gym
import numpy as np
from Entornos_Complejos.src.agents import AgentSarsaSemiGradient
from Entornos_Complejos.src.features_extractors.polynomial import polynomial_feature_extractor
from Entornos_Complejos.src.policies import EpsilonGreedyPolicy
from Entornos_Complejos.src.train import train_agent


def main():
    print("=" * 70)
    print("DEMO: SARSA Semi-Gradiente en MountainCar-v0")
    print("=" * 70)

    # Configuración
    env_name = 'MountainCar-v0'
    n_episodes = 5000
    #alpha = 0.005  # Tasa de aprendizaje más alta para MountainCar
    alpha = 0.01
    gamma = 0.99
    epsilon = 0.1

    # Crear entorno
    env = gym.make(env_name)
    n_actions = env.action_space.n

    # Calcular número de características
    test_state, _ = env.reset()
    test_features = polynomial_feature_extractor(test_state, 0, env)
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
        feature_extractor=polynomial_feature_extractor,
        n_features=n_features,
        alpha=alpha,
        gamma=gamma
    )

    print("\n" + "=" * 70)
    print("INICIANDO ENTRENAMIENTO...")
    print("=" * 70)

    episode_returns, episode_lengths = train_agent(env, agent, n_episodes=5000, max_steps=300)

    avg_last_100 = np.mean(episode_returns[-100:])
    avg_first_100 = np.mean(episode_returns[:100])
    max_reward = np.max(episode_returns)

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


if __name__ == "__main__":
    main()
