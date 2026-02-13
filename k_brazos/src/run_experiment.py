from typing import List

import numpy as np

from .algorithms import Algorithm
from .arms import Bandit
from .utils import SEMILLA


def run_experiment(bandit: Bandit, algorithms: List[Algorithm], steps: int, runs: int):

    optimal_arm = bandit.optimal_arm  # Necesario para calcular el porcentaje de selecciones óptimas.

    rewards = np.zeros((len(algorithms), steps))  # Matriz para almacenar las recompensas promedio.

    optimal_selections = np.zeros(
        (len(algorithms), steps))  # Matriz para almacenar el porcentaje de selecciones óptimas.

    # Rastrear victorias y pérdidas para cada algoritmo
    arm_stats = []
    for _ in algorithms:
        arm_stats.append({
            "wins": np.zeros(steps),
            "losses": np.zeros(steps)
        })

    np.random.seed(SEMILLA)  # Asegurar reproducibilidad de resultados.

    for run in range(runs):
        current_bandit = Bandit(arms=bandit.arms)

        for algo in algorithms:
            algo.reset()  # Reiniciar los valores de los algoritmos.

        total_rewards_per_algo = np.zeros(
            len(algorithms))  # Acumulador de recompensas por algoritmo. Necesario para calcular el promedio.

        for step in range(steps):
            for idx, algo in enumerate(algorithms):
                chosen_arm = algo.select_arm()  # Seleccionar un brazo según la política del algoritmo.
                reward = current_bandit.pull_arm(chosen_arm)  # Obtener la recompensa del brazo seleccionado.
                algo.update(chosen_arm, reward)  # Actualizar el valor estimado del brazo seleccionado.

                rewards[
                    idx, step] += reward  # Acumular la recompensa obtenida en la matriz rewards para el algoritmo idx en el paso step.
                total_rewards_per_algo[
                    idx] += reward  # Acumular la recompensa obtenida en total_rewards_per_algo para el algoritmo idx.

                if chosen_arm == optimal_arm:
                    optimal_selections[
                        idx, step] += 1  # Acumular la selección del brazo óptimo en optimal_selections para el algoritmo idx en el paso step.
                    arm_stats[idx]["wins"][step] += 1  # Contar como victoria
                else:
                    arm_stats[idx]["losses"][step] += 1  # Contar como pérdida

    rewards /= runs
    optimal_selections /= runs

    # Promediar las victorias y pérdidas sobre las ejecuciones
    for idx in range(len(algorithms)):
        arm_stats[idx]["wins"] /= runs
        arm_stats[idx]["losses"] /= runs

    return rewards, optimal_selections, arm_stats