"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ..algorithms.algorithm import Algorithm
from ..algorithms.epsilon_greedy import EpsilonGreedy


def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
    label = type(algo).__name__
    if isinstance(algo, EpsilonGreedy):
        label += f" (epsilon={algo.epsilon})"
    # elif isinstance(algo, OtroAlgoritmo):
    #     label += f" (parametro={algo.parametro})"
    # Añadir más condiciones para otros algoritmos aquí
    else:
        raise ValueError("El algoritmo debe ser de la clase Algorithm o una subclase.")
    return label


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    plt.figure(figsize=(14, 7))
    for idx, algorithm in enumerate(algorithms):
        label = get_algorithm_label(algorithm)
        plt.plot(range(steps), optimal_selections[idx] * 100, label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('% Selección Óptima', fontsize=14)
    plt.title('Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.ylim([0, 100])
    plt.tight_layout()
    plt.show()


def plot_arm_statistics(arm_stats: List[dict], algorithms: List[Algorithm], *args):
    """
    Mostrar la evolución del rechazo, de forma análoga a cómo se muestra la evolución de las recompensas.
    Genera gráficas separadas de Selección de Arms:
    Ganancias vs Pérdidas para cada algoritmo.
    :param arm_stats: Lista (de diccionarios) con estadísticas de cada brazo por algoritmo.
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param args: Opcional. Parámetros que consideres
    """
    raise NotImplementedError("Esta función aún no está implementada.")


def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: List[Algorithm], *args):
    """
    Función OPCIONAL.
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo.
    :param steps: Número de pasos de tiempo.
    :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos).
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param args: Opcional. Parámetros que consideres. P.e. la cota teórica Cte * ln(T).
    """
    raise NotImplementedError("Esta función aún no está implementada.")

# TODO Modifique los ficheros bandit_experiment.ipynb y https://github.com/ldaniel-hm/eml_k_bandit/blob/main/plotting/plotting.pyplotting.py para incluir en el estudio original las gráficas indicadas.

# TODO Cuando complete el estudio de la familia de algoritmos epsilon-greedy, indique cuál de todas las gráficas que se piden son las más relevantes ¿por qué es esa gráfica o esas gráficas las más importantes?

