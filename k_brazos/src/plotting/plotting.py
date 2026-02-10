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
from ..algorithms.softmax import Softmax

def get_algorithm_label(algo) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :return: Cadena descriptiva para el algoritmo.
    """
    label = type(algo).__name__
    
    if isinstance(algo, EpsilonGreedy):
        label += f" (epsilon={algo.epsilon})"
    
    elif isinstance(algo, Softmax):
        # Aquí añadimos la lógica para tu nuevo algoritmo
        label += f" (tau={algo.tau})"

    
    
    else:

        raise ValueError("El algoritmo debe ser de la clase Algorithm o una subclase.")
    
        
    return label

def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.
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
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        
        # Convertimos a porcentaje (0-100) si viene en formato 0.0-1.0
        data = optimal_selections[idx]
        if np.max(data) <= 1.0:
            data = data * 100
            
        plt.plot(range(steps), data, label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('% Selección Óptima', fontsize=14)
    plt.title('Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo', fontsize=16)
    plt.ylim(0, 105)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()