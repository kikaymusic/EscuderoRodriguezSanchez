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
import pandas as pd
import matplotlib.pyplot as plt
from src.arms.bandit import Bandit
from ..algorithms.algorithm import Algorithm
from ..algorithms.epsilon_greedy import EpsilonGreedy
from ..algorithms.softmax import Softmax
from ..algorithms.ucb1 import UCB1

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

def get_algorithm_label(algo) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :return: Cadena descriptiva para el algoritmo.
    """

    label = type(algo).__name__

    if isinstance(algo, EpsilonGreedy):
        label += f" (epsilon={algo.epsilon})"

    elif isinstance(algo, UCB1):
        label += f" (c={algo.c})"
    
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
    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm]):
    """
    Función OBLIGATORIA.
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.
    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    plt.figure(figsize=(14, 7))
    for idx, algorithm in enumerate(algorithms):
        label = get_algorithm_label(algorithm)
        plt.plot(range(steps), optimal_selections[idx] * 100, label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('% Selección Óptima', fontsize=14)
    plt.title(f'Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim([0, 100])
    plt.tight_layout()
    plt.show()


def plot_arm_statistics(arm_stats: List[dict], algorithms: List[Algorithm], *args):
    """
    Función OPCIONAL.
    Genera gráficas separadas de Selección de Arms: Ganancias vs Pérdidas para cada algoritmo.
    :param arm_stats: Lista (de diccionarios) con estadísticas de cada brazo por algoritmo.
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param args: Opcional. Parámetros que consideres
    """
    num_algos = len(algorithms)

    # Creamos subplots: uno para cada algoritmo, dispuestos en fila
    fig, axes = plt.subplots(1, num_algos, figsize=(6 * num_algos, 5), sharey=True)

    # Aseguramos que axes sea una lista si solo hay un algoritmo
    if num_algos == 1:
        axes = [axes]

    for idx, algo in enumerate(algorithms):
        ax = axes[idx]
        label = get_algorithm_label(algo)

        steps = range(len(arm_stats[idx]["wins"]))

        # Graficamos Wins (Óptimo) y Losses (No óptimo)
        ax.plot(steps, arm_stats[idx]["wins"] * 100, label="Wins (Optimal)", color='green', linewidth=2)
        ax.plot(steps, arm_stats[idx]["losses"] * 100, label="Losses (Suboptimal)", color='red', linestyle='--',
                linewidth=2)

        ax.set_title(f"Algoritmo: {label}")
        ax.set_xlabel('Pasos de Tiempo')
        ax.set_ylim([-5, 105])  # Margen para ver bien el 0 y 100

        if idx == 0:
            ax.set_ylabel('% Selección de Brazo')

        ax.legend(loc='center right')

    plt.suptitle('Estadísticas de Selección por Algoritmo', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()


def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: List[Algorithm], *args):
    """
    Función OBLIGATORIA.
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo.
    :param steps: Número de pasos de tiempo.
    :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos).
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param args: Opcional. Parámetros que consideres. P.e. la cota teórica Cte * ln(T).
    """
    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), regret_accumulated[idx], label=label, linewidth=2)
    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Regret Acumulado', fontsize=14)
    plt.title(f'Regret Acumulado vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

#def plot_best_comparison(df_eps, df_softmax, df_ucb, distribution: str):
#    """
#    Genera un plot comparando el mejor parámetro de cada familia de algoritmos
#    (EpsilonGreedy, Softmax, UCB1) para una distribución dada, mostrando el
#    % de Selección Óptima a lo largo del tiempo.
#
#    :param df_eps:        DataFrame con los resultados de Epsilon-Greedy.
#    :param df_softmax:    DataFrame con los resultados de Softmax.
#    :param df_ucb:        DataFrame con los resultados de UCB1.
#    :param distribution:  Nombre de la distribución: 'Normal', 'Bernoulli' o 'Binomial'.
#    """
#    import numpy as np
#    import pandas as pd
#    import matplotlib.pyplot as plt
#    import seaborn as sns
#
#    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
#
#    dist = distribution.capitalize()
#
#    # ── Columnas de % selección óptima ──────────────────────────────────────
#    eps_cols   = {0:    f"{dist}_Opt_Eps_0",
#                  0.01: f"{dist}_Opt_Eps_0.01",
#                  0.1:  f"{dist}_Opt_Eps_0.1"}
#
#    tau_cols   = {0.1: f"{dist}_Opt_Tau_0.1",
#                  1:   f"{dist}_Opt_Tau_1",
#                  5:   f"{dist}_Opt_Tau_5"}
#
#    k_cols     = {0.1: f"{dist}_Opt_k_0.1",
#                  1:   f"{dist}_Opt_k_1",
#                  5:   f"{dist}_Opt_k_5"}
#
#    def _best_param(df, col_map):
#        """Devuelve (mejor_param, serie) según la media del último 20% de pasos."""
#        tail = max(1, len(df) // 5)
#        best_param = max(col_map, key=lambda p: df[col_map[p]].iloc[-tail:].mean())
#        return best_param, df[col_map[best_param]].values
#
#    best_eps,  series_eps  = _best_param(df_eps,     eps_cols)
#    best_tau,  series_tau  = _best_param(df_softmax, tau_cols)
#    best_k,    series_k    = _best_param(df_ucb,     k_cols)
#
#    steps = len(series_eps)
#
#    # ── Rango dinámico del eje Y ─────────────────────────────────────────────
#    # Calcula el máximo real de todas las series y añade un 5 % de margen
#    # para que las líneas cercanas al 100 % no queden pegadas al borde.
#    all_values = np.concatenate([series_eps, series_tau, series_k]) * 100
#    y_max = min(all_values.max() * 1.05, 110)   # nunca supera 110 %
#    y_min = max(all_values.min() * 0.95, 0)      # nunca baja de 0 %
#
#    # ── Plot ─────────────────────────────────────────────────────────────────
#    fig, ax = plt.subplots(figsize=(14, 6))
#
#    ax.plot(range(steps), series_eps * 100,
#            label=f"EpsilonGreedy  (ε={best_eps})", linewidth=2)
#    ax.plot(range(steps), series_tau * 100,
#            label=f"Softmax        (τ={best_tau})", linewidth=2)
#    ax.plot(range(steps), series_k * 100,
#            label=f"UCB1           (c={best_k})",   linewidth=2)
#
#    ax.set_title(f"Mejor Algoritmo por Familia — Distribución {dist}", fontsize=16)
#    ax.set_xlabel("Pasos de Tiempo", fontsize=13)
#    ax.set_ylabel("% Selección Óptima", fontsize=13)
#    ax.set_ylim([y_min, y_max])
#    ax.legend(title="Algoritmo (mejor parámetro)", fontsize=11)
#    plt.tight_layout()
#    plt.show()
#
#    # ── Resumen en consola ────────────────────────────────────────────────────
#    print(f"[{dist}]  Mejor EpsilonGreedy → ε={best_eps}  |  "
#          f"Mejor Softmax → τ={best_tau}  |  Mejor UCB1 → c={best_k}")



def plot_best_comparison(df_eps, df_softmax, df_ucb, distribution: str):
    """
    Genera un plot comparando el mejor parámetro de cada familia de algoritmos
    (EpsilonGreedy, Softmax, UCB1) para una distribución dada, mostrando el
    % de Selección Óptima a lo largo del tiempo.

    :param df_eps:        DataFrame con los resultados de Epsilon-Greedy.
    :param df_softmax:    DataFrame con los resultados de Softmax.
    :param df_ucb:        DataFrame con los resultados de UCB1.
    :param distribution:  Nombre de la distribución: 'Normal', 'Bernoulli' o 'Binomial'.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    dist = distribution.capitalize()

    # ── Columnas de % selección óptima ──────────────────────────────────────
    eps_opt_cols = {0:    f"{dist}_Opt_Eps_0",
                    0.01: f"{dist}_Opt_Eps_0.01",
                    0.1:  f"{dist}_Opt_Eps_0.1"}

    tau_opt_cols = {0.1: f"{dist}_Opt_Tau_0.1",
                    1:   f"{dist}_Opt_Tau_1",
                    5:   f"{dist}_Opt_Tau_5"}

    k_opt_cols   = {0.1: f"{dist}_Opt_k_0.1",
                    1:   f"{dist}_Opt_k_1",
                    5:   f"{dist}_Opt_k_5"}

    # ── Columnas de recompensa promedio ──────────────────────────────────────
    eps_rew_cols = {0:    f"{dist}_Reward_Eps_0",
                    0.01: f"{dist}_Reward_Eps_0.01",
                    0.1:  f"{dist}_Reward_Eps_0.1"}

    tau_rew_cols = {0.1: f"{dist}_Reward_Tau_0.1",
                    1:   f"{dist}_Reward_Tau_1",
                    5:   f"{dist}_Reward_Tau_5"}

    k_rew_cols   = {0.1: f"{dist}_Reward_k_0.1",
                    1:   f"{dist}_Reward_k_1",
                    5:   f"{dist}_Reward_k_5"}

    def _best_param(df, opt_col_map, rew_col_map):
        """
        Devuelve (mejor_param, serie_opt, serie_reward) eligiendo el parámetro
        con mayor media de % selección óptima en el último 20 % de pasos.
        """
        tail = max(1, len(df) // 5)
        best_param = max(opt_col_map, key=lambda p: df[opt_col_map[p]].iloc[-tail:].mean())
        return (best_param,
                df[opt_col_map[best_param]].values,
                df[rew_col_map[best_param]].values)

    best_eps, opt_eps, rew_eps = _best_param(df_eps,     eps_opt_cols, eps_rew_cols)
    best_tau, opt_tau, rew_tau = _best_param(df_softmax, tau_opt_cols, tau_rew_cols)
    best_k,   opt_k,   rew_k   = _best_param(df_ucb,     k_opt_cols,   k_rew_cols)

    steps = len(opt_eps)

    labels = [f"EpsilonGreedy  (ε={best_eps})",
              f"Softmax        (τ={best_tau})",
              f"UCB1           (c={best_k})"]

    # ── Rangos dinámicos ─────────────────────────────────────────────────────
    all_opt = np.concatenate([opt_eps, opt_tau, opt_k]) * 100
    opt_ymax = min(all_opt.max() * 1.05, 110)
    opt_ymin = max(all_opt.min() * 0.95, 0)

    all_rew = np.concatenate([rew_eps, rew_tau, rew_k])
    rew_ymax = all_rew.max() * 1.05
    rew_ymin = all_rew.min() * 0.95

    # ── Figura con dos subplots ───────────────────────────────────────────────
    fig, (ax_opt, ax_rew) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"Mejor Algoritmo por Familia — Distribución {dist}",
                 fontsize=16, y=1.02)

    colors = sns.color_palette("muted", 3)

    for series, label, color in zip(
            [opt_eps * 100, opt_tau * 100, opt_k * 100], labels, colors):
        ax_opt.plot(range(steps), series, label=label, linewidth=2, color=color)

    ax_opt.set_title("% Selección Óptima", fontsize=14)
    ax_opt.set_xlabel("Pasos de Tiempo", fontsize=12)
    ax_opt.set_ylabel("% Selección Óptima", fontsize=12)
    ax_opt.set_ylim([opt_ymin, opt_ymax])
    ax_opt.legend(title="Algoritmo (mejor parámetro)", fontsize=10)

    for series, label, color in zip(
            [rew_eps, rew_tau, rew_k], labels, colors):
        ax_rew.plot(range(steps), series, label=label, linewidth=2, color=color)

    ax_rew.set_title("Recompensa Promedio", fontsize=14)
    ax_rew.set_xlabel("Pasos de Tiempo", fontsize=12)
    ax_rew.set_ylabel("Recompensa Promedio", fontsize=12)
    ax_rew.set_ylim([rew_ymin, rew_ymax])
    ax_rew.legend(title="Algoritmo (mejor parámetro)", fontsize=10)

    plt.tight_layout()
    plt.show()

    # ── Resumen en consola ────────────────────────────────────────────────────
    print(f"[{dist}]  EpsilonGreedy → ε={best_eps}  |  "
          f"Softmax → τ={best_tau}  |  UCB1 → c={best_k}")
    




def _get_param_label(algo: Algorithm) -> str:
    """Devuelve el nombre y valor del parámetro propio de cada algoritmo."""
    if hasattr(algo, 'tau'):
        return f"τ={algo.tau}"
    elif hasattr(algo, 'epsilon'):
        return f"ε={algo.epsilon}"
    elif hasattr(algo, 'c'):
        return f"c={algo.c}"
    return ""
 
 
def plot_best_regret_comparison(
    bandidos: List[Bandit],
    rewards_list: List[np.ndarray],
    algorithms: List[Algorithm],
    dist_names: List[str],
):
    """
    Genera una gráfica comparando el mejor regret acumulado de cada distribución.
    Para cada distribución selecciona el algoritmo con menor regret acumulado final.
 
    :param bandidos: Lista de instancias de Bandit, una por distribución.
    :param rewards_list: Lista de matrices (n_algoritmos x steps) devueltas por run_experiment.
    :param algorithms: Lista de instancias de algoritmos (mismo orden que rewards_list).
    :param dist_names: Nombres de las distribuciones, e.g. ["Normal", "Binomial", "Bernoulli"].
    """
    styles = [
        {"color": "#2196F3", "linestyle": "-"},
        {"color": "#FF9800", "linestyle": "-"},
        {"color": "#4CAF50", "linestyle": "-"},
    ]
 
    steps = rewards_list[0].shape[1]
    plt.figure(figsize=(14, 7))
 
    for i, (bandido, rewards, dist_name) in enumerate(zip(bandidos, rewards_list, dist_names)):
        # mu* real, idéntico a calculate_regret
        mu_star = bandido.get_expected_value(bandido.optimal_arm)
 
        best_regret = None
        best_algo   = None
        best_final  = np.inf
 
        for j, algo in enumerate(algorithms):
            regret_accum = np.cumsum(mu_star - rewards[j])
            if regret_accum[-1] < best_final:
                best_final  = regret_accum[-1]
                best_regret = regret_accum
                best_algo   = algo
 
        param_label = _get_param_label(best_algo)
        algo_name   = type(best_algo).__name__
        label = f"{dist_name} (mejor {algo_name} {param_label})"
        plt.plot(range(steps), best_regret, label=label, linewidth=2.5, **styles[i])
 
    plt.xlabel("Pasos de Tiempo", fontsize=14)
    plt.ylabel("Regret Acumulado", fontsize=14)
    plt.title("Comparación del Mejor Regret por Distribución", fontsize=16)
    plt.legend(title="Distribución (mejor algoritmo)", bbox_to_anchor=(1.05, 1),
               loc="upper left", fontsize=11)
    plt.tight_layout()
    plt.show()