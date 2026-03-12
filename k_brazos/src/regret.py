import numpy as np

from src.arms import Bandit


def calculate_regret(bandido: Bandit, rewards: np.ndarray) -> np.ndarray:
    """
    Calcula el regret acumulado empírico a partir de las recompensas promedio.
    Funciona para CUALQUIER algoritmo y CUALQUIER distribución.

    :param bandido: El entorno del bandido instanciado.
    :param rewards: Matriz NumPy de recompensas promedio devuelta por run_experiment (algoritmos x pasos).
    :return: Matriz NumPy de regret acumulado (algoritmos x pasos).
    """
    # 1. Obtener la media teórica del brazo óptimo (mu*)
    brazo_optimo = bandido.optimal_arm
    mu_star = bandido.get_expected_value(brazo_optimo)

    # 2. Calcular el regret instantáneo empírico en cada paso
    # (Lo que deberías haber ganado de media - lo que ganaste de media)
    regret_instantaneo = mu_star - rewards

    # 3. El regret acumulado es la suma acumulativa a lo largo del tiempo (axis=1)
    regret_accumulated = np.cumsum(regret_instantaneo, axis=1)

    return regret_accumulated