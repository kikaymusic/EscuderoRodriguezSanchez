import numpy as np

from Entornos_Complejos.src.policies.policy import Policy


# def random_epsilon_greedy_policy(Q, epsilon, state, nA):
#    pi_A = np.ones(nA, dtype=float) * epsilon / nA
#    best_action = np.argmax(Q[state])
#    pi_A[best_action] += (1.0 - epsilon)
#    return pi_A
#
## Política epsilon-greedy a partir de una epsilon-soft
# def epsilon_greedy_policy(Q, epsilon, state, nA):
#    pi_A = random_epsilon_greedy_policy(Q, epsilon, state, nA)
#    return np.random.choice(np.arange(nA), p=pi_A)


class EpsilonGreedyPolicy(Policy):
    """
    Implementa una política Epsilon-Greedy para la selección de acciones en Aprendizaje por Refuerzo.

    Esta política equilibra la exploración y la explotación: con una probabilidad 'epsilon'
    selecciona una acción de forma completamente aleatoria (exploración), y con una
    probabilidad '1 - epsilon' selecciona la mejor acción conocida según los valores Q
    (explotación).

    Attributes:
        epsilon (float): La probabilidad de elegir una acción aleatoria (rango [0.0, 1.0]).
        n_actions (int): El número total de acciones discretas disponibles en el entorno.
        rng (numpy.random.Generator): Generador de números aleatorios para reproducibilidad.
    """

    def __init__(self, epsilon, n_actions, seed=None):
        """

        :param epsilon: Tasa de exploración (0.0 significa 100% greedy, 1.0 significa 100% aleatorio).
        :param n_actions: Cantidad de acciones posibles que el agente puede tomar.
        :param seed: Semilla para el generador de números aleatorios. Por defecto es None.
        """

        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def get_action(self, state, q_values=None):
        """
        Selecciona una acción para un estado dado siguiendo la política epsilon-greedy.

        :param state: Estado actual del agente.
        :param q_values: Estructura de datos que contiene los valores Q.
                                          Se espera que `q_values[state]` devuelva un array
                                          con los valores Q para todas las acciones en ese estado.
        :return: int: El índice de la acción seleccionada.
        """

        # Exploración aleatoria con probabilidad epsilon
        if self.rng.random() < self.epsilon:
            return self.rng.integers(self.n_actions)

        # Explotación: Selección de la acción con el mayor valor Q.
        # Funcionalidad para romper empates de forma aleatoria.
        # np.argmax siempre devuelve el primer índice si hay empate (ej. todos los Q son 0 al inicio),
        # lo que introduce un sesgo hacia la acción 0.
        max_q = np.max(q_values[state])
        best_actions = np.where(q_values[state] == max_q)[0]
        return self.rng.choice(best_actions)

    def get_probability(self, state, action, q_values=None):
        """
        Calcula la probabilidad de que una acción específica sea seleccionada en un estado dado.

        La probabilidad se define matemáticamente como:
        - Si la acción es la mejor (greedy): ((1 - epsilon) / num_mejores_acciones) + (epsilon / total_acciones)
        - Si la acción no es la mejor: epsilon / total_acciones
        :param state: (int o tuple): El estado actual del entorno.
        :param action: (int): La acción cuya probabilidad se desea calcular.
        :param q_values: (dict o np.ndarray): Estructura de datos con los valores Q actuales.
        :return: float: La probabilidad de seleccionar la acción dada en el estado dado, rango [0.0, 1.0].
        """

        max_q = np.max(q_values[state])
        best_actions = np.where(q_values[state] == max_q)[0]

        # Si la acción es una de las mejores, tiene la prob de ser elegida greedy + su parte aleatoria
        if action in best_actions:
            return ((1 - self.epsilon) / len(best_actions)) + (self.epsilon / self.n_actions)

        # Si no, solo tiene la probabilidad de ser elegida por exploración
        return self.epsilon / self.n_actions
