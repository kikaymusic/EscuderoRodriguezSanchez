import numpy as np

from Entornos_Complejos.src.policies.policy import Policy

class EpsilonGreedyPolicy(Policy):
    """
    Implementa una política Epsilon-Greedy para la selección de acciones según la definición:
    - Probabilidad de acción greedy (arg max Q(s, a)): 1 - epsilon + (epsilon / |A(s)|)
    - Probabilidad de acción no greedy: epsilon / |A(s)|
    Siendo |A(s)| el número total de acciones disponibles en el estado s.

    Esta política esta basada en epsilon-soft, pero con la diferencia de que en epsilon-greedy tenemos una probabilidad
    aleatoría de seleccionar otra acción que no sea la greedy (exploración de manera aleatoria).
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

        # Buscamos el valor más alto dentro del array Q
        max_q = np.max(q_values[state])
        # Obtenemos los índices de todas las acciones que tengan el valor Q máximo
        best_actions = np.where(q_values[state] == max_q)[0]
        # Seleccionamos aleatoriamente entre las acciones que tengan el valor Q máximo
        return self.rng.choice(best_actions)

    def get_probability(self, state, action, q_values=None):
        """
        Calcula la probabilidad de que una acción específica sea seleccionada en un estado dado.

        :param state: (int o tuple): El estado actual del entorno.
        :param action: (int): La acción cuya probabilidad se desea calcular.
        :param q_values: (dict o np.ndarray): Estructura de datos con los valores Q actuales.
        :return: float: La probabilidad de seleccionar la acción dada en el estado dado, rango [0.0, 1.0].
        """
        # Buscamos el valor más alto dentro del array Q
        max_q = np.max(q_values[state])
        # Obtenemos los índices de todas las acciones que tengan el valor Q máximo
        best_actions = np.where(q_values[state] == max_q)[0]

        # Comprobamos si la acción pasada como parámetro es una de las acciones greedy
        if action in best_actions:
            # Devolvemos la probabilidad de acción greedy (1 - epsilon + (epsilon / |A(s)|))
            # Por si hay varios empates, dividimos la parte greedy entre el número de acciones greedy
            return ((1 - self.epsilon) / len(best_actions)) + (self.epsilon / self.n_actions)

        # Si no es una acción greedy, devolvemos la probabilidad básica de acción no greedy (epsilon / |A(s)|)
        return self.epsilon / self.n_actions
