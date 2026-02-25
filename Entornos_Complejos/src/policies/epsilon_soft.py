import numpy as np

from .policy import Policy

class EpsilonSoftPolicy(Policy):
    """
    Implementa una política epsilon-soft para la selección de acciones según la definición:
    - Probabilidad de acción greedy: 1 - epsilon + (epsilon / |A(s)|)
    - Probabilidad de acción no greedy: epsilon / |A(s)|
    Siendo |A(s)| el número total de acciones disponibles en el estado s.
    """

    def __init__(self, epsilon, n_actions, seed=None):
        """
        :param epsilon: Tasa de exploración.
        :param n_actions: Cantidad de acciones posibles que el agente puede tomar.
        :param seed: Semilla para el generador de números aleatorios. Por defecto es None.
        """
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def get_action(self, state, q_values=None):
        """
        Selecciona una acción para un estado dado siguiendo la política epsilon-soft.

        :param state: Estado actual del agente.
        :param q_values: Estructura de datos que contiene los valores Q.
                                          Se espera que `q_values[state]` devuelva un array
                                          con los valores Q para todas las acciones en ese estado.
        :return: int: El índice de la acción seleccionada.
        """
        # Buscamos el valor más alto dentro del array Q
        max_q = np.max(q_values[state])
        # Obtenemos los índices de todas las acciones que tengan el valor Q máximo
        best_actions = np.where(q_values[state] == max_q)[0]
        # Obtenemos un índice aleatorio de entre todos los posibles
        # Esta será nuestra acción greedy
        greedy_action = self.rng.choice(best_actions)

        # Asignamos a todas las acciones la probabilidad base (epsilon / |A(s)|)
        # De esta forma, todas las acciones tendrán la probabilidad de acción no greedy
        probabilities = np.full(self.n_actions, self.epsilon / self.n_actions)
        
        # Sumamos a la acción codiciosa (1 - epsilon)
        # De esta forma, la cción codiciosa tendrá la probabilidad de acción greedy (1 - epsilon + (epsilon / |A(s)|))
        probabilities[greedy_action] += (1 - self.epsilon)

        # Seleccionamos una acción según las probabilidades calculadas
        return self.rng.choice(self.n_actions, p=probabilities)

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
    
    def get_action_probabilities(self, state, q_values=None):
        """
        Calcula la distribución de probabilidad completa para todas las acciones en un estado.

        :param state: (int o tuple): El estado actual del entorno.
        :param q_values: (dict o np.ndarray): Estructura de datos con los valores Q actuales.
        :return: np.ndarray: Un array de tamaño `n_actions` con la probabilidad de cada acción.
        """
        # Inicializamos todas las acciones con la probabilidad base (epsilon / |A(s)|)
        probabilities = np.ones(self.n_actions, dtype=float) * (self.epsilon / self.n_actions)

        # Buscamos el valor más alto dentro del array Q
        max_q = np.max(q_values[state])
        # Obtenemos los índices de todas las acciones que tengan el valor Q máximo
        best_actions = np.where(q_values[state] == max_q)[0]

        # Sumamos a las acciones codiciosas (1 - epsilon)
        # Por si hay varios empates, dividimos la parte greedy entre el número de acciones greedy
        probabilities[best_actions] += (1.0 - self.epsilon) / len(best_actions)

        return probabilities