import numpy as np
from gymnasium.core import Env
from .agent import Agent
from ..policies.policy import Policy


class AgentSarsaSemiGradient(Agent):
    """
    Implementación del agente SARSA Semi-Gradiente para aprendizaje por refuerzo
    con aproximación de funciones lineales.

    A diferencia del SARSA tabular, este agente utiliza aproximación de funciones
    para estimar los valores Q, lo que permite trabajar con espacios de estados
    continuos o muy grandes.

    La actualización semi-gradiente se realiza mediante:
    w <- w + alpha * [R + gamma * q_hat(S', A', w) - q_hat(S, A, w)] * grad_w[q_hat(S, A, w)]

    Donde:
    - w: vector de pesos de la aproximación lineal
    - q_hat(S, A, w): aproximación del valor Q(S, A) = w^T * phi(S, A)
    - phi(S, A): vector de características (features) del par estado-acción
    - grad_w[q_hat(S, A, w)] = phi(S, A) (para aproximación lineal)

    Attributes:
        env (Env): El entorno de Gymnasium con el que interactúa el agente.
        policy (Policy): La política que el agente usa (SARSA es On-Policy).
        alpha (float): Tasa de aprendizaje.
        gamma (float): Factor de descuento para futuras recompensas.
        weights (np.ndarray): Vector de pesos para la aproximación lineal.
        feature_extractor (callable): Función que extrae características del par (estado, acción).
    """

    def __init__(self, env: Env, policy: Policy, feature_extractor,
                 n_features: int, alpha: float = 0.01, gamma: float = 0.99):
        """
        Constructor del agente SARSA Semi-Gradiente.

        :param env: Entorno de Gymnasium.
        :param policy: Política del agente. Como SARSA es On-Policy, solo necesitamos una.
        :param feature_extractor: Función que toma (state, action, env) y devuelve un vector de características.
                                  Debe retornar un numpy array de tamaño n_features.
        :param n_features: Número de características en el vector de features.
        :param alpha: Tasa de aprendizaje (step-size).
        :param gamma: Factor de descuento.
        """
        super().__init__(env)
        self.policy = policy
        self.alpha = alpha
        self.gamma = gamma
        self.feature_extractor = feature_extractor
        self.n_features = n_features

        # Inicializamos el vector de pesos con valores pequeños aleatorios
        # o con ceros (depende de la preferencia)
        self.weights = np.zeros(n_features)

        # Cache para almacenar los valores Q aproximados de cada acción
        # Esto es útil para la política y para evitar recalcular
        self._q_cache = {}

    def _get_features(self, state, action):
        """
        Extrae el vector de características para un par estado-acción.

        :param state: Estado del entorno.
        :param action: Acción a tomar.
        :return: Vector de características (numpy array).
        """
        return self.feature_extractor(state, action, self.env)

    def _get_q_value(self, state, action):
        """
        Calcula el valor Q aproximado para un par estado-acción.
        q_hat(s, a) = w^T * phi(s, a)

        :param state: Estado del entorno.
        :param action: Acción a tomar.
        :return: Valor Q aproximado (float).
        """
        features = self._get_features(state, action)
        return np.dot(self.weights, features)

    def _get_all_q_values(self, state):
        """
        Calcula los valores Q aproximados para todas las acciones posibles en un estado.

        :param state: Estado del entorno.
        :return: Array con los valores Q de todas las acciones.
        """
        n_actions = self.env.action_space.n
        q_values = np.zeros(n_actions)

        for action in range(n_actions):
            q_values[action] = self._get_q_value(state, action)

        return q_values

    def get_action(self, state):
        """
        Selecciona una acción siguiendo la política actual.

        Para que la política funcione correctamente, creamos un diccionario temporal
        que simula la q_table pero con los valores aproximados.

        :param state: Estado actual del entorno.
        :return: Acción seleccionada (int).
        """
        # Calculamos los valores Q para todas las acciones
        q_values = self._get_all_q_values(state)

        # Convertimos el estado a tupla para que sea hashable (necesario para diccionarios)
        # Si el estado ya es hashable (int, tuple), lo usamos directamente
        try:
            state_key = tuple(state) if hasattr(state, '__iter__') and not isinstance(state, (str, tuple)) else state
        except TypeError:
            state_key = state

        # Creamos un diccionario temporal para que la política pueda acceder a los valores Q
        # Esto mantiene la compatibilidad con las políticas existentes
        temp_q_table = {state_key: q_values}

        # Obtenemos la acción según la política
        return self.policy.get_action(state_key, temp_q_table)

    def update(self, state, action, reward, next_state, done):
        """
        Actualiza los pesos usando el algoritmo SARSA Semi-Gradiente.

        La actualización se realiza mediante:
        w <- w + alpha * [R + gamma * q_hat(S', A', w) - q_hat(S, A, w)] * phi(S, A)

        Donde:
        - phi(S, A) es el vector de características del par (S, A)
        - q_hat(S, A, w) es la aproximación del valor Q
        - A' es la siguiente acción seleccionada por la política en S'

        :param state: Estado actual (S).
        :param action: Acción tomada (A).
        :param reward: Recompensa recibida (R).
        :param next_state: Siguiente estado (S').
        :param done: Booleano que indica si el episodio ha terminado.
        :return: La siguiente acción (A') para mantener compatibilidad con el flujo SARSA.
        """
        # Obtenemos el vector de características para el par (S, A)
        features = self._get_features(state, action)

        # Calculamos q_hat(S, A, w)
        current_q = self._get_q_value(state, action)

        # Si el episodio ha terminado, no hay siguiente acción
        if done:
            next_action = None
            next_q = 0.0
        else:
            # Seleccionamos la siguiente acción A' usando la política
            next_action = self.get_action(next_state)
            # Calculamos q_hat(S', A', w)
            next_q = self._get_q_value(next_state, next_action)

        # Calculamos el error TD (TD error)
        # td_error = R + gamma * q_hat(S', A', w) - q_hat(S, A, w)
        td_error = reward + self.gamma * next_q - current_q

        # Actualizamos los pesos usando el gradiente semi-gradiente
        # w <- w + alpha * td_error * phi(S, A)
        # Para aproximación lineal, el gradiente de q_hat respecto a w es simplemente phi(S, A)
        self.weights += self.alpha * td_error * features

        # Devolvemos la siguiente acción (útil para el flujo de control en algunos casos)
        return next_action

    def get_weights(self):
        """
        Devuelve una copia del vector de pesos actual.

        :return: Copia del vector de pesos.
        """
        return self.weights.copy()

    def set_weights(self, weights):
        """
        Establece el vector de pesos.

        :param weights: Nuevo vector de pesos.
        """
        if len(weights) != self.n_features:
            raise ValueError(f"El vector de pesos debe tener {self.n_features} elementos.")
        self.weights = np.array(weights)

    def reset_weights(self):
        """
        Reinicia los pesos a ceros.
        """
        self.weights = np.zeros(self.n_features)
