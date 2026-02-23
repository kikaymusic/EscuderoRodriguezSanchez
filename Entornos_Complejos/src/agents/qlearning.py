import numpy as np
from gymnasium.core import Env
from .agent import Agent
from Entornos_Complejos.src.policies.policy import Policy

class AgentQLearning(Agent):
    """
    Implementación del agente Q-Learning para aprendizaje por refuerzo.
    Al ser un algoritmo Off-Policy, necesitaremos dos políticas diferentes.
    """

    def __init__(self, env: Env, target_policy: Policy, behavior_policy: Policy, alpha: float = 0.1, gamma: float = 0.99):
        """
        :param env: Entorno de Gymnasium.
        :param behavior_policy: Política de comportamiento.
        :param target_policy: Política objetivo.
        :param alpha: Tasa de aprendizaje.
        :param gamma: Factor de descuento.
        """
        super().__init__(env)
        self.behavior_policy = behavior_policy
        self.target_policy = target_policy
        self.alpha = alpha
        self.gamma = gamma

    def get_action(self, state):
        """
        Obtiene una acción siguiendo la política actual.
        """
        # Aseguramos que el estado exista en la tabla Q antes de obtener la acción
        self._ensure_state_exists(state)
        # Obtenemos la acción según la política de comportamiento y los valores Q actuales
        return self.behavior_policy.get_action(state, self.q_table)

    def update(self, state, action, reward, next_state, done):
        """
        Función de actualización de Q-Learning, utilizando la función:
        Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', a) - Q(S, A)]
        Siendo:
        - S: estado actual
        - A: acción tomada
        - R: recompensa recibida
        - S': siguiente estado
        - a: acción con el valor Q más alto en S' (seleccionada por la política target)
        
        :param state: Estado actual (S).
        :param action: Acción tomada (A).
        :param reward: Recompensa recibida (R).
        :param next_state: Siguiente estado (S').
        :param done: Booleano que indica si el episodio ha terminado.
        """
        # Nos aseguramos que los estados existan en la tabla Q
        self._ensure_state_exists(state)
        self._ensure_state_exists(next_state)

        # Si estamos en un estado terminal, el valor esperado (Q(S', a)) es solo la recompensa recibida
        if done:
            target_value = 0.0
        # Si no, calculamos el valor esperado (Q(S', a)) usando la política target
        else:
            # Obtenemos las probabilidades de cada acción con la política target
            target_probs = self.target_policy.get_action_probabilities(next_state, self.q_table)
            # Calculamos el valor esperado usando la política target
            target_value = np.dot(target_probs, self.q_table[next_state])

        # Obtenemos el valor de Q(S, A)
        current_q = self.q_table[state][action]
        # Calculamos el valor de la formula [R + gamma * Q(S', a) - Q(S, A)]
        td_target = reward + self.gamma * target_value - current_q
        # Actualizamos el valor de Q(S, A) siguiendo la formula:
        # Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', a) - Q(S, A)]
        self.q_table[state][action] += self.alpha * (td_target - self.q_table[state][action])

    def _ensure_state_exists(self, state):
        """
        Función para asegurar que un estado exista en la tabla Q. Si no existe, lo inicializa con un array de ceros.
        """
        # Comprobamos si el estado no existe en la tabla Q
        if state not in self.q_table:
            # Creamos un array de ceros para las N acciones posibles
            self.q_table[state] = np.zeros(self.env.action_space.n)