import numpy as np
from gymnasium.core import Env
from .agent import Agent
from ..policies import Policy

class AgentSarsa(Agent):
    """
    Implementación del agente SARSA para aprendizaje por refuerzo. 
    """

    def __init__(self, env: Env, policy: Policy, alpha: float = 0.1, gamma: float = 0.99):
        """
        :param env: Entorno de Gymnasium.
        :param policy: Política del agente. Como SARSA es On-Policy, solo necesitamos una.
        :param alpha: Tasa de aprendizaje.
        :param gamma: Factor de descuento.
        """
        super().__init__(env)
        self.policy = policy
        self.alpha = alpha
        self.gamma = gamma

    def get_action(self, state):
        """
        Obtiene una acción siguiendo la política actual.
        """
        # Aseguramos que el estado exista en la tabla Q antes de obtener la acción
        self._ensure_state_exists(state)
        # Obtenemos la acción según la política y los valores Q actuales
        return self.policy.get_action(state, self.q_table)

    def update(self, state, action, reward, next_state, done):
        """
        Función de actualización de SARSA, utilizando la función:
        Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
        Siendo:
        - S: estado actual
        - A: acción tomada
        - R: recompensa recibida
        - S': siguiente estado
        - A': siguiente acción tomada en S' (seleccionada por la política)
        
        :param state: Estado actual (S).
        :param action: Acción tomada (A).
        :param reward: Recompensa recibida (R).
        :param next_state: Siguiente estado (S').
        :param done: Booleano que indica si el episodio ha terminado.
        """
        # Nos aseguramos que los estados existan en la tabla Q
        self._ensure_state_exists(state)
        self._ensure_state_exists(next_state)

        # Si estamos en un estado terminal, no tomamos ninguna acción
        if done:
            next_action = None
            next_q = 0.0
        # Si no, seleccionamos la acción A' para el siguiente estado S' y obtenemos su valor Q
        else:
            # Elegimos A' para el siguiente estado S'
            next_action = self.get_action(next_state)
            # Obtenemos el valor Q(S', A')
            next_q = self.q_table[next_state][next_action]

        # Obtenemos el valor de Q(S, A)
        current_q = self.q_table[state][action]
        # Calculamos el valor de la formula [R + gamma * Q(S', A') - Q(S, A)]
        td_target = reward + self.gamma * next_q - current_q
        # Actualizamos el valor de Q(S, A) siguiendo la formula:
        # Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
        self.q_table[state][action] += self.alpha * td_target

        # Devolvemos la acción elegida
        return next_action

    def _ensure_state_exists(self, state):
        """
        Función para asegurar que un estado exista en la tabla Q. Si no existe, lo inicializa con un array de ceros.
        """
        # Comprobamos si el estado no existe en la tabla Q
        if state not in self.q_table:
            # Creamos un array de ceros para las N acciones posibles
            self.q_table[state] = np.zeros(self.env.action_space.n)