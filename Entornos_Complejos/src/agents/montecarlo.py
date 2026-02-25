from gymnasium.core import Env
import numpy as np
from .agent import Agent
from ..policies.policy import Policy


class AgentMonteCarlo(Agent):
    """
    Implementa el algoritmo de Control de Monte Carlo Off-policy con
    Muestreo de Importancia Ponderado (Weighted Importance Sampling).

    Es una versión de "todas las visitas" (every-visit), donde los valores Q
    se actualizan cada vez que se encuentra un par estado-acción en el episodio,
    no solo la primera vez.

    Attributes:
        env (Env): El entorno de Gymnasium con el que interactúa el agente.
        target_policy (Policy): La política que el agente intenta aprender y optimizar (pi).
        behavior_policy (Policy): La política que el agente usa para generar comportamiento
                                  y explorar el entorno (b). Si no se provee, asume On-policy.
        gamma (float): Factor de descuento para futuras recompensas.
        q_table (dict): Diccionario que almacena los valores Q estimados. Mapea estados a arrays de acciones.
        c_table (dict): Diccionario que almacena la suma acumulada de los pesos (W).
                        Crucial para el Weighted Importance Sampling.
        episode_memory (list): Buffer que guarda la trayectoria del episodio actual como tuplas (S_t, A_t, R_{t+1}).
    """
    def __init__(self, env: Env, target_policy: Policy, behavior_policy: Policy | None, gamma: float = 0.99):
        """
        Constructor del agente Monte Carlo Off-policy con Weighted Importance Sampling.
        :param env: Entorno de Gymnasium.
        :param target_policy: Política objetivo.
        :param behavior_policy: Política de comportamiento.
        :param gamma: Factor de descuento.
        """
        super().__init__(env)

        self.target_policy = target_policy
        # Si no se define behavior_policy, es On-policy
        self.behavior_policy = behavior_policy if behavior_policy else target_policy
        self.gamma = gamma

        # Inicializamos C-Table (para Importance Sampling en off-policy)
        self.c_table = {}

        # Buffer del episodio actual
        self.episode_memory = []


    
    def get_action(self, state):
        """
        Selecciona una acción basándose en la política de comportamiento (behavior_policy).
        :param state: El estado actual del entorno.
        :return: int: El índice de la acción a tomar.
        """
        
        # Ensure state exists in q_table before policy tries to access it
        if state not in self.q_table:
            n_actions = self.env.action_space.n
            self.q_table[state] = np.zeros(n_actions)
            self.c_table[state] = np.zeros(n_actions)
        
        return self.behavior_policy.get_action(state, self.q_table)


    def update(self, state, action, reward, next_state, done):
        """
        Recibe la transición actual del entorno y la almacena.
        Si el episodio ha terminado, dispara el proceso de aprendizaje.
        :param state: El estado actual del entorno.
        :param action: La acción tomada en el estado actual.
        :param reward: La recompensa recibida tras tomar la acción.
        :param next_state: El siguiente estado del entorno tras tomar la acción.
        :param done: Booleano que indica si el episodio ha terminado.
        :return: None
        """
        # 1. Guardamos la experiencia del paso actual
        self.episode_memory.append((state, action, reward))

        # 2. Monte Carlo SOLO aprende al final del episodio (todas las visitas AKA every-visit)
        if done:
            self._learn_from_episode()
            self.episode_memory = []  # Vaciamos la memoria para el siguiente episodio
        

    def _learn_from_episode(self):
        """
        Aprende de la experiencia acumulada en el episodio calculando el retorno hacia atrás y aplicando Weighted Importance Sampling.
        Se realiza recorriendo self.episode_memory hacia atrás (G = R + gamma * G).

        Si self.behavior_policy != self.target_policy:
            Calculas la variable rho (Importance Sampling) usando el métod get_probability()
            de ambas políticas.

        :return:
        """
        G = 0.0  # Retorno acumulado (Suma de recompensas descontadas)
        W = 1.0  # Peso del Importance Sampling (rho). Comienza en 1.0 al final del episodio.
        # Recorremos la experiencia del episodio de atrás hacia adelante (reversed)
        for state, action, reward in reversed(self.episode_memory):

            # TODO revisar, no tiene sentido aprender de un episodio y que su estado no exista
            # --- Inicialización perezosa (Lazy initialization) ---
            # Si es la primera vez que visitamos este estado, lo creamos en las tablas
            if state not in self.q_table:
                n_actions = self.env.action_space.n  # Asumimos espacio de acciones discreto
                self.q_table[state] = np.zeros(n_actions)
                self.c_table[state] = np.zeros(n_actions)

            # 1. Calculamos el retorno descontado G
            G = self.gamma * G + reward
            # 2. Actualizamos la suma acumulada de los pesos (C-Table)
            self.c_table[state][action] += W

            # 3. Actualizamos el valor Q (Media móvil ponderada  por el peso W)
            # Formula: Q(s,a) = Q(s,a) + (W / C(s,a)) * [G - Q(s,a)]
            # El factor (W / C) actúa como la tasa de aprendizaje (alpha)
            self.q_table[state][action] += (W / self.c_table[state][action]) * (G - self.q_table[state][action])

            # 4. Calculamos las probabilidades de haber tomado esa acción bajo ambas políticas

            pi_prob = self.target_policy.get_probability(state, action, self.q_table)
            b_prob = self.behavior_policy.get_probability(state, action, self.q_table)

            # 5. Actualizamos el peso W para el paso anterior (que procesaremos en la siguiente iteración)
            W = W * (pi_prob / b_prob)

            # 6. Optimización vital para Off-policy:
            # Si W se vuelve 0, significa que la política objetivo NUNCA habría tomado esa acción. No tiene sentido
            # seguir hacia atrás porque el peso de toda la experiencia anterior será 0. Rompemos el bucle para ahorrar cómputo.
            if W == 0.0:
                break
