from gymnasium.core import Env
import numpy as np
from .agent import Agent
#from Entornos_Complejos.src.policies.epsilon_greedy import random_epsilon_greedy_policy


class AgentMonteCarlo(Agent):
    #def __init__(self, env: Env, epsilon=0.4, decay=False, discount_factor=1.0, seed=100):
    def __init__(self, env: Env, target_policy, behavior_policy, gamma=0.99):
        super().__init__(env)

        self.target_policy = target_policy
        # Si no se define behavior_policy, es On-policy
        self.behavior_policy = behavior_policy if behavior_policy else target_policy
        self.gamma = gamma

        # Inicializamos Q-Table y C-Table (esta última es para Importance Sampling en off-policy)
        # Asumiendo espacios discretos por el momento
        self.q_table = {}
        self.c_table = {}

        # Buffer del episodio actual
        self.episode_memory = []
        
        #self.nA = env.action_space.n
        #self.nS = env.observation_space.n
#
        #self.epsilon_init = float(epsilon)
        #self.epsilon = float(epsilon)
        #self.decay = bool(decay)
        #self.gamma = float(discount_factor)
#
        ## Q(s,a) y contador de visitas para promedio incremental
        #self.Q = np.zeros((self.nS, self.nA), dtype=float)
        #self.n_visits = np.zeros((self.nS, self.nA), dtype=float)
#
        ## Buffer del episodio: (s, a, r)
        #self.episode = []
#
        ## Estadísticas
        #self.episode_returns = []
        #self.episode_lengths = []
        #self.seed = seed

    def get_action(self, state):
        # Siempre nos movemos en el entorno usando la behavior_policy
        return self.behavior_policy.get_action(state, self.q_table)

    def update(self, state, action, reward, next_state, done):
        # 1. Guardamos la experiencia del paso actual
        self.episode_memory.append((state, action, reward))

        # 2. Monte Carlo SOLO aprende al final del episodio
        if done:
            self._learn_from_episode()
            self.episode_memory = []  # Vaciamos la memoria para el siguiente episodio

    def _learn_from_episode(self):
        """
        Recorres self.episode_memory hacia atrás (G = R + gamma * G).

        Si self.behavior_policy != self.target_policy:
            Calculas la variable rho (Importance Sampling) usando el método get_probability()
            de ambas políticas.

        :return:
        """
        G = 0.0  # Retorno acumulado
        W = 1.0  # Peso del Importance Sampling (rho)
        # Recorremos la experiencia del episodio de atrás hacia adelante (reversed)
        for state, action, reward in reversed(self.episode_memory):

            # 1. Calculamos el retorno descontado G
            G = self.gamma * G + reward

            # --- Inicialización perezosa (Lazy initialization) ---
            # Si es la primera vez que visitamos este estado, lo creamos en las tablas
            if state not in self.q_table:
                n_actions = self.env.action_space.n  # Asumimos espacio de acciones discreto
                self.q_table[state] = np.zeros(n_actions)
                self.c_table[state] = np.zeros(n_actions)

            # 2. Actualizamos la suma acumulada de los pesos (C-Table)
            self.c_table[state][action] += W

            # 3. Actualizamos el valor Q (Media móvil ponderada)
            # Formula: Q(s,a) = Q(s,a) + (W / C(s,a)) * [G - Q(s,a)]
            self.q_table[state][action] += (W / self.c_table[state][action]) * (G - self.q_table[state][action])

            # 4. Calculamos las probabilidades de haber tomado esa acción bajo ambas políticas
            pi_prob = self.target_policy.get_probability(state, action, self.q_table)
            b_prob = self.behavior_policy.get_probability(state, action, self.q_table)

            # 5. Actualizamos el peso W para el paso anterior (que procesaremos en la siguiente iteración)
            W = W * (pi_prob / b_prob)

            # 6. Optimización vital para Off-policy:
            # Si W se vuelve 0, significa que la política objetivo NUNCA habría tomado
            # esa acción. No tiene sentido seguir hacia atrás porque el peso de todo lo
            # anterior será 0. Rompemos el bucle para ahorrar cómputo.
            if W == 0.0:
                break
#   
    # def _epsilon_schedule(self, t):
    #     # Misma idea que en tu código: epsilon grande al inicio, va bajando
    #     # Ojo: si quieres, puedes cambiar la fórmula, pero que sea determinista.
    #     return min(1.0, 1000.0 / (t + 1))
# 
    # def get_action(self, state, t=0):
    #     if self.decay:
    #         self.epsilon = self._epsilon_schedule(t)
    #     else:
    #         self.epsilon = self.epsilon_init
# 
    #     pi_A = random_epsilon_greedy_policy(self.Q, self.epsilon, state, self.nA)
    #     action = np.random.choice(np.arange(self.nA), p=pi_A)
    #     return action
# 
    # def update(self, obs, action, next_obs, reward, terminated, truncated, info):
    #     # Guardamos transición del episodio
    #     self.episode.append((obs, action, reward))
# 
    #     done = terminated or truncated
    #     if not done:
    #         return
# 
    #     # Al terminar: Monte Carlo hacia atrás
    #     G = 0.0
    #     for (s, a, r) in reversed(self.episode):
    #         G = r + self.gamma * G
    #         self.n_visits[s, a] += 1.0
    #         alpha = 1.0 / self.n_visits[s, a]
    #         self.Q[s, a] += alpha * (G - self.Q[s, a])
# 
    #     # Estadísticas del episodio
    #     ep_return = sum([tr[2] * (self.gamma ** i) for i, tr in enumerate(self.episode)])
    #     self.episode_returns.append(ep_return)
    #     self.episode_lengths.append(len(self.episode))
# 
    #     # Reset del buffer
    #     self.episode = []
# 
    # def stats(self):
    #     # Devuelve series para graficar
    #     returns = np.array(self.episode_returns, dtype=float)
    #     lengths = np.array(self.episode_lengths, dtype=float)
# 
    #     if len(returns) == 0:
    #         return {
    #             "returns": [],
    #             "returns_mean": [],
    #             "lengths": [],
    #             "lengths_mean": []
    #         }
# 
    #     returns_mean = np.cumsum(returns) / (np.arange(len(returns)) + 1)
    #     lengths_mean = np.cumsum(lengths) / (np.arange(len(lengths)) + 1)
# 
    #     return {
    #         "returns": returns.tolist(),
    #         "returns_mean": returns_mean.tolist(),
    #         "lengths": lengths.tolist(),
    #         "lengths_mean": lengths_mean.tolist()
    #     }