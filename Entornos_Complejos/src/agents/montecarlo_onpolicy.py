import numpy as np


class AgentMonteCarloEveryVisitOnPolicy:
    def __init__(self, env, epsilon=0.4, decay=False, discount_factor=1.0, seed=100):
        self.env = env
        self.nA = env.action_space.n
        self.nS = env.observation_space.n

        self.epsilon_init = float(epsilon)
        self.epsilon = float(epsilon)
        self.decay = bool(decay)
        self.gamma = float(discount_factor)

        # Q(s,a) y contador de visitas para promedio incremental
        self.Q = np.zeros((self.nS, self.nA), dtype=float)
        self.n_visits = np.zeros((self.nS, self.nA), dtype=float)

        # Buffer del episodio: (s, a, r)
        self.episode = []

        # Estadísticas
        self.episode_returns = []
        self.episode_lengths = []
        self.seed = seed

    def _epsilon_schedule(self, t):
        # Misma idea que en tu código: epsilon grande al inicio, va bajando
        # Ojo: si quieres, puedes cambiar la fórmula, pero que sea determinista.
        return min(1.0, 1000.0 / (t + 1))

    def get_action(self, state, t=0):
        if self.decay:
            self.epsilon = self._epsilon_schedule(t)
        else:
            self.epsilon = self.epsilon_init

        pi_A = random_epsilon_greedy_policy(self.Q, self.epsilon, state, self.nA)
        action = np.random.choice(np.arange(self.nA), p=pi_A)
        return action

    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        # Guardamos transición del episodio
        self.episode.append((obs, action, reward))

        done = terminated or truncated
        if not done:
            return

        # Al terminar: Monte Carlo hacia atrás
        G = 0.0
        for (s, a, r) in reversed(self.episode):
            G = r + self.gamma * G
            self.n_visits[s, a] += 1.0
            alpha = 1.0 / self.n_visits[s, a]
            self.Q[s, a] += alpha * (G - self.Q[s, a])

        # Estadísticas del episodio
        ep_return = sum([tr[2] * (self.gamma ** i) for i, tr in enumerate(self.episode)])
        self.episode_returns.append(ep_return)
        self.episode_lengths.append(len(self.episode))

        # Reset del buffer
        self.episode = []

    def stats(self):
        # Devuelve series para graficar
        returns = np.array(self.episode_returns, dtype=float)
        lengths = np.array(self.episode_lengths, dtype=float)

        if len(returns) == 0:
            return {
                "returns": [],
                "returns_mean": [],
                "lengths": [],
                "lengths_mean": []
            }

        returns_mean = np.cumsum(returns) / (np.arange(len(returns)) + 1)
        lengths_mean = np.cumsum(lengths) / (np.arange(len(lengths)) + 1)

        return {
            "returns": returns.tolist(),
            "returns_mean": returns_mean.tolist(),
            "lengths": lengths.tolist(),
            "lengths_mean": lengths_mean.tolist()
        }