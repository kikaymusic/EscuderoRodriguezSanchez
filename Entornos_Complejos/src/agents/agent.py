import gymnasium as gym
from abc import ABC, abstractmethod

class Agent(ABC):

    def __init__(self, env: gym.Env):
        """
        Inicializamos tod lo necesario para el aprendizaje
        """
        self.env = env
        self.training_rewards = []

    @abstractmethod
    def get_action(self, state):
        """
        Indicará qué acción realizar de acuerdo al estado.
        Responde a la política del agente.
        Construir tantas funciones de este tipo como políticas se quieran usar.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """
        Con la muestra (s, a, s', r) e información complementaria aplicamos el algoritmo.
        update() no es más que el algoritmo de aprendizaje del agente.
        Se añadirá lo necesario para obtener resultados estadísticos, evolución, etc ...
        """
        raise NotImplementedError("This method must be implemented by the subclass.")