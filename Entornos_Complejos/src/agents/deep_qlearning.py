import numpy as np
from gymnasium.core import Env
from .agent import Agent
from ..policies.policy import Policy
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class AgentDeepQLearning(Agent):
    """
    Implementación del agente Deep Q-Learning con Experience Replay para aprendizaje por refuerzo.
    Extensión de Q-Learning la cual, en vez de utilizar la tabla Q para almacenar los valores de acción-estado, utilizamos una red neuronal para aproximar la función Q.
    En esta implementación introducimos una memoria de replay, para ...
   
    Attributes:
        env (Env): El entorno de Gymnasium con el que interactúa el agente.
        behavior_policy (Policy): Política de comportamiento.
        gamma (float): Factor de descuento para futuras recompensas.
        model (nn.Module): Red neuronal de PyTorch que actúa como aproximador de Q.
        optimizer (torch.optim.Optimizer): Optimizador para actualizar los pesos de la red.
        criterion (nn.Module): Función de pérdida para el entrenamiento de la red.
        memory (deque): Memoria de replay (D).
        batch_size (int): Tamaño del minibatch para el entrenamiento.
    """

    def __init__(self, env: Env, behavior_policy: Policy, q_network: nn.Module, lr: float = 1e-3, buffer_size: int = 10000, batch_size: int = 64, gamma: float = 0.99):
        """
        :param env: Entorno de Gymnasium.
        :param behavior_policy: Política de comportamiento.
        :param q_network: Red neuronal de PyTorch que actúa como aproximador de Q.
        :param lr: Learning rate para la red neuronal.
        :param batch_size: Tamaño del minibatch para el entrenamiento.
        :param buffer_size: Capacidad N de la memoria de replay (D).
        :param gamma: Factor de descuento.
        """
        super().__init__(env)
        # Inicializamos la tabla Q vacia, ya que no la utilizaremos
        self.q_table = None 
        # Política de comportamiento
        self.behavior_policy = behavior_policy
        # Factor de descuento
        self.gamma = gamma

        # Red neuronal para aproximar la función Q
        self.model = q_network
        # Optimizador y función de pérdida para entrenar la red
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        # Tamaño del minibatch para el entrenamiento
        self.batch_size = batch_size

        # Inicializamos la memoria de replay D con el tamaño especificado N
        self.memory = deque(maxlen=buffer_size)


    def get_action(self, state):
        """
        Obtiene una acción siguiendo la política actual y la red neuronal.
        """
        # Convertimos el estado a tensor para la red
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Desactivamos el cálculo de gradientes
        with torch.no_grad():
            # Predecimos los valores de Q para todas las acciones
            q_values = self.model(state_tensor).numpy()
            # Creamos un diccionario falso para el formato esperado por la política
            q_values_dict = {state: q_values}
        
        # Obtenemos la acción según la política de comportamiento y los valores Q predichos
        return self.behavior_policy.get_action(state, q_values_dict)
    

    def update(self, state, action, reward, next_state, done):
        """
        Función de actualización de Deep Q-Learning.

        Para cada transición, guardaremos la experiencia en la memoria de replay. La experiencia estará formada por:
        (phi, a, r, phi', done)
        Siendo:
        - phi: estado actual
        - a: acción tomada
        - r: recompensa recibida
        - phi': siguiente estado
        - done: booleano que indica si el episodio ha terminado

        Para calcular la recompensa objetivo, obtendremos un minibatch de transiciones (j) de la memoria de replay y utilizaremos la fórmula:
        y_j = r_j + gamma * max_a' Q(phi_j', a'; theta)
        Siendo:
        - r_j: recompensa recibida en la transición j
        - gamma: factor de descuento
        - phi_j': siguiente estado en la transición j
        - a': acción con el valor Q más alto en phi_j'
        - theta: pesos actuales de la red neuronal

        Por último, actualizaremos los pesos de la red neuronal utilizando la función de pérdida entre:
        - Predicciones actuales Q(phi_j, a_j; theta)
        - Objetivos calculados y_j.

        :param state: Estado actual (phi).
        :param action: Acción tomada (a).
        :param reward: Recompensa recibida (r).
        :param next_state: Siguiente estado (phi').
        :param done: Booleano que indica si el episodio ha terminado.
        """
        # Guardamos la transición en la memoria de replay 
        self.memory.append((state, action, reward, next_state, done))


        # Comprobamos si tenemos suficientes experiencias en la memoria para realizar un entrenamiento
        if len(self.memory) < self.batch_size:
            return

        # Extraemos un minibatch de transiciones de D
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convertimos todo a tensores de PyTorch
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)


        # Desactivamos el cálculo de gradientes
        with torch.no_grad():
            # Obtenemos el valor Q máximo para el siguiente estado phi_j' utilizando la red neuronal (max_a' Q(phi_next, a'; theta))
            max_next_q = self.model(next_states).max(1)[0]
            # Calculamos los objetivos para cada transición en el minibatch (y_j = r_j + gamma * max_a' Q(phi_next, a'; theta))
            targets = rewards + (self.gamma * max_next_q * (1 - dones))

        # Obtenemos las predicciones actuales Q(phi_j, a_j; theta)
        current_q = self.model(states).gather(1, actions).squeeze()
        # Aplicamos la función de perdida entre las predicciones actuales (Q(phi_j, a_j; theta)) y los objetivos calculados (y_j)
        loss = self.criterion(current_q, targets)
        
        # Reseteamos los gradientes del optimizador
        self.optimizer.zero_grad()

        # Actualizamos los pesos de la red neuronal utilizando backpropagation
        loss.backward()
        self.optimizer.step()