import numpy as np
from gymnasium.core import Env

from src.rnn.qnetwork import QNetwork
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

    def __init__(self,
                 env: Env,
                 behavior_policy: Policy,
                 q_network: QNetwork,
                 lr: float = 1e-3,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 device: str = None,
                 update_frequency: int = 1,
                 seed: int = None):
        """
        :param env: Entorno de Gymnasium.
        :param behavior_policy: Política de comportamiento.
        :param q_network: Red neuronal de PyTorch que actúa como aproximador de Q.
        :param lr: Learning rate para la red neuronal.
        :param batch_size: Tamaño del minibatch para el entrenamiento.
        :param buffer_size: Capacidad N de la memoria de replay (D).
        :param gamma: Factor de descuento.
        :param device: Dispositivo para ejecutar el modelo ('cuda', 'cpu', o None para auto-detectar).
        :param update_frequency: Frecuencia de actualizaciones (entrenar cada N pasos).
        """
        super().__init__(env)
        # Inicializamos la tabla Q vacia, ya que no la utilizaremos
        self.q_table = None
        # Política de comportamiento
        self.behavior_policy = behavior_policy
        # Factor de descuento
        self.gamma = gamma

        # Detectar dispositivo (GPU si está disponible, sino CPU)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Red neuronal para aproximar la función Q
        self.model = q_network.to(self.device)
        # Inicializamos los pesos de la red neuronal
        q_network.init_weights_fijos()

        # Optimizador y función de pérdida para entrenar la red
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        # Tamaño del minibatch para el entrenamiento
        self.batch_size = batch_size

        # Inicializamos la memoria de replay D con el tamaño especificado N
        self.memory = deque(maxlen=buffer_size)

        # Contador de pasos y frecuencia de actualización
        self.update_frequency = update_frequency
        self.step_count = 0

        # RNG para reproducibilidad
        self.rng = np.random.default_rng(seed)


    def get_action(self, state):
        """
        Obtiene una acción siguiendo la política actual y la red neuronal.
        """
        # Convertimos el estado a tensor para la red y movemos al dispositivo correcto
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Desactivamos el cálculo de gradientes
        with torch.no_grad():
            # Predecimos los valores de Q para todas las acciones
            q_values = self.model(state_tensor).cpu().numpy().squeeze()
            # Creamos un diccionario con una clave hashable (tupla) para el formato esperado por la política
            q_values_dict = {tuple(state): q_values}

        # Obtenemos la acción según la política de comportamiento y los valores Q predichos
        return self.behavior_policy.get_action(tuple(state), q_values_dict)
    

    def update(self, state, action, reward, next_state, done):
        """
        Función de actualización de Deep Q-Learning con optimizaciones de rendimiento.

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

        # Incrementar contador de pasos
        self.step_count += 1

        # Comprobamos si tenemos suficientes experiencias en la memoria y si toca entrenar
        if len(self.memory) < self.batch_size or self.step_count % self.update_frequency != 0:
            return

        # Extraemos un minibatch de transiciones de D usando índices aleatorios (más rápido)
        indices = self.rng.integers(0, len(self.memory), size=self.batch_size)

        # Preparar arrays numpy primero (más eficiente)
        states_np = np.zeros((self.batch_size, len(state)), dtype=np.float32)
        next_states_np = np.zeros((self.batch_size, len(state)), dtype=np.float32)
        actions_np = np.zeros(self.batch_size, dtype=np.int64)
        rewards_np = np.zeros(self.batch_size, dtype=np.float32)
        dones_np = np.zeros(self.batch_size, dtype=np.float32)

        for i, idx in enumerate(indices):
            s, a, r, ns, d = self.memory[idx]
            states_np[i] = s
            actions_np[i] = a
            rewards_np[i] = r
            next_states_np[i] = ns
            dones_np[i] = d

        # Convertir a tensores de una vez (una sola transferencia CPU->GPU)
        states = torch.from_numpy(states_np).to(self.device, non_blocking=True)
        actions = torch.from_numpy(actions_np).unsqueeze(1).to(self.device, non_blocking=True)
        rewards = torch.from_numpy(rewards_np).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(next_states_np).to(self.device, non_blocking=True)
        dones = torch.from_numpy(dones_np).to(self.device, non_blocking=True)

        # Desactivamos el cálculo de gradientes para el cálculo de targets
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

        # Actualizamos los gradientes mediante backpropagation
        loss.backward()

        # Aplicamos gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Actualizamos los pesos
        self.optimizer.step()