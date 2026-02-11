import numpy as np
from .algorithm import Algorithm

class Softmax(Algorithm):

    def __init__(self, k: int, tau: float = 1.0):
        """
        Inicializa el algoritmo Softmax.

        :param k: Número de brazos.
        :param tau: Parámetro de temperatura (tau).
                    Controla la exploración vs explotación.
        :raises ValueError: Si la temperatura no es mayor que 0.
        """
        # Validación similar a la que hace epsilon_greedy con epsilon
        assert tau > 0, "El parámetro tau debe ser mayor que 0."

        # Llamada al constructor de la clase padre (Algorithm)
        super().__init__(k)
        
        # Guardamos la temperatura propia de este algoritmo
        self.tau = tau

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la probabilidad de Boltzmann.
        
        Pseudocódigo:
        P(a) = exp(Q(a) / tau) / sum(exp(Q(b) / tau) para todo b)
        """
        
        # 1. Recuperamos Q(a) (valores estimados) de la clase padre
        q_values = self.values
        
        # 2. Aplicamos la fórmula del exponente: exp(Q(a) / tau)
        # Nota técnica: Restamos el máximo (z - max(z)) antes de exponenciar 
        # para evitar errores numéricos (overflow) en Python. La matemática se mantiene proporcional.
        z = q_values / self.tau
        max_z = np.max(z)
        exp_values = np.exp(z - max_z)
        
        # 3. Calculamos la suma de los exponentes (el denominador)
        sum_exp_values = np.sum(exp_values)
        
        # 4. Calculamos las probabilidades finales P(a)
        probabilities = exp_values / sum_exp_values
        
        # 5. Seleccionamos el brazo usando esas probabilidades
        chosen_arm = np.random.choice(self.k, p=probabilities)
        
        return chosen_arm

    # NO implementamos update().
    # Usamos el de la clase padre Algorithm, que ya hace:
    # Q(a) <- Q(a) + 1/n * (R - Q(a))