
import numpy as np
from arms.arm import Arm

class ArmBernoulli(Arm):
    def __init__(self, p: float):
        """
        Inicializa el brazo con distribución de Bernoulli Be(p).
        
        Representa un experimento con dos resultados posibles: 
        1 (éxito) con probabilidad p, y 0 (fracaso) con probabilidad 1-p.

        :param p: Probabilidad de éxito. Debe estar en el rango [0, 1].
        """
        assert 0 <= p <= 1, f"La probabilidad p debe estar entre 0 y 1. Recibido: {p}"
        
        self.p = p

    def pull(self) -> int:
        """
        Genera una recompensa basada en la probabilidad p.
        
        :return: 1 si hay éxito, 0 si hay fracaso.
        """
        # np.random.binomial(1, p) es matemáticamente equivalente a Bernoulli(p)
        return np.random.binomial(1, self.p)

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución Bernoulli.
        
        Para una Be(p), el valor esperado E[X] = p.
        """
        return self.p

    def __str__(self):
        return f"ArmBernoulli(p={self.p:.2f})"

    @classmethod
    def generate_arms(cls, k: int) -> list:
        """
        Genera k brazos de Bernoulli con probabilidades p aleatorias.

        :param k: Número de brazos a generar.
        :return: Lista de instancias de ArmBernoulli.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        
        # Generamos k probabilidades aleatorias uniformes entre 0 y 1
        p_values = np.random.uniform(0, 1, k)
        return [cls(p=p) for p in p_values]