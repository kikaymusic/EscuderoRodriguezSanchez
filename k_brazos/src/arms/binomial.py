import numpy as np
from arms.arm import Arm  # Ajusta el import según tu estructura (igual que en armnormal.py)

class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        """
        Inicializa el brazo con distribución Binomial B(n, p).
        
        Modela la probabilidad de obtener k éxitos en n ensayos, 
        donde cada ensayo tiene probabilidad p.

        :param n: Número de ensayos (trials). Debe ser entero >= 1.
        :param p: Probabilidad de éxito en cada ensayo. Debe estar en [0, 1].
        """
        assert isinstance(n, int) and n >= 1, "El número de ensayos n debe ser un entero >= 1."
        assert 0 <= p <= 1, "La probabilidad p debe estar entre 0 y 1."

        self.n = n
        self.p = p

    def pull(self) -> int:
        """
        Genera una recompensa siguiendo una distribución Binomial.
        
        Simula n lanzamientos de moneda con probabilidad p.
        
        :return: Número de éxitos obtenidos (entero entre 0 y n).
        """
        # Usamos numpy para generar la muestra eficientemente
        reward = np.random.binomial(self.n, self.p)
        return reward

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución Binomial.
        
        La esperanza matemática de una Binomial B(n, p) es E[X] = n * p.

        :return: Valor esperado de la distribución.
        """
        return self.n * self.p

    def __str__(self):
        """
        Representación en cadena del brazo binomial.

        :return: Descripción detallada del brazo.
        """
        return f"ArmBinomial(n={self.n}, p={self.p:.2f})"

    @classmethod
    def generate_arms(cls, k: int, n: int = 10) -> list:
        """
        Genera k brazos binomiales con el mismo n pero diferentes probabilidades p.
        
        Las probabilidades p se generan aleatoriamente de forma uniforme entre 0 y 1.

        :param k: Número de brazos a generar.
        :param n: Número de ensayos para cada brazo (fijo para todos). Por defecto 10.
        :return: Lista de objetos ArmBinomial.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert n >= 1, "El número de ensayos n debe ser mayor o igual a 1."

        arms = []
        # Generamos k probabilidades aleatorias entre 0 y 1
        p_values = np.random.uniform(0, 1, k)
        
        for p in p_values:
            arms.append(cls(n=n, p=p))
            
        return arms