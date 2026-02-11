"""
Module: algorithms/ucb1.py
Description: Implementación del algoritmo UCB1 para el problema de los k-brazos.
"""

import numpy as np

from .algorithm import Algorithm

class UCB1(Algorithm):
    def __init__(self, k: int, c: float = 1.0):
        """
        Inicializa el algoritmo UCB1.

        :param k: Número de brazos.
        :param c: Parámetro de ajuste de exploración. (usualmente c=1)
        """
        assert 0 < c, "El parámetro c debe ser mayor o igual a 0."

        super().__init__(k)
        self.c = c

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB1.
        Valor UCB1(a) = Qt(a) + u(a)
        Siendo u(a) = c * sqrt((log(t)) / N(a))
        
        :return: Índice del brazo seleccionado.
        """
        # Si hay algún brazo que no se ha seleccionado todavía, lo seleccionamos
        # Esto evita problemas de división por cero en el cálculo de u(a) y asegura que cada brazo se seleccione al menos una vez.
        if 0 in self.counts:
            # Buscamos el primer brazo que no ha sido seleccionado
            return int(np.argmin(self.counts))

        # Calculamos el paso en el que estamos contando el total de selecciones de brazos
        t = np.sum(self.counts)

        # Obtenemos el término de explotación Qt(a)
        q_a = self.values

        # Calculamos el término de exploración u(a) = (c * sqrt((log(t)) / N(a)))
        ua = self.c * np.sqrt(np.log(t) / self.counts)

        # Calculamos el valor de UCB1 = Qt(a) + u(a)
        ucb_values = q_a + ua

        # Seleccionamos el brazo con el valor de UCB1 más alto
        return int(np.argmax(ucb_values))