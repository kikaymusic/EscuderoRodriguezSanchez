from abc import ABC, abstractmethod

class Policy(ABC):

    @abstractmethod
    def get_action(self, state, q_values=None):
        """Devuelve una acción dado un estado (y opcionalmente los valores Q)."""
        raise NotImplementedError("This method must be implemented by the subclass.")

    @abstractmethod
    def get_probability(self, state, action, q_values=None):
        """
        Devuelve la probabilidad de tomar una 'action' en un 'state'.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")

    @abstractmethod
    def get_action_probabilities(self, state, q_values=None):
        """
        Calcula la distribución de probabilidad completa para todas las acciones en un estado.
        Útil para algoritmos como Expected Sarsa que necesitan el vector completo de probabilidades.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")
