from abc import ABC, abstractmethod

class Policy(ABC):
    @classmethod
    def __init__(self):
        """
        Inicializamos todo lo necesario para la política
        """
        raise NotImplementedError("This method must be implemented by the subclass.")

    @abstractmethod
    def select_action(self, state, training=True):
        """
        Lógica para elegir acción
        """
        raise NotImplementedError("This method must be implemented by the subclass.")

    @abstractmethod
    def update(self, trajectory):
        """
        Actualización de la política
        """
        raise NotImplementedError("This method must be implemented by the subclass.")