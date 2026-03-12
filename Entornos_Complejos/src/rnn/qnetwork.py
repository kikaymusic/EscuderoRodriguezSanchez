import torch.nn as nn

class QNetwork(nn.Module):
    """
    Red neuronal para aproximar Q(s,a) en MountainCar.
    Arquitectura: 2 inputs (posición, velocidad) -> hidden layers -> 3 outputs (Q-values por acción)
    """
    def __init__(self, state_dim=2, action_dim=3, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def init_weights_fijos(self):
        """
        Método interno para forzar pesos constantes y asegurar reproducibilidad.
        """
        # Definimos la función interna que aplicaremos a cada capa
        def _aplicar_constantes(m):
            if isinstance(m, nn.Linear):
                # Peso fijo 0.01 y sesgo 0.0
                nn.init.constant_(m.weight, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        # El self hace referencia a la instancia actual de la red
        self.apply(_aplicar_constantes)
        print("Pesos de la red inicializados a valores fijos (0.01)")

    def forward(self, x):
        return self.network(x)