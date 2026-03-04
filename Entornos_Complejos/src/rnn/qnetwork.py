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
    
    def forward(self, x):
        return self.network(x)