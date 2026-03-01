import numpy as np


class TileCodingExtractor:
    """
    Extractor de características basado en Tile Coding para Aprendizaje por Refuerzo.

    Tile Coding es una técnica de discretización que divide el espacio continuo de estados
    en múltiples grillas (tilings) superpuestas con desplazamientos. Cada grilla proporciona
    una granularidad diferente, permitiendo generalización local y aproximación suave.

    Ventajas:
    - Generalización local: estados cercanos comparten tiles
    - Múltiples resoluciones: diferentes tilings capturan patrones a diferentes escalas
    - Eficiencia computacional: representación sparse
    - Control de generalización mediante el número de tilings y tiles
    """

    def __init__(self, env, num_tilings=8, tiles_per_dim=8):
        """
        Inicializa el extractor de características con Tile Coding.

        Args:
            env: Entorno de Gymnasium
            num_tilings: Número de tilings superpuestos (capas de grillas)
            tiles_per_dim: Número de tiles por dimensión en cada tiling
        """
        self.n_actions = env.action_space.n
        self.num_tilings = num_tilings
        self.tiles_per_dim = tiles_per_dim

        # Límites del espacio de observación
        self.low = env.observation_space.low
        self.high = env.observation_space.high
        self.n_dims = len(self.low)

        # Tamaño de cada tile en cada dimensión
        self.tile_width = (self.high - self.low) / tiles_per_dim

        # Calcular desplazamientos para cada tiling
        # Los desplazamientos distribuyen uniformemente los tilings
        self.offsets = np.array([
            (i / num_tilings) * self.tile_width
            for i in range(num_tilings)
        ])

        # Número total de tiles por tiling
        self.tiles_per_tiling = tiles_per_dim ** self.n_dims

        # Dimensión total del vector de características
        # num_tilings * tiles_per_tiling por cada acción
        self.n_base_features = num_tilings * self.tiles_per_tiling
        self.n_features = self.n_base_features * self.n_actions

    def _get_tile_indices(self, state):
        """
        Calcula los índices de los tiles activados para un estado dado.

        Args:
            state: Estado del entorno

        Returns:
            Array con los índices de los tiles activados (uno por tiling)
        """
        state = np.array(state)
        tile_indices = []

        for tiling_idx in range(self.num_tilings):
            # Aplicar desplazamiento del tiling actual
            offset_state = state + self.offsets[tiling_idx]

            # Calcular la posición del tile en cada dimensión
            tile_coords = np.floor(
                (offset_state - self.low) / self.tile_width
            ).astype(int)

            # Clip para asegurar que estamos dentro de los límites
            tile_coords = np.clip(tile_coords, 0, self.tiles_per_dim - 1)

            # Convertir coordenadas multidimensionales a índice único
            # Similar a raveling en numpy
            tile_idx = 0
            multiplier = 1
            for dim in range(self.n_dims - 1, -1, -1):
                tile_idx += tile_coords[dim] * multiplier
                multiplier *= self.tiles_per_dim

            # Añadir offset para este tiling
            tile_idx += tiling_idx * self.tiles_per_tiling
            tile_indices.append(tile_idx)

        return np.array(tile_indices)

    def __call__(self, state, action, env=None):
        """
        Extrae características usando Tile Coding.

        Args:
            state: Estado actual del entorno
            action: Acción tomada
            env: Entorno (opcional, para compatibilidad)

        Returns:
            Vector de características sparse con ones en los tiles activados
        """
        # Crear vector de características vacío
        features = np.zeros(self.n_features)

        # Obtener índices de tiles activados para el estado
        tile_indices = self._get_tile_indices(state)

        # Activar los tiles correspondientes a la acción
        # Cada acción tiene su propio conjunto de tiles
        action_offset = action * self.n_base_features
        for tile_idx in tile_indices:
            features[action_offset + tile_idx] = 1.0

        return features

    def get_feature_size(self):
        """Retorna el tamaño total del vector de características."""
        return self.n_features
