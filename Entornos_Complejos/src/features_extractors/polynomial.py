import numpy as np
from sklearn.preprocessing import PolynomialFeatures

#def polinomial_feature_extractor(state, action, env):
#    """
#    Extractor de características simple para espacios continuos.
#    Usa características polinómicas de grado 2.
#    Debido a que la posición y la velocidad no son independientes; el efecto de la posición sobre el valor de una acción
#    depende de cuánta velocidad lleve el coche. Los términos cruzados (como pos*vel) de grado 2 permiten al agente
#    "entender" estas dependencias.
#    """
#    state = np.array(state).flatten()  # Asegurar que sea 1D
#    n_actions = env.action_space.n
#
#    # Características base: [1, s1, s2, s1^2, s2^2, s1*s2]
#    # Para MountainCar: posición y velocidad
#    features_list = [1.0]  # Bias
#
#    # Términos lineales
#    for s in state:
#        features_list.append(s)
#
#    # Términos cuadráticos
#    for s in state:
#        features_list.append(s ** 2)
#
#    # Términos cruzados (si hay más de una dimensión)
#    if len(state) > 1:
#        for i in range(len(state)):
#            for j in range(i + 1, len(state)):
#                features_list.append(state[i] * state[j])
#
#    base_features = np.array(features_list)
#    n_base_features = len(base_features)
#
#    # Crear vector de características con one-hot encoding para la acción
#    features = np.zeros(n_base_features * n_actions)
#    start_idx = action * n_base_features
#    end_idx = start_idx + n_base_features
#    features[start_idx:end_idx] = base_features
#
#    return features

class OptimizedPolynomialExtractor:
    """
    Extractor de características polinómicas optimizado para Aprendizaje por Refuerzo.
    Incluye normalización automática de los estados (Min-Max a [-1, 1]) y
    pre-inicialización de scikit-learn para máxima eficiencia computacional.
    """

    def __init__(self, env, degree=2):
        self.n_actions = env.action_space.n

        # 1. Obtenemos los límites del entorno para la normalización
        self.low = env.observation_space.low
        self.high = env.observation_space.high

        # 2. Inicializamos scikit-learn UNA SOLA VEZ
        self.poly = PolynomialFeatures(degree=degree)

        # Hacemos un 'fit' inicial con un estado de prueba para fijar las dimensiones
        dummy_state = np.zeros((1, len(self.low)))
        self.n_base_features = self.poly.fit_transform(dummy_state).shape[1]

    def __call__(self, state, action, env=None):
        """
        Al usar __call__, la instancia de esta clase se puede usar exactamente
        igual que tu función original: extractor(state, action, env)
        """
        # Asegurar formato numpy array
        state = np.array(state)

        # 1. Normalización Min-Max al rango [-1, 1]
        # Esto previene que los gradientes exploten al elevar números muy grandes o pequeños
        state_norm = 2.0 * (state - self.low) / (self.high - self.low) - 1.0
        state_norm = state_norm.reshape(1, -1)

        # 2. Transformación Polinómica (ahora es ultrarrápido porque solo usa transform)
        base_features = self.poly.transform(state_norm).flatten()

        # 3. One-hot encoding para asociar las características a la acción elegida
        features = np.zeros(self.n_base_features * self.n_actions)
        start_idx = action * self.n_base_features
        end_idx = start_idx + self.n_base_features
        features[start_idx:end_idx] = base_features

        return features

#def polynomial_feature_extractor(state, action, env, degree=2):
#    """
#    Extractor robusto usando scikit-learn.
#    Garantiza la creación correcta de todos los términos cruzados para cualquier grado.
#    """
#    state = np.array(state).reshape(1, -1)
#    n_actions = env.action_space.n
#
#    # Genera [1, s1, s2, s1^2, s1*s2, s2^2] automáticamente
#    poly = PolynomialFeatures(degree=degree)
#    base_features = poly.fit_transform(state).flatten()
#
#    n_base_features = len(base_features)
#    features = np.zeros(n_base_features * n_actions)
#
#    start_idx = action * n_base_features
#    end_idx = start_idx + n_base_features
#    features[start_idx:end_idx] = base_features
#
#    return features