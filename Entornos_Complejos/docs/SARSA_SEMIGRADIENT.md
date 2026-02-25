# Agente SARSA Semi-Gradiente

## Descripción

El **AgentSarsaSemiGradient** es una implementación del algoritmo SARSA (State-Action-Reward-State-Action) que utiliza **aproximación de funciones lineales** en lugar de una tabla Q tradicional. Esto permite trabajar con espacios de estados continuos o muy grandes donde mantener una tabla Q completa sería impracticable.

## Diferencias con SARSA Tabular

| Característica | SARSA Tabular | SARSA Semi-Gradiente |
|----------------|---------------|----------------------|
| **Representación** | Tabla Q discreta | Vector de pesos continuo |
| **Espacio de estados** | Discreto y pequeño | Continuo o muy grande |
| **Memoria requerida** | O(|S| × |A|) | O(n_features) |
| **Generalización** | No generaliza entre estados | Generaliza entre estados similares |
| **Convergencia** | Garantizada (bajo ciertas condiciones) | No garantizada (método semi-gradiente) |

## Fundamento Matemático

### Aproximación Lineal

En lugar de mantener un valor Q(s, a) para cada par estado-acción, aproximamos:

```
q̂(s, a, w) = w^T · φ(s, a)
```

Donde:
- **w**: Vector de pesos (parámetros a aprender)
- **φ(s, a)**: Vector de características (features) del par estado-acción
- **q̂(s, a, w)**: Aproximación del valor Q verdadero

### Actualización Semi-Gradiente

La actualización de los pesos sigue la regla:

```
w ← w + α · [R + γ · q̂(S', A', w) - q̂(S, A, w)] · ∇_w q̂(S, A, w)
```

Para aproximación lineal, el gradiente es simplemente:

```
∇_w q̂(S, A, w) = φ(S, A)
```

Por lo tanto, la actualización se simplifica a:

```
w ← w + α · [R + γ · q̂(S', A', w) - q̂(S, A, w)] · φ(S, A)
```

### ¿Por qué "Semi-Gradiente"?

Se llama "semi-gradiente" porque **no calculamos el gradiente completo**. El término `q̂(S', A', w)` también depende de `w`, pero lo tratamos como una constante durante la actualización. Esto hace que el método sea más eficiente computacionalmente, pero pierde las garantías de convergencia del verdadero descenso de gradiente.

## Uso del Agente

### Inicialización

```python
from Entornos_Complejos.src.agents import AgentSarsaSemiGradient
from Entornos_Complejos.src.policies import EpsilonGreedyPolicy
import gymnasium as gym

# Crear el entorno
env = gym.make('CartPole-v1')

# Definir el extractor de características
def feature_extractor(state, action, env):
    # Tu implementación aquí
    pass

# Crear la política
policy = EpsilonGreedyPolicy(epsilon=0.1)

# Crear el agente
agent = AgentSarsaSemiGradient(
    env=env,
    policy=policy,
    feature_extractor=feature_extractor,
    n_features=100,  # Número de características
    alpha=0.01,      # Tasa de aprendizaje
    gamma=0.99       # Factor de descuento
)
```

### Bucle de Entrenamiento

```python
for episode in range(n_episodes):
    state, _ = env.reset()
    action = agent.get_action(state)
    done = False

    while not done:
        next_state, reward, done, truncated, _ = env.step(action)

        # Actualizar y obtener la siguiente acción
        next_action = agent.update(state, action, reward, next_state, done or truncated)

        state = next_state
        action = next_action if next_action is not None else agent.get_action(state)
```

## Extractores de Características

El componente más importante del SARSA Semi-Gradiente es el **extractor de características**. Esta función transforma el par (estado, acción) en un vector de características que captura la información relevante.

### Requisitos del Extractor

```python
def feature_extractor(state, action, env):
    """
    :param state: Estado del entorno
    :param action: Acción a tomar
    :param env: Entorno de Gymnasium
    :return: numpy array de tamaño n_features
    """
    # Tu implementación
    return features
```

### Tipos de Extractores

#### 1. Características Polinómicas

Útil para capturar relaciones no lineales:

```python
def polynomial_features(state, action, env, degree=2):
    state = np.array(state)
    n_actions = env.action_space.n

    # Crear características: [1, s1, s2, ..., s1², s2², ..., s1*s2, ...]
    features_list = [1.0]  # Bias

    # Términos lineales
    features_list.extend(state)

    # Términos cuadráticos
    if degree >= 2:
        features_list.extend([s**2 for s in state])
        # Términos cruzados
        for i in range(len(state)):
            for j in range(i+1, len(state)):
                features_list.append(state[i] * state[j])

    # One-hot encoding para la acción
    base_features = np.array(features_list)
    features = np.zeros(len(base_features) * n_actions)
    features[action * len(base_features):(action + 1) * len(base_features)] = base_features

    return features
```

#### 2. Tile Coding

Excelente para espacios continuos, crea múltiples discretizaciones:

```python
def tile_coding(state, action, env, n_tilings=8, n_tiles_per_dim=8):
    # Implementación en el archivo de ejemplo
    pass
```

**Ventajas:**
- Buena generalización local
- Eficiente computacionalmente
- Funciona bien en la práctica

#### 3. Funciones de Base Radial (RBF)

Captura similitudes locales usando funciones gaussianas:

```python
def rbf_features(state, action, env, n_centers=10, sigma=1.0):
    # Implementación en el archivo de ejemplo
    pass
```

**Ventajas:**
- Suavidad en la aproximación
- Buena para funciones continuas
- Interpretable

#### 4. Características Simples (Baseline)

Para espacios pequeños o como punto de partida:

```python
def simple_features(state, action, env):
    state = np.array(state)
    n_actions = env.action_space.n

    # Concatenar estado con bias
    base_features = np.concatenate([[1.0], state])

    # One-hot para acción
    features = np.zeros(len(base_features) * n_actions)
    features[action * len(base_features):(action + 1) * len(base_features)] = base_features

    return features
```

## Hiperparámetros

### Tasa de Aprendizaje (alpha)

- **Rango típico:** 0.0001 - 0.1
- **Valor recomendado:** 0.001 - 0.01
- **Consideraciones:**
  - Valores muy altos pueden causar inestabilidad
  - Valores muy bajos ralentizan el aprendizaje
  - Puede ser útil usar un schedule decreciente

### Factor de Descuento (gamma)

- **Rango típico:** 0.9 - 0.999
- **Valor recomendado:** 0.99
- **Consideraciones:**
  - Valores cercanos a 1 consideran más el futuro
  - Valores menores priorizan recompensas inmediatas

### Número de Características (n_features)

- **Depende del extractor elegido**
- **Consideraciones:**
  - Más características = mayor capacidad de representación
  - Más características = más lento y más datos necesarios
  - Balance entre expresividad y eficiencia

## Ventajas y Desventajas

### Ventajas

✅ **Escalabilidad:** Funciona con espacios de estados continuos o muy grandes
✅ **Generalización:** Aprende patrones que se transfieren entre estados similares
✅ **Eficiencia de memoria:** O(n_features) en lugar de O(|S| × |A|)
✅ **Flexibilidad:** Permite usar diferentes representaciones de características

### Desventajas

❌ **Sin garantías de convergencia:** El método semi-gradiente puede diverger
❌ **Diseño de características:** Requiere conocimiento del dominio para elegir buenas características
❌ **Sensibilidad a hiperparámetros:** Requiere ajuste cuidadoso de alpha
❌ **Capacidad limitada:** La aproximación lineal puede no capturar funciones muy complejas

## Consejos Prácticos

### 1. Normalización de Características

Normaliza las características para que tengan media 0 y varianza 1:

```python
features = (features - mean) / std
```

### 2. Inicialización de Pesos

- Por defecto, los pesos se inicializan a cero
- Para algunos problemas, inicialización aleatoria pequeña puede ayudar:

```python
agent.weights = np.random.randn(n_features) * 0.01
```

### 3. Monitoreo del Aprendizaje

Observa la norma de los pesos para detectar divergencia:

```python
weight_norm = np.linalg.norm(agent.get_weights())
if weight_norm > 1000:
    print("¡Advertencia! Los pesos están creciendo demasiado")
```

### 4. Ajuste de Alpha

Si el aprendizaje es inestable, reduce alpha:

```python
# Alpha decreciente
alpha = alpha_initial / (1 + episode / decay_rate)
```

### 5. Debugging

Verifica que las características sean razonables:

```python
features = feature_extractor(state, action, env)
print(f"Min: {features.min()}, Max: {features.max()}, Norm: {np.linalg.norm(features)}")
```

## Comparación con Otros Métodos

### vs. SARSA Tabular

- Usa SARSA Semi-Gradiente cuando el espacio de estados es continuo o muy grande
- Usa SARSA Tabular cuando el espacio es pequeño y discreto

### vs. Q-Learning Semi-Gradiente

- SARSA es on-policy (más estable)
- Q-Learning es off-policy (potencialmente más eficiente en datos)

### vs. Deep Q-Networks (DQN)

- SARSA Semi-Gradiente usa aproximación lineal (más simple, más rápido)
- DQN usa redes neuronales (más expresivo, más complejo)

## Referencias

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
  - Capítulo 9: On-policy Prediction with Approximation
  - Capítulo 10: On-policy Control with Approximation
  - Sección 10.1: Episodic Semi-gradient Control

## Ejemplo Completo

Ver el archivo `examples/ejemplo_sarsa_semigradient.py` para ejemplos completos de uso con diferentes extractores de características.
