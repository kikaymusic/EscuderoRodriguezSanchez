# Resumen: Implementación del Agente SARSA Semi-Gradiente

## 📋 Archivos Creados

### 1. **Agente Principal**
- **Ruta:** `src/agents/sarsa_semigradient.py`
- **Clase:** `AgentSarsaSemiGradient`
- **Descripción:** Implementación completa del algoritmo SARSA con aproximación de funciones lineales

### 2. **Archivo de Ejemplos**
- **Ruta:** `examples/ejemplo_sarsa_semigradient.py`
- **Contenido:**
  - 4 extractores de características diferentes:
    - Tile Coding
    - Características Polinómicas
    - Funciones de Base Radial (RBF)
    - Características Simples
  - Función de entrenamiento completa
  - Ejemplo ejecutable

### 3. **Documentación**
- **Ruta:** `docs/SARSA_SEMIGRADIENT.md`
- **Contenido:**
  - Explicación teórica del algoritmo
  - Fundamento matemático
  - Comparación con SARSA tabular
  - Guía de uso completa
  - Consejos prácticos
  - Referencias

### 4. **Tests**
- **Ruta:** `tests/test_sarsa_semigradient.py`
- **Tests incluidos:**
  - Inicialización del agente
  - Selección de acciones
  - Actualización de pesos
  - Cálculo de valores Q
  - Entrenamiento de episodios
  - Gestión de pesos

### 5. **Archivos Actualizados**
- `src/agents/__init__.py` - Añadido `AgentSarsaSemiGradient` a las exportaciones
- `src/policies/__init__.py` - Añadidas las políticas a las exportaciones

---

## 🎯 Características Principales del Agente

### Parámetros del Constructor

```python
    AgentSarsaSemiGradient(
        env: Env,                    # Entorno de Gymnasium
        policy: Policy,              # Política (on-policy)
        feature_extractor: callable, # Función de extracción de características
        n_features: int,             # Número de características
        alpha: float = 0.01,         # Tasa de aprendizaje
        gamma: float = 0.99          # Factor de descuento
    )
```

### Métodos Principales

1. **`get_action(state)`** - Selecciona una acción según la política
2. **`update(state, action, reward, next_state, done)`** - Actualiza los pesos usando SARSA semi-gradiente
3. **`get_weights()`** - Obtiene una copia del vector de pesos
4. **`set_weights(weights)`** - Establece el vector de pesos
5. **`reset_weights()`** - Reinicia los pesos a cero

### Métodos Internos

- **`_get_features(state, action)`** - Extrae características del par estado-acción
- **`_get_q_value(state, action)`** - Calcula q̂(s, a) = w^T · φ(s, a)
- **`_get_all_q_values(state)`** - Calcula valores Q para todas las acciones

---

## 🔬 Algoritmo Implementado

### Fórmula de Actualización

```
w ← w + α · [R + γ · q̂(S', A', w) - q̂(S, A, w)] · φ(S, A)
```

Donde:
- **w**: Vector de pesos
- **α**: Tasa de aprendizaje (alpha)
- **R**: Recompensa recibida
- **γ**: Factor de descuento (gamma)
- **q̂(S, A, w)**: Aproximación del valor Q = w^T · φ(S, A)
- **φ(S, A)**: Vector de características

### Flujo del Algoritmo

1. Inicializar pesos w arbitrariamente
2. Para cada episodio:
   - Inicializar S
   - Elegir A usando la política derivada de q̂
   - Para cada paso del episodio:
     - Tomar acción A, observar R, S'
     - Elegir A' usando la política derivada de q̂
     - Actualizar: w ← w + α · [R + γ · q̂(S', A', w) - q̂(S, A, w)] · φ(S, A)
     - S ← S', A ← A'
   - Hasta que S sea terminal

---

## 📊 Extractores de Características Incluidos

### 1. Tile Coding
- **Uso:** Espacios continuos multidimensionales
- **Ventajas:** Buena generalización local, eficiente
- **Parámetros:** `n_tilings`, `n_tiles_per_dim`

### 2. Características Polinómicas
- **Uso:** Aproximar funciones no lineales
- **Ventajas:** Captura interacciones entre variables
- **Parámetros:** `degree` (grado del polinomio)

### 3. Funciones de Base Radial (RBF)
- **Uso:** Funciones suaves, similitudes locales
- **Ventajas:** Aproximación suave, interpretable
- **Parámetros:** `n_centers`, `sigma`

### 4. Características Simples
- **Uso:** Baseline, espacios pequeños
- **Ventajas:** Simple, rápido
- **Parámetros:** Ninguno

---

## 🚀 Ejemplo de Uso Rápido

```python
import gymnasium as gym
from Entornos_Complejos.src.agents import AgentSarsaSemiGradient
from Entornos_Complejos.src.policies import EpsilonGreedyPolicy
import numpy as np

# Crear entorno
env = gym.make('CartPole-v1')

# Definir extractor de características
def feature_extractor(state, action, env):
    state = np.array(state)
    n_actions = env.action_space.n
    base_features = np.concatenate([[1.0], state])
    features = np.zeros(len(base_features) * n_actions)
    features[action * len(base_features):(action + 1) * len(base_features)] = base_features
    return features

# Crear política
policy = EpsilonGreedyPolicy(epsilon=0.1, n_actions=env.action_space.n)

# Crear agente
agent = AgentSarsaSemiGradient(
    env=env,
    policy=policy,
    feature_extractor=feature_extractor,
    n_features=10,  # (1 + 4 dimensiones) * 2 acciones
    alpha=0.01,
    gamma=0.99
)

# Entrenar
for episode in range(500):
    state, _ = env.reset()
    action = agent.get_action(state)
    done = False

    while not done:
        next_state, reward, done, truncated, _ = env.step(action)
        next_action = agent.update(state, action, reward, next_state, done or truncated)
        state = next_state
        action = next_action if next_action is not None else agent.get_action(state)
```

---

## ✅ Tests Disponibles

Para ejecutar los tests:

```bash
python Entornos_Complejos/tests/test_sarsa_semigradient.py
```

Tests incluidos:
- ✓ Inicialización del agente
- ✓ Selección de acciones válidas
- ✓ Actualización de pesos
- ✓ Cálculo de valores Q
- ✓ Entrenamiento de episodios completos
- ✓ Gestión de pesos (get/set/reset)

---

## 📚 Diferencias Clave con Otros Agentes

### vs. SARSA Tabular (`AgentSarsa`)
- **SARSA Tabular:** Usa tabla Q discreta, solo espacios discretos pequeños
- **SARSA Semi-Gradiente:** Usa aproximación lineal, espacios continuos o grandes

### vs. Q-Learning (`AgentQLearning`)
- **Q-Learning:** Off-policy, actualiza hacia el máximo Q
- **SARSA Semi-Gradiente:** On-policy, actualiza hacia la acción seleccionada

### vs. Monte Carlo (`AgentMonteCarlo`)
- **Monte Carlo:** Aprende al final del episodio, sin bootstrapping
- **SARSA Semi-Gradiente:** Aprende en cada paso, con bootstrapping

---

## 🎓 Cuándo Usar SARSA Semi-Gradiente

### ✅ Usar cuando:
- El espacio de estados es continuo (ej: CartPole, MountainCar)
- El espacio de estados es muy grande
- Necesitas generalización entre estados similares
- Quieres un método on-policy (más estable que off-policy)

### ❌ No usar cuando:
- El espacio de estados es pequeño y discreto (usa SARSA tabular)
- Necesitas garantías de convergencia
- La función Q es muy compleja (considera redes neuronales/DQN)

---

## 📖 Referencias y Recursos

- **Libro:** Sutton & Barto (2018) - "Reinforcement Learning: An Introduction"
  - Capítulo 9: On-policy Prediction with Approximation
  - Capítulo 10: On-policy Control with Approximation

- **Documentación completa:** `docs/SARSA_SEMIGRADIENT.md`
- **Ejemplos:** `examples/ejemplo_sarsa_semigradient.py`

---

## 🔧 Próximos Pasos Sugeridos

1. **Experimentar con diferentes extractores de características**
   - Probar tile coding con diferentes configuraciones
   - Ajustar el grado de las características polinómicas

2. **Ajustar hiperparámetros**
   - Probar diferentes valores de alpha (0.0001 - 0.1)
   - Experimentar con diferentes valores de epsilon

3. **Probar en diferentes entornos**
   - MountainCar-v0 (continuo)
   - Acrobot-v1
   - LunarLander-v2

4. **Implementar mejoras**
   - Alpha decreciente (learning rate decay)
   - Normalización de características
   - Eligibility traces (SARSA(λ))

---

## 📝 Notas Importantes

- El agente es **on-policy**, lo que significa que aprende sobre la misma política que usa para actuar
- El método es **semi-gradiente** porque no calcula el gradiente completo (trata el target como constante)
- La **elección del extractor de características** es crucial para el rendimiento
- Requiere **ajuste cuidadoso de alpha** para evitar divergencia
- **No garantiza convergencia** como los métodos tabulares

---

## 🎉 Resumen

Se ha implementado exitosamente el agente **SARSA Semi-Gradiente** como una nueva subclase de `Agent`, incluyendo:

- ✅ Implementación completa del algoritmo
- ✅ 4 extractores de características diferentes
- ✅ Documentación detallada
- ✅ Ejemplos de uso
- ✅ Suite de tests
- ✅ Integración con el sistema de políticas existente

El agente está listo para ser usado en problemas de aprendizaje por refuerzo con espacios de estados continuos o muy grandes.
