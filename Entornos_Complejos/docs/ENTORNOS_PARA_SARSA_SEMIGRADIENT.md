# Entornos Apropiados para SARSA Semi-Gradiente

## ¿Qué tipo de entornos son adecuados?

SARSA Semi-Gradiente con **aproximación lineal** está diseñado para:

✅ **Espacios de estados CONTINUOS** (o muy grandes)
✅ **Acciones DISCRETAS**
✅ **Dimensionalidad baja-media** (vectores, no imágenes)

## ❌ Entornos NO Apropiados

### CarRacing-v3
- **Problema:** Observaciones son imágenes (96×96×3 = 27,648 dimensiones)
- **Solución:** Requiere redes neuronales convolucionales (CNN) → usar DQN, no SARSA lineal

### MountainCarContinuous-v0
- **Problema:** Acciones continuas (no discretas)
- **Solución:** Requiere métodos actor-crítico o discretizar acciones

### Atari Games
- **Problema:** Imágenes de alta dimensión
- **Solución:** Usar Deep Q-Networks (DQN)

## ✅ Entornos Apropiados

### 1. MountainCar-v0 ⭐ (RECOMENDADO)

**Espacio de observación:** Box(2,) - Continuo
- `position`: [-1.2, 0.6]
- `velocity`: [-0.07, 0.07]

**Espacio de acciones:** Discrete(3)
- 0: Acelerar a la izquierda
- 1: No acelerar
- 2: Acelerar a la derecha

**Por qué es ideal:**
- ✅ 2 dimensiones continuas (fácil de visualizar)
- ✅ 3 acciones discretas
- ✅ Problema clásico de RL
- ✅ Requiere generalización (no se puede tabular)
- ✅ Características polinómicas funcionan bien

**Características sugeridas:**
```python
# Polinomio grado 2: [1, pos, vel, pos², vel², pos×vel]
# Total: 6 características × 3 acciones = 18 features
```

**Dificultad:** Media-Alta
- Recompensa: -1 por cada paso
- Objetivo: Llegar a la bandera (position ≥ 0.5)
- Desafío: El coche no tiene suficiente potencia para subir directamente

---

### 2. CartPole-v1

**Espacio de observación:** Box(4,) - Continuo
- `cart_position`: [-4.8, 4.8]
- `cart_velocity`: [-∞, ∞]
- `pole_angle`: [-0.418, 0.418] rad
- `pole_angular_velocity`: [-∞, ∞]

**Espacio de acciones:** Discrete(2)
- 0: Empujar carrito a la izquierda
- 1: Empujar carrito a la derecha

**Por qué es bueno:**
- ✅ 4 dimensiones continuas
- ✅ 2 acciones discretas
- ✅ Episodios cortos (más rápido de entrenar)
- ✅ Feedback inmediato

**Características sugeridas:**
```python
# Polinomio grado 2: [1, x1, x2, x3, x4, x1², x2², x3², x4², x1×x2, ...]
# Total: ~15 características × 2 acciones = 30 features
```

**Dificultad:** Baja-Media
- Recompensa: +1 por cada paso que el palo se mantiene vertical
- Objetivo: Mantener el palo vertical el mayor tiempo posible

---

### 3. Acrobot-v1

**Espacio de observación:** Box(6,) - Continuo
- `cos(θ1)`, `sin(θ1)`, `cos(θ2)`, `sin(θ2)`, `θ̇1`, `θ̇2`

**Espacio de acciones:** Discrete(3)
- 0: Torque negativo
- 1: Sin torque
- 2: Torque positivo

**Por qué es interesante:**
- ✅ 6 dimensiones continuas
- ✅ 3 acciones discretas
- ✅ Problema de control subactuado
- ✅ Más complejo que CartPole

**Características sugeridas:**
```python
# Características trigonométricas ya incluidas
# Polinomio grado 1-2
# Total: ~20-40 features
```

**Dificultad:** Media-Alta
- Recompensa: -1 por cada paso
- Objetivo: Balancear el acróbata hasta cierta altura

---

### 4. LunarLander-v2 (con acciones discretas)

**Espacio de observación:** Box(8,) - Continuo
- Posición x, y
- Velocidad x, y
- Ángulo, velocidad angular
- Contacto pata izquierda, derecha

**Espacio de acciones:** Discrete(4)
- 0: No hacer nada
- 1: Motor izquierdo
- 2: Motor principal
- 3: Motor derecho

**Por qué es desafiante:**
- ✅ 8 dimensiones continuas
- ✅ 4 acciones discretas
- ✅ Recompensas shaped (más fácil de aprender)
- ⚠️ Más complejo, puede requerir más características

**Características sugeridas:**
```python
# Tile coding o RBF recomendado
# Polinomio grado 1 como baseline
# Total: 50-200 features
```

**Dificultad:** Alta
- Recompensa: Shaped (por posición, velocidad, aterrizaje)
- Objetivo: Aterrizaje suave entre las banderas

---

## Comparación de Entornos

| Entorno | Obs. Dim | Acciones | Dificultad | Tiempo/Ep | Recomendado |
|---------|----------|----------|------------|-----------|-------------|
| **MountainCar-v0** | 2 | 3 | Media-Alta | Medio | ⭐⭐⭐⭐⭐ |
| **CartPole-v1** | 4 | 2 | Baja-Media | Corto | ⭐⭐⭐⭐ |
| **Acrobot-v1** | 6 | 3 | Media-Alta | Medio | ⭐⭐⭐ |
| **LunarLander-v2** | 8 | 4 | Alta | Largo | ⭐⭐ |

## Extractores de Características Recomendados

### Para MountainCar-v0

#### 1. Características Polinómicas (Grado 2) ⭐
```python
# [1, pos, vel, pos², vel², pos×vel]
# Simple y efectivo
```

#### 2. Tile Coding ⭐⭐⭐
```python
# 8 tilings, 8×8 tiles
# Mejor rendimiento, más complejo
```

#### 3. RBF (Funciones de Base Radial)
```python
# 10×10 centros gaussianos
# Buena aproximación suave
```

### Para CartPole-v1

#### 1. Características Polinómicas (Grado 1-2)
```python
# Grado 1: [1, x1, x2, x3, x4]
# Grado 2: + términos cuadráticos y cruzados
```

#### 2. Características Simples
```python
# [1, x1, x2, x3, x4] con one-hot para acciones
# Suficiente para este problema
```

## Consejos de Hiperparámetros

### MountainCar-v0
- **Alpha:** 0.05 - 0.2 (más alto que otros entornos)
- **Gamma:** 0.99 - 1.0
- **Epsilon:** 0.1 - 0.2
- **Episodios:** 500-1000

### CartPole-v1
- **Alpha:** 0.001 - 0.01 (más bajo, aprende rápido)
- **Gamma:** 0.99
- **Epsilon:** 0.05 - 0.1
- **Episodios:** 300-500

### Acrobot-v1
- **Alpha:** 0.01 - 0.1
- **Gamma:** 0.99
- **Epsilon:** 0.1
- **Episodios:** 500-1000

## Resumen

### Para empezar: **MountainCar-v0**
- Problema clásico de RL
- Demuestra la necesidad de aproximación de funciones
- Buen balance entre simplicidad y desafío

### Para experimentar: **CartPole-v1**
- Más fácil de resolver
- Feedback rápido
- Bueno para probar diferentes extractores

### Para desafío: **Acrobot-v1** o **LunarLander-v2**
- Más dimensiones
- Más complejo
- Requiere mejores características

## Nota Importante sobre "Control con Aproximaciones"

**SARSA Semi-Gradiente** es un método de **Control con Aproximaciones** porque:

1. ✅ **Control:** Aprende una política (no solo predice valores)
2. ✅ **Aproximaciones:** Usa aproximación de funciones (no tabla)
3. ✅ **Espacios continuos:** Diseñado para estados continuos
4. ✅ **Generalización:** Aprende patrones que se transfieren entre estados

**Pero requiere:**
- ❌ Acciones **discretas** (no continuas)
- ❌ Dimensionalidad **baja-media** (no imágenes)
- ❌ Aproximación **lineal** (no deep learning)

Para acciones continuas → Actor-Critic, DDPG, TD3
Para imágenes → DQN, A3C, PPO con CNNs
