# HNAF Mejorado - Optimizaciones Avanzadas

## 🚀 Mejoras Implementadas

Este proyecto incluye una versión mejorada del HNAF (Hybrid Normalized Advantage Function) con optimizaciones avanzadas que mejoran significativamente el rendimiento y la estabilidad del entrenamiento.

### 📊 Comparación: Original vs Mejorado

| Aspecto | HNAF Original | HNAF Mejorado |
|---------|---------------|----------------|
| **ε-greedy** | ε fijo = 0.2 | ε decay: 0.5 → 0.05 |
| **Red Neuronal** | 1 capa, 32 unidades | 3 capas, 64 unidades + BatchNorm |
| **Experience Replay** | Buffer uniforme 5000 | Buffer priorizado 10000 |
| **Normalización** | Sin normalización | Estados y recompensas normalizados |
| **Reward Shaping** | r = -\|‖x'‖-‖x₀‖\|/15 | + bonus por modo óptimo |
| **Evaluación** | Solo episodios | Grid 100x100 + episodios |
| **Horizonte** | 20 pasos/episodio | 50 pasos/episodio |

## 🎯 Optimizaciones Detalladas

### 1. ε-greedy Decay Lineal
```python
# Original: ε fijo
epsilon = 0.2

# Mejorado: ε decay
epsilon = max(final_epsilon, initial_epsilon - episode * decay_rate)
```
**Beneficio**: Más exploración al inicio, más explotación al final.

### 2. Red Neuronal Profunda
```python
# Original: 1 capa
self.fc1 = nn.Linear(state_dim, 32)
self.fc2 = nn.Linear(32, hidden_dim)

# Mejorado: 3 capas + BatchNorm
self.layers = nn.ModuleList([
    nn.Linear(state_dim, hidden_dim),
    nn.Linear(hidden_dim, hidden_dim),
    nn.Linear(hidden_dim, hidden_dim)
])
self.batch_norms = nn.ModuleList([
    nn.BatchNorm1d(hidden_dim) for _ in range(3)
])
```
**Beneficio**: Mayor capacidad de aprendizaje y estabilidad.

### 3. Prioritized Experience Replay
```python
# Original: Buffer uniforme
self.buffer = deque(maxlen=5000)

# Mejorado: Buffer priorizado
class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # Prioridad exponencial
        self.beta = beta    # Corrección de sesgo
```
**Beneficio**: Enfoque en transiciones difíciles y mejor aprendizaje.

### 4. Normalización de Estados y Recompensas
```python
# Normalización de estados
normalized_state = (state - state_mean) / (state_std + 1e-8)

# Normalización de recompensas
normalized_reward = (reward - reward_mean) / (reward_std + 1e-8)
```
**Beneficio**: Entrenamiento más estable y convergencia más rápida.

### 5. Reward Shaping Local
```python
# Original
reward = -abs(reward) / 15.0

# Mejorado
reward_base = -abs(reward) / 15.0
mode_bonus = 0.1 if mode == optimal_mode else 0.0
reward_final = reward_base + mode_bonus
```
**Beneficio**: Guía adicional para la selección correcta de modos.

### 6. Evaluación Mejorada
```python
# Evaluación en grid 100x100
def evaluate_policy_grid(self, grid_size=100):
    x = np.linspace(-0.5, 0.5, grid_size)
    y = np.linspace(-0.5, 0.5, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Calcular precisión en cada punto
    optimal_accuracy = sum(mode == optimal_mode for mode in mode_selections)
    return optimal_accuracy / (grid_size * grid_size)
```
**Beneficio**: Métrica más robusta de precisión del modelo.

### 7. Horizonte Más Largo
```python
# Original: 20 pasos
max_steps = 20

# Mejorado: 50 pasos
max_steps = 50
```
**Beneficio**: Mejor comprensión de las consecuencias de las acciones.

## 🎮 Uso de la GUI Mejorada

### Configuración Recomendada

1. **HNAF Mejorado**: ✅ Activado por defecto
2. **Parámetros de Red**:
   - Dimensión oculta: 64
   - Número de capas: 3
3. **Parámetros de Entrenamiento**:
   - ε inicial: 0.5
   - ε final: 0.05
   - Max Steps: 50
   - Learning Rate: 1e-4

### Funciones Personalizadas

La GUI permite definir funciones personalizadas de transformación y recompensa:

```python
# Ejemplo de funciones personalizadas
def transform_mode_0(x0, y0):
    """Transformación para el modo 0"""
    A = np.array([[1, 50], [-1, 1]])
    result = A @ np.array([[x0], [y0]])
    return result[0, 0], result[1, 0]

def reward_function(x, y, x0, y0):
    """Función de recompensa personalizada"""
    norm_xy = np.linalg.norm([x, y])
    norm_x0y0 = np.linalg.norm([x0, y0])
    return abs(norm_xy - norm_x0y0)

transformation_functions = [transform_mode_0, transform_mode_1]
```

## 📈 Métricas de Rendimiento

### Métricas Disponibles

1. **Recompensa de Episodio**: Promedio de recompensas por episodio
2. **Recompensa de Evaluación**: Recompensa en episodios de evaluación
3. **Precisión en Grid**: Porcentaje de decisiones correctas en grid 100x100
4. **Pérdida de Entrenamiento**: Pérdida de la red neuronal
5. **Distribución de Modos**: Frecuencia de selección de cada modo

### Visualización Mejorada

La GUI ahora muestra:
- Gráfico de recompensas con promedio móvil
- Gráfico de precisión en grid (HNAF mejorado)
- Métricas en tiempo real durante el entrenamiento

## 🧪 Scripts de Demostración

### `demo_mejorado.py`
Compara el rendimiento del HNAF original vs mejorado:
```bash
python demo_mejorado.py
```

### `demo_completo.py`
Demuestra el funcionamiento completo del HNAF:
```bash
python demo_completo.py
```

## 🚀 Ejecución

### GUI Principal
```bash
python hnaf_gui.py
```

### Entrenamiento Directo
```python
from src.hnaf_improved import train_improved_hnaf

# Entrenar HNAF mejorado
hnaf, results = train_improved_hnaf(
    num_episodes=1000,
    initial_epsilon=0.5,
    final_epsilon=0.05,
    hidden_dim=64,
    num_layers=3,
    lr=1e-4
)
```

## 📊 Resultados Esperados

Con las optimizaciones implementadas, se espera:

1. **Convergencia más rápida**: 20-30% menos episodios para convergencia
2. **Mejor precisión**: 85-95% de precisión en grid de evaluación
3. **Estabilidad mejorada**: Menos variabilidad en recompensas
4. **Exploración más eficiente**: Mejor balance exploración/explotación

## 🔧 Dependencias

```bash
pip install torch numpy matplotlib scipy tkinter
```

## 📝 Notas Técnicas

- **Reproducibilidad**: Semilla fija (42) para resultados consistentes
- **Compatibilidad**: Mantiene compatibilidad con implementación original
- **Escalabilidad**: Fácil extensión a más modos y dimensiones
- **Debugging**: Logs detallados para diagnóstico de problemas

## 🎯 Próximas Mejoras

1. **Multi-agent HNAF**: Extensión a múltiples agentes
2. **Curriculum Learning**: Entrenamiento progresivo de dificultad
3. **Meta-learning**: Adaptación rápida a nuevos entornos
4. **Distributed Training**: Entrenamiento distribuido para grandes redes 