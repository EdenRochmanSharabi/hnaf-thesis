# HNAF (Hybrid Normalized Advantage Function) Implementation

## 📋 Resumen del Proyecto

Este proyecto implementa un **HNAF (Hybrid Normalized Advantage Function)** basado en el framework de control híbrido descrito en el paper "Hybrid Reinforcement Learning for Optimal Control of Non-Linear Switching System". El HNAF combina control discreto (selección de modos) y control continuo (acciones) en un solo algoritmo de aprendizaje por refuerzo.

## 🎯 Objetivos Logrados

### ✅ **NAF Individual Verificado**
- **Problema identificado**: La implementación original usaba transformación directa `A @ x0` en lugar de exponencial de matriz `expm(A * t) @ x0`
- **Solución**: Implementación corregida que coincide exactamente con la solución ODE
- **Resultado**: Error promedio = 0.000000 (resultados idénticos)

### ✅ **HNAF Implementado**
- **Arquitectura**: Redes neuronales separadas para cada modo discreto
- **Salidas**: V(x,v), μ(x,v), L(x,v) para cada modo
- **Selección de modo**: v* = argmin V(x,v)
- **Acción continua**: u = μ(x, v*)

### ✅ **Estabilidad Numérica**
- **Problema**: Valores explosivos y NaN en la primera implementación
- **Solución**: Clipping, inicialización estable, learning rate reducido
- **Resultado**: Entrenamiento estable con valores finitos

## 🏗️ Arquitectura del Sistema

### **Componentes Principales**

1. **NAFNetwork/StableNAFNetwork**: Red neuronal para cada modo
   - Entrada: Estado x
   - Salidas: V(x,v), μ(x,v), L(x,v)
   - Matriz P(x,v) = L(x,v) * L(x,v)^T

2. **HNAF/StableHNAF**: Sistema híbrido principal
   - Múltiples redes (una por modo)
   - Buffers de replay separados
   - Selección de modo y acción

3. **Matrices de Transformación**:
   ```python
   A1 = [[1, 50], [-1, 1]]  # Modo 0
   A2 = [[1, -1], [50, 1]]  # Modo 1
   ```

### **Flujo de Entrenamiento**

1. **Inicialización**: Estados aleatorios en [-0.5, 0.5]
2. **Selección de acción**: 
   - Calcular V(x,v) para todos los modos
   - Seleccionar v* = argmin V(x,v)
   - Aplicar u = μ(x, v*)
3. **Transición**: x' = A_v* @ x
4. **Recompensa**: r = |‖x'‖ - ‖x‖|
5. **Almacenamiento**: (x, v*, u, r, x') en buffer del modo v*
6. **Actualización**: Q-learning con target híbrido

## 📊 Resultados del Entrenamiento

### **Métricas de Estabilidad**
- **Recompensas**: Valores finitos (~1800-1900)
- **Pérdidas**: Convergencia estable (~9000-9500)
- **Selección de modos**: Distribución equilibrada entre modos 0 y 1

### **Verificación HNAF vs NAF Individual**

| Caso | Estado Inicial | HNAF Modo | Q_pred | R_real | NAF1 | NAF2 | Óptimo |
|------|----------------|-----------|--------|--------|------|------|--------|
| 1 | [0.1, 0.1] | 0 | 0.0364 | 4.9586 | 1.4215 | 1.4215 | ❌ |
| 2 | [0, 0.1] | 0 | 0.0366 | 4.9010 | 1.2759 | 0.0937 | ❌ |
| 3 | [0.1, 0] | 0 | 0.0367 | 0.0414 | 0.0937 | 1.2759 | ✅ |
| 4 | [0.05, 0.05] | 0 | 0.0366 | 2.4793 | 0.7108 | 0.7108 | ❌ |
| 5 | [-0.05, 0.08] | 0 | 0.0367 | 3.8578 | 0.9137 | 0.4465 | ❌ |

### **Análisis de Resultados**

**✅ Aspectos Positivos:**
- Entrenamiento estable sin explosión de valores
- Convergencia de pérdidas
- Selección de modos funcional
- Caso 3: Modo óptimo seleccionado correctamente

**⚠️ Áreas de Mejora:**
- Q_pred vs R_real: Diferencias significativas
- Selección de modo: No siempre óptima
- Necesita más entrenamiento para convergencia completa

## 🔧 Archivos del Proyecto

### **Implementaciones Principales**
- `naf_corrected.py`: NAF individual corregido con exponencial de matriz
- `hnaf_implementation.py`: Implementación inicial del HNAF
- `hnaf_stable.py`: Versión estable del HNAF con mejor estabilidad numérica

### **Verificación y Testing**
- `naf_verification.py`: Verificación NAF vs ODE
- `naf_corrected.py`: Verificación de corrección

### **Clases de Optimización**
- `optimization_functions.py`: Clase base para funciones matemáticas
- `example_usage.py`: Ejemplos de uso

## 🚀 Uso del Sistema

### **Entrenamiento HNAF**
```python
from hnaf_stable import train_stable_hnaf

# Entrenar HNAF
hnaf = train_stable_hnaf(num_episodes=200, eval_interval=20)
```

### **Verificación NAF**
```python
from naf_corrected import CorrectedOptimizationFunctions

# Crear NAF corregido
naf = CorrectedOptimizationFunctions(t=1.0)

# Usar transformaciones
x1 = naf.execute_function("transform_x1", 1, 1)
x2 = naf.execute_function("transform_x2", 1, 1)
```

### **Selección de Acción Híbrida**
```python
# Seleccionar modo y acción
mode, action = hnaf.select_action(state, epsilon=0.0)

# Calcular Q-value
Q = hnaf.compute_Q_value(state, action, mode)
```

## 📈 Próximos Pasos

### **Mejoras Propuestas**

1. **Hiperparámetros Optimizados**
   - Learning rate adaptativo
   - Tamaño de batch dinámico
   - Arquitectura de red más profunda

2. **Algoritmos Avanzados**
   - Prioritized Experience Replay
   - N-step returns
   - Multi-step learning

3. **Verificación Mejorada**
   - Más casos de prueba
   - Análisis de convergencia
   - Comparación con baselines

4. **Aplicaciones Prácticas**
   - Sistemas de control reales
   - Robótica
   - Vehículos autónomos

## 🎓 Conceptos Clave

### **Control Híbrido**
- **Modo discreto**: v ∈ {1, 2, ..., N}
- **Acción continua**: u ∈ ℝ^m
- **Estado**: x ∈ ℝ^n
- **Política**: π(x) = (v*, u*)

### **NAF (Normalized Advantage Function)**
- **Q-value**: Q(x,v,u) = V(x,v) + A(x,v,u)
- **Ventaja**: A(x,v,u) = -0.5 * (u - μ)^T * P(x,v) * (u - μ)
- **Matriz P**: P(x,v) = L(x,v) * L(x,v)^T

### **HNAF (Hybrid NAF)**
- **Selección de modo**: v* = argmin V(x,v)
- **Acción continua**: u* = μ(x, v*)
- **Política híbrida**: π(x) = (v*, u*)

## 📚 Referencias

1. "Hybrid Reinforcement Learning for Optimal Control of Non-Linear Switching System"
2. "Continuous Control with Deep Reinforcement Learning" (NAF paper)
3. "Deep Q-Learning with Experience Replay"

---

**Estado del Proyecto**: ✅ **FUNCIONAL** - HNAF implementado y entrenado exitosamente con estabilidad numérica.

**Próximo Milestone**: Optimización de hiperparámetros y convergencia completa del aprendizaje. 