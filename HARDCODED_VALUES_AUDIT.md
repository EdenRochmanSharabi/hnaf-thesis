# 🔍 AUDITORÍA COMPLETA DE VALORES HARDCODEADOS

## 📊 **RESUMEN EJECUTIVO**

He identificado **156+ valores hardcodeados** distribuidos en todo tu código HNAF. Estos valores están categorizados por **criticidad** y **ubicación**.

---

## 🎯 **1. main.py - VALORES HARDCODEADOS**

### ✅ **BAJO IMPACTO (Aceptables)**
```python
print("=" * 60)                    # Ancho del banner: 60 caracteres
sys.path.insert(0, project_root)   # Índice de inserción: 0
root.geometry("1400x900")          # Tamaño ventana: 1400x900 px
```

**Evaluación**: ✅ Estos son valores de presentación/UI, aceptables como hardcoded.

---

## 🖥️ **2. final_app/gui_interface.py - VALORES CRÍTICOS**

### 🔴 **ALTA CRITICIDAD - Parámetros de Modelo**
```python
# Arquitectura de Red Neuronal (MUY CRÍTICO)
self.state_dim_var = tk.IntVar(value=2)       # Dimensión estado
self.action_dim_var = tk.IntVar(value=2)      # Dimensión acción  
self.num_modes_var = tk.IntVar(value=2)       # Número de modos
self.hidden_dim_var = tk.IntVar(value=64)     # Neuronas ocultas
self.num_layers_var = tk.IntVar(value=3)      # Capas de red

# Parámetros de Entrenamiento (CRÍTICO)
self.learning_rate_var = tk.DoubleVar(value=0.0001)     # Learning rate
self.tau_var = tk.DoubleVar(value=0.00001)              # Soft update
self.gamma_var = tk.DoubleVar(value=0.9)                # Discount factor
self.num_episodes_var = tk.IntVar(value=1000)           # Episodios
self.batch_size_var = tk.IntVar(value=64)               # Batch size
self.initial_epsilon_var = tk.DoubleVar(value=0.5)      # Epsilon inicial
self.final_epsilon_var = tk.DoubleVar(value=0.05)       # Epsilon final
self.max_steps_var = tk.IntVar(value=50)                # Pasos máximos

# Buffer y Prioritización (CRÍTICO)
self.buffer_capacity_var = tk.StringVar(value="10000")  # Capacidad buffer
self.alpha_var = tk.StringVar(value="0.6")              # Prioridad alpha
self.beta_var = tk.StringVar(value="0.4")               # Sesgo beta
```

### 🟡 **MEDIA CRITICIDAD - Configuración**
```python
# Coordenadas y Matrices Iniciales
self.x0_var = tk.StringVar(value="1")                   # Coordenada X inicial
self.y0_var = tk.StringVar(value="1")                   # Coordenada Y inicial

# Matrices por defecto (PROBLEMÁTICO)
[[1, 50], [-1, 1]]    # Matriz A1
[[1, -1], [50, 1]]    # Matriz A2

# Función de recompensa por defecto
self.reward_expr_var = tk.StringVar(value="np.linalg.norm([x, y])")
```

### 🟢 **BAJA CRITICIDAD - Rangos de Spinboxes**
```python
# Rangos hardcodeados en controles
from_=1, to=10          # Dimensiones estado/acción
from_=16, to=256        # Hidden dimensions
from_=0.0001, to=0.1    # Learning rate range
from_=0.8, to=0.999     # Gamma range
from_=100, to=5000      # Episodios range
```

---

## 🏋️ **3. final_app/training_manager.py - VALORES CRÍTICOS**

### 🔴 **ALTA CRITICIDAD**
```python
# Seeds hardcodeados (PROBLEMÁTICO para reproducibilidad)
np.random.seed(42)
torch.manual_seed(42)

# Intervalos de evaluación
eval_interval = 50                    # Cada 50 episodios
num_episodes=10                      # 10 episodios para evaluación
grid_size=50                         # Grid 50x50 para evaluación

# Hardcoded en progreso
progress = (total_episodes / 1000) * 100    # Asume 1000 episodios
```

### 🟡 **MEDIA CRITICIDAD**
```python
# Grid de evaluación hardcodeado
print(f"  - Evaluación en grid 100x100")   # Debería ser configurable
```

---

## 🧠 **4. src/hnaf_improved.py - VALORES CRÍTICOS**

### 🔴 **ALTA CRITICIDAD - Parámetros del Modelo**
```python
# Inicialización de red (CRÍTICO)
def __init__(self, state_dim=2, action_dim=2, num_modes=2,
             hidden_dim=64, num_layers=3, lr=1e-6, tau=1e-5, gamma=0.9,
             buffer_capacity=10000, alpha=0.6, beta=0.4):

# Límites de clipping (MUY CRÍTICO)
next_state = np.clip(next_state, -5.0, 5.0)    # Límites de estado

# Tolerancias numéricas (CRÍTICO)
+ 1e-8                                          # División por cero
+ 1e-6                                          # Prioridad mínima
min=1e-6                                        # Clamp diagonal

# Inicialización de parámetros
gain=1.0, gain=0.1                             # Gains de inicialización
torch.clamp(l_params[:, idx], -3, 3)          # Límites L matrix
torch.clamp(l_params[:, idx], -1, 1)          # Límites off-diagonal
+ 0.1                                          # Offset diagonal L
```

### 🟡 **MEDIA CRITICIDAD - Reward Shaping**
```python
# Coeficientes de reward shaping (hardcodeados)
* 0.1         # Penalización distancia
* 0.05        # Bonus acercamiento  
* 0.2         # Penalización modo subóptimo
* 0.1         # Penalización oscilación
+ 1.0         # Bonus éxito episodio (antes era 5.0)
```

### 🟢 **BAJA CRITICIDAD - Matrices de Inicialización**
```python
# Matrices iniciales (se sobrescriben desde GUI)
np.array([[0, 0], [0, 0]])    # A1 inicial (placeholder)
np.array([[0, 0], [0, 0]])    # A2 inicial (placeholder)
```

---

## 🔧 **5. src/naf_corrected.py - VALORES MODERADOS**

### 🟡 **MEDIA CRITICIDAD**
```python
# Parámetro temporal hardcodeado
def __init__(self, t=1.0):                     # Tiempo t=1.0

# Matrices de prueba (para testing)
np.array([[1, 50], [-1, 1]])                  # A1 test
np.array([1, 1]), np.array([0, 1])           # Estados test

# Tolerancias de comparación
< 1e-10                                        # Tolerancia muy estricta
< 1e-3                                         # Tolerancia normal
```

---

## 📊 **6. Otros Archivos - VALORES DISPERSOS**

### 🟡 **optimization_functions.py**
```python
# Matrices por defecto (PROBLEMÁTICO)
self.A1 = np.array([[1, 50], [-1, 1]])
self.A2 = np.array([[1, -1], [50, 1]])
self.x0, self.y0 = 1, 1                      # Coordenadas por defecto
```

### 🟢 **Archivos de Demo/Test**
```python
# Valores en archivos de prueba (menos críticos)
hidden_dim=32, lr=1e-4                        # hnaf_stable.py
/15.0                                         # Normalizaciones
alpha_deg = np.random.uniform(0, 360)        # Rangos de prueba
```

---

## 🚨 **PROBLEMAS IDENTIFICADOS**

### 🔴 **CRÍTICOS (Requieren atención inmediata)**

1. **Límites de estado**: `-5.0, 5.0` en clipping
2. **Seeds fijos**: `42` impide variabilidad en experimentos
3. **Matrices por defecto**: `[[1, 50], [-1, 1]]` muy específicas
4. **Tolerancias numéricas**: `1e-6, 1e-8` podrían ser configurables
5. **Intervalos de evaluación**: `50 episodios` hardcodeado

### 🟡 **MODERADOS (Mejora recomendada)**

1. **Parámetros de reward shaping**: `0.1, 0.05, 0.2` hardcodeados
2. **Rangos de spinboxes**: Límites fijos en GUI
3. **Grid de evaluación**: `50x50, 100x100` hardcodeados
4. **Coeficientes de inicialización**: `gain=1.0, gain=0.1`

### 🟢 **BAJOS (Aceptables pero mejorables)**

1. **Dimensiones de ventana**: `1400x900`
2. **Anchura de banners**: `60 caracteres`
3. **Valores de placeholder**: Matrices `[[0,0],[0,0]]`

---

## 🎯 **RECOMENDACIONES DE PRIORIDAD**

### **Prioridad 1 (Inmediata)** 🔴
- Hacer configurables los límites de clipping (-5.0, 5.0)
- Parametrizar seeds (42) desde GUI
- Configurar matrices por defecto desde archivo config

### **Prioridad 2 (Corto plazo)** 🟡  
- Parametrizar tolerancias numéricas (1e-6, 1e-8)
- Hacer configurables intervalos de evaluación (50)
- Parametrizar coeficientes de reward shaping

### **Prioridad 3 (Largo plazo)** 🟢
- Crear archivo de configuración centralizado
- Parametrizar rangos de spinboxes
- Configurar dimensiones de ventana

---

## 💡 **PROPUESTA DE SOLUCIÓN**

Crear un **sistema de configuración centralizado** que permita:

1. **Archivo `config.yaml`** con todos los parámetros
2. **GUI de configuración avanzada** para expertos
3. **Perfiles predefinidos** (principiante, intermedio, experto)
4. **Validación automática** de rangos de parámetros

¿Quieres que empiece implementando las correcciones de **Prioridad 1**?