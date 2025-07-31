# HNAF - Aplicación Modular

## 📁 Estructura del Proyecto

```
final_app/
├── main.py                    # Punto de entrada principal
├── gui_interface.py           # Interfaz gráfica
├── training_manager.py        # Lógica de entrenamiento
├── evaluation_manager.py      # Evaluación de modelos
├── visualization_manager.py   # Gráficos y visualizaciones
└── README.md                 # Este archivo
```

## 🚀 Cómo Ejecutar

### Opción 1: Desde el directorio raíz
```bash
cd final_app
python main.py
```

### Opción 2: Desde el directorio padre
```bash
python final_app/main.py
```

## 🧩 Arquitectura Modular

### 1. **main.py** - Punto de Entrada
- Inicializa la aplicación
- Configura el path para importar módulos
- Crea la ventana principal
- Maneja el cierre limpio

### 2. **gui_interface.py** - Interfaz Gráfica
- Maneja toda la interfaz de usuario
- Controles de parámetros
- Editor de funciones personalizadas
- Visualización en tiempo real
- **No contiene lógica de entrenamiento**

### 3. **training_manager.py** - Entrenamiento
- Maneja toda la lógica de entrenamiento
- Creación del modelo HNAF
- Configuración de parámetros
- Ejecución de episodios
- **Separado de la interfaz**

### 4. **evaluation_manager.py** - Evaluación
- Evaluación de modelos entrenados
- Comparación entre modelos
- Verificación de funcionamiento
- Métricas de rendimiento

### 5. **visualization_manager.py** - Visualización
- Gráficos de entrenamiento
- Mapas de calor
- Comparaciones visuales
- Guardado de gráficos

## 🎯 Ventajas de la Arquitectura Modular

### ✅ **Separación de Responsabilidades**
- GUI solo maneja la interfaz
- Entrenamiento solo maneja la lógica
- Evaluación solo maneja métricas
- Visualización solo maneja gráficos

### ✅ **Mantenibilidad**
- Cada módulo es independiente
- Fácil de modificar sin afectar otros
- Código más limpio y organizado

### ✅ **Reutilización**
- Módulos pueden usarse independientemente
- Fácil de integrar en otros proyectos
- Testing individual por módulo

### ✅ **Escalabilidad**
- Fácil agregar nuevos módulos
- Fácil modificar funcionalidades
- Fácil de limpiar y reorganizar

## 🔧 Uso de los Módulos

### Entrenamiento Independiente
```python
from training_manager import TrainingManager

manager = TrainingManager()
params = {
    'state_dim': 2,
    'action_dim': 2,
    'num_modes': 2,
    'hidden_dim': 64,
    'num_layers': 3,
    'lr': 1e-4,
    'tau': 0.001,
    'gamma': 0.9,
    'num_episodes': 1000,
    'batch_size': 32,
    'initial_epsilon': 0.5,
    'final_epsilon': 0.05,
    'max_steps': 50
}

model, results = manager.train_hnaf(params)
```

### Evaluación Independiente
```python
from evaluation_manager import EvaluationManager

eval_manager = EvaluationManager()
results = eval_manager.evaluate_model(model)
```

### Visualización Independiente
```python
from visualization_manager import VisualizationManager

viz_manager = VisualizationManager()
fig = viz_manager.create_comparison_plot(results1, results2)
```

## 📊 Funcionalidades

### 🎮 **Interfaz Gráfica**
- Parámetros configurables
- Editor de funciones personalizadas
- Visualización en tiempo real
- Barra de progreso
- Salida de terminal integrada

### 🧠 **Entrenamiento Avanzado**
- ε-greedy decay (0.5 → 0.05)
- Prioritized Experience Replay
- Normalización de estados y recompensas
- Reward shaping local
- Red neuronal profunda (3 capas, 64 unidades)

### 📈 **Evaluación Robusta**
- Evaluación en grid 100x100
- Métricas de precisión
- Comparación entre modelos
- Verificación de funcionamiento

### 📊 **Visualización Completa**
- Gráficos de recompensas
- Mapas de calor de decisiones
- Comparaciones visuales
- Guardado de gráficos

## 🛠️ Dependencias

```bash
pip install torch numpy matplotlib scipy tkinter
```

## 🎯 Resultados Esperados

Con esta arquitectura modular, obtienes:
- **Código más limpio** y organizado
- **Fácil mantenimiento** y debugging
- **Reutilización** de componentes
- **Escalabilidad** para futuras mejoras
- **Testing individual** por módulo

## 🔄 Flujo de Trabajo

1. **Ejecutar aplicación**: `python final_app/main.py`
2. **Configurar parámetros** en la interfaz
3. **Iniciar entrenamiento** con el botón
4. **Evaluar modelo** cuando termine
5. **Verificar funcionamiento** si es necesario
6. **Analizar gráficos** para entender resultados

¡La aplicación modular está lista para usar! 🚀 