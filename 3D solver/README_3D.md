# HNAF 3D - Sistema Autosuficiente

## 🎯 **Descripción**

Este directorio contiene una implementación completa y autosuficiente del sistema HNAF (Hybrid Neural Adaptive Feedback) para sistemas 3D. Todos los archivos necesarios están incluidos para ejecutar entrenamiento, evaluación y generación de resultados.

## 📁 **Estructura de Archivos**

### 🔧 **Archivos Principales**
- `app.py` - Aplicación principal
- `training_monitor.py` - Monitor de entrenamiento
- `training_manager.py` - Gestor de entrenamiento
- `hnaf_improved.py` - Modelo HNAF mejorado
- `config_manager.py` - Gestor de configuración
- `logging_manager.py` - Gestor de logs

### 🎮 **Interfaz y GUI**
- `gui_interface.py` - Interfaz gráfica
- `app.py` - Punto de entrada principal

### 📊 **Evaluación y Resultados**
- `evaluate_policy.py` - Evaluación de políticas
- `evaluate_current_model.py` - Evaluación del modelo actual
- `generate_thesis_results.py` - Generación de resultados para tesis
- `generate_final_results.py` - Generación de resultados finales
- `reporting_utils.py` - Utilidades de reportes
- `plot_utils.py` - Utilidades de gráficos

### 🔬 **Optimización**
- `optuna_optimizer.py` - Optimizador Optuna
- `run_multiple_trainings.py` - Múltiples entrenamientos
- `run_trainings_with_monitor.py` - Entrenamientos con monitor

### 📋 **Configuración y Logs**
- `config.yaml` - Configuración del sistema
- `detailed_logger.py` - Logger detallado
- `noise_process.py` - Procesos de ruido

### 🚀 **Scripts de Automatización**
- `run_complete_pipeline.sh` - Pipeline completo
- `monitor_3d_training.py` - Monitor específico para 3D

### 🧪 **Testing**
- `test_detailed_logging.py` - Tests de logging
- `test_optimization.py` - Tests de optimización

## 🚀 **Cómo Usar**

### 1. **Entrenamiento Básico**
```bash
python training_monitor.py --run 1
```

### 2. **Entrenamiento con Monitor**
```bash
python run_trainings_with_monitor.py
```

### 3. **Pipeline Completo**
```bash
./run_complete_pipeline.sh
```

### 4. **Evaluación de Modelo**
```bash
python evaluate_policy.py --auto-find-model
```

### 5. **Generación de Resultados**
```bash
python generate_thesis_results.py --auto
```

### 6. **Interfaz Gráfica**
```bash
python app.py
```

## ⚙️ **Configuración**

### Matrices 3D (Forma de Jordan)
El sistema está configurado para usar matrices 3x3 individualmente inestables:

```yaml
A1: [[1.0, 0.0, 0.0], [0.0, 2.0, -1.0], [0.0, 1.0, 2.0]]
A2: [[1.0, 0.0, 0.0], [0.0, 2.0, -1.0], [0.0, 1.0, 2.0]]
A3: [[1.0, 0.0, 0.0], [0.0, 2.0, -1.0], [0.0, 1.0, 2.0]]
```

### Dimensiones del Sistema
- **State Dimension:** 3
- **Action Dimension:** 3
- **Number of Modes:** 3

## 🔍 **Problema Actual**

**Ver archivo:** `PROBLEMA_MATRICES_3D.md`

El sistema está configurado correctamente para 3D, pero hay un problema donde el código sigue leyendo matrices 2x2 en lugar de 3x3. Esto causa el error:

```
ERROR en episodio supervisado 1: mat1 and mat2 shapes cannot be multiplied (1x2 and 3x512)
```

## 📊 **Resultados**

Los resultados se guardan en:
- `logs/` - Logs del sistema
- `resultados_evaluacion_*/` - Resultados de evaluación
- `tesis_resultados_*/` - Resultados para tesis

## 🛠️ **Troubleshooting**

### Limpiar Cache
```bash
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete
```

### Verificar Configuración
```bash
python -c "from config_manager import get_config_manager; cm = get_config_manager(); print(f'A1: {cm.get(\"defaults.matrices.A1\")}')"
```

### Debug de Matrices
```bash
grep -r "range(2)" *.py
grep -r "\[\[.*\],.*\[.*\]\]" *.py
```

## 📚 **Documentación**

- `README_TESIS.md` - Documentación para tesis
- `PROBLEMA_MATRICES_3D.md` - Análisis del problema actual

## 🎯 **Objetivo**

Completar el entrenamiento 3D exitosamente para encontrar una ley de conmutación estabilizadora para sistemas híbridos 3D.

---
*Sistema HNAF 3D - Autosuficiente*
*Última actualización: 2025-08-05* 