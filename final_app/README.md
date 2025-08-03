# HNAF Application - Versión Final

Esta carpeta contiene la aplicación HNAF completamente funcional y optimizada.

## 📁 Estructura de Archivos

### Archivos Principales
- **`app.py`** - Punto de entrada principal de la aplicación
- **`gui_interface.py`** - Interfaz gráfica completa
- **`training_manager.py`** - Gestor de entrenamiento optimizado
- **`config_manager.py`** - Gestor de configuración sin hardcode
- **`optuna_optimizer.py`** - Optimización automática de hiperparámetros
- **`logging_manager.py`** - Sistema de logging estructurado
- **`config.yaml`** - Configuración centralizada

## 🚀 Uso Rápido

### Ejecutar GUI (Interfaz Gráfica)
```bash
python app.py
```

### Ejecutar en Modo CLI (Línea de Comandos)
```bash
python app.py --cli
```

### Solo Entrenamiento
```bash
python app.py --train --iterations 3
```

### Optimización Automática
```bash
python app.py --optimize
```

### Análisis de Estabilidad
```bash
python app.py --stability
```

### Loop de Mejora Automática
```bash
python app.py --improve --iterations 5 --target 80.0
```

## 📊 Funcionalidades Principales

### 1. **GUI Completa**
- Interfaz gráfica moderna y responsive
- Configuración dinámica desde `config.yaml`
- Visualización de resultados en tiempo real
- Optimización integrada con Optuna
- Auto-recarga de configuración

### 2. **Entrenamiento Avanzado**
- HNAF mejorado con estabilización
- Gradient clipping para prevenir explosión de gradientes
- Normalización de estados y recompensas
- Curriculum learning automático
- Warm-up supervisado balanceado

### 3. **Optimización Automática**
- Búsqueda de hiperparámetros con Optuna
- Actualización automática de configuración
- Análisis de rendimiento detallado
- Backup automático de configuraciones

### 4. **Configuración Centralizada**
- Todo configurado desde `config.yaml`
- Sin valores hardcodeados
- Auto-recarga de cambios
- Validación estricta de parámetros

### 5. **Estabilización Avanzada**
- Recompensas normalizadas con `tanh`
- Clipping de gradientes
- Normalización de estados
- Manejo robusto de errores

## 🔧 Configuración

La configuración se maneja completamente a través de `config.yaml`:

- **Parámetros de red**: Dimensiones, capas, inicialización
- **Parámetros de entrenamiento**: Learning rate, episodios, batch size
- **Funciones de recompensa**: Expresiones personalizables
- **Matrices de transformación**: A1 y A2 configurables
- **Optimización**: Parámetros de Optuna y estabilización

## 📈 Monitoreo y Logging

### Sistema de Logs
- Logs estructurados en `logs/`
- Niveles configurables (DEBUG, INFO, WARNING, ERROR)
- Rotación automática de archivos
- Contexto detallado de errores

### Métricas en Tiempo Real
- Precisión del modelo
- Estabilidad del sistema
- Pérdidas durante entrenamiento
- Balance de modos
- Progreso de optimización

## 🎯 Características Técnicas

### Estabilización
- **Gradient Clipping**: `max_norm=1.0`
- **Recompensas Normalizadas**: Rango [-1, 1] con `tanh`
- **Normalización de Estados**: Media=0, Std=1
- **Clipping de Recompensas**: Prevención de valores extremos

### Optimizaciones
- **Prioritized Experience Replay**: Con parámetros α y β
- **ε-greedy Decay**: Exploración controlada
- **Curriculum Learning**: Dificultad progresiva
- **Warm-up Supervisado**: Enseñanza balanceada

### Configuración Dinámica
- **Auto-recarga**: Monitoreo de cambios en `config.yaml`
- **Validación Estricta**: Sin fallbacks, errores inmediatos
- **Backup Automático**: Preservación de configuraciones
- **Sincronización GUI**: Cambios reflejados automáticamente

## 🛠️ Comandos Útiles

### Entrenamiento Rápido
```bash
python app.py --train --iterations 1
```

### Optimización Completa
```bash
python app.py --optimize
```

### Análisis de Estabilidad
```bash
python app.py --stability
```

### Mejora Automática
```bash
python app.py --improve --iterations 10 --target 85.0
```

## 📋 Funcionalidades de la GUI

### Controles Principales
- **Iniciar Entrenamiento**: Entrenamiento completo con GUI
- **Evaluar Modelo**: Análisis de rendimiento
- **Verificar HNAF**: Validación del modelo
- **Limpiar Salida**: Limpieza de logs

### Configuración Dinámica
- **Recargar Config**: Carga manual desde `config.yaml`
- **Auto-recargar**: Monitoreo automático de cambios
- **Mostrar Config**: Visualización de configuración actual
- **Guardar**: Persistencia de cambios en `config.yaml`

### Optimización
- **Optimización Optuna**: Búsqueda automática de hiperparámetros
- **Aplicar a Config.yaml**: Actualización automática de configuración
- **Cargar Mejores Parámetros**: Aplicación de resultados optimizados

## 🎯 Objetivos de Rendimiento

- **Precisión objetivo**: >80%
- **Estabilidad objetivo**: >70%
- **Balance de modos**: 50/50 ±10%
- **Pérdidas controladas**: <10K

## 📞 Soporte

### Logs Detallados
Los logs se guardan en `logs/` con información completa de:
- Entrenamiento y optimización
- Errores y advertencias
- Análisis de estabilidad
- Configuración y cambios

### Debugging
Para obtener información detallada:
```bash
python app.py --cli --iterations 1
```

### Configuración
- Verificar `config.yaml` para parámetros
- Usar "Mostrar Config" en la GUI
- Revisar logs para errores específicos

---

**¡La aplicación está completamente funcional y lista para usar!** 🚀

**Características destacadas:**
- ✅ **Estable**: Gradient clipping y normalización
- ✅ **Configurable**: Todo desde `config.yaml`
- ✅ **Optimizada**: Auto-recarga y sincronización
- ✅ **Robusta**: Manejo completo de errores
- ✅ **Completa**: GUI, CLI y optimización automática 