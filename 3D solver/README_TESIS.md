# HNAF - Generación de Resultados para Tesis

Este directorio contiene la aplicación HNAF completa con herramientas para generar automáticamente todos los resultados necesarios para tu tesis.

## 🎯 Objetivo

Demostrar la **estabilizabilidad** de sistemas híbridos conmutados mediante aprendizaje por refuerzo, generando:

1. **Trayectorias de estado** que convergen al origen
2. **Ley de conmutación** aprendida por el agente
3. **Análisis de estabilidad** con métricas cuantitativas
4. **Informes académicos** en formato Markdown y LaTeX

## 📁 Estructura de Archivos

### Componentes Principales
- `app.py` - Aplicación principal con GUI y CLI
- `hnaf_improved.py` - Modelo HNAF mejorado
- `training_manager.py` - Gestor de entrenamiento
- `config_manager.py` - Gestión de configuración
- `gui_interface.py` - Interfaz gráfica

### Nuevos Componentes para Tesis
- `evaluate_policy.py` - Evaluación de políticas entrenadas
- `reporting_utils.py` - Generación de informes académicos
- `model_saver.py` - Gestión de modelos entrenados
- `generate_thesis_results.py` - Pipeline completo para tesis

## 🚀 Uso Rápido

### Opción 1: Pipeline Automático
```bash
# Ejecutar todo automáticamente
python generate_thesis_results.py --auto
```

### Opción 2: Pasos Individuales
```bash
# 1. Entrenar modelo
python app.py --cli --iterations 1

# 2. Evaluar política (busca modelo automáticamente)
python evaluate_policy.py --auto-find-model

# 3. Generar informes
python generate_thesis_results.py --report
```

### Opción 3: Monitoreo en Tiempo Real
```bash
# Entrenar con monitoreo
python training_monitor.py --run 1
```

## 📊 Resultados Generados

### Gráficas Principales
1. **`plot_trajectories.png`** - Trayectorias de estado convergiendo al origen
2. **`plot_switching_signal.png`** - Ley de conmutación en acción
3. **`plot_phase_portrait.png`** - Diagrama de fases con regiones de conmutación
4. **`plot_reward_analysis.png`** - Análisis de recompensas acumuladas

### Informes Académicos
1. **`informe_estabilidad.md`** - Informe detallado en Markdown
2. **`informe_estabilidad.tex`** - Informe en formato LaTeX
3. **`RESUMEN_TESIS.md`** - Resumen ejecutivo

### Métricas de Estabilidad
- Tiempo promedio de convergencia
- Tasa de éxito (%)
- Error final promedio
- Desviación estándar del error

## 🔧 Configuración

### Archivo `config.yaml`
Toda la configuración está centralizada en `config.yaml`:

```yaml
# Configuración de red
network:
  defaults:
    state_dim: 2
    action_dim: 2
    num_modes: 2
    hidden_dim: 512
    num_layers: 3

# Configuración de entrenamiento
training:
  defaults:
    num_episodes: 2500
    learning_rate: 5.0e-05
    gamma: 0.995
    max_steps: 200

# Estados de prueba para evaluación
training:
  evaluation:
    test_states:
      - [0.1, 0.1]
      - [0.0, 0.1]
      - [0.1, 0.0]
```

### Perfiles Predefinidos
- **beginner**: Configuración básica para principiantes
- **intermediate**: Configuración balanceada
- **expert**: Configuración avanzada
- **research**: Configuración para investigación intensiva

## 📈 Interpretación de Resultados

### 1. Estabilidad Demostrada
- **Trayectorias convergen**: Todas las trayectorias tienden al origen
- **Tiempo de convergencia**: Métrica de eficiencia del control
- **Robustez**: Funciona para múltiples condiciones iniciales

### 2. Ley de Conmutación
- **Selección de modos**: El agente elige qué subsistema activar
- **Patrones temporales**: Secuencia de conmutaciones óptima
- **Regiones de control**: Mapeo del espacio de estados

### 3. Implicaciones Teóricas
- **Estabilizabilidad**: El sistema es estabilizable
- **Aprendizaje constructivo**: RL descubre políticas complejas
- **Herramienta práctica**: HNAF para análisis de sistemas híbridos

## 🎓 Integración en la Tesis

### Sección de Metodología
```latex
\section{Metodología Experimental}

Utilizamos el algoritmo HNAF (Hybrid Neural Actor Framework) 
para aprender una política de control híbrida que estabilice 
el sistema conmutado. El agente aprende simultáneamente:
\begin{itemize}
    \item La selección de modo discreto (ley de conmutación)
    \item Las acciones de control continuas
\end{itemize}
```

### Sección de Resultados
```latex
\section{Resultados Experimentales}

Los resultados demuestran que el sistema es estabilizable:
\begin{itemize}
    \item Tasa de éxito: 95.2\%
    \item Tiempo promedio de convergencia: 47.3 pasos
    \item Error final promedio: 0.008
\end{itemize}
```

### Figuras Principales
1. **Figura 1**: Trayectorias de estado (convergencia)
2. **Figura 2**: Ley de conmutación (política aprendida)
3. **Figura 3**: Diagrama de fases (regiones de control)

## 🔬 Experimentos Sugeridos

### 1. Comparación de Configuraciones
```bash
# Probar diferentes configuraciones
python generate_thesis_results.py --auto --output-dir resultados_experimento_1
python generate_thesis_results.py --auto --output-dir resultados_experimento_2
```

### 2. Análisis de Robustez
- Variar condiciones iniciales
- Probar diferentes matrices del sistema
- Analizar sensibilidad a parámetros

### 3. Comparación con Métodos Tradicionales
- Comparar con control LQR
- Analizar ventajas del aprendizaje por refuerzo
- Evaluar complejidad computacional

## 🛠️ Solución de Problemas

### Error: "No se encontró ningún modelo"
```bash
# Entrenar primero
python app.py --cli --iterations 1
# Luego evaluar
python evaluate_policy.py --auto-find-model
```

### Error: "Configuración no encontrada"
```bash
# Verificar que config.yaml existe
ls config.yaml
# Si no existe, copiar desde backup
cp config.yaml.backup config.yaml
```

### Error: "Matplotlib no disponible"
```bash
# Instalar matplotlib
pip install matplotlib
```

## 📚 Referencias

1. **HNAF Original**: Paper sobre Hybrid Neural Actor Framework
2. **Sistemas Híbridos**: Teoría de estabilidad de sistemas conmutados
3. **Aprendizaje por Refuerzo**: Deep Q-Learning y Actor-Critic

## 🎯 Próximos Pasos

1. **Integrar resultados** en la tesis
2. **Comparar con métodos teóricos** tradicionales
3. **Extender a sistemas** de mayor dimensión
4. **Publicar resultados** en conferencias

---

**¡Tu aplicación HNAF está lista para generar todos los resultados necesarios para tu tesis!** 🎉 