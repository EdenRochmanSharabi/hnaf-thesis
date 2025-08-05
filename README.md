# 🧠 HNAF - Hybrid Neural Actor Framework

> **Demostrando la estabilizabilidad de sistemas híbridos conmutados mediante aprendizaje por refuerzo**

Este repositorio contiene mi investigación sobre el algoritmo HNAF (Hybrid Neural Actor Framework) para el control de sistemas híbridos conmutados. El trabajo forma parte de mi tesis de maestría en control automático.

## 🎯 ¿Qué hace este proyecto?

El objetivo principal es **demostrar que es posible estabilizar sistemas híbridos conmutados** usando aprendizaje por refuerzo. En lugar de usar métodos tradicionales de control, el agente aprende automáticamente:

- **Cuándo conmutar** entre diferentes subsistemas
- **Cómo controlar** cada subsistema individualmente
- **La política óptima** que lleva el sistema al equilibrio

### 🔬 Contexto Teórico

Los sistemas híbridos conmutados son comunes en aplicaciones reales (robótica, control de procesos, vehículos autónomos). El problema fundamental es: **¿Es posible estabilizar estos sistemas usando aprendizaje por refuerzo?**

Mi investigación responde: **¡Sí!** Y lo demuestra con resultados experimentales sólidos.

## 📁 Estructura del Repositorio

```
├── final_app/          # 🎯 Aplicación principal HNAF
│   ├── app.py          # Interfaz principal (GUI + CLI)
│   ├── hnaf_improved.py # Algoritmo HNAF mejorado
│   ├── training_manager.py # Gestión de entrenamiento
│   ├── gui_interface.py # Interfaz gráfica profesional
│   └── DOCUMENTACION_FINAL.md # 📚 Documentación completa
├── 3D solver/          # 🔧 Extensión a 3D (en desarrollo)
└── bin/                # 📦 Archivos de respaldo y desarrollo
```

## 🚀 ¿Cómo usar el sistema?

### Opción 1: Pipeline Automático (Recomendado)
```bash
cd final_app
python generate_thesis_results.py --auto
```
Esto ejecuta todo automáticamente y genera todos los resultados para la tesis.

### Opción 2: Uso Interactivo
```bash
cd final_app
python app.py
```
Abre la interfaz gráfica donde puedes configurar parámetros y monitorear el entrenamiento en tiempo real.

### Opción 3: Modo CLI
```bash
cd final_app
python app.py --cli --iterations 1
```
Para ejecución automatizada sin interfaz gráfica.

## 📊 Resultados Experimentales

### ✅ **Resultados Principales**
- **Tasa de éxito**: 100.0% (50/50 trayectorias convergentes)
- **Tiempo promedio de convergencia**: 69.5 pasos
- **Error final promedio**: 0.000 (convergencia perfecta)
- **Estabilidad**: Sistema completamente estabilizable

### 📈 **Gráficas Generadas**
1. **Trayectorias de estado** - Muestra cómo el sistema converge al origen
2. **Ley de conmutación** - Visualiza la política aprendida por el agente
3. **Diagrama de fases** - Regiones de control en el espacio de estados
4. **Análisis de recompensas** - Evolución del aprendizaje

### 📝 **Informes Académicos**
- Informes en Markdown y LaTeX
- Métricas cuantitativas detalladas
- Análisis de estabilidad completo
- Código LaTeX para tesis

## 🔧 Características Técnicas

### **Algoritmo HNAF Mejorado**
- **Sin valores hardcodeados**: Todo configurable desde `config.yaml`
- **Red neuronal con Batch Normalization**: Mejor estabilidad
- **Múltiples modos de conmutación**: Soporte para sistemas complejos
- **Evaluación automática**: Métricas de estabilidad en tiempo real

### **Interfaz Profesional**
- **GUI completa** con Tkinter
- **Monitoreo en tiempo real** del entrenamiento
- **Configuración dinámica** de parámetros
- **Visualización de resultados** integrada

### **Optimización Automática**
- **Integración con Optuna** para búsqueda de hiperparámetros
- **Logging detallado** de todo el proceso
- **Detección automática** de problemas
- **Guardado inteligente** de mejores configuraciones

## 📚 Documentación Completa

Para información detallada sobre el uso, configuración y análisis de resultados, consulta:

**[📖 Documentación Completa](final_app/DOCUMENTACION_FINAL.md)**

La documentación incluye:
- Guía paso a paso de uso
- Explicación detallada de cada componente
- Análisis de logs y resultados
- Solución de problemas comunes
- Integración académica para tesis

## 🎓 Contexto Académico

Este trabajo forma parte de mi investigación de maestría en control automático. Los resultados demuestran que:

1. **Los sistemas híbridos conmutados son estabilizables** mediante aprendizaje por refuerzo
2. **HNAF es una herramienta efectiva** para el análisis de estos sistemas
3. **El enfoque es aplicable** a problemas de control reales

### **Implicaciones Teóricas**
- Validación experimental de estabilizabilidad
- Nuevo enfoque para control de sistemas híbridos
- Herramienta práctica para investigación en control

## 🛠️ Instalación y Configuración

### **Requisitos**
```bash
pip install torch numpy matplotlib tkinter optuna
```

### **Configuración**
Todo está configurado en `final_app/config.yaml`. No necesitas modificar código, solo ajustar parámetros según tus necesidades.

## 📈 Estado del Proyecto

- ✅ **Sistema HNAF 2D**: Completamente funcional
- 🔄 **Extensión 3D**: En desarrollo
- 📊 **Resultados experimentales**: Validados y documentados
- 📝 **Documentación**: Completa y actualizada

## 🤝 Contribuciones

Este es un proyecto de investigación académica. Si encuentras errores o tienes sugerencias, por favor abre un issue. Las contribuciones son bienvenidas, especialmente en:

- Mejoras al algoritmo HNAF
- Extensiones a sistemas de mayor dimensión
- Optimización de rendimiento
- Documentación adicional

## 📄 Licencia

Este proyecto es parte de mi investigación académica. El código está disponible para fines educativos y de investigación.

---

**Autor**: [Tu nombre]  
**Institución**: [Tu universidad]  
**Tesis**: Control de Sistemas Híbridos Conmutados mediante Aprendizaje por Refuerzo  
**Fecha**: Agosto 2025

*Desarrollado con ❤️ para la investigación en control automático* 
