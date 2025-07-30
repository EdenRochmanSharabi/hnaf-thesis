# Hybrid Normalized Advantage Function (HNAF) - Tesis de Grado

## Descripción

Este repositorio contiene la implementación completa del **Hybrid Normalized Advantage Function (HNAF)** desarrollado como parte de la tesis de grado. El HNAF es un algoritmo de aprendizaje por refuerzo que combina control discreto y continuo para sistemas de control híbridos.

## Objetivo

Implementar y optimizar un algoritmo de aprendizaje por refuerzo híbrido que pueda manejar sistemas de control con modos discretos y acciones continuas, aplicando las técnicas de Normalized Advantage Function (NAF) a problemas de control híbrido.

## Metodología

### Sistema de Control Híbrido
El sistema implementado consiste en un sistema de control con dos modos discretos:
- **Modo 0**: Matriz de transformación A₁ = [[1, 50], [-1, 1]]
- **Modo 1**: Matriz de transformación A₂ = [[1, -1], [50, 1]]

### Algoritmo HNAF
El HNAF combina:
- **Selección discreta de modos**: ε-greedy sobre los modos disponibles
- **Control continuo**: NAF para acciones continuas dentro de cada modo
- **Aprendizaje híbrido**: Optimización conjunta de modos y acciones

## Características Principales

### Mejoras Implementadas
1. **Recompensas reescaladas**: `r = -abs(||x'|| - ||x₀||) / 15` para estabilidad numérica
2. **Factor de descuento optimizado**: γ = 0.9 para mejor convergencia
3. **Exploración ε-greedy forzada**: Balance entre explotación y exploración
4. **Buffer de replay ampliado**: 5000 transiciones para mejor experiencia
5. **Batch size optimizado**: 32 muestras para gradientes estables
6. **Entrenamiento extendido**: 1000 épocas para convergencia completa

### Componentes Técnicos
- **Arquitectura de red**: Redes neuronales separadas para cada modo
- **Función de valor**: V(x,v) para cada modo discreto
- **Función de ventaja**: A(x,v,u) para acciones continuas
- **Actualización suave**: Soft update de redes objetivo
- **Clipping de gradientes**: Para estabilidad numérica

## Estructura del Proyecto

```
HNAF-Jose/
├── src/                          # Módulo principal
│   ├── __init__.py              # Inicialización del módulo
│   ├── hnaf_stable.py           # Implementación HNAF mejorada
│   ├── naf_corrected.py         # NAF corregido con exponencial de matriz
│   └── optimization_functions.py # Funciones de optimización base
├── demo_completo.py              # Demostración completa del sistema
├── test_hnaf_improvements.py     # Tests de mejoras implementadas
├── hnaf_stable.py                # Implementación principal HNAF
├── naf_corrected.py              # NAF corregido principal
├── optimization_functions.py     # Funciones de optimización
├── requirements.txt              # Dependencias del proyecto
├── README.md                     # Este archivo
└── RESUMEN_MEJORAS_HNAF.md      # Documentación detallada de mejoras
```

## Instalación

### Prerrequisitos
- Python 3.8+
- PyTorch
- NumPy
- SciPy
- Matplotlib

### Instalación de Dependencias
```bash
pip install -r requirements.txt
```

## Uso

### Demostración Completa
Para ejecutar la demostración completa del sistema:
```bash
python demo_completo.py
```

### Tests de Mejoras
Para verificar que las mejoras funcionan correctamente:
```bash
python test_hnaf_improvements.py
```

### Entrenamiento Personalizado
```python
from src.hnaf_stable import train_stable_hnaf

# Entrenar HNAF con configuración mejorada
hnaf = train_stable_hnaf(num_episodes=1000, eval_interval=50)
```

## Resultados Experimentales

### Validación Científica vs Solución Exacta

**Verificación de Corrección Matemática:**
- **NAF vs ODE**: Diferencias de 0.00e+00 (resultados idénticos)
- **Transformaciones**: Uso correcto de exponencial de matriz `expm(A * t)`
- **Recompensas**: Cálculo exacto coincidente con solución ODE

**Casos de Prueba Validados:**
- Estado [1, 1]: NAF1=14.2150, NAF2=14.2150, ODE=14.2150 ✅
- Estado [0, 1]: NAF1=12.7594, NAF2=0.9366, ODE=12.7594/0.9366 ✅
- Estado [1, 0]: NAF1=0.9366, NAF2=12.7594, ODE=0.9366/12.7594 ✅
- Estado [0.5, 0.5]: NAF1=7.1075, NAF2=7.1075, ODE=7.1075 ✅

### Rendimiento del HNAF Entrenado

**Métricas de Entrenamiento:**
- **Épocas completadas**: 1000
- **Recompensa final promedio**: -19.4811
- **Pérdida final**: 0.196338
- **Convergencia**: Estable y consistente

**Selección de Modos Óptimos:**
- **Rendimiento**: 3/5 casos (60.0%)
- **Casos exitosos**: Estados [0.1, 0.1], [0, 0.1], [0.05, 0.05]
- **Casos con mejora**: Estados [0.1, 0], [-0.05, 0.08]

### Visualización de Resultados

![Análisis de Recompensas y Modos Óptimos](figure%201.png)

**Análisis de la Visualización:**
- **Panel 1**: Recompensa NAF1 (Modo 0) - Patrón diagonal de bajo a alto
- **Panel 2**: Recompensa NAF2 (Modo 1) - Patrón diagonal perpendicular
- **Panel 3**: Modo óptimo - Regiones bien definidas con transiciones claras

### Configuración Final Optimizada
- **Factor de descuento**: γ = 0.9
- **Buffer de replay**: 5000 transiciones
- **Batch size**: 32 muestras
- **Exploración**: ε-greedy con balance 60-40%
- **Recompensas**: Reescaladas a r ∈ [-1, 0]

## Validación Científica

### Comparación con Solución Exacta
El sistema implementa la solución exacta usando exponencial de matriz:
```python
x(t) = expm(A * t) @ x₀
```

### Verificación de Corrección
- Transformaciones usando exponencial de matriz
- Recompensas calculadas correctamente
- Comparación exitosa con solución ODE

## Referencias

1. **Normalized Advantage Functions**: Gu et al. (2016)
2. **Hybrid Control Systems**: Branicky et al. (1998)
3. **Deep Reinforcement Learning**: Sutton & Barto (2018)

## Autor

**Eden Rochman**  
Estudiante de Ingeniería  
Tesis de Grado

## Licencia

Este proyecto es de código abierto (Open Source).

## Contribuciones

Este es un proyecto de tesis académica. Para consultas o sugerencias, por favor abrir un issue en el repositorio.

## Contacto

- **Email**: eden@example.com
- **GitHub**: [@edenrochman](https://github.com/edenrochman)

---

**Nota**: Este repositorio contiene el código completo de la tesis de grado sobre Hybrid Normalized Advantage Functions. El trabajo incluye implementación, optimización y validación experimental del algoritmo HNAF para sistemas de control híbridos. 