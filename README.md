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

## Resultados

### Rendimiento vs Solución Exacta
- **Recompensas reescaladas**: Implementadas correctamente
- **Gamma = 0.9**: Efecto significativo en convergencia
- **Exploración mejorada**: ε-greedy funcionando
- **Buffer y batch optimizados**: Configuración estable

### Métricas de Entrenamiento
- **Épocas de entrenamiento**: 1000
- **Intervalo de evaluación**: 50 épocas
- **Batch size**: 32
- **Buffer capacity**: 5000
- **Factor de descuento**: 0.9

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