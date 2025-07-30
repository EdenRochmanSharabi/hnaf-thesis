# Interfaz Gráfica HNAF

## Descripción

Esta interfaz gráfica proporciona un control completo sobre los hiperparámetros del algoritmo HNAF (Hybrid Normalized Advantage Function) para tu tesis de investigación. La interfaz es profesional y está diseñada específicamente para entornos académicos.

## Características

- **Control de Hiperparámetros**: Ajuste completo de todos los parámetros de red y entrenamiento
- **Funciones Personalizadas**: Editor de código para definir tus propias transformaciones y recompensas
- **Entrenamiento en Tiempo Real**: Visualización del progreso con barra de progreso
- **Salida de Terminal Integrada**: Todos los resultados se muestran en la interfaz
- **Gráficos Automáticos**: Visualización de resultados de entrenamiento
- **Evaluación y Verificación**: Herramientas para validar el modelo entrenado
- **Interfaz Profesional**: Diseño limpio y académico sin emojis

## Instalación

1. Asegúrate de tener Python 3.7+ instalado
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

### Método 1: Script de lanzamiento
```bash
python run_gui.py
```

### Método 2: Ejecución directa
```bash
python hnaf_gui.py
```

## Parámetros Configurables

### Parámetros de Red Neuronal
- **Dimensión del Estado**: Tamaño del vector de estado (1-10)
- **Dimensión de Acción**: Tamaño del vector de acción (1-10)
- **Número de Modos**: Cantidad de modos discretos (1-5)
- **Capas Ocultas**: Tamaño de las capas ocultas (16-256)

### Parámetros de Entrenamiento
- **Learning Rate**: Tasa de aprendizaje (0.0001-0.1)
- **Tau (Soft Update)**: Parámetro de actualización suave (0.0001-0.01)
- **Gamma (Discount)**: Factor de descuento (0.8-0.999)
- **Episodios**: Número de episodios de entrenamiento (100-5000)
- **Batch Size**: Tamaño del batch (8-128)
- **Epsilon**: Parámetro de exploración (0.0-0.5)
- **Max Steps**: Pasos máximos por episodio (10-200)

## Funcionalidades

### 1. Funciones Personalizadas
- **Editor de Código**: Escribe tus propias funciones de transformación y recompensa
- **Plantilla Incluida**: Código de ejemplo listo para usar
- **Prueba de Funciones**: Valida tu código antes del entrenamiento
- **Guardar/Cargar**: Persistencia de tus funciones personalizadas
- **Checkbox de Activación**: Activa/desactiva el uso de funciones personalizadas

### 2. Iniciar Entrenamiento
- Configura los hiperparámetros deseados
- Activa funciones personalizadas si las necesitas
- Haz clic en "Iniciar Entrenamiento"
- El progreso se muestra en tiempo real
- Los resultados aparecen en la terminal integrada

### 3. Evaluar Modelo
- Disponible después del entrenamiento
- Evalúa el modelo en 20 episodios
- Muestra recompensa promedio y distribución de modos

### 4. Verificar HNAF
- Compara el HNAF con NAF individual
- Valida la selección óptima de modos
- Muestra diferencias entre predicción y realidad

### 5. Gráficos de Resultados
- Recompensas por episodio
- Recompensas de evaluación
- Promedio móvil de recompensas

## Funciones Personalizadas

### Estructura Requerida
Tu código debe incluir estas funciones obligatorias:

```python
# Lista de funciones de transformación (una por modo)
transformation_functions = [transform_mode_0, transform_mode_1]

# Función de recompensa
def reward_function(x, y, x0, y0):
    """Función de recompensa personalizada"""
    # Tu lógica aquí
    return reward_value
```

### Ejemplo de Plantilla
La interfaz incluye una plantilla completa con:
- Funciones de transformación para cada modo
- Función de recompensa personalizable
- Ejemplos de uso y pruebas

### Validación Automática
- **Prueba de Funciones**: Valida sintaxis y estructura
- **Verificación de Dependencias**: Asegura que todas las funciones requeridas estén presentes
- **Test de Ejecución**: Prueba las funciones con valores de ejemplo

## Estructura de Archivos

```
HNAF Jose/
├── hnaf_gui.py          # Interfaz gráfica principal
├── run_gui.py           # Script de lanzamiento
├── hnaf_implementation.py # Implementación HNAF
├── optimization_functions.py # Funciones de optimización
├── custom_functions.py   # Funciones personalizadas (se crea al guardar)
├── requirements.txt     # Dependencias
└── GUI_README.md        # Esta documentación
```

## Solución de Problemas

### Error de Importación
Si aparece un error de importación:
```bash
pip install -r requirements.txt
```

### Error de Tkinter
En algunos sistemas Linux, tkinter puede no estar instalado:
```bash
sudo apt-get install python3-tk  # Ubuntu/Debian
```

### Error de Matplotlib
Si hay problemas con matplotlib:
```bash
pip install --upgrade matplotlib
```

## Notas Técnicas

- La interfaz utiliza threading para evitar bloqueos durante el entrenamiento
- Todos los prints se redirigen automáticamente a la terminal integrada
- Los gráficos se actualizan automáticamente al completar el entrenamiento
- La interfaz es completamente profesional y apropiada para tesis

## Contribución

Para reportar problemas o sugerir mejoras, por favor:
1. Revisa que el problema no esté ya documentado
2. Proporciona información detallada sobre el error
3. Incluye tu sistema operativo y versión de Python 