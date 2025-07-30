# Clase OptimizationFunctions

Una clase flexible para almacenar, gestionar y ejecutar funciones de optimización matemática.

## 📋 Características

- **Almacenamiento de funciones**: Guarda funciones con nombres y descripciones
- **Ejecución dinámica**: Ejecuta funciones por nombre con argumentos flexibles
- **Gestión de resultados**: Almacena y recupera resultados de ejecuciones
- **Visualización**: Incluye herramientas de visualización para transformaciones
- **Optimización**: Métodos de optimización integrados
- **Extensibilidad**: Fácil agregar nuevas funciones

## 🚀 Instalación

```bash
# Asegúrate de tener las dependencias instaladas
pip install numpy matplotlib
```

## 📖 Uso Básico

### 1. Crear una instancia

```python
from optimization_functions import OptimizationFunctions

# Crear instancia
opt_funcs = OptimizationFunctions()
```

### 2. Ver funciones disponibles

```python
# Listar todas las funciones
opt_funcs.list_functions()
```

### 3. Ejecutar funciones existentes

```python
# Transformaciones lineales
x1 = opt_funcs.execute_function("transform_x1")
x2 = opt_funcs.execute_function("transform_x2")

# Función recompensa
reward = opt_funcs.execute_function("reward_function", 3, 4)

# Calcular todas las transformaciones
transforms = opt_funcs.execute_function("calculate_transformations")
```

### 4. Agregar nuevas funciones

```python
# Definir nueva función
def mi_funcion(x, y, parametro=1):
    return x**2 + y**2 + parametro

# Agregar a la clase
opt_funcs.add_function("mi_funcion", mi_funcion, "Descripción de mi función")

# Ejecutar
resultado = opt_funcs.execute_function("mi_funcion", 2, 3, parametro=5)
```

## 🔧 Funciones Incluidas

### Transformaciones Lineales
- **`transform_x1`**: Aplica transformación A1 = [[1, 50], [-1, 1]]
- **`transform_x2`**: Aplica transformación A2 = [[1, -1], [50, 1]]
- **`calculate_transformations`**: Calcula ambas transformaciones

### Función Recompensa
- **`reward_function`**: Minimiza diferencia de distancias al origen
  - Fórmula: `| ||(x, y)|| - ||(x0, y0)|| |`

## 📊 Visualización

```python
# Visualizar transformaciones
opt_funcs.visualize_transformations()

# Con coordenadas personalizadas
opt_funcs.visualize_transformations(x0=2, y0=3)
```

## 🎯 Optimización

```python
# Optimizar función recompensa
resultado = opt_funcs.optimize_reward(
    method='grid',
    bounds=((-10, 10), (-10, 10)),
    n_points=100
)

print(f"Mínimo: {resultado['minimum_reward']}")
print(f"En punto: ({resultado['optimal_x']}, {resultado['optimal_y']})")
```

## 📈 Gestión de Resultados

```python
# Obtener todos los resultados
todos_resultados = opt_funcs.get_results()

# Obtener resultado específico
resultado_x1 = opt_funcs.get_results("transform_x1")
```

## 🔍 Ejemplos de Funciones Personalizadas

### Distancia Euclidiana
```python
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

opt_funcs.add_function("euclidean_distance", euclidean_distance)
```

### Producto Escalar
```python
def dot_product(x1, y1, x2, y2):
    return x1 * x2 + y1 * y2

opt_funcs.add_function("dot_product", dot_product)
```

### Función Cuadrática
```python
def quadratic_function(x, a=1, b=0, c=0):
    return a * x**2 + b * x + c

opt_funcs.add_function("quadratic_function", quadratic_function)
```

## 📁 Estructura de Archivos

```
├── optimization_functions.py    # Clase principal
├── example_usage.py            # Ejemplo de uso
└── README_OptimizationFunctions.md  # Esta documentación
```

## 🧪 Ejecutar Ejemplos

```bash
# Ejecutar ejemplo básico
python optimization_functions.py

# Ejecutar ejemplo completo
python example_usage.py
```

## 🔧 Configuración

### Coordenadas Iniciales
Por defecto, las coordenadas iniciales son `(x0, y0) = (1, 1)`. Puedes cambiarlas modificando:

```python
opt_funcs.x0 = 2
opt_funcs.y0 = 3
```

### Matrices de Transformación
Las matrices A1 y A2 están definidas como:

```python
A1 = [[1, 50],
      [-1, 1]]

A2 = [[1, -1],
      [50, 1]]
```

## 🎨 Personalización

### Agregar Métodos de Optimización
```python
def optimize_with_gradient_descent(self, func_name, initial_point, learning_rate=0.01, iterations=1000):
    # Implementar descenso de gradiente
    pass

opt_funcs.optimize_with_gradient_descent = optimize_with_gradient_descent
```

### Agregar Visualizaciones
```python
def plot_function_3d(self, func_name, x_range, y_range):
    # Implementar visualización 3D
    pass

opt_funcs.plot_function_3d = plot_function_3d
```

## 📝 Notas

- Todas las funciones se almacenan en `self.functions` como diccionario
- Los resultados se almacenan en `self.results`
- Las funciones pueden usar argumentos posicionales y con nombre
- La clase es extensible y modular

## 🤝 Contribuir

Para agregar nuevas funcionalidades:

1. Define tu función
2. Usa `add_function()` para agregarla
3. Documenta su propósito
4. Prueba con diferentes argumentos

## 📄 Licencia

Este código es de uso libre para fines educativos y de investigación. 