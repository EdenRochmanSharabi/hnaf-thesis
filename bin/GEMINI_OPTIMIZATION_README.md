# 🤖 Optimización Automática con Gemini

## Descripción

El sistema de optimización automática utiliza la API de Google Gemini para buscar los mejores hiperparámetros para el algoritmo HNAF de forma automática.

## Características

### 🔧 Optimización Automática
- **Búsqueda inteligente**: Gemini analiza resultados previos y sugiere mejoras
- **Evaluación automática**: Entrena modelos con parámetros sugeridos
- **Aprendizaje continuo**: Mejora iterativamente basándose en resultados
- **Timeout configurable**: Se detiene automáticamente después de un tiempo límite

### 📊 Métricas de Evaluación
- **Precisión en grid**: Porcentaje de selección correcta de modos
- **Recompensa promedio**: Estabilidad del sistema
- **Score combinado**: Métrica unificada para optimización

### 💾 Persistencia de Datos
- **Guardado automático**: Mejores parámetros se guardan en JSON
- **Carga automática**: Al iniciar, carga mejores parámetros encontrados
- **Historial completo**: Guarda todas las iteraciones para análisis

## Configuración

### 1. API Key de Gemini
Crear archivo `config.py` en el directorio raíz:
```python
GEMINI_API_KEY = "tu_api_key_aqui"
```

### 2. Configuración de Optimización
```python
AUTO_OPTIMIZATION_CONFIG = {
    "max_iterations": 20,        # Máximo número de iteraciones
    "evaluation_episodes": 100,   # Episodios por evaluación
    "timeout_minutes": 30,        # Timeout en minutos
    "best_params_file": "best_hyperparameters.json"
}
```

## Uso en la Interfaz

### 🚀 Iniciar Optimización
1. Marcar checkbox "Usar optimización automática con Gemini"
2. Hacer clic en "🚀 Iniciar Optimización"
3. El sistema comenzará a buscar automáticamente

### 🛑 Detener Optimización
- Hacer clic en "🛑 Detener Optimización" en cualquier momento
- El sistema guardará los mejores parámetros encontrados

### 📥 Cargar Mejores Parámetros
- Hacer clic en "📥 Cargar Mejores Parámetros"
- Los parámetros optimizados se aplicarán automáticamente a la UI

## Parámetros Optimizados

### Red Neuronal
- `hidden_dim`: [16, 32, 64, 128, 256]
- `num_layers`: [2, 3, 4, 5]

### Entrenamiento
- `lr`: [1e-5, 1e-4, 1e-3, 1e-2]
- `batch_size`: [16, 32, 64, 128]
- `initial_epsilon`: [0.1, 0.3, 0.5, 0.7]
- `final_epsilon`: [0.01, 0.05, 0.1]
- `max_steps`: [10, 20, 30, 50]

### Experience Replay
- `buffer_capacity`: [1000, 5000, 10000]
- `alpha`: [0.4, 0.6, 0.8]
- `beta`: [0.2, 0.4, 0.6]

## Archivos Generados

### `best_hyperparameters.json`
```json
{
  "best_params": {
    "hidden_dim": 128,
    "num_layers": 4,
    "lr": 0.0001,
    "batch_size": 64,
    "initial_epsilon": 0.5,
    "final_epsilon": 0.05,
    "max_steps": 30,
    "buffer_capacity": 10000,
    "alpha": 0.6,
    "beta": 0.4
  },
  "best_score": 0.8542,
  "history": [...],
  "last_updated": "2024-01-15T10:30:00"
}
```

## Proceso de Optimización

### 1. Consulta a Gemini
- Envía contexto del problema y parámetros disponibles
- Incluye historial de resultados previos
- Solicita parámetros optimizados en formato JSON

### 2. Evaluación Automática
- Entrena modelo con parámetros sugeridos
- Evalúa precisión en grid y recompensa promedio
- Calcula score combinado

### 3. Actualización de Mejores Parámetros
- Compara score actual con mejor score histórico
- Guarda nuevos mejores parámetros si corresponde
- Actualiza historial de optimización

### 4. Iteración
- Repite proceso hasta timeout o máximo iteraciones
- Cada iteración mejora basándose en resultados previos

## Seguridad

### 🔒 API Key
- **NO se sube al repositorio**: `config.py` está en `.gitignore`
- **Uso local**: La API key solo se usa en tu máquina
- **Configuración segura**: Se puede cambiar fácilmente

### 📁 Archivos Sensibles
- `config.py`: Contiene API key (no se sube)
- `best_hyperparameters.json`: Resultados de optimización
- `.gitignore`: Protege archivos sensibles

## Instalación

### 1. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 2. Configurar API Key
```bash
# Crear archivo config.py
echo 'GEMINI_API_KEY = "tu_api_key_aqui"' > config.py
```

### 3. Ejecutar Aplicación
```bash
python final_app/main.py
```

## Troubleshooting

### Error de API Key
```
❌ Error iniciando optimización: Invalid API key
```
**Solución**: Verificar que `config.py` existe y contiene una API key válida

### Error de Conexión
```
❌ Error consultando Gemini: Network error
```
**Solución**: Verificar conexión a internet y API key válida

### Timeout
```
⏰ Timeout alcanzado
```
**Solución**: Aumentar `timeout_minutes` en configuración

## Ventajas

### 🎯 Optimización Inteligente
- Gemini entiende el contexto del problema
- Aprende de resultados previos
- Sugiere mejoras basadas en patrones

### ⚡ Automatización Completa
- No requiere intervención manual
- Ejecuta en segundo plano
- Guarda resultados automáticamente

### 📈 Mejora Continua
- Cada ejecución mejora los parámetros
- Historial completo para análisis
- Persistencia entre sesiones

### 🔄 Fácil de Usar
- Interfaz gráfica intuitiva
- Un solo clic para iniciar
- Resultados inmediatamente aplicables 