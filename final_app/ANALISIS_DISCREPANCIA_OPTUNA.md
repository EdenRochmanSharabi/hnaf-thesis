# 🕵️‍♂️ Análisis de Discrepancia: Optuna vs Ejecución Manual

## ❓ Problema

**Optuna reporta una precisión del 80% con ciertos hiperparámetros, pero al ejecutar el sistema manualmente con esos mismos parámetros, la precisión cae a ~43%.**

---

## 1. Posibles Causas Técnicas

### 1.1. Diferencia en la función de recompensa
- **Optuna** podría estar usando una función de recompensa diferente (por ejemplo, `np.linalg.norm([x, y])`), mientras que la ejecución manual usa otra (por ejemplo, `-np.tanh(np.linalg.norm([x, y]) * 0.1)`).
- **Impacto:** Cambios sutiles en la función de recompensa pueden alterar radicalmente el aprendizaje y la métrica de precisión.
- **Verificar:** Que el string de la función de recompensa en `config.yaml` sea **idéntico** al usado en el trial de Optuna.

### 1.2. Semillas aleatorias y determinismo
- **Optuna** puede fijar la semilla (`numpy_seed`, `torch_seed`) de forma diferente o en un lugar distinto del código respecto a la ejecución manual.
- **Impacto:** En RL, pequeñas diferencias en la semilla pueden causar grandes variaciones en el resultado.
- **Verificar:** Que las semillas sean exactamente iguales y se fijen antes de cualquier inicialización de red o buffer.

### 1.3. Diferencias en el pipeline de evaluación
- **Optuna** podría estar evaluando la precisión de forma diferente (por ejemplo, usando un grid distinto, o evaluando en un punto diferente del entrenamiento).
- **Impacto:** Si la métrica de evaluación no es idéntica, los resultados no son comparables.
- **Verificar:** Que el método de evaluación (`evaluate_policy_grid`, etc.) sea exactamente el mismo en ambos casos.

### 1.4. Parámetros ocultos o no aplicados
- Puede que algunos parámetros del trial de Optuna no se estén aplicando realmente en la ejecución manual (por ejemplo, si el código no lee todos los parámetros desde el config, o si hay un bug en la actualización de la configuración).
- **Impacto:** Un solo parámetro diferente puede cambiar radicalmente el resultado.
- **Verificar:** Que todos los parámetros del trial ganador estén realmente en uso en la ejecución manual.

### 1.5. Optuna sobreajustando a una métrica específica
- Si la métrica de Optuna no es exactamente la misma que la reportada en el CLI, puede haber un sesgo.
- **Verificar:** Que la métrica de score/precisión sea exactamente la misma en ambos contextos.

### 1.6. Estado interno o inicialización diferente
- Puede haber diferencias en el estado inicial del entorno, buffers, o incluso en la inicialización de la red.
- **Verificar:** Que todo el pipeline de inicialización sea idéntico.

---

## 2. Checklist de Verificación

- [ ] ¿La función de recompensa (`reward_function`) es **idéntica** en Optuna y en el config.yaml?
- [ ] ¿Las semillas (`numpy_seed`, `torch_seed`) son iguales y se fijan antes de cualquier inicialización?
- [ ] ¿El método de evaluación es exactamente el mismo (grid, episodios, etc.)?
- [ ] ¿Todos los hiperparámetros del trial ganador están realmente aplicados en la ejecución manual?
- [ ] ¿No hay parámetros por defecto o "ocultos" que difieran?
- [ ] ¿El entorno y la red se inicializan exactamente igual?
- [ ] ¿El código de entrenamiento y evaluación es el mismo en ambos casos?
- [ ] ¿No hay diferencias en la versión de librerías (numpy, torch, etc.)?

---

## 3. Recomendaciones para Depuración

1. **Haz un diff entre el config.yaml y los parámetros del trial ganador de Optuna.**
2. **Fuerza la función de recompensa y todos los hiperparámetros a ser idénticos.**
3. **Imprime todos los hiperparámetros y la función de recompensa al inicio de cada run.**
4. **Fuerza la semilla aleatoria antes de cualquier inicialización.**
5. **Verifica que el método de evaluación sea exactamente el mismo.**
6. **Ejecuta varias veces para descartar variabilidad estocástica.**
7. **Si la discrepancia persiste, prueba a ejecutar el trial de Optuna manualmente (fuera del framework) con los mismos parámetros y observa el resultado.**

---

## 4. Conclusión

**Las discrepancias entre Optuna y la ejecución manual suelen deberse a diferencias sutiles en la función de recompensa, semillas, parámetros no aplicados o diferencias en el pipeline de evaluación.**

**La reproducibilidad en RL es difícil, pero siguiendo este checklist puedes identificar y corregir la causa raíz.**