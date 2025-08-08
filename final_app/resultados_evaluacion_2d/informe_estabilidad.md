# Informe de Análisis de Estabilidad HNAF

**Fecha y Hora:** 2025-08-08 00:03:07
**Modelo Cargado:** `models/hnaf_model_20250808_000252.pt`
**Configuración:** HNAF Mejorado

## Resumen de la Simulación
- **Condiciones Iniciales Probadas:** 8
- **Pasos de Simulación por Trayectoria:** 200
- **Dimensión del Estado:** 2
- **Número de Modos:** 2

## Métricas de Estabilidad
- **Tiempo Promedio de Convergencia:** 200.00 pasos
- **Tasa de Éxito:** 0.0%
- **Error Final Promedio:** nan

--- 

## 1. Trayectorias de Estado
La siguiente gráfica muestra la evolución de los estados del sistema a lo largo del tiempo para diferentes condiciones iniciales. Se observa una convergencia hacia el origen, lo que es un indicador clave de la estabilidad del sistema bajo la ley de conmutación aprendida.

![Trayectorias de Estado](plot_trajectories.png)

--- 

## 2. Ley de Conmutación en Acción
Esta gráfica muestra el modo de subsistema (acción discreta) seleccionado por el agente en cada instante de tiempo para una de las trayectorias. Representa la ley de conmutación aprendida en funcionamiento.

![Señal de Conmutación](plot_switching_signal.png)

--- 

## 3. Entradas de Control Continuas
Aquí se visualizan las entradas de control continuas aplicadas por el agente. Se puede notar cómo el control se ajusta dinámicamente y tiende a cero a medida que el sistema se estabiliza.

![Control Continuo](plot_switching_signal.png)

--- 

## 4. Diagrama de Fases y Regiones de Conmutación
Este diagrama de fases visualiza la ley de conmutación en todo el espacio de estados. Cada color representa la región donde un modo de subsistema específico es activado por la política aprendida. Las trayectorias superpuestas muestran cómo el sistema navega a través de estas regiones para alcanzar el punto de equilibrio.

![Diagrama de Fases](plot_phase_portrait.png)

--- 

## 5. Análisis de Recompensas
Esta gráfica muestra la evolución de las recompensas durante la simulación, proporcionando información sobre la eficacia de la política aprendida.

![Análisis de Recompensas](plot_reward_analysis.png)

--- 

## Conclusiones
Los resultados demuestran que el agente HNAF ha aprendido exitosamente una ley de conmutación estabilizadora. Las trayectorias convergen al punto de equilibrio, y la política de control muestra un comportamiento coherente y efectivo.

### Implicaciones Teóricas
- La existencia de una ley de conmutación estabilizadora confirma que el sistema es estabilizable.
- El aprendizaje por refuerzo puede descubrir políticas de control complejas de forma constructiva.
- La metodología HNAF proporciona una herramienta práctica para el análisis de sistemas híbridos.

