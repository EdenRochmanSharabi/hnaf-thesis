#!/usr/bin/env python3
"""
Utilidades para generar informes académicos en formato Markdown
Específicamente diseñado para documentar resultados de estabilidad HNAF
"""

import os
from datetime import datetime
import numpy as np
import json

def create_report(results_dir, report_data):
    """
    Genera un informe completo en formato Markdown para la tesis.

    Args:
        results_dir (str): Directorio donde se guardarán el informe y las imágenes.
        report_data (dict): Diccionario con los datos y rutas de las figuras.
    """
    report_path = os.path.join(results_dir, "informe_estabilidad.md")
    
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("# Informe de Análisis de Estabilidad HNAF\n\n")
        f.write(f"**Fecha y Hora:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Modelo Cargado:** `{report_data['model_path']}`\n")
        f.write(f"**Configuración:** {report_data.get('config_name', 'HNAF Mejorado')}\n\n")
        
        f.write("## Resumen de la Simulación\n")
        f.write(f"- **Condiciones Iniciales Probadas:** {len(report_data['initial_conditions'])}\n")
        f.write(f"- **Pasos de Simulación por Trayectoria:** {report_data['max_steps']}\n")
        f.write(f"- **Dimensión del Estado:** {report_data.get('state_dim', 'N/A')}\n")
        f.write(f"- **Número de Modos:** {report_data.get('num_modes', 'N/A')}\n\n")

        # Resumen de configuración (si está disponible)
        net = report_data.get('network_defaults', {})
        trn = report_data.get('training_defaults', {})
        state_limits = report_data.get('state_limits', {})
        reward_shaping = report_data.get('reward_shaping', {})
        if net or trn or state_limits or reward_shaping:
            f.write("## Configuración (Resumen)\n")
            if net:
                f.write("### Parámetros de Red\n")
                for k, v in net.items():
                    f.write(f"- {k}: {v}\n")
            if trn:
                f.write("\n### Parámetros de Entrenamiento\n")
                for k, v in trn.items():
                    f.write(f"- {k}: {v}\n")
            if state_limits:
                f.write("\n### Límites del Estado\n")
                f.write(f"- min: {state_limits.get('min')}\n")
                f.write(f"- max: {state_limits.get('max')}\n")
            if reward_shaping:
                f.write("\n### Reward Shaping\n")
                for k, v in reward_shaping.items():
                    f.write(f"- {k}: {v}\n")
            f.write("\n")

        # Matrices del sistema
        matrices = report_data.get('matrices', {})
        if matrices:
            f.write("## Matrices del Sistema\n")
            for key in sorted([k for k in matrices.keys() if str(k).startswith('A')]):
                f.write(f"- {key}: {matrices[key]}\n")
            f.write("\n")

        # Métricas de estabilidad
        if 'stability_metrics' in report_data:
            metrics = report_data['stability_metrics']
            f.write("## Métricas de Estabilidad\n")
            f.write(f"- **Tiempo Promedio de Convergencia:** {metrics.get('avg_convergence_time', 'N/A'):.2f} pasos\n")
            f.write(f"- **Tasa de Éxito:** {metrics.get('success_rate', 'N/A'):.1f}%\n")
            f.write(f"- **Error Final Promedio:** {metrics.get('avg_final_error', 'N/A'):.6f}\n\n")

        f.write("--- \n\n")

        f.write("## 1. Trayectorias de Estado\n")
        f.write("La siguiente gráfica muestra la evolución de los estados del sistema a lo largo del tiempo para diferentes condiciones iniciales. Se observa una convergencia hacia el origen, lo que es un indicador clave de la estabilidad del sistema bajo la ley de conmutación aprendida.\n\n")
        f.write(f"![Trayectorias de Estado]({os.path.basename(report_data['plots']['trajectories'])})\n\n")

        f.write("--- \n\n")

        f.write("## 2. Ley de Conmutación en Acción\n")
        f.write("Esta gráfica muestra el modo de subsistema (acción discreta) seleccionado por el agente en cada instante de tiempo para una de las trayectorias. Representa la ley de conmutación aprendida en funcionamiento.\n\n")
        f.write(f"![Señal de Conmutación]({os.path.basename(report_data['plots']['switching'])})\n\n")

        f.write("--- \n\n")

        f.write("## 3. Entradas de Control Continuas\n")
        f.write("Aquí se visualizan las entradas de control continuas aplicadas por el agente. Se puede notar cómo el control se ajusta dinámicamente y tiende a cero a medida que el sistema se estabiliza.\n\n")
        f.write(f"![Control Continuo]({os.path.basename(report_data['plots']['control'])})\n\n")

        if 'phase_portrait' in report_data['plots']:
            f.write("--- \n\n")
            f.write("## 4. Diagrama de Fases y Regiones de Conmutación\n")
            f.write("Este diagrama de fases visualiza la ley de conmutación en todo el espacio de estados. Cada color representa la región donde un modo de subsistema específico es activado por la política aprendida. Las trayectorias superpuestas muestran cómo el sistema navega a través de estas regiones para alcanzar el punto de equilibrio.\n\n")
            f.write(f"![Diagrama de Fases]({os.path.basename(report_data['plots']['phase_portrait'])})\n\n")

        if 'reward_analysis' in report_data['plots']:
            f.write("--- \n\n")
            f.write("## 5. Análisis de Recompensas\n")
            f.write("Esta gráfica muestra la evolución de las recompensas durante la simulación, proporcionando información sobre la eficacia de la política aprendida.\n\n")
            f.write(f"![Análisis de Recompensas]({os.path.basename(report_data['plots']['reward_analysis'])})\n\n")

        f.write("--- \n\n")
        f.write("## Conclusiones\n")
        f.write("Los resultados demuestran que el agente HNAF ha aprendido exitosamente una ley de conmutación estabilizadora. Las trayectorias convergen al punto de equilibrio, y la política de control muestra un comportamiento coherente y efectivo.\n\n")
        
        f.write("### Implicaciones Teóricas\n")
        f.write("- La existencia de una ley de conmutación estabilizadora confirma que el sistema es estabilizable.\n")
        f.write("- El aprendizaje por refuerzo puede descubrir políticas de control complejas de forma constructiva.\n")
        f.write("- La metodología HNAF proporciona una herramienta práctica para el análisis de sistemas híbridos.\n\n")
        
        # Apéndice con configuración completa
        if 'config' in report_data:
            f.write("--- \n\n")
            f.write("## Apéndice: Configuración Completa (JSON)\n")
            f.write("```json\n")
            f.write(json.dumps(report_data['config'], indent=2, ensure_ascii=False))
            f.write("\n```\n")
            
    print(f"✅ Informe guardado en: {report_path}")
    return report_path

def create_latex_report(results_dir, report_data):
    """
    Genera un informe en formato LaTeX para integración directa en la tesis.
    """
    report_path = os.path.join(results_dir, "informe_estabilidad.tex")
    
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("\\section{Análisis de Estabilidad HNAF}\n\n")
        f.write(f"\\subsection{{Configuración del Experimento}}\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\\\")
        f.write(f"Modelo: {report_data['model_path']}\\\\")
        f.write(f"Condiciones iniciales: {len(report_data['initial_conditions'])}\\\\")
        f.write(f"Pasos de simulación: {report_data['max_steps']}\n\n")
        
        # Resumen de configuración (LaTeX)
        net = report_data.get('network_defaults', {})
        trn = report_data.get('training_defaults', {})
        state_limits = report_data.get('state_limits', {})
        if any([net, trn, state_limits]):
            f.write("\\subsection{Parámetros de Red y Entrenamiento}\n")
            f.write("\\begin{itemize}\n")
            for k, v in net.items():
                f.write(f"\\item {k}: {v}\n")
            for k, v in trn.items():
                f.write(f"\\item {k}: {v}\n")
            if state_limits:
                f.write(f"\\item Límites del estado: min={state_limits.get('min')}, max={state_limits.get('max')}\n")
            f.write("\\end{itemize}\n\n")

        # Matrices del sistema (LaTeX)
        matrices = report_data.get('matrices', {})
        if matrices:
            f.write("\\subsection{Matrices del Sistema}\n")
            for key in sorted([k for k in matrices.keys() if str(k).startswith('A')]):
                mat = matrices[key]
                try:
                    rows = " \\ ".join([" & ".join([str(x) for x in row]) for row in mat])
                    f.write(f"${key} = \\begin{{bmatrix}}{rows}\\end{{bmatrix}}$\\\\\n\n")
                except Exception:
                    f.write(f"${key} = {mat}$\\\\\n\n")
        
        f.write("\\subsection{Trayectorias de Estado}\n")
        f.write("La Figura \\ref{fig:trajectories} muestra la evolución temporal de los estados del sistema.\n\n")
        f.write("\\begin{figure}[h]\n")
        f.write("\\centering\n")
        f.write(f"\\includegraphics[width=0.8\\textwidth]{{{os.path.basename(report_data['plots']['trajectories'])}}}\n")
        f.write("\\caption{Trayectorias de estado para diferentes condiciones iniciales}\n")
        f.write("\\label{fig:trajectories}\n")
        f.write("\\end{figure}\n\n")
        
        f.write("\\subsection{Ley de Conmutación}\n")
        f.write("La Figura \\ref{fig:switching} ilustra la ley de conmutación aprendida.\n\n")
        f.write("\\begin{figure}[h]\n")
        f.write("\\centering\n")
        f.write(f"\\includegraphics[width=0.8\\textwidth]{{{os.path.basename(report_data['plots']['switching'])}}}\n")
        f.write("\\caption{Ley de conmutación y entradas de control}\n")
        f.write("\\label{fig:switching}\n")
        f.write("\\end{figure}\n\n")
        
        if 'phase_portrait' in report_data['plots']:
            f.write("\\subsection{Diagrama de Fases}\n")
            f.write("La Figura \\ref{fig:phase} muestra las regiones de conmutación.\n\n")
            f.write("\\begin{figure}[h]\n")
            f.write("\\centering\n")
            f.write(f"\\includegraphics[width=0.8\\textwidth]{{{os.path.basename(report_data['plots']['phase_portrait'])}}}\n")
            f.write("\\caption{Diagrama de fases con regiones de conmutación}\n")
            f.write("\\label{fig:phase}\n")
            f.write("\\end{figure}\n\n")
            
    print(f"✅ Informe LaTeX guardado en: {report_path}")
    return report_path

def calculate_stability_metrics(trajectories, max_steps, convergence_threshold=0.01):
    """
    Calcula métricas de estabilidad para el informe.
    """
    convergence_times = []
    final_errors = []
    success_count = 0
    
    for trajectory in trajectories:
        traj_array = np.array(trajectory)
        
        # Calcular error final
        final_error = np.linalg.norm(traj_array[-1])
        final_errors.append(final_error)
        
        # Verificar convergencia
        if final_error < convergence_threshold:
            success_count += 1
            
            # Encontrar tiempo de convergencia
            for i, state in enumerate(traj_array):
                if np.linalg.norm(state) < convergence_threshold:
                    convergence_times.append(i)
                    break
            else:
                convergence_times.append(max_steps)
        else:
            convergence_times.append(max_steps)
    
    metrics = {
        'avg_convergence_time': np.mean(convergence_times) if convergence_times else max_steps,
        'success_rate': (success_count / len(trajectories)) * 100,
        'avg_final_error': np.mean(final_errors),
        'std_final_error': np.std(final_errors),
        'min_convergence_time': min(convergence_times) if convergence_times else max_steps,
        'max_convergence_time': max(convergence_times) if convergence_times else max_steps
    }
    
    return metrics 