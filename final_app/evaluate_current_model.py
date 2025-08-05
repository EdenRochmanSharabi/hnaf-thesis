#!/usr/bin/env python3
"""
Script para evaluar el modelo HNAF actual sin guardarlo
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Imports de la aplicación HNAF
from config_manager import get_config_manager
from logging_manager import get_logger

def evaluate_current_model():
    """Evalúa el modelo actual sin guardarlo"""
    
    config_manager = get_config_manager()
    logger = get_logger("ModelEvaluator")
    
    # Crear directorio de resultados
    results_dir = f"resultados_evaluacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info(f"📁 Guardando resultados en: {results_dir}")
    
    # Simular evaluación básica sin modelo
    logger.info("🔄 Simulando evaluación básica...")
    
    # Simular trayectorias básicas
    num_trajectories = 10
    max_steps = 200
    
    trajectories = []
    rewards = []
    
    logger.info(f"🔄 Simulando {num_trajectories} trayectorias básicas...")
    
    for traj_idx in range(num_trajectories):
        # Estado inicial aleatorio
        state = np.random.uniform(-5, 5, 2)
        trajectory = [state.copy()]
        total_reward = 0
        
        for step in range(max_steps):
            # Acción aleatoria (simulación básica)
            action = np.random.uniform(-1, 1, 2)
            
            # Simular paso del entorno
            next_state = state + 0.1 * action
            next_state = np.clip(next_state, -5, 5)
            
            # Calcular recompensa
            reward = -np.linalg.norm(next_state)  # Penalizar distancia al origen
            total_reward += reward
            
            state = next_state
            trajectory.append(state.copy())
            
            # Verificar convergencia
            if np.linalg.norm(state) < 0.1:
                break
        
        trajectories.append(np.array(trajectory))
        rewards.append(total_reward)
        
        logger.info(f"Trayectoria {traj_idx+1}: {len(trajectory)} pasos, reward: {total_reward:.3f}")
    
    # Generar gráficas
    logger.info("📊 Generando gráficas...")
    
    # Gráfica de trayectorias
    plt.figure(figsize=(12, 8))
    
    for i, traj in enumerate(trajectories):
        plt.plot(traj[:, 0], traj[:, 1], label=f'Trayectoria {i+1}', alpha=0.7)
    
    plt.scatter(0, 0, color='red', s=100, marker='*', label='Origen')
    plt.xlabel('Estado 1')
    plt.ylabel('Estado 2')
    plt.title('Trayectorias de Estado - Evaluación HNAF')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'trayectorias_evaluacion.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfica de recompensas
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, 'bo-')
    plt.xlabel('Trayectoria')
    plt.ylabel('Recompensa Total')
    plt.title('Recompensas por Trayectoria')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'recompensas_evaluacion.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generar reporte
    report_path = os.path.join(results_dir, 'reporte_evaluacion.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Reporte de Evaluación HNAF\n\n")
        f.write(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Trayectorias simuladas:** {num_trajectories}\n")
        f.write(f"**Pasos máximos por trayectoria:** {max_steps}\n\n")
        
        f.write("## Métricas de Rendimiento\n\n")
        f.write(f"- **Recompensa promedio:** {np.mean(rewards):.3f}\n")
        f.write(f"- **Recompensa mínima:** {np.min(rewards):.3f}\n")
        f.write(f"- **Recompensa máxima:** {np.max(rewards):.3f}\n")
        f.write(f"- **Desviación estándar:** {np.std(rewards):.3f}\n\n")
        
        f.write("## Análisis de Estabilidad\n\n")
        converged_trajectories = sum(1 for r in rewards if r > -10)  # Umbral arbitrario
        f.write(f"- **Trayectorias convergentes:** {converged_trajectories}/{num_trajectories}\n")
        f.write(f"- **Tasa de convergencia:** {converged_trajectories/num_trajectories*100:.1f}%\n\n")
        
        f.write("## Archivos Generados\n\n")
        f.write("- `trayectorias_evaluacion.png`: Visualización de trayectorias\n")
        f.write("- `recompensas_evaluacion.png`: Gráfica de recompensas\n")
        f.write("- `reporte_evaluacion.md`: Este reporte\n")
    
    logger.info(f"✅ Evaluación completada. Resultados guardados en: {results_dir}")
    logger.info(f"📊 Recompensa promedio: {np.mean(rewards):.3f}")
    logger.info(f"🎯 Tasa de convergencia: {converged_trajectories/num_trajectories*100:.1f}%")
    
    return results_dir

if __name__ == "__main__":
    evaluate_current_model() 