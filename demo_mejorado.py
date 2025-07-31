#!/usr/bin/env python3
"""
DEMOSTRACIÓN COMPARATIVA: HNAF ORIGINAL vs HNAF MEJORADO
Muestra las mejoras implementadas en la nueva versión.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from src.hnaf_stable import train_stable_hnaf
from src.hnaf_improved import train_improved_hnaf

def comparar_entrenamientos():
    """Comparar entrenamiento del HNAF original vs mejorado"""
    print("="*80)
    print("COMPARACIÓN: HNAF ORIGINAL vs HNAF MEJORADO")
    print("="*80)
    
    # Parámetros de comparación
    num_episodes = 500  # Menos episodios para comparación rápida
    eval_interval = 50
    
    print("\n🔍 Entrenando HNAF ORIGINAL...")
    start_time = time.time()
    
    # Entrenar HNAF original
    hnaf_original, results_original = train_stable_hnaf(
        num_episodes=num_episodes,
        eval_interval=eval_interval
    )
    
    time_original = time.time() - start_time
    print(f"⏱️  Tiempo HNAF original: {time_original:.2f} segundos")
    
    print("\n🚀 Entrenando HNAF MEJORADO...")
    start_time = time.time()
    
    # Entrenar HNAF mejorado
    hnaf_mejorado, results_mejorado = train_improved_hnaf(
        num_episodes=num_episodes,
        eval_interval=eval_interval,
        initial_epsilon=0.5,
        final_epsilon=0.05,
        hidden_dim=64,
        num_layers=3,
        lr=1e-4
    )
    
    time_mejorado = time.time() - start_time
    print(f"⏱️  Tiempo HNAF mejorado: {time_mejorado:.2f} segundos")
    
    # Comparar resultados
    print("\n" + "="*80)
    print("RESULTADOS COMPARATIVOS")
    print("="*80)
    
    # Recompensas finales
    final_reward_original = np.mean(results_original['episode_rewards'][-50:])
    final_reward_mejorado = np.mean(results_mejorado['episode_rewards'][-50:])
    
    print(f"📊 Recompensa final (últimos 50 episodios):")
    print(f"   HNAF Original: {final_reward_original:.4f}")
    print(f"   HNAF Mejorado: {final_reward_mejorado:.4f}")
    print(f"   Mejora: {((final_reward_mejorado - final_reward_original) / abs(final_reward_original) * 100):.1f}%")
    
    # Evaluación en grid
    if 'grid_accuracies' in results_mejorado and results_mejorado['grid_accuracies']:
        final_accuracy = results_mejorado['grid_accuracies'][-1]
        print(f"\n🎯 Precisión en grid (HNAF Mejorado): {final_accuracy:.2%}")
    
    # Tiempo de entrenamiento
    print(f"\n⏱️  Tiempo de entrenamiento:")
    print(f"   HNAF Original: {time_original:.2f}s")
    print(f"   HNAF Mejorado: {time_mejorado:.2f}s")
    print(f"   Diferencia: {time_mejorado - time_original:.2f}s")
    
    # Visualización comparativa
    visualizar_comparacion(results_original, results_mejorado)
    
    return hnaf_original, hnaf_mejorado, results_original, results_mejorado

def visualizar_comparacion(results_original, results_mejorado):
    """Visualizar comparación de resultados"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Gráfico 1: Recompensas de episodios
    axes[0, 0].plot(results_original['episode_rewards'], alpha=0.6, label='HNAF Original', color='blue')
    axes[0, 0].plot(results_mejorado['episode_rewards'], alpha=0.6, label='HNAF Mejorado', color='red')
    axes[0, 0].set_title('Recompensas de Episodios')
    axes[0, 0].set_xlabel('Episodio')
    axes[0, 0].set_ylabel('Recompensa')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: Recompensas de evaluación
    eval_episodes_orig = np.arange(50, len(results_original['episode_rewards']) + 1, 50)
    eval_episodes_mejor = np.arange(50, len(results_mejorado['episode_rewards']) + 1, 50)
    
    axes[0, 1].plot(eval_episodes_orig, results_original['eval_rewards'], 'b-o', label='HNAF Original')
    axes[0, 1].plot(eval_episodes_mejor, results_mejorado['eval_rewards'], 'r-o', label='HNAF Mejorado')
    axes[0, 1].set_title('Recompensas de Evaluación')
    axes[0, 1].set_xlabel('Episodio')
    axes[0, 1].set_ylabel('Recompensa')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfico 3: Pérdidas
    if results_original['losses'] and results_mejorado['losses']:
        axes[1, 0].plot(results_original['losses'], alpha=0.6, label='HNAF Original', color='blue')
        axes[1, 0].plot(results_mejorado['losses'], alpha=0.6, label='HNAF Mejorado', color='red')
        axes[1, 0].set_title('Pérdidas de Entrenamiento')
        axes[1, 0].set_xlabel('Actualización')
        axes[1, 0].set_ylabel('Pérdida')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Gráfico 4: Precisión en grid (solo HNAF mejorado)
    if 'grid_accuracies' in results_mejorado and results_mejorado['grid_accuracies']:
        eval_episodes = np.arange(50, len(results_mejorado['episode_rewards']) + 1, 50)
        axes[1, 1].plot(eval_episodes, results_mejorado['grid_accuracies'], 'purple', linewidth=2, marker='o')
        axes[1, 1].set_title('Precisión en Grid (HNAF Mejorado)')
        axes[1, 1].set_xlabel('Episodio')
        axes[1, 1].set_ylabel('Precisión')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1)
    else:
        axes[1, 1].text(0.5, 0.5, 'Grid accuracy\nno disponible', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Precisión en Grid')
    
    plt.tight_layout()
    plt.show()

def demostrar_mejoras():
    """Demostrar las mejoras específicas implementadas"""
    print("\n" + "="*80)
    print("DEMOSTRACIÓN DE MEJORAS IMPLEMENTADAS")
    print("="*80)
    
    print("\n1. 🎯 ε-greedy decay:")
    print("   - Original: ε fijo = 0.2")
    print("   - Mejorado: ε decay de 0.5 → 0.05")
    print("   - Beneficio: Más exploración al inicio, más explotación al final")
    
    print("\n2. 🧠 Red más profunda:")
    print("   - Original: 1 capa de 32 unidades")
    print("   - Mejorado: 3 capas de 64 unidades + BatchNorm")
    print("   - Beneficio: Mayor capacidad de aprendizaje")
    
    print("\n3. 📊 Prioritized Experience Replay:")
    print("   - Original: Buffer uniforme de 5000")
    print("   - Mejorado: Buffer priorizado de 10000")
    print("   - Beneficio: Enfoque en transiciones difíciles")
    
    print("\n4. 📈 Normalización:")
    print("   - Original: Sin normalización")
    print("   - Mejorado: Normalización de estados y recompensas")
    print("   - Beneficio: Entrenamiento más estable")
    
    print("\n5. 🎁 Reward shaping:")
    print("   - Original: r = -|‖x'‖-‖x₀‖|/15")
    print("   - Mejorado: r = -|‖x'‖-‖x₀‖|/15 + 0.1·(modo_pred==modo_opt)")
    print("   - Beneficio: Guía adicional para selección de modos")
    
    print("\n6. 🔍 Evaluación mejorada:")
    print("   - Original: Evaluación en episodios")
    print("   - Mejorado: Evaluación en grid 100x100 + episodios")
    print("   - Beneficio: Métrica más robusta de precisión")
    
    print("\n7. ⏱️  Horizonte más largo:")
    print("   - Original: 20 pasos por episodio")
    print("   - Mejorado: 50 pasos por episodio")
    print("   - Beneficio: Mejor comprensión de consecuencias")

def main():
    """Función principal"""
    print("🎉 DEMOSTRACIÓN COMPARATIVA HNAF")
    print("="*80)
    print("Este script compara el HNAF original con el HNAF mejorado")
    print("para mostrar las optimizaciones implementadas.")
    print("="*80)
    
    try:
        # Demostrar mejoras
        demostrar_mejoras()
        
        # Comparar entrenamientos
        hnaf_original, hnaf_mejorado, results_original, results_mejorado = comparar_entrenamientos()
        
        print("\n" + "="*80)
        print("🎊 ¡COMPARACIÓN COMPLETADA!")
        print("="*80)
        print("✅ HNAF Original: Entrenado exitosamente")
        print("✅ HNAF Mejorado: Entrenado con optimizaciones")
        print("✅ Comparación: Métricas calculadas")
        print("✅ Visualización: Gráficos generados")
        print("\n🚀 El HNAF mejorado debería mostrar mejor rendimiento")
        print("   en términos de convergencia y precisión.")
        
    except Exception as e:
        print(f"\n❌ Error en la demostración: {e}")
        print("Por favor, verifica que todos los archivos estén presentes.")

if __name__ == "__main__":
    main() 