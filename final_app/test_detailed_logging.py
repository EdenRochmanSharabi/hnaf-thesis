#!/usr/bin/env python3
"""
Script de prueba para entrenamiento con logging detallado
Ejecuta un entrenamiento corto para verificar que el sistema funciona correctamente
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training_manager import TrainingManager
from config_manager import get_config_manager
import numpy as np

def test_detailed_logging():
    """Ejecutar prueba de entrenamiento con logging detallado"""
    print("🚀 INICIANDO PRUEBA DE LOGGING DETALLADO")
    print("=" * 60)
    
    # Configurar parámetros de prueba
    test_params = {
        'state_dim': 2,
        'action_dim': 2,
        'num_modes': 2,
        'hidden_dim': 256,  # Reducido para prueba rápida
        'num_layers': 2,    # Reducido para prueba rápida
        'lr': 0.0001,
        'tau': 0.01,
        'gamma': 0.999,
        'num_episodes': 5,  # Solo 5 episodios para prueba
        'batch_size': 64,   # Reducido para prueba rápida
        'initial_epsilon': 0.8,
        'final_epsilon': 0.4,
        'max_steps': 50,    # Reducido para prueba rápida
        'buffer_capacity': 1000,
        'alpha': 0.9,
        'beta': 0.9,
        'supervised_episodes': 2,  # Solo 2 episodios supervisados
        'reward_normalize': False,
        'reward_shaping': True,
        'reward_optimization': 'minimizar',
        'gui_reward_function': 'mode_aware_reward',
        'gui_matrices': {
            'A1': [[1.0, 50.0], [-1.0, 1.0]],
            'A2': [[1.0, -1.0], [50.0, 1.0]]
        }
    }
    
    try:
        # Crear training manager
        training_manager = TrainingManager()
        
        # Ejecutar entrenamiento
        print("📊 Ejecutando entrenamiento de prueba...")
        hnaf_model, training_results = training_manager.train_hnaf(test_params)
        
        print("✅ Entrenamiento completado exitosamente")
        print(f"📈 Precisión final: {training_results['final_precision']:.2%}")
        print(f"🔧 Score de estabilidad: {training_results['stability_score']:.3f}")
        
        # Verificar archivos de log
        if training_manager.detailed_logger:
            print(f"📝 Log detallado guardado en: {training_manager.detailed_logger.log_file}")
            print(f"📊 Datos JSON guardados en: {training_manager.detailed_logger.json_file}")
            
            # Mostrar resumen del análisis
            analysis = training_manager.detailed_logger._generate_final_analysis()
            print("\n📋 RESUMEN DEL ANÁLISIS:")
            print(f"  Episodios totales: {analysis['total_episodes']}")
            print(f"  Pasos totales: {analysis['total_steps']}")
            print(f"  Selecciones de modo: {analysis['mode_analysis']['total_selections']}")
            print(f"  Distribución de modos: {analysis['mode_analysis']['mode_ratios']}")
            print(f"  Colapso de modos detectado: {analysis['mode_analysis']['mode_collapse_detected']}")
            print(f"  Recompensa promedio: {analysis['reward_analysis']['average_reward']:.4f}")
            print(f"  Loss promedio: {analysis['loss_analysis']['average_loss']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR en prueba: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_detailed_logging()
    if success:
        print("\n🎉 PRUEBA EXITOSA - Sistema de logging detallado funcionando correctamente")
    else:
        print("\n💥 PRUEBA FALLIDA - Revisar errores")
    
    sys.exit(0 if success else 1) 