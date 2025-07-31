#!/usr/bin/env python3
"""
Script de debugging para HNAF
Ejecutar desde terminal: python debug.py
"""

from final_app.training_manager import TrainingManager
from final_app.evaluation_manager import EvaluationManager
import numpy as np

def debug_hnaf():
    """Debug del HNAF desde terminal"""
    print("🔧 DEBUGGING HNAF")
    print("="*60)
    
    # Parámetros de prueba
    params = {
        'state_dim': 2,
        'action_dim': 2,
        'num_modes': 2,
        'hidden_dim': 64,
        'num_layers': 3,
        'lr': 0.0001,
        'tau': 0.001,
        'gamma': 0.9,
        'num_episodes': 50,  # Reducido para debugging
        'batch_size': 16,
        'initial_epsilon': 0.5,
        'final_epsilon': 0.05,
        'max_steps': 20
    }
    
    print("Parámetros:", params)
    
    try:
        # Entrenamiento
        print("\n🚀 Entrenando...")
        manager = TrainingManager()
        model, results = manager.train_hnaf(params)
        
        print(f"\n✅ Entrenamiento completado!")
        print(f"   Recompensa final: {np.mean(results['episode_rewards'][-10:]):.4f}")
        
        # Evaluación
        print("\n📊 Evaluando...")
        eval_manager = EvaluationManager()
        eval_results = eval_manager.evaluate_model(model)
        
        print("✅ Debug completado sin errores!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_hnaf() 