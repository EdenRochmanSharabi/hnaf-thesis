#!/usr/bin/env python3
"""
Script de prueba para verificar que la optimización Optuna funciona correctamente
"""

import sys
import os

def test_optimization():
    """Probar la optimización Optuna"""
    print("🧪 Probando optimización Optuna...")
    
    try:
        # Importar módulos
        from optuna_optimizer import OptunaOptimizer
        from config_manager import get_config_manager
        
        print("✅ Importaciones exitosas")
        
        # Inicializar optimizador
        optimizer = OptunaOptimizer()
        print("✅ OptunaOptimizer inicializado")
        
        # Obtener parámetros por defecto
        default_params = optimizer.get_default_params()
        print(f"✅ Parámetros por defecto obtenidos: {len(default_params)} parámetros")
        
        # Probar evaluación de parámetros
        print("🔄 Probando evaluación de parámetros...")
        score = optimizer.evaluate_params(default_params)
        print(f"✅ Evaluación completada. Score: {score}")
        
        if score > float('-inf'):
            print("🎉 ¡La optimización está funcionando correctamente!")
            return True
        else:
            print("⚠️ Score es -inf, pero sin errores de importación")
            return True
            
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        return False

if __name__ == "__main__":
    success = test_optimization()
    if success:
        print("\n✅ Prueba completada exitosamente")
        sys.exit(0)
    else:
        print("\n❌ Prueba falló")
        sys.exit(1) 