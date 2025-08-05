#!/usr/bin/env python3
"""
Monitor para entrenamiento 3D y evaluación automática
"""

import os
import time
import subprocess
from datetime import datetime

def monitor_3d_training():
    """Monitorea el entrenamiento 3D y ejecuta evaluación cuando termine"""
    
    print("🔍 Monitoreando entrenamiento 3D...")
    print("📊 Esperando que termine el entrenamiento...")
    
    # Esperar a que termine el entrenamiento
    while True:
        # Verificar si el proceso de entrenamiento sigue corriendo
        try:
            result = subprocess.run(['pgrep', '-f', 'training_monitor.py'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("✅ Entrenamiento 3D completado!")
                break
        except:
            pass
        
        time.sleep(10)  # Verificar cada 10 segundos
        print(f"⏳ Esperando... {datetime.now().strftime('%H:%M:%S')}")
    
    print("🚀 Iniciando evaluación 3D...")
    
    # Ejecutar evaluación
    try:
        subprocess.run(['python', 'evaluate_current_model.py'], check=True)
        print("✅ Evaluación 3D completada!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en evaluación: {e}")
    
    # Ejecutar generación de resultados finales
    try:
        subprocess.run(['python', 'generate_final_results.py'], check=True)
        print("✅ Resultados finales 3D generados!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generando resultados: {e}")
    
    print("🎉 Pipeline 3D completado!")

if __name__ == "__main__":
    monitor_3d_training() 