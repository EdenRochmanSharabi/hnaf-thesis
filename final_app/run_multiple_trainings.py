#!/usr/bin/env python3
"""
Script para ejecutar múltiples entrenamientos completos
Guarda los resultados en carpetas separadas para análisis comparativo
"""

import sys
import os
import shutil
import json
from datetime import datetime
import subprocess
import time

def create_training_directory(training_num):
    """Crear directorio para un entrenamiento específico"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"training_{training_num}_{timestamp}"
    dir_path = os.path.join("logs", "trainings", dir_name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def run_training(training_num, max_episodes=2000):
    """Ejecutar un entrenamiento completo"""
    print(f"🚀 INICIANDO ENTRENAMIENTO {training_num}")
    print("=" * 60)
    
    # Crear directorio para este entrenamiento
    training_dir = create_training_directory(training_num)
    print(f"📁 Directorio de resultados: {training_dir}")
    
    # Ejecutar el entrenamiento
    start_time = time.time()
    try:
        # Ejecutar el entrenamiento usando app.py
        cmd = f"python app.py --cli --iterations 1"
        print(f"📊 Ejecutando: {cmd}")
        
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=os.getcwd()
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Guardar la salida del entrenamiento
        output_file = os.path.join(training_dir, "training_output.txt")
        with open(output_file, 'w') as f:
            f.write(f"=== ENTRENAMIENTO {training_num} ===\n")
            f.write(f"Fecha: {datetime.now().isoformat()}\n")
            f.write(f"Duración: {duration:.2f} segundos\n")
            f.write(f"Comando: {cmd}\n")
            f.write(f"Exit code: {result.returncode}\n")
            f.write("\n=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)
        
        # Copiar archivos de log detallado si existen
        detailed_logs_dir = "logs/detailed"
        if os.path.exists(detailed_logs_dir):
            # Encontrar los archivos más recientes
            log_files = []
            for file in os.listdir(detailed_logs_dir):
                if file.startswith("detailed_training_") or file.startswith("training_data_"):
                    log_files.append(file)
            
            # Copiar los archivos más recientes
            for file in log_files:
                src = os.path.join(detailed_logs_dir, file)
                dst = os.path.join(training_dir, file)
                shutil.copy2(src, dst)
                print(f"📋 Copiado: {file}")
        
        # Crear resumen del entrenamiento
        summary = {
            "training_num": training_num,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "exit_code": result.returncode,
            "success": result.returncode == 0,
            "output_file": output_file,
            "training_dir": training_dir
        }
        
        summary_file = os.path.join(training_dir, "training_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if result.returncode == 0:
            print(f"✅ ENTRENAMIENTO {training_num} COMPLETADO EXITOSAMENTE")
            print(f"⏱️  Duración: {duration:.2f} segundos")
        else:
            print(f"❌ ENTRENAMIENTO {training_num} FALLÓ")
            print(f"Exit code: {result.returncode}")
        
        return summary
        
    except Exception as e:
        print(f"💥 ERROR en entrenamiento {training_num}: {e}")
        return {
            "training_num": training_num,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "success": False,
            "training_dir": training_dir
        }

def analyze_training_results(trainings_data):
    """Analizar los resultados de todos los entrenamientos"""
    print("\n" + "=" * 60)
    print("📊 ANÁLISIS DE RESULTADOS DE TODOS LOS ENTRENAMIENTOS")
    print("=" * 60)
    
    successful_trainings = [t for t in trainings_data if t.get('success', False)]
    failed_trainings = [t for t in trainings_data if not t.get('success', False)]
    
    print(f"✅ Entrenamientos exitosos: {len(successful_trainings)}")
    print(f"❌ Entrenamientos fallidos: {len(failed_trainings)}")
    
    if successful_trainings:
        avg_duration = sum(t.get('duration_seconds', 0) for t in successful_trainings) / len(successful_trainings)
        print(f"⏱️  Duración promedio: {avg_duration:.2f} segundos")
    
    # Crear resumen general
    summary = {
        "total_trainings": len(trainings_data),
        "successful_trainings": len(successful_trainings),
        "failed_trainings": len(failed_trainings),
        "success_rate": len(successful_trainings) / len(trainings_data) if trainings_data else 0,
        "trainings": trainings_data
    }
    
    return summary

def main():
    """Función principal"""
    print("🎯 EJECUTOR DE MÚLTIPLES ENTRENAMIENTOS HNAF")
    print("=" * 60)
    
    # Configurar número de entrenamientos
    num_trainings = 3  # Puedes cambiar este número
    
    print(f"📈 Se ejecutarán {num_trainings} entrenamientos completos")
    print("⚠️  Cada entrenamiento puede tomar varios minutos")
    
    # Crear directorio principal para todos los entrenamientos
    main_dir = os.path.join("logs", "trainings")
    os.makedirs(main_dir, exist_ok=True)
    
    # Ejecutar entrenamientos
    trainings_data = []
    
    for i in range(1, num_trainings + 1):
        print(f"\n{'='*20} ENTRENAMIENTO {i}/{num_trainings} {'='*20}")
        
        summary = run_training(i)
        trainings_data.append(summary)
        
        # Pausa entre entrenamientos
        if i < num_trainings:
            print("⏳ Esperando 5 segundos antes del siguiente entrenamiento...")
            time.sleep(5)
    
    # Analizar resultados
    overall_summary = analyze_training_results(trainings_data)
    
    # Guardar resumen general
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_summary_file = os.path.join(main_dir, f"overall_summary_{timestamp}.json")
    with open(overall_summary_file, 'w') as f:
        json.dump(overall_summary, f, indent=2)
    
    print(f"\n📋 Resumen general guardado en: {overall_summary_file}")
    print("\n🎉 TODOS LOS ENTRENAMIENTOS COMPLETADOS")
    
    return overall_summary

if __name__ == "__main__":
    try:
        summary = main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n⏹️  Ejecución interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 ERROR GENERAL: {e}")
        sys.exit(1) 