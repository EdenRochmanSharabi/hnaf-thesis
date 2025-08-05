#!/usr/bin/env python3
"""
Script para ejecutar múltiples entrenamientos con monitoreo en tiempo real
"""

import sys
import os
import time
import json
import subprocess
import threading
from datetime import datetime
import shutil
from training_monitor import TrainingMonitor

class TrainingRunner:
    """Ejecutor de entrenamientos con monitoreo"""
    
    def __init__(self):
        self.monitor = TrainingMonitor()
        self.trainings_dir = os.path.join("logs", "trainings")
        os.makedirs(self.trainings_dir, exist_ok=True)
        
    def create_training_directory(self, training_num):
        """Crear directorio para un entrenamiento específico"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"training_{training_num}_{timestamp}"
        dir_path = os.path.join(self.trainings_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
    
    def run_single_training(self, training_num, max_episodes=2500):
        """Ejecutar un entrenamiento individual con monitoreo"""
        print(f"\n{'='*20} ENTRENAMIENTO {training_num} {'='*20}")
        
        # Crear directorio para este entrenamiento
        training_dir = self.create_training_directory(training_num)
        print(f"📁 Directorio: {training_dir}")
        
        # Configurar parámetros de entrenamiento
        config_file = "config.yaml"
        if os.path.exists(config_file):
            # Hacer backup del config actual
            backup_file = os.path.join(training_dir, "config_backup.yaml")
            shutil.copy2(config_file, backup_file)
            print(f"📋 Backup de config guardado en: {backup_file}")
        
        start_time = time.time()
        
        # Ejecutar entrenamiento con monitoreo
        try:
            # Iniciar entrenamiento en hilo separado
            def run_training():
                try:
                    cmd = "python app.py --cli --iterations 1"
                    result = subprocess.run(
                        cmd, 
                        shell=True, 
                        capture_output=True, 
                        text=True, 
                        cwd=os.getcwd()
                    )
                    return result
                except Exception as e:
                    print(f"❌ Error en entrenamiento: {e}")
                    return None
            
            # Iniciar hilo de entrenamiento
            training_thread = threading.Thread(target=run_training)
            training_thread.daemon = True
            training_thread.start()
            
            # Esperar un momento para que se inicie
            time.sleep(3)
            
            # Monitorear progreso
            self.monitor.monitor_training()
            
            # Esperar a que termine el entrenamiento
            training_thread.join(timeout=1)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Guardar resultados
            self.save_training_results(training_num, training_dir, duration)
            
            print(f"✅ ENTRENAMIENTO {training_num} COMPLETADO")
            print(f"⏱️  Duración: {duration:.2f} segundos")
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR en entrenamiento {training_num}: {e}")
            return False
    
    def save_training_results(self, training_num, training_dir, duration):
        """Guardar resultados del entrenamiento"""
        # Copiar archivos de log detallado
        detailed_logs_dir = "logs/detailed"
        if os.path.exists(detailed_logs_dir):
            log_files = []
            for file in os.listdir(detailed_logs_dir):
                if file.startswith("detailed_training_") or file.startswith("training_data_"):
                    log_files.append(file)
            
            for file in log_files:
                src = os.path.join(detailed_logs_dir, file)
                dst = os.path.join(training_dir, file)
                shutil.copy2(src, dst)
        
        # Crear resumen
        summary = {
            "training_num": training_num,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "training_dir": training_dir,
            "config_backup": os.path.join(training_dir, "config_backup.yaml")
        }
        
        summary_file = os.path.join(training_dir, "training_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def run_multiple_trainings(self, num_trainings=3):
        """Ejecutar múltiples entrenamientos secuencialmente"""
        print("🎯 EJECUTOR DE MÚLTIPLES ENTRENAMIENTOS CON MONITOREO")
        print("=" * 60)
        print(f"📈 Se ejecutarán {num_trainings} entrenamientos completos")
        print("⚠️  Cada entrenamiento puede tomar varios minutos")
        print("🔄 El monitoreo mostrará el progreso en tiempo real")
        
        results = []
        
        for i in range(1, num_trainings + 1):
            success = self.run_single_training(i)
            results.append({
                "training_num": i,
                "success": success,
                "timestamp": datetime.now().isoformat()
            })
            
            # Pausa entre entrenamientos
            if i < num_trainings:
                print(f"\n⏳ Esperando 10 segundos antes del siguiente entrenamiento...")
                time.sleep(10)
        
        # Análisis final
        self.analyze_results(results)
        
        return results
    
    def analyze_results(self, results):
        """Analizar resultados de todos los entrenamientos"""
        print("\n" + "=" * 60)
        print("📊 ANÁLISIS FINAL DE TODOS LOS ENTRENAMIENTOS")
        print("=" * 60)
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"✅ Entrenamientos exitosos: {len(successful)}")
        print(f"❌ Entrenamientos fallidos: {len(failed)}")
        print(f"📊 Tasa de éxito: {(len(successful)/len(results)*100):.1f}%")
        
        # Guardar análisis
        analysis = {
            "total_trainings": len(results),
            "successful_trainings": len(successful),
            "failed_trainings": len(failed),
            "success_rate": len(successful) / len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = os.path.join(self.trainings_dir, f"overall_analysis_{timestamp}.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\n📋 Análisis guardado en: {analysis_file}")
        print("\n🎉 TODOS LOS ENTRENAMIENTOS COMPLETADOS")

def main():
    """Función principal"""
    runner = TrainingRunner()
    
    if len(sys.argv) > 1:
        try:
            num_trainings = int(sys.argv[1])
        except ValueError:
            print("❌ Error: El número de entrenamientos debe ser un entero")
            sys.exit(1)
    else:
        num_trainings = 3  # Por defecto
    
    try:
        results = runner.run_multiple_trainings(num_trainings)
        print("\n🎯 Proceso completado exitosamente")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n⏹️  Proceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 ERROR GENERAL: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 