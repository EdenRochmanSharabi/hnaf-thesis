#!/usr/bin/env python3
"""
Monitor de progreso para entrenamientos HNAF
Muestra el progreso en tiempo real y permite monitorear múltiples entrenamientos
"""

import os
import sys
import time
import json
import subprocess
import threading
from datetime import datetime
import glob

class TrainingMonitor:
    """Monitor de progreso para entrenamientos HNAF"""
    
    def __init__(self):
        self.logs_dir = "logs"
        self.detailed_logs_dir = os.path.join(self.logs_dir, "detailed")
        self.trainings_dir = os.path.join(self.logs_dir, "trainings")
        
    def get_latest_log_file(self):
        """Obtener el archivo de log más reciente"""
        if not os.path.exists(self.detailed_logs_dir):
            return None
            
        log_files = glob.glob(os.path.join(self.detailed_logs_dir, "detailed_training_*.log"))
        if not log_files:
            return None
            
        # Obtener el archivo más reciente
        latest_file = max(log_files, key=os.path.getctime)
        return latest_file
    
    def get_training_progress(self, log_file):
        """Extraer progreso del entrenamiento desde el archivo de log"""
        if not log_file or not os.path.exists(log_file):
            return None
            
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Buscar información de progreso
            progress_info = {
                'current_episode': 0,
                'total_episodes': 0,
                'current_step': 0,
                'max_steps': 0,
                'total_reward': 0.0,
                'mode_counts': {},
                'loss': None,
                'precision': None,
                'status': 'running'
            }
            
            for line in lines[-100:]:  # Últimas 100 líneas
                line = line.strip()
                
                # Buscar episodio actual
                if '=== EPISODIO' in line and '/' in line:
                    try:
                        parts = line.split('EPISODIO')[1].strip().split('/')
                        progress_info['current_episode'] = int(parts[0].strip())
                        progress_info['total_episodes'] = int(parts[1].split()[0].strip())
                    except:
                        pass
                
                # Buscar paso actual
                if '--- PASO' in line and '/' in line:
                    try:
                        parts = line.split('PASO')[1].strip().split('/')
                        progress_info['current_step'] = int(parts[0].strip())
                        progress_info['max_steps'] = int(parts[1].split()[0].strip())
                    except:
                        pass
                
                # Buscar recompensa total
                if 'Total reward acumulado:' in line:
                    try:
                        reward = float(line.split('Total reward acumulado:')[1].strip())
                        progress_info['total_reward'] = reward
                    except:
                        pass
                
                # Buscar conteo de modos
                if 'Mode counts:' in line:
                    try:
                        mode_str = line.split('Mode counts:')[1].strip()
                        if mode_str.startswith('{') and mode_str.endswith('}'):
                            mode_str = mode_str.replace("'", '"')
                            progress_info['mode_counts'] = json.loads(mode_str)
                    except:
                        pass
                
                # Buscar loss
                if 'Loss del step:' in line and 'None' not in line:
                    try:
                        loss = float(line.split('Loss del step:')[1].strip())
                        progress_info['loss'] = loss
                    except:
                        pass
                
                # Buscar precisión
                if 'Precisión en grid' in line:
                    try:
                        precision = float(line.split('Precisión en grid')[1].split(':')[1].strip().replace('%', ''))
                        progress_info['precision'] = precision
                    except:
                        pass
                
                # Buscar fin de entrenamiento
                if 'Entrenamiento completado' in line:
                    progress_info['status'] = 'completed'
                elif 'ERROR' in line:
                    progress_info['status'] = 'error'
            
            return progress_info
            
        except Exception as e:
            return None
    
    def display_progress(self, progress_info):
        """Mostrar progreso en consola"""
        if not progress_info:
            print("⏳ Esperando inicio del entrenamiento...")
            return
        
        # Limpiar pantalla (funciona en terminales Unix/Linux/Mac)
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🎯 MONITOR DE ENTRENAMIENTO HNAF")
        print("=" * 50)
        print(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        # Estado del entrenamiento
        status_emoji = {
            'running': '🔄',
            'completed': '✅',
            'error': '❌'
        }
        status = progress_info.get('status', 'running')
        print(f"{status_emoji.get(status, '❓')} Estado: {status.upper()}")
        print()
        
        # Progreso de episodios
        current_ep = progress_info.get('current_episode', 0)
        total_ep = progress_info.get('total_episodes', 0)
        if total_ep > 0:
            ep_percent = (current_ep / total_ep) * 100
            ep_bar = '█' * int(ep_percent / 5) + '░' * (20 - int(ep_percent / 5))
            print(f"📊 Episodios: {current_ep}/{total_ep} ({ep_percent:.1f}%)")
            print(f"   [{ep_bar}]")
        else:
            print(f"📊 Episodios: {current_ep}/?")
        print()
        
        # Progreso de pasos
        current_step = progress_info.get('current_step', 0)
        max_steps = progress_info.get('max_steps', 0)
        if max_steps > 0:
            step_percent = (current_step / max_steps) * 100
            step_bar = '█' * int(step_percent / 5) + '░' * (20 - int(step_percent / 5))
            print(f"👣 Pasos: {current_step}/{max_steps} ({step_percent:.1f}%)")
            print(f"   [{step_bar}]")
        else:
            print(f"👣 Pasos: {current_step}/?")
        print()
        
        # Métricas
        total_reward = progress_info.get('total_reward', 0.0)
        print(f"💰 Recompensa total: {total_reward:.4f}")
        
        loss = progress_info.get('loss')
        if loss is not None:
            print(f"📉 Loss actual: {loss:.6f}")
        
        precision = progress_info.get('precision')
        if precision is not None:
            print(f"🎯 Precisión: {precision:.2f}%")
        
        # Distribución de modos
        mode_counts = progress_info.get('mode_counts', {})
        if mode_counts:
            total_modes = sum(mode_counts.values())
            if total_modes > 0:
                print(f"🎲 Distribución de modos:")
                for mode, count in mode_counts.items():
                    percentage = (count / total_modes) * 100
                    bar = '█' * int(percentage / 5) + '░' * (10 - int(percentage / 5))
                    print(f"   Modo {mode}: {count} ({percentage:.1f}%) [{bar}]")
        
        print()
        print("=" * 50)
        print("💡 Presiona Ctrl+C para detener el monitoreo")
    
    def monitor_training(self, update_interval=2):
        """Monitorear entrenamiento en tiempo real"""
        print("🔍 Iniciando monitoreo de entrenamiento...")
        print("⏳ Esperando archivos de log...")
        
        try:
            while True:
                # Obtener archivo de log más reciente
                log_file = self.get_latest_log_file()
                
                # Obtener progreso
                progress = self.get_training_progress(log_file)
                
                # Mostrar progreso
                self.display_progress(progress)
                
                # Verificar si el entrenamiento ha terminado
                if progress and progress.get('status') in ['completed', 'error']:
                    print("\n🎉 Entrenamiento finalizado!")
                    break
                
                # Esperar antes de la siguiente actualización
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoreo detenido por el usuario")
    
    def run_training_with_monitor(self, training_num=1):
        """Ejecutar entrenamiento con monitoreo en tiempo real"""
        print(f"🚀 INICIANDO ENTRENAMIENTO {training_num} CON MONITOREO")
        print("=" * 60)
        
        # Iniciar entrenamiento en un hilo separado
        def run_training():
            try:
                cmd = "python app.py --cli --iterations 1"
                subprocess.run(cmd, shell=True, cwd=os.getcwd())
            except Exception as e:
                print(f"❌ Error en entrenamiento: {e}")
        
        # Iniciar hilo de entrenamiento
        training_thread = threading.Thread(target=run_training)
        training_thread.daemon = True
        training_thread.start()
        
        # Esperar un momento para que se inicie
        time.sleep(3)
        
        # Iniciar monitoreo
        self.monitor_training()

def main():
    """Función principal"""
    monitor = TrainingMonitor()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--monitor":
            # Solo monitorear entrenamiento existente
            monitor.monitor_training()
        elif sys.argv[1] == "--run":
            # Ejecutar entrenamiento con monitoreo
            training_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            monitor.run_training_with_monitor(training_num)
        else:
            print("Uso:")
            print("  python training_monitor.py --monitor    # Monitorear entrenamiento existente")
            print("  python training_monitor.py --run [num]  # Ejecutar entrenamiento con monitoreo")
    else:
        # Por defecto, monitorear entrenamiento existente
        monitor.monitor_training()

if __name__ == "__main__":
    main() 