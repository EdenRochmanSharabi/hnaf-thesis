#!/usr/bin/env python3
"""
HNAF Application - Versión Simplificada y Limpia
==================================================

Este archivo contiene todo lo necesario para ejecutar la aplicación HNAF.
Incluye:
- GUI principal
- Entrenamiento
- Optimización
- Análisis de estabilidad
- CLI para ejecución automática

Uso:
    python app.py                    # Ejecutar GUI
    python app.py --cli              # Ejecutar en modo CLI
    python app.py --train            # Solo entrenamiento
    python app.py --optimize         # Solo optimización
"""

import sys
import os
import argparse
from pathlib import Path

# Añadir el directorio padre al path para importar src
sys.path.append(str(Path(__file__).parent.parent))

# Imports principales
from gui_interface import HNAFGUI
from config_manager import get_config_manager
from training_manager import TrainingManager
from optuna_optimizer import OptunaOptimizer

from logging_manager import get_logger

def run_gui():
    """Ejecutar la interfaz gráfica principal"""
    print("🎮 Iniciando HNAF GUI...")
    try:
        import tkinter as tk
        root = tk.Tk()
        config_manager = get_config_manager()
        gui = HNAFGUI(root=root, config_manager=config_manager)
        gui.run()
    except Exception as e:
        print(f"❌ Error al iniciar GUI: {e}")
        return False
    return True

def run_cli_training(iterations=1):
    """Ejecutar entrenamiento en modo CLI"""
    print(f"🔄 Ejecutando entrenamiento CLI ({iterations} iteraciones)...")
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # Ocultar ventana en modo CLI
        config_manager = get_config_manager()
        gui = HNAFGUI(root=root, config_manager=config_manager, cli_mode=True)
        
        for i in range(iterations):
            print(f"\n📊 Iteración {i+1}/{iterations}")
            precision = gui.run_training_cli()
            print(f"✅ Precisión obtenida: {precision:.2f}%")
            
    except Exception as e:
        print(f"❌ Error en entrenamiento CLI: {e}")
        return False
    return True

def run_optimization():
    """Ejecutar optimización con Optuna"""
    print("🔍 Iniciando optimización Optuna...")
    try:
        optimizer = OptunaOptimizer()
        
        # Configurar logging detallado
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('optuna_optimization.log'),
                logging.StreamHandler()
            ]
        )
        
        print("🔄 Iniciando optimización continua...")
        print("📝 Logs guardados en: optuna_optimization.log")
        print("⏰ Para detener: Ctrl+C")
        
        # Ejecutar optimización en el hilo principal (no en background)
        optimizer.optimize_loop()
        
        return True
    except KeyboardInterrupt:
        print("\n⏹️ Optimización detenida por el usuario")
        return True
    except Exception as e:
        print(f"❌ Error en optimización: {e}")
        return False

def run_stability_analysis():
    """Ejecutar análisis de estabilidad"""
    print("📊 Analizando estabilidad del sistema...")
    try:
        config_manager = get_config_manager()
        # Ejecutar un entrenamiento rápido para análisis
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # Ocultar ventana en modo CLI
        gui = HNAFGUI(root=root, config_manager=config_manager, cli_mode=True)
        precision = gui.run_training_cli()
        
        # Análisis simple
        training_data = gui.training_results if hasattr(gui, 'training_results') else {}
        if training_data and 'stability_score' in training_data:
            stability = training_data['stability_score']
            print(f"\n📈 Análisis de Estabilidad:")
            print(f"   Puntuación: {stability:.3f}")
            if stability < 0.6:
                print("   ⚠️ Estabilidad baja detectada")
            else:
                print("   ✅ Estabilidad aceptable")
        
        return True
    except Exception as e:
        print(f"❌ Error en análisis de estabilidad: {e}")
        return False

def run_improvement_loop(max_iterations=5, target_precision=80.0):
    """Ejecutar loop de mejora automática"""
    print(f"🔄 Iniciando loop de mejora ({max_iterations} iteraciones, objetivo: {target_precision}%)")
    try:
        # Loop simple de mejora
        for i in range(max_iterations):
            print(f"\n📊 Iteración {i+1}/{max_iterations}")
            precision = run_cli_training(1)
            if precision >= target_precision:
                print(f"✅ Objetivo alcanzado: {precision:.2f}%")
                break
        return True
    except Exception as e:
        print(f"❌ Error en loop de mejora: {e}")
        return False

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="HNAF Application")
    parser.add_argument("--cli", action="store_true", help="Ejecutar en modo CLI")
    parser.add_argument("--train", action="store_true", help="Solo entrenamiento")
    parser.add_argument("--optimize", action="store_true", help="Solo optimización")
    parser.add_argument("--stability", action="store_true", help="Análisis de estabilidad")
    parser.add_argument("--improve", action="store_true", help="Loop de mejora automática")
    parser.add_argument("--iterations", type=int, default=1, help="Número de iteraciones")
    parser.add_argument("--target", type=float, default=80.0, help="Precision objetivo (porcentaje)")
    
    args = parser.parse_args()
    
    print("🚀 HNAF Application - Versión Simplificada")
    print("=" * 50)
    
    try:
        if args.cli:
            success = run_cli_training(args.iterations)
        elif args.train:
            success = run_cli_training(args.iterations)
        elif args.optimize:
            success = run_optimization()
        elif args.stability:
            success = run_stability_analysis()
        elif args.improve:
            success = run_improvement_loop(args.iterations, args.target)
        else:
            # Modo por defecto: GUI
            success = run_gui()
        
        if success:
            print("\n✅ Operación completada exitosamente")
        else:
            print("\n❌ Operación falló")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Operación interrumpida por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 