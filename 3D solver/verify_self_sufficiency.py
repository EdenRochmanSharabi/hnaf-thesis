#!/usr/bin/env python3
"""
Script de verificación para asegurar que el directorio 3D sea completamente autosuficiente
"""

import os
import sys
import importlib
import subprocess

def check_file_exists(filename):
    """Verificar si un archivo existe"""
    if os.path.exists(filename):
        print(f"✅ {filename}")
        return True
    else:
        print(f"❌ {filename} - FALTANTE")
        return False

def check_import(module_name):
    """Verificar si un módulo se puede importar"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} - Importable")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - Error de importación: {e}")
        return False

def check_config():
    """Verificar configuración"""
    try:
        from config_manager import get_config_manager
        cm = get_config_manager()
        a1 = cm.get("defaults.matrices.A1")
        if len(a1) == 3 and len(a1[0]) == 3:
            print(f"✅ Configuración 3D correcta: A1 = {a1}")
            return True
        else:
            print(f"❌ Configuración incorrecta: A1 = {a1}")
            return False
    except Exception as e:
        print(f"❌ Error verificando configuración: {e}")
        return False

def main():
    """Función principal de verificación"""
    print("🔍 VERIFICANDO AUTOSUFICIENCIA DEL DIRECTORIO 3D")
    print("=" * 60)
    
    # Lista de archivos críticos
    critical_files = [
        "app.py",
        "training_monitor.py", 
        "training_manager.py",
        "hnaf_improved.py",
        "config_manager.py",
        "logging_manager.py",
        "gui_interface.py",
        "config.yaml",
        "evaluate_policy.py",
        "generate_thesis_results.py",
        "generate_final_results.py",
        "reporting_utils.py",
        "plot_utils.py",
        "model_saver.py",
        "noise_process.py",
        "optuna_optimizer.py",
        "run_multiple_trainings.py",
        "run_trainings_with_monitor.py",
        "detailed_logger.py",
        "test_detailed_logging.py",
        "test_optimization.py",
        "README_3D.md",
        "PROBLEMA_MATRICES_3D.md",
        "README_TESIS.md",
        ".gitignore",
        "__init__.py",
        "run_complete_pipeline.sh",
        "monitor_3d_training.py",
        "evaluate_current_model.py"
    ]
    
    print("\n📁 VERIFICANDO ARCHIVOS CRÍTICOS:")
    missing_files = []
    for file in critical_files:
        if not check_file_exists(file):
            missing_files.append(file)
    
    print(f"\n📊 RESUMEN ARCHIVOS:")
    print(f"   Total archivos críticos: {len(critical_files)}")
    print(f"   Archivos encontrados: {len(critical_files) - len(missing_files)}")
    print(f"   Archivos faltantes: {len(missing_files)}")
    
    if missing_files:
        print(f"\n❌ ARCHIVOS FALTANTES:")
        for file in missing_files:
            print(f"   - {file}")
    
    print(f"\n🔧 VERIFICANDO IMPORTS:")
    modules_to_check = [
        "config_manager",
        "logging_manager", 
        "training_manager",
        "hnaf_improved",
        "gui_interface",
        "evaluate_policy",
        "generate_thesis_results",
        "reporting_utils",
        "plot_utils",
        "model_saver",
        "noise_process",
        "optuna_optimizer",
        "detailed_logger"
    ]
    
    failed_imports = []
    for module in modules_to_check:
        if not check_import(module):
            failed_imports.append(module)
    
    print(f"\n📊 RESUMEN IMPORTS:")
    print(f"   Total módulos: {len(modules_to_check)}")
    print(f"   Imports exitosos: {len(modules_to_check) - len(failed_imports)}")
    print(f"   Imports fallidos: {len(failed_imports)}")
    
    if failed_imports:
        print(f"\n❌ IMPORTS FALLIDOS:")
        for module in failed_imports:
            print(f"   - {module}")
    
    print(f"\n⚙️ VERIFICANDO CONFIGURACIÓN:")
    config_ok = check_config()
    
    print(f"\n🎯 VERIFICACIÓN COMPLETA")
    print("=" * 60)
    
    if not missing_files and not failed_imports and config_ok:
        print("✅ DIRECTORIO 3D COMPLETAMENTE AUTOSUFICIENTE")
        print("   - Todos los archivos críticos presentes")
        print("   - Todos los módulos importables")
        print("   - Configuración 3D correcta")
        return True
    else:
        print("❌ DIRECTORIO 3D NO AUTOSUFICIENTE")
        if missing_files:
            print(f"   - {len(missing_files)} archivos faltantes")
        if failed_imports:
            print(f"   - {len(failed_imports)} imports fallidos")
        if not config_ok:
            print("   - Configuración incorrecta")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 