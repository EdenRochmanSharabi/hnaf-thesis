#!/usr/bin/env python3
"""
Script de lanzamiento para la interfaz gráfica de HNAF
"""

import sys
import os

# Añadir el directorio actual al path para importar módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from hnaf_gui import main
    print("Iniciando interfaz gráfica de HNAF...")
    main()
except ImportError as e:
    print(f"Error al importar módulos: {e}")
    print("Asegúrate de tener todas las dependencias instaladas:")
    print("pip install -r requirements.txt")
except Exception as e:
    print(f"Error al iniciar la interfaz: {e}")
    print("Verifica que todos los archivos estén presentes en el directorio.") 