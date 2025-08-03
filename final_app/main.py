#!/usr/bin/env python3
"""
HNAF - Hybrid Normalized Advantage Function
Aplicación principal para entrenamiento y evaluación
SIN VALORES HARDCODEADOS - Todo desde config.yaml
"""

import sys
import os
import tkinter as tk

# Configurar path del proyecto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from final_app.config_manager import get_config_manager
from final_app.gui_interface import HNAFGUI


def main():
    """Punto de entrada principal de la aplicación HNAF."""
    try:
        # Cargar configuración (sin valores hardcodeados)
        config_manager = get_config_manager()
        config_manager.validate_config()
        
        app_config = config_manager.get_app_config()
        
        # Banner de inicio (configurable)
        banner_width = app_config['banner']['width']
        banner_char = app_config['banner']['char']
        print(banner_char * banner_width)
        print("HNAF - Hybrid Normalized Advantage Function")
        print("Aplicación Modular - SIN VALORES HARDCODEADOS")
        print(banner_char * banner_width)
        
        # Crear interfaz gráfica (dimensiones configurables)
        root = tk.Tk()
        root.title(app_config['title'])
        window_size = f"{app_config['window']['width']}x{app_config['window']['height']}"
        root.geometry(window_size)
        
        app = HNAFGUI(root, config_manager)
        
        # Configurar cierre limpio
        def on_closing():
            sys.stdout = sys.__stdout__  # Restaurar stdout original
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Iniciar aplicación
        print("🎮 Iniciando interfaz gráfica con configuración desde config.yaml")
        root.mainloop()
        
    except Exception as e:
        error_msg = f"❌ ERROR CRÍTICO EN INICIALIZACIÓN:\n" \
                   f"   Error: {e}\n" \
                   f"   APLICACIÓN ABORTADA - Revisa config.yaml"
        print(error_msg)
        return 1
    
    return 0


if __name__ == "__main__":
    main() 