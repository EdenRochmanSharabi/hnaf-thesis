#!/usr/bin/env python3
"""
Aplicación principal HNAF
Punto de entrada para la aplicación modular
"""

import sys
import os

# Agregar el directorio raíz del proyecto al sys.path
# para que los módulos 'src' y 'final_app' puedan ser encontrados.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import tkinter as tk
from final_app.gui_interface import HNAFGUI

def main():
    """Función principal"""
    print("🚀 Iniciando aplicación HNAF modular")
    print("📁 Estructura modular:")
    print("   - gui_interface.py: Interfaz gráfica")
    print("   - training_manager.py: Lógica de entrenamiento")
    print("   - evaluation_manager.py: Evaluación de modelos")
    print("   - visualization_manager.py: Gráficos y visualizaciones")
    print("   - main.py: Punto de entrada")
    
    # Crear ventana principal
    root = tk.Tk()
    print("✅ Ventana principal creada")
    
    # Crear aplicación GUI
    app = HNAFGUI(root)
    print("✅ Interfaz HNAF creada")
    
    # Configurar cierre limpio
    def on_closing():
        print("🔄 Cerrando aplicación")
        import sys
        sys.stdout = sys.__stdout__  # Restaurar stdout
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    print("🎮 Iniciando interfaz gráfica")
    print("="*60)
    print("HNAF - Hybrid Normalized Advantage Function")
    print("Aplicación Modular")
    print("="*60)
    
    # Iniciar mainloop
    root.mainloop()
    print("✅ Aplicación cerrada exitosamente")

if __name__ == "__main__":
    main() 