#!/usr/bin/env python3
"""
HNAF - Hybrid Normalized Advantage Function
Aplicación principal para entrenamiento y evaluación
"""

import sys
import os
import tkinter as tk

# Configurar path del proyecto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from final_app.gui_interface import HNAFGUI


def main():
    """Punto de entrada principal de la aplicación HNAF."""
    # Banner de inicio
    print("=" * 60)
    print("HNAF - Hybrid Normalized Advantage Function")
    print("Aplicación Modular")
    print("=" * 60)
    
    # Crear interfaz gráfica
    root = tk.Tk()
    app = HNAFGUI(root)
    
    # Configurar cierre limpio
    def on_closing():
        sys.stdout = sys.__stdout__  # Restaurar stdout original
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Iniciar aplicación
    print("🎮 Iniciando interfaz gráfica")
    root.mainloop()


if __name__ == "__main__":
    main() 