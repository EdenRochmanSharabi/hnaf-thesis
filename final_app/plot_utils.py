# final_app/plot_utils.py

import matplotlib
matplotlib.use('Agg')  # Usar backend no-GUI para evitar problemas en macOS
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_successful_trajectories(trajectories, save_path='trayectorias'):
    """
    Dibuja y guarda las trayectorias de estabilización exitosas.

    Args:
        trajectories (list): Una lista de trayectorias. Cada trayectoria es una lista de estados (np.array).
        save_path (str): Directorio donde se guardarán las imágenes.
    """
    if not trajectories:
        print("No se encontraron trayectorias exitosas para dibujar.")
        return

    # Crear el directorio si no existe
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Seleccionar hasta 5 trayectorias de ejemplo para no saturar
    num_plots = min(len(trajectories), 5)
    
    print(f"Dibujando {num_plots} trayectorias de ejemplo...")

    fig, ax = plt.subplots(figsize=(12, 10))  # Más ancho para la leyenda
    
    # Dibujar la circunferencia unidad como referencia
    circle = plt.Circle((0, 0), 1, color='lightgray', linestyle='--', fill=False, label='Circunferencia Unidad')
    ax.add_artist(circle)
    
    # Dibujar cada trayectoria
    for i in range(num_plots):
        path = np.array(trajectories[i])
        ax.plot(path[:, 0], path[:, 1], marker='o', linestyle='-', label=f'Trayectoria {i+1}')
        # Marcar el inicio (verde) y el fin (rojo)
        ax.plot(path[0, 0], path[0, 1], 'go', markersize=10, label=f'Inicio {i+1}') 
        ax.plot(path[-1, 0], path[-1, 1], 'rx', markersize=10, label=f'Fin {i+1}')

    ax.set_title('Ejemplos de Trayectorias de Estabilización Exitosas')
    ax.set_xlabel('Estado X1')
    ax.set_ylabel('Estado X2')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    # Mover leyenda fuera del gráfico
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Guardar la figura con leyenda completa y nombre único
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'trayectorias_exitosas_{timestamp}.png'
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"✅ Gráfico de trayectorias guardado en: {filepath}")

def plot_3d_trajectories(trajectories, save_path='trayectorias'):
    """
    Dibuja trayectorias en 3D para sistemas con 3 estados.
    """
    if not trajectories:
        print("No se encontraron trayectorias exitosas para dibujar.")
        return

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    num_plots = min(len(trajectories), 3)  # Menos trayectorias en 3D para claridad
    
    print(f"Dibujando {num_plots} trayectorias 3D de ejemplo...")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Dibujar esfera unidad como referencia
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.1, color='lightgray')
    
    # Dibujar cada trayectoria
    for i in range(num_plots):
        path = np.array(trajectories[i])
        ax.plot(path[:, 0], path[:, 1], path[:, 2], marker='o', linestyle='-', label=f'Trayectoria {i+1}')
        # Marcar inicio y fin
        ax.scatter(path[0, 0], path[0, 1], path[0, 2], c='green', s=100, marker='o')
        ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], c='red', s=100, marker='x')

    ax.set_title('Trayectorias de Estabilización 3D')
    ax.set_xlabel('Estado X1')
    ax.set_ylabel('Estado X2')
    ax.set_zlabel('Estado X3')
    
    # Guardar la figura con nombre único
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'trayectorias_3d_exitosas_{timestamp}.png'
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"✅ Gráfico 3D de trayectorias guardado en: {filepath}") 