#!/usr/bin/env python3
"""
Módulo de visualización para HNAF
Maneja gráficos y visualizaciones
"""

import numpy as np
import matplotlib.pyplot as plt

class VisualizationManager:
    """Manager para visualizaciones del HNAF"""
    
    def __init__(self):
        pass
    
    def update_plots(self, fig, ax, canvas, training_results):
        """
        Actualizar gráficos con los resultados del entrenamiento
        
        Args:
            fig: Figura de matplotlib
            ax: Ejes de matplotlib
            canvas: Canvas de tkinter
            training_results: Resultados del entrenamiento
        """
        if training_results is None:
            return
        
        # Datos
        episode_rewards = training_results['episode_rewards']
        eval_rewards = training_results['eval_rewards']
        eval_interval = training_results['eval_interval']
        grid_accuracies = training_results.get('grid_accuracies', [])
        
        # Crear subplots si hay métricas adicionales
        if grid_accuracies:
            # Crear figura con subplots
            fig.clear()
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)
            
            # Gráfico superior: recompensas
            ax1.plot(episode_rewards, alpha=0.6, label='Episodio', color='blue')
            
            if eval_rewards:
                eval_episodes = np.arange(eval_interval, len(episode_rewards) + 1, eval_interval)
                ax1.plot(eval_episodes, eval_rewards, 'r-', linewidth=2, label='Evaluación')
            
            # Promedio móvil
            window = min(100, len(episode_rewards) // 10)
            if len(episode_rewards) >= window:
                moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 'g-', 
                        linewidth=2, label=f'Promedio móvil ({window})')
            
            ax1.set_title("Recompensas del Entrenamiento")
            ax1.set_ylabel("Recompensa")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico inferior: precisión en grid
            eval_episodes = np.arange(eval_interval, len(episode_rewards) + 1, eval_interval)
            ax2.plot(eval_episodes, grid_accuracies, 'purple', linewidth=2, marker='o')
            ax2.set_title("Precisión en Grid de Evaluación")
            ax2.set_xlabel("Episodio")
            ax2.set_ylabel("Precisión")
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
            
        else:
            # Gráfico simple para HNAF estable
            ax.plot(episode_rewards, alpha=0.6, label='Episodio', color='blue')
            
            if eval_rewards:
                eval_episodes = np.arange(eval_interval, len(episode_rewards) + 1, eval_interval)
                ax.plot(eval_episodes, eval_rewards, 'r-', linewidth=2, label='Evaluación')
            
            # Promedio móvil
            window = min(100, len(episode_rewards) // 10)
            if len(episode_rewards) >= window:
                moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(episode_rewards)), moving_avg, 'g-', 
                        linewidth=2, label=f'Promedio móvil ({window})')
            
            ax.set_title("Resultados del Entrenamiento HNAF")
            ax.set_xlabel("Episodio")
            ax.set_ylabel("Recompensa")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Actualizar canvas
        canvas.draw()
    
    def create_comparison_plot(self, results1, results2, labels=None):
        """
        Crear gráfico de comparación entre dos modelos
        
        Args:
            results1: Resultados del primer modelo
            results2: Resultados del segundo modelo
            labels: Etiquetas para los modelos
            
        Returns:
            fig: Figura de matplotlib
        """
        if labels is None:
            labels = ['Modelo 1', 'Modelo 2']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Gráfico 1: Recompensas de episodios
        axes[0, 0].plot(results1['episode_rewards'], alpha=0.6, label=labels[0], color='blue')
        axes[0, 0].plot(results2['episode_rewards'], alpha=0.6, label=labels[1], color='red')
        axes[0, 0].set_title('Recompensas de Episodios')
        axes[0, 0].set_xlabel('Episodio')
        axes[0, 0].set_ylabel('Recompensa')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gráfico 2: Recompensas de evaluación
        eval_episodes1 = np.arange(50, len(results1['episode_rewards']) + 1, 50)
        eval_episodes2 = np.arange(50, len(results2['episode_rewards']) + 1, 50)
        
        axes[0, 1].plot(eval_episodes1, results1['eval_rewards'], 'b-o', label=labels[0])
        axes[0, 1].plot(eval_episodes2, results2['eval_rewards'], 'r-o', label=labels[1])
        axes[0, 1].set_title('Recompensas de Evaluación')
        axes[0, 1].set_xlabel('Episodio')
        axes[0, 1].set_ylabel('Recompensa')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gráfico 3: Pérdidas
        if results1['losses'] and results2['losses']:
            axes[1, 0].plot(results1['losses'], alpha=0.6, label=labels[0], color='blue')
            axes[1, 0].plot(results2['losses'], alpha=0.6, label=labels[1], color='red')
            axes[1, 0].set_title('Pérdidas de Entrenamiento')
            axes[1, 0].set_xlabel('Actualización')
            axes[1, 0].set_ylabel('Pérdida')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Gráfico 4: Precisión en grid
        if 'grid_accuracies' in results1 and 'grid_accuracies' in results2:
            eval_episodes1 = np.arange(50, len(results1['episode_rewards']) + 1, 50)
            eval_episodes2 = np.arange(50, len(results2['episode_rewards']) + 1, 50)
            
            axes[1, 1].plot(eval_episodes1, results1['grid_accuracies'], 'blue', linewidth=2, marker='o', label=labels[0])
            axes[1, 1].plot(eval_episodes2, results2['grid_accuracies'], 'red', linewidth=2, marker='s', label=labels[1])
            axes[1, 1].set_title('Precisión en Grid')
            axes[1, 1].set_xlabel('Episodio')
            axes[1, 1].set_ylabel('Precisión')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 1)
        else:
            axes[1, 1].text(0.5, 0.5, 'Grid accuracy\nno disponible', 
                            ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Precisión en Grid')
        
        plt.tight_layout()
        return fig
    
    def create_heatmap(self, hnaf_model, grid_size=50):
        """
        Crear mapa de calor de las decisiones del modelo
        
        Args:
            hnaf_model: Modelo HNAF entrenado
            grid_size: Tamaño de la rejilla
            
        Returns:
            fig: Figura de matplotlib
        """
        if not hasattr(hnaf_model, 'evaluate_policy_grid'):
            return None
        
        # Evaluar en grid
        grid_results = hnaf_model.evaluate_policy_grid(grid_size=grid_size)
        
        # Crear figura
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Mapa de calor de selección de modos
        im1 = axes[0].contourf(grid_results['X'], grid_results['Y'], 
                               grid_results['mode_selections'], levels=[-0.5, 0.5, 1.5], 
                               cmap='RdYlBu')
        axes[0].set_title('Selección de Modos')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0])
        
        # Mapa de calor de Q-values
        im2 = axes[1].contourf(grid_results['X'], grid_results['Y'], 
                               grid_results['q_values'], levels=20, cmap='viridis')
        axes[1].set_title('Q-values')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        return fig
    
    def save_plots(self, fig, filename):
        """
        Guardar gráficos en archivo
        
        Args:
            fig: Figura de matplotlib
            filename: Nombre del archivo
        """
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado como: {filename}") 