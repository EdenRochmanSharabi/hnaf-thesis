#!/usr/bin/env python3
"""
Módulo de visualización para HNAF
Maneja gráficos y visualizaciones
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class VisualizationManager:
    """Manager para visualizaciones del HNAF"""
    
    def __init__(self):
        pass
    
    def update_plots(self, fig, canvas, training_results, show_rewards=True, show_precision=True, show_loss=True):
        """
        Genera y muestra los gráficos seleccionados con un layout dinámico que ocupa todo el espacio.
        """
        if not training_results:
            return

        sns.set_theme(style="whitegrid", palette="viridis")
        fig.clear()

        # Determinar qué gráficos mostrar
        plots_to_show = []
        if show_rewards: plots_to_show.append('rewards')
        if show_precision: plots_to_show.append('precision')
        if show_loss: plots_to_show.append('loss')

        num_plots = len(plots_to_show)
        if num_plots == 0:
            canvas.draw()
            return
        
        # --- Crear layout dinámico que ocupe todo el espacio ---
        if num_plots == 1:
            axes = [fig.add_subplot(1, 1, 1)]
        elif num_plots == 2:
            axes = fig.subplots(1, 2)
            # Convertir a lista si es necesario
            if hasattr(axes, '__iter__') and not isinstance(axes, list):
                axes = list(axes)
            elif not hasattr(axes, '__iter__'):
                axes = [axes]
        else:  # num_plots == 3
            gs = fig.add_gridspec(2, 2)
            ax1 = fig.add_subplot(gs[0, :])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[1, 1])
            axes = [ax1, ax2, ax3]

        # Mapear gráficos a ejes
        plot_axis_map = dict(zip(plots_to_show, axes))

        # --- Gráfico de Recompensas ---
        if 'rewards' in plot_axis_map:
            ax = plot_axis_map['rewards']
            rewards = training_results.get('episode_rewards', [])
            eval_rewards = training_results.get('eval_rewards', [])
            eval_interval = training_results.get('eval_interval', 50)
            
            ax.plot(rewards, color='lightgray', alpha=0.6, label='Recompensa Episodio')
            if len(rewards) >= 100:
                moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
                ax.plot(np.arange(99, len(rewards)), moving_avg, color=sns.color_palette("viridis")[2], linewidth=2, label='Promedio Móvil')
            
            if eval_rewards:
                eval_episodes = np.arange(eval_interval, len(rewards) + 1, eval_interval)
                ax.plot(eval_episodes, eval_rewards, 'o-', color=sns.color_palette("viridis")[0], markersize=5, label='Evaluación')

            ax.set_title("Evolución de la Recompensa", fontsize=14, weight='bold')
            ax.set_xlabel("Episodio", fontsize=12)
            ax.set_ylabel("Recompensa", fontsize=12)
            ax.legend(frameon=True, shadow=True, fancybox=True)

        # --- Gráfico de Precisión ---
        if 'precision' in plot_axis_map:
            ax = plot_axis_map['precision']
            grid_accuracies = training_results.get('grid_accuracies', [])
            if grid_accuracies:
                rewards = training_results.get('episode_rewards', [])
                eval_interval = training_results.get('eval_interval', 50)
                eval_episodes = np.arange(eval_interval, len(rewards) + 1, eval_interval)
                ax.plot(eval_episodes, grid_accuracies, 'o-', color=sns.color_palette("viridis")[3], markersize=5)
                ax.set_ylim(0, 1)
                ax.axhline(y=0.5, color='gray', linestyle='--', label='Azar (50%)')

            ax.set_title("Precisión de Selección de Modo", fontsize=12, weight='bold')
            ax.set_xlabel("Episodio", fontsize=10)
            ax.set_ylabel("Precisión en Grid", fontsize=10)
            ax.legend(frameon=True, shadow=True, fancybox=True)
        
        # --- Gráfico de Pérdida ---
        if 'loss' in plot_axis_map:
            ax = plot_axis_map['loss']
            losses = training_results.get('losses', [])
            if losses:
                loss_ma = np.convolve(losses, np.ones(50)/50, mode='valid') if len(losses) > 50 else losses
                ax.plot(loss_ma, color=sns.color_palette("viridis")[5])
                ax.set_yscale('log')
            
            ax.set_title("Pérdida del Crítico (Log)", fontsize=12, weight='bold')
            ax.set_xlabel("Paso de Actualización", fontsize=10)
            ax.set_ylabel("Pérdida", fontsize=10)

        fig.suptitle("Análisis del Entrenamiento HNAF", fontsize=18, weight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
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