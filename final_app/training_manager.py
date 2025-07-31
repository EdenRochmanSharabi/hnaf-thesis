#!/usr/bin/env python3
"""
Módulo de entrenamiento para HNAF
Maneja toda la lógica de entrenamiento
"""

import numpy as np
import torch
from src.hnaf_improved import ImprovedHNAF

class TrainingManager:
    """Manager para el entrenamiento del HNAF"""
    
    def __init__(self):
        self.hnaf_model = None
        self.training_results = None
    
    def train_hnaf(self, params):
        """
        Entrenar HNAF con parámetros dados
        
        Args:
            params (dict): Parámetros de entrenamiento
            
        Returns:
            tuple: (hnaf_model, training_results)
        """
        try:
            print("="*60)
            print("INICIANDO ENTRENAMIENTO HNAF MEJORADO")
            print("="*60)
            print("🚀 Usando HNAF MEJORADO con optimizaciones avanzadas:")
            print(f"  - Red: {params['num_layers']} capas de {params['hidden_dim']} unidades")
            print(f"  - ε-greedy decay: {params['initial_epsilon']} -> {params['final_epsilon']}")
            print(f"  - Learning rate: {params['lr']}")
            print(f"  - Prioritized replay con buffer de 10000")
            print(f"  - Normalización de estados y recompensas")
            print(f"  - Reward shaping local")
            print(f"  - Evaluación en grid 100x100")
            print(f"  - Horizonte más largo: {params['max_steps']} pasos")
            
            if 'custom_functions' in params:
                print("✅ Usando funciones personalizadas")
            else:
                print("✅ Usando funciones por defecto")
            print()
            
            # Fijar semilla para reproducibilidad
            np.random.seed(42)
            torch.manual_seed(42)
            
            # Crear modelo HNAF mejorado
            self.hnaf_model = ImprovedHNAF(
                state_dim=params['state_dim'],
                action_dim=params['action_dim'],
                num_modes=params['num_modes'],
                hidden_dim=params['hidden_dim'],
                num_layers=params['num_layers'],
                lr=params['lr'],
                tau=params['tau'],
                gamma=params['gamma']
            )
            
            # Si hay funciones personalizadas, actualizar el modelo
            if 'custom_functions' in params:
                custom_funcs = params['custom_functions']
                self.hnaf_model.transformation_functions = custom_funcs['transformation_functions']
                self.hnaf_model.reward_function = custom_funcs['reward_function']
            
            # Métricas de entrenamiento
            episode_rewards = []
            losses = []
            eval_rewards = []
            grid_accuracies = []
            eval_interval = 50
            
            # Entrenamiento mejorado con ε-greedy decay
            epsilon_decay = (params['initial_epsilon'] - params['final_epsilon']) / params['num_episodes']
            
            for episode in range(params['num_episodes']):
                # Calcular epsilon actual
                epsilon = max(params['final_epsilon'], 
                            params['initial_epsilon'] - episode * epsilon_decay)
                
                # Entrenar episodio
                reward, _ = self.hnaf_model.train_episode(
                    max_steps=params['max_steps'],
                    epsilon=epsilon
                )
                episode_rewards.append(reward)
                
                # Actualizar redes
                loss = self.hnaf_model.update(batch_size=params['batch_size'])
                if loss is not None:
                    losses.append(loss)
                
                # Actualizar redes objetivo
                if hasattr(self.hnaf_model, 'update_target_networks'):
                    self.hnaf_model.update_target_networks()
                
                # Evaluación periódica
                if (episode + 1) % eval_interval == 0:
                    eval_reward, mode_selections = self.hnaf_model.evaluate_policy(num_episodes=10)
                    eval_rewards.append(eval_reward)
                    
                    # Evaluación en grid para HNAF mejorado
                    if hasattr(self.hnaf_model, 'evaluate_policy_grid'):
                        grid_results = self.hnaf_model.evaluate_policy_grid(grid_size=50)
                        grid_accuracies.append(grid_results['optimal_accuracy'])
                        
                        print(f"Episodio {episode+1}/{params['num_episodes']}")
                        print(f"  ε: {epsilon:.3f}")
                        print(f"  Recompensa promedio: {np.mean(episode_rewards[-eval_interval:]):.4f}")
                        print(f"  Recompensa evaluación: {eval_reward:.4f}")
                        print(f"  Precisión grid: {grid_results['optimal_accuracy']:.2%}")
                        print(f"  Selección modos: {mode_selections}")
                        if losses:
                            print(f"  Pérdida promedio: {np.mean(losses[-eval_interval:]):.6f}")
                        print()
                    else:
                        print(f"Episodio {episode+1}/{params['num_episodes']}")
                        print(f"  Recompensa promedio: {np.mean(episode_rewards[-eval_interval:]):.4f}")
                        print(f"  Recompensa evaluación: {eval_reward:.4f}")
                        print(f"  Selección de modos: {mode_selections}")
                        if losses:
                            print(f"  Pérdida promedio: {np.mean(losses[-eval_interval:]):.6f}")
                        print()
            
            # Verificación final
            self.hnaf_model.verify_hnaf()
            
            # Guardar resultados
            self.training_results = {
                'episode_rewards': episode_rewards,
                'losses': losses,
                'eval_rewards': eval_rewards,
                'grid_accuracies': grid_accuracies,
                'eval_interval': eval_interval
            }
            
            return self.hnaf_model, self.training_results
            
        except Exception as e:
            print(f"Error durante el entrenamiento: {str(e)}")
            raise e
    
    def get_training_progress(self):
        """Obtener progreso del entrenamiento"""
        if self.training_results is None:
            return 0, "No iniciado"
        
        total_episodes = len(self.training_results['episode_rewards'])
        if total_episodes == 0:
            return 0, "Iniciando..."
        
        progress = (total_episodes / 1000) * 100  # Asumiendo 1000 episodios
        status = f"Episodio {total_episodes}/1000"
        
        return progress, status 