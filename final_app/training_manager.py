#!/usr/bin/env python3
"""
Módulo de entrenamiento para HNAF
Maneja toda la lógica de entrenamiento
"""

import time
import numpy as np
import torch
from src.hnaf_improved import ImprovedHNAF

class TrainingManager:
    """Manager para el entrenamiento del HNAF"""
    
    def __init__(self):
        self.hnaf_model = None
        self.training_results = None
    
    def _create_gui_reward_function(self, gui_reward_expr, params):
        """Crear función de recompensa desde la expresión del GUI."""
        def gui_reward_function(x, y, x0, y0):
            try:
                # Crear namespace seguro para eval
                safe_dict = {
                    'x': x, 'y': y, 'x0': x0, 'y0': y0,
                    'np': np, 'abs': abs, 'sqrt': np.sqrt, 'log': np.log,
                    '__builtins__': {}
                }
                raw_reward = eval(gui_reward_expr, safe_dict)
                return raw_reward  # Usar la función tal como está definida
            except Exception as e:
                error_msg = f"❌ ERROR CRÍTICO: Función de recompensa GUI falló\n" \
                           f"   Expresión: {gui_reward_expr}\n" \
                           f"   Error: {e}\n" \
                           f"   Estado: x={x}, y={y}, x0={x0}, y0={y0}\n" \
                           f"   ENTRENAMIENTO DETENIDO - Corrige la función de recompensa"
                print(error_msg)
                raise RuntimeError(f"Función de recompensa GUI inválida: {e}")
        return gui_reward_function
    
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
            print(f"  - Prioritized replay con buffer de {params['buffer_capacity']}")
            print(f"  - Alpha (prioridad): {params['alpha']}, Beta (sesgo): {params['beta']}")
            print(f"  - Normalización de recompensas: {'Habilitada' if params['reward_normalize'] else 'Deshabilitada'}")
            print(f"  - Reward shaping: {'Habilitado' if params.get('reward_shaping', False) else 'Deshabilitado'}")
            print(f"  - Evaluación en grid 100x100")
            print(f"  - Horizonte más largo: {params['max_steps']} pasos")
            
            # Si hay funciones personalizadas, actualizar el modelo
            if 'custom_functions' in params:
                custom_funcs = params['custom_functions']
                
                # Crear funciones de transformación dinámicamente
                def create_transform_function(matrix):
                    def transform_func(x, y):
                        A = np.array(matrix)
                        result = A @ np.array([[x], [y]])
                        return float(result[0, 0]), float(result[1, 0])
                    return transform_func
                
                # Crear funciones de transformación para cada modo
                transform_x1 = create_transform_function(custom_funcs['A1'])
                transform_x2 = create_transform_function(custom_funcs['A2'])
                transformation_functions = [transform_x1, transform_x2]
                
                # Crear función de recompensa dinámica
                def custom_reward_function(x, y, x0, y0):
                    try:
                        # Crear namespace seguro para eval
                        safe_dict = {
                            'x': x, 'y': y, 'x0': x0, 'y0': y0,
                            'np': np, 'abs': abs, 'sqrt': np.sqrt,
                            '__builtins__': {}
                        }
                        raw_reward = eval(custom_funcs['reward_expr'], safe_dict)
                        
                        # **CORREGIDO**: Respetar la función tal como la define el usuario
                        return raw_reward  # Usar la función tal como está definida
                    except Exception as e:
                        error_msg = f"❌ ERROR CRÍTICO: Función de recompensa personalizada falló\n" \
                                   f"   Expresión: {custom_funcs['reward_expr']}\n" \
                                   f"   Error: {e}\n" \
                                   f"   ENTRENAMIENTO DETENIDO - Corrige la función de recompensa"
                        print(error_msg)
                        raise RuntimeError(f"Función de recompensa personalizada inválida: {e}")
                
                # Actualizar el modelo con las funciones personalizadas
                self.hnaf_model.transformation_functions = transformation_functions
                self.hnaf_model.reward_function = custom_reward_function
                
                print(f"✅ Funciones personalizadas cargadas:")
                print(f"   - Coordenadas iniciales: ({custom_funcs['x0']}, {custom_funcs['y0']})")
                print(f"   - Matriz A1: {custom_funcs['A1']}")
                print(f"   - Matriz A2: {custom_funcs['A2']}")
                print(f"   - Función de recompensa: {custom_funcs['reward_expr']}")
                print(f"   - Optimización: {params.get('reward_optimization', 'minimizar')}")
                print(f"   - Reward shaping: {'Habilitado' if params.get('reward_shaping', False) else 'Deshabilitado'}")
            else:
                print("✅ Usando funciones por defecto")
                if 'gui_reward_function' in params:
                    print(f"   - Función de recompensa: {params['gui_reward_function']}")
                else:
                    print(f"   - Función de recompensa: No definida")
            print()
            
            # Fijar semilla para reproducibilidad
            np.random.seed(42)
            torch.manual_seed(42)
            
            # Crear modelo HNAF mejorado
            self.hnaf_model = ImprovedHNAF(
                state_dim=int(params['state_dim']),
                action_dim=int(params['action_dim']),
                num_modes=int(params['num_modes']),
                hidden_dim=int(params['hidden_dim']),
                num_layers=int(params['num_layers']),
                lr=float(params['lr']),
                tau=float(params['tau']),
                gamma=float(params['gamma']),
                buffer_capacity=int(params['buffer_capacity']),
                alpha=float(params['alpha']),
                beta=float(params['beta']),
                reward_normalize=bool(params['reward_normalize'])
            )
            
            # **NUEVO**: Configurar reward shaping
            self.hnaf_model.reward_shaping_enabled = params.get('reward_shaping', True)
            
            # Configurar función de recompensa desde GUI
            if 'gui_reward_function' in params:
                gui_reward_function = self._create_gui_reward_function(
                    params['gui_reward_function'], params)
                self.hnaf_model.reward_function = gui_reward_function
            
            # **NUEVO**: SIEMPRE actualizar matrices desde el GUI
            if 'gui_matrices' in params:
                gui_matrices = params['gui_matrices']
                A1_matrix = gui_matrices['A1']
                A2_matrix = gui_matrices['A2']
                self.hnaf_model.update_transformation_matrices(A1_matrix, A2_matrix)
            elif 'custom_functions' in params:
                # Fallback para compatibilidad
                custom_funcs = params['custom_functions']
                A1_matrix = custom_funcs['A1']
                A2_matrix = custom_funcs['A2']
                self.hnaf_model.update_transformation_matrices(A1_matrix, A2_matrix)
            
            # Métricas de entrenamiento
            episode_rewards = []
            losses = []
            eval_rewards = []
            grid_accuracies = []
            eval_interval = 50
            
            # Entrenamiento mejorado con ε-greedy decay
            epsilon_decay = (float(params['initial_epsilon']) - float(params['final_epsilon'])) / int(params['num_episodes'])
            
            for episode in range(int(params['num_episodes'])):
                # Calcular epsilon actual
                epsilon = max(float(params['final_epsilon']), 
                            float(params['initial_epsilon']) - episode * epsilon_decay)
                
                # Entrenar episodio
                reward, _ = self.hnaf_model.train_episode(
                    max_steps=int(params['max_steps']),
                    epsilon=epsilon
                )
                episode_rewards.append(reward)
                
                # Actualizar redes
                loss = self.hnaf_model.update(batch_size=int(params['batch_size']))
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
                        
                        print(f"Episodio {episode+1}/{int(params['num_episodes'])}")
                        print(f"  ε: {epsilon:.3f}")
                        print(f"  Recompensa promedio: {np.mean(episode_rewards[-eval_interval:]):.4f}")
                        print(f"  Recompensa evaluación: {eval_reward:.4f}")
                        print(f"  Precisión grid: {grid_results['optimal_accuracy']:.2%}")
                        print(f"  Selección modos: {mode_selections}")
                        if losses:
                            print(f"  Pérdida promedio: {np.mean(losses[-eval_interval:]):.6f}")
                        print()
                    else:
                        print(f"Episodio {episode+1}/{int(params['num_episodes'])}")
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
            error_msg = f"❌ ERROR CRÍTICO DE ENTRENAMIENTO:\n" \
                       f"   Error: {e}\n" \
                       f"   Tipo: {type(e).__name__}\n" \
                       f"   ENTRENAMIENTO FALLIDO - Revisa configuración"
            print(error_msg)
            raise RuntimeError(f"Entrenamiento falló: {e}")
    
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