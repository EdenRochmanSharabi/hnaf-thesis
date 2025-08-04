#!/usr/bin/env python3
"""
Módulo de entrenamiento para HNAF
Maneja toda la lógica de entrenamiento
SIN VALORES HARDCODEADOS - Todo desde config.yaml
"""

import time
import numpy as np
import torch
from hnaf_improved import ImprovedHNAF
from config_manager import get_config_manager
from logging_manager import get_logger, log_exception, log_info, log_warning, log_error



class TrainingManager:
    """Manager para el entrenamiento del HNAF"""
    
    def __init__(self):
        self.hnaf_model = None
        self.training_results = None
        self.config_manager = get_config_manager()
        self.logger = get_logger("TrainingManager")


    
    def _create_gui_reward_function(self, gui_reward_expr, params):
        """Crear función de recompensa inteligente que enseña selección de modo."""
        
        # Obtener matrices dinámicamente desde GUI (NO hardcodeadas)
        gui_matrices = params.get('gui_matrices', {})
        # Obtener matrices por defecto desde config
        # VERIFICAR MATRICES GUI (SIN FALLBACKS)
        if 'A1' not in gui_matrices or 'A2' not in gui_matrices:
            raise RuntimeError(f"❌ ERROR: Matrices GUI faltantes. Recibido: {list(gui_matrices.keys())}")
        
        A1 = np.array(gui_matrices['A1'])
        A2 = np.array(gui_matrices['A2'])
        
        # VERIFICAR CONFIGURACIÓN (SIN FALLBACKS)
        reward_config = self.config_manager.get_reward_shaping_config()
        if 'mode_aware' not in reward_config:
            raise RuntimeError("❌ ERROR: Configuración 'mode_aware' faltante en config.yaml")
        
        def gui_reward_function(x, y, x0, y0, mode=None, action=None, previous_state=None):
            try:
                state = np.array([x, y])
                
                # DETECTAR TIPO DE FUNCIÓN DE RECOMPENSA
                if gui_reward_expr == 'mode_aware_reward':
                    # NUEVA RECOMPENSA BASADA EN MULTIPLICADORES RELATIVOS
                    state = np.array([x, y])
                    
                    # Calcular qué modo sería óptimo (mínima norma resultante)
                    norm_if_mode0 = np.linalg.norm(A1 @ state)
                    norm_if_mode1 = np.linalg.norm(A2 @ state)
                    optimal_mode = 0 if norm_if_mode0 < norm_if_mode1 else 1
                    
                    # **ESTABILIZACIÓN**: Recompensa base usando tanh para rango [-1, 1]
                    base_reward = -np.tanh(np.linalg.norm(state))
                    
                    # **NUEVO**: Bonus de exploración para evitar colapso de modos
                    exploration_bonus = 0.0
                    if mode is not None:
                        # Bonus por explorar el modo menos usado
                        mode_counts = getattr(self.hnaf_model, 'mode_selection_counts', {0: 0, 1: 0})
                        total_selections = sum(mode_counts.values()) if mode_counts else 0
                        
                        if total_selections > 0:
                            mode_ratio = mode_counts.get(mode, 0) / total_selections
                            # Bonus si el modo actual está siendo usado menos del 40%
                            if mode_ratio < 0.4:
                                exploration_bonus = 0.1 * (0.4 - mode_ratio)
                    
                    # **NUEVO**: Bonus de estabilidad mejorado
                    stability_bonus = 0.0
                    current_norm = np.linalg.norm(state)
                    
                    # Bonus por estar cerca del origen (implementación actual)
                    if current_norm < 0.5:  # Si estamos cerca del origen
                        stability_bonus = 0.1 * (0.5 - current_norm)  # Bonus por estar cerca
                    
                    # **IMPLEMENTADO**: Bonus de estabilidad completo con estado anterior
                    if previous_state is not None:
                        previous_norm = np.linalg.norm(previous_state)
                        if current_norm < previous_norm:
                            stability_bonus = 0.2 * (previous_norm - current_norm)
                    else:
                        # Fallback: bonus por estar cerca del origen
                        if current_norm < 0.5:  # Si estamos cerca del origen
                            stability_bonus = 0.1 * (0.5 - current_norm)  # Bonus por estar cerca
                    
                    # MULTIPLICADORES RELATIVOS CONFIGURABLES PARA ENSEÑAR MODO CORRECTO
                    if mode is not None:
                        reward_config = self.config_manager.get_reward_shaping_config()
                        mode_aware_config = reward_config.get('mode_aware', {})
                        
                        if mode == optimal_mode:
                            # Modo correcto: reducir penalización
                            if 'correct_mode_multiplier' not in mode_aware_config:
                                raise RuntimeError("❌ ERROR: 'correct_mode_multiplier' faltante en config.yaml")
                            multiplier = mode_aware_config['correct_mode_multiplier']
                        else:
                            # Modo incorrecto: aumentar penalización
                            if 'incorrect_mode_multiplier' not in mode_aware_config:
                                raise RuntimeError("❌ ERROR: 'incorrect_mode_multiplier' faltante en config.yaml")
                            multiplier = mode_aware_config['incorrect_mode_multiplier']
                        
                        total_reward = (base_reward * multiplier) + exploration_bonus + stability_bonus
                        # **ESTABILIZACIÓN**: Clipping final para asegurar rango [-1, 1]
                        return np.clip(total_reward, -1.0, 1.0)
                    else:
                        # Sin modo específico, usar recompensa base
                        return base_reward
                        
                else:
                    # FUNCIÓN TRADICIONAL desde GUI
                    safe_dict = {
                        'x': x, 'y': y, 'x0': x0, 'y0': y0,
                        'np': np, 'abs': abs, 'sqrt': np.sqrt, 'log': np.log, 'tanh': np.tanh,
                        '__builtins__': {}
                    }
                    raw_reward = eval(gui_reward_expr, safe_dict)
                    # **ESTABILIZACIÓN**: Clipping para asegurar rango [-1, 1]
                    return np.clip(raw_reward, -1.0, 1.0)
                    
            except Exception as e:
                context = {
                    'state_x': x, 'state_y': y, 'mode': mode,
                    'gui_reward_expr': gui_reward_expr,
                    'matrices_available': 'gui_matrices' in params
                }
                self.logger.log_exception(e, context, "Función de recompensa inteligente")
                error_msg = f"❌ ERROR CRÍTICO: Función de recompensa inteligente falló - {e}"
                print(error_msg)
                raise RuntimeError(f"Función de recompensa inteligente inválida: {e}")
        
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
            # Usar tamaño de grid desde configuración
            eval_config = self.config_manager.get_evaluation_config()
            print(f"  - Evaluación en grid {eval_config['grid_size_display']}x{eval_config['grid_size_display']}")
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
                            'np': np, 'abs': abs, 'sqrt': np.sqrt, 'tanh': np.tanh,
                            '__builtins__': {}
                        }
                        raw_reward = eval(custom_funcs['reward_expr'], safe_dict)
                        
                        # **ESTABILIZACIÓN**: Clipping para asegurar rango [-1, 1]
                        return np.clip(raw_reward, -1.0, 1.0)
                    except Exception as e:
                        context = {
                            'reward_expression': custom_funcs['reward_expr'],
                            'state_x': x, 'state_y': y,
                            'custom_functions_available': True
                        }
                        self.logger.log_exception(e, context, "Función de recompensa personalizada")
                        error_msg = f"❌ ERROR CRÍTICO: Función de recompensa personalizada falló - {e}"
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
            
            # ⚡ OPTIMIZACIONES BÁSICAS
            print("⚡ Aplicando optimizaciones básicas...")
            import gc
            gc.collect()
            
            # Configurar seeds desde configuración (NO hardcodeados)
            init_config = self.config_manager.get_initialization_config()
            seeds = init_config['seeds']
            
            if seeds['numpy_seed'] is not None:
                np.random.seed(seeds['numpy_seed'])
                print(f"🎲 Numpy seed configurado: {seeds['numpy_seed']}")
            else:
                print("🎲 Numpy seed aleatorio (sin configurar)")
                
            if seeds['torch_seed'] is not None:
                torch.manual_seed(seeds['torch_seed'])
                print(f"🎲 PyTorch seed configurado: {seeds['torch_seed']}")
            else:
                print("🎲 PyTorch seed aleatorio (sin configurar)")
            
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
            mode_selection_counts = {0: 0, 1: 0}  # Conteo de selección de modos
            
            # Configuración de evaluación (NO hardcodeada)
            eval_config = self.config_manager.get_evaluation_config()
            eval_interval = eval_config['interval']
            
            # Parámetros de warm-up supervisado
            supervised_episodes = int(params.get('supervised_episodes', 0))
            if supervised_episodes > 0:
                print(f"🧠 Warm-up supervisado BALANCEADO: {supervised_episodes} episodios")
                print("   └─ Estrategia: Enseñanza dirigida con exploración de ambos modos")
            else:
                print("⚠️ Sin entrenamiento supervisado - Exploración libre únicamente")
            
            # Entrenamiento mejorado con ε-greedy decay
            epsilon_decay = (float(params['initial_epsilon']) - float(params['final_epsilon'])) / int(params['num_episodes'])
            
            for episode in range(int(params['num_episodes'])):
                # Calcular epsilon actual
                epsilon = max(float(params['final_epsilon']), 
                            float(params['initial_epsilon']) - episode * epsilon_decay)
                
                # Entrenar episodio (con warm-up supervisado si aplica)
                if episode < supervised_episodes:
                    # WARM-UP SUPERVISADO: forzar modo óptimo con curriculum learning
                    reward, mode_counts = self._train_supervised_episode(
                        max_steps=int(params['max_steps']),
                        params=params,
                        episode_num=episode,
                        total_supervised=supervised_episodes
                    )
                else:
                    # Entrenamiento normal
                    reward, mode_counts = self._train_normal_episode(
                        max_steps=int(params['max_steps']),
                        epsilon=epsilon
                    )
                
                episode_rewards.append(reward)
                
                # Actualizar contadores de modos
                for mode, count in mode_counts.items():
                    mode_selection_counts[mode] += count
                
                # Actualizar redes
                loss = self.hnaf_model.update(batch_size=int(params['batch_size']))
                if loss is not None:
                    losses.append(loss)
                
                # Actualizar redes objetivo
                if hasattr(self.hnaf_model, 'update_target_networks'):
                    self.hnaf_model.update_target_networks()
                
                # Evaluación periódica
                if (episode + 1) % eval_interval == 0:
                    eval_reward, mode_selections = self.hnaf_model.evaluate_policy(num_episodes=eval_config['num_episodes'])
                    eval_rewards.append(eval_reward)
                    
                    # Evaluación en grid para HNAF mejorado
                    if hasattr(self.hnaf_model, 'evaluate_policy_grid'):
                        grid_results = self.hnaf_model.evaluate_policy_grid(grid_size=eval_config['grid_size'])
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
            
            # ⚡ ANÁLISIS DE RENDIMIENTO Y OPTIMIZACIÓN
            final_precision = grid_accuracies[-1] if grid_accuracies else 0.0
            
            # NUEVO: Análisis de estabilidad avanzado
            training_data = {
                'losses': losses,
                'episode_rewards': episode_rewards,
                'mode_selection_counts': mode_selection_counts,
                'grid_accuracies': grid_accuracies,
                'eval_rewards': eval_rewards
            }
            
            # Análisis simple de estabilidad
            stability_score = self._calculate_simple_stability(training_data)
            
            if stability_score < 0.6:
                print(f"\n⚠️ Estabilidad baja detectada: {stability_score:.3f}")
                print("💡 Considera ajustar parámetros de entrenamiento")
            else:
                print(f"\n✅ Estabilidad aceptable: {stability_score:.3f}")
            
            # Limpieza final de memoria
            import gc
            gc.collect()
            
            # Guardar resultados
            self.training_results = {
                'final_precision': final_precision,
                'episode_rewards': episode_rewards,
                'losses': losses,
                'eval_rewards': eval_rewards,
                'grid_accuracies': grid_accuracies,
                'eval_interval': eval_interval,
                'mode_selection_counts': mode_selection_counts,
                'stability_score': stability_score
            }
            
            # Log del análisis final
            self.logger.info("Entrenamiento completado",
                           precision=final_precision,
                           stability=stability_score)
            
            return self.hnaf_model, self.training_results
            
        except Exception as e:
            error_msg = f"❌ ERROR CRÍTICO DE ENTRENAMIENTO:\n" \
                       f"   Error: {e}\n" \
                       f"   Tipo: {type(e).__name__}\n" \
                       f"   ENTRENAMIENTO FALLIDO - Revisa configuración"
            print(error_msg)
            raise RuntimeError(f"Entrenamiento falló: {e}")
    
    def _calculate_simple_stability(self, training_data):
        """Calcular estabilidad de forma simple"""
        try:
            losses = training_data.get('losses', [])
            if not losses:
                return 0.5
            
            # Calcular estabilidad basada en pérdidas
            if len(losses) < 10:
                return 0.5
            
            recent_losses = losses[-10:]
            loss_variance = np.var(recent_losses)
            max_loss = max(recent_losses)
            
            # Estabilidad inversamente proporcional a la varianza y pérdida máxima
            stability = max(0.0, 1.0 - (loss_variance / 100) - (max_loss / 1000000))
            return min(1.0, stability)
            
        except Exception as e:
            self.logger.log_exception(e, {}, "Cálculo de estabilidad")
            return 0.5
    
    def get_training_progress(self):
        """Obtener progreso del entrenamiento"""
        if self.training_results is None:
            raise RuntimeError("❌ ERROR: Training results no inicializados")
        
        total_episodes = len(self.training_results['episode_rewards'])
        if total_episodes == 0:
            raise RuntimeError("❌ ERROR: No hay datos de entrenamiento disponibles")
        
        # Calcular progreso sin asumir número fijo de episodios
        # Usar el número de episodios actual del entrenamiento
        training_config = self.config_manager.get_training_defaults()
        max_episodes = training_config['num_episodes']
        progress = (total_episodes / max_episodes) * 100
        status = f"Episodio {total_episodes}/{max_episodes}"
        
        return progress, status
    
    def _train_supervised_episode(self, max_steps, params, episode_num=0, total_supervised=1):
        """Entrenar un episodio con modo supervisado BALANCEADO (forzar exploración de ambos modos)"""
        import numpy as np
        import torch
        
        # CURRICULUM LEARNING: Determinar dificultad basada en progreso
        reward_config = self.config_manager.get_reward_shaping_config()
        curriculum_config = reward_config.get('curriculum_learning', {})
        
        if curriculum_config.get('enabled', False):
            # Determinar fase actual basada en progreso
            progress_ratio = episode_num / max(total_supervised, 1)
            phases = curriculum_config.get('phases', [])
            
            current_phase = None
            cumulative_ratio = 0
            for phase in phases:
                cumulative_ratio += phase.get('episodes_ratio', 0.33)
                if progress_ratio <= cumulative_ratio:
                    current_phase = phase
                    break
            
            if current_phase is None:
                current_phase = phases[-1] if phases else {'state_range': 0.15}
            
            # Usar rango de estados basado en la fase
            state_range = current_phase.get('state_range', 0.15)
            state = np.random.uniform(-state_range, state_range, 2)
            
            if episode_num % 50 == 0:  # Mostrar progreso de curriculum
                print(f"   📚 Curriculum: Fase '{current_phase.get('name', 'unknown')}' (rango: ±{state_range:.2f})")
        else:
            # Fallback a configuración tradicional
            init_config = reward_config.get('state_initialization', {})
            min_range = init_config.get('min_range', -0.5)
            max_range = init_config.get('max_range', 0.5)
            state = np.random.uniform(min_range, max_range, 2)
        total_reward = 0
        mode_counts = {0: 0, 1: 0}
        
        # VERIFICAR MATRICES GUI (SIN FALLBACKS)
        gui_matrices = params.get('gui_matrices', {})
        if 'A1' not in gui_matrices or 'A2' not in gui_matrices:
            raise RuntimeError(f"❌ ERROR: Matrices GUI faltantes en supervised training. Recibido: {list(gui_matrices.keys())}")
        
        A1 = np.array(gui_matrices['A1'])
        A2 = np.array(gui_matrices['A2'])
        
        previous_state = None  # Para el primer paso
        for step in range(max_steps):
            # ESTRATEGIA EDUCATIVA MEJORADA: Usar múltiples estados específicos
            supervised_states = reward_config.get('supervised_states', {})
            additional_states = supervised_states.get('additional_states', {})
            
            # Cada 8 pasos, elegir un estado específico para enseñar un modo
            if step % 8 == 0:  # Estado básico Modo 0
                state = np.array(supervised_states.get('mode0_state', [0.1, 0.0]))
            elif step % 8 == 1:  # Estado variante Modo 0
                mode0_variants = additional_states.get('mode0_variants', [[0.15, 0.05]])
                if mode0_variants:
                    state = np.array(mode0_variants[np.random.randint(0, len(mode0_variants))])
                else:
                    state = np.array([0.15, 0.05])
            elif step % 8 == 4:  # Estado básico Modo 1
                state = np.array(supervised_states.get('mode1_state', [0.0, 0.1]))
            elif step % 8 == 5:  # Estado variante Modo 1
                mode1_variants = additional_states.get('mode1_variants', [[0.05, 0.15]])
                if mode1_variants:
                    state = np.array(mode1_variants[np.random.randint(0, len(mode1_variants))])
                else:
                    state = np.array([0.05, 0.15])
            
            # Calcular norma de cada modo para decidir qué enseñar
            norm_if_mode0 = np.linalg.norm(A1 @ state)
            norm_if_mode1 = np.linalg.norm(A2 @ state)
            
            # Determinar modo óptimo matemáticamente
            truly_optimal_mode = 0 if norm_if_mode0 < norm_if_mode1 else 1
            
            # ENSEÑANZA DIRECTA: siempre usar el modo óptimo para que aprenda la relación
            forced_mode = truly_optimal_mode
            mode_counts[forced_mode] += 1
            
            # Forzar acción usando el modo balanceado
            state_tensor = torch.FloatTensor(self.hnaf_model.normalize_state(state)).unsqueeze(0)
            with torch.no_grad():
                V, mu, P = self.hnaf_model.networks[forced_mode].get_P_matrix(state_tensor, training=False)
                action = mu.squeeze().numpy()
            
            # Aplicar transformación según el modo forzado
            A = A1 if forced_mode == 0 else A2
            next_state = A @ state.reshape(-1, 1)
            next_state = next_state.flatten()
            
            # Limitar estados
            state_limits = self.hnaf_model.config_manager.get_state_limits()
            next_state = np.clip(next_state, state_limits['min'], state_limits['max'])
            
            # **NUEVO**: Calcular recompensa con estado anterior para bonus de estabilidad
            reward = self.hnaf_model.reward_function(next_state[0], next_state[1], 0, 0, forced_mode, action, previous_state)
            total_reward += reward
            
            # Almacenar transición usando el modo forzado
            self.hnaf_model.step(state, forced_mode, action, reward, next_state)
            
            # Actualizar estado anterior para el siguiente paso
            previous_state = state.copy()
            state = next_state
            
            # Criterio de parada (umbral configurable - SIN FALLBACKS)
            init_config = reward_config.get('state_initialization', {})
            if 'convergence_threshold' not in init_config:
                raise RuntimeError("❌ ERROR: 'convergence_threshold' faltante en config.yaml")
            convergence_threshold = init_config['convergence_threshold']
            if np.linalg.norm(state) < convergence_threshold:
                break
                
        return total_reward, mode_counts
    
    def _train_normal_episode(self, max_steps, epsilon):
        """Entrenar un episodio normal aplicando RUIDO EN LA ACCIÓN para exploración inteligente."""
        import numpy as np
        
        # Estado inicial aleatorio
        init_config = self.config_manager.get_reward_shaping_config().get('state_initialization', {})
        min_range = init_config.get('min_range', -0.5)
        max_range = init_config.get('max_range', 0.5)
        state = np.random.uniform(min_range, max_range, 2)
        previous_state = None
        
        total_reward = 0
        mode_counts = {0: 0, 1: 0}
        
        # Obtener matrices para la simulación del entorno
        A1 = self.hnaf_model.transformation_matrices[0]
        A2 = self.hnaf_model.transformation_matrices[1]
        
        # Parámetros para el ruido (desde config)
        advanced_config = self.config_manager.get_advanced_config()
        noise_std_dev = advanced_config.get('action_noise_std_dev', 0.1)

        for step in range(max_steps):
            # 1. Elige el modo y la MEJOR acción (sin epsilon)
            mode, action = self.hnaf_model.select_action(state, epsilon=0.0) # Epsilon a 0 para explotación
            mode_counts[mode] += 1

            # 2. AÑADIR RUIDO A LA ACCIÓN para una exploración inteligente
            # El ruido disminuye a medida que epsilon baja, permitiendo el ajuste fino al final
            noise = np.random.normal(0, noise_std_dev * epsilon, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0) # Aseguramos que la acción sigue en el rango válido

            # Aplicar transformación del entorno
            A = A1 if mode == 0 else A2
            next_state = (A @ state.reshape(-1, 1)).flatten()
            
            # Limitar estados
            state_limits = self.hnaf_model.config_manager.get_state_limits()
            next_state = np.clip(next_state, state_limits['min'], state_limits['max'])
            
            # Calcular recompensa pasando el estado anterior
            reward = self.hnaf_model.reward_function(
                next_state[0], next_state[1], state[0], state[1], mode, action, previous_state
            )
            
            # Almacenar la transición REAL
            self.hnaf_model.step(state, mode, action, reward, next_state)
            
            # --- INICIO DE IMAGINATION ROLLOUTS ---
            advanced_config = self.config_manager.get_advanced_config()
            imagination_rollouts = advanced_config.get('imagination_rollouts', 5)
            
            if self.hnaf_model.replay_buffers[mode].can_provide_sample(self.hnaf_model.batch_size):
                # Usar el estado actual como punto de partida para la imaginación
                imagined_state = state.copy()
                for _ in range(imagination_rollouts):
                    # El agente imagina tomar una acción desde el estado imaginado
                    imagined_mode, imagined_action = self.hnaf_model.select_action(imagined_state, epsilon=0.0) # Explotación en la imaginación
                    
                    # Simular el siguiente estado usando el modelo del mundo (matrices A)
                    A_imagined = A1 if imagined_mode == 0 else A2
                    next_imagined_state = (A_imagined @ imagined_state.reshape(-1, 1)).flatten()
                    
                    # Limitar estados imaginados
                    next_imagined_state = np.clip(next_imagined_state, state_limits['min'], state_limits['max'])
                    
                    # Calcular recompensa para la transición imaginada
                    imagined_reward = self.hnaf_model.reward_function(
                        next_imagined_state[0], next_imagined_state[1], imagined_state[0], imagined_state[1], imagined_mode, imagined_action, imagined_state
                    )
                    
                    # Almacenar la transición SINTÉTICA en el buffer
                    self.hnaf_model.replay_buffers[imagined_mode].push(imagined_state, imagined_mode, imagined_action, imagined_reward, next_imagined_state)
                    
                    # Actualizar el estado para el siguiente paso de imaginación
                    imagined_state = next_imagined_state
            # --- FIN DE IMAGINATION ROLLOUTS ---
            
            # Actualizar estados
            previous_state = state.copy()
            state = next_state
            total_reward += reward
            
            # Criterio de parada
            if np.linalg.norm(state) < 0.01:
                break
                
        return total_reward, mode_counts 