#!/usr/bin/env python3
"""
Módulo de entrenamiento para HNAF
Maneja toda la lógica de entrenamiento
SIN VALORES HARDCODEADOS - Todo desde config.yaml
"""

import time
import numpy as np
import torch
from hnaf_improved import HNAFImproved
from config_manager import get_config_manager
from logging_manager import get_logger, log_exception, log_info, log_warning, log_error
from plot_utils import plot_successful_trajectories, plot_3d_trajectories
from noise_process import OUNoise
from detailed_logger import DetailedLogger



class TrainingManager:
    """Manager para el entrenamiento del HNAF"""
    
    def __init__(self):
        self.hnaf_model = None
        self.training_results = None
        self.config_manager = get_config_manager()
        self.logger = get_logger("TrainingManager")
        
        # AÑADIR: Contadores para la entropía
        self.mode_selection_history = [] 
        self.entropy_window_size = 100  # Miraremos los últimos 100 pasos para calcular la entropía
        
        # AÑADIR: Inicializar el proceso de ruido OU (se configurará cuando se cree el modelo)
        self.noise = None
        
        # AÑADIR: Logger detallado para análisis completo
        self.detailed_logger = None


    
    def _create_gui_reward_function(self, gui_reward_expr, params):
        """Crear función de recompensa inteligente que enseña selección de modo."""
        
        # Obtener matrices dinámicamente desde GUI (NO hardcodeadas)
        gui_matrices = params.get('gui_matrices', {})
        # Obtener matrices por defecto desde config
        # VERIFICAR MATRICES GUI (SIN FALLBACKS)
        if 'A1' not in gui_matrices or 'A2' not in gui_matrices:
            raise RuntimeError(f"❌ ERROR: Matrices GUI faltantes. Recibido: {list(gui_matrices.keys())}")
        
        # **NUEVO**: Soporte dinámico para múltiples matrices
        matrices = {}
        for key, matrix in gui_matrices.items():
            if key.startswith('A'):
                matrices[key] = np.array(matrix)
        
        # Verificar que tenemos al menos 2 matrices
        if len(matrices) < 2:
            raise RuntimeError(f"❌ ERROR: Se requieren al menos 2 matrices. Encontradas: {list(matrices.keys())}")
        
        A1 = matrices['A1']
        A2 = matrices['A2']
        A_list = [matrices[k] for k in sorted(matrices.keys())]
        
        # VERIFICAR CONFIGURACIÓN (SIN FALLBACKS)
        reward_config = self.config_manager.get_reward_shaping_config()
        if 'mode_aware' not in reward_config:
            raise RuntimeError("❌ ERROR: Configuración 'mode_aware' faltante en config.yaml")
        
        def gui_reward_function(x, y, x0, y0, mode=None, action=None, previous_state=None):
            try:
                # Construir vector de estado de dimensión completa
                sd = int(self.hnaf_model.state_dim) if hasattr(self, 'hnaf_model') and self.hnaf_model else 2
                state = np.zeros(sd)
                state[0] = x
                if sd > 1:
                    state[1] = y
                
                # DETECTAR TIPO DE FUNCIÓN DE RECOMPENSA
                if gui_reward_expr == 'mode_aware_reward':
                    # NUEVA RECOMPENSA BASADA EN MULTIPLICADORES RELATIVOS (usar estado completo)
                    
                    # Calcular qué modo sería óptimo (mínima norma resultante)
                    # **NUEVO**: Soporte dinámico para múltiples modos
                    norms = []
                    for i, (key, matrix) in enumerate(matrices.items()):
                        norm = np.linalg.norm(matrix @ state)
                        norms.append((i, norm))
                    
                    optimal_mode = min(norms, key=lambda x: x[1])[0]
                    
                    # **ESTABILIZACIÓN**: Recompensa base usando tanh para rango [-1, 1]
                    base_reward = -np.tanh(np.linalg.norm(state))
                    
                    # **MEJORADO**: Bonus de exploración más agresivo para evitar colapso de modos
                    exploration_bonus = 0.0
                    if mode is not None:
                        # Bonus por explorar el modo menos usado
                        mode_counts = getattr(self.hnaf_model, 'mode_selection_counts', {i: 0 for i in range(self.hnaf_model.num_modes)})
                        total_selections = sum(mode_counts.values()) if mode_counts else 0
                        
                        if total_selections > 0:
                            mode_ratio = mode_counts.get(mode, 0) / total_selections
                            # Bonus más agresivo si el modo actual está siendo usado menos del 50%
                            if mode_ratio < 0.5:
                                exploration_bonus = 0.3 * (0.5 - mode_ratio)  # Bonus más fuerte
                            # Penalización si un modo está siendo usado más del 80%
                            elif mode_ratio > 0.8:
                                exploration_bonus = -0.2 * (mode_ratio - 0.8)  # Penalización
                    
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
                        
                elif gui_reward_expr == 'entropy_bonus_reward':
                    # NUEVA: Recompensa con bonificación por entropía
                    return self.entropy_bonus_reward(x, y, x0, y0, mode, action, previous_state)
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
    
    def unit_circle_reward(self, x, y, x0, y0, mode=None, action=None, previous_state=None):
        """
        Recompensa al agente por acercarse y mantenerse en la circunferencia unidad.
        """
        state_norm = np.linalg.norm([x, y])
        
        # El error es la distancia a la circunferencia unidad (radio=1)
        error = abs(state_norm - 1.0)
        
        # La recompensa es mayor cuanto menor es el error.
        # Usamos una exponencial negativa para una penalización suave.
        reward = -np.tanh(error)
        
        # Bonus de estabilidad si se está acercando a la circunferencia
        if previous_state is not None:
            previous_norm = np.linalg.norm(previous_state)
            if abs(state_norm - 1.0) < abs(previous_norm - 1.0):
                reward += 0.1  # Bonus por acercarse a la circunferencia
        
        return np.clip(reward, -1.0, 1.0)
    
    def entropy_bonus_reward(self, x, y, x0, y0, mode=None, action=None, previous_state=None):
        """
        Recompensa que combina la estabilización con un bonus por exploración (entropía).
        """
        # 1. Recompensa base por estabilidad (lógica actual)
        base_reward = -np.tanh(np.linalg.norm([x, y]) * 0.1)

        # 2. Bonificación por Entropía
        entropy_bonus = 0.0
        if mode is not None and self.hnaf_model is not None:
            # Añadir la selección actual al historial
            self.mode_selection_history.append(mode)
            
            # Mantener el historial con un tamaño de ventana fijo
            if len(self.mode_selection_history) > self.entropy_window_size:
                self.mode_selection_history.pop(0)

            # Calcular la distribución de modos en la ventana reciente
            if len(self.mode_selection_history) >= 10:  # Mínimo 10 muestras para calcular entropía
                counts = np.bincount(self.mode_selection_history, minlength=self.hnaf_model.num_modes)
                probabilities = counts / len(self.mode_selection_history)
                
                # Penalizar las probabilidades muy bajas (cercanas a cero)
                # Si una probabilidad es 0, el agente no lo está explorando.
                if np.any(probabilities < 0.1):  # Si algún modo se usa menos del 10% de las veces
                    # Dar un pequeño bonus por elegir el modo menos frecuente
                    least_frequent_mode = np.argmin(probabilities)
                    if mode == least_frequent_mode:
                        entropy_bonus = 0.2  # Recompensa por ser "curioso"
                        
                        # Bonus adicional si la probabilidad es muy baja
                        if probabilities[mode] < 0.05:  # Menos del 5%
                            entropy_bonus += 0.1  # Bonus extra por exploración rara
        
        # La recompensa total es la suma de la estabilidad y la curiosidad
        return np.clip(base_reward + entropy_bonus, -1.0, 1.0)
    
    def train_hnaf(self, params):
        """
        Entrenar HNAF con parámetros dados
        
        Args:
            params (dict): Parámetros de entrenamiento
            
        Returns:
            tuple: (hnaf_model, training_results)
        """
        try:
            # DEBUG: Verificar parámetros recibidos
            print(f"🔍 DEBUG - Parámetros recibidos:")
            print(f"   Keys en params: {list(params.keys())}")
            if 'gui_matrices' in params:
                print(f"   gui_matrices keys: {list(params['gui_matrices'].keys())}")
                print(f"   A1 from gui_matrices: {params['gui_matrices']['A1']}")
                print(f"   A2 from gui_matrices: {params['gui_matrices']['A2']}")
            if 'custom_functions' in params:
                print(f"   custom_functions keys: {list(params['custom_functions'].keys())}")
                print(f"   A1 from custom_functions: {params['custom_functions']['A1']}")
                print(f"   A2 from custom_functions: {params['custom_functions']['A2']}")
            
            # Inicializar logger detallado
            self.detailed_logger = DetailedLogger()
            
            self.logger.info("="*60)
            self.logger.info("INICIANDO ENTRENAMIENTO HNAF MEJORADO")
            self.logger.info("="*60)
            self.logger.info("🚀 Usando HNAF MEJORADO con optimizaciones avanzadas:")
            self.logger.info(f"  - Red: {params['num_layers']} capas de {params['hidden_dim']} unidades")
            self.logger.info(f"  - ε-greedy decay: {params['initial_epsilon']} -> {params['final_epsilon']}")
            self.logger.info(f"  - Learning rate: {params['lr']}")
            self.logger.info(f"  - Prioritized replay con buffer de {params['buffer_capacity']}")
            self.logger.info(f"  - Alpha (prioridad): {params['alpha']}, Beta (sesgo): {params['beta']}")
            self.logger.info(f"  - Normalización de recompensas: {'Habilitada' if params['reward_normalize'] else 'Deshabilitada'}")
            self.logger.info(f"  - Reward shaping: {'Habilitado' if params.get('reward_shaping', False) else 'Deshabilitado'}")
            # Usar tamaño de grid desde configuración
            eval_config = self.config_manager.get_evaluation_config()
            self.logger.info(f"  - Evaluación en grid {eval_config['grid_size_display']}x{eval_config['grid_size_display']}")
            self.logger.info(f"  - Horizonte más largo: {params['max_steps']} pasos")
            
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
                # Construir funciones para todas las matrices Ax presentes
                transformation_functions = []
                for key in sorted([k for k in custom_funcs.keys() if str(k).startswith('A')]):
                    transformation_functions.append(create_transform_function(custom_funcs[key]))
                
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
                
                # **NUEVO**: Actualizar matrices de transformación con todas las disponibles
                # Pasar todas las matrices al modelo (2 o más)
                matrix_list = [matrices[key] for key in sorted(matrices.keys())]
                self.hnaf_model.update_transformation_matrices(*matrix_list)
                
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
            self.logger.info("Inicializando modelo HNAFImproved...")
            try:
                self.hnaf_model = HNAFImproved(
                    config=self.config_manager.config,
                    logger=self.logger
                )
                # Establecer config_manager en el modelo
                self.hnaf_model.config_manager = self.config_manager
                self.logger.info("Modelo HNAFImproved inicializado exitosamente")
            except Exception as e:
                self.logger.error(f"ERROR al inicializar HNAFImproved: {e}")
                raise
            
            # AÑADIR: Inicializar el proceso de ruido OU
            self.logger.info("Inicializando ruido OU...")
            try:
                action_dim = int(params['action_dim'])
                self.logger.info(f"Action dim: {action_dim}")
                self.noise = OUNoise(size=action_dim, seed=42)
                self.logger.info("Ruido OU inicializado exitosamente")
            except Exception as e:
                self.logger.error(f"ERROR al inicializar ruido OU: {e}")
                raise
            
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
                # DEBUG: Verificar matrices antes de actualizar
                print(f"🔍 DEBUG - Matrices desde training_manager:")
                print(f"   A1_matrix: {gui_matrices['A1']}")
                print(f"   A2_matrix: {gui_matrices['A2']}")
                print(f"   A1_matrix shape: {len(gui_matrices['A1'])}x{len(gui_matrices['A1'][0])}")
                print(f"   A2_matrix shape: {len(gui_matrices['A2'])}x{len(gui_matrices['A2'][0])}")

                # Pasar todas las matrices disponibles (A1, A2, A3, ...)
                matrix_list = [gui_matrices[key] for key in sorted(gui_matrices.keys()) if str(key).startswith('A')]
                self.hnaf_model.update_transformation_matrices(*matrix_list)
            elif 'custom_functions' in params:
                # Fallback para compatibilidad
                custom_funcs = params['custom_functions']
                A1_matrix = custom_funcs['A1']
                A2_matrix = custom_funcs['A2']
                # También incluir A3 si existe
                matrix_list = [A1_matrix, A2_matrix]
                if 'A3' in custom_funcs:
                    matrix_list.append(custom_funcs['A3'])
                self.hnaf_model.update_transformation_matrices(*matrix_list)
            
            # Métricas de entrenamiento
            episode_rewards = []
            losses = []
            eval_rewards = []
            grid_accuracies = []
            mode_selection_counts = {i: 0 for i in range(self.hnaf_model.num_modes)}  # Conteo por modo dinámico
            
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
            self.logger.info(f"Epsilon decay: {epsilon_decay}")
            
            for episode in range(int(params['num_episodes'])):
                # Iniciar logging detallado del episodio
                episode_type = "supervised" if episode < supervised_episodes else "normal"
                episode_idx = self.detailed_logger.log_episode_start(
                    episode + 1, int(params['num_episodes']), episode_type
                )
                
                self.logger.info(f"=== EPISODIO {episode + 1}/{int(params['num_episodes'])} ===")
                
                # Calcular epsilon actual
                epsilon = max(float(params['final_epsilon']), 
                            float(params['initial_epsilon']) - episode * epsilon_decay)
                self.logger.info(f"Epsilon actual: {epsilon}")
                
                # Entrenar episodio (con warm-up supervisado si aplica)
                if episode < supervised_episodes:
                    self.logger.info(f"Entrenamiento SUPERVISADO (episodio {episode + 1}/{supervised_episodes})")
                    # WARM-UP SUPERVISADO: forzar modo óptimo con curriculum learning
                    try:
                        reward, mode_counts = self._train_supervised_episode(
                            max_steps=int(params['max_steps']),
                            params=params,
                            episode_num=episode,
                            total_supervised=supervised_episodes,
                            episode_idx=episode_idx
                        )
                        self.logger.info(f"Episodio supervisado completado - Reward: {reward}, Mode counts: {mode_counts}")
                    except Exception as e:
                        self.logger.error(f"ERROR en episodio supervisado {episode + 1}: {e}")
                        raise
                else:
                    self.logger.info("Entrenamiento NORMAL")
                    # Entrenamiento normal
                    try:
                        reward, mode_counts = self._train_normal_episode(
                            max_steps=int(params['max_steps']),
                            epsilon=epsilon,
                            episode_idx=episode_idx
                        )
                        self.logger.info(f"Episodio normal completado - Reward: {reward}, Mode counts: {mode_counts}")
                    except Exception as e:
                        self.logger.error(f"ERROR en episodio normal {episode + 1}: {e}")
                        raise
                
                self.logger.info(f"Reward del episodio: {reward} (tipo: {type(reward)})")
                episode_rewards.append(reward)
                
                # Actualizar contadores de modos
                for mode, count in mode_counts.items():
                    mode_selection_counts[mode] += count
                self.logger.info(f"Mode selection counts actualizados: {mode_selection_counts}")
                
                # Actualizar redes
                self.logger.info("Actualizando redes...")
                try:
                    loss = self.hnaf_model.update(batch_size=int(params['batch_size']))
                    self.logger.info(f"Loss del update: {loss} (tipo: {type(loss)})")
                    if loss is not None:
                        losses.append(loss)
                        self.logger.info(f"Loss añadido a lista. Total losses: {len(losses)}")
                except Exception as e:
                    self.logger.error(f"ERROR en update de redes: {e}")
                    raise
                
                # Actualizar redes objetivo
                if hasattr(self.hnaf_model, 'update_target_networks'):
                    self.logger.info("Actualizando redes objetivo...")
                    self.hnaf_model.update_target_networks()
                
                # Evaluación periódica (menos frecuente para acelerar)
                if (episode + 1) % (eval_interval * 2) == 0:  # Evaluar cada 100 episodios en lugar de 50
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
            
            # Marcar evaluación final para generar gráficos
            self.hnaf_model._is_final_evaluation = True
            
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
            
            if stability_score is not None and stability_score < 0.6:
                print(f"\n⚠️ Estabilidad baja detectada: {stability_score:.3f}")
                print("💡 Considera ajustar parámetros de entrenamiento")
            elif stability_score is not None:
                print(f"\n✅ Estabilidad aceptable: {stability_score:.3f}")
            else:
                print(f"\n⚠️ No se pudo calcular estabilidad")
            
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
            
            # Finalizar logging detallado
            final_results = {
                'final_precision': final_precision,
                'stability_score': stability_score,
                'total_episodes': int(params['num_episodes']),
                'supervised_episodes': supervised_episodes
            }
            self.detailed_logger.log_training_end(final_results)
            
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
            self.logger.info("Calculando estabilidad simple...")
            losses = training_data.get('losses', [])
            self.logger.info(f"Losses disponibles: {len(losses)}")
            
            if not losses:
                self.logger.info("No hay losses, devolviendo 0.5")
                return 0.5
            
            # Calcular estabilidad basada en pérdidas
            if len(losses) < 10:
                self.logger.info(f"Pocas losses ({len(losses)}), devolviendo 0.5")
                return 0.5
            
            recent_losses = losses[-10:]
            loss_variance = np.var(recent_losses)
            max_loss = max(recent_losses)
            
            self.logger.info(f"Recent losses: {recent_losses}")
            self.logger.info(f"Loss variance: {loss_variance}")
            self.logger.info(f"Max loss: {max_loss}")
            
            # Estabilidad inversamente proporcional a la varianza y pérdida máxima
            stability = max(0.0, 1.0 - (loss_variance / 100) - (max_loss / 1000000))
            stability = min(1.0, stability)
            
            self.logger.info(f"Stability calculada: {stability}")
            return stability
            
        except Exception as e:
            self.logger.error(f"ERROR en cálculo de estabilidad: {e}")
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
    
    def _train_supervised_episode(self, max_steps, params, episode_num=0, total_supervised=1, episode_idx=None):
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
                if progress_ratio is not None and progress_ratio <= cumulative_ratio:
                    current_phase = phase
                    break
            
            if current_phase is None:
                current_phase = phases[-1] if phases else {'state_range': 0.15}
            
            # Usar rango de estados basado en la fase
            state_range = current_phase.get('state_range', 0.15)
            state = np.random.uniform(-state_range, state_range, self.hnaf_model.state_dim)
            
            if episode_num % 50 == 0:  # Mostrar progreso de curriculum
                print(f"   📚 Curriculum: Fase '{current_phase.get('name', 'unknown')}' (rango: ±{state_range:.2f})")
        else:
            # Fallback a configuración tradicional
            init_config = reward_config.get('state_initialization', {})
            min_range = init_config.get('min_range', -0.5)
            max_range = init_config.get('max_range', 0.5)
            state = np.random.uniform(min_range, max_range, self.hnaf_model.state_dim)
        total_reward = 0
        mode_counts = {i: 0 for i in range(self.hnaf_model.num_modes)}
        
        # VERIFICAR MATRICES GUI (SIN FALLBACKS)
        gui_matrices = params.get('gui_matrices', {})
        if 'A1' not in gui_matrices or 'A2' not in gui_matrices:
            raise RuntimeError(f"❌ ERROR: Matrices GUI faltantes en supervised training. Recibido: {list(gui_matrices.keys())}")
        
        A1 = np.array(gui_matrices['A1'])
        A2 = np.array(gui_matrices['A2'])
        # Usar todas las matrices disponibles en el modelo (A1, A2, A3, ...)
        A_list = self.hnaf_model.transformation_matrices
        
        previous_state = None  # Para el primer paso
        for step in range(max_steps):
            # ESTRATEGIA EDUCATIVA MEJORADA: Usar múltiples estados específicos
            supervised_states = reward_config.get('supervised_states', {})
            additional_states = supervised_states.get('additional_states', {})
            
            # Cada 8 pasos, elegir un estado específico para enseñar un modo
            if step % 8 == 0:  # Estado básico Modo 0
                state = np.array(supervised_states.get('mode0_state', [0.1, 0.0, 0.0]))
                if state.shape[0] < self.hnaf_model.state_dim:
                    state = np.pad(state, (0, self.hnaf_model.state_dim - state.shape[0]))
                else:
                    state = state[: self.hnaf_model.state_dim]
            elif step % 8 == 1:  # Estado variante Modo 0
                mode0_variants = additional_states.get('mode0_variants', [[0.15, 0.05, 0.0]])
                if mode0_variants:
                    state = np.array(mode0_variants[np.random.randint(0, len(mode0_variants))])
                else:
                    state = np.array([0.15, 0.05, 0.0])
                if state.shape[0] < self.hnaf_model.state_dim:
                    state = np.pad(state, (0, self.hnaf_model.state_dim - state.shape[0]))
                else:
                    state = state[: self.hnaf_model.state_dim]
            elif step % 8 == 4:  # Estado básico Modo 1
                state = np.array(supervised_states.get('mode1_state', [0.0, 0.1, 0.0]))
                if state.shape[0] < self.hnaf_model.state_dim:
                    state = np.pad(state, (0, self.hnaf_model.state_dim - state.shape[0]))
                else:
                    state = state[: self.hnaf_model.state_dim]
            elif step % 8 == 5:  # Estado variante Modo 1
                mode1_variants = additional_states.get('mode1_variants', [[0.05, 0.15, 0.0]])
                if mode1_variants:
                    state = np.array(mode1_variants[np.random.randint(0, len(mode1_variants))])
                else:
                    state = np.array([0.05, 0.15, 0.0])
                if state.shape[0] < self.hnaf_model.state_dim:
                    state = np.pad(state, (0, self.hnaf_model.state_dim - state.shape[0]))
                else:
                    state = state[: self.hnaf_model.state_dim]
            
            # Calcular norma de cada modo para decidir qué enseñar (dinámico)
            norms = []
            for idx, A in enumerate(A_list):
                try:
                    norms.append((idx, np.linalg.norm(A @ state)))
                except Exception:
                    # Aceptar reshape si fuese necesario
                    norms.append((idx, np.linalg.norm((A @ state.reshape(-1, 1)).flatten())))
            # Determinar modo óptimo matemáticamente
            truly_optimal_mode = min(norms, key=lambda x: x[1])[0]
            
            # ENSEÑANZA DIRECTA: siempre usar el modo óptimo para que aprenda la relación
            forced_mode = truly_optimal_mode
            mode_counts[forced_mode] += 1
            
            # Forzar acción usando el modo balanceado
            mode, action = self.hnaf_model.select_action(state)
            
            # Aplicar transformación según el modo forzado
            A = A_list[forced_mode]
            next_state = A @ state.reshape(-1, 1)
            next_state = next_state.flatten()
            
            # Limitar estados
            state_limits = self.hnaf_model.config_manager.get_state_limits()
            next_state = np.clip(next_state, state_limits['min'], state_limits['max'])
            
            # **NUEVO**: Calcular recompensa con estado anterior para bonus de estabilidad
            # Pasar solo x,y para compatibilidad; z se ignora en recompensa actual
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
    
    def _train_normal_episode(self, max_steps, epsilon, episode_idx=None):
        """Entrenar un episodio normal con EXPLORACIÓN INTELIGENTE CON RUIDO OU."""
        import numpy as np
        
        self.logger.info("=== INICIANDO EPISODIO NORMAL ===")
        self.logger.info(f"max_steps: {max_steps}, epsilon: {epsilon}")
        
        # Reiniciar el ruido OU al inicio de cada episodio
        if self.noise is not None:
            self.noise.reset()
            self.logger.info("Ruido OU reiniciado")
        else:
            self.logger.info("No hay ruido OU configurado")
        
        # Estado inicial aleatorio
        init_config = self.config_manager.get_reward_shaping_config().get('state_initialization', {})
        min_range = init_config.get('min_range', -0.5)
        max_range = init_config.get('max_range', 0.5)
        state = np.random.uniform(min_range, max_range, self.hnaf_model.state_dim)
        previous_state = None
        self.logger.info(f"Estado inicial: {state} (rango: [{min_range}, {max_range}])")
        
        total_reward = 0
        mode_counts = {i: 0 for i in range(self.hnaf_model.num_modes)}
        
        # Obtener matrices para la simulación del entorno (todas)
        A_list = self.hnaf_model.transformation_matrices
        try:
            self.logger.info(f"Matrices formas: {[A.shape for A in A_list]}")
        except Exception:
            pass
        
        # --- EXPLORACIÓN INTELIGENTE CON RUIDO OU ---
        # Contar episodios para alternar modos
        episode_count = getattr(self, '_episode_count', 0)
        self._episode_count = episode_count + 1
        
        # Forzar exploración de modos alternados en episodios tempranos
        force_mode_exploration = epsilon > 0.3  # Solo en episodios tempranos
        self.logger.info(f"Episode count: {self._episode_count}, Force mode exploration: {force_mode_exploration}")

        for step in range(max_steps):
            self.logger.info(f"--- PASO {step + 1}/{max_steps} ---")
            self.logger.info(f"Estado actual: {state}")
            
            # 1. SELECCIÓN DE MODO Y ACCIÓN ÓPTIMA (sin epsilon)
            if force_mode_exploration and step % 10 == 0:  # Cada 10 pasos
                # Alternar entre modos para forzar exploración
                mode = step // 10 % self.hnaf_model.num_modes  # Alternar entre todos los modos
                self.logger.info(f"Forzando modo alternado: {mode}")
                _, action = self.hnaf_model.select_action(state)
                selection_reason = "forced_alternation"
            else:
                # Selección normal SIN epsilon (la exploración viene del ruido OU)
                self.logger.info("Selección normal de modo")
                mode, action = self.hnaf_model.select_action(state)
                selection_reason = "normal_selection"
            
            self.logger.info(f"Modo seleccionado: {mode}, Acción base: {action}")
            mode_counts[mode] += 1
            
            # Logging detallado de selección de modo
            if episode_idx is not None and self.detailed_logger:
                # Obtener Q-values para todos los modos (usando eval mode)
                all_q_values = []
                self.hnaf_model.model.eval()
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.hnaf_model.device)
                    all_head_outputs = self.hnaf_model.model(state_tensor)
                    for mode_idx, head_output in enumerate(all_head_outputs):
                        action_candidate = head_output[:, 1:(1 + self.hnaf_model.action_dim)]
                        q_value = self.hnaf_model._get_q_value_from_output(head_output, action_candidate)[0]
                        all_q_values.append(float(q_value))
                self.hnaf_model.model.train()
                
                self.detailed_logger.log_mode_selection(
                    episode_idx, step + 1, all_q_values, mode, selection_reason
                )

            # 2. AÑADIR RUIDO OU A LA ACCIÓN para exploración inteligente
            if self.noise is not None:
                noise_sample = self.noise.sample()
                # La magnitud del ruido disminuye a medida que epsilon baja
                action_with_noise = action + epsilon * noise_sample
                action_with_noise = np.clip(action_with_noise, -1.0, 1.0)  # Asegurar que la acción es válida
                self.logger.info(f"Ruido OU añadido: {noise_sample}, Acción final: {action_with_noise}")
            else:
                # Fallback si no hay ruido OU configurado
                action_with_noise = action
                self.logger.info("Sin ruido OU, usando acción base")

            # Aplicar transformación del entorno
            A = A_list[mode]
            next_state = (A @ state.reshape(-1, 1)).flatten()
            self.logger.info(f"Matriz aplicada: A{mode+1}, Estado siguiente (antes de clip): {next_state}")
            
            # Limitar estados
            state_limits = self.hnaf_model.config_manager.get_state_limits()
            next_state = np.clip(next_state, state_limits['min'], state_limits['max'])
            self.logger.info(f"Límites de estado: {state_limits}, Estado siguiente (después de clip): {next_state}")
            
            # Calcular recompensa pasando el estado anterior
            self.logger.info("Calculando recompensa...")
            try:
                # Pasar solo x,y para compatibilidad; z se ignora en recompensa actual
                reward = self.hnaf_model.reward_function(
                    next_state[0], next_state[1], state[0], state[1], mode, action_with_noise, previous_state
                )
                self.logger.info(f"Recompensa calculada: {reward} (tipo: {type(reward)})")
                
                # Verificar que reward no sea None
                if reward is None:
                    self.logger.error(f"ERROR: Recompensa es None en paso {step + 1}")
                    raise ValueError(f"Recompensa es None en paso {step + 1}")
                    
            except Exception as e:
                self.logger.error(f"ERROR al calcular recompensa: {e}")
                raise
            
            # Almacenar la transición REAL
            self.logger.info("Guardando transición en replay buffer...")
            try:
                loss = self.hnaf_model.step(state, mode, action_with_noise, reward, next_state)
                self.logger.info(f"Loss del step: {loss} (tipo: {type(loss)})")
            except Exception as e:
                self.logger.error(f"ERROR en hnaf_model.step: {e}")
                raise
            
            # --- INICIO DE IMAGINATION ROLLOUTS ---
            advanced_config = self.config_manager.get_advanced_config()
            imagination_rollouts = advanced_config.get('imagination_rollouts', 5)
            
            if len(self.hnaf_model.replay_buffer) >= self.hnaf_model.batch_size:
                self.logger.info(f"Iniciando imagination rollouts ({imagination_rollouts} rollouts)")
                # Usar el estado actual como punto de partida para la imaginación
                imagined_state = state.copy()
                for rollout_idx in range(imagination_rollouts):
                    # El agente imagina tomar una acción desde el estado imaginado
                    imagined_mode, imagined_action = self.hnaf_model.select_action(imagined_state) # Explotación en la imaginación
                    
                    # Simular el siguiente estado usando el modelo del mundo (matrices A)
                    A_imagined = A_list[imagined_mode]
                    next_imagined_state = (A_imagined @ imagined_state.reshape(-1, 1)).flatten()
                    
                    # Limitar estados imaginados
                    next_imagined_state = np.clip(next_imagined_state, state_limits['min'], state_limits['max'])
                    
                    # Calcular recompensa para la transición imaginada
                    imagined_reward = self.hnaf_model.reward_function(
                        next_imagined_state[0], next_imagined_state[1], imagined_state[0], imagined_state[1], imagined_mode, imagined_action, imagined_state
                    )
                    
                    # Almacenar la transición SINTÉTICA en el buffer
                    self.hnaf_model.replay_buffer.append((imagined_state, imagined_mode, imagined_action, imagined_reward, next_imagined_state))
                    
                    # Actualizar el estado para el siguiente paso de imaginación
                    imagined_state = next_imagined_state
                    
                    self.logger.info(f"Rollout {rollout_idx + 1}: modo={imagined_mode}, recompensa={imagined_reward}")
            else:
                self.logger.info(f"No hay suficientes muestras para imagination rollouts (buffer: {len(self.hnaf_model.replay_buffer)}/{self.hnaf_model.batch_size})")
            # --- FIN DE IMAGINATION ROLLOUTS ---
            
            # Actualizar estados
            previous_state = state.copy()
            state = next_state
            total_reward += reward
            self.logger.info(f"Total reward acumulado: {total_reward}")
            
            # Criterio de parada
            state_norm = np.linalg.norm(state)
            self.logger.info(f"Norma del estado: {state_norm}")
            if state_norm < 0.01:
                self.logger.info("Sistema estable, terminando episodio")
                break
                
        self.logger.info(f"=== EPISODIO COMPLETADO ===")
        self.logger.info(f"Total reward: {total_reward}")
        self.logger.info(f"Mode counts: {mode_counts}")
        self.logger.info(f"Pasos realizados: {step + 1}")
        
        # Logging detallado del fin del episodio
        if episode_idx is not None and self.detailed_logger:
            # Calcular loss promedio y stability score
            avg_loss = 0.0  # Se calculará si hay losses disponibles
            stability_score = 1.0 - np.linalg.norm(state)  # Score simple basado en distancia al origen
            
            self.detailed_logger.log_episode_end(
                episode_idx, total_reward, mode_counts, avg_loss, stability_score
            )
            
            # Detectar colapso de modos
            self.detailed_logger.log_mode_collapse_detection(
                episode_idx + 1, mode_counts
            )
        
        return total_reward, mode_counts 