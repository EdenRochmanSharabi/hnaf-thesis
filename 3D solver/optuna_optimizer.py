#!/usr/bin/env python3
"""
Optimizador de hiperparámetros usando Optuna
Actualiza automáticamente config.yaml con mejores parámetros
"""

import optuna
import json
import numpy as np
import threading
import time
from typing import Dict, Any, Optional, Callable
import os
import sys

# Importar config manager para actualizar config.yaml
from config_manager import get_config_manager

class OptunaOptimizer:
    def __init__(self):
        self.is_running = True  # Iniciar como True para que funcione continuamente
        self.current_iteration = 0
        self.best_score = float('-inf')
        self.best_params = None
        self.optimization_history = []
        self.study = None
        self.optimization_thread = None
        self.progress_callback = None
        
        # Configuración para optimización nocturna
        self.max_iterations = None  # Sin límite de trials - solo detener manualmente
        # Cargar desde configuración en lugar de hardcodear
        config_manager = get_config_manager()
        hardcode_config = config_manager.get_hardcode_elimination_config()
        self.evaluation_episodes = hardcode_config['optuna']['default_evaluation_episodes']
        self.timeout_minutes = None  # Sin timeout - continúa hasta parar manualmente
        self.best_params_file = "best_hyperparameters_optuna.json"
        
        # Cargar mejores parámetros existentes
        self.load_best_params()
    
    def load_best_params(self):
        """Cargar mejores parámetros desde archivo"""
        try:
            if os.path.exists(self.best_params_file):
                with open(self.best_params_file, 'r') as f:
                    data = json.load(f)
                    self.best_score = data.get('score', float('-inf'))
                    self.best_params = data.get('params', self.get_default_params())
                    print(f"Parámetros Optuna cargados - Mejor score: {self.best_score:.4f}")
        except Exception as e:
            print(f"Error cargando parámetros Optuna: {e}")
            self.best_params = self.get_default_params()
    
    def save_best_params(self):
        """Guardar mejores parámetros en archivo JSON"""
        try:
            data = {
                'score': self.best_score,
                'params': self.best_params,
                'timestamp': time.time()
            }
            with open(self.best_params_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Mejores parámetros Optuna guardados (score: {self.best_score:.4f})")
        except Exception as e:
            print(f"Error guardando parámetros Optuna: {e}")
    
    def update_config_yaml(self):
        """Actualizar config.yaml con los mejores parámetros encontrados por Optuna"""
        if not self.best_params:
            print("⚠️  No hay mejores parámetros para actualizar config.yaml")
            return False
            
        try:
            config_manager = get_config_manager()
            
            print("🔄 Actualizando config.yaml con mejores parámetros de Optuna...")
            
            # Crear backup del config actual
            import shutil
            config_backup = config_manager.config_path + ".backup_pre_optuna"
            shutil.copy2(config_manager.config_path, config_backup)
            print(f"📋 Backup creado: {config_backup}")
            
            # Actualizar parámetros de red neuronal
            if 'hidden_dim' in self.best_params:
                config_manager.config['network']['defaults']['hidden_dim'] = self.best_params['hidden_dim']
            if 'num_layers' in self.best_params:
                config_manager.config['network']['defaults']['num_layers'] = self.best_params['num_layers']
            
            # Actualizar parámetros de entrenamiento
            if 'lr' in self.best_params:
                config_manager.config['training']['defaults']['learning_rate'] = self.best_params['lr']
            if 'batch_size' in self.best_params:
                config_manager.config['training']['defaults']['batch_size'] = self.best_params['batch_size']
            if 'initial_epsilon' in self.best_params:
                config_manager.config['training']['defaults']['initial_epsilon'] = self.best_params['initial_epsilon']
            if 'final_epsilon' in self.best_params:
                config_manager.config['training']['defaults']['final_epsilon'] = self.best_params['final_epsilon']
            if 'max_steps' in self.best_params:
                config_manager.config['training']['defaults']['max_steps'] = self.best_params['max_steps']
            if 'buffer_capacity' in self.best_params:
                config_manager.config['training']['defaults']['buffer_capacity'] = self.best_params['buffer_capacity']
            if 'alpha' in self.best_params:
                config_manager.config['training']['defaults']['alpha'] = self.best_params['alpha']
            if 'beta' in self.best_params:
                config_manager.config['training']['defaults']['beta'] = self.best_params['beta']
            if 'tau' in self.best_params:
                config_manager.config['training']['defaults']['tau'] = self.best_params['tau']
            if 'gamma' in self.best_params:
                config_manager.config['training']['defaults']['gamma'] = self.best_params['gamma']
            if 'supervised_episodes' in self.best_params:
                config_manager.config['training']['defaults']['supervised_episodes'] = self.best_params['supervised_episodes']
            if 'reward_normalize' in self.best_params:
                config_manager.config['training']['defaults']['reward_normalize'] = self.best_params['reward_normalize']
            if 'reward_shaping' in self.best_params:
                config_manager.config['training']['defaults']['reward_shaping'] = self.best_params['reward_shaping']
            
            # Agregar metadatos de optimización
            if 'optuna_optimization' not in config_manager.config:
                config_manager.config['optuna_optimization'] = {}
            
            config_manager.config['optuna_optimization'] = {
                'last_update': time.strftime('%Y-%m-%d %H:%M:%S'),
                'best_score': self.best_score,
                'total_trials': len(self.optimization_history),
                'optimized_params': list(self.best_params.keys()),
                'backup_file': config_backup
            }
            
            # Guardar config.yaml actualizado
            config_manager.save_config(config_manager.config)
            
            print("✅ config.yaml actualizado exitosamente con mejores parámetros de Optuna")
            print(f"📊 Mejor score: {self.best_score:.4f}")
            print(f"🔧 Parámetros optimizados: {list(self.best_params.keys())}")
            print(f"📁 Backup disponible en: {config_backup}")
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR actualizando config.yaml: {e}")
            print("🔄 Intentando restaurar desde backup...")
            try:
                if os.path.exists(config_backup):
                    shutil.copy2(config_backup, config_manager.config_path)
                    print("✅ Config.yaml restaurado desde backup")
            except Exception as restore_error:
                print(f"❌ Error restaurando backup: {restore_error}")
            return False
    
    def get_default_params(self):
        """Parámetros por defecto (desde configuración)"""
        config_manager = get_config_manager()
        network_defaults = config_manager.get_network_defaults()
        training_defaults = config_manager.get_training_defaults()
        interface_config = config_manager.get_interface_config()
        
        return {
            "state_dim": network_defaults['state_dim'],
            "action_dim": network_defaults['action_dim'],
            "num_modes": network_defaults['num_modes'],
            "hidden_dim": network_defaults['hidden_dim'],
            "num_layers": network_defaults['num_layers'],
            "lr": training_defaults['learning_rate'],
            "batch_size": training_defaults['batch_size'],
            "initial_epsilon": training_defaults['initial_epsilon'],
            "final_epsilon": training_defaults['final_epsilon'],
            "max_steps": training_defaults['max_steps'],
            "buffer_capacity": training_defaults['buffer_capacity'],
            "alpha": training_defaults['alpha'],
            "beta": training_defaults['beta'],
            "tau": training_defaults['tau'],
            "gamma": training_defaults['gamma'],
            "num_episodes": self.evaluation_episodes,
            "reward_normalize": interface_config['checkboxes']['reward_normalize'],
            "reward_shaping": interface_config['checkboxes']['reward_shaping']
        }
    
    def objective(self, trial):
        """Función objetivo para Optuna - CADA TRIAL ES INDEPENDIENTE"""
        if not self.is_running:
            raise optuna.TrialPruned()
        
        # 🚀 IMPORTANTE: Cada trial debe ser completamente independiente
        # No hay fuga de estado entre trials - cada uno empieza desde cero
        
        # Definir espacio de búsqueda (solo parámetros optimizables)
        optuna_params = {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64, 128, 256, 512]),
            "num_layers": trial.suggest_categorical("num_layers", [2, 3, 4, 5, 6]),
            "lr": trial.suggest_float("lr", 1e-6, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256]),
            "initial_epsilon": trial.suggest_categorical("initial_epsilon", [0.3, 0.5, 0.7, 0.9]),
            "final_epsilon": trial.suggest_categorical("final_epsilon", [0.1, 0.2, 0.3, 0.4]),
            "max_steps": trial.suggest_categorical("max_steps", [10, 20, 50, 100, 200]),
            "buffer_capacity": trial.suggest_categorical("buffer_capacity", [1000, 5000, 10000, 20000, 50000]),
            "alpha": trial.suggest_categorical("alpha", [0.1, 0.3, 0.5, 0.7, 0.9]),
            "beta": trial.suggest_categorical("beta", [0.1, 0.3, 0.5, 0.7, 0.9]),
            "tau": trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.05, 0.1]),
            "gamma": trial.suggest_categorical("gamma", [0.9, 0.95, 0.99, 0.995, 0.999]),
            "supervised_episodes": trial.suggest_categorical("supervised_episodes", [0, 50, 100, 200])
        }
        
        # Agregar parámetros fijos requeridos por el training manager
        config_manager = get_config_manager()
        base_params = {
            'state_dim': 2,  # Fijo para este problema
            'action_dim': 2,  # Fijo para este problema
            'num_modes': 2,   # Fijo para este problema
            'num_episodes': self.evaluation_episodes,  # Episodios para evaluación robusta
            'reward_normalize': trial.suggest_categorical("reward_normalize", [True, False]),
            'reward_shaping': trial.suggest_categorical("reward_shaping", [True, False]),
            'reward_optimization': 'minimizar',  # Fijo
            'gui_reward_function': trial.suggest_categorical("reward_function", [
                'np.linalg.norm([x, y])',  # Función original
                'mode_aware_reward'  # Función de recompensa original centrada en estabilización
            ]),
            'gui_matrices': self._get_dynamic_matrices(trial, config_manager)
        }
        # Nota: tau y gamma ahora son optimizados por Optuna, no fijos
        
        # Combinar parámetros optimizables con parámetros fijos
        params = {**base_params, **optuna_params}
        
        # Evaluar parámetros
        try:
            score = self.evaluate_params(params)
            return score
        except Exception as e:
            print(f"Error evaluando parámetros Optuna: {e}")
            return float('-inf')
    
    def evaluate_params(self, params):
        """Evaluar parámetros usando el training manager - NUEVA INSTANCIA POR TRIAL"""
        try:
            import logging
            logger = logging.getLogger(__name__)
            
            from training_manager import TrainingManager
            
            logger.info(f"🔄 Evaluando trial con parámetros: {params}")
            
            # 🚀 SOLUCIÓN: Crear NUEVA instancia del TrainingManager para cada trial
            # Esto evita la "fuga de estado" entre pruebas de Optuna
            print(f"🆕 Creando NUEVA instancia para trial - Sin fuga de estado")
            training_manager = TrainingManager()
            hnaf_model, training_results = training_manager.train_hnaf(params)
            
            if training_results and 'grid_accuracies' in training_results:
                final_accuracy = training_results['grid_accuracies'][-1] if training_results['grid_accuracies'] else 0
                avg_reward = np.mean(training_results['episode_rewards'][-50:]) if training_results['episode_rewards'] else 0
                
                # Normalizar recompensa (asumiendo rango típico -20 a 0)
                normalized_reward = max(0, (avg_reward + 20) / 20)
                
                # Score combinado: 50% precisión + 50% recompensa (más énfasis en precisión)
                score = 0.5 * final_accuracy + 0.5 * normalized_reward
                
                # PENALIZAR monopolio de un solo modo - necesitamos exploración balanceada
                mode_counts = training_results.get('mode_selection_counts', {})
                if mode_counts:
                    mode_balance = min(mode_counts.values()) / max(mode_counts.values()) if max(mode_counts.values()) > 0 else 0
                    # Bonus por balance: +0.2 si usa ambos modos equitativamente
                    balance_bonus = 0.2 * mode_balance
                    score += balance_bonus
                
                logger.info(f"✅ Trial completado - Score: {score:.4f} (precision: {final_accuracy:.2%}, recompensa: {avg_reward:.4f}, normalizada: {normalized_reward:.3f}, balance: {mode_balance:.3f})")
                print(f"   Score Optuna: {score:.4f} (precision: {final_accuracy:.2%}, recompensa: {avg_reward:.4f}, normalizada: {normalized_reward:.3f})")
                
                return score
            else:
                logger.warning("❌ Trial falló - No hay resultados de entrenamiento")
                return float('-inf')
                
        except Exception as e:
            logger.error(f"❌ Error en evaluación Optuna: {e}")
            print(f"Error en evaluación Optuna: {e}")
            return float('-inf')
    
    def optimize_loop(self):
        """Bucle principal de optimización"""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("🚀 Iniciando optimización automática con Optuna")
        print("Iniciando optimización automática con Optuna")
        
        if self.timeout_minutes:
            logger.info(f"Timeout: {self.timeout_minutes} minutos")
            print(f"Timeout: {self.timeout_minutes} minutos")
        else:
            logger.info("⏰ Sin límite de tiempo - continúa hasta parar manualmente")
            print("⏰ Sin límite de tiempo - continúa hasta parar manualmente")
        
        if self.max_iterations:
            logger.info(f"Máximo iteraciones: {self.max_iterations}")
            print(f"Máximo iteraciones: {self.max_iterations}")
        else:
            logger.info("🔄 Sin límite de trials - optimización continua")
            print("🔄 Sin límite de trials - optimización continua")
        
        logger.info("Explorando espacio de hiperparámetros con Optuna")
        print("Explorando espacio de hiperparámetros con Optuna")
        
        try:
            # Crear estudio Optuna
            self.study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            
            logger.info("✅ Estudio Optuna creado exitosamente")
            
            # Ejecutar optimización continua
            while self.is_running:
                try:
                    # Ejecutar un trial
                    trial = self.study.ask()
                    value = self.objective(trial)
                    self.study.tell(trial, value)
                    
                    # Log del progreso
                    logger.info(f"📊 Trial {len(self.study.trials)} completado - Score: {value:.4f}")
                    if value > self.best_score:
                        logger.info(f"🎉 Nuevo mejor score: {value:.4f} (anterior: {self.best_score:.4f})")
                        self.best_score = value
                        self.best_params = trial.params
                        self.save_best_params()
                    
                except KeyboardInterrupt:
                    logger.info("⏹️ Optimización detenida por el usuario")
                    break
                except Exception as e:
                    logger.error(f"❌ Error en trial: {e}")
                    continue
            
            logger.info("✅ Optimización Optuna completada")
            logger.info(f"🏆 Mejor score final: {self.best_score}")
            logger.info(f"📈 Total trials: {len(self.study.trials)}")
            
            print("Optimización Optuna completada")
            print(f"Mejor score: {self.best_score}")
            print(f"Total iteraciones: {len(self.study.trials)}")
            
        except Exception as e:
            logger.error(f"❌ Error en optimización Optuna: {e}")
            print(f"Error en optimización Optuna: {e}")
        finally:
            self.is_running = False
    
    def _optuna_callback(self, study, trial):
        """Callback para Optuna - Actualiza config.yaml automáticamente"""
        if not self.is_running:
            return
        
        self.current_iteration = len(study.trials)
        current_score = trial.value if trial.value is not None else float('-inf')
        
        # Actualizar mejor score
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_params = trial.params
            
            # Guardar en archivo JSON
            self.save_best_params()
            
            # 🎯 NUEVA FUNCIONALIDAD: Actualizar config.yaml automáticamente
            print(f"🎉 Nuevo mejor score Optuna: {self.best_score:.4f}!")
            print("🔄 Actualizando config.yaml con nuevos mejores parámetros...")
            
            success = self.update_config_yaml()
            if success:
                print("✅ config.yaml actualizado automáticamente por Optuna")
            else:
                print("❌ Error actualizando config.yaml - Mejores parámetros guardados solo en JSON")
        
        # Guardar en historial
        self.optimization_history.append({
            'iteration': self.current_iteration,
            'score': current_score,
            'params': trial.params
        })
        
        # Llamar callback de progreso
        if self.progress_callback:
            self.progress_callback(self.current_iteration, current_score, self.best_score)
    
    def start_optimization(self, progress_callback=None):
        """Iniciar optimización"""
        if self.is_running:
            print("Optimización Optuna ya está ejecutándose")
            return
        
        self.is_running = True
        self.current_iteration = 0
        self.progress_callback = progress_callback
        
        # Ejecutar en thread separado
        self.optimization_thread = threading.Thread(target=self.optimize_loop)
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
        
        print("Optimización Optuna iniciada en segundo plano")
    
    def stop_optimization(self):
        """Detener optimización"""
        if not self.is_running:
            print("Optimización Optuna no está ejecutándose")
            return
        
        self.is_running = False
        print("Optimización Optuna detenida por usuario")
    
    def get_best_params(self):
        """Obtener mejores parámetros"""
        return self.best_params if self.best_params else self.get_default_params()
    
    def _get_dynamic_matrices(self, trial, config_manager):
        """Obtener matrices dinámicamente - usar config por defecto o optimizar"""
        
        # Opción 1: Usar matrices del config.yaml por defecto
        defaults_config = config_manager.get_defaults_config()
        matrices_config = defaults_config.get('matrices', {})
        
        # VERIFICAR MATRICES (SIN FALLBACKS)
        if 'A1' not in matrices_config or 'A2' not in matrices_config:
            raise RuntimeError("❌ ERROR: Matrices A1/A2 faltantes en config.yaml")
        default_A1 = matrices_config['A1']
        default_A2 = matrices_config['A2']
        
        # Opción 2: Permitir que Optuna optimice elementos de matrices (opcional)
        optimize_matrices = trial.suggest_categorical("optimize_matrices", [False, True])
        
        if optimize_matrices:
            # Optimizar elementos clave de las matrices
            # A1 = [[a11, a12], [a21, a22]]
            a11 = trial.suggest_float("A1_11", 0.1, 2.0)
            a12 = trial.suggest_float("A1_12", 1.0, 100.0)
            a21 = trial.suggest_float("A1_21", -10.0, 0.0) 
            a22 = trial.suggest_float("A1_22", 0.1, 2.0)
            
            # A2 = [[b11, b12], [b21, b22]]  
            b11 = trial.suggest_float("A2_11", 0.1, 2.0)
            b12 = trial.suggest_float("A2_12", -10.0, 0.0)
            b21 = trial.suggest_float("A2_21", 1.0, 100.0)
            b22 = trial.suggest_float("A2_22", 0.1, 2.0)
            
            # Mantener compatibilidad: estas variables ya no se usan directamente en 3D
            optimized_A1 = [[a11, a12], [a21, a22]]
            optimized_A2 = [[b11, b12], [b21, b22]]
            
            print(f"🔬 Optuna optimizando matrices:")
            print(f"   A1: {optimized_A1}")
            print(f"   A2: {optimized_A2}")
            
            return {
                'A1': optimized_A1,
                'A2': optimized_A2
            }
        else:
            # Usar matrices por defecto del config
            print(f"📊 Usando matrices por defecto del config:")
            print(f"   A1: {default_A1}")
            print(f"   A2: {default_A2}")
            
            return {
                'A1': default_A1,
                'A2': default_A2
            }

    def get_optimization_status(self):
        """Obtener estado de optimización"""
        return {
            'is_running': self.is_running,
            'current_iteration': self.current_iteration,
            'best_score': self.best_score,
            'total_iterations': self.max_iterations if self.max_iterations else "Sin límite",
            'completed_trials': len(self.study.trials) if self.study else 0
        } 