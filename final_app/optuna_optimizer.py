#!/usr/bin/env python3
"""
Optimizador de hiperparámetros usando Optuna
"""

import optuna
import json
import numpy as np
import threading
import time
from typing import Dict, Any, Optional, Callable
import os

class OptunaOptimizer:
    def __init__(self):
        self.is_running = False
        self.current_iteration = 0
        self.best_score = float('-inf')
        self.best_params = None
        self.optimization_history = []
        self.study = None
        self.optimization_thread = None
        self.progress_callback = None
        
        # Configuración
        self.max_iterations = 50
        self.evaluation_episodes = 100
        self.timeout_minutes = 60
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
        """Guardar mejores parámetros en archivo"""
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
    
    def get_default_params(self):
        """Parámetros por defecto"""
        return {
            "hidden_dim": 64,
            "num_layers": 3,
            "lr": 1e-3,
            "batch_size": 64,
            "initial_epsilon": 0.5,
            "final_epsilon": 0.01,
            "max_steps": 30,
            "buffer_capacity": 5000,
            "alpha": 0.6,
            "beta": 0.4
        }
    
    def objective(self, trial):
        """Función objetivo para Optuna"""
        if not self.is_running:
            raise optuna.TrialPruned()
        
        # Definir espacio de búsqueda
        params = {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64, 128, 256, 512]),
            "num_layers": trial.suggest_categorical("num_layers", [2, 3, 4, 5, 6]),
            "lr": trial.suggest_float("lr", 1e-6, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256]),
            "initial_epsilon": trial.suggest_categorical("initial_epsilon", [0.1, 0.3, 0.5, 0.7, 0.9]),
            "final_epsilon": trial.suggest_categorical("final_epsilon", [0.001, 0.01, 0.05, 0.1, 0.2]),
            "max_steps": trial.suggest_categorical("max_steps", [10, 20, 50, 100, 200]),
            "buffer_capacity": trial.suggest_categorical("buffer_capacity", [1000, 5000, 10000, 20000, 50000]),
            "alpha": trial.suggest_categorical("alpha", [0.1, 0.3, 0.5, 0.7, 0.9]),
            "beta": trial.suggest_categorical("beta", [0.1, 0.3, 0.5, 0.7, 0.9])
        }
        
        # Evaluar parámetros
        try:
            score = self.evaluate_params(params)
            return score
        except Exception as e:
            print(f"Error evaluando parámetros Optuna: {e}")
            return float('-inf')
    
    def evaluate_params(self, params):
        """Evaluar parámetros usando el training manager"""
        try:
            from .training_manager import TrainingManager
            
            training_manager = TrainingManager()
            training_results = training_manager.train_hnaf(params)
            
            if training_results and 'grid_accuracies' in training_results:
                final_accuracy = training_results['grid_accuracies'][-1] if training_results['grid_accuracies'] else 0
                avg_reward = np.mean(training_results['episode_rewards'][-50:]) if training_results['episode_rewards'] else 0
                
                # Normalizar recompensa (asumiendo rango típico -20 a 0)
                normalized_reward = max(0, (avg_reward + 20) / 20)
                
                # Score combinado: 30% precisión + 70% recompensa
                score = 0.3 * final_accuracy + 0.7 * normalized_reward
                
                print(f"   Score Optuna: {score:.4f} (precision: {final_accuracy:.2%}, recompensa: {avg_reward:.4f}, normalizada: {normalized_reward:.3f})")
                
                return score
            else:
                return float('-inf')
                
        except Exception as e:
            print(f"Error en evaluación Optuna: {e}")
            return float('-inf')
    
    def optimize_loop(self):
        """Bucle principal de optimización"""
        print("Iniciando optimización automática con Optuna")
        print(f"Timeout: {self.timeout_minutes} minutos")
        print(f"Maximo iteraciones: {self.max_iterations}")
        print("Explorando espacio de hiperparámetros con Optuna")
        
        try:
            # Crear estudio Optuna
            self.study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            
            # Ejecutar optimización
            self.study.optimize(
                self.objective,
                n_trials=self.max_iterations,
                timeout=self.timeout_minutes * 60,
                callbacks=[self._optuna_callback]
            )
            
            print("Optimización Optuna completada")
            print(f"Mejor score: {self.best_score}")
            print(f"Total iteraciones: {len(self.study.trials)}")
            
        except Exception as e:
            print(f"Error en optimización Optuna: {e}")
        finally:
            self.is_running = False
    
    def _optuna_callback(self, study, trial):
        """Callback para Optuna"""
        if not self.is_running:
            return
        
        self.current_iteration = len(study.trials)
        current_score = trial.value if trial.value is not None else float('-inf')
        
        # Actualizar mejor score
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_params = trial.params
            self.save_best_params()
            print(f"Nuevo mejor score Optuna: {self.best_score:.4f}!")
        
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
    
    def get_optimization_status(self):
        """Obtener estado de optimización"""
        return {
            'is_running': self.is_running,
            'current_iteration': self.current_iteration,
            'best_score': self.best_score,
            'total_iterations': self.max_iterations
        } 