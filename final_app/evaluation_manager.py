#!/usr/bin/env python3
"""
Módulo de evaluación para HNAF
Maneja la evaluación del modelo entrenado
"""

import numpy as np

class EvaluationManager:
    """Manager para la evaluación del HNAF"""
    
    def __init__(self):
        pass
    
    def evaluate_model(self, hnaf_model):
        """
        Evaluar el modelo entrenado
        
        Args:
            hnaf_model: Modelo HNAF entrenado
            
        Returns:
            dict: Resultados de la evaluación
        """
        print("DEBUG: Iniciando evaluación del modelo")
        print("\n" + "="*60)
        print("EVALUACIÓN DEL MODELO")
        print("="*60)
        
        # Evaluación estándar
        eval_reward, mode_selections = hnaf_model.evaluate_policy(num_episodes=20)
        
        print(f"Recompensa promedio de evaluación: {eval_reward:.4f}")
        print(f"Distribución de selección de modos: {mode_selections}")
        
        results = {
            'eval_reward': eval_reward,
            'mode_selections': mode_selections
        }
        
        # Evaluación en grid si está disponible
        if hasattr(hnaf_model, 'evaluate_policy_grid'):
            print("\n📊 Evaluación en grid 100x100:")
            grid_results = hnaf_model.evaluate_policy_grid(grid_size=100)
            print(f"Precisión en grid: {grid_results['optimal_accuracy']:.2%}")
            print(f"Q-values promedio: {np.mean(grid_results['q_values']):.4f}")
            print(f"Q-values std: {np.std(grid_results['q_values']):.4f}")
            
            results.update({
                'grid_accuracy': grid_results['optimal_accuracy'],
                'q_values_mean': np.mean(grid_results['q_values']),
                'q_values_std': np.std(grid_results['q_values'])
            })
        
        print("="*60)
        
        return results
    
    def verify_hnaf(self, hnaf_model):
        """
        Verificar el funcionamiento del HNAF
        
        Args:
            hnaf_model: Modelo HNAF entrenado
        """
        print("DEBUG: Iniciando verificación HNAF")
        print("\n" + "="*60)
        print("VERIFICACIÓN HNAF")
        print("="*60)
        
        hnaf_model.verify_hnaf()
    
    def compare_models(self, model1, model2, test_states=None):
        """
        Comparar dos modelos HNAF
        
        Args:
            model1: Primer modelo
            model2: Segundo modelo
            test_states: Estados de prueba (opcional)
            
        Returns:
            dict: Resultados de la comparación
        """
        if test_states is None:
            test_states = [
                np.array([0.1, 0.1]),
                np.array([0, 0.1]),
                np.array([0.1, 0]),
                np.array([0.05, 0.05]),
                np.array([-0.05, 0.08])
            ]
        
        print("="*60)
        print("COMPARACIÓN DE MODELOS")
        print("="*60)
        
        results = {
            'model1_accuracy': 0,
            'model2_accuracy': 0,
            'model1_rewards': [],
            'model2_rewards': []
        }
        
        correct1 = 0
        correct2 = 0
        
        for i, state in enumerate(test_states):
            print(f"\nEstado {i+1}: {state}")
            
            # Evaluar modelo 1
            mode1, action1 = model1.select_action(state, epsilon=0.0)
            Q1 = model1.compute_Q_value(state, action1, mode1)
            
            # Evaluar modelo 2
            mode2, action2 = model2.select_action(state, epsilon=0.0)
            Q2 = model2.compute_Q_value(state, action2, mode2)
            
            # Determinar modo óptimo
            optimal_mode = self._get_optimal_mode(state)
            
            # Verificar precisión
            if mode1 == optimal_mode:
                correct1 += 1
            if mode2 == optimal_mode:
                correct2 += 1
            
            print(f"  Modelo 1: Modo {mode1}, Q={Q1:.4f}")
            print(f"  Modelo 2: Modo {mode2}, Q={Q2:.4f}")
            print(f"  Óptimo: Modo {optimal_mode}")
        
        # Calcular precisión
        accuracy1 = correct1 / len(test_states)
        accuracy2 = correct2 / len(test_states)
        
        results['model1_accuracy'] = accuracy1
        results['model2_accuracy'] = accuracy2
        
        print(f"\n📊 RESULTADOS:")
        print(f"  Modelo 1 precisión: {accuracy1:.2%}")
        print(f"  Modelo 2 precisión: {accuracy2:.2%}")
        
        return results
    
    def _get_optimal_mode(self, state):
        """Determinar el modo óptimo para un estado"""
        # Esta función debería usar la misma lógica que el HNAF
        # Por simplicidad, usamos una implementación básica
        from src.naf_corrected import CorrectedOptimizationFunctions
        
        naf = CorrectedOptimizationFunctions(t=1.0)
        
        # Calcular recompensas para ambos modos
        x1 = naf.execute_function("transform_x1", state[0], state[1])
        x2 = naf.execute_function("transform_x2", state[0], state[1])
        
        r1 = naf.execute_function("reward_function", x1[0, 0], x1[1, 0], state[0], state[1])
        r2 = naf.execute_function("reward_function", x2[0, 0], x2[1, 0], state[0], state[1])
        
        # Retornar el modo con menor recompensa (mejor)
        return 0 if abs(r1) < abs(r2) else 1 