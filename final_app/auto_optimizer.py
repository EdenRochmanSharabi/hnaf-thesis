#!/usr/bin/env python3
"""
Optimizador automático con Gemini API
Busca los mejores hiperparámetros para HNAF
"""

import json
import time
import threading
import numpy as np
import google.generativeai as genai
from datetime import datetime
import os
import sys

# Agregar el directorio raíz al path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    from config import GEMINI_API_KEY, AUTO_OPTIMIZATION_CONFIG
except ImportError:
    print("⚠️  config.py no encontrado. Usando configuración por defecto.")
    GEMINI_API_KEY = "AIzaSyAmN3fmfw4whoCWMgH88YvBe-bNmwM84B0"
    AUTO_OPTIMIZATION_CONFIG = {
    "max_iterations": 50,
    "evaluation_episodes": 100,
    "timeout_minutes": 60,
    "best_params_file": "best_hyperparameters.json"
}

class AutoOptimizer:
    """Optimizador automático con Gemini"""
    
    def __init__(self):
        self.is_running = False
        self.current_iteration = 0
        self.best_score = float('-inf')
        self.best_params = None
        self.optimization_history = []
        
        # Configurar Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Cargar mejores parámetros existentes
        self.load_best_params()
    
    def load_best_params(self):
        """Cargar mejores parámetros desde archivo"""
        try:
            if os.path.exists(AUTO_OPTIMIZATION_CONFIG['best_params_file']):
                with open(AUTO_OPTIMIZATION_CONFIG['best_params_file'], 'r') as f:
                    data = json.load(f)
                    self.best_params = data.get('best_params')
                    self.best_score = data.get('best_score', float('-inf'))
                    self.optimization_history = data.get('history', [])
                print(f"Cargados mejores parámetros (score: {self.best_score:.4f})")
        except Exception as e:
            print(f"⚠️  Error cargando mejores parámetros: {e}")
    
    def save_best_params(self):
        """Guardar mejores parámetros en archivo"""
        try:
            data = {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'history': self.optimization_history,
                'last_updated': datetime.now().isoformat()
            }
            with open(AUTO_OPTIMIZATION_CONFIG['best_params_file'], 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Mejores parámetros guardados (score: {self.best_score:.4f})")
        except Exception as e:
            print(f"❌ Error guardando parámetros: {e}")
    
    def get_optimization_prompt(self, current_results=None):
        """Generar prompt para Gemini"""
        base_prompt = """
Eres un experto en optimización de hiperparámetros para redes neuronales de aprendizaje por refuerzo.

Tarea: Optimizar hiperparámetros para HNAF (Hybrid Normalized Advantage Function)

Contexto del problema:
- Sistema de control híbrido con 2 modos de funcionamiento
- Estado: 2 dimensiones, Acción: 2 dimensiones
- Objetivo: Estabilizar sistema seleccionando modo óptimo
- Función de recompensa: r(x_t, i) = -|‖x_{t+1}‖ - ‖x_t‖|

IMPORTANTE: Debes explorar DIVERSIDAD de configuraciones. NO te limites a valores similares.

Parámetros a optimizar (RANGOS AMPLIOS):
1. hidden_dim: [16, 32, 64, 128, 256, 512] - Explora toda la gama
2. num_layers: [2, 3, 4, 5, 6] - Prueba diferentes profundidades
3. lr: [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1] - RANGO AMPLIO de learning rates
4. batch_size: [8, 16, 32, 64, 128, 256] - Explora diferentes tamaños
5. initial_epsilon: [0.1, 0.3, 0.5, 0.7, 0.9] - Diferentes niveles de exploración
6. final_epsilon: [0.001, 0.01, 0.05, 0.1, 0.2] - Rango amplio de explotación
7. max_steps: [10, 20, 50, 100, 200] - Diferentes horizontes temporales
8. buffer_capacity: [1000, 5000, 10000, 20000, 50000] - Diferentes capacidades
9. alpha: [0.1, 0.3, 0.5, 0.7, 0.9] - Diferentes prioridades
10. beta: [0.1, 0.3, 0.5, 0.7, 0.9] - Diferentes correcciones de sesgo

ESTRATEGIA DE BÚSQUEDA:
- Varía SIGNIFICATIVAMENTE los parámetros entre iteraciones
- Prueba learning rates MUY diferentes (1e-6 hasta 1e-1)
- Explora batch sizes desde 8 hasta 256
- Prueba max_steps desde 10 hasta 200
- Varía buffer_capacity desde 1000 hasta 50000
- NO te quedes en valores similares

Métricas de evaluación:
- Precisión en grid (0-100%)
- Recompensa promedio (métrica principal)
- Score combinado: 0.3 × precisión + 0.7 × recompensa_normalizada

Responde SOLO con un JSON válido con los parámetros optimizados:
{
    "hidden_dim": int,
    "num_layers": int,
    "lr": float,
    "batch_size": int,
    "initial_epsilon": float,
    "final_epsilon": float,
    "max_steps": int,
    "buffer_capacity": int,
    "alpha": float,
    "beta": float,
    "reasoning": "explicación de la elección y diversidad explorada"
}
"""
        
        if current_results:
            # Analizar diversidad de configuraciones anteriores
            recent_params = [r['params'] for r in self.optimization_history[-5:]]
            lr_values = [p.get('lr', 0) for p in recent_params]
            batch_values = [p.get('batch_size', 0) for p in recent_params]
            hidden_values = [p.get('hidden_dim', 0) for p in recent_params]
            
            base_prompt += f"""

Resultados actuales:
- Mejor score: {self.best_score:.4f}
- Iteración actual: {self.current_iteration}
- Historial de scores: {[r['score'] for r in self.optimization_history[-5:]]}

ANÁLISIS DE DIVERSIDAD:
- Learning rates probados: {lr_values}
- Batch sizes probados: {batch_values}
- Hidden dims probados: {hidden_values}

Últimos resultados:
{json.dumps(current_results, indent=2)}

INSTRUCCIONES CRÍTICAS:
- Varía SIGNIFICATIVAMENTE los parámetros
- Prueba learning rates MUY diferentes (1e-6 hasta 1e-1)
- Explora batch sizes desde 8 hasta 256
- Prueba max_steps desde 10 hasta 200
- Varía buffer_capacity desde 1000 hasta 50000
- NO repitas configuraciones similares
- La recompensa es la métrica principal (70% del score)

Basándote en estos resultados, sugiere parámetros DIVERSOS y mejorados.
"""
        
        return base_prompt
    
    def parse_gemini_response(self, response):
        """Parsear respuesta de Gemini"""
        try:
            # Extraer JSON de la respuesta
            text = response.text
            start = text.find('{')
            end = text.rfind('}') + 1
            
            if start != -1 and end != 0:
                json_str = text[start:end]
                params = json.loads(json_str)
                
                # Validar parámetros
                required_params = [
                    'hidden_dim', 'num_layers', 'lr', 'batch_size',
                    'initial_epsilon', 'final_epsilon', 'max_steps',
                    'buffer_capacity', 'alpha', 'beta'
                ]
                
                for param in required_params:
                    if param not in params:
                        raise ValueError(f"Falta parámetro: {param}")
                
                return params
            else:
                raise ValueError("No se encontró JSON en la respuesta")
                
        except Exception as e:
            print(f"❌ Error parseando respuesta de Gemini: {e}")
            return self.get_default_params()
    
    def get_default_params(self):
        """Parámetros por defecto con mayor diversidad"""
        import random
        
        # Generar parámetros diversos por defecto
        return {
            "hidden_dim": random.choice([16, 32, 64, 128, 256, 512]),
            "num_layers": random.choice([2, 3, 4, 5, 6]),
            "lr": random.choice([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
            "batch_size": random.choice([8, 16, 32, 64, 128, 256]),
            "initial_epsilon": random.choice([0.1, 0.3, 0.5, 0.7, 0.9]),
            "final_epsilon": random.choice([0.001, 0.01, 0.05, 0.1, 0.2]),
            "max_steps": random.choice([10, 20, 50, 100, 200]),
            "buffer_capacity": random.choice([1000, 5000, 10000, 20000, 50000]),
            "alpha": random.choice([0.1, 0.3, 0.5, 0.7, 0.9]),
            "beta": random.choice([0.1, 0.3, 0.5, 0.7, 0.9]),
            "reasoning": "Parámetros diversos por defecto (fallback)"
        }
    
    def evaluate_params(self, params, training_manager):
        """Evaluar parámetros con entrenamiento"""
        try:
            print(f"Evaluando parámetros (iteración {self.current_iteration + 1}):")
            print(f"   hidden_dim: {params['hidden_dim']}")
            print(f"   num_layers: {params['num_layers']}")
            print(f"   lr: {params['lr']}")
            print(f"   batch_size: {params['batch_size']}")
            
            # Configurar parámetros base
            base_params = {
                'state_dim': 2,
                'action_dim': 2,
                'num_modes': 2,
                'num_episodes': AUTO_OPTIMIZATION_CONFIG['evaluation_episodes'],
                'tau': 0.001,
                'gamma': 0.9,
                'reward_normalize': True,
                'reward_optimization': 'minimizar'
            }
            
            # Combinar con parámetros optimizados
            test_params = {**base_params, **params}
            
            # Entrenar modelo
            hnaf_model, training_results = training_manager.train_hnaf(test_params)
            
            # Calcular score
            if training_results and 'grid_accuracies' in training_results:
                final_accuracy = training_results['grid_accuracies'][-1] if training_results['grid_accuracies'] else 0
                avg_reward = np.mean(training_results['episode_rewards'][-50:]) if training_results['episode_rewards'] else 0
                
                # Normalizar recompensa (asumiendo rango típico -20 a 0)
                normalized_reward = max(0, (avg_reward + 20) / 20)  # Normalizar a [0,1]
                
                # Score combinado: 30% precisión + 70% recompensa (métrica principal)
                score = 0.3 * final_accuracy + 0.7 * normalized_reward
            else:
                score = 0
            
            print(f"   Score: {score:.4f} (precision: {final_accuracy:.2%}, recompensa: {avg_reward:.4f}, normalizada: {normalized_reward:.3f})")
            
            return {
                'params': params,
                'score': score,
                'accuracy': final_accuracy,
                'avg_reward': avg_reward,
                'training_results': training_results
            }
            
        except Exception as e:
            print(f"❌ Error evaluando parámetros: {e}")
            return {
                'params': params,
                'score': float('-inf'),
                'accuracy': 0,
                'avg_reward': 0,
                'training_results': None
            }
    
    def optimize_loop(self, training_manager, callback=None):
        """Bucle principal de optimización"""
        self.is_running = True
        start_time = time.time()
        
        print("Iniciando optimización automática con Gemini")
        print(f"Timeout: {AUTO_OPTIMIZATION_CONFIG['timeout_minutes']} minutos")
        print(f"Maximo iteraciones: {AUTO_OPTIMIZATION_CONFIG['max_iterations']}")
        print("IMPORTANTE: Explorando diversidad de configuraciones")
        
        while self.is_running and self.current_iteration < AUTO_OPTIMIZATION_CONFIG['max_iterations']:
            try:
                # Verificar timeout
                elapsed_minutes = (time.time() - start_time) / 60
                if elapsed_minutes > AUTO_OPTIMIZATION_CONFIG['timeout_minutes']:
                    print("Timeout alcanzado")
                    break
                
                # Generar parámetros con Gemini
                print(f"Consultando Gemini (iteración {self.current_iteration + 1})...")
                
                # Preparar historial para Gemini
                recent_results = None
                if self.optimization_history:
                    recent_results = {
                        'best_score': self.best_score,
                        'recent_scores': [r['score'] for r in self.optimization_history[-3:]]
                    }
                
                prompt = self.get_optimization_prompt(recent_results)
                response = self.model.generate_content(prompt)
                
                # Parsear respuesta
                new_params = self.parse_gemini_response(response)
                
                # Evaluar parámetros
                result = self.evaluate_params(new_params, training_manager)
                
                # Actualizar mejores parámetros
                if result['score'] > self.best_score:
                    self.best_score = result['score']
                    self.best_params = new_params
                    self.save_best_params()
                    print(f"Nuevo mejor score: {self.best_score:.4f}!")
                
                # Guardar historial
                self.optimization_history.append(result)
                
                # Callback para actualizar UI
                if callback:
                    callback(self.current_iteration + 1, result['score'], self.best_score)
                
                self.current_iteration += 1
                
                # Pausa entre iteraciones
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error en iteración {self.current_iteration + 1}: {e}")
                self.current_iteration += 1
                time.sleep(5)
        
        self.is_running = False
        print(f"Optimización completada")
        print(f"Mejor score: {self.best_score:.4f}")
        print(f"Total iteraciones: {self.current_iteration}")
    
    def start_optimization(self, training_manager, callback=None):
        """Iniciar optimización en thread separado"""
        thread = threading.Thread(
            target=self.optimize_loop,
            args=(training_manager, callback)
        )
        thread.daemon = True
        thread.start()
        return thread
    
    def stop_optimization(self):
        """Detener optimización"""
        self.is_running = False
        print("Optimización detenida por usuario")
    
    def get_best_params(self):
        """Obtener mejores parámetros"""
        return self.best_params if self.best_params else self.get_default_params() 