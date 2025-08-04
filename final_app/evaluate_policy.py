#!/usr/bin/env python3
"""
Script de Evaluación de Política HNAF
=====================================

Este script carga un modelo HNAF entrenado y genera todos los resultados
necesarios para demostrar la estabilidad del sistema en la tesis.

Uso:
    python evaluate_policy.py --model-path modelo_entrenado.pt
    python evaluate_policy.py --config config.yaml --auto-find-model
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import glob

# Imports de la aplicación HNAF
from hnaf_improved import HNAFImproved
from config_manager import get_config_manager
from logging_manager import get_logger
from reporting_utils import create_report, create_latex_report, calculate_stability_metrics

class PolicyEvaluator:
    """Evaluador de políticas HNAF para análisis de estabilidad"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = get_logger("PolicyEvaluator")
        
        # Configuración del sistema
        network_config = config_manager.get_network_defaults()
        self.state_dim = network_config['state_dim']
        self.action_dim = network_config['action_dim']
        self.num_modes = network_config['num_modes']
        
        # Configuración de evaluación
        eval_config = config_manager.get_evaluation_config()
        self.grid_size = eval_config.get('grid_size', 50)
        self.test_states = eval_config.get('test_states', [])
        
        # Configuración de simulación
        training_config = config_manager.get_training_defaults()
        self.max_steps = training_config.get('max_steps', 200)
        
        # Límites del estado
        state_limits = config_manager.get_state_limits()
        self.state_min = state_limits['min']
        self.state_max = state_limits['max']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Evaluador inicializado en dispositivo: {self.device}")
        
    def load_model(self, model_path):
        """Cargar modelo HNAF entrenado"""
        try:
            # Crear instancia del modelo
            self.model = HNAFImproved(self.config_manager.config, self.logger)
            
            # Cargar pesos del modelo
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            
            # Poner en modo evaluación
            self.model.model.eval()
            self.model.target_model.eval()
            
            self.logger.info(f"✅ Modelo cargado exitosamente desde: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error cargando modelo: {e}")
            return False
    
    def simulate_trajectory(self, initial_state, max_steps=None):
        """Simular una trayectoria desde un estado inicial"""
        if max_steps is None:
            max_steps = self.max_steps
            
        # Convertir estado inicial a tensor
        if isinstance(initial_state, np.ndarray):
            state = torch.FloatTensor(initial_state).to(self.device)
        else:
            state = torch.FloatTensor(initial_state).to(self.device)
        
        trajectory = [state.cpu().numpy()]
        switching_signals = []
        control_inputs = []
        rewards = []
        
        for step in range(max_steps):
            # Obtener acción del modelo (sin exploración)
            with torch.no_grad():
                action = self.model.select_action(state)
            
            # Simular paso del sistema
            next_state, reward, done = self._simulate_step(state, action)
            
            # Guardar datos
            trajectory.append(next_state.cpu().numpy())
            switching_signals.append(action[0] if isinstance(action, (list, tuple)) else action)
            control_inputs.append(action[1] if isinstance(action, (list, tuple)) else action)
            rewards.append(reward)
            
            # Actualizar estado
            state = next_state
            
            # Verificar convergencia
            if np.linalg.norm(state.cpu().numpy()) < 0.01:
                break
                
        return trajectory, switching_signals, control_inputs, rewards
    
    def _simulate_step(self, state, action):
        """Simular un paso del sistema híbrido"""
        state_np = state.cpu().numpy()
        
        # Extraer modo y control
        if isinstance(action, (list, tuple)):
            mode = int(action[0])
            control = action[1] if len(action) > 1 else np.zeros(self.action_dim)
        else:
            mode = int(action)
            control = np.zeros(self.action_dim)
        
        # Obtener matrices del sistema desde configuración
        matrices_config = self.config_manager.get_defaults_config()
        matrices = matrices_config.get('matrices', {})
        
        # Seleccionar matriz según modo
        matrix_key = f'A{mode + 1}' if f'A{mode + 1}' in matrices else 'A1'
        A = np.array(matrices[matrix_key])
        
        # Simular dinámica del sistema
        # x(t+1) = A*x(t) + B*u(t)
        B = np.eye(self.state_dim)  # Matriz de control por defecto
        next_state_np = A @ state_np + B @ control
        
        # Calcular recompensa (función de coste cuadrática)
        Q = np.eye(self.state_dim)  # Matriz de ponderación del estado
        R = 0.1 * np.eye(self.action_dim)  # Matriz de ponderación del control
        cost = next_state_np.T @ Q @ next_state_np + control.T @ R @ control
        reward = -cost
        
        # Verificar si el episodio termina
        done = np.linalg.norm(next_state_np) < 0.01 or np.linalg.norm(next_state_np) > 10.0
        
        return torch.FloatTensor(next_state_np).to(self.device), reward, done
    
    def generate_test_conditions(self, num_conditions=8):
        """Generar condiciones iniciales de prueba"""
        conditions = []
        
        # Usar estados de prueba de la configuración
        if self.test_states:
            conditions.extend(self.test_states)
        
        # Generar condiciones adicionales
        remaining = num_conditions - len(conditions)
        if remaining > 0:
            for i in range(remaining):
                # Generar estados aleatorios en el rango permitido
                state = np.random.uniform(self.state_min, self.state_max, self.state_dim)
                conditions.append(state.tolist())
        
        return conditions[:num_conditions]
    
    def plot_state_trajectories(self, results_dir, all_trajectories, initial_conditions):
        """Generar gráfica de trayectorias de estado"""
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_trajectories)))
        
        for i, (traj, ic) in enumerate(zip(all_trajectories, initial_conditions)):
            traj_array = np.array(traj)
            
            # Plotear cada componente del estado
            for j in range(traj_array.shape[1]):
                label = f'x{j+1} (IC {i+1})' if j == 0 else None
                plt.plot(traj_array[:, j], color=colors[i], alpha=0.8, label=label)
                
        plt.title("Evolución de las Trayectorias de Estado", fontsize=16)
        plt.xlabel("Pasos de Tiempo", fontsize=12)
        plt.ylabel("Valor del Estado", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        plt.tight_layout()
        
        save_path = os.path.join(results_dir, "plot_trajectories.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def plot_switching_and_control(self, results_dir, switching_signal, control_inputs):
        """Generar gráfica de señal de conmutación y control"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Gráfica de la señal de conmutación
        ax1.step(range(len(switching_signal)), switching_signal, where='post', linewidth=2)
        ax1.set_title("Ley de Conmutación (Señal de Modo vs. Tiempo)", fontsize=14)
        ax1.set_ylabel("Modo Activo (v)", fontsize=12)
        ax1.set_yticks(np.unique(switching_signal))
        ax1.grid(True, alpha=0.3)
        
        # Gráfica de las entradas de control
        control_np = np.array(control_inputs)
        for i in range(control_np.shape[1]):
            ax2.plot(control_np[:, i], label=f'u{i+1}', linewidth=2)
        ax2.set_title("Entradas de Control Continuas vs. Tiempo", fontsize=14)
        ax2.set_xlabel("Pasos de Tiempo", fontsize=12)
        ax2.set_ylabel("Valor de Control (u)", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        save_path = os.path.join(results_dir, "plot_switching_signal.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def plot_phase_portrait(self, results_dir, trajectory_to_plot):
        """Generar diagrama de fases (solo para sistemas 2D)"""
        if self.state_dim != 2:
            self.logger.warning("Diagrama de fases solo disponible para sistemas 2D")
            return None
        
        # Crear grilla del espacio de estados
        x_range = np.linspace(self.state_min, self.state_max, 30)
        y_range = np.linspace(self.state_min, self.state_max, 30)
        xx, yy = np.meshgrid(x_range, y_range)
        
        # Evaluar la elección de modo en cada punto de la grilla
        modes = np.zeros_like(xx)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                state = torch.FloatTensor([xx[i, j], yy[i, j]]).to(self.device)
                with torch.no_grad():
                    action = self.model.select_action(state)
                modes[i, j] = action[0] if isinstance(action, (list, tuple)) else action
        
        plt.figure(figsize=(10, 10))
        
        # Colorear las regiones
        contour = plt.contourf(xx, yy, modes, alpha=0.6, cmap='viridis')
        plt.colorbar(contour, label="Modo Activo (v)")
        
        # Superponer la trayectoria
        traj_np = np.array(trajectory_to_plot)
        plt.plot(traj_np[:, 0], traj_np[:, 1], 'r-', linewidth=3, label="Trayectoria de Ejemplo")
        plt.plot(traj_np[0, 0], traj_np[0, 1], 'go', markersize=12, label="Inicio")
        plt.plot(traj_np[-1, 0], traj_np[-1, 1], 'kx', markersize=12, label="Fin")
        
        plt.title("Diagrama de Fases y Regiones de Conmutación", fontsize=16)
        plt.xlabel("Estado x1", fontsize=12)
        plt.ylabel("Estado x2", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        save_path = os.path.join(results_dir, "plot_phase_portrait.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def plot_reward_analysis(self, results_dir, all_rewards):
        """Generar análisis de recompensas"""
        plt.figure(figsize=(12, 6))
        
        for i, rewards in enumerate(all_rewards):
            cumulative_rewards = np.cumsum(rewards)
            plt.plot(cumulative_rewards, label=f'Trayectoria {i+1}', alpha=0.8)
        
        plt.title("Análisis de Recompensas Acumuladas", fontsize=16)
        plt.xlabel("Pasos de Tiempo", fontsize=12)
        plt.ylabel("Recompensa Acumulada", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        save_path = os.path.join(results_dir, "plot_reward_analysis.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def evaluate_policy(self, model_path, results_dir):
        """Evaluar la política completa y generar todos los resultados"""
        self.logger.info("🚀 Iniciando evaluación de política...")
        
        # Cargar modelo
        if not self.load_model(model_path):
            return False
        
        # Generar condiciones de prueba
        initial_conditions = self.generate_test_conditions()
        self.logger.info(f"📊 Probando {len(initial_conditions)} condiciones iniciales")
        
        # Simular trayectorias
        all_trajectories = []
        all_switching_signals = []
        all_control_inputs = []
        all_rewards = []
        
        for i, ic in enumerate(initial_conditions):
            self.logger.info(f"🔄 Simulando trayectoria {i+1}/{len(initial_conditions)}")
            trajectory, switching, control, rewards = self.simulate_trajectory(ic)
            
            all_trajectories.append(trajectory)
            all_switching_signals.append(switching)
            all_control_inputs.append(control)
            all_rewards.append(rewards)
        
        # Calcular métricas de estabilidad
        stability_metrics = calculate_stability_metrics(all_trajectories, self.max_steps)
        self.logger.info(f"📈 Métricas de estabilidad calculadas: {stability_metrics}")
        
        # Generar gráficas
        self.logger.info("📊 Generando gráficas...")
        plots_paths = {}
        
        # Gráfica 1: Trayectorias de estado
        plots_paths['trajectories'] = self.plot_state_trajectories(results_dir, all_trajectories, initial_conditions)
        
        # Gráfica 2: Señal de conmutación y control (usar primera trayectoria)
        plots_paths['switching'] = self.plot_switching_and_control(results_dir, all_switching_signals[0], all_control_inputs[0])
        plots_paths['control'] = plots_paths['switching']  # Misma imagen para el informe
        
        # Gráfica 3: Diagrama de fases (si es 2D)
        if self.state_dim == 2:
            phase_portrait_path = self.plot_phase_portrait(results_dir, all_trajectories[0])
            if phase_portrait_path:
                plots_paths['phase_portrait'] = phase_portrait_path
        
        # Gráfica 4: Análisis de recompensas
        plots_paths['reward_analysis'] = self.plot_reward_analysis(results_dir, all_rewards)
        
        # Generar informes
        report_data = {
            'model_path': model_path,
            'initial_conditions': initial_conditions,
            'max_steps': self.max_steps,
            'state_dim': self.state_dim,
            'num_modes': self.num_modes,
            'config_name': 'HNAF Mejorado',
            'stability_metrics': stability_metrics,
            'plots': plots_paths
        }
        
        # Generar informe Markdown
        create_report(results_dir, report_data)
        
        # Generar informe LaTeX
        create_latex_report(results_dir, report_data)
        
        self.logger.info("✅ Evaluación completada exitosamente")
        return True

def find_latest_model():
    """Buscar el modelo más reciente en el directorio"""
    model_patterns = [
        "*.pt",
        "*.pth", 
        "model_*.pt",
        "hnaf_*.pt"
    ]
    
    for pattern in model_patterns:
        models = glob.glob(pattern)
        if models:
            # Ordenar por fecha de modificación
            latest = max(models, key=os.path.getmtime)
            return latest
    
    return None

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Evaluar política HNAF entrenada")
    parser.add_argument("--model-path", type=str, help="Ruta al modelo entrenado")
    parser.add_argument("--config", type=str, default="config.yaml", help="Archivo de configuración")
    parser.add_argument("--auto-find-model", action="store_true", help="Buscar modelo automáticamente")
    parser.add_argument("--output-dir", type=str, help="Directorio de salida")
    
    args = parser.parse_args()
    
    # Configurar directorio de salida
    if args.output_dir:
        results_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"resultados_evaluacion_{timestamp}"
    
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 Guardando resultados en: {results_dir}")
    
    # Buscar modelo si no se especifica
    model_path = args.model_path
    if not model_path and args.auto_find_model:
        model_path = find_latest_model()
        if model_path:
            print(f"🔍 Modelo encontrado automáticamente: {model_path}")
        else:
            print("❌ No se encontró ningún modelo. Entrena primero el agente.")
            return False
    
    if not model_path:
        print("❌ Debes especificar --model-path o usar --auto-find-model")
        return False
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado: {model_path}")
        return False
    
    # Inicializar evaluador
    config_manager = get_config_manager()
    evaluator = PolicyEvaluator(config_manager)
    
    # Ejecutar evaluación
    success = evaluator.evaluate_policy(model_path, results_dir)
    
    if success:
        print(f"✅ Evaluación completada. Resultados en: {results_dir}")
        print(f"📄 Informe Markdown: {results_dir}/informe_estabilidad.md")
        print(f"📄 Informe LaTeX: {results_dir}/informe_estabilidad.tex")
    else:
        print("❌ Error en la evaluación")
        return False
    
    return True

if __name__ == "__main__":
    main() 