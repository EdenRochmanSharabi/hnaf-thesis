import logging
import json
import os
from datetime import datetime
import numpy as np
import torch

class DetailedLogger:
    """Sistema de logging detallado para análisis completo del entrenamiento HNAF"""
    
    def __init__(self, log_dir="logs/detailed"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Crear archivo de log detallado
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"detailed_training_{timestamp}.log")
        self.json_file = os.path.join(log_dir, f"training_data_{timestamp}.json")
        
        # Configurar logger
        self.logger = logging.getLogger(f"DetailedLogger_{timestamp}")
        self.logger.setLevel(logging.DEBUG)
        
        # Handler para archivo
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Formato detallado
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Datos estructurados para análisis posterior
        self.training_data = {
            'metadata': {
                'start_time': timestamp,
                'version': 'HNAF_Detailed_Logger_v1.0'
            },
            'episodes': [],
            'mode_selections': [],
            'rewards': [],
            'losses': [],
            'states': [],
            'actions': [],
            'q_values': [],
            'stability_metrics': []
        }
        
        self.logger.info("=== INICIO DE LOGGING DETALLADO HNAF ===")
    
    def log_episode_start(self, episode_num, total_episodes, episode_type="normal"):
        """Registrar inicio de episodio"""
        episode_data = {
            'episode_num': episode_num,
            'total_episodes': total_episodes,
            'episode_type': episode_type,
            'start_time': datetime.now().isoformat(),
            'steps': []
        }
        
        self.training_data['episodes'].append(episode_data)
        
        self.logger.info(f"=== EPISODIO {episode_num}/{total_episodes} ({episode_type.upper()}) ===")
        return len(self.training_data['episodes']) - 1  # Índice del episodio
    
    def log_step(self, episode_idx, step_num, state, mode, action, reward, next_state, 
                 q_values=None, loss=None, epsilon=None, noise=None):
        """Registrar cada paso del entrenamiento"""
        step_data = {
            'step_num': step_num,
            'state': state.tolist() if isinstance(state, np.ndarray) else state,
            'mode': mode,
            'action': action.tolist() if isinstance(action, np.ndarray) else action,
            'reward': float(reward),
            'next_state': next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
            'q_values': q_values,
            'loss': float(loss) if loss is not None else None,
            'epsilon': epsilon,
            'noise': noise.tolist() if isinstance(noise, np.ndarray) else noise,
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_data['episodes'][episode_idx]['steps'].append(step_data)
        
        # Log detallado
        self.logger.info(f"PASO {step_num}:")
        self.logger.info(f"  Estado: {state}")
        self.logger.info(f"  Modo seleccionado: {mode}")
        self.logger.info(f"  Acción: {action}")
        self.logger.info(f"  Recompensa: {reward}")
        self.logger.info(f"  Estado siguiente: {next_state}")
        if q_values is not None:
            self.logger.info(f"  Q-values: {q_values}")
        if loss is not None:
            self.logger.info(f"  Loss: {loss}")
        if epsilon is not None:
            self.logger.info(f"  Epsilon: {epsilon}")
        if noise is not None:
            self.logger.info(f"  Ruido: {noise}")
    
    def log_mode_selection(self, episode_idx, step_num, all_q_values, selected_mode, selection_reason):
        """Registrar proceso de selección de modo"""
        mode_data = {
            'episode_idx': episode_idx,
            'step_num': step_num,
            'all_q_values': all_q_values,
            'selected_mode': selected_mode,
            'selection_reason': selection_reason,
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_data['mode_selections'].append(mode_data)
        
        self.logger.info(f"SELECCIÓN DE MODO (Ep{episode_idx}, Step{step_num}):")
        self.logger.info(f"  Q-values por modo: {all_q_values}")
        self.logger.info(f"  Modo seleccionado: {selected_mode}")
        self.logger.info(f"  Razón: {selection_reason}")
    
    def log_episode_end(self, episode_idx, total_reward, mode_counts, avg_loss, stability_score):
        """Registrar fin de episodio"""
        episode_data = self.training_data['episodes'][episode_idx]
        episode_data.update({
            'end_time': datetime.now().isoformat(),
            'total_reward': float(total_reward),
            'mode_counts': mode_counts,
            'avg_loss': float(avg_loss) if avg_loss is not None else None,
            'stability_score': float(stability_score) if stability_score is not None else None,
            'total_steps': len(episode_data['steps'])
        })
        
        self.logger.info(f"=== FIN EPISODIO {episode_data['episode_num']} ===")
        self.logger.info(f"  Recompensa total: {total_reward}")
        self.logger.info(f"  Conteo de modos: {mode_counts}")
        self.logger.info(f"  Loss promedio: {avg_loss}")
        self.logger.info(f"  Score de estabilidad: {stability_score}")
        self.logger.info(f"  Pasos totales: {len(episode_data['steps'])}")
    
    def log_training_metrics(self, episode_num, epsilon, episode_rewards, losses, mode_selection_counts):
        """Registrar métricas de entrenamiento"""
        metrics_data = {
            'episode_num': episode_num,
            'epsilon': epsilon,
            'recent_rewards': episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards,
            'recent_losses': losses[-10:] if len(losses) >= 10 else losses,
            'mode_selection_counts': mode_selection_counts,
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_data['stability_metrics'].append(metrics_data)
        
        self.logger.info(f"MÉTRICAS EPISODIO {episode_num}:")
        self.logger.info(f"  Epsilon: {epsilon}")
        self.logger.info(f"  Recompensas recientes: {episode_rewards[-5:] if len(episode_rewards) >= 5 else episode_rewards}")
        self.logger.info(f"  Losses recientes: {losses[-5:] if len(losses) >= 5 else losses}")
        self.logger.info(f"  Distribución de modos: {mode_selection_counts}")
    
    def log_q_value_analysis(self, episode_idx, step_num, q_values_by_mode, selected_mode):
        """Análisis detallado de Q-values"""
        q_analysis = {
            'episode_idx': episode_idx,
            'step_num': step_num,
            'q_values_by_mode': q_values_by_mode,
            'selected_mode': selected_mode,
            'q_value_differences': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Calcular diferencias entre Q-values
        if len(q_values_by_mode) > 1:
            for i in range(len(q_values_by_mode)):
                for j in range(i+1, len(q_values_by_mode)):
                    diff = q_values_by_mode[i] - q_values_by_mode[j]
                    q_analysis['q_value_differences'][f'mode_{i}_vs_mode_{j}'] = float(diff)
        
        self.training_data['q_values'].append(q_analysis)
        
        self.logger.info(f"ANÁLISIS Q-VALUES (Ep{episode_idx}, Step{step_num}):")
        self.logger.info(f"  Q-values por modo: {q_values_by_mode}")
        self.logger.info(f"  Modo seleccionado: {selected_mode}")
        if q_analysis['q_value_differences']:
            self.logger.info(f"  Diferencias Q-values: {q_analysis['q_value_differences']}")
    
    def log_mode_collapse_detection(self, episode_num, mode_counts, threshold=0.8):
        """Detectar y registrar colapso de modos"""
        total_selections = sum(mode_counts.values())
        if total_selections == 0:
            return
        
        mode_ratios = {mode: count/total_selections for mode, count in mode_counts.items()}
        dominant_mode = max(mode_ratios, key=mode_ratios.get)
        dominant_ratio = mode_ratios[dominant_mode]
        
        if dominant_ratio > threshold:
            collapse_data = {
                'episode_num': episode_num,
                'mode_ratios': mode_ratios,
                'dominant_mode': dominant_mode,
                'dominant_ratio': dominant_ratio,
                'threshold': threshold,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.warning(f"⚠️ COLAPSO DE MODOS DETECTADO EN EPISODIO {episode_num}:")
            self.logger.warning(f"  Modo dominante: {dominant_mode} ({dominant_ratio:.2%})")
            self.logger.warning(f"  Distribución completa: {mode_ratios}")
            self.logger.warning(f"  Umbral: {threshold}")
    
    def log_training_end(self, final_results):
        """Registrar fin del entrenamiento"""
        self.training_data['metadata'].update({
            'end_time': datetime.now().isoformat(),
            'final_results': final_results
        })
        
        # Guardar datos en JSON
        with open(self.json_file, 'w') as f:
            json.dump(self.training_data, f, indent=2, default=str)
        
        self.logger.info("=== FIN DE ENTRENAMIENTO ===")
        self.logger.info(f"Datos guardados en: {self.json_file}")
        self.logger.info(f"Log detallado en: {self.log_file}")
        
        # Análisis final
        self._generate_final_analysis()
    
    def _generate_final_analysis(self):
        """Generar análisis final del entrenamiento"""
        total_episodes = len(self.training_data['episodes'])
        total_steps = sum(len(ep['steps']) for ep in self.training_data['episodes'])
        
        # Análisis de modos
        all_mode_selections = []
        for episode in self.training_data['episodes']:
            for step in episode['steps']:
                all_mode_selections.append(step['mode'])
        
        mode_counts = {}
        for mode in all_mode_selections:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        total_selections = len(all_mode_selections)
        mode_ratios = {mode: count/total_selections for mode, count in mode_counts.items()}
        
        # Análisis de recompensas
        all_rewards = [step['reward'] for episode in self.training_data['episodes'] for step in episode['steps']]
        avg_reward = np.mean(all_rewards) if all_rewards else 0
        reward_std = np.std(all_rewards) if all_rewards else 0
        
        # Análisis de losses
        all_losses = [step['loss'] for episode in self.training_data['episodes'] for step in episode['steps'] if step['loss'] is not None]
        avg_loss = np.mean(all_losses) if all_losses else 0
        loss_std = np.std(all_losses) if all_losses else 0
        
        analysis = {
            'total_episodes': total_episodes,
            'total_steps': total_steps,
            'mode_analysis': {
                'total_selections': total_selections,
                'mode_counts': mode_counts,
                'mode_ratios': mode_ratios,
                'mode_collapse_detected': any(ratio > 0.8 for ratio in mode_ratios.values())
            },
            'reward_analysis': {
                'total_rewards': len(all_rewards),
                'average_reward': avg_reward,
                'reward_std': reward_std,
                'min_reward': min(all_rewards) if all_rewards else 0,
                'max_reward': max(all_rewards) if all_rewards else 0
            },
            'loss_analysis': {
                'total_losses': len(all_losses),
                'average_loss': avg_loss,
                'loss_std': loss_std,
                'min_loss': min(all_losses) if all_losses else 0,
                'max_loss': max(all_losses) if all_losses else 0
            }
        }
        
        self.logger.info("=== ANÁLISIS FINAL ===")
        self.logger.info(f"Episodios totales: {total_episodes}")
        self.logger.info(f"Pasos totales: {total_steps}")
        self.logger.info(f"Selecciones de modo: {total_selections}")
        self.logger.info(f"Distribución de modos: {mode_ratios}")
        self.logger.info(f"Colapso de modos detectado: {analysis['mode_analysis']['mode_collapse_detected']}")
        self.logger.info(f"Recompensa promedio: {avg_reward:.4f} ± {reward_std:.4f}")
        self.logger.info(f"Loss promedio: {avg_loss:.6f} ± {loss_std:.6f}")
        
        # Guardar análisis
        analysis_file = self.json_file.replace('.json', '_analysis.json')
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        self.logger.info(f"Análisis guardado en: {analysis_file}")
        
        return analysis 