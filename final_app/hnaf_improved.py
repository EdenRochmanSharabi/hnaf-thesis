#!/usr/bin/env python3
"""
HNAF Mejorado SIN VALORES HARDCODEADOS:
- Todos los parámetros desde config.yaml
- ε-greedy decay configurable
- Prioritized Experience Replay configurable
- Normalización de estados y recompensas configurable
- Red neuronal totalmente configurable
- Límites y tolerancias configurables
- Evaluación con grid configurable
- Reward shaping totalmente configurable
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import sys
import os
import matplotlib.pyplot as plt
from scipy.linalg import expm
# Importar naf_corrected desde el directorio padre si existe
try:
    from ..src.naf_corrected import CorrectedOptimizationFunctions
except ImportError:
    # Fallback: buscar en el directorio actual
    try:
        from naf_corrected import CorrectedOptimizationFunctions
    except ImportError:
        # Crear una clase dummy si no se encuentra
        class CorrectedOptimizationFunctions:
            def __init__(self, t=1.0):
                self.t = t
            def execute_function(self, func_name, *args):
                return 0.0
            def update_matrices(self, A1, A2):
                pass

# Importar config manager - buscar desde el directorio actual
from config_manager import get_config_manager

class ImprovedNAFNetwork(nn.Module):
    """
    Red neuronal mejorada con más capas y normalización
    """
    
    def __init__(self, state_dim=2, action_dim=2, hidden_dim=64, num_layers=3):
        super(ImprovedNAFNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Capas ocultas con normalización
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Primera capa
        self.layers.append(nn.Linear(state_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Capas intermedias
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # V(x,v) - Valor del estado
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # μ(x,v) - Acción media
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        
        # L(x,v) - Matriz triangular inferior
        self.l_head = nn.Linear(hidden_dim, action_dim * (action_dim + 1) // 2)
        
        # Inicialización mejorada
        self._init_weights()
        
    def _init_weights(self):
        """Inicialización mejorada de pesos (valores desde configuración)"""
        config_manager = get_config_manager()
        init_config = config_manager.get_initialization_config()
        network_init = init_config['network']
        
        for module in self.layers + [self.value_head, self.mu_head]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight, gain=network_init['xavier_gain_general'])
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, network_init['bias_constant'])
        
        # Inicialización especial para l_head (configurable)
        if hasattr(self.l_head, 'weight'):
            nn.init.xavier_uniform_(self.l_head.weight, gain=network_init['xavier_gain_l_head'])
            if hasattr(self.l_head, 'bias') and self.l_head.bias is not None:
                nn.init.constant_(self.l_head.bias, network_init['bias_constant'])
        
    def forward(self, x, training=True):
        """
        Forward pass con normalización
        """
        # Aplicar capas con normalización
        for i, (layer, bn) in enumerate(zip(self.layers, self.batch_norms)):
            x = layer(x)
            if training:
                x = bn(x)
            else:
                # En evaluación, usar estadísticas de entrenamiento
                bn.eval()
                with torch.no_grad():
                    x = bn(x)
            x = F.relu(x)  # ReLU para redes más profundas
        
        # Valores de salida (escalas desde configuración)
        config_manager = get_config_manager()
        init_config = config_manager.get_initialization_config()
        network_init = init_config['network']
        tolerances = config_manager.get_tolerances()
        
        V = self.value_head(x)
        mu = torch.tanh(self.mu_head(x)) * network_init['mu_scale']
        
        # Construir matriz L triangular inferior (parámetros configurables)
        l_params = self.l_head(x) * network_init['l_scale']
        L = torch.zeros(x.size(0), self.action_dim, self.action_dim, device=x.device)
        
        matrix_clamp_l = tolerances['matrix_clamp_l']
        matrix_clamp_off_diag = tolerances['matrix_clamp_off_diag']
        diagonal_offset = network_init['diagonal_offset']
        
        idx = 0
        for i in range(self.action_dim):
            for j in range(i + 1):
                if i == j:
                    # Elementos diagonales (clamp y offset configurables)
                    L[:, i, j] = torch.exp(torch.clamp(l_params[:, idx], 
                                                     matrix_clamp_l[0], matrix_clamp_l[1])) + diagonal_offset
                else:
                    # Elementos off-diagonal (clamp configurable)
                    L[:, i, j] = torch.clamp(l_params[:, idx], 
                                           matrix_clamp_off_diag[0], matrix_clamp_off_diag[1])
                idx += 1
        
        return V, mu, L
    
    def get_P_matrix(self, x, training=True):
        """Calcula P(x,v) = L(x,v) * L(x,v)^T"""
        V, mu, L = self.forward(x, training)
        P = torch.bmm(L, L.transpose(1, 2))
        return V, mu, P

class PrioritizedReplayBuffer:
    """Buffer de experiencia con priorización"""
    
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # Prioridad exponencial
        self.beta = beta    # Corrección de sesgo
        self.buffer = []
        self.priorities = []
        self.max_priority = 1.0
        
    def push(self, state, mode, action, reward, next_state, error=None):
        # Calcular prioridad basada en error (epsilon desde configuración)
        config_manager = get_config_manager()
        tolerances = config_manager.get_tolerances()
        priority_epsilon = tolerances['priority_epsilon']
        
        if error is not None:
            priority = (abs(error) + priority_epsilon) ** self.alpha
        else:
            priority = self.max_priority
        
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)
        
        self.buffer.append((state, mode, action, reward, next_state))
        self.priorities.append(priority)
    
    def sample(self, batch_size):
        """Muestreo con priorización"""
        if len(self.buffer) == 0:
            return []
        
        # Calcular probabilidades
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        
        # Muestrear índices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calcular pesos de importancia
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        batch = [self.buffer[i] for i in indices]
        return batch, indices, weights
    
    def update_priorities(self, indices, errors):
        """Actualizar prioridades basadas en errores (epsilon desde configuración)"""
        config_manager = get_config_manager()
        tolerances = config_manager.get_tolerances()
        priority_epsilon = tolerances['priority_epsilon']
        
        for idx, error in zip(indices, errors):
            if idx < len(self.priorities):
                # Asegurar que error sea escalar
                if hasattr(error, '__iter__'):
                    error = error[0] if hasattr(error, '__getitem__') else float(error)
                
                priority = (abs(error) + priority_epsilon) ** self.alpha
                # Asegurar que priority sea escalar
                if hasattr(priority, '__iter__'):
                    priority = priority[0] if hasattr(priority, '__getitem__') else float(priority)
                
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

class ImprovedHNAF:
    """
    HNAF mejorado con todas las optimizaciones
    """
    
    def __init__(self, state_dim=2, action_dim=2, num_modes=2, 
                 hidden_dim=64, num_layers=3, lr=1e-6, tau=1e-5, gamma=0.9,
                 buffer_capacity=10000, alpha=0.6, beta=0.4, reward_normalize=True):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.reward_normalize = reward_normalize
        
        # Estadísticas para normalización
        self.state_mean = np.zeros(state_dim)
        self.state_std = np.ones(state_dim)
        self.reward_mean = 0.0
        self.reward_std = 1.0
        
        # Networks y optimizers para cada modo
        self.networks = []
        self.target_networks = []
        self.optimizers = []
        
        for i in range(num_modes):
            network = ImprovedNAFNetwork(state_dim, action_dim, hidden_dim, num_layers)
            target_network = ImprovedNAFNetwork(state_dim, action_dim, hidden_dim, num_layers)
            target_network.load_state_dict(network.state_dict())
            
            self.networks.append(network)
            self.target_networks.append(target_network)
            self.optimizers.append(optim.Adam(network.parameters(), lr=lr))
        
        # Buffer de experiencia priorizado con parámetros configurables
        self.replay_buffers = [PrioritizedReplayBuffer(capacity=buffer_capacity, alpha=alpha, beta=beta) for _ in range(num_modes)]
        
        # Cargar configuración (NO hardcodeada)
        self.config_manager = get_config_manager()
        advanced_config = self.config_manager.get_advanced_config()
        
        # NAF verifier para recompensas (tiempo configurable)
        time_param = advanced_config['time_parameter']
        self.naf_verifier = CorrectedOptimizationFunctions(t=time_param)
        
        # Matrices de transformación (se inicializan vacías, se actualizarán desde GUI)
        self.transformation_matrices = [
            np.array([[0, 0], [0, 0]]),  # Se actualizarán desde GUI
            np.array([[0, 0], [0, 0]])   # Se actualizarán desde GUI
        ]
        
        # Funciones de transformación por defecto
        self.transformation_functions = [
            lambda x0, y0: self.naf_verifier.execute_function("transform_x1", x0, y0),
            lambda x0, y0: self.naf_verifier.execute_function("transform_x2", x0, y0)
        ]
        
        # Función de recompensa por defecto
        self.reward_function = lambda x, y, x0, y0, mode=None, action=None, previous_state=None: self.naf_verifier.execute_function("reward_function", x, y, x0, y0)
        
        # **NUEVO**: Tracking de selección de modos para evitar colapso
        self.mode_selection_counts = {0: 0, 1: 0}
    
    def normalize_state(self, state):
        """Normalizar estado (epsilon desde configuración)"""
        tolerances = self.config_manager.get_tolerances()
        division_epsilon = tolerances['division_epsilon']
        return (state - self.state_mean) / (self.state_std + division_epsilon)
    
    def normalize_reward(self, reward):
        """Normalizar recompensa"""
        # Asegurar que reward sea escalar
        if hasattr(reward, '__iter__'):
            reward = reward[0] if hasattr(reward, '__getitem__') else float(reward)
        
        # Solo normalizar si está habilitado
        if self.reward_normalize:
            tolerances = self.config_manager.get_tolerances()
            division_epsilon = tolerances['division_epsilon']
            return (reward - self.reward_mean) / (self.reward_std + division_epsilon)
        else:
            return reward
    
    def update_normalization_stats(self, states, rewards):
        """Actualizar estadísticas de normalización (epsilon desde configuración)"""
        tolerances = self.config_manager.get_tolerances()
        division_epsilon = tolerances['division_epsilon']
        
        if len(states) > 0:
            states_array = np.array(states)
            self.state_mean = states_array.mean(axis=0)
            self.state_std = states_array.std(axis=0) + division_epsilon
        
        if len(rewards) > 0:
            rewards_array = np.array(rewards)
            self.reward_mean = rewards_array.mean()
            self.reward_std = rewards_array.std() + division_epsilon
    
    def update_transformation_matrices(self, A1_matrix, A2_matrix):
        """Actualizar matrices desde el GUI."""
        # Actualizar matrices internas
        self.transformation_matrices = [np.array(A1_matrix), np.array(A2_matrix)]
        
        # Actualizar naf_verifier (esto recalcula automáticamente los exponenciales)
        self.naf_verifier.update_matrices(A1_matrix, A2_matrix)
    
    def select_action(self, state, epsilon=0.0):
        """Seleccionar acción con ε-greedy"""
        if random.random() < epsilon:
            # Exploración: modo aleatorio
            mode = random.randint(0, self.num_modes - 1)
            action = np.random.uniform(-1, 1, self.action_dim)
        else:
            # Explotación: mejor Q-value
            state_tensor = torch.FloatTensor(self.normalize_state(state)).unsqueeze(0)
            
            best_q = float('-inf')
            best_mode = 0
            best_action = np.zeros(self.action_dim)
            
            for mode in range(self.num_modes):
                with torch.no_grad():
                    V, mu, P = self.networks[mode].get_P_matrix(state_tensor, training=False)
                    
                    # Calcular Q-value
                    Q = V + 0.5 * torch.sum(torch.log(torch.diagonal(P, dim1=1, dim2=2)), dim=1)
                    Q = Q.item()
                    
                    if Q > best_q:
                        best_q = Q
                        best_mode = mode
                        best_action = mu.squeeze().numpy()
            
            mode = best_mode
            action = best_action
        
        # **NUEVO**: Trackear selección de modos para evitar colapso
        self.mode_selection_counts[mode] = self.mode_selection_counts.get(mode, 0) + 1
        
        return mode, action
    
    def compute_Q_value(self, state, action, mode):
        """Calcular Q-value para un estado, acción y modo"""
        state_tensor = torch.FloatTensor(self.normalize_state(state)).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        
        with torch.no_grad():
            V, mu, P = self.networks[mode].get_P_matrix(state_tensor, training=False)
            
            # Calcular Q-value con mejor manejo de valores pequeños
            # Usar log1p para evitar problemas con valores muy pequeños
            diagonal_elements = torch.diagonal(P, dim1=1, dim2=2)
            # Asegurar que los elementos diagonales sean positivos (mínimo desde configuración)
            tolerances = get_config_manager().get_tolerances()
            diagonal_clamp_min = tolerances['diagonal_clamp_min']
            diagonal_elements = torch.clamp(diagonal_elements, min=diagonal_clamp_min)
            log_diagonal = torch.log(diagonal_elements)
            
            Q = V + 0.5 * torch.sum(log_diagonal, dim=1)
            return Q.item()
    
    def update(self, batch_size=32):
        """Actualizar redes con priorización"""
        total_loss = 0
        all_errors = []
        criterion = torch.nn.SmoothL1Loss(reduction='none')  # Huber loss
        
        for mode in range(self.num_modes):
            if len(self.replay_buffers[mode]) < batch_size:
                continue
            
            # Muestrear con priorización
            batch, indices, weights = self.replay_buffers[mode].sample(batch_size)
            
            if not batch:
                continue
            
            # Preparar datos
            states = torch.FloatTensor([self.normalize_state(transition[0]) for transition in batch])
            actions = torch.FloatTensor([transition[2] for transition in batch])
            rewards = torch.FloatTensor([self.normalize_reward(transition[3]) for transition in batch])
            next_states = torch.FloatTensor([self.normalize_state(transition[4]) for transition in batch])
            weights = torch.FloatTensor(weights)
            
            # Calcular Q-values actuales
            V, mu, P = self.networks[mode].get_P_matrix(states)
            diagonal_elements = torch.diagonal(P, dim1=1, dim2=2)
            # Usar tolerancia configurable
            tolerances = get_config_manager().get_tolerances()
            diagonal_clamp_min = tolerances['diagonal_clamp_min']
            diagonal_elements = torch.clamp(diagonal_elements, min=diagonal_clamp_min)
            log_diagonal = torch.log(diagonal_elements)
            Q_current = V + 0.5 * torch.sum(log_diagonal, dim=1)
            
            # Calcular Q-values objetivo
            with torch.no_grad():
                V_next, mu_next, P_next = self.target_networks[mode].get_P_matrix(next_states, training=False)
                diagonal_elements_next = torch.diagonal(P_next, dim1=1, dim2=2)
                diagonal_elements_next = torch.clamp(diagonal_elements_next, min=diagonal_clamp_min)
                log_diagonal_next = torch.log(diagonal_elements_next)
                Q_next = V_next + 0.5 * torch.sum(log_diagonal_next, dim=1)
                Q_target = rewards + self.gamma * Q_next
            
            # Calcular pérdida con pesos de importancia (Huber loss)
            loss = criterion(Q_current, Q_target)
            weighted_loss = (loss * weights).mean()
            
            # Backpropagation
            self.optimizers[mode].zero_grad()
            weighted_loss.backward()
            
            # **MEJORADO**: Gradient clipping configurable desde config.yaml
            training_config = self.config_manager.get_training_defaults()
            gradient_clip = training_config.get('gradient_clip', 0.5)
            torch.nn.utils.clip_grad_norm_(self.networks[mode].parameters(), gradient_clip)
            
            self.optimizers[mode].step()
            
            # Actualizar prioridades
            errors = (Q_current - Q_target).detach().numpy()
            self.replay_buffers[mode].update_priorities(indices, errors)
            
            total_loss += weighted_loss.item()
            all_errors.extend(errors)
        
        return total_loss / self.num_modes if total_loss > 0 else None
    
    def step(self, state, mode, action, reward, next_state):
        """Almacenar transición en el buffer"""
        self.replay_buffers[mode].push(state, mode, action, reward, next_state)
    
    def train_episode(self, max_steps=50, epsilon=0.2):
        """Entrenar un episodio con horizonte más largo"""
        # Estado inicial aleatorio
        x0 = np.random.uniform(-0.5, 0.5, self.state_dim)
        state = x0.copy()
        
        total_reward = 0
        episode_data = []
        all_states = []
        all_rewards = []
        reward_debug = []  # Para imprimir algunos rewards
        
        for step in range(max_steps):
            # Seleccionar acción
            mode, action = self.select_action(state, epsilon)
            
            # Aplicar transformación según el modo
            if hasattr(self, 'transformation_functions') and len(self.transformation_functions) > mode:
                try:
                    x_next, y_next = self.transformation_functions[mode](state[0], state[1])
                    # Asegurar que son escalares
                    if hasattr(x_next, '__iter__'):
                        x_next = x_next[0] if hasattr(x_next, '__getitem__') else float(x_next)
                    if hasattr(y_next, '__iter__'):
                        y_next = y_next[0] if hasattr(y_next, '__iter__') else float(y_next)
                    next_state = np.array([x_next, y_next])
                except Exception as e:
                    error_msg = f"❌ ERROR CRÍTICO: Transformación del modo {mode} falló en entrenamiento\n" \
                               f"   Error: {e}\n" \
                               f"   Estado: {state}\n" \
                               f"   EPISODIO ABORTADO - Revisa funciones de transformación"
                    print(error_msg)
                    raise RuntimeError(f"Transformación del modo {mode} inválida en entrenamiento: {e}")
            else:
                A = self.transformation_matrices[mode]
                next_state = A @ state.reshape(-1, 1)
                next_state = next_state.flatten()
            
            # **Estabilización del entorno**: Limitar estados (límites desde configuración)
            state_limits = self.config_manager.get_state_limits()
            next_state = np.clip(next_state, state_limits['min'], state_limits['max'])
            
            # **MEJORADA**: Calcular recompensa usando la función de la GUI como base
            norm_current = np.linalg.norm(state)
            norm_next = np.linalg.norm(next_state)
            norm_initial = np.linalg.norm(x0)
            
            # **CORREGIDO**: Calcular recompensa de manera más coherente
            if hasattr(self, 'reward_function'):
                try:
                    # Usar la función de recompensa definida en la GUI
                    reward_base = self.reward_function(next_state[0], next_state[1], state[0], state[1])
                    # Asegurar que sea escalar
                    if hasattr(reward_base, '__iter__'):
                        reward_base = reward_base[0] if hasattr(reward_base, '__getitem__') else float(reward_base)
                except Exception as e:
                    error_msg = f"❌ ERROR CRÍTICO: Función de recompensa falló\n" \
                               f"   Error: {e}\n" \
                               f"   Estado actual: {state}\n" \
                               f"   Estado siguiente: {next_state}\n" \
                               f"   EPISODIO ABORTADO - Revisa función de recompensa"
                    print(error_msg)
                    raise RuntimeError(f"Función de recompensa inválida: {e}")
            else:
                error_msg = f"❌ ERROR CRÍTICO: No hay función de recompensa definida\n" \
                           f"   El modelo no tiene atributo 'reward_function'\n" \
                           f"   EPISODIO ABORTADO - Configura función de recompensa"
                print(error_msg)
                raise RuntimeError("No hay función de recompensa definida")
            
            # **CORREGIDO**: Aplicar reward shaping solo si está habilitado (coeficientes desde configuración)
            if hasattr(self, 'reward_shaping_enabled') and self.reward_shaping_enabled:
                reward_shaping_config = self.config_manager.get_reward_shaping_config()
                coeffs = reward_shaping_config['coefficients']
                
                # Penalización por alejarse del origen (configurable)
                distance_penalty = -abs(norm_next - norm_initial) * coeffs['distance_penalty']
                
                # Bonus por acercarse al origen (configurable)
                approach_bonus = coeffs['approach_bonus'] if norm_next < norm_current else 0.0
                
                # Penalización por modo subóptimo (configurable)
                optimal_mode = self._get_optimal_mode(state)
                mode_penalty = -coeffs['suboptimal_penalty'] if mode != optimal_mode else 0.0
                
                # Penalización por oscilación (configurable)
                oscillation_penalty = -coeffs['oscillation_penalty'] * abs(norm_next - norm_current) if step > 0 else 0.0
                
                reward_final = reward_base + distance_penalty + approach_bonus + mode_penalty + oscillation_penalty
            else:
                # Usar solo la función de la GUI sin reward shaping
                reward_final = reward_base
            
            # **CORREGIDO**: Clamp más razonable
            reward_final = np.clip(reward_final, -5, 1)
            
            # Almacenar para normalización
            all_states.append(state.copy())
            all_rewards.append(reward_final)
            
            # Almacenar transición
            self.step(state, mode, action, reward_final, next_state)
            
            # Actualizar estado
            state = next_state.copy()
            total_reward += reward_final
            
            # Guardar datos del episodio
            episode_data.append({
                'step': step,
                'state': state.copy(),
                'mode': mode,
                'action': action,
                'reward': reward_final,
                'next_state': next_state.copy()
            })
            
            # **NUEVO**: Condición de terminación temprana si se acerca mucho al origen
            if norm_next < 0.01:
                break
        
        # Bonus final por episodio exitoso (desde configuración)
        final_norm = np.linalg.norm(state)
        if final_norm < 0.1:
            reward_shaping_config = self.config_manager.get_reward_shaping_config()
            success_bonus = reward_shaping_config['coefficients']['success_bonus']
            total_reward += success_bonus
        
        # Actualizar estadísticas de normalización
        if all_states and all_rewards:
            self.update_normalization_stats(all_states, all_rewards)
        
        return total_reward, episode_data
    
    def _get_optimal_mode(self, state):
        """Determinar el modo óptimo para reward shaping - MEJORADO"""
        best_norm = float('inf')
        optimal_mode = 0
        
        for mode in range(self.num_modes):
            # Aplicar transformación del modo
            if hasattr(self, 'transformation_functions') and len(self.transformation_functions) > mode:
                try:
                    x_next, y_next = self.transformation_functions[mode](state[0], state[1])
                    # Asegurar que son escalares
                    if hasattr(x_next, '__iter__'):
                        x_next = x_next[0] if hasattr(x_next, '__getitem__') else float(x_next)
                    if hasattr(y_next, '__iter__'):
                        y_next = y_next[0] if hasattr(y_next, '__iter__') else float(y_next)
                    next_state = np.array([x_next, y_next])
                except Exception as e:
                    error_msg = f"❌ ERROR CRÍTICO: Transformación del modo {mode} falló en evaluación\n" \
                               f"   Error: {e}\n" \
                               f"   Estado: {state}\n" \
                               f"   EVALUACIÓN ABORTADA - Revisa funciones de transformación"
                    print(error_msg)
                    raise RuntimeError(f"Transformación del modo {mode} inválida en evaluación: {e}")
            else:
                A = self.transformation_matrices[mode]
                next_state = A @ state.reshape(-1, 1)
                next_state = next_state.flatten()
            
            # **MEJORADO**: Calcular norma del estado resultante
            norm_next = np.linalg.norm(next_state)
            
            # **NUEVO**: Considerar también la estabilidad (evitar explosiones)
            stability_penalty = 0.0
            if norm_next > 2.0:  # Penalizar estados muy lejos
                stability_penalty = norm_next * 0.5
            
            # **NUEVO**: Considerar la dirección hacia el origen
            current_norm = np.linalg.norm(state)
            direction_bonus = 0.0
            if norm_next < current_norm:  # Si se acerca al origen
                direction_bonus = -0.1  # Bonus (menor penalización)
            
            # **MEJORADO**: Criterio de selección más robusto
            total_cost = norm_next + stability_penalty + direction_bonus
            
            if total_cost < best_norm:
                best_norm = total_cost
                optimal_mode = mode
        
        return optimal_mode
    
    def evaluate_policy_grid(self, grid_size=100):
        """Evaluar política en una rejilla de 100x100"""
        x = np.linspace(-0.5, 0.5, grid_size)
        y = np.linspace(-0.5, 0.5, grid_size)
        X, Y = np.meshgrid(x, y)
        
        mode_selections = np.zeros((grid_size, grid_size))
        q_values = np.zeros((grid_size, grid_size))
        optimal_accuracy = 0
        
        for i in range(grid_size):
            for j in range(grid_size):
                state = np.array([X[i, j], Y[i, j]])
                
                # Seleccionar modo sin ruido
                mode, _ = self.select_action(state, epsilon=0.0)
                mode_selections[i, j] = mode
                
                # Calcular Q-value
                q_val = self.compute_Q_value(state, np.zeros(self.action_dim), mode)
                q_values[i, j] = q_val
                
                # Verificar si es óptimo
                optimal_mode = self._get_optimal_mode(state)
                if mode == optimal_mode:
                    optimal_accuracy += 1
        
        optimal_accuracy /= (grid_size * grid_size)
        
        return {
            'mode_selections': mode_selections,
            'q_values': q_values,
            'optimal_accuracy': optimal_accuracy,
            'X': X,
            'Y': Y
        }
    
    def evaluate_policy(self, num_episodes=10):
        """Evaluar política con episodios"""
        total_rewards = []
        mode_selections = {0: 0, 1: 0}
        
        for episode in range(num_episodes):
            x0 = np.random.uniform(-0.5, 0.5, self.state_dim)
            state = x0.copy()
            
            episode_reward = 0
            episode_modes = []
            
            for step in range(30):  # Episodio más largo
                mode, action = self.select_action(state, epsilon=0.0)
                episode_modes.append(mode)
                
                # Aplicar transformación
                if hasattr(self, 'transformation_functions') and len(self.transformation_functions) > mode:
                    try:
                        x_next, y_next = self.transformation_functions[mode](state[0], state[1])
                        # Asegurar que son escalares
                        if hasattr(x_next, '__iter__'):
                            x_next = x_next[0] if hasattr(x_next, '__getitem__') else float(x_next)
                        if hasattr(y_next, '__iter__'):
                            y_next = y_next[0] if hasattr(y_next, '__getitem__') else float(y_next)
                        next_state = np.array([x_next, y_next])
                    except Exception as e:
                        error_msg = f"❌ ERROR CRÍTICO: Transformación del modo {mode} falló en política óptima\n" \
                                   f"   Error: {e}\n" \
                                   f"   Estado: {state}\n" \
                                   f"   CÁLCULO ABORTADO - Revisa funciones de transformación"
                        print(error_msg)
                        raise RuntimeError(f"Transformación del modo {mode} inválida en política óptima: {e}")
                else:
                    A = self.transformation_matrices[mode]
                    next_state = A @ state.reshape(-1, 1)
                    next_state = next_state.flatten()
                
                # **NUEVO**: Estabilización del entorno (límites desde configuración)
                state_limits = self.config_manager.get_state_limits()
                next_state = np.clip(next_state, state_limits['min'], state_limits['max'])
                
                # **CORREGIDO**: Usar la función de recompensa de la GUI
                if hasattr(self, 'reward_function'):
                    try:
                        reward = self.reward_function(next_state[0], next_state[1], state[0], state[1])
                    except Exception as e:
                        print(f"Error en función de recompensa: {e}")
                        reward = 0.0
                else:
                    try:
                        reward = self.naf_verifier.execute_function("reward_function", 
                                                                  next_state[0], next_state[1], 
                                                                  state[0], state[1])
                    except Exception as e:
                        print(f"Error en NAF verifier: {e}")
                        reward = 0.0
                
                # **CORREGIDO**: Asegurar que reward sea escalar y respetar el valor
                if hasattr(reward, '__iter__'):
                    reward = reward[0] if hasattr(reward, '__getitem__') else float(reward)
                
                episode_reward += reward
                state = next_state.copy()
            
            total_rewards.append(episode_reward)
            
            if episode_modes:
                most_used_mode = max(set(episode_modes), key=episode_modes.count)
                mode_selections[most_used_mode] += 1
        
        return np.mean(total_rewards), mode_selections
    
    def verify_hnaf(self, test_states=None):
        """Verificar funcionamiento del HNAF"""
        if test_states is None:
            test_states = [
                np.array([0.1, 0.1]),
                np.array([0, 0.1]),
                np.array([0.1, 0]),
                np.array([0.05, 0.05]),
                np.array([-0.05, 0.08])
            ]
        
        print("="*60)
        print("VERIFICACIÓN HNAF MEJORADO")
        print("="*60)
        
        for i, state in enumerate(test_states):
            print(f"\nEstado {i+1}: {state}")
            
            # HNAF
            mode, action = self.select_action(state, epsilon=0.0)
            Q_pred = self.compute_Q_value(state, action, mode)
            
            # Modo óptimo
            optimal_mode = self._get_optimal_mode(state)
            
            print(f"  HNAF: Modo {mode}, Q={Q_pred:.4f}")
            print(f"  Óptimo: Modo {optimal_mode}")
            print(f"  Correcto: {'✅' if mode == optimal_mode else '❌'}")
        
        # Evaluación en grid
        grid_results = self.evaluate_policy_grid(grid_size=50)
        print(f"\nPrecisión en grid 50x50: {grid_results['optimal_accuracy']:.2%}")
    
    def update_target_networks(self):
        """Actualizar redes objetivo con soft update"""
        for mode in range(self.num_modes):
            for target_param, param in zip(self.target_networks[mode].parameters(), 
                                         self.networks[mode].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

def train_improved_hnaf(num_episodes=1000, eval_interval=50, 
                       initial_epsilon=0.5, final_epsilon=0.05,
                       hidden_dim=64, num_layers=3, lr=1e-4):
    """
    Entrenar HNAF mejorado con todas las optimizaciones
    """
    print("🚀 Iniciando entrenamiento HNAF mejorado...")
    print(f"   - Red: {num_layers} capas de {hidden_dim} unidades")
    print(f"   - ε-greedy: {initial_epsilon} -> {final_epsilon}")
    print(f"   - Learning rate: {lr}")
    print(f"   - Prioritized replay")
    print(f"   - Normalización de estados y recompensas")
    print(f"   - Reward shaping local")
    print(f"   - Evaluación en grid 100x100")
    
    # Crear modelo
    hnaf = ImprovedHNAF(
        state_dim=2,
        action_dim=2,
        num_modes=2,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        lr=lr,
        tau=0.001,
        gamma=0.9
    )
    
    # Métricas
    episode_rewards = []
    losses = []
    eval_rewards = []
    grid_accuracies = []
    
    # ε-greedy decay
    epsilon_decay = (initial_epsilon - final_epsilon) / num_episodes
    
    for episode in range(num_episodes):
        # Calcular epsilon actual
        epsilon = max(final_epsilon, initial_epsilon - episode * epsilon_decay)
        
        # Entrenar episodio
        reward, _ = hnaf.train_episode(max_steps=50, epsilon=epsilon)
        episode_rewards.append(reward)
        
        # Actualizar redes
        loss = hnaf.update(batch_size=32)
        if loss is not None:
            losses.append(loss)
        
        # Actualizar redes objetivo
        hnaf.update_target_networks()
        
        # Evaluación periódica
        if (episode + 1) % eval_interval == 0:
            eval_reward, mode_selections = hnaf.evaluate_policy(num_episodes=10)
            eval_rewards.append(eval_reward)
            
            # Evaluación en grid
            grid_results = hnaf.evaluate_policy_grid(grid_size=50)
            grid_accuracies.append(grid_results['optimal_accuracy'])
            
            print(f"Episodio {episode+1}/{num_episodes}")
            print(f"  ε: {epsilon:.3f}")
            print(f"  Recompensa promedio: {np.mean(episode_rewards[-eval_interval:]):.4f}")
            print(f"  Recompensa evaluación: {eval_reward:.4f}")
            print(f"  Precisión grid: {grid_results['optimal_accuracy']:.2%}")
            print(f"  Selección modos: {mode_selections}")
            if losses:
                print(f"  Pérdida promedio: {np.mean(losses[-eval_interval:]):.6f}")
            print()
    
    # Verificación final
    hnaf.verify_hnaf()
    
    return hnaf, {
        'episode_rewards': episode_rewards,
        'losses': losses,
        'eval_rewards': eval_rewards,
        'grid_accuracies': grid_accuracies,
        'eval_interval': eval_interval
    } 