#!/usr/bin/env python3
"""
HNAF Mejorado Simplificado
Versión que evita problemas de compatibilidad con funciones de transformación
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
from scipy.linalg import expm
from .naf_corrected import CorrectedOptimizationFunctions

class SimpleImprovedNAFNetwork(nn.Module):
    """
    Red neuronal mejorada simplificada
    """
    
    def __init__(self, state_dim=2, action_dim=2, hidden_dim=64, num_layers=3):
        super(SimpleImprovedNAFNetwork, self).__init__()
        
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
        """Inicialización mejorada de pesos"""
        for module in self.layers + [self.value_head, self.mu_head, self.l_head]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
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
        
        # Valores de salida
        V = self.value_head(x)
        mu = torch.tanh(self.mu_head(x)) * 0.1
        
        # Construir matriz L triangular inferior
        l_params = self.l_head(x) * 0.1
        L = torch.zeros(x.size(0), self.action_dim, self.action_dim, device=x.device)
        
        idx = 0
        for i in range(self.action_dim):
            for j in range(i + 1):
                if i == j:
                    L[:, i, j] = torch.exp(torch.clamp(l_params[:, idx], -5, 5))
                else:
                    L[:, i, j] = torch.clamp(l_params[:, idx], -1, 1)
                idx += 1
        
        return V, mu, L
    
    def get_P_matrix(self, x, training=True):
        """Calcula P(x,v) = L(x,v) * L(x,v)^T"""
        V, mu, L = self.forward(x, training)
        P = torch.bmm(L, L.transpose(1, 2))
        return V, mu, P

class SimplePrioritizedReplayBuffer:
    """Buffer de experiencia con priorización simplificado"""
    
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # Prioridad exponencial
        self.beta = beta    # Corrección de sesgo
        self.buffer = []
        self.priorities = []
        self.max_priority = 1.0
        
    def push(self, state, mode, action, reward, next_state, error=None):
        # Calcular prioridad basada en error o prioridad máxima
        if error is not None:
            priority = (abs(error) + 1e-6) ** self.alpha
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
        """Actualizar prioridades basadas en errores"""
        for idx, error in zip(indices, errors):
            if idx < len(self.priorities):
                priority = (abs(error) + 1e-6) ** self.alpha
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

class SimpleImprovedHNAF:
    """
    HNAF mejorado simplificado que evita problemas de compatibilidad
    """
    
    def __init__(self, state_dim=2, action_dim=2, num_modes=2, 
                 hidden_dim=64, num_layers=3, lr=1e-4, tau=0.001, gamma=0.9):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        
        # Estadísticas para normalización
        self.state_mean = np.zeros(state_dim)
        self.state_std = np.ones(state_dim)
        self.reward_mean = 0.0
        self.reward_std = 1.0
        
        # Redes para cada modo
        self.networks = []
        self.target_networks = []
        self.optimizers = []
        
        for mode in range(num_modes):
            network = SimpleImprovedNAFNetwork(state_dim, action_dim, hidden_dim, num_layers)
            target_network = SimpleImprovedNAFNetwork(state_dim, action_dim, hidden_dim, num_layers)
            target_network.load_state_dict(network.state_dict())
            
            self.networks.append(network)
            self.target_networks.append(target_network)
            self.optimizers.append(optim.Adam(network.parameters(), lr=lr))
        
        # Buffer de experiencia priorizado
        self.replay_buffers = [SimplePrioritizedReplayBuffer() for _ in range(num_modes)]
        
        # NAF verifier para recompensas
        self.naf_verifier = CorrectedOptimizationFunctions(t=1.0)
        
        # Matrices de transformación (solo usar matrices, no funciones personalizadas)
        self.transformation_matrices = [
            np.array([[1, 50], [-1, 1]]),
            np.array([[1, -1], [50, 1]])
        ]
    
    def normalize_state(self, state):
        """Normalizar estado"""
        return (state - self.state_mean) / (self.state_std + 1e-8)
    
    def normalize_reward(self, reward):
        """Normalizar recompensa"""
        return (reward - self.reward_mean) / (self.reward_std + 1e-8)
    
    def update_normalization_stats(self, states, rewards):
        """Actualizar estadísticas de normalización"""
        if len(states) > 0:
            states_array = np.array(states)
            self.state_mean = states_array.mean(axis=0)
            self.state_std = states_array.std(axis=0) + 1e-8
        
        if len(rewards) > 0:
            rewards_array = np.array(rewards)
            self.reward_mean = rewards_array.mean()
            self.reward_std = rewards_array.std() + 1e-8
    
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
        
        return mode, action
    
    def compute_Q_value(self, state, action, mode):
        """Calcular Q-value para un estado, acción y modo"""
        state_tensor = torch.FloatTensor(self.normalize_state(state)).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        
        with torch.no_grad():
            V, mu, P = self.networks[mode].get_P_matrix(state_tensor, training=False)
            
            # Calcular Q-value
            Q = V + 0.5 * torch.sum(torch.log(torch.diagonal(P, dim1=1, dim2=2)), dim=1)
            return Q.item()
    
    def update(self, batch_size=32):
        """Actualizar redes con priorización"""
        total_loss = 0
        all_errors = []
        
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
            Q_current = V + 0.5 * torch.sum(torch.log(torch.diagonal(P, dim1=1, dim2=2)), dim=1)
            
            # Calcular Q-values objetivo
            with torch.no_grad():
                V_next, mu_next, P_next = self.target_networks[mode].get_P_matrix(next_states, training=False)
                Q_next = V_next + 0.5 * torch.sum(torch.log(torch.diagonal(P_next, dim1=1, dim2=2)), dim=1)
                Q_target = rewards + self.gamma * Q_next
            
            # Calcular pérdida con pesos de importancia
            loss = F.mse_loss(Q_current, Q_target, reduction='none')
            weighted_loss = (loss * weights).mean()
            
            # Backpropagation
            self.optimizers[mode].zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.networks[mode].parameters(), 1.0)
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
    
    def _get_optimal_mode(self, state):
        """Determinar el modo óptimo usando solo matrices"""
        best_reward = float('inf')
        optimal_mode = 0
        
        for mode in range(self.num_modes):
            A = self.transformation_matrices[mode]
            next_state = A @ state.reshape(-1, 1)
            next_state = next_state.flatten()
            
            # Calcular recompensa usando NAF verifier
            reward = self.naf_verifier.execute_function("reward_function", 
                                                      next_state[0], next_state[1], 
                                                      state[0], state[1])
            
            if abs(reward) < best_reward:
                best_reward = abs(reward)
                optimal_mode = mode
        
        return optimal_mode
    
    def train_episode(self, max_steps=50, epsilon=0.2):
        """Entrenar un episodio con horizonte más largo"""
        # Estado inicial aleatorio
        x0 = np.random.uniform(-0.5, 0.5, self.state_dim)
        state = x0.copy()
        
        total_reward = 0
        episode_data = []
        all_states = []
        all_rewards = []
        
        for step in range(max_steps):
            # Seleccionar acción
            mode, action = self.select_action(state, epsilon)
            
            # Aplicar transformación usando matrices
            A = self.transformation_matrices[mode]
            next_state = A @ state.reshape(-1, 1)
            next_state = next_state.flatten()
            
            # Calcular recompensa base
            reward = self.naf_verifier.execute_function("reward_function", 
                                                      next_state[0], next_state[1], 
                                                      state[0], state[1])
            
            # Reward shaping: recompensa base + bonus por modo óptimo
            reward_base = -abs(reward) / 15.0  # Normalizar
            
            # Determinar modo óptimo para reward shaping
            optimal_mode = self._get_optimal_mode(state)
            mode_bonus = 0.1 if mode == optimal_mode else 0.0
            
            reward_final = reward_base + mode_bonus
            reward_final = np.clip(reward_final, -1, 0)
            
            # Almacenar para normalización
            all_states.append(state.copy())
            all_rewards.append(reward_final)
            
            # Almacenar transición
            self.step(state, mode, action, reward_final, next_state)
            
            episode_data.append({
                'state': state.copy(),
                'mode': mode,
                'action': action.copy(),
                'reward': reward_final,
                'next_state': next_state.copy()
            })
            
            total_reward += reward_final
            state = next_state.copy()
        
        # Actualizar estadísticas de normalización
        self.update_normalization_stats(all_states, all_rewards)
        
        return total_reward, episode_data
    
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
                
                # Aplicar transformación usando matrices
                A = self.transformation_matrices[mode]
                next_state = A @ state.reshape(-1, 1)
                next_state = next_state.flatten()
                
                # Recompensa
                reward = self.naf_verifier.execute_function("reward_function", 
                                                          next_state[0], next_state[1], 
                                                          state[0], state[1])
                
                reward = -abs(reward) / 15.0
                reward = np.clip(reward, -1, 0)
                
                episode_reward += reward
                state = next_state.copy()
            
            total_rewards.append(episode_reward)
            
            # Contar selecciones de modo
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
        print("VERIFICACIÓN HNAF MEJORADO SIMPLIFICADO")
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

def train_simple_improved_hnaf(num_episodes=1000, eval_interval=50, 
                              initial_epsilon=0.5, final_epsilon=0.05,
                              hidden_dim=64, num_layers=3, lr=1e-4):
    """
    Entrenar HNAF mejorado simplificado
    """
    print("🚀 Iniciando entrenamiento HNAF mejorado simplificado...")
    print(f"   - Red: {num_layers} capas de {hidden_dim} unidades")
    print(f"   - ε-greedy decay: {initial_epsilon} -> {final_epsilon}")
    print(f"   - Learning rate: {lr}")
    print(f"   - Prioritized replay con buffer de 10000")
    print(f"   - Normalización de estados y recompensas")
    print(f"   - Reward shaping local")
    print(f"   - Evaluación en grid 100x100")
    print(f"   - Horizonte más largo: 50 pasos")
    
    # Crear modelo
    hnaf = SimpleImprovedHNAF(
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