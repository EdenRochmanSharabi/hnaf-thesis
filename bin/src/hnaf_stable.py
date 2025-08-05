#!/usr/bin/env python3
"""
Versión estable de HNAF (Hybrid Normalized Advantage Function)
Con mejor estabilidad numérica y debugging.
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

class StableNAFNetwork(nn.Module):
    """
    Red neuronal estable para un modo discreto específico.
    Salida: V(x,v), μ(x,v), L(x,v)
    """
    
    def __init__(self, state_dim=2, action_dim=2, hidden_dim=32):
        super(StableNAFNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Capas compartidas más pequeñas para estabilidad
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # V(x,v) - Valor del estado
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # μ(x,v) - Acción media
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        
        # L(x,v) - Matriz triangular inferior (diagonal exponencial)
        self.l_head = nn.Linear(hidden_dim, action_dim * (action_dim + 1) // 2)
        
        # Inicialización más estable
        self._init_weights()
        
    def _init_weights(self):
        """Inicialización estable de pesos"""
        for module in [self.fc1, self.fc2, self.value_head, self.mu_head, self.l_head]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
    def forward(self, x):
        """
        Forward pass: x -> (V, μ, L)
        """
        # Capas compartidas con activación más estable
        x = torch.tanh(self.fc1(x))  # tanh en lugar de ReLU para estabilidad
        x = torch.tanh(self.fc2(x))
        
        # Valores de salida
        V = self.value_head(x)
        mu = torch.tanh(self.mu_head(x)) * 0.1  # Escalar y limitar acciones
        
        # Construir matriz L triangular inferior con valores más pequeños
        l_params = self.l_head(x) * 0.1  # Escalar parámetros
        L = torch.zeros(x.size(0), self.action_dim, self.action_dim, device=x.device)
        
        # Llenar matriz triangular inferior con diagonal exponencial
        idx = 0
        for i in range(self.action_dim):
            for j in range(i + 1):
                if i == j:
                    # Diagonal: exponencial limitada
                    L[:, i, j] = torch.exp(torch.clamp(l_params[:, idx], -5, 5))
                else:
                    # Subdiagonal: valor directo limitado
                    L[:, i, j] = torch.clamp(l_params[:, idx], -1, 1)
                idx += 1
        
        return V, mu, L
    
    def get_P_matrix(self, x):
        """
        Calcula P(x,v) = L(x,v) * L(x,v)^T
        """
        V, mu, L = self.forward(x)
        P = torch.bmm(L, L.transpose(1, 2))
        return V, mu, P

class StableReplayBuffer:
    """Buffer de experiencia estable para un modo específico."""
    
    def __init__(self, capacity=5000):  # Buffer más grande para más datos
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, mode, action, reward, next_state):
        # Verificar que los datos son válidos
        if not (np.isfinite(state).all() and np.isfinite(action).all() and 
                np.isfinite(reward) and np.isfinite(next_state).all()):
            return  # Ignorar datos inválidos
        
        self.buffer.append((state, mode, action, reward, next_state))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        state, mode, action, reward, next_state = zip(*batch)
        
        return (torch.FloatTensor(np.array(state)),
                torch.LongTensor(mode),
                torch.FloatTensor(np.array(action)),
                torch.FloatTensor(reward),
                torch.FloatTensor(np.array(next_state)))
    
    def __len__(self):
        return len(self.buffer)

class StableHNAF:
    """
    Hybrid Normalized Advantage Function con estabilidad numérica
    """
    
    def __init__(self, state_dim=2, action_dim=2, num_modes=2, 
                 hidden_dim=32, lr=1e-4, tau=0.001, gamma=0.9):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_modes = num_modes
        self.gamma = gamma  # Cambiado a 0.9 como recomendado
        self.tau = tau
        
        # Redes para cada modo
        self.networks = {}
        self.target_networks = {}
        self.optimizers = {}
        self.replay_buffers = {}
        
        for mode in range(num_modes):
            # Red principal
            self.networks[mode] = StableNAFNetwork(state_dim, action_dim, hidden_dim)
            # Red target
            self.target_networks[mode] = StableNAFNetwork(state_dim, action_dim, hidden_dim)
            self.target_networks[mode].load_state_dict(self.networks[mode].state_dict())
            # Optimizador con learning rate más pequeño
            self.optimizers[mode] = optim.Adam(self.networks[mode].parameters(), lr=lr, weight_decay=1e-5)
            # Buffer de replay
            self.replay_buffers[mode] = StableReplayBuffer()
        
        # Matrices de transformación (modos discretos)
        self.A1 = np.array([[1, 50], [-1, 1]])
        self.A2 = np.array([[1, -1], [50, 1]])
        self.transformation_matrices = {0: self.A1, 1: self.A2}
        
        # NAF corregido para verificación
        self.naf_verifier = CorrectedOptimizationFunctions(t=1.0)
    
    def select_action(self, state, epsilon=0.0):
        """
        Selecciona acción híbrida (modo, acción_continua)
        Con exploración forzada de ambos modos
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Calcular V(x,v) para cada modo
        values = {}
        mus = {}
        
        for mode in range(self.num_modes):
            with torch.no_grad():
                V, mu, _ = self.networks[mode](state_tensor)
                values[mode] = V.item()
                mus[mode] = mu.squeeze().numpy()
        
        # Exploración ε-greedy sobre selección discreta v
        if epsilon > 0 and random.random() < epsilon:
            # Exploración: seleccionar modo aleatorio
            selected_mode = random.randint(0, self.num_modes - 1)
        else:
            # Explotación: seleccionar modo óptimo
            selected_mode = min(values.keys(), key=lambda v: values[v])
        
        # Seleccionar acción continua: u = μ(x, v*)
        selected_action = mus[selected_mode]
        
        # Añadir ruido pequeño a la acción continua
        if epsilon > 0 and random.random() < epsilon:
            selected_action += np.random.normal(0, 0.01, self.action_dim)
        
        return selected_mode, selected_action
    
    def compute_Q_value(self, state, action, mode):
        """
        Calcula Q(x,v,u) = V(x,v) + A(x,v,u)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        
        with torch.no_grad():
            V, mu, P = self.networks[mode].get_P_matrix(state_tensor)
            
            # Calcular ventaja con clipping para estabilidad
            diff = action_tensor - mu
            diff = torch.clamp(diff, -10, 10)  # Clipping para evitar valores extremos
            
            advantage = -0.5 * torch.bmm(torch.bmm(diff.unsqueeze(1), P), diff.unsqueeze(2))
            advantage = torch.clamp(advantage, -100, 100)  # Clipping adicional
            
            # Q-value
            Q = V + advantage
            
            return Q.item()
    
    def update(self, batch_size=32):
        """
        Actualiza todas las redes con mejor estabilidad y batch más grande
        """
        total_loss = 0
        num_updates = 0
        
        for mode in range(self.num_modes):
            batch_data = self.replay_buffers[mode].sample(batch_size)
            if batch_data is None:
                continue
                
            states, modes, actions, rewards, next_states = batch_data
            
            # Clipping de recompensas para estabilidad (ya normalizadas)
            rewards = torch.clamp(rewards, -1, 0)
            
            # Calcular target Q-value
            target_Q = torch.zeros(batch_size, 1)
            
            for i in range(batch_size):
                # Encontrar mínimo V(x',v') para todos los modos
                min_next_value = float('inf')
                for next_mode in range(self.num_modes):
                    with torch.no_grad():
                        V_next, _, _ = self.target_networks[next_mode](next_states[i:i+1])
                        min_next_value = min(min_next_value, V_next.item())
                
                target_Q[i] = rewards[i] + self.gamma * min_next_value
            
            # Clipping del target (recompensas normalizadas)
            target_Q = torch.clamp(target_Q, -10, 0)
            
            # Calcular Q-value actual
            current_Q = torch.zeros(batch_size, 1)
            for i in range(batch_size):
                V, mu, P = self.networks[mode](states[i:i+1])
                diff = actions[i:i+1] - mu
                diff = torch.clamp(diff, -10, 10)
                
                advantage = -0.5 * torch.bmm(torch.bmm(diff.unsqueeze(1), P), diff.unsqueeze(2))
                advantage = torch.clamp(advantage, -100, 100)
                
                current_Q[i] = V + advantage
            
            # Clipping del Q actual (recompensas normalizadas)
            current_Q = torch.clamp(current_Q, -10, 0)
            
            # Calcular pérdida y actualizar
            loss = F.mse_loss(current_Q, target_Q)
            
            # Verificar que la pérdida es finita
            if not torch.isfinite(loss):
                continue
            
            self.optimizers[mode].zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.networks[mode].parameters(), max_norm=1.0)
            
            self.optimizers[mode].step()
            
            total_loss += loss.item()
            num_updates += 1
            
            # Soft update del target network
            for target_param, param in zip(self.target_networks[mode].parameters(), 
                                         self.networks[mode].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return total_loss / max(num_updates, 1)
    
    def step(self, state, mode, action, reward, next_state):
        """
        Almacena transición
        """
        self.replay_buffers[mode].push(state, mode, action, reward, next_state)
    
    def train_episode(self, max_steps=20, epsilon=0.2):
        """
        Entrena un episodio completo con exploración aumentada
        """
        # Estado inicial aleatorio más pequeño
        x0 = np.random.uniform(-0.5, 0.5, self.state_dim)
        state = x0.copy()
        
        total_reward = 0
        episode_data = []
        
        for step in range(max_steps):
            # Seleccionar acción
            mode, action = self.select_action(state, epsilon)
            
            # Aplicar transformación según el modo
            A = self.transformation_matrices[mode]
            next_state = A @ state.reshape(-1, 1)
            next_state = next_state.flatten()
            
            # Calcular recompensa usando NAF corregido
            reward = self.naf_verifier.execute_function("reward_function", 
                                                      next_state[0], next_state[1], 
                                                      state[0], state[1])
            
            # Reescalar recompensa: reward = -abs(||x'|| - ||x₀||) / 15
            # Esto hace que el agente "maximice" acercarse al origen
            reward = -abs(reward) / 15.0  # Normalizar a r ∈ [-1, 0]
            
            # Clipping de recompensa
            reward = np.clip(reward, -1, 0)
            
            # Almacenar transición
            self.step(state, mode, action, reward, next_state)
            
            episode_data.append({
                'state': state.copy(),
                'mode': mode,
                'action': action.copy(),
                'reward': reward,
                'next_state': next_state.copy()
            })
            
            total_reward += reward
            state = next_state.copy()
        
        return total_reward, episode_data
    
    def evaluate_policy(self, num_episodes=10):
        """
        Evalúa la política aprendida
        """
        total_rewards = []
        mode_selections = {0: 0, 1: 0}
        
        for episode in range(num_episodes):
            # Estado inicial aleatorio
            x0 = np.random.uniform(-0.5, 0.5, self.state_dim)
            state = x0.copy()
            
            episode_reward = 0
            episode_modes = []
            
            for step in range(20):  # Episodio más corto
                # Seleccionar acción sin ruido
                mode, action = self.select_action(state, epsilon=0.0)
                episode_modes.append(mode)
                
                # Aplicar transformación
                A = self.transformation_matrices[mode]
                next_state = A @ state.reshape(-1, 1)
                next_state = next_state.flatten()
                
                # Recompensa
                reward = self.naf_verifier.execute_function("reward_function", 
                                                          next_state[0], next_state[1], 
                                                          state[0], state[1])
                # Reescalar recompensa igual que en entrenamiento
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
        """
        Verifica que el HNAF funciona correctamente
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
        print("VERIFICACIÓN HNAF ESTABLE")
        print("="*60)
        
        for i, x0 in enumerate(test_states):
            print(f"\nCaso {i+1}: Estado inicial {x0}")
            
            # HNAF: seleccionar modo y acción
            selected_mode, selected_action = self.select_action(x0, epsilon=0.0)
            Q_pred = self.compute_Q_value(x0, selected_action, selected_mode)
            
            # Aplicar transformación del modo seleccionado
            A = self.transformation_matrices[selected_mode]
            x_next = A @ x0.reshape(-1, 1)
            x_next = x_next.flatten()
            
            # Recompensa real
            r_real = self.naf_verifier.execute_function("reward_function", 
                                                      x_next[0], x_next[1], 
                                                      x0[0], x0[1])
            
            # NAF individual para comparar
            x1_naf = self.naf_verifier.execute_function("transform_x1", x0[0], x0[1])
            x2_naf = self.naf_verifier.execute_function("transform_x2", x0[0], x0[1])
            r1_naf = self.naf_verifier.execute_function("reward_function", x1_naf[0, 0], x1_naf[1, 0], x0[0], x0[1])
            r2_naf = self.naf_verifier.execute_function("reward_function", x2_naf[0, 0], x2_naf[1, 0], x0[0], x0[1])
            
            print(f"  HNAF: Modo {selected_mode}, Q_pred={Q_pred:.4f}, R_real={r_real:.4f}")
            print(f"  NAF1: R={r1_naf:.4f}, NAF2: R={r2_naf:.4f}")
            print(f"  Diferencia Q-R: {abs(Q_pred - r_real):.6f}")
            
            # Verificar que el modo seleccionado es el óptimo
            optimal_mode = 0 if r1_naf < r2_naf else 1
            if selected_mode == optimal_mode:
                print(f"  ✅ Modo óptimo seleccionado")
            else:
                print(f"  ❌ Modo subóptimo seleccionado")

def train_stable_hnaf(num_episodes=1000, eval_interval=50):
    """
    Entrena el HNAF estable con más épocas y mejor configuración
    """
    print("="*60)
    print("ENTRENAMIENTO HNAF ESTABLE MEJORADO")
    print("="*60)
    
    # Crear HNAF estable con gamma=0.9
    hnaf = StableHNAF(state_dim=2, action_dim=2, num_modes=2, 
                      hidden_dim=32, lr=1e-4, tau=0.001, gamma=0.9)
    
    # Métricas de entrenamiento
    episode_rewards = []
    losses = []
    eval_rewards = []
    
    for episode in range(num_episodes):
        # Entrenar episodio con exploración aumentada
        reward, _ = hnaf.train_episode(max_steps=20, epsilon=0.2)
        episode_rewards.append(reward)
        
        # Actualizar redes con batch más grande
        loss = hnaf.update(batch_size=32)
        if loss is not None and np.isfinite(loss):
            losses.append(loss)
        
        # Evaluación periódica
        if (episode + 1) % eval_interval == 0:
            eval_reward, mode_selections = hnaf.evaluate_policy(num_episodes=10)
            eval_rewards.append(eval_reward)
            
            print(f"Episodio {episode+1}/{num_episodes}")
            print(f"  Recompensa promedio: {np.mean(episode_rewards[-eval_interval:]):.4f}")
            print(f"  Recompensa evaluación: {eval_reward:.4f}")
            print(f"  Selección de modos: {mode_selections}")
            if losses:
                print(f"  Pérdida promedio: {np.mean(losses[-eval_interval:]):.6f}")
            print()
    
    # Verificación final
    hnaf.verify_hnaf()
    
    return hnaf

if __name__ == "__main__":
    # Entrenar HNAF estable con más épocas
    hnaf = train_stable_hnaf(num_episodes=1000, eval_interval=50)
    
    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*60) 