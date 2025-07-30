#!/usr/bin/env python3
"""
Implementación completa de HNAF (Hybrid Normalized Advantage Function)
Basado en el NAF corregido y el framework de control híbrido.
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
from naf_corrected import CorrectedOptimizationFunctions

class NAFNetwork(nn.Module):
    """
    Red neuronal para un modo discreto específico.
    Salida: V(x,v), μ(x,v), L(x,v)
    """
    
    def __init__(self, state_dim=2, action_dim=2, hidden_dim=64):
        super(NAFNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Capas compartidas
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # V(x,v) - Valor del estado
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # μ(x,v) - Acción media
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        
        # L(x,v) - Matriz triangular inferior (diagonal exponencial)
        self.l_head = nn.Linear(hidden_dim, action_dim * (action_dim + 1) // 2)
        
    def forward(self, x):
        """
        Forward pass: x -> (V, μ, L)
        """
        # Capas compartidas
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Valores de salida
        V = self.value_head(x)
        mu = self.mu_head(x)
        
        # Construir matriz L triangular inferior
        l_params = self.l_head(x)
        L = torch.zeros(x.size(0), self.action_dim, self.action_dim, device=x.device)
        
        # Llenar matriz triangular inferior con diagonal exponencial
        idx = 0
        for i in range(self.action_dim):
            for j in range(i + 1):
                if i == j:
                    # Diagonal: exponencial
                    L[:, i, j] = torch.exp(l_params[:, idx])
                else:
                    # Subdiagonal: valor directo
                    L[:, i, j] = l_params[:, idx]
                idx += 1
        
        return V, mu, L
    
    def get_P_matrix(self, x):
        """
        Calcula P(x,v) = L(x,v) * L(x,v)^T
        """
        V, mu, L = self.forward(x)
        P = torch.bmm(L, L.transpose(1, 2))
        return V, mu, P

class ReplayBuffer:
    """Buffer de experiencia para un modo específico."""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, mode, action, reward, next_state):
        self.buffer.append((state, mode, action, reward, next_state))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        state, mode, action, reward, next_state = zip(*batch)
        return (torch.FloatTensor(state),
                torch.LongTensor(mode),
                torch.FloatTensor(action),
                torch.FloatTensor(reward),
                torch.FloatTensor(next_state))
    
    def __len__(self):
        return len(self.buffer)

class HNAF:
    """
    Hybrid Normalized Advantage Function
    """
    
    def __init__(self, state_dim=2, action_dim=2, num_modes=2, 
                 hidden_dim=64, lr=1e-3, tau=0.001, gamma=0.99):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_modes = num_modes
        self.gamma = gamma
        self.tau = tau
        
        # Redes para cada modo
        self.networks = {}
        self.target_networks = {}
        self.optimizers = {}
        self.replay_buffers = {}
        
        for mode in range(num_modes):
            # Red principal
            self.networks[mode] = NAFNetwork(state_dim, action_dim, hidden_dim)
            # Red target
            self.target_networks[mode] = NAFNetwork(state_dim, action_dim, hidden_dim)
            self.target_networks[mode].load_state_dict(self.networks[mode].state_dict())
            # Optimizador
            self.optimizers[mode] = optim.Adam(self.networks[mode].parameters(), lr=lr)
            # Buffer de replay
            self.replay_buffers[mode] = ReplayBuffer()
        
        # Matrices de transformación (modos discretos) - por defecto
        self.A1 = np.array([[1, 50], [-1, 1]])
        self.A2 = np.array([[1, -1], [50, 1]])
        self.transformation_matrices = {0: self.A1, 1: self.A2}
        
        # Funciones personalizadas (se pueden sobrescribir)
        self.transformation_functions = None
        self.reward_function = None
        
        # NAF corregido para verificación
        self.naf_verifier = CorrectedOptimizationFunctions(t=1.0)
    
    def select_action(self, state, epsilon=0.0):
        """
        Selecciona acción híbrida (modo, acción_continua)
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
        
        # Seleccionar modo: v* = argmin V(x,v)
        selected_mode = min(values.keys(), key=lambda v: values[v])
        
        # Seleccionar acción continua: u = μ(x, v*)
        selected_action = mus[selected_mode]
        
        # Añadir ruido si es necesario
        if epsilon > 0 and random.random() < epsilon:
            selected_action += np.random.normal(0, 0.1, self.action_dim)
        
        return selected_mode, selected_action
    
    def compute_advantage(self, state, action, mode):
        """
        Calcula A(x,v,u) = -0.5 * (u - μ)^T * P(x,v) * (u - μ)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        
        with torch.no_grad():
            V, mu, P = self.networks[mode].get_P_matrix(state_tensor)
            
            # Calcular ventaja
            diff = action_tensor - mu
            advantage = -0.5 * torch.bmm(torch.bmm(diff.unsqueeze(1), P), diff.unsqueeze(2))
            
            return advantage.item()
    
    def compute_Q_value(self, state, action, mode):
        """
        Calcula Q(x,v,u) = V(x,v) + A(x,v,u)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        
        with torch.no_grad():
            V, mu, P = self.networks[mode].get_P_matrix(state_tensor)
            
            # Calcular ventaja
            diff = action_tensor - mu
            advantage = -0.5 * torch.bmm(torch.bmm(diff.unsqueeze(1), P), diff.unsqueeze(2))
            
            # Q-value
            Q = V + advantage
            
            return Q.item()
    
    def update(self, batch_size=32):
        """
        Actualiza todas las redes
        """
        total_loss = 0
        
        for mode in range(self.num_modes):
            if len(self.replay_buffers[mode]) < batch_size:
                continue
                
            # Muestrear batch
            states, modes, actions, rewards, next_states = self.replay_buffers[mode].sample(batch_size)
            
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
            
            # Calcular Q-value actual
            current_Q = torch.zeros(batch_size, 1)
            for i in range(batch_size):
                V, mu, P = self.networks[mode](states[i:i+1])
                diff = actions[i:i+1] - mu
                advantage = -0.5 * torch.bmm(torch.bmm(diff.unsqueeze(1), P), diff.unsqueeze(2))
                current_Q[i] = V + advantage
            
            # Calcular pérdida y actualizar
            loss = F.mse_loss(current_Q, target_Q)
            
            self.optimizers[mode].zero_grad()
            loss.backward()
            self.optimizers[mode].step()
            
            total_loss += loss.item()
            
            # Soft update del target network
            for target_param, param in zip(self.target_networks[mode].parameters(), 
                                         self.networks[mode].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return total_loss / self.num_modes
    
    def step(self, state, mode, action, reward, next_state):
        """
        Almacena transición y actualiza redes
        """
        self.replay_buffers[mode].push(state, mode, action, reward, next_state)
    
    def train_episode(self, max_steps=100, epsilon=0.1):
        """
        Entrena un episodio completo
        """
        # Estado inicial aleatorio
        x0 = np.random.uniform(-1, 1, self.state_dim)
        state = x0.copy()
        
        total_reward = 0
        episode_data = []
        
        for step in range(max_steps):
            # Seleccionar acción
            mode, action = self.select_action(state, epsilon)
            
            # Aplicar transformación según el modo
            if self.transformation_functions is not None:
                # Usar funciones personalizadas
                transform_func = self.transformation_functions[mode]
                x1, y1 = transform_func(state[0], state[1])
                next_state = np.array([x1, y1])
            else:
                # Usar matrices por defecto
                A = self.transformation_matrices[mode]
                next_state = A @ state.reshape(-1, 1)
                next_state = next_state.flatten()
            
            # Calcular recompensa
            if self.reward_function is not None:
                # Usar función de recompensa personalizada
                reward = self.reward_function(next_state[0], next_state[1], 
                                           state[0], state[1])
            else:
                # Usar NAF corregido por defecto
                reward = self.naf_verifier.execute_function("reward_function", 
                                                          next_state[0], next_state[1], 
                                                          state[0], state[1])
            
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
            x0 = np.random.uniform(-1, 1, self.state_dim)
            state = x0.copy()
            
            episode_reward = 0
            episode_modes = []
            
            for step in range(50):  # Episodio más corto para evaluación
                # Seleccionar acción sin ruido
                mode, action = self.select_action(state, epsilon=0.0)
                episode_modes.append(mode)
                
                # Aplicar transformación
                if self.transformation_functions is not None:
                    # Usar funciones personalizadas
                    transform_func = self.transformation_functions[mode]
                    x1, y1 = transform_func(state[0], state[1])
                    next_state = np.array([x1, y1])
                else:
                    # Usar matrices por defecto
                    A = self.transformation_matrices[mode]
                    next_state = A @ state.reshape(-1, 1)
                    next_state = next_state.flatten()
                
                # Recompensa
                if self.reward_function is not None:
                    # Usar función de recompensa personalizada
                    reward = self.reward_function(next_state[0], next_state[1], 
                                               state[0], state[1])
                else:
                    # Usar NAF corregido por defecto
                    reward = self.naf_verifier.execute_function("reward_function", 
                                                              next_state[0], next_state[1], 
                                                              state[0], state[1])
                
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
        Verifica que el HNAF funciona correctamente comparando con NAF individual
        """
        if test_states is None:
            test_states = [
                np.array([1, 1]),
                np.array([0, 1]),
                np.array([1, 0]),
                np.array([0.5, 0.5]),
                np.array([-0.5, 0.8])
            ]
        
        print("="*60)
        print("VERIFICACIÓN HNAF vs NAF INDIVIDUAL")
        print("="*60)
        
        for i, x0 in enumerate(test_states):
            print(f"\nCaso {i+1}: Estado inicial {x0}")
            
            # HNAF: seleccionar modo y acción
            selected_mode, selected_action = self.select_action(x0, epsilon=0.0)
            Q_pred = self.compute_Q_value(x0, selected_action, selected_mode)
            
            # Aplicar transformación del modo seleccionado
            if self.transformation_functions is not None:
                # Usar funciones personalizadas
                transform_func = self.transformation_functions[selected_mode]
                x1, y1 = transform_func(x0[0], x0[1])
                x_next = np.array([x1, y1])
            else:
                # Usar matrices por defecto
                A = self.transformation_matrices[selected_mode]
                x_next = A @ x0.reshape(-1, 1)
                x_next = x_next.flatten()
            
            # Recompensa real
            if self.reward_function is not None:
                # Usar función de recompensa personalizada
                r_real = self.reward_function(x_next[0], x_next[1], x0[0], x0[1])
            else:
                # Usar NAF corregido por defecto
                r_real = self.naf_verifier.execute_function("reward_function", 
                                                          x_next[0], x_next[1], 
                                                          x0[0], x0[1])
            
            # NAF individual para comparar
            if self.transformation_functions is not None:
                # Usar funciones personalizadas para comparación
                x1_naf_x, x1_naf_y = self.transformation_functions[0](x0[0], x0[1])
                x2_naf_x, x2_naf_y = self.transformation_functions[1](x0[0], x0[1])
                
                if self.reward_function is not None:
                    r1_naf = self.reward_function(x1_naf_x, x1_naf_y, x0[0], x0[1])
                    r2_naf = self.reward_function(x2_naf_x, x2_naf_y, x0[0], x0[1])
                else:
                    r1_naf = self.naf_verifier.execute_function("reward_function", x1_naf_x, x1_naf_y, x0[0], x0[1])
                    r2_naf = self.naf_verifier.execute_function("reward_function", x2_naf_x, x2_naf_y, x0[0], x0[1])
            else:
                # Usar NAF corregido por defecto
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

def train_hnaf(num_episodes=1000, eval_interval=100):
    """
    Entrena el HNAF completo
    """
    print("="*60)
    print("ENTRENAMIENTO HNAF")
    print("="*60)
    
    # Crear HNAF
    hnaf = HNAF(state_dim=2, action_dim=2, num_modes=2, 
                hidden_dim=64, lr=1e-3, tau=0.001, gamma=0.99)
    
    # Métricas de entrenamiento
    episode_rewards = []
    losses = []
    eval_rewards = []
    
    for episode in range(num_episodes):
        # Entrenar episodio
        reward, _ = hnaf.train_episode(max_steps=50, epsilon=0.1)
        episode_rewards.append(reward)
        
        # Actualizar redes
        loss = hnaf.update(batch_size=32)
        if loss is not None:
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
    
    # Visualización
    visualize_training(episode_rewards, losses, eval_rewards, eval_interval)
    
    return hnaf

def visualize_training(episode_rewards, losses, eval_rewards, eval_interval):
    """
    Visualiza el entrenamiento
    """
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Recompensas por episodio
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards, alpha=0.6, label='Episodio')
    if eval_rewards:
        eval_episodes = np.arange(eval_interval, len(episode_rewards) + 1, eval_interval)
        plt.plot(eval_episodes, eval_rewards, 'r-', linewidth=2, label='Evaluación')
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.title("Recompensas durante el entrenamiento")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Pérdida
    if losses:
        plt.subplot(1, 3, 2)
        plt.plot(losses)
        plt.xlabel("Actualización")
        plt.ylabel("Pérdida")
        plt.title("Pérdida del critic")
        plt.grid(True, alpha=0.3)
    
    # Subplot 3: Recompensa promedio móvil
    plt.subplot(1, 3, 3)
    window = 100
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_rewards)), moving_avg, 'g-', linewidth=2)
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa promedio")
        plt.title(f"Recompensa promedio móvil (ventana={window})")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Entrenar HNAF
    hnaf = train_hnaf(num_episodes=500, eval_interval=50)
    
    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*60) 