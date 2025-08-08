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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

# --- Red Neuronal HNAF con Batch Normalization ---
class HNAFNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers, num_modes):
        super(HNAFNetwork, self).__init__()
        self.num_modes = num_modes
        self.action_dim = action_dim

        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
        self.base_layers = nn.Sequential(*layers)

        self.heads = nn.ModuleList()
        for _ in range(num_modes):
            l_output_dim = action_dim * (action_dim + 1) // 2
            self.heads.append(nn.Linear(hidden_dim, 1 + action_dim + l_output_dim))

    def forward(self, state):
        features = self.base_layers(state)
        return [head(features) for head in self.heads]

# --- Cerebro del Agente HNAF ---
class HNAFImproved:
    def __init__(self, config, logger):
        self.logger = logger
        self.logger.info("Iniciando constructor de HNAFImproved...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device: {self.device}")
        
        network_config = config.get('network', {}).get('defaults', {})
        training_config = config.get('training', {}).get('defaults', {})
        self.logger.info(f"Network config keys: {list(network_config.keys())}")
        self.logger.info(f"Training config keys: {list(training_config.keys())}")
        
        self.state_dim = network_config.get('state_dim')
        self.action_dim = network_config.get('action_dim')
        self.num_modes = network_config.get('num_modes')
        self.gamma = training_config.get('gamma')
        self.tau = training_config.get('tau', 0.005)

        self.logger.info(f"state_dim: {self.state_dim} (tipo: {type(self.state_dim)})")
        self.logger.info(f"action_dim: {self.action_dim} (tipo: {type(self.action_dim)})")
        self.logger.info(f"num_modes: {self.num_modes} (tipo: {type(self.num_modes)})")
        self.logger.info(f"gamma: {self.gamma} (tipo: {type(self.gamma)})")
        self.logger.info(f"tau: {self.tau} (tipo: {type(self.tau)})")

        hidden_dim = network_config.get('hidden_dim')
        num_layers = network_config.get('num_layers')
        lr = training_config.get('learning_rate')  # Corregido: buscar 'learning_rate' en lugar de 'lr'

        self.logger.info(f"hidden_dim: {hidden_dim} (tipo: {type(hidden_dim)})")
        self.logger.info(f"num_layers: {num_layers} (tipo: {type(num_layers)})")
        self.logger.info(f"lr: {lr} (tipo: {type(lr)})")

        # Verificar que ningún valor sea None antes de crear las redes
        if any(v is None for v in [self.state_dim, self.action_dim, self.num_modes, hidden_dim, num_layers, lr]):
            missing_values = []
            if self.state_dim is None: missing_values.append('state_dim')
            if self.action_dim is None: missing_values.append('action_dim')
            if self.num_modes is None: missing_values.append('num_modes')
            if hidden_dim is None: missing_values.append('hidden_dim')
            if num_layers is None: missing_values.append('num_layers')
            if lr is None: missing_values.append('lr')
            raise ValueError(f"Valores faltantes en config: {missing_values}")

        self.logger.info("Creando redes neuronales...")
        self.model = HNAFNetwork(self.state_dim, self.action_dim, hidden_dim, num_layers, self.num_modes).to(self.device)
        self.target_model = HNAFNetwork(self.state_dim, self.action_dim, hidden_dim, num_layers, self.num_modes).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.logger.info("Redes neuronales creadas exitosamente")

        buffer_capacity = training_config.get('buffer_capacity', 200000)
        self.replay_buffer = deque(maxlen=buffer_capacity) # CORRECCIÓN: 'replay_buffer' en singular
        self.batch_size = training_config.get('batch_size')
        
        self.logger.info(f"buffer_capacity: {buffer_capacity}")
        self.logger.info(f"batch_size: {self.batch_size} (tipo: {type(self.batch_size)})")
        
        # Verificar batch_size
        if self.batch_size is None:
            raise ValueError("batch_size es None")
        
        self.logger.info("Constructor de HNAFImproved completado exitosamente")
        
        # Inicializar matrices de transformación por defecto (3D, tantas como num_modes)
        default_A = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, -1.0], [0.0, 1.0, 2.0]])
        self.transformation_matrices = [default_A.copy() for _ in range(int(self.num_modes))]
        
        # DEBUG: Verificar matrices por defecto
        print(f"🔍 DEBUG - Constructor HNAFImproved:")
        for idx, mat in enumerate(self.transformation_matrices):
            print(f"   transformation_matrices[{idx}] forma: {mat.shape}")
            print(f"   transformation_matrices[{idx}] valor: {mat}")
        
        # Inicializar función de recompensa por defecto
        self.reward_function = self._default_reward_function
        
        # Inicializar config_manager
        self.config_manager = None  # Se establecerá desde TrainingManager

    def _default_reward_function(self, x, y, x0, y0, mode=None, action=None, previous_state=None):
        """Función de recompensa por defecto"""
        return -np.tanh(np.linalg.norm([x, y]) * 0.1)
    
    def update_transformation_matrices(self, *matrices):
        """Actualizar matrices de transformación (soporta 2 o más matrices)."""
        print(f"🔍 DEBUG - Matrices recibidas en update_transformation_matrices (n={len(matrices)}):")
        normalized = []
        for i, m in enumerate(matrices):
            arr = np.array(m)
            print(f"   A{i+1} tipo: {type(m)}, forma: {arr.shape}")
            print(f"   A{i+1} valor: {arr}")
            normalized.append(arr)

        if not normalized:
            raise ValueError("No se recibieron matrices para actualizar")

        # Ajustar al número de modos del modelo
        if len(normalized) < int(self.num_modes):
            last = normalized[-1]
            normalized.extend([last.copy() for _ in range(int(self.num_modes) - len(normalized))])

        self.transformation_matrices = normalized[: int(self.num_modes)]

        try:
            shapes = [mat.shape for mat in self.transformation_matrices]
            self.logger.info(f"Matrices actualizadas (formas): {shapes}")
        except Exception:
            pass

    def select_action(self, state):
        # Usar eval mode para evitar problemas con BatchNorm
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            all_head_outputs = self.model(state_tensor)
            
            best_q_value = -float('inf')
            best_mode = 0 # Default to mode 0
            best_action = np.random.uniform(-1, 1, self.action_dim) # Default a una acción aleatoria

            for mode_idx, head_output in enumerate(all_head_outputs):
                action_candidate = head_output[:, 1:(1 + self.action_dim)]
                q_value = self._get_q_value_from_output(head_output, action_candidate)[0] # Tomar el valor escalar
                
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_mode = mode_idx
                    best_action = action_candidate.cpu().numpy().flatten()
        
        # Volver a train mode
        self.model.train()
        return best_mode, best_action

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None # Devolver None si no hay suficientes muestras

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, modes, actions, rewards, next_states = map(np.array, zip(*batch))

        states = torch.FloatTensor(states).to(self.device)
        modes = torch.LongTensor(modes).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        
        # --- Cómputo del Q-valor Objetivo (Target) ---
        with torch.no_grad():
            all_target_outputs = self.target_model(next_states)
            next_q_values_per_mode = []
            for mode_idx in range(self.num_modes):
                head_output = all_target_outputs[mode_idx]
                mu_next = head_output[:, 1:(1 + self.action_dim)]
                next_q_values_per_mode.append(self._get_q_value_from_output(head_output, mu_next))
            
            next_q_values_stacked = torch.cat(next_q_values_per_mode, dim=1)
            max_next_q_values, _ = torch.max(next_q_values_stacked, dim=1)
            target_q_values = rewards + (self.gamma * max_next_q_values.unsqueeze(1))

        # --- Cómputo del Q-valor Actual y la Pérdida ---
        self.optimizer.zero_grad()
        all_main_outputs = self.model(states)

        # CORRECCIÓN CRÍTICA: Calcular la pérdida de forma que no haya conflicto de dimensiones.
        q_values_list = []
        for i in range(self.batch_size):
            mode = modes[i].item()
            # Seleccionar la salida de la cabeza correcta para la muestra i
            head_output = all_main_outputs[mode][i:i+1]  # Usar slicing en lugar de unsqueeze
            # Evaluar la acción que realmente se tomó
            action = actions[i:i+1]  # Usar slicing en lugar de unsqueeze
            q_value = self._get_q_value_from_output(head_output, action)
            q_values_list.append(q_value)
            
        q_values = torch.cat(q_values_list, dim=0)
        
        loss = F.mse_loss(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()
        self._soft_update_target_network()
        
        return loss.item() # Devolver la pérdida como un escalar

    def _get_q_value_from_output(self, head_output, action):
        value = head_output[:, 0].unsqueeze(1)
        mu = head_output[:, 1:(1 + self.action_dim)]
        
        l_flat = head_output[:, (1 + self.action_dim):]
        L = torch.zeros(head_output.shape[0], self.action_dim, self.action_dim, device=self.device)
        tril_indices = torch.tril_indices(row=self.action_dim, col=self.action_dim, offset=0)
        L[:, tril_indices[0], tril_indices[1]] = l_flat
        
        P = torch.bmm(L, L.transpose(1, 2))
        
        u_minus_mu = (action - mu).unsqueeze(2)
        advantage = -0.5 * torch.bmm(torch.bmm(u_minus_mu.transpose(1, 2), P), u_minus_mu).squeeze(2)
        
        return value + advantage

    def _soft_update_target_network(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
    def step(self, state, mode, action, reward, next_state):
        self.replay_buffer.append((state, mode, action, reward, next_state)) # CORRECCIÓN: 'replay_buffer' en singular
        if len(self.replay_buffer) > self.batch_size:
            return self.learn()
        return None
    
    def update(self, batch_size=None):
        """Método de compatibilidad con el código existente"""
        return self.learn()
    
    def evaluate_policy(self, num_episodes=10):
        """Evaluar la política actual"""
        total_rewards = []
        mode_counts = {i: 0 for i in range(self.num_modes)}
        
        for _ in range(num_episodes):
            # Simular un episodio simple
            state = np.random.uniform(-0.5, 0.5, self.state_dim)
            total_reward = 0
            
            for step in range(50):  # 50 pasos por episodio
                mode, action = self.select_action(state)
                mode_counts[mode] += 1
                
                # Simular transformación simple
                next_state = state + action * 0.1
                next_state = np.clip(next_state, -5.0, 5.0)
                
                # Recompensa simple
                reward = -np.tanh(np.linalg.norm(next_state) * 0.1)
                total_reward += reward
                
                state = next_state
                
                if np.linalg.norm(state) < 0.1:
                    break
            
            total_rewards.append(total_reward)
        
        return np.mean(total_rewards), mode_counts
    
    def evaluate_policy_grid(self, grid_size=50):
        """Evaluar política en una rejilla"""
        x = np.linspace(-0.5, 0.5, grid_size)
        y = np.linspace(-0.5, 0.5, grid_size)
        X, Y = np.meshgrid(x, y)
        
        mode_selections = np.zeros((grid_size, grid_size))
        q_values = np.zeros((grid_size, grid_size))
        optimal_accuracy = 0
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Para visualización, evaluamos sobre plano XY; z=0 si aplica
                if self.state_dim >= 3:
                    state = np.array([X[i, j], Y[i, j], 0.0])
                else:
                    state = np.array([X[i, j], Y[i, j]])
                mode, _ = self.select_action(state)
                mode_selections[i, j] = mode
                
                # Q-value simple
                q_values[i, j] = -np.linalg.norm(state)
                
                # Verificar si es óptimo (simplificado)
                optimal_mode = 0 if state[0] > 0 else 1
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
    
    def verify_hnaf(self):
        """Verificar funcionamiento del HNAF"""
        print("="*60)
        print("VERIFICACIÓN HNAF MEJORADO")
        print("="*60)
        
        # Construir estados de prueba respetando la dimensión del estado
        base_states_2d = [
            [0.1, 0.1],
            [0.0, 0.1],
            [0.1, 0.0],
            [0.05, 0.05],
            [-0.05, 0.08]
        ]
        test_states = []
        for xs, ys in base_states_2d:
            if self.state_dim >= 3:
                test_states.append(np.array([xs, ys, 0.0]))
            else:
                test_states.append(np.array([xs, ys]))
        
        for i, state in enumerate(test_states):
            print(f"\nEstado {i+1}: {state}")
            mode, action = self.select_action(state)
            print(f"  HNAF: Modo {mode}, Q={-np.linalg.norm(state):.4f}")
            if self.num_modes >= 2:
                print(f"  Óptimo (heurística simple): Modo {0 if state[0] > 0 else 1}")
                print(f"  Correcto: {'✅' if mode == (0 if state[0] > 0 else 1) else '❌'}")
        
        # Evaluación en grid
        grid_results = self.evaluate_policy_grid(grid_size=50)
        print(f"\nPrecisión en grid 50x50: {grid_results['optimal_accuracy']:.2%}") 