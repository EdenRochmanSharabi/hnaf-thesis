#!/usr/bin/env python3
"""
Verificación de la implementación NAF comparando con el código de referencia.
Este script compara las transformaciones lineales de NAF con la solución de ODE.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from optimization_functions import OptimizationFunctions

def solve_linear_ode(A, x0, t):
    """Solución: x(t) = expm(A * t) @ x0"""
    return expm(A * t) @ x0

def reward(x, x0):
    """Calcular la recompensa como la diferencia de normas"""
    return abs(np.linalg.norm(x) - np.linalg.norm(x0))

def generate_random_initial_state():
    """Generar nuevo estado inicial aleatorio en coordenadas polares"""
    alpha_deg = np.random.uniform(0, 360)
    alpha_rad = np.deg2rad(alpha_deg)
    r = 1
    x0 = r * np.cos(alpha_rad)
    y0 = r * np.sin(alpha_rad)
    return np.array([x0, y0]), alpha_deg

def simulate_reference(n_iter=10, t=1.0):
    """Simulación de referencia usando ODE"""
    A1 = np.array([[1, 50],
                   [-1, 1]])

    rewards = []
    alphas = []
    states = []

    for _ in range(n_iter):
        x0, alpha = generate_random_initial_state()
        x0_col = x0.reshape((2, 1))

        # Solución ODE
        x = solve_linear_ode(A1, x0_col, t)

        # Recompensa
        r = reward(x, x0_col)
        rewards.append(r)
        alphas.append(alpha)
        states.append((x0, x.flatten()))

        print(f"α = {alpha:.2f}°, Recompensa ODE = {r:.4f}")

    return alphas, rewards, states

def simulate_naf(n_iter=10):
    """Simulación usando NAF (transformación directa)"""
    opt_funcs = OptimizationFunctions()
    
    rewards = []
    alphas = []
    states = []

    for _ in range(n_iter):
        x0, alpha = generate_random_initial_state()
        
        # Usar NAF para transformación
        x1 = opt_funcs.execute_function("transform_x1", x0[0], x0[1])
        x1_flat = x1.flatten()
        
        # Recompensa usando NAF
        r = opt_funcs.execute_function("reward_function", x1_flat[0], x1_flat[1], x0[0], x0[1])
        rewards.append(r)
        alphas.append(alpha)
        states.append((x0, x1_flat))

        print(f"α = {alpha:.2f}°, Recompensa NAF = {r:.4f}")

    return alphas, rewards, states

def compare_methods(n_iter=20, t=1.0):
    """Comparar ambos métodos"""
    print("="*60)
    print("COMPARACIÓN NAF vs SOLUCIÓN ODE")
    print("="*60)
    
    print(f"\nEjecutando {n_iter} iteraciones...")
    print("-" * 40)
    
    # Simulación de referencia (ODE)
    print("1. SIMULACIÓN DE REFERENCIA (ODE):")
    alphas_ref, rewards_ref, states_ref = simulate_reference(n_iter, t)
    
    print("\n2. SIMULACIÓN NAF (TRANSFORMACIÓN DIRECTA):")
    alphas_naf, rewards_naf, states_naf = simulate_naf(n_iter)
    
    # Comparar resultados
    print("\n3. COMPARACIÓN DE RESULTADOS:")
    print("-" * 40)
    
    # Calcular estadísticas
    mean_ref = np.mean(rewards_ref)
    mean_naf = np.mean(rewards_naf)
    std_ref = np.std(rewards_ref)
    std_naf = np.std(rewards_naf)
    
    print(f"Recompensa promedio ODE:  {mean_ref:.4f} ± {std_ref:.4f}")
    print(f"Recompensa promedio NAF:  {mean_naf:.4f} ± {std_naf:.4f}")
    print(f"Diferencia:                {abs(mean_ref - mean_naf):.4f}")
    
    # Comparar casos específicos
    print(f"\nComparación caso por caso:")
    for i in range(min(5, n_iter)):
        print(f"  Caso {i+1}: ODE={rewards_ref[i]:.4f}, NAF={rewards_naf[i]:.4f}, Diff={abs(rewards_ref[i]-rewards_naf[i]):.4f}")
    
    # Visualización
    visualize_comparison(alphas_ref, rewards_ref, alphas_naf, rewards_naf)
    
    return alphas_ref, rewards_ref, alphas_naf, rewards_naf

def visualize_comparison(alphas_ref, rewards_ref, alphas_naf, rewards_naf):
    """Visualizar comparación de ambos métodos"""
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Recompensas vs ángulo
    plt.subplot(1, 3, 1)
    plt.scatter(alphas_ref, rewards_ref, alpha=0.7, label='ODE', color='blue')
    plt.scatter(alphas_naf, rewards_naf, alpha=0.7, label='NAF', color='red')
    plt.xlabel("Ángulo α (grados)")
    plt.ylabel("Recompensa")
    plt.title("Recompensa vs Estado Inicial")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Distribución de recompensas
    plt.subplot(1, 3, 2)
    plt.hist(rewards_ref, alpha=0.7, label='ODE', bins=10, color='blue')
    plt.hist(rewards_naf, alpha=0.7, label='NAF', bins=10, color='red')
    plt.xlabel("Recompensa")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de Recompensas")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Comparación directa
    plt.subplot(1, 3, 3)
    min_len = min(len(rewards_ref), len(rewards_naf))
    plt.scatter(rewards_ref[:min_len], rewards_naf[:min_len], alpha=0.7)
    plt.plot([0, max(max(rewards_ref), max(rewards_naf))], 
             [0, max(max(rewards_ref), max(rewards_naf))], 'k--', alpha=0.5)
    plt.xlabel("Recompensa ODE")
    plt.ylabel("Recompensa NAF")
    plt.title("ODE vs NAF")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def test_specific_cases():
    """Probar casos específicos para verificar la implementación"""
    print("\n" + "="*60)
    print("PRUEBAS ESPECÍFICAS")
    print("="*60)
    
    opt_funcs = OptimizationFunctions()
    
    # Caso 1: Estado inicial (1, 1)
    x0 = np.array([1, 1])
    print(f"\nCaso 1: Estado inicial {x0}")
    
    # ODE
    A1 = np.array([[1, 50], [-1, 1]])
    x_ode = solve_linear_ode(A1, x0.reshape(2, 1), 1.0)
    r_ode = reward(x_ode, x0.reshape(2, 1))
    
    # NAF
    x_naf = opt_funcs.execute_function("transform_x1", x0[0], x0[1])
    r_naf = opt_funcs.execute_function("reward_function", x_naf[0, 0], x_naf[1, 0], x0[0], x0[1])
    
    print(f"  ODE: x = {x_ode.flatten()}, r = {r_ode:.4f}")
    print(f"  NAF: x = {x_naf.flatten()}, r = {r_naf:.4f}")
    print(f"  Diferencia: {abs(r_ode - r_naf):.6f}")
    
    # Caso 2: Estado inicial (0, 1)
    x0 = np.array([0, 1])
    print(f"\nCaso 2: Estado inicial {x0}")
    
    # ODE
    x_ode = solve_linear_ode(A1, x0.reshape(2, 1), 1.0)
    r_ode = reward(x_ode, x0.reshape(2, 1))
    
    # NAF
    x_naf = opt_funcs.execute_function("transform_x1", x0[0], x0[1])
    r_naf = opt_funcs.execute_function("reward_function", x_naf[0, 0], x_naf[1, 0], x0[0], x0[1])
    
    print(f"  ODE: x = {x_ode.flatten()}, r = {r_ode:.4f}")
    print(f"  NAF: x = {x_naf.flatten()}, r = {r_naf:.4f}")
    print(f"  Diferencia: {abs(r_ode - r_naf):.6f}")

def main():
    """Función principal"""
    print("VERIFICACIÓN DE IMPLEMENTACIÓN NAF")
    print("="*60)
    
    # Pruebas específicas
    test_specific_cases()
    
    # Comparación completa
    alphas_ref, rewards_ref, alphas_naf, rewards_naf = compare_methods(n_iter=30)
    
    print("\n" + "="*60)
    print("CONCLUSIÓN")
    print("="*60)
    
    # Análisis final
    mean_diff = np.mean([abs(r_ref - r_naf) for r_ref, r_naf in zip(rewards_ref, rewards_naf)])
    max_diff = max([abs(r_ref - r_naf) for r_ref, r_naf in zip(rewards_ref, rewards_naf)])
    
    print(f"Error promedio: {mean_diff:.6f}")
    print(f"Error máximo:   {max_diff:.6f}")
    
    if mean_diff < 1e-10:
        print("✅ NAF funciona correctamente - resultados idénticos a ODE")
    elif mean_diff < 1e-6:
        print("✅ NAF funciona correctamente - diferencias mínimas")
    elif mean_diff < 1e-3:
        print("⚠️  NAF funciona con pequeñas diferencias")
    else:
        print("❌ NAF no funciona correctamente - diferencias significativas")

if __name__ == "__main__":
    main() 