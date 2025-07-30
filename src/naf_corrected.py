#!/usr/bin/env python3
"""
Versión corregida de NAF que usa exponencial de matriz para coincidir con la solución ODE.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimization_functions import OptimizationFunctions

class CorrectedOptimizationFunctions(OptimizationFunctions):
    """
    Versión corregida que usa exponencial de matriz en lugar de transformación directa.
    """
    
    def __init__(self, t=1.0):
        """
        Inicializa con tiempo t para la exponencial de matriz.
        
        Args:
            t: Tiempo para la evolución ODE
        """
        super().__init__()
        self.t = t
        self._update_transformations()
    
    def _update_transformations(self):
        """Actualiza las transformaciones para usar exponencial de matriz."""
        # Calcular exponenciales de matriz
        self.exp_A1_t = expm(self.A1 * self.t)
        self.exp_A2_t = expm(self.A2 * self.t)
        
        print(f"Exponencial A1*t (t={self.t}):")
        print(self.exp_A1_t)
        print(f"Exponencial A2*t (t={self.t}):")
        print(self.exp_A2_t)
    
    def _transform_x1(self, x0=None, y0=None):
        """
        Aplica la transformación corregida usando exponencial de matriz.
        x1 = expm(A1 * t) @ x0
        """
        if x0 is None:
            x0 = self.x0
        if y0 is None:
            y0 = self.y0
            
        x1 = self.exp_A1_t @ np.array([[x0], [y0]])
        return x1
    
    def _transform_x2(self, x0=None, y0=None):
        """
        Aplica la transformación corregida usando exponencial de matriz.
        x2 = expm(A2 * t) @ x0
        """
        if x0 is None:
            x0 = self.x0
        if y0 is None:
            y0 = self.y0
            
        x2 = self.exp_A2_t @ np.array([[x0], [y0]])
        return x2

def verify_correction():
    """Verificar que la corrección funciona."""
    print("="*60)
    print("VERIFICACIÓN DE LA CORRECCIÓN")
    print("="*60)
    
    # Crear instancia corregida
    opt_funcs_corrected = CorrectedOptimizationFunctions(t=1.0)
    
    # Casos de prueba
    test_cases = [
        np.array([1, 1]),
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([0.5, 0.5])
    ]
    
    for i, x0 in enumerate(test_cases):
        print(f"\nCaso {i+1}: Estado inicial {x0}")
        
        # ODE directo
        A1 = np.array([[1, 50], [-1, 1]])
        x_ode = expm(A1 * 1.0) @ x0.reshape(2, 1)
        r_ode = abs(np.linalg.norm(x_ode) - np.linalg.norm(x0))
        
        # NAF corregido
        x_naf = opt_funcs_corrected.execute_function("transform_x1", x0[0], x0[1])
        r_naf = opt_funcs_corrected.execute_function("reward_function", x_naf[0, 0], x_naf[1, 0], x0[0], x0[1])
        
        print(f"  ODE: x = {x_ode.flatten()}, r = {r_ode:.4f}")
        print(f"  NAF: x = {x_naf.flatten()}, r = {r_naf:.4f}")
        print(f"  Diferencia: {abs(r_ode - r_naf):.6f}")
        
        # Verificar que son idénticos
        if abs(r_ode - r_naf) < 1e-10:
            print("  ✅ PERFECTO - Resultados idénticos")
        else:
            print("  ❌ ERROR - Diferencias encontradas")

def compare_all_methods(n_iter=20):
    """Comparar todos los métodos: ODE, NAF original, NAF corregido."""
    print("\n" + "="*60)
    print("COMPARACIÓN COMPLETA DE MÉTODOS")
    print("="*60)
    
    # Métodos
    opt_funcs_original = OptimizationFunctions()
    opt_funcs_corrected = CorrectedOptimizationFunctions(t=1.0)
    
    # Resultados
    rewards_ode = []
    rewards_naf_original = []
    rewards_naf_corrected = []
    alphas = []
    
    for _ in range(n_iter):
        # Generar estado aleatorio
        alpha_deg = np.random.uniform(0, 360)
        alpha_rad = np.deg2rad(alpha_deg)
        r = 1
        x0 = r * np.cos(alpha_rad)
        y0 = r * np.sin(alpha_rad)
        
        # ODE
        A1 = np.array([[1, 50], [-1, 1]])
        x_ode = expm(A1 * 1.0) @ np.array([[x0], [y0]])
        r_ode = abs(np.linalg.norm(x_ode) - np.linalg.norm([x0, y0]))
        
        # NAF original
        x_naf_orig = opt_funcs_original.execute_function("transform_x1", x0, y0)
        r_naf_orig = opt_funcs_original.execute_function("reward_function", x_naf_orig[0, 0], x_naf_orig[1, 0], x0, y0)
        
        # NAF corregido
        x_naf_corr = opt_funcs_corrected.execute_function("transform_x1", x0, y0)
        r_naf_corr = opt_funcs_corrected.execute_function("reward_function", x_naf_corr[0, 0], x_naf_corr[1, 0], x0, y0)
        
        rewards_ode.append(r_ode)
        rewards_naf_original.append(r_naf_orig)
        rewards_naf_corrected.append(r_naf_corr)
        alphas.append(alpha_deg)
        
        print(f"α = {alpha_deg:.2f}°: ODE={r_ode:.4f}, NAF_orig={r_naf_orig:.4f}, NAF_corr={r_naf_corr:.4f}")
    
    # Estadísticas
    print(f"\nESTADÍSTICAS:")
    print(f"ODE promedio:           {np.mean(rewards_ode):.4f} ± {np.std(rewards_ode):.4f}")
    print(f"NAF original promedio:  {np.mean(rewards_naf_original):.4f} ± {np.std(rewards_naf_original):.4f}")
    print(f"NAF corregido promedio: {np.mean(rewards_naf_corrected):.4f} ± {np.std(rewards_naf_corrected):.4f}")
    
    # Errores
    error_original = np.mean([abs(r_ode - r_naf) for r_ode, r_naf in zip(rewards_ode, rewards_naf_original)])
    error_corrected = np.mean([abs(r_ode - r_naf) for r_ode, r_naf in zip(rewards_ode, rewards_naf_corrected)])
    
    print(f"\nERRORES:")
    print(f"Error NAF original:  {error_original:.6f}")
    print(f"Error NAF corregido: {error_corrected:.6f}")
    
    # Visualización
    visualize_all_methods(alphas, rewards_ode, rewards_naf_original, rewards_naf_corrected)
    
    return rewards_ode, rewards_naf_original, rewards_naf_corrected

def visualize_all_methods(alphas, rewards_ode, rewards_naf_original, rewards_naf_corrected):
    """Visualizar comparación de todos los métodos."""
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Recompensas vs ángulo
    plt.subplot(1, 3, 1)
    plt.scatter(alphas, rewards_ode, alpha=0.7, label='ODE', color='blue')
    plt.scatter(alphas, rewards_naf_original, alpha=0.7, label='NAF Original', color='red')
    plt.scatter(alphas, rewards_naf_corrected, alpha=0.7, label='NAF Corregido', color='green')
    plt.xlabel("Ángulo α (grados)")
    plt.ylabel("Recompensa")
    plt.title("Recompensa vs Estado Inicial")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Comparación ODE vs NAF Original
    plt.subplot(1, 3, 2)
    plt.scatter(rewards_ode, rewards_naf_original, alpha=0.7, color='red', label='Original')
    plt.plot([0, max(max(rewards_ode), max(rewards_naf_original))], 
             [0, max(max(rewards_ode), max(rewards_naf_original))], 'k--', alpha=0.5)
    plt.xlabel("Recompensa ODE")
    plt.ylabel("Recompensa NAF Original")
    plt.title("ODE vs NAF Original")
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Comparación ODE vs NAF Corregido
    plt.subplot(1, 3, 3)
    plt.scatter(rewards_ode, rewards_naf_corrected, alpha=0.7, color='green', label='Corregido')
    plt.plot([0, max(max(rewards_ode), max(rewards_naf_corrected))], 
             [0, max(max(rewards_ode), max(rewards_naf_corrected))], 'k--', alpha=0.5)
    plt.xlabel("Recompensa ODE")
    plt.ylabel("Recompensa NAF Corregido")
    plt.title("ODE vs NAF Corregido")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Función principal"""
    print("CORRECCIÓN DE IMPLEMENTACIÓN NAF")
    print("="*60)
    
    # Verificar corrección
    verify_correction()
    
    # Comparación completa
    rewards_ode, rewards_naf_original, rewards_naf_corrected = compare_all_methods(n_iter=20)
    
    print("\n" + "="*60)
    print("CONCLUSIÓN FINAL")
    print("="*60)
    
    error_original = np.mean([abs(r_ode - r_naf) for r_ode, r_naf in zip(rewards_ode, rewards_naf_original)])
    error_corrected = np.mean([abs(r_ode - r_naf) for r_ode, r_naf in zip(rewards_ode, rewards_naf_corrected)])
    
    print(f"NAF Original:  {'❌ NO funciona' if error_original > 1e-3 else '✅ Funciona'}")
    print(f"NAF Corregido: {'❌ NO funciona' if error_corrected > 1e-3 else '✅ Funciona'}")
    
    if error_corrected < 1e-10:
        print("\n🎉 ¡NAF corregido funciona perfectamente!")
        print("La implementación ahora usa exponencial de matriz como la solución ODE.")
    else:
        print("\n⚠️  Aún hay diferencias en NAF corregido.")

if __name__ == "__main__":
    main() 