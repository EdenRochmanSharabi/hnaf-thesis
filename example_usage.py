#!/usr/bin/env python3
"""
Ejemplo de uso de la clase OptimizationFunctions
Demuestra cómo usar las funciones existentes y agregar nuevas funciones.
"""

from optimization_functions import OptimizationFunctions
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("="*60)
    print("EJEMPLO DE USO DE LA CLASE OptimizationFunctions")
    print("="*60)
    
    # Crear instancia de la clase
    opt_funcs = OptimizationFunctions()
    
    print("\n1. FUNCIONES DISPONIBLES:")
    opt_funcs.list_functions()
    
    print("\n2. EJECUTANDO FUNCIONES EXISTENTES:")
    print("-" * 40)
    
    # Ejecutar transformaciones
    x1 = opt_funcs.execute_function("transform_x1")
    x2 = opt_funcs.execute_function("transform_x2")
    
    print(f"Transformación x1 (A1 @ [1,1]):\n{x1}")
    print(f"Transformación x2 (A2 @ [1,1]):\n{x2}")
    
    # Calcular recompensa para diferentes puntos
    test_points = [(3, 4), (0, 0), (1, 1), (5, 5)]
    print("\nRecompensas para diferentes puntos:")
    for x, y in test_points:
        r = opt_funcs.execute_function("reward_function", x, y)
        print(f"  ({x}, {y}) -> Recompensa: {r:.4f}")
    
    print("\n3. AGREGANDO NUEVAS FUNCIONES:")
    print("-" * 40)
    
    # Función 1: Distancia euclidiana
    def euclidean_distance(x1, y1, x2, y2):
        """Calcula la distancia euclidiana entre dos puntos."""
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    opt_funcs.add_function("euclidean_distance", euclidean_distance, "Distancia euclidiana entre dos puntos")
    
    # Función 2: Producto escalar
    def dot_product(x1, y1, x2, y2):
        """Calcula el producto escalar entre dos vectores."""
        return x1 * x2 + y1 * y2
    
    opt_funcs.add_function("dot_product", dot_product, "Producto escalar entre dos vectores")
    
    # Función 3: Ángulo entre vectores
    def angle_between_vectors(x1, y1, x2, y2):
        """Calcula el ángulo entre dos vectores en radianes."""
        dot = opt_funcs.execute_function("dot_product", x1, y1, x2, y2)
        norm1 = np.sqrt(x1**2 + y1**2)
        norm2 = np.sqrt(x2**2 + y2**2)
        cos_angle = dot / (norm1 * norm2)
        return np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    opt_funcs.add_function("angle_between_vectors", angle_between_vectors, "Ángulo entre dos vectores")
    
    # Función 4: Función cuadrática
    def quadratic_function(x, a=1, b=0, c=0):
        """Función cuadrática f(x) = ax² + bx + c"""
        return a * x**2 + b * x + c
    
    opt_funcs.add_function("quadratic_function", quadratic_function, "Función cuadrática f(x) = ax² + bx + c")
    
    print("\n4. PROBANDO NUEVAS FUNCIONES:")
    print("-" * 40)
    
    # Probar distancia euclidiana
    dist = opt_funcs.execute_function("euclidean_distance", 0, 0, 3, 4)
    print(f"Distancia entre (0,0) y (3,4): {dist}")
    
    # Probar producto escalar
    dot = opt_funcs.execute_function("dot_product", 1, 2, 3, 4)
    print(f"Producto escalar entre (1,2) y (3,4): {dot}")
    
    # Probar ángulo entre vectores
    angle = opt_funcs.execute_function("angle_between_vectors", 1, 0, 0, 1)
    print(f"Ángulo entre (1,0) y (0,1): {angle:.4f} radianes ({np.degrees(angle):.1f}°)")
    
    # Probar función cuadrática
    quad_result = opt_funcs.execute_function("quadratic_function", 2, a=1, b=2, c=1)
    print(f"f(2) = 1*2² + 2*2 + 1 = {quad_result}")
    
    print("\n5. OPTIMIZACIÓN:")
    print("-" * 40)
    
    # Optimizar recompensa
    result = opt_funcs.optimize_reward(n_points=50)
    print(f"Optimización de recompensa:")
    print(f"  Mínimo: {result['minimum_reward']:.6f}")
    print(f"  En punto: ({result['optimal_x']:.4f}, {result['optimal_y']:.4f})")
    print(f"  Método: {result['method']}")
    
    print("\n6. RESULTADOS ALMACENADOS:")
    print("-" * 40)
    
    all_results = opt_funcs.get_results()
    print("Resultados de todas las funciones ejecutadas:")
    for func_name, result in all_results.items():
        if isinstance(result, np.ndarray):
            print(f"  {func_name}: {result.flatten()}")
        else:
            print(f"  {func_name}: {result}")
    
    print("\n7. FUNCIONES FINALES DISPONIBLES:")
    opt_funcs.list_functions()
    
    print("\n" + "="*60)
    print("EJEMPLO COMPLETADO EXITOSAMENTE")
    print("="*60)
    
    # Opcional: Descomenta para ver la visualización
    # print("\nGenerando visualización...")
    # opt_funcs.visualize_transformations()

if __name__ == "__main__":
    main() 