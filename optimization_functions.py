import numpy as np
from typing import Tuple, Callable, Dict, Any
import matplotlib.pyplot as plt

class OptimizationFunctions:
    """
    Clase para almacenar y gestionar funciones de optimización.
    Permite agregar, ejecutar y visualizar diferentes funciones matemáticas.
    """
    
    def __init__(self):
        """Inicializa la clase con un diccionario para almacenar las funciones."""
        self.functions = {}
        self.results = {}
        self._initialize_default_functions()
    
    def _initialize_default_functions(self):
        """Inicializa las funciones por defecto."""
        # Coordenadas iniciales
        self.x0, self.y0 = 1, 1
        
        # Matrices de transformación
        self.A1 = np.array([[1, 50],
                           [-1, 1]])
        self.A2 = np.array([[1, -1],
                           [50, 1]])
        
        # Agregar funciones por defecto
        self.add_function("transform_x1", self._transform_x1)
        self.add_function("transform_x2", self._transform_x2)
        self.add_function("reward_function", self._reward_function)
        self.add_function("calculate_transformations", self._calculate_transformations)
    
    def add_function(self, name: str, func: Callable, description: str = ""):
        """
        Agrega una nueva función al almacén.
        
        Args:
            name: Nombre de la función
            func: Función a almacenar
            description: Descripción de la función
        """
        self.functions[name] = {
            'function': func,
            'description': description
        }
        print(f"Función '{name}' agregada exitosamente.")
    
    def execute_function(self, name: str, *args, **kwargs) -> Any:
        """
        Ejecuta una función almacenada.
        
        Args:
            name: Nombre de la función a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos con nombre
            
        Returns:
            Resultado de la función
        """
        if name not in self.functions:
            raise ValueError(f"Función '{name}' no encontrada. Funciones disponibles: {list(self.functions.keys())}")
        
        func = self.functions[name]['function']
        result = func(*args, **kwargs)
        
        # Guardar resultado
        self.results[name] = result
        return result
    
    def _transform_x1(self, x0: float = None, y0: float = None) -> np.ndarray:
        """
        Aplica la transformación A1 a las coordenadas (x0, y0).
        
        Args:
            x0: Coordenada x inicial (usa self.x0 si no se proporciona)
            y0: Coordenada y inicial (usa self.y0 si no se proporciona)
            
        Returns:
            Vector x1 después de la transformación
        """
        if x0 is None:
            x0 = self.x0
        if y0 is None:
            y0 = self.y0
            
        x1 = self.A1 @ np.array([[x0], [y0]])
        return x1
    
    def _transform_x2(self, x0: float = None, y0: float = None) -> np.ndarray:
        """
        Aplica la transformación A2 a las coordenadas (x0, y0).
        
        Args:
            x0: Coordenada x inicial (usa self.x0 si no se proporciona)
            y0: Coordenada y inicial (usa self.y0 si no se proporciona)
            
        Returns:
            Vector x2 después de la transformación
        """
        if x0 is None:
            x0 = self.x0
        if y0 is None:
            y0 = self.y0
            
        x2 = self.A2 @ np.array([[x0], [y0]])
        return x2
    
    def _reward_function(self, x: float, y: float, x0: float = None, y0: float = None) -> float:
        """
        Calcula la función recompensa: minimizar diferencia de distancias al origen.
        Reward = | ||(x, y)|| - ||(x0, y0)|| |
        
        Args:
            x: Coordenada x del punto
            y: Coordenada y del punto
            x0: Coordenada x inicial (usa self.x0 si no se proporciona)
            y0: Coordenada y inicial (usa self.y0 si no se proporciona)
            
        Returns:
            Valor de la recompensa
        """
        if x0 is None:
            x0 = self.x0
        if y0 is None:
            y0 = self.y0
            
        norm_xy = np.linalg.norm([x, y])
        norm_x0y0 = np.linalg.norm([x0, y0])
        return abs(norm_xy - norm_x0y0)
    
    def _calculate_transformations(self, x0: float = None, y0: float = None) -> Dict[str, np.ndarray]:
        """
        Calcula ambas transformaciones x1 y x2.
        
        Args:
            x0: Coordenada x inicial (usa self.x0 si no se proporciona)
            y0: Coordenada y inicial (usa self.y0 si no se proporciona)
            
        Returns:
            Diccionario con x1 y x2
        """
        if x0 is None:
            x0 = self.x0
        if y0 is None:
            y0 = self.y0
            
        x1 = self._transform_x1(x0, y0)
        x2 = self._transform_x2(x0, y0)
        
        return {
            'x1': x1,
            'x2': x2,
            'original': np.array([[x0], [y0]])
        }
    
    def list_functions(self) -> None:
        """Lista todas las funciones disponibles."""
        print("Funciones disponibles:")
        for name, info in self.functions.items():
            print(f"  - {name}: {info['description']}")
    
    def get_results(self, name: str = None) -> Any:
        """
        Obtiene los resultados de una función específica o todos los resultados.
        
        Args:
            name: Nombre de la función (si es None, retorna todos los resultados)
            
        Returns:
            Resultados de la función o diccionario con todos los resultados
        """
        if name is None:
            return self.results
        return self.results.get(name, None)
    
    def visualize_transformations(self, x0: float = None, y0: float = None):
        """
        Visualiza las transformaciones en un gráfico 2D.
        
        Args:
            x0: Coordenada x inicial (usa self.x0 si no se proporciona)
            y0: Coordenada y inicial (usa self.y0 si no se proporciona)
        """
        if x0 is None:
            x0 = self.x0
        if y0 is None:
            y0 = self.y0
            
        # Calcular transformaciones
        transforms = self._calculate_transformations(x0, y0)
        
        # Crear gráfico
        plt.figure(figsize=(10, 8))
        
        # Punto original
        plt.scatter(x0, y0, color='blue', s=100, label='Punto original', zorder=5)
        plt.annotate(f'({x0}, {y0})', (x0, y0), xytext=(5, 5), textcoords='offset points')
        
        # Transformación x1
        x1_x, x1_y = transforms['x1'][0, 0], transforms['x1'][1, 0]
        plt.scatter(x1_x, x1_y, color='red', s=100, label='x1 (A1)', zorder=5)
        plt.annotate(f'x1 ({x1_x:.1f}, {x1_y:.1f})', (x1_x, x1_y), xytext=(5, 5), textcoords='offset points')
        
        # Transformación x2
        x2_x, x2_y = transforms['x2'][0, 0], transforms['x2'][1, 0]
        plt.scatter(x2_x, x2_y, color='green', s=100, label='x2 (A2)', zorder=5)
        plt.annotate(f'x2 ({x2_x:.1f}, {x2_y:.1f})', (x2_x, x2_y), xytext=(5, 5), textcoords='offset points')
        
        # Líneas de conexión
        plt.plot([x0, x1_x], [y0, x1_y], 'r--', alpha=0.5, label='Transformación A1')
        plt.plot([x0, x2_x], [y0, x2_y], 'g--', alpha=0.5, label='Transformación A2')
        
        # Configurar gráfico
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Visualización de Transformaciones Lineales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Mostrar recompensas
        reward_x1 = self._reward_function(x1_x, x1_y, x0, y0)
        reward_x2 = self._reward_function(x2_x, x2_y, x0, y0)
        
        plt.figtext(0.02, 0.02, f'Recompensa x1: {reward_x1:.2f}\nRecompensa x2: {reward_x2:.2f}', 
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def optimize_reward(self, method: str = 'grid', bounds: Tuple[Tuple[float, float], Tuple[float, float]] = None, 
                       n_points: int = 100) -> Dict[str, Any]:
        """
        Optimiza la función recompensa para encontrar el mínimo.
        
        Args:
            method: Método de optimización ('grid' para búsqueda en grilla)
            bounds: Límites de búsqueda ((x_min, x_max), (y_min, y_max))
            n_points: Número de puntos en cada dimensión para la grilla
            
        Returns:
            Diccionario con el mínimo encontrado y su posición
        """
        if bounds is None:
            # Límites por defecto basados en las transformaciones
            transforms = self._calculate_transformations()
            x1_x, x1_y = transforms['x1'][0, 0], transforms['x1'][1, 0]
            x2_x, x2_y = transforms['x2'][0, 0], transforms['x2'][1, 0]
            
            x_min = min(self.x0, x1_x, x2_x) - 10
            x_max = max(self.x0, x1_x, x2_x) + 10
            y_min = min(self.y0, x1_y, x2_y) - 10
            y_max = max(self.y0, x1_y, x2_y) + 10
            
            bounds = ((x_min, x_max), (y_min, y_max))
        
        if method == 'grid':
            x_range = np.linspace(bounds[0][0], bounds[0][1], n_points)
            y_range = np.linspace(bounds[1][0], bounds[1][1], n_points)
            
            min_reward = float('inf')
            min_x, min_y = 0, 0
            
            for x in x_range:
                for y in y_range:
                    reward = self._reward_function(x, y)
                    if reward < min_reward:
                        min_reward = reward
                        min_x, min_y = x, y
            
            return {
                'minimum_reward': min_reward,
                'optimal_x': min_x,
                'optimal_y': min_y,
                'method': method
            }
        
        return None


# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancia de la clase
    opt_funcs = OptimizationFunctions()
    
    # Listar funciones disponibles
    opt_funcs.list_functions()
    
    print("\n" + "="*50)
    print("EJEMPLOS DE USO")
    print("="*50)
    
    # Ejecutar transformaciones
    x1 = opt_funcs.execute_function("transform_x1")
    x2 = opt_funcs.execute_function("transform_x2")
    
    print("x1 =\n", x1)
    print("x2 =\n", x2)
    
    # Calcular recompensa
    x, y = 3, 4
    r = opt_funcs.execute_function("reward_function", x, y)
    print(f"Recompensa para ({x}, {y}): {r}")
    
    # Calcular todas las transformaciones
    transforms = opt_funcs.execute_function("calculate_transformations")
    print("\nTodas las transformaciones:")
    for key, value in transforms.items():
        print(f"{key}:\n{value}")
    
    # Optimizar recompensa
    print("\nOptimizando recompensa...")
    result = opt_funcs.optimize_reward()
    print(f"Mínimo encontrado: {result}")
    
    # Visualizar (descomenta la siguiente línea para ver el gráfico)
    # opt_funcs.visualize_transformations() 