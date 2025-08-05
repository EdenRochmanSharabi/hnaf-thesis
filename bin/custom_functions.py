import numpy as np

# Coordenadas iniciales
x0, y0 = 1, 1  # puedes cambiar estos valores

# Definimos los vectores x1 y x2
A1 = np.array([[1, 50],
               [-1, 1]])
x1 = A1 @ np.array([[x0], [y0]])

A2 = np.array([[1, -1],
               [50, 1]])
x2 = A2 @ np.array([[x0], [y0]])

# Imprimir resultados de las transformaciones
print("x1 =\n", x1)
print("x2 =\n", x2)

# Función recompensa: minimizar diferencia de distancias al origen
# Reward = | ||(x, y)|| - ||(x0, y0)|| |
def reward(x, y, x0, y0):
    norm_xy = np.linalg.norm([x, y])
    norm_x0y0 = np.linalg.norm([x0, y0])
    return abs(norm_xy - norm_x0y0)

# Ejemplo de uso
x, y = 3, 4  # ejemplo de punto
r = reward(x, y, x0, y0)
print("Recompensa:", r)

# Funciones de transformación para cada modo
def transform_mode_0(x0, y0):
    """Transformación para el modo 0"""
    A = np.array([[1, 50],
                  [-1, 1]])
    result = A @ np.array([[x0], [y0]])
    return result[0, 0], result[1, 0]

def transform_mode_1(x0, y0):
    """Transformación para el modo 1"""
    A = np.array([[1, -1],
                  [50, 1]])
    result = A @ np.array([[x0], [y0]])
    return result[0, 0], result[1, 0]

# Lista de funciones de transformación (una por modo)
transformation_functions = [transform_mode_0, transform_mode_1]

# Función de recompensa
def reward_function(x, y, x0, y0):
    """Función de recompensa personalizada"""
    norm_xy = np.linalg.norm([x, y])
    norm_x0y0 = np.linalg.norm([x0, y0])
    return abs(norm_xy - norm_x0y0)

