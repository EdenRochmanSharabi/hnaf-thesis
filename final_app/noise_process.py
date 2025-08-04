import numpy as np

class OUNoise:
    """
    Implementa un proceso de Ornstein-Uhlenbeck para la exploración 
    correlacionada en el tiempo.
    """
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """
        Inicializa los parámetros del ruido.

        Args:
            size (int): Dimensión del espacio de acción.
            seed (int): Semilla para la aleatoriedad para reproducibilidad.
            mu (float): Punto medio al que el ruido tiende a volver (generalmente 0).
            theta (float): La "fuerza" con la que el ruido vuelve a la media.
            sigma (float): La volatilidad o magnitud del ruido.
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        np.random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reinicia el estado interno del ruido a la media (mu). Se llama al inicio de cada episodio."""
        self.state = np.copy(self.mu)

    def sample(self):
        """Actualiza y devuelve una muestra del proceso de ruido."""
        x = self.state
        # La fórmula de OU: dx = theta * (mu - x) + sigma * RuidoGaussiano
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state 