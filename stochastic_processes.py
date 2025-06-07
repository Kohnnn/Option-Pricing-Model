"""
This module defines classes for various stochastic processes used in financial modeling.
"""

import numpy as np

class OrnsteinUhlenbeckProcess:
    """
    A class to represent the Ornstein-Uhlenbeck process.

    This process is often used to model mean-reverting phenomena.

    Attributes:
        speed (float): The speed of mean reversion.
        mean (float): The long-term mean of the process.
        volatility (float): The volatility of the process.
    """
    def __init__(self, speed, mean, volatility):
        """
        Initializes the Ornstein-Uhlenbeck process.

        Args:
            speed (float): The speed of mean reversion (theta).
            mean (float): The long-term mean (mu).
            volatility (float): The volatility (sigma).
        """
        self.speed = speed
        self.mean = mean
        self.volatility = volatility

    def simulate_path(self, initial_value, T, num_steps):
        """
        Simulates a path of the Ornstein-Uhlenbeck process.

        Args:
            initial_value (float): The starting value of the process.
            T (float): The total time period for the simulation.
            num_steps (int): The number of time steps in the simulation.

        Returns:
            A numpy array representing the simulated path.
        """
        dt = T / num_steps
        path = np.zeros(num_steps + 1)
        path[0] = initial_value
        for t in range(1, num_steps + 1):
            dW = np.random.normal(0, np.sqrt(dt))
            path[t] = path[t-1] + self.speed * (self.mean - path[t-1]) * dt + self.volatility * dW
        return path
class GeometricBrownianMotion:
    """
    A class to represent Geometric Brownian Motion (GBM).

    This process is commonly used to model stock prices.

    Attributes:
        mu (float): The drift coefficient.
        sigma (float): The diffusion coefficient (volatility).
    """
    def __init__(self, mu, sigma):
        """
        Initializes the Geometric Brownian Motion process.

        Args:
            mu (float): The drift coefficient.
            sigma (float): The diffusion coefficient (volatility).
        """
        self.mu = mu
        self.sigma = sigma

    def simulate_path(self, initial_value, T, num_steps):
        """
        Simulates a path of the Geometric Brownian Motion process.

        Args:
            initial_value (float): The starting value of the process.
            T (float): The total time period for the simulation.
            num_steps (int): The number of time steps in the simulation.

        Returns:
            A numpy array representing the simulated path.
        """
        dt = T / num_steps
        path = np.zeros(num_steps + 1)
        path[0] = initial_value
        for t in range(1, num_steps + 1):
            dW = np.random.normal(0, np.sqrt(dt))
            path[t] = path[t-1] * np.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * dW)
        return path