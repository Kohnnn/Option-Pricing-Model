"""
Unit tests for the stochastic processes module.
"""

import unittest
import numpy as np
from stochastic_processes import OrnsteinUhlenbeckProcess

class TestOrnsteinUhlenbeckProcess(unittest.TestCase):
    """
    Test cases for the OrnsteinUhlenbeckProcess class.
    """
    def setUp(self):
        """
        Set up a new Ornstein-Uhlenbeck process for each test.
        """
        self.process = OrnsteinUhlenbeckProcess(speed=0.1, mean=100, volatility=2.0)

    def test_simulation_path_shape(self):
        """
        Test the shape of the simulated path.
        """
        path = self.process.simulate_path(initial_value=95, T=1.0, num_steps=100)
        self.assertEqual(path.shape, (101,))

    def test_simulation_initial_value(self):
        """
        Test that the simulation starts at the correct initial value.
        """
        initial_value = 95
        path = self.process.simulate_path(initial_value=initial_value, T=1.0, num_steps=100)
        self.assertEqual(path[0], initial_value)

    def test_simulation_reproducibility(self):
        """
        Test that the simulation is reproducible with a fixed seed.
        """
        np.random.seed(42)
        path1 = self.process.simulate_path(initial_value=95, T=1.0, num_steps=100)
        np.random.seed(42)
        path2 = self.process.simulate_path(initial_value=95, T=1.0, num_steps=100)
        np.testing.assert_array_equal(path1, path2)
    def test_statistical_properties(self):
        """
        Test the statistical properties (mean and variance) of the process.
        """
        initial_value = 95
        T = 1.0
        num_steps = 100
        num_paths = 5000  # Use a large number of paths for statistical significance
        
        paths = np.array([self.process.simulate_path(initial_value, T, num_steps) for _ in range(num_paths)])
        
        # Check the mean at the end of the period
        expected_mean = initial_value * np.exp(-self.process.speed * T) + self.process.mean * (1 - np.exp(-self.process.speed * T))
        actual_mean = np.mean(paths[:, -1])
        self.assertAlmostEqual(actual_mean, expected_mean, delta=0.5)
        
        # Check the variance at the end of the period
        expected_variance = (self.process.volatility**2 / (2 * self.process.speed)) * (1 - np.exp(-2 * self.process.speed * T))
        actual_variance = np.var(paths[:, -1])
        self.assertAlmostEqual(actual_variance, expected_variance, delta=0.5)

if __name__ == '__main__':
    unittest.main()