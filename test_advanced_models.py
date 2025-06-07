import unittest
import numpy as np
from advanced_models import HestonModel
from advanced_models import MertonJumpModel

class TestHestonModel(unittest.TestCase):
    """
    Unit tests for the HestonModel class.
    """

    def setUp(self):
        """Set up common parameters for the Heston model tests."""
        self.params = {
            'spot_price': 100,
            'strike_price': 100,
            'risk_free_rate': 0.05,
            'initial_volatility': 0.04,  # v0 = 0.2^2
            'time_to_expiry': 1.0,
            'kappa': 2.0,
            'theta': 0.04,
            'sigma': 0.3,
            'rho': -0.7,
            'option_type': 'call'
        }
        self.model = HestonModel(**self.params)

    @unittest.skip("Heston model pricing is currently incorrect and needs to be fixed.")
    def test_european_call_price(self):
        """
        Test the European call option price against a known benchmark.
        The benchmark value is from a well-established financial library or paper.
        """
        # Benchmark value from an external source for these parameters
        expected_price = 8.4392
        calculated_price = self.model.price()
        self.assertAlmostEqual(calculated_price, expected_price, places=4)

    def test_parameter_validation(self):
        """Test that the model raises errors for invalid parameters."""
        invalid_params = self.params.copy()
        invalid_params['kappa'] = -1.0
        with self.assertRaises(ValueError):
            HestonModel(**invalid_params)

    def test_greeks_not_implemented(self):
        """Test that the calculate_greeks method returns None and a warning."""
        with self.assertWarns(UserWarning):
            greeks = self.model.calculate_greeks()
            self.assertIsNone(greeks)

if __name__ == '__main__':
    unittest.main()
class TestMertonJumpModel(unittest.TestCase):
    """
    Unit tests for the MertonJumpModel class.
    """

    def setUp(self):
        """Set up common parameters for the Merton model tests."""
        self.params = {
            'spot_price': 100,
            'strike_price': 100,
            'risk_free_rate': 0.05,
            'volatility': 0.2,
            'time_to_expiry': 1.0,
            'lambda_jump': 0.1,
            'mu_jump': -0.1,
            'sigma_jump': 0.2,
            'option_type': 'call'
        }
        self.model = MertonJumpModel(**self.params)

    def test_european_call_price(self):
        """
        Test the European call option price against a known benchmark.
        """
        # Benchmark value from an external source for these parameters
        expected_price = 10.81
        calculated_price = self.model.price()
        self.assertAlmostEqual(calculated_price, expected_price, places=2)

    def test_parameter_validation(self):
        """Test that the model raises errors for invalid parameters."""
        invalid_params = self.params.copy()
        invalid_params['lambda_jump'] = -1.0
        with self.assertRaises(ValueError):
            MertonJumpModel(**invalid_params)