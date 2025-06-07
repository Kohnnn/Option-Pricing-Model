import unittest
from portfolio import Portfolio
import numpy as np
from risk_management import ValueAtRisk, StressTester

class TestRiskManagement(unittest.TestCase):
    """
    Unit tests for the risk management classes.
    """
    def setUp(self):
        """
        Set up a sample portfolio for testing.
        """
        self.portfolio = Portfolio()
        self.portfolio.add_position('AAPL', 100, 150.0)
        self.portfolio.add_position('GOOG', 50, 2800.0)

    def test_value_at_risk_initialization(self):
        """
        Tests that the ValueAtRisk class can be initialized.
        """
        var = ValueAtRisk(self.portfolio)
        self.assertIsInstance(var, ValueAtRisk)
        self.assertEqual(var.portfolio, self.portfolio)

    def test_stress_tester_initialization(self):
        """
        Tests that the StressTester class can be initialized.
        """
        st = StressTester(self.portfolio)
        self.assertIsInstance(st, StressTester)
        self.assertEqual(st.portfolio, self.portfolio)

    def test_historical_var_cvar(self):
        """
        Tests the Historical VaR and CVaR calculations.
        """
        # Sample historical returns
        returns = np.array([-0.02, -0.01, 0.0, 0.01, 0.02, -0.03, 0.04])
        
        # Initialize ValueAtRisk with a 95% confidence level
        var_calculator = ValueAtRisk(self.portfolio, confidence_level=0.95)
        
        # Calculate VaR and CVaR
        calculated_var = var_calculator.historical_var(returns)
        calculated_cvar = var_calculator.historical_cvar(returns)
        
        # Expected VaR is the 5th percentile of the returns
        expected_var = np.percentile(returns, 5)
        
        # Expected CVaR is the mean of returns less than VaR
        expected_cvar = returns[returns < expected_var].mean()
        
        self.assertAlmostEqual(calculated_var, expected_var, places=4)
        self.assertAlmostEqual(calculated_cvar, expected_cvar, places=4)

    def test_parametric_var(self):
        """
        Tests the Parametric VaR calculation.
        """
        returns = np.array([-0.02, -0.01, 0.0, 0.01, 0.02, -0.03, 0.04])
        var_calculator = ValueAtRisk(self.portfolio, confidence_level=0.95)
        
        calculated_var = var_calculator.parametric_var(returns)
        
        mean = np.mean(returns)
        std_dev = np.std(returns)
        z_score = -1.64485  # Z-score for 95% confidence
        expected_var = mean + z_score * std_dev
        
        self.assertAlmostEqual(calculated_var, -0.038, places=2)

    def test_monte_carlo_var(self):
        """
        Tests the Monte Carlo VaR calculation.
        """
        var_calculator = ValueAtRisk(self.portfolio, confidence_level=0.95)
        
        # This is a stochastic test, so we can't have an exact expected value.
        # We'll run it and check if the result is reasonable.
        calculated_var = var_calculator.monte_carlo_var(num_simulations=1000)
        
        # Expected value is hard to pin down, but it should be negative
        self.assertLess(calculated_var, 0)

    def test_run_stress_test(self):
        """
        Tests the stress testing functionality.
        """
        st = StressTester(self.portfolio)
        
        scenarios = [
            {'name': 'AAPL Price Drop 10%', 'symbol': 'AAPL', 'type': 'percentage', 'shock': -0.1},
            {'name': 'GOOG Price Up 20', 'symbol': 'GOOG', 'type': 'absolute', 'shock': 20},
        ]
        
        results = st.run_stress_test(scenarios)
        
        # Expected P&L for AAPL shock: 100 shares * (150 * -0.1) = -1500
        self.assertAlmostEqual(results['AAPL Price Drop 10%'], -1500, places=2)
        
        # Expected P&L for GOOG shock: 50 shares * 20 = 1000
        self.assertAlmostEqual(results['GOOG Price Up 20'], 1000, places=2)

if __name__ == '__main__':
    unittest.main()