"""
Integration tests for the entire application.
"""

import unittest
import numpy as np
from unittest.mock import patch
from data_provider import FinancialDataAPI
from portfolio import Portfolio
from risk_management import ValueAtRisk, StressTester

class TestIntegration(unittest.TestCase):
    """
    Test cases for the integration of different modules.
    """
    def setUp(self):
        """
        Set up the necessary objects for integration testing.
        """
        # We will use a mock data provider to avoid actual API calls
        self.mock_data_provider = unittest.mock.MagicMock(spec=FinancialDataAPI)
        self.mock_data_provider.get_latest_stock_price.return_value = 100.0
        
        historical_prices = [{'c': 100 + i} for i in range(-10, 10)]
        self.mock_data_provider.get_historical_stock_prices.return_value = historical_prices
        
        self.portfolio = Portfolio()
        self.value_at_risk = ValueAtRisk(self.portfolio)
        self.stress_tester = StressTester(self.portfolio)

    def test_portfolio_creation_and_valuation(self):
        """
        Test creating a portfolio, adding assets, and calculating its value.
        """
        self.portfolio.add_position("AAPL", 10, 100)
        self.portfolio.add_position("GOOGL", 5, 200)
        
        self.assertEqual(len(self.portfolio.positions), 2)
        self.assertAlmostEqual(self.portfolio.get_total_value(), 2000.0)

    def test_risk_model_calculations(self):
        """
        Test that the risk models run without errors.
        """
        self.portfolio.add_position("AAPL", 10, 100)
        
        # We are not testing the accuracy of the calculations here,
        # just that they run without raising exceptions.
        # Mocking returns for historical VaR
        returns = np.random.normal(-0.01, 0.02, 1000)
        self.assertIsInstance(self.value_at_risk.historical_var(returns), float)
        self.assertIsInstance(self.value_at_risk.historical_cvar(returns), float)
        self.assertIsInstance(self.value_at_risk.parametric_var(returns), float)
        
        pnl_dist = self.value_at_risk.monte_carlo_simulation(1000, 1)
        self.assertIsInstance(self.value_at_risk.get_var_from_pnl(pnl_dist), float)
        self.assertIsInstance(self.value_at_risk.get_cvar_from_pnl(pnl_dist), float)

    def test_stress_testing(self):
        """
        Test the stress testing functionality.
        """
        self.portfolio.add_position("AAPL", 10, 100)
        scenarios = [{'name': 'Scenario 1', 'symbol': 'AAPL', 'type': 'percentage', 'shock': -0.1}]
        results = self.stress_tester.run_stress_test(scenarios)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results['Scenario 1'], -100.0)

    def test_add_and_remove_assets(self):
        """
        Test adding and removing assets from the portfolio.
        """
        self.portfolio.add_position("AAPL", 10, 100)
        self.portfolio.add_position("GOOGL", 5, 200)
        self.assertEqual(len(self.portfolio.positions), 2)

        self.portfolio.remove_position("AAPL")
        self.assertEqual(len(self.portfolio.positions), 1)
        self.assertNotIn("AAPL", self.portfolio.positions)

    def test_var_cvar_calculations(self):
        """
        Test VaR and CVaR calculations with different parameters.
        """
        self.portfolio.add_position("AAPL", 10, 100)
        
        # Test with different confidence levels
        self.value_at_risk.confidence_level = 0.95
        returns = np.random.normal(-0.01, 0.02, 1000)
        self.assertIsInstance(self.value_at_risk.historical_var(returns), float)
        self.value_at_risk.confidence_level = 0.99
        self.assertIsInstance(self.value_at_risk.historical_var(returns), float)

    def test_stress_testing_scenarios(self):
        """
        Test the stress testing functionality with various scenarios.
        """
        self.portfolio.add_position("AAPL", 10, 100)
        self.portfolio.add_position("GOOGL", 5, 200)

        scenarios = [
            {'name': 'Scenario 1', 'symbol': 'AAPL', 'type': 'percentage', 'shock': -0.1},
            {'name': 'Scenario 2', 'symbol': 'GOOGL', 'type': 'absolute', 'shock': 10}
        ]
        results = self.stress_tester.run_stress_test(scenarios)
        self.assertEqual(len(results), 2)
        self.assertIn('Scenario 1', results)
        self.assertIn('Scenario 2', results)
        self.assertAlmostEqual(results['Scenario 1'], -100.0)
        self.assertAlmostEqual(results['Scenario 2'], 50.0)

if __name__ == '__main__':
    unittest.main()