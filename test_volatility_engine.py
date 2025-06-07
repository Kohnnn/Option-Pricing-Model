import unittest
from unittest.mock import MagicMock
from volatility_engine import VolatilityEngine
from data_provider import FinancialDataAPI

class TestVolatilityEngine(unittest.TestCase):
    """
    Unit tests for the VolatilityEngine class.
    """

    def setUp(self):
        """Set up the test environment with a mock data provider."""
        self.mock_data_provider = MagicMock(spec=FinancialDataAPI)
        self.volatility_engine = VolatilityEngine(self.mock_data_provider)

    def test_calculate_historical_volatility_success(self):
        """Test successful calculation of historical volatility."""
        # Mock the historical data to return a list of closing prices
        self.mock_data_provider.get_historical_stock_prices.return_value = [
            {'c': 150}, {'c': 151}, {'c': 150.5}, {'c': 152}, {'c': 153}, {'c': 152.5}
        ] * 50 # Repeat to meet window size

        volatility = self.volatility_engine.calculate_historical_volatility("AAPL", window=252)
        self.assertIsInstance(volatility, float)
        self.assertGreater(volatility, 0)

    def test_calculate_historical_volatility_insufficient_data(self):
        """Test that an error is raised if there is not enough historical data."""
        self.mock_data_provider.get_historical_stock_prices.return_value = [{'c': 150}] * 10
        with self.assertRaises(ValueError):
            self.volatility_engine.calculate_historical_volatility("AAPL", window=20)

    def test_calculate_implied_volatility_success(self):
        """Test successful calculation of implied volatility."""
        # Mock the required market data
        self.mock_data_provider.get_latest_stock_price.return_value = 150.0
        self.mock_data_provider.get_risk_free_rate.return_value = 0.05
        # Mock historical volatility to be used as an initial guess
        self.mock_data_provider.get_historical_stock_prices.return_value = [{'c': 150}] * 252

        # Parameters for the option
        strike_price = 155.0
        time_to_expiry = 0.25  # 3 months
        option_market_price = 5.0 # Example market price

        implied_vol = self.volatility_engine.calculate_implied_volatility(
            "AAPL", strike_price, time_to_expiry, option_market_price
        )
        self.assertIsInstance(implied_vol, float)
        self.assertGreater(implied_vol, 0)

    def test_calculate_implied_volatility_missing_data(self):
        """Test that an error is raised if market data is missing."""
        self.mock_data_provider.get_latest_stock_price.return_value = None
        with self.assertRaises(ValueError):
            self.volatility_engine.calculate_implied_volatility("AAPL", 155.0, 0.25, 5.0)

if __name__ == '__main__':
    unittest.main()