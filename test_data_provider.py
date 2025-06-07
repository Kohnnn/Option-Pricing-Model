import unittest
from unittest.mock import patch, Mock
import os
from data_provider import FinancialDataAPI

class TestFinancialDataAPI(unittest.TestCase):
    """
    Unit tests for the FinancialDataAPI class.
    """

    @patch('requests.get')
    def test_get_latest_stock_price_success(self, mock_get):
        """Test successful retrieval of the latest stock price."""
        data_provider = FinancialDataAPI(api_key="test_api_key")
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Global Quote": {
                "05. price": "150.75"
            }
        }
        mock_get.return_value = mock_response

        price = data_provider.get_latest_stock_price("AAPL")
        self.assertEqual(price, 150.75)
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_get_latest_stock_price_failure(self, mock_get):
        """Test failure case for retrieving the latest stock price."""
        data_provider = FinancialDataAPI(api_key="test_api_key")
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"Global Quote": {}}
        mock_get.return_value = mock_response

        price = data_provider.get_latest_stock_price("FAIL")
        self.assertIsNone(price)

    @patch('requests.get')
    def test_get_historical_stock_prices_success(self, mock_get):
        """Test successful retrieval of historical stock prices."""
        data_provider = FinancialDataAPI(api_key="test_api_key")
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Time Series (Daily)": {
                "2023-01-02": {
                    "1. open": "130.0", "2. high": "132.0", "3. low": "129.0", "4. close": "131.0", "6. volume": "100000"
                },
                "2023-01-01": {
                    "1. open": "128.0", "2. high": "130.0", "3. low": "127.0", "4. close": "129.0", "6. volume": "90000"
                }
            }
        }
        mock_get.return_value = mock_response

        prices = data_provider.get_historical_stock_prices("AAPL")
        self.assertEqual(len(prices), 2)
        self.assertEqual(prices[0]['c'], 129.0) # Check sorting
        self.assertEqual(prices[1]['c'], 131.0)

    @patch('requests.get')
    def test_get_risk_free_rate_success(self, mock_get):
        """Test successful retrieval of the risk-free rate."""
        data_provider = FinancialDataAPI(api_key="test_api_key")
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"date": "2023-01-02", "value": "4.5"},
                {"date": "2023-01-01", "value": "4.4"}
            ]
        }
        mock_get.return_value = mock_response

        rate = data_provider.get_risk_free_rate()
        self.assertEqual(rate, 0.045)

    @patch('requests.get')
    def test_get_risk_free_rate_failure(self, mock_get):
        """Test failure case for retrieving the risk-free rate."""
        data_provider = FinancialDataAPI(api_key="test_api_key")
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        mock_get.return_value = mock_response

        rate = data_provider.get_risk_free_rate()
        self.assertEqual(rate, 0.05)

    @patch.dict(os.environ, {}, clear=True)
    def test_api_key_missing(self):
        """Test that a ValueError is raised if the API key is missing."""
        with self.assertRaises(ValueError):
            FinancialDataAPI(api_key=None)

if __name__ == '__main__':
    unittest.main()