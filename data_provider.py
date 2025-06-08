import os
import requests
from cachetools import cached, TTLCache
import datetime

class FinancialDataAPI:
    """
    Provides financial data from Alpha Vantage API.
    This class is responsible for fetching all external financial data,
    including stock prices, historical data, and risk-free rates.
    """
    def __init__(self, api_key=None):
        api_keys_str = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
        if not api_keys_str:
            self.api_keys = []
        else:
            self.api_keys = [key.strip() for key in api_keys_str.split(',')]
        self.base_url = "https://www.alphavantage.co/query"
        self.current_key_index = 0

    def get_next_api_key(self):
        """
        Rotates through the available API keys.
        """
        if not self.api_keys:
            return None
        key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return key

    def validate_api_key(self):
        """
        Validates the API key by making a lightweight test call.
        A successful call to the GLOBAL_QUOTE endpoint for a major ticker
        should return a 'Global Quote' object. An invalid key will not.
        """
        if not self.api_keys:
            return False
        
        initial_key_index = self.current_key_index
        
        while True:
            api_key = self.get_next_api_key()
            if not api_key:
                return False

            try:
                params = {
                    "function": "GLOBAL_QUOTE",
                    "symbol": "IBM",  # Use a standard, reliable ticker for testing
                    "apikey": api_key
                }
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()

                if "Error Message" not in data and "Information" not in data and data.get("Global Quote"):
                    return True
            except (requests.exceptions.RequestException, ValueError, KeyError):
                pass  # Try the next key

            if self.current_key_index == initial_key_index:
                # Cycled through all keys and none are valid
                return False

    @cached(cache=TTLCache(maxsize=100, ttl=300))
    def get_latest_stock_price(self, ticker):
        """
        Fetches the latest stock price for a given ticker.
        
        Args:
            ticker (str): The stock ticker symbol.
            
        Returns:
            float: The latest stock price, or None if not available.
        """
        api_key = self.get_next_api_key()
        if not api_key:
            raise ValueError("An API key is required to fetch latest stock prices.")
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": ticker,
            "apikey": api_key
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()
        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")
        if "Information" in data:
            raise ValueError(f"API Info: {data['Information']}")
        global_quote = data.get('Global Quote', {})
        price_str = global_quote.get('05. price')
        if price_str is None:
            # If the key is invalid, Alpha Vantage often returns an empty 'Global Quote'
            # or a note about usage.
            if not global_quote:
                 raise ValueError("Invalid API key, API call limit reached, or no data available. Please check your key and API usage.")
            return None
        return float(price_str)

    @cached(cache=TTLCache(maxsize=100, ttl=300))
    def get_historical_stock_prices(self, ticker, outputsize='compact'):
        """
        Fetches historical daily stock prices.
        
        Args:
            ticker (str): The stock ticker symbol.
            outputsize (str): 'compact' for latest 100 data points, 'full' for the full-length time series.
            
        Returns:
            list: A sorted list of historical price data points (dictionaries).
        """
        api_key = self.get_next_api_key()
        if not api_key:
            raise ValueError("An API key is required to fetch historical stock prices.")
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": outputsize,
            "apikey": api_key
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")
        if "Information" in data:
            raise ValueError(f"API Info: {data['Information']}")

        time_series = data.get("Time Series (Daily)", {})
        results = []
        for date, values in time_series.items():
            results.append({
                't': int(datetime.datetime.strptime(date, '%Y-%m-%d').timestamp() * 1000),
                'o': float(values['1. open']),
                'h': float(values['2. high']),
                'l': float(values['3. low']),
                'c': float(values['4. close']),
                'v': int(values['5. volume'])
            })
        return sorted(results, key=lambda x: x['t'])

    @cached(cache=TTLCache(maxsize=1, ttl=86400)) # Cache for 24 hours
    def get_risk_free_rate(self):
        """
        Fetches the latest 3-month Treasury yield as a proxy for the risk-free rate.
        
        Returns:
            float: The risk-free rate as a decimal (e.g., 0.05 for 5%).
        """
        api_key = self.get_next_api_key()
        if not api_key:
            raise ValueError("An API key is required to fetch the risk-free rate.")
        params = {
            "function": "TREASURY_YIELD",
            "interval": "daily",
            "maturity": "3month",
            "apikey": api_key
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()
        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")
        if "Information" in data:
            raise ValueError(f"API Info: {data['Information']}")
        
        yield_data = data.get('data', [])
        if yield_data:
            latest_yield = yield_data[0].get('value')
            if latest_yield and latest_yield != ".":
                return float(latest_yield) / 100
        return 0.05  # Fallback to a default value if API fails