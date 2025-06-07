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
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not provided.")
        self.base_url = "https://www.alphavantage.co/query"

    @cached(cache=TTLCache(maxsize=100, ttl=300))
    def get_latest_stock_price(self, ticker):
        """
        Fetches the latest stock price for a given ticker.
        
        Args:
            ticker (str): The stock ticker symbol.
            
        Returns:
            float: The latest stock price, or None if not available.
        """
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": ticker,
            "apikey": self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json().get('Global Quote', {})
        price_str = data.get('05. price')
        return float(price_str) if price_str else None

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
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": outputsize,
            "apikey": self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        time_series = data.get("Time Series (Daily)", {})
        results = []
        for date, values in time_series.items():
            results.append({
                't': int(datetime.datetime.strptime(date, '%Y-%m-%d').timestamp() * 1000),
                'o': float(values['1. open']),
                'h': float(values['2. high']),
                'l': float(values['3. low']),
                'c': float(values['4. close']),
                'v': int(values['6. volume'])
            })
        return sorted(results, key=lambda x: x['t'])

    @cached(cache=TTLCache(maxsize=1, ttl=86400)) # Cache for 24 hours
    def get_risk_free_rate(self):
        """
        Fetches the latest 3-month Treasury yield as a proxy for the risk-free rate.
        
        Returns:
            float: The risk-free rate as a decimal (e.g., 0.05 for 5%).
        """
        params = {
            "function": "TREASURY_YIELD",
            "interval": "daily",
            "maturity": "3month",
            "apikey": self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json().get('data', [])
        
        if data:
            latest_yield = data[0].get('value')
            if latest_yield and latest_yield != ".":
                return float(latest_yield) / 100
        return 0.05 # Fallback to a default value if API fails