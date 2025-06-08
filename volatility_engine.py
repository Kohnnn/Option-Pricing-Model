import numpy as np
import datetime
import logging
from models import BlackScholesOption

class VolatilityEngine:
    """
    Calculates historical and implied volatility for financial assets.
    """
    def __init__(self, data_provider):
        self.data_provider = data_provider

    def calculate_historical_volatility(self, ticker, window=252, historical_data=None):
        """
        Calculates the annualized historical volatility of a stock.
        
        Args:
            ticker (str): The stock ticker symbol.
            window (int): The number of trading days for the volatility calculation.
            historical_data (list, optional): Pre-fetched historical data to avoid redundant API calls.
            
        Returns:
            float: The annualized historical volatility as a decimal.
        """
        if historical_data is None:
            historical_data = self.data_provider.get_historical_stock_prices(ticker, outputsize='full')
        
        if not historical_data or len(historical_data) < window:
            # Try to fetch with a larger window if the provided data is insufficient
            if historical_data is None:
                historical_data = self.data_provider.get_historical_stock_prices(ticker, outputsize='full')
            
            if not historical_data or len(historical_data) < window:
                logging.warning(f"Insufficient historical data for {ticker} to calculate volatility with window {window}. Returning default value 0.20.")
                return 0.20
            
        closes = [d['c'] for d in historical_data[-window:]]
        
        log_returns = np.log(np.array(closes[1:]) / np.array(closes[:-1]))
        
        daily_volatility = np.std(log_returns)
        
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        return annualized_volatility

    def calculate_implied_volatility(self, ticker, strike_price, time_to_expiry, option_market_price, option_type='call'):
        """
        Calculates the implied volatility using the Black-Scholes model.
        
        Args:
            ticker (str): The stock ticker symbol.
            strike_price (float): The option's strike price.
            time_to_expiry (float): The time to expiration in years.
            option_market_price (float): The current market price of the option.
            option_type (str): 'call' or 'put'.
            
        Returns:
            float: The implied volatility as a decimal.
        """
        spot_price = self.data_provider.get_latest_stock_price(ticker)
        risk_free_rate = self.data_provider.get_risk_free_rate()

        if spot_price is None or risk_free_rate is None:
            raise ValueError("Could not retrieve required market data (spot price or risk-free rate).")

        # Initial guess for volatility
        initial_volatility = self.calculate_historical_volatility(ticker)

        bs_option = BlackScholesOption(
            spot_price=spot_price,
            strike_price=strike_price,
            risk_free_rate=risk_free_rate,
            volatility=initial_volatility,
            time_to_expiry=time_to_expiry,
            option_type=option_type
        )

        try:
            implied_vol = bs_option.implied_volatility(option_market_price)
            return implied_vol
        except ValueError as e:
            # Re-raise with a more informative message
            raise ValueError(f"Could not calculate implied volatility: {e}")

    def get_implied_volatility_from_chain(self, ticker, strike_price, expiry_date):
        """
        Fetches implied volatility from the options chain.

        Args:
            ticker (str): The stock ticker symbol.
            strike_price (float): The option's strike price.
            expiry_date (datetime.date): The option's expiry date.

        Returns:
            float: The implied volatility, or None if not found.
        """
        options_chain = self.data_provider.get_options_chain(ticker)
        if not options_chain:
            return None

        expiry_date_str = expiry_date.strftime('%Y-%m-%d')

        for option in options_chain:
            if (option['details']['strike_price'] == strike_price and
                option['details']['expiration_date'] == expiry_date_str):
                return option.get('implied_volatility')

        return None