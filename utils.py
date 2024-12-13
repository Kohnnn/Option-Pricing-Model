import numpy as np
from scipy.stats import norm

def cumulative_normal_distribution(x):
    """
    Compute the cumulative standard normal distribution.
    
    Args:
        x (float): Input value
    
    Returns:
        float: Cumulative probability
    """
    return norm.cdf(x)

def standard_normal_distribution(x):
    """
    Compute the standard normal probability density function.
    
    Args:
        x (float): Input value
    
    Returns:
        float: Probability density
    """
    return norm.pdf(x)

def validate_option_parameters(spot_price, strike_price, risk_free_rate, volatility, time_to_expiry):
    """
    Validate input parameters for option pricing.
    
    Args:
        spot_price (float): Current stock price
        strike_price (float): Option strike price
        risk_free_rate (float): Risk-free interest rate
        volatility (float): Stock price volatility
        time_to_expiry (float): Time to option expiration
    
    Raises:
        ValueError: If any parameter is invalid
    """
    if spot_price <= 0:
        raise ValueError("Spot price must be positive")
    if strike_price <= 0:
        raise ValueError("Strike price must be positive")
    if time_to_expiry <= 0:
        raise ValueError("Time to expiry must be positive")
    if volatility < 0:
        raise ValueError("Volatility cannot be negative")
