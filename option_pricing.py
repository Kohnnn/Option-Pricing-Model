from models import BlackScholesOption, BinomialTreeOption

def calculate_option_price(pricing_model='black_scholes', **kwargs):
    """
    Calculate option price using specified pricing model
    
    Args:
        pricing_model (str): Option pricing model to use
        **kwargs: Parameters for option pricing
    
    Returns:
        float: Option price
    """
    if pricing_model == 'black_scholes':
        option = BlackScholesOption(**kwargs)
    elif pricing_model == 'binomial_tree':
        option = BinomialTreeOption(**kwargs)
    else:
        raise ValueError(f"Unsupported pricing model: {pricing_model}")
    
    return option.price()

def calculate_option_greeks(pricing_model='black_scholes', **kwargs):
    """
    Calculate option Greeks
    
    Args:
        pricing_model (str): Option pricing model to use
        **kwargs: Parameters for option pricing
    
    Returns:
        dict: Option Greeks (delta, gamma, vega, theta)
    """
    if pricing_model == 'black_scholes':
        option = BlackScholesOption(**kwargs)
        return {
            'delta': option.delta(),
            'gamma': option.gamma(),
            'vega': option.vega(),
            'theta': option.theta()
        }
    else:
        raise ValueError("Greeks calculation only supported for Black-Scholes model")

# Example usage
if __name__ == "__main__":
    # Example option pricing
    call_price = calculate_option_price(
        pricing_model='black_scholes',
        spot_price=100,
        strike_price=100,
        risk_free_rate=0.05,
        volatility=0.2,
        time_to_expiry=1.0,
        option_type='call'
    )
    
    print(f"Call Option Price: ${call_price:.2f}")
    
    # Example Greeks calculation
    greeks = calculate_option_greeks(
        pricing_model='black_scholes',
        spot_price=100,
        strike_price=100,
        risk_free_rate=0.05,
        volatility=0.2,
        time_to_expiry=1.0,
        option_type='call'
    )
    
    print("Option Greeks:")
    for greek, value in greeks.items():
        print(f"{greek.capitalize()}: {value:.4f}")
