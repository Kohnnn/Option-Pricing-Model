from models import BlackScholesOption, BinomialTreeOption, AdvancedOptionPricing
from advanced_models import HestonModel, MertonJumpModel, KouJumpModel

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
    elif pricing_model == 'heston':
        # Extract Heston-specific parameters
        kappa = kwargs.pop('kappa', 2.0)
        theta = kwargs.pop('theta', kwargs.get('initial_volatility', kwargs.get('volatility')**2))
        sigma = kwargs.pop('sigma', 0.3)
        rho = kwargs.pop('rho', -0.7)
        initial_volatility = kwargs.pop('initial_volatility', kwargs.get('volatility')**2)
        
        # Replace volatility with initial_volatility for Heston model
        if 'volatility' in kwargs:
            del kwargs['volatility']
            
        option = HestonModel(
            initial_volatility=initial_volatility,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            rho=rho,
            **kwargs
        )
    elif pricing_model == 'merton_jump':
        # Extract Merton jump model specific parameters
        lambda_jump = kwargs.pop('lambda_jump', 1.0)
        mu_jump = kwargs.pop('mu_jump', -0.1)
        sigma_jump = kwargs.pop('sigma_jump', 0.2)
        
        option = MertonJumpModel(
            lambda_jump=lambda_jump,
            mu_jump=mu_jump,
            sigma_jump=sigma_jump,
            **kwargs
        )
    elif pricing_model == 'kou_jump':
        # Extract Kou jump model specific parameters
        lambda_jump = kwargs.pop('lambda_jump', 1.0)
        p_up = kwargs.pop('p_up', 0.3)
        eta1 = kwargs.pop('eta1', 10.0)
        eta2 = kwargs.pop('eta2', 5.0)
        
        option = KouJumpModel(
            lambda_jump=lambda_jump,
            p_up=p_up,
            eta1=eta1,
            eta2=eta2,
            **kwargs
        )
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
        dict: Option Greeks (delta, gamma, vega, theta, rho)
    """
    if pricing_model == 'black_scholes':
        option = BlackScholesOption(**kwargs)
        return {
            'delta': option.delta(),
            'gamma': option.gamma(),
            'vega': option.vega(),
            'theta': option.theta()
        }
    elif pricing_model == 'heston':
        # Extract Heston-specific parameters
        kappa = kwargs.pop('kappa', 2.0)
        theta = kwargs.pop('theta', kwargs.get('initial_volatility', kwargs.get('volatility')**2))
        sigma = kwargs.pop('sigma', 0.3)
        rho = kwargs.pop('rho', -0.7)
        initial_volatility = kwargs.pop('initial_volatility', kwargs.get('volatility')**2)
        
        # Replace volatility with initial_volatility for Heston model
        if 'volatility' in kwargs:
            del kwargs['volatility']
            
        option = HestonModel(
            initial_volatility=initial_volatility,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            rho=rho,
            **kwargs
        )
        return option.calculate_greeks()
    elif pricing_model == 'merton_jump':
        # Extract Merton jump model specific parameters
        lambda_jump = kwargs.pop('lambda_jump', 1.0)
        mu_jump = kwargs.pop('mu_jump', -0.1)
        sigma_jump = kwargs.pop('sigma_jump', 0.2)
        
        option = MertonJumpModel(
            lambda_jump=lambda_jump,
            mu_jump=mu_jump,
            sigma_jump=sigma_jump,
            **kwargs
        )
        return option.calculate_greeks()
    elif pricing_model == 'kou_jump':
        # Extract Kou jump model specific parameters
        lambda_jump = kwargs.pop('lambda_jump', 1.0)
        p_up = kwargs.pop('p_up', 0.3)
        eta1 = kwargs.pop('eta1', 10.0)
        eta2 = kwargs.pop('eta2', 5.0)
        
        option = KouJumpModel(
            lambda_jump=lambda_jump,
            p_up=p_up,
            eta1=eta1,
            eta2=eta2,
            **kwargs
        )
        return option.calculate_greeks()
    elif pricing_model == 'advanced':
        option = AdvancedOptionPricing(**kwargs)
        return option.calculate_greeks()
    else:
        raise ValueError(f"Greeks calculation not supported for model: {pricing_model}")

def calibrate_model(pricing_model, market_prices, spot_price, risk_free_rate, initial_params=None, **kwargs):
    """
    Calibrate model parameters to market prices.
    
    Args:
        pricing_model (str): Model to calibrate ('heston', 'merton_jump', or 'kou_jump')
        market_prices (dict): Dictionary of market prices with (strike, expiry) tuples as keys
        spot_price (float): Current spot price
        risk_free_rate (float): Risk-free interest rate
        initial_params (dict, optional): Initial parameter estimates
        **kwargs: Additional parameters
    
    Returns:
        dict: Calibrated parameters
    """
    option_type = kwargs.get('option_type', 'call')
    
    if pricing_model == 'heston':
        # Default initial parameters for Heston model
        if initial_params is None:
            initial_params = {
                'kappa': 2.0,
                'theta': kwargs.get('initial_volatility', 0.04),
                'sigma': 0.3,
                'rho': -0.7
            }
        
        # Initial volatility (either from kwargs or derived from regular volatility)
        initial_volatility = kwargs.get('initial_volatility', kwargs.get('volatility', 0.2)**2)
        
        # Create Heston model instance
        model = HestonModel(
            spot_price=spot_price,
            strike_price=100,  # Dummy value, will be overridden during calibration
            risk_free_rate=risk_free_rate,
            initial_volatility=initial_volatility,
            time_to_expiry=1.0,  # Dummy value, will be overridden during calibration
            kappa=initial_params['kappa'],
            theta=initial_params['theta'],
            sigma=initial_params['sigma'],
            rho=initial_params['rho'],
            option_type=option_type
        )
        
        # Perform calibration
        return model.calibrate(market_prices, initial_params)
        
    elif pricing_model == 'merton_jump':
        # Default initial parameters for Merton jump model
        if initial_params is None:
            initial_params = {
                'lambda_jump': 1.0,
                'mu_jump': -0.1,
                'sigma_jump': 0.2
            }
        
        # Create Merton jump model instance
        model = MertonJumpModel(
            spot_price=spot_price,
            strike_price=100,  # Dummy value, will be overridden during calibration
            risk_free_rate=risk_free_rate,
            volatility=kwargs.get('volatility', 0.2),
            time_to_expiry=1.0,  # Dummy value, will be overridden during calibration
            lambda_jump=initial_params['lambda_jump'],
            mu_jump=initial_params['mu_jump'],
            sigma_jump=initial_params['sigma_jump'],
            option_type=option_type
        )
        
        # Perform calibration
        return model.calibrate(market_prices, initial_params)
        
    elif pricing_model == 'kou_jump':
        # Default initial parameters for Kou jump model
        if initial_params is None:
            initial_params = {
                'lambda_jump': 1.0,
                'p_up': 0.3,
                'eta1': 10.0,
                'eta2': 5.0
            }
        
        # Create Kou jump model instance
        model = KouJumpModel(
            spot_price=spot_price,
            strike_price=100,  # Dummy value, will be overridden during calibration
            risk_free_rate=risk_free_rate,
            volatility=kwargs.get('volatility', 0.2),
            time_to_expiry=1.0,  # Dummy value, will be overridden during calibration
            lambda_jump=initial_params['lambda_jump'],
            p_up=initial_params['p_up'],
            eta1=initial_params['eta1'],
            eta2=initial_params['eta2'],
            option_type=option_type
        )
        
        # Perform calibration
        return model.calibrate(market_prices, initial_params)
    
    else:
        raise ValueError(f"Calibration not supported for model: {pricing_model}")

# Example usage
if __name__ == "__main__":
    # Basic Black-Scholes example
    print("\n=== Black-Scholes Model ===")
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
    
    # Basic Greeks calculation
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
    
    # Heston model example
    print("\n=== Heston Model ===")
    heston_price = calculate_option_price(
        pricing_model='heston',
        spot_price=100,
        strike_price=100,
        risk_free_rate=0.05,
        initial_volatility=0.04,  # v_0 = 0.2^2
        time_to_expiry=1.0,
        kappa=2.0,        # Mean reversion speed
        theta=0.04,       # Long-term variance
        sigma=0.3,        # Volatility of variance
        rho=-0.7,         # Correlation
        option_type='call'
    )
    
    print(f"Heston Model Call Price: ${heston_price:.2f}")
    
    # Merton Jump Diffusion example
    print("\n=== Merton Jump Diffusion Model ===")
    merton_price = calculate_option_price(
        pricing_model='merton_jump',
        spot_price=100,
        strike_price=100,
        risk_free_rate=0.05,
        volatility=0.2,
        time_to_expiry=1.0,
        lambda_jump=1.0,   # Jump intensity
        mu_jump=-0.1,      # Average jump size
        sigma_jump=0.2,    # Jump size volatility
        option_type='call'
    )
    
    print(f"Merton Jump Model Call Price: ${merton_price:.2f}")
    
    # Kou Jump Diffusion example
    print("\n=== Kou Jump Diffusion Model ===")
    kou_price = calculate_option_price(
        pricing_model='kou_jump',
        spot_price=100,
        strike_price=100,
        risk_free_rate=0.05,
        volatility=0.2,
        time_to_expiry=1.0,
        lambda_jump=1.0,  # Jump intensity
        p_up=0.3,         # Probability of upward jump
        eta1=10.0,        # Upward jump rate parameter
        eta2=5.0,         # Downward jump rate parameter
        option_type='call'
    )
    
    print(f"Kou Jump Model Call Price: ${kou_price:.2f}")
