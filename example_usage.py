"""
Example Usage of Advanced Option Pricing Models

This script demonstrates how to use the Heston, Merton Jump, and Kou Jump
option pricing models for various applications.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from advanced_models import HestonModel, MertonJumpModel, KouJumpModel
from option_pricing import calibrate_model

def basic_pricing_example():
    """
    Basic example showing how to price options using all three advanced models.
    """
    print("\n=== Basic Pricing Example ===")
    
    # Option parameters
    spot_price = 100.0
    strike_price = 100.0
    risk_free_rate = 0.05
    volatility = 0.2
    time_to_expiry = 1.0
    
    # Initialize models
    heston = HestonModel(
        spot_price=spot_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        initial_volatility=volatility**2,  # convert volatility to variance
        time_to_expiry=time_to_expiry,
        kappa=2.0,
        theta=volatility**2,
        sigma=0.3,
        rho=-0.7,
        option_type='call'
    )
    
    merton_jump = MertonJumpModel(
        spot_price=spot_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiry=time_to_expiry,
        lambda_jump=1.0,
        mu_jump=-0.1,
        sigma_jump=0.2,
        option_type='call'
    )
    
    kou_jump = KouJumpModel(
        spot_price=spot_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiry=time_to_expiry,
        lambda_jump=1.0,
        p_up=0.3,
        eta1=10.0,
        eta2=5.0,
        option_type='call'
    )
    
    # Price European options
    heston_price = heston.price(american=False)
    merton_price = merton_jump.price(american=False)
    kou_price = kou_jump.price(american=False)
    
    print(f"Heston Model (European Call): ${heston_price:.4f}")
    print(f"Merton Jump Model (European Call): ${merton_price:.4f}")
    print(f"Kou Jump Model (European Call): ${kou_price:.4f}")
    
    # Price American options
    heston_american = heston.price(american=True, simulations=10000, time_steps=50)
    merton_american = merton_jump.price(american=True, simulations=10000, time_steps=50)
    kou_american = kou_jump.price(american=True, simulations=10000, time_steps=50)
    
    print(f"\nHeston Model (American Call): ${heston_american:.4f}")
    print(f"Merton Jump Model (American Call): ${merton_american:.4f}")
    print(f"Kou Jump Model (American Call): ${kou_american:.4f}")

def calculate_greeks_example():
    """
    Example showing how to calculate option Greeks for advanced models.
    """
    print("\n=== Option Greeks Example ===")
    
    # Option parameters
    spot_price = 100.0
    strike_price = 100.0
    risk_free_rate = 0.05
    volatility = 0.2
    time_to_expiry = 1.0
    
    models = {
        "Heston": HestonModel(
            spot_price=spot_price,
            strike_price=strike_price,
            risk_free_rate=risk_free_rate,
            initial_volatility=volatility**2,
            time_to_expiry=time_to_expiry,
            kappa=2.0,
            theta=volatility**2,
            sigma=0.3,
            rho=-0.7,
            option_type='call'
        ),
        "Merton Jump": MertonJumpModel(
            spot_price=spot_price,
            strike_price=strike_price,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            time_to_expiry=time_to_expiry,
            lambda_jump=1.0,
            mu_jump=-0.1,
            sigma_jump=0.2,
            option_type='call'
        ),
        "Kou Jump": KouJumpModel(
            spot_price=spot_price,
            strike_price=strike_price,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            time_to_expiry=time_to_expiry,
            lambda_jump=1.0,
            p_up=0.3,
            eta1=10.0,
            eta2=5.0,
            option_type='call'
        )
    }
    
    # Calculate Greeks for each model
    for name, model in models.items():
        greeks = model.calculate_greeks()
        print(f"\n{name} Model Greeks:")
        for greek, value in greeks.items():
            print(f"  {greek.capitalize()}: {value:.6f}")

def sensitivity_analysis():
    """
    Example showing how to perform sensitivity analysis.
    """
    print("\n=== Sensitivity Analysis Example ===")
    
    # Base parameters
    spot_price = 100.0
    strike_price = 100.0
    risk_free_rate = 0.05
    volatility = 0.2
    time_to_expiry = 1.0
    
    # Heston model sensitivity to correlation parameter
    rho_values = np.linspace(-0.9, 0.9, 19)
    heston_prices = []
    
    for rho in rho_values:
        model = HestonModel(
            spot_price=spot_price,
            strike_price=strike_price,
            risk_free_rate=risk_free_rate,
            initial_volatility=volatility**2,
            time_to_expiry=time_to_expiry,
            kappa=2.0,
            theta=volatility**2,
            sigma=0.3,
            rho=rho,
            option_type='call'
        )
        heston_prices.append(model.price(american=False))
    
    # Merton model sensitivity to jump intensity
    lambda_values = np.linspace(0.1, 3.0, 15)
    merton_prices = []
    
    for lambda_jump in lambda_values:
        model = MertonJumpModel(
            spot_price=spot_price,
            strike_price=strike_price,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            time_to_expiry=time_to_expiry,
            lambda_jump=lambda_jump,
            mu_jump=-0.1,
            sigma_jump=0.2,
            option_type='call'
        )
        merton_prices.append(model.price(american=False))
    
    # Kou model sensitivity to probability of upward jump
    p_up_values = np.linspace(0.1, 0.9, 9)
    kou_prices = []
    
    for p_up in p_up_values:
        model = KouJumpModel(
            spot_price=spot_price,
            strike_price=strike_price,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            time_to_expiry=time_to_expiry,
            lambda_jump=1.0,
            p_up=p_up,
            eta1=10.0,
            eta2=5.0,
            option_type='call'
        )
        kou_prices.append(model.price(american=False))
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(rho_values, heston_prices, 'o-', linewidth=2)
    plt.xlabel('Correlation (ρ)')
    plt.ylabel('Option Price')
    plt.title('Heston Model: Sensitivity to Correlation')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(lambda_values, merton_prices, 'o-', linewidth=2)
    plt.xlabel('Jump Intensity (λ)')
    plt.ylabel('Option Price')
    plt.title('Merton Model: Sensitivity to Jump Intensity')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(p_up_values, kou_prices, 'o-', linewidth=2)
    plt.xlabel('Probability of Upward Jump')
    plt.ylabel('Option Price')
    plt.title('Kou Model: Sensitivity to P(up)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png')
    print("Sensitivity analysis chart saved as 'sensitivity_analysis.png'")

def calibration_example():
    """
    Example showing how to calibrate model parameters to market prices.
    """
    print("\n=== Model Calibration Example ===")
    
    # Define market prices for different strikes and maturities
    # (strike, expiry) -> market_price
    market_prices = {
        (90, 0.5): 15.2,
        (100, 0.5): 8.7,
        (110, 0.5): 4.1,
        (90, 1.0): 18.1,
        (100, 1.0): 11.2,
        (110, 1.0): 6.3
    }
    
    print("Market prices for calibration:")
    for (strike, expiry), price in market_prices.items():
        print(f"  Strike={strike}, Expiry={expiry} years: ${price:.2f}")
    
    # Current spot price and risk-free rate
    spot_price = 100.0
    risk_free_rate = 0.05
    
    # Calibrate Heston model
    print("\nCalibrating Heston model...")
    heston_params = calibrate_model(
        'heston',
        market_prices,
        spot_price,
        risk_free_rate,
        initial_params={
            'kappa': 1.5,
            'theta': 0.04,
            'sigma': 0.4,
            'rho': -0.5
        }
    )
    
    print("Calibrated Heston Parameters:")
    print(f"  Mean Reversion Speed (κ): {heston_params['kappa']:.4f}")
    print(f"  Long-term Variance (θ): {heston_params['theta']:.4f}")
    print(f"  Volatility of Variance (σ): {heston_params['sigma']:.4f}")
    print(f"  Correlation (ρ): {heston_params['rho']:.4f}")
    
    # Calibrate Merton Jump model
    print("\nCalibrating Merton Jump model...")
    merton_params = calibrate_model(
        'merton_jump',
        market_prices,
        spot_price,
        risk_free_rate,
        volatility=0.2,
        initial_params={
            'lambda_jump': 0.8,
            'mu_jump': -0.05,
            'sigma_jump': 0.15
        }
    )
    
    print("Calibrated Merton Jump Parameters:")
    print(f"  Jump Intensity (λ): {merton_params['lambda_jump']:.4f}")
    print(f"  Average Jump Size (μ): {merton_params['mu_jump']:.4f}")
    print(f"  Jump Size Volatility (σ_j): {merton_params['sigma_jump']:.4f}")
    
    # Calibrate Kou Jump model
    print("\nCalibrating Kou Jump model...")
    kou_params = calibrate_model(
        'kou_jump',
        market_prices,
        spot_price,
        risk_free_rate,
        volatility=0.2,
        initial_params={
            'lambda_jump': 0.8,
            'p_up': 0.4,
            'eta1': 8.0,
            'eta2': 4.0
        }
    )
    
    print("Calibrated Kou Jump Parameters:")
    print(f"  Jump Intensity (λ): {kou_params['lambda_jump']:.4f}")
    print(f"  Probability of Upward Jump: {kou_params['p_up']:.4f}")
    print(f"  Upward Jump Rate (η₁): {kou_params['eta1']:.4f}")
    print(f"  Downward Jump Rate (η₂): {kou_params['eta2']:.4f}")

def term_structure_analysis():
    """
    Example showing how to analyze the term structure of option prices.
    """
    print("\n=== Term Structure Analysis Example ===")
    
    # Option parameters
    spot_price = 100.0
    strike_price = 100.0
    risk_free_rate = 0.05
    volatility = 0.2
    
    # Time points
    times = np.linspace(0.1, 2.0, 20)
    
    # Initialize price arrays
    heston_prices = []
    merton_prices = []
    kou_prices = []
    
    # Calculate prices for different maturities
    for t in times:
        # Heston model
        heston = HestonModel(
            spot_price=spot_price,
            strike_price=strike_price,
            risk_free_rate=risk_free_rate,
            initial_volatility=volatility**2,
            time_to_expiry=t,
            kappa=2.0,
            theta=volatility**2,
            sigma=0.3,
            rho=-0.7,
            option_type='call'
        )
        heston_prices.append(heston.price(american=False))
        
        # Merton Jump model
        merton_jump = MertonJumpModel(
            spot_price=spot_price,
            strike_price=strike_price,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            time_to_expiry=t,
            lambda_jump=1.0,
            mu_jump=-0.1,
            sigma_jump=0.2,
            option_type='call'
        )
        merton_prices.append(merton_jump.price(american=False))
        
        # Kou Jump model
        kou_jump = KouJumpModel(
            spot_price=spot_price,
            strike_price=strike_price,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            time_to_expiry=t,
            lambda_jump=1.0,
            p_up=0.3,
            eta1=10.0,
            eta2=5.0,
            option_type='call'
        )
        kou_prices.append(kou_jump.price(american=False))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(times, heston_prices, 'o-', label='Heston', linewidth=2)
    plt.plot(times, merton_prices, 's-', label='Merton Jump', linewidth=2)
    plt.plot(times, kou_prices, '^-', label='Kou Jump', linewidth=2)
    plt.xlabel('Time to Expiry (Years)')
    plt.ylabel('Option Price')
    plt.title('Term Structure of Option Prices')
    plt.grid(True)
    plt.legend()
    plt.savefig('term_structure.png')
    print("Term structure chart saved as 'term_structure.png'")

if __name__ == "__main__":
    basic_pricing_example()
    calculate_greeks_example()
    sensitivity_analysis()
    calibration_example()
    term_structure_analysis()