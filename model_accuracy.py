import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import BlackScholesOption, AdvancedOptionPricing, BinomialTreeOption

def calculate_model_accuracy(spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, option_type='call'):
    """
    Compare accuracy of different option pricing models
    
    Args:
        spot_price (float): Current stock price
        strike_price (float): Option strike price
        risk_free_rate (float): Risk-free interest rate
        volatility (float): Stock price volatility
        time_to_expiry (float): Time to option expiration
        option_type (str): Option type 'call' or 'put'
    
    Returns:
        pd.DataFrame: Comparison of different pricing models
    """
    # Initialize pricing models
    black_scholes = BlackScholesOption(
        spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, option_type
    )
    advanced_option = AdvancedOptionPricing(
        spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, option_type
    )
    binomial_tree = BinomialTreeOption(
        spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, option_type=option_type
    )
    
    # Calculate prices using different methods
    pricing_methods = {
        'Black-Scholes (Analytical)': black_scholes.price(),
        'Monte Carlo (Basic)': advanced_option.monte_carlo_pricing(simulations=50000)['price'],
        'Quasi-Monte Carlo': advanced_option.quasi_monte_carlo_pricing(simulations=50000),
        'Binomial Tree': binomial_tree.price()
    }
    
    # Calculate Greeks
    greeks = advanced_option.calculate_greeks()
    
    # Prepare results DataFrame
    results_df = pd.DataFrame.from_dict(pricing_methods, orient='index', columns=['Price'])
    
    # Calculate relative differences
    base_price = results_df.loc['Black-Scholes (Analytical)', 'Price']
    results_df['Absolute Difference'] = np.abs(results_df['Price'] - base_price)
    results_df['Relative Difference (%)'] = np.abs((results_df['Price'] - base_price) / base_price * 100)
    
    # Add Greeks to the results
    for greek, value in greeks.items():
        results_df[f'{greek.capitalize()} Greek'] = value
    
    return results_df

def visualize_pricing_accuracy(spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, option_type='call'):
    """
    Visualize the accuracy of different option pricing models
    """
    # Calculate accuracy metrics
    accuracy_df = calculate_model_accuracy(
        spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, option_type
    )
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Pricing Comparison
    plt.subplot(1, 2, 1)
    accuracy_df['Price'].plot(kind='bar')
    plt.title('Option Pricing Comparison')
    plt.xlabel('Pricing Method')
    plt.ylabel('Option Price')
    plt.xticks(rotation=45, ha='right')
    
    # Relative Difference
    plt.subplot(1, 2, 2)
    accuracy_df['Relative Difference (%)'].plot(kind='bar')
    plt.title('Relative Difference from Black-Scholes')
    plt.xlabel('Pricing Method')
    plt.ylabel('Relative Difference (%)')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

def run_comprehensive_analysis():
    """
    Run comprehensive option pricing model analysis
    """
    # Test scenarios
    scenarios = [
        {
            'name': 'At-the-Money Call',
            'spot_price': 100,
            'strike_price': 100,
            'risk_free_rate': 0.05,
            'volatility': 0.2,
            'time_to_expiry': 1,
            'option_type': 'call'
        },
        {
            'name': 'Out-of-the-Money Put',
            'spot_price': 100,
            'strike_price': 110,
            'risk_free_rate': 0.05,
            'volatility': 0.3,
            'time_to_expiry': 0.5,
            'option_type': 'put'
        },
        {
            'name': 'In-the-Money Call',
            'spot_price': 120,
            'strike_price': 100,
            'risk_free_rate': 0.03,
            'volatility': 0.25,
            'time_to_expiry': 0.75,
            'option_type': 'call'
        }
    ]
    
    # Analyze each scenario
    for scenario in scenarios:
        print(f"\nAnalysis for {scenario['name']} Option:")
        print("-" * 50)
        
        # Calculate accuracy
        accuracy_df = calculate_model_accuracy(**{k: v for k, v in scenario.items() if k != 'name'})
        print(accuracy_df)
        
        # Optional: Visualize results
        # Uncomment the following line if you want to generate plots
        # visualize_pricing_accuracy(**{k: v for k, v in scenario.items() if k != 'name'})

if __name__ == "__main__":
    run_comprehensive_analysis()
