import numpy as np
import pandas as pd
import time
from models import BlackScholesOption
from advanced_models import HestonModel, MertonJumpModel

def run_validation_and_profiling():
    """
    Performs accuracy and performance validation for advanced option pricing models.
    """
    # Define test scenarios
    scenarios = [
        {
            'name': 'At-the-Money',
            'spot_price': 100, 'strike_price': 100, 'risk_free_rate': 0.05,
            'volatility': 0.2, 'time_to_expiry': 1.0, 'option_type': 'call'
        },
        {
            'name': 'In-the-Money',
            'spot_price': 110, 'strike_price': 100, 'risk_free_rate': 0.05,
            'volatility': 0.25, 'time_to_expiry': 0.75, 'option_type': 'call'
        },
        {
            'name': 'Out-of-the-Money',
            'spot_price': 90, 'strike_price': 100, 'risk_free_rate': 0.05,
            'volatility': 0.15, 'time_to_expiry': 0.5, 'option_type': 'put'
        }
    ]

    # --- Accuracy Validation for MertonJumpModel ---
    accuracy_results = []
    print("--- Merton Jump Model Accuracy Validation ---")
    print("-" * 50)

    for scenario in scenarios:
        # Benchmark price from Black-Scholes
        bs_model = BlackScholesOption(
            scenario['spot_price'], scenario['strike_price'], scenario['risk_free_rate'],
            scenario['volatility'], scenario['time_to_expiry'], scenario['option_type']
        )
        bs_price = bs_model.price()

        # Merton Jump Model
        merton_model = MertonJumpModel(
            scenario['spot_price'], scenario['strike_price'], scenario['risk_free_rate'],
            scenario['volatility'], scenario['time_to_expiry'],
            lambda_jump=0.1, mu_jump=0.0, sigma_jump=0.1, option_type=scenario['option_type']
        )
        merton_price = merton_model.price()

        accuracy_results.append({
            'Scenario': scenario['name'],
            'Merton Price': merton_price,
            'Benchmark (BS) Price': bs_price,
            'Difference': merton_price - bs_price
        })

    accuracy_df = pd.DataFrame(accuracy_results)
    print(accuracy_df.to_string())
    print("\n")


    # --- Performance Profiling for Merton and Heston Models ---
    performance_results = []
    print("--- Model Performance Profiling ---")
    print("-" * 50)

    # Use a consistent scenario for performance testing
    perf_scenario = scenarios[0]

    # Profile MertonJumpModel
    merton_model = MertonJumpModel(
        perf_scenario['spot_price'], perf_scenario['strike_price'], perf_scenario['risk_free_rate'],
        perf_scenario['volatility'], perf_scenario['time_to_expiry'],
        lambda_jump=0.1, mu_jump=0.0, sigma_jump=0.1, option_type=perf_scenario['option_type']
    )
    start_time = time.time()
    merton_model.price()
    end_time = time.time()
    performance_results.append({
        'Model': 'MertonJumpModel',
        'Execution Time (s)': end_time - start_time
    })

    # Profile HestonModel
    heston_model = HestonModel(
        perf_scenario['spot_price'], perf_scenario['strike_price'], perf_scenario['risk_free_rate'],
        perf_scenario['volatility']**2, perf_scenario['time_to_expiry'],
        kappa=2.0, theta=perf_scenario['volatility']**2, sigma=0.3, rho=-0.7,
        option_type=perf_scenario['option_type']
    )
    start_time = time.time()
    heston_model.price()
    end_time = time.time()
    performance_results.append({
        'Model': 'HestonModel',
        'Execution Time (s)': end_time - start_time
    })

    performance_df = pd.DataFrame(performance_results)
    print(performance_df.to_string())

    return accuracy_df, performance_df


if __name__ == "__main__":
    run_validation_and_profiling()
