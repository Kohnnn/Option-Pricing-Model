import numpy as np
from portfolio import Portfolio
from stochastic_processes import GeometricBrownianMotion
from scipy.stats import norm

class ValueAtRisk:
    """
    Provides methods to calculate Value at Risk (VaR) and Conditional Value at
    Risk (CVaR) using various methodologies.

    This class supports historical, parametric, and Monte Carlo-based risk
    calculations, offering a comprehensive toolkit for portfolio risk analysis.
    """
    def __init__(self, portfolio, confidence_level=0.99):
        """
        Initializes the ValueAtRisk calculator.

        Args:
            portfolio (Portfolio): The portfolio for which to calculate risk.
            confidence_level (float): The confidence level for VaR calculations
                                      (e.g., 0.99 for 99% confidence).
        """
        self.portfolio = portfolio
        self.confidence_level = confidence_level

    def historical_var(self, returns, horizon_days=1):
        """
        Calculates VaR using the historical simulation method.

        This method relies on historical price data to simulate potential
        portfolio returns and identifies the loss at the specified confidence
        level.

        Args:
            returns (np.ndarray): A NumPy array of historical portfolio returns.
            horizon_days (int): The time horizon in days for the VaR calculation.

        Returns:
            float: The calculated historical VaR, scaled by the time horizon.
        """
        if not isinstance(returns, np.ndarray):
            returns = np.array(returns)
        
        var = np.percentile(returns, 100 * (1 - self.confidence_level))
        return var * np.sqrt(horizon_days)

    def historical_cvar(self, returns, horizon_days=1):
        """
        Calculates Conditional Value at Risk (CVaR) using historical simulation.

        CVaR, or Expected Shortfall, is the expected loss given that the loss is
        greater than the VaR. It provides a more comprehensive measure of tail risk.

        Args:
            returns (np.ndarray): A NumPy array of historical portfolio returns.
            horizon_days (int): The time horizon in days for the CVaR calculation.

        Returns:
            float: The calculated historical CVaR, scaled by the time horizon.
        """
        if not isinstance(returns, np.ndarray):
            returns = np.array(returns)
            
        var = np.percentile(returns, 100 * (1 - self.confidence_level))
        cvar = returns[returns < var].mean()
        return cvar * np.sqrt(horizon_days)

    def monte_carlo_var(self, num_simulations, horizon_days=1):
        """
        Calculates VaR using the Monte Carlo simulation method.
        """
        pnl_distribution = self.monte_carlo_simulation(num_simulations, horizon_days)
        return self.get_var_from_pnl(pnl_distribution)

    def monte_carlo_simulation(self, num_simulations, horizon_days=1):
        """
        Generates a distribution of portfolio Profit and Loss (P&L) using a
        Monte Carlo simulation.

        This method simulates future asset prices using Geometric Brownian Motion
        (GBM) and calculates the resulting portfolio P&L for each simulation.

        Args:
            num_simulations (int): The number of Monte Carlo simulations to run.
            horizon_days (int): The simulation time horizon in days.

        Returns:
            np.ndarray: A NumPy array of simulated portfolio P&L values.
        """
        portfolio_pnl = []
        initial_value = self.portfolio.get_total_value()
        if initial_value == 0:
            return np.array([])

        for _ in range(num_simulations):
            simulated_value = 0
            for position in self.portfolio.positions.values():
                # Assuming mu=0.1, sigma=0.2 for simulation.
                # In a real scenario, these would be fetched or estimated per asset.
                gbm = GeometricBrownianMotion(mu=0.1, sigma=0.2)
                # Simulate path for the given horizon
                time_step = 1/252
                steps = horizon_days
                path = gbm.simulate_path(position['price'], time_step, steps)
                simulated_value += position['quantity'] * path[-1]
            
            pnl = simulated_value - initial_value
            portfolio_pnl.append(pnl)

        return np.array(portfolio_pnl)

    def get_var_from_pnl(self, pnl_distribution):
        """
        Calculates VaR from a given Profit and Loss (P&L) distribution.

        This is a helper method used by the Monte Carlo simulation to derive VaR
        from the simulated P&L data.

        Args:
            pnl_distribution (np.ndarray): A distribution of P&L values.

        Returns:
            float: The calculated VaR.
        """
        if pnl_distribution.size == 0:
            return 0
        return np.percentile(pnl_distribution, 100 * (1 - self.confidence_level))

    def get_cvar_from_pnl(self, pnl_distribution):
        """
        Calculates CVaR from a given Profit and Loss (P&L) distribution.

        This helper method computes CVaR from simulated P&L data by averaging
        the losses that exceed the VaR threshold.

        Args:
            pnl_distribution (np.ndarray): A distribution of P&L values.

        Returns:
            float: The calculated CVaR.
        """
        if pnl_distribution.size == 0:
            return 0
        var = self.get_var_from_pnl(pnl_distribution)
        cvar_values = pnl_distribution[pnl_distribution < var]
        return cvar_values.mean() if cvar_values.size > 0 else 0

    def parametric_var(self, returns, horizon_days=1):
        """
        Calculates VaR using the parametric (variance-covariance) method.

        This method assumes that portfolio returns are normally distributed and
        uses the mean and standard deviation to estimate VaR. It is fast but
        may be inaccurate if the normality assumption is violated.

        Args:
            returns (np.ndarray): A NumPy array of historical portfolio returns.
            horizon_days (int): The time horizon in days for the VaR calculation.

        Returns:
            float: The calculated parametric VaR.
        """
        if not isinstance(returns, np.ndarray):
            returns = np.array(returns)
            
        mean = np.mean(returns)
        std_dev = np.std(returns)
        
        # Z-score for the confidence level
        z_score = norm.ppf(1 - self.confidence_level)
        
        var = (mean + z_score * std_dev)
        return var * np.sqrt(horizon_days)

class StressTester:
    """
    Performs stress testing on a portfolio to assess its resilience to
    extreme market events.

    The StressTester allows for the creation of custom scenarios to simulate
    the impact of market shocks on portfolio value.
    """
    def __init__(self, portfolio):
        """
        Initializes the StressTester.

        Args:
            portfolio (Portfolio): The portfolio to be stress-tested.
        """
        self.portfolio = portfolio

    def run_stress_test(self, scenarios):
        """
        Runs a stress test on the portfolio using a set of defined scenarios.

        Each scenario applies a shock to a specific asset's price, and the
        method calculates the resulting Profit and Loss (P&L).

        Args:
            scenarios (list of dict): A list of stress scenarios, where each
                scenario is a dictionary defining the shock parameters.
                Example:
                {
                    'name': 'Market Crash',
                    'symbol': 'SPY',
                    'type': 'percentage',
                    'shock': -0.20  # 20% drop
                }

        Returns:
            dict: A dictionary mapping scenario names to the calculated P&L.
        """
        initial_value = self.portfolio.get_total_value()
        results = {}

        for scenario in scenarios:
            stressed_portfolio = self.portfolio.clone()
            symbol = scenario['symbol']
            shock = scenario['shock']
            shock_type = scenario['type']

            if symbol in stressed_portfolio.positions:
                original_price = stressed_portfolio.positions[symbol]['price']
                if shock_type == 'percentage':
                    new_price = original_price * (1 + shock)
                elif shock_type == 'absolute':
                    new_price = original_price + shock
                else:
                    raise ValueError(f"Invalid shock type: {shock_type}")
                
                stressed_portfolio.positions[symbol]['price'] = new_price

            stressed_value = stressed_portfolio.get_total_value()
            pnl = stressed_value - initial_value
            results[scenario['name']] = pnl
            
        return results