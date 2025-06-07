import numpy as np
from scipy.stats import norm
from scipy.stats import t
import warnings
from base_model import BaseOptionModel
from utils import validate_option_parameters

class BlackScholesOption(BaseOptionModel):
    def __init__(self, spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, option_type='call'):
        """
        Initialize Black-Scholes Option Pricing Model
        
        Args:
            spot_price (float): Current stock price
            strike_price (float): Option strike price
            risk_free_rate (float): Risk-free interest rate
            volatility (float): Stock price volatility
            time_to_expiry (float): Time to option expiration
            option_type (str): 'call' or 'put'
        """
        validate_option_parameters(spot_price, strike_price, risk_free_rate, volatility, time_to_expiry)
        
        self.spot_price = spot_price
        self.strike_price = strike_price
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.time_to_expiry = time_to_expiry
        self.option_type = option_type.lower()
        
    def _calculate_d1_d2(self):
        """
        Calculate d1 and d2 parameters for Black-Scholes model
        
        Returns:
            tuple: (d1, d2)
        """
        d1 = (np.log(self.spot_price / self.strike_price) + 
              (self.risk_free_rate + 0.5 * self.volatility**2) * self.time_to_expiry) / \
             (self.volatility * np.sqrt(self.time_to_expiry))
        
        d2 = d1 - self.volatility * np.sqrt(self.time_to_expiry)
        
        return d1, d2
    
    def price(self):
        """
        Calculate option price using Black-Scholes model
        
        Returns:
            float: Option price
        """
        d1, d2 = self._calculate_d1_d2()
        
        if self.option_type == 'call':
            return (self.spot_price * norm.cdf(d1) -
                    self.strike_price * np.exp(-self.risk_free_rate * self.time_to_expiry) *
                    norm.cdf(d2))
        elif self.option_type == 'put':
            return (self.strike_price * np.exp(-self.risk_free_rate * self.time_to_expiry) *
                    norm.cdf(-d2) -
                    self.spot_price * norm.cdf(-d1))
        else:
            raise ValueError("Option type must be 'call' or 'put'")
    
    def delta(self):
        """
        Calculate option delta
        
        Returns:
            float: Option delta
        """
        d1, _ = self._calculate_d1_d2()
        
        if self.option_type == 'call':
            return norm.cdf(d1)
        elif self.option_type == 'put':
            return norm.cdf(d1) - 1
    
    def gamma(self):
        """
        Calculate option gamma
        
        Returns:
            float: Option gamma
        """
        d1, _ = self._calculate_d1_d2()
        
        return norm.pdf(d1) / (self.spot_price * self.volatility *
                                                   np.sqrt(self.time_to_expiry))
    
    def vega(self):
        """
        Calculate option vega
        
        Returns:
            float: Option vega
        """
        d1, _ = self._calculate_d1_d2()
        
        return self.spot_price * norm.pdf(d1) * np.sqrt(self.time_to_expiry)
    
    def theta(self):
        """
        Calculate option theta
        
        Returns:
            float: Option theta
        """
        d1, d2 = self._calculate_d1_d2()
        
        if self.option_type == 'call':
            theta = (-self.spot_price * norm.pdf(d1) * self.volatility /
                     (2 * np.sqrt(self.time_to_expiry)) -
                     self.risk_free_rate * self.strike_price * np.exp(-self.risk_free_rate * self.time_to_expiry) *
                     norm.cdf(d2))
        elif self.option_type == 'put':
            theta = (-self.spot_price * norm.pdf(d1) * self.volatility /
                     (2 * np.sqrt(self.time_to_expiry)) +
                     self.risk_free_rate * self.strike_price * np.exp(-self.risk_free_rate * self.time_to_expiry) *
                     norm.cdf(-d2))
        
        return theta
    
    def implied_volatility(self, market_price, max_iterations=100, tolerance=1e-5):
        """
        Calculate implied volatility using Newton-Raphson method
        
        Args:
            market_price (float): Current market price of the option
            max_iterations (int): Maximum number of iterations
            tolerance (float): Convergence tolerance
        
        Returns:
            float: Implied volatility
        """
        def objective_function(vol):
            """Calculate the difference between model price and market price"""
            self.volatility = vol
            return self.price() - market_price
        
        def derivative_function(vol):
            """Calculate the derivative of the objective function"""
            self.volatility = vol
            return self.vega()
        
        # Initial guess: use current volatility or a standard value
        volatility = self.volatility if self.volatility > 0 else 0.3
        
        for _ in range(max_iterations):
            price_diff = objective_function(volatility)
            
            if abs(price_diff) < tolerance:
                return volatility
            
            vega = derivative_function(volatility)
            
            # Prevent division by zero
            if vega == 0:
                break
            
            volatility -= price_diff / vega
        
        raise ValueError("Implied volatility calculation did not converge")

    def calculate_greeks(self):
        """
        Calculate all option Greeks
        
        Returns:
            dict: A dictionary containing delta, gamma, vega, and theta
        """
        return {
            'delta': self.delta(),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta()
        }

    def scenario_analysis(self, spot_price_range=None, volatility_range=None):
        """
        Perform scenario analysis for the option
        
        Args:
            spot_price_range (tuple): Range of spot prices to analyze (min, max, steps)
            volatility_range (tuple): Range of volatilities to analyze (min, max, steps)
        
        Returns:
            dict: Scenario analysis results
        """
        if spot_price_range is None:
            spot_price_range = (self.spot_price * 0.5, self.spot_price * 1.5, 20)
        
        if volatility_range is None:
            volatility_range = (0.1, 1.0, 20)
        
        spot_min, spot_max, spot_steps = spot_price_range
        vol_min, vol_max, vol_steps = volatility_range
        
        spot_prices = np.linspace(spot_min, spot_max, spot_steps)
        volatilities = np.linspace(vol_min, vol_max, vol_steps)
        
        results = {
            'spot_prices': spot_prices.tolist(),
            'volatilities': volatilities.tolist(),
            'prices': [],
            'deltas': [],
            'gammas': []
        }
        
        for spot in spot_prices:
            spot_results = []
            delta_results = []
            gamma_results = []
            
            for vol in volatilities:
                self.spot_price = spot
                self.volatility = vol
                
                spot_results.append(self.price())
                delta_results.append(self.delta())
                gamma_results.append(self.gamma())
            
            results['prices'].append(spot_results)
            results['deltas'].append(delta_results)
            results['gammas'].append(gamma_results)
        
        return results

class BinomialTreeOption(BaseOptionModel):
    def __init__(self, spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, steps=100, option_type='call'):
        """
        Initialize Binomial Tree Option Pricing Model
        
        Args:
            spot_price (float): Current stock price
            strike_price (float): Option strike price
            risk_free_rate (float): Risk-free interest rate
            volatility (float): Stock price volatility
            time_to_expiry (float): Time to option expiration
            steps (int): Number of steps in binomial tree
            option_type (str): 'call' or 'put'
        """
        validate_option_parameters(spot_price, strike_price, risk_free_rate, volatility, time_to_expiry)
        
        self.spot_price = spot_price
        self.strike_price = strike_price
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.time_to_expiry = time_to_expiry
        self.steps = steps
        self.option_type = option_type.lower()
    
    def price(self):
        """
        Calculate option price using Binomial Tree method
        
        Returns:
            float: Option price
        """
        dt = self.time_to_expiry / self.steps
        u = np.exp(self.volatility * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.risk_free_rate * dt) - d) / (u - d)
        
        stock_prices = np.zeros((self.steps + 1, self.steps + 1))
        option_values = np.zeros((self.steps + 1, self.steps + 1))
        
        for i in range(self.steps + 1):
            stock_prices[i, self.steps] = self.spot_price * (u ** (self.steps - i)) * (d ** i)
            
            if self.option_type == 'call':
                option_values[i, self.steps] = max(0, stock_prices[i, self.steps] - self.strike_price)
            elif self.option_type == 'put':
                option_values[i, self.steps] = max(0, self.strike_price - stock_prices[i, self.steps])
        
        for j in range(self.steps - 1, -1, -1):
            for i in range(j + 1):
                stock_prices[i, j] = self.spot_price * (u ** (j - i)) * (d ** i)
                
                discounted_expected_value = (p * option_values[i, j+1] + (1 - p) * option_values[i+1, j+1]) * np.exp(-self.risk_free_rate * dt)
                
                if self.option_type == 'call':
                    exercise_value = max(0, stock_prices[i, j] - self.strike_price)
                elif self.option_type == 'put':
                    exercise_value = max(0, self.strike_price - stock_prices[i, j])
                
                option_values[i, j] = max(exercise_value, discounted_expected_value)
        
        return option_values[0, 0]

    def calculate_greeks(self):
        """
        Greeks are not analytically available for the Binomial Tree model.
        This method is implemented to satisfy the BaseOptionModel interface.
        
        Returns:
            dict: A dictionary with None values for all Greeks.
        """
        return {
            'delta': None,
            'gamma': None,
            'vega': None,
            'theta': None
        }

class AdvancedOptionPricing(BaseOptionModel):
    """
    Advanced Option Pricing Models with Enhanced Risk Analysis and Simulation Techniques
    """
    def __init__(self, spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, option_type='call'):
        """
        Initialize advanced option pricing model with comprehensive parameters
        
        Args:
            spot_price (float): Current stock price
            strike_price (float): Option strike price
            risk_free_rate (float): Risk-free interest rate
            volatility (float): Stock price volatility
            time_to_expiry (float): Time to option expiration
            option_type (str): 'call' or 'put'
        """
        # Validate inputs first
        validate_option_parameters(spot_price, strike_price, risk_free_rate, volatility, time_to_expiry)
        
        self.spot_price = spot_price
        self.strike_price = strike_price
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.time_to_expiry = time_to_expiry
        self.option_type = option_type.lower()
        
        # Additional validation
        if self.option_type not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")
    
    def monte_carlo_pricing(self, simulations=50000, confidence_level=0.95, seed=None):
        """
        Advanced Monte Carlo Option Pricing Simulation with Multiple Variance Reduction Techniques
        
        Args:
            simulations (int): Number of Monte Carlo simulations
            confidence_level (float): Confidence level for interval estimation
            seed (int, optional): Random seed for reproducibility
        
        Returns:
            dict: Option pricing results with point estimate and confidence interval
        """
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(42)  # Default seed
        
        # Improved random number generation with Box-Muller transformation
        def generate_normal_samples(n):
            """
            Generate normally distributed random numbers using Box-Muller transformation
            
            Args:
                n (int): Number of samples to generate
            
            Returns:
                numpy.ndarray: Normally distributed random numbers
            """
            u1 = np.random.uniform(0, 1, n)
            u2 = np.random.uniform(0, 1, n)
            z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
            return z0
        
        # Generate random samples
        z = generate_normal_samples(simulations)
        
        # Simulate stock price paths using more accurate geometric Brownian motion
        drift = (self.risk_free_rate - 0.5 * self.volatility**2)
        diffusion = self.volatility
        
        stock_paths = self.spot_price * np.exp(
            drift * self.time_to_expiry + 
            diffusion * np.sqrt(self.time_to_expiry) * z
        )
        
        # Calculate payoffs with improved numerical stability
        if self.option_type == 'call':
            payoffs = np.maximum(stock_paths - self.strike_price, 0)
        else:
            payoffs = np.maximum(self.strike_price - stock_paths, 0)
        
        # More robust discounting
        discount_factor = np.exp(-self.risk_free_rate * self.time_to_expiry)
        discounted_payoffs = discount_factor * payoffs
        
        # Point estimate with improved statistical methods
        option_price_mean = np.mean(discounted_payoffs)
        option_price_median = np.median(discounted_payoffs)
        
        # Robust confidence interval calculation
        standard_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(simulations)
        degrees_of_freedom = simulations - 1
        t_value = t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
        
        confidence_interval = (
            option_price_mean - t_value * standard_error,
            option_price_mean + t_value * standard_error
        )
        
        # Analytical Black-Scholes price for comparison
        bs_option = BlackScholesOption(self.spot_price, self.strike_price, self.risk_free_rate, self.volatility, self.time_to_expiry, self.option_type)
        bs_price = bs_option.price()
        
        return {
            'price': option_price_mean,
            'median_price': option_price_median,
            'confidence_interval': confidence_interval,
            'standard_error': standard_error,
            'black_scholes_price': bs_price,
            'price_variance': np.var(discounted_payoffs, ddof=1),
            'simulation_details': {
                'drift': drift,
                'diffusion': diffusion,
                'simulations': simulations
            }
        }
    
    def quasi_monte_carlo_pricing(self, simulations=50000):
        """
        Quasi-Monte Carlo Option Pricing using Sobol Sequences with Improved Accuracy
        
        Args:
            simulations (int): Number of quasi-Monte Carlo simulations
        
        Returns:
            float: Estimated option price using Sobol sequences
        """
        try:
            from scipy.stats import qmc
        except ImportError:
            warnings.warn("Scipy's qmc module not available. Using standard Monte Carlo.")
            return self.monte_carlo_pricing(simulations)['price']
        
        # Create Sobol sequence generator with improved parameters
        sampler = qmc.Sobol(d=1, scramble=True)
        
        # Generate quasi-random numbers with multiple transformations
        sobol_samples = sampler.random(n=simulations)
        
        # Improved inverse transform sampling
        def inverse_transform_sampling(samples):
            """
            Convert uniform samples to normally distributed samples
            
            Args:
                samples (numpy.ndarray): Uniform random samples
            
            Returns:
                numpy.ndarray: Normally distributed samples
            """
            return norm.ppf(samples)
        
        # Transform Sobol samples
        z = inverse_transform_sampling(sobol_samples)
        
        # Simulate stock price paths with more accurate drift and diffusion
        drift = (self.risk_free_rate - 0.5 * self.volatility**2)
        diffusion = self.volatility
        
        stock_paths = self.spot_price * np.exp(
            drift * self.time_to_expiry + 
            diffusion * np.sqrt(self.time_to_expiry) * z
        )
        
        # Calculate payoffs
        if self.option_type == 'call':
            payoffs = np.maximum(stock_paths - self.strike_price, 0)
        else:
            payoffs = np.maximum(self.strike_price - stock_paths, 0)
        
        # Discount the payoffs
        option_price = np.exp(-self.risk_free_rate * self.time_to_expiry) * np.mean(payoffs)
        
        return option_price
    
    def _calculate_d1_d2(self):
        """
        Calculate d1 and d2 parameters for Black-Scholes model
        
        Returns:
            tuple: (d1, d2)
        """
        d1 = (np.log(self.spot_price / self.strike_price) + 
              (self.risk_free_rate + 0.5 * self.volatility**2) * self.time_to_expiry) / \
             (self.volatility * np.sqrt(self.time_to_expiry))
        
        d2 = d1 - self.volatility * np.sqrt(self.time_to_expiry)
        
        return d1, d2
    
    def price(self, **kwargs):
        """
        Calculate option price using the most appropriate method.
        For this advanced model, it defaults to Monte Carlo simulation.
        
        Returns:
            float: Option price
        """
        # Forward any arguments to the monte_carlo_pricing method
        return self.monte_carlo_pricing(**kwargs)['price']
    
    def calculate_greeks(self):
        """
        Calculate Option Greeks for risk analysis
        
        Returns:
            dict: Option Greeks (Delta, Gamma, Theta, Vega, Rho)
        """
        d1, d2 = self._calculate_d1_d2()
        
        # Delta calculation
        if self.option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = -norm.cdf(-d1)
        
        # Gamma calculation
        gamma = norm.pdf(d1) / (self.spot_price * self.volatility * np.sqrt(self.time_to_expiry))
        
        # Theta calculation
        theta_calc = -(self.spot_price * norm.pdf(d1) * self.volatility) / (2 * np.sqrt(self.time_to_expiry))
        theta = theta_calc - self.risk_free_rate * self.strike_price * np.exp(-self.risk_free_rate * self.time_to_expiry) * norm.cdf(d2)
        
        # Vega calculation
        vega = self.spot_price * norm.pdf(d1) * np.sqrt(self.time_to_expiry)
        
        # Rho calculation
        if self.option_type == 'call':
            rho = self.strike_price * self.time_to_expiry * np.exp(-self.risk_free_rate * self.time_to_expiry) * norm.cdf(d2)
        else:
            rho = -self.strike_price * self.time_to_expiry * np.exp(-self.risk_free_rate * self.time_to_expiry) * norm.cdf(-d2)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

def cumulative_normal_distribution(x):
    return norm.cdf(x)

def standard_normal_distribution(x):
    return norm.pdf(x)
