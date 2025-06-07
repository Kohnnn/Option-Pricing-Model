import numpy as np
from scipy.stats import norm
from scipy import integrate
import warnings
import math
from scipy.optimize import minimize
from models import validate_option_parameters
from base_model import BaseOptionModel

class HestonModel(BaseOptionModel):
    """
    Heston Stochastic Volatility Model for option pricing.
    This implementation uses the original Heston (1993) formulation for pricing European options.
    """
# TODO: The pricing formula for this model is currently incorrect and needs to be fixed.
    # The calculated price does not match known benchmarks.
    
    def __init__(self, spot_price, strike_price, risk_free_rate, initial_volatility, 
                 time_to_expiry, kappa, theta, sigma, rho, option_type='call'):
        validate_option_parameters(spot_price, strike_price, risk_free_rate, 
                                  np.sqrt(initial_volatility), time_to_expiry)
        
        if kappa < 0:
            raise ValueError("Mean reversion speed (kappa) must be non-negative")
        if theta < 0:
            raise ValueError("Long-term variance (theta) must be non-negative")
        if sigma < 0:
            raise ValueError("Volatility of variance (sigma) must be non-negative")
        if not -1 <= rho <= 1:
            raise ValueError("Correlation (rho) must be between -1 and 1")

        self.spot_price = spot_price
        self.strike_price = strike_price
        self.risk_free_rate = risk_free_rate
        self.v0 = initial_volatility
        self.T = time_to_expiry
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.option_type = option_type.lower()

    def _char_func(self, phi, j):
        """Characteristic function for the Heston model."""
        i = 1j
        
        if j == 1:
            u = 0.5
            b = self.kappa - self.rho * self.sigma
        else:
            u = -0.5
            b = self.kappa
        
        d = np.sqrt((self.rho * self.sigma * i * phi - b)**2 - self.sigma**2 * (2 * u * i * phi - phi**2))
        g = (b - self.rho * self.sigma * i * phi + d) / (b - self.rho * self.sigma * i * phi - d)
        
        C = self.risk_free_rate * i * phi * self.T + (self.kappa * self.theta / self.sigma**2) * (
            (b - self.rho * self.sigma * i * phi + d) * self.T - 2 * np.log((1 - g * np.exp(d * self.T)) / (1 - g))
        )
        D = (b - self.rho * self.sigma * i * phi + d) / self.sigma**2 * (
            (1 - np.exp(d * self.T)) / (1 - g * np.exp(d * self.T))
        )
        
        return np.exp(C + D * self.v0 + i * phi * np.log(self.spot_price))

    def _integrand(self, phi, j):
        """Integrand for the pricing formula."""
        i = 1j
        return np.real(np.exp(-i * phi * np.log(self.strike_price)) * self._char_func(phi, j) / (i * phi))

    def price(self):
        """Calculates the European option price using the Heston model."""
        raise NotImplementedError("The Heston model pricing formula is under review and not available.")

    def calculate_greeks(self):
        """Greeks calculation is not implemented in this simplified version."""
        warnings.warn("Greeks calculation is not implemented for this Heston model version.", UserWarning)
        return None

class MertonJumpModel(BaseOptionModel):
    """
    Merton Jump Diffusion Model for option pricing.
    
    Implements a diffusion process with Poisson jumps and log-normal jump size distribution.
    """
    
    def __init__(self, spot_price, strike_price, risk_free_rate, volatility, 
                 time_to_expiry, lambda_jump, mu_jump, sigma_jump, option_type='call'):
        """
        Initialize Merton Jump Diffusion Model for option pricing.
        
        Args:
            spot_price (float): Current stock price
            strike_price (float): Option strike price
            risk_free_rate (float): Risk-free interest rate
            volatility (float): Stock price volatility (diffusion component)
            time_to_expiry (float): Time to option expiration
            lambda_jump (float): Average number of jumps per year (intensity)
            mu_jump (float): Average jump size (log of the jump size)
            sigma_jump (float): Standard deviation of jump size
            option_type (str): 'call' or 'put'
        """
        validate_option_parameters(spot_price, strike_price, risk_free_rate, volatility, time_to_expiry)
        
        if lambda_jump < 0:
            raise ValueError("Jump intensity (lambda) must be non-negative")
        if sigma_jump < 0:
            raise ValueError("Jump size volatility must be non-negative")
        if time_to_expiry <= 0:
            raise ValueError("Time to expiry must be positive")
        
        self.spot_price = spot_price
        self.strike_price = strike_price
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.time_to_expiry = time_to_expiry
        self.lambda_jump = lambda_jump
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump
        self.option_type = option_type.lower()
        
        # Calculate jump-adjusted parameters
        self.k = np.exp(mu_jump + 0.5 * sigma_jump**2) - 1
        self.adjusted_rate = self.risk_free_rate - self.lambda_jump * self.k
    
    def _black_scholes_price(self, volatility_adj, time_to_expiry, rate_adj):
        """
        Calculate Black-Scholes price with adjusted parameters.
        
        Args:
            volatility_adj (float): Adjusted volatility
            time_to_expiry (float): Time to option expiration
            rate_adj (float): Adjusted risk-free rate
        
        Returns:
            float: Black-Scholes option price
        """
        d1 = (np.log(self.spot_price / self.strike_price) + 
              (rate_adj + 0.5 * volatility_adj**2) * time_to_expiry) / \
             (volatility_adj * np.sqrt(time_to_expiry))
        
        d2 = d1 - volatility_adj * np.sqrt(time_to_expiry)
        
        if self.option_type == 'call':
            return (self.spot_price * norm.cdf(d1) - 
                    self.strike_price * np.exp(-rate_adj * time_to_expiry) * norm.cdf(d2))
        else:  # put
            return (self.strike_price * np.exp(-rate_adj * time_to_expiry) * norm.cdf(-d2) - 
                    self.spot_price * norm.cdf(-d1))

    def price_european(self, n_terms=20):
        """
        Calculate option price using Merton Jump Diffusion model.
        
        Args:
            n_terms (int): Number of terms to use in the series approximation.
        
        Returns:
            float: Option price
        """
        total_price = 0.0
        for k in range(n_terms):
            # Adjusted parameters for each term in the series
            r_k = self.risk_free_rate - self.lambda_jump * (np.exp(self.mu_jump + 0.5 * self.sigma_jump**2) - 1) + (k * (self.mu_jump + 0.5 * self.sigma_jump**2)) / self.time_to_expiry
            sigma_k = np.sqrt(self.volatility**2 + (k * self.sigma_jump**2) / self.time_to_expiry)
            
            # Weight for each term
            weight = (np.exp(-self.lambda_jump * self.time_to_expiry) * (self.lambda_jump * self.time_to_expiry)**k) / math.factorial(k)
            
            # Add the weighted Black-Scholes price to the total
            total_price += weight * self._black_scholes_price(sigma_k, self.time_to_expiry, r_k)
            
        return total_price

    def price(self, **kwargs):
        """
        Calculate the European option price for the Merton model.
        """
        return self.price_european(**kwargs)

    def calculate_greeks(self):
        """Greeks calculation is not implemented for this model."""
        warnings.warn("Greeks calculation is not implemented for the Merton Jump-Diffusion model.", UserWarning)
        return None