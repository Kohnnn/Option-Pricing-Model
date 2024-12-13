# Option Pricing Module

## Overview

This module provides comprehensive option pricing tools using various mathematical models and techniques.

## Option Pricing Models: Mathematical Foundations

### 1. Black-Scholes Option Pricing Model

#### Mathematical Formula

The Black-Scholes formula calculates the theoretical price of European-style options:

For a Call Option:

```
C = S * N(d1) - K * e^(-r*T) * N(d2)
```

For a Put Option:

```
P = K * e^(-r*T) * N(-d2) - S * N(-d1)
```

Where:

- `C`: Call Option Price
- `P`: Put Option Price
- `S`: Current Stock Price
- `K`: Strike Price
- `r`: Risk-Free Interest Rate
- `T`: Time to Expiration
- `N()`: Cumulative Standard Normal Distribution Function

#### Key Parameters Calculation

1. **d1 Calculation**:

```
d1 = [ln(S/K) + (r + σ²/2) * T] / (σ * √T)
```

- `ln()`: Natural Logarithm
- `σ`: Stock Price Volatility

2. **d2 Calculation**:

```
d2 = d1 - σ * √T
```

#### Greeks Calculation

1. **Delta (Δ)**:

   - Measures rate of change in option price relative to stock price
   - Call Option: `N(d1)`
   - Put Option: `N(d1) - 1`
2. **Gamma (Γ)**:

   - Measures rate of change in delta

   ```
   Γ = φ(d1) / (S * σ * √T)
   ```

   Where `φ()` is the standard normal probability density function
3. **Vega (ν)**:

   - Measures sensitivity to volatility

   ```
   ν = S * φ(d1) * √T
   ```
4. **Theta (Θ)**:

   - Measures time decay of option value
   - Complex calculation involving stock price, volatility, and time

### 2. Binomial Tree Option Pricing Model

#### Mathematical Approach

The Binomial Tree model discretizes time into multiple steps, creating a tree of possible stock price movements.

#### Key Calculations

1. **Up and Down Factors**:

```
u = e^(σ * √Δt)  # Up factor
d = 1/u          # Down factor
```

2. **Risk-Neutral Probability**:

```
p = (e^(r*Δt) - d) / (u - d)
```

3. **Option Value Calculation**:
   - Backward induction from expiration to present
   - At each node, compare intrinsic value with expected discounted value

#### Advantages

- Handles American-style options
- More flexible than Black-Scholes
- Can incorporate early exercise

### 3. Implied Volatility Estimation (Future Enhancement)

#### Concept

- Reverse-engineer volatility from market option prices
- Uses iterative numerical methods (e.g., Newton-Raphson)

## Mathematical Assumptions

### Black-Scholes Model Assumptions

1. Log-normal distribution of stock prices
2. No transaction costs
3. No dividends
4. Risk-free rate and volatility are constant
5. European-style options only

### Binomial Tree Model Assumptions

1. Discrete-time model
2. Stock price can move up or down
3. Risk-neutral pricing framework

## Option Pricing Models

## Overview

This interactive Streamlit application provides comprehensive option pricing tools using various mathematical models and techniques, including:
- Black-Scholes Option Pricing Model
- Binomial Tree Option Pricing Model
- Advanced Option Pricing Techniques

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/option-pricing-model.git
cd option-pricing-model
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

To launch the Streamlit application:
```bash
streamlit run streamlit_app.py
```

## Features

- Interactive option pricing calculation
- Multiple pricing model comparisons
- Visualization of pricing accuracy
- Greeks calculation
- Model performance analysis

## Mathematical Models

### 1. Black-Scholes Option Pricing Model
Calculates theoretical prices for European-style options.

### 2. Binomial Tree Option Pricing Model
Provides a more flexible approach to option pricing by discretizing time.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]

## Contact

[Your contact information]

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Running the Web Application

```bash
# On Windows
run.bat

# On macOS/Linux
python app.py
```

Navigate to `http://localhost:5000` in your web browser.

## Usage

### Python Module

```python
from option_pricing import calculate_option_price

# Calculate option price
price = calculate_option_price(
    spot_price=100,
    strike_price=100,
    risk_free_rate=0.05,
    volatility=0.2,
    time_to_expiry=1.0,
    option_type='call'
)
```

### Web Interface

1. Select Pricing Model (Black-Scholes or Binomial Tree)
2. Choose Option Type (Call or Put)
3. Enter Option Parameters
4. Click "Calculate Option Price"

## Update 12/13/24

**Added:**

- Implied Volatility Calculation
- Scenario Analysis
- Greeks Calculation (Delta, Gamma, Vega, Theta)

## License

MIT License

## WIP

* Fomula accuracy
* Pricing for other assets classes
