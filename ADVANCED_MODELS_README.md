# Software Architecture: Advanced Option Pricing Models and Data Integration

## 1. Overall Architecture Design

The proposed architecture introduces two new core components: a **Data Provider** for fetching market data and a **Volatility Engine** for calculating historical and implied volatility. These will integrate with the existing model and UI layers to create a more robust and data-driven pricing system.

### 1.1. High-Level Architecture Diagram

```mermaid
graph TD
    subgraph User Interface
        A[streamlit_app.py]
    end

    subgraph Application Logic
        B[option_pricing.py]
        C[data_provider.py]
        D[volatility_engine.py]
    end

    subgraph Modeling Layer
        E[models.py]
        F[advanced_models.py]
        G[BaseOptionModel (Interface)]
    end

    subgraph External Services
        H[Polygon.io API]
    end

    A -- Fetches Data --> C
    A -- Triggers Volatility Calculation --> D
    A -- Selects Model & Parameters --> B

    B -- Instantiates Models --> E
    B -- Instantiates Models --> F

    C -- Fetches Market Data --> H
    D -- Uses Data From --> C
    D -- Provides Volatility --> B

    E -- Implements --> G
    F -- Implements --> G
```

### 1.2. Detailed Architectural Description

The architecture is designed to be modular and extensible.

*   **`streamlit_app.py` (UI Layer):** Remains the entry point for the user. It will be enhanced to interact with the new `data_provider.py` to fetch real-time data for the underlying asset. It will also trigger the `volatility_engine.py` to calculate volatility based on the fetched data.
*   **`data_provider.py` (New Module):** This new module will be responsible for all interactions with the Polygon.io API. It will abstract away the complexities of API calls, data parsing, and error handling.
*   **`volatility_engine.py` (New Module):** This module will contain the logic for calculating different types of volatility. It will use the `data_provider.py` to get the necessary historical price data.
*   **`option_pricing.py` (Factory):** This will continue to act as a factory for the pricing models. It will be updated to accept volatility data from the `volatility_engine.py` and pass it to the models.
*   **`models.py` & `advanced_models.py` (Modeling Layer):** These files will be refactored to inherit from a common `BaseOptionModel` interface, ensuring that all models have a consistent API for pricing and calculating Greeks.

### 2. Data Integration Module Design (`data_provider.py`)

This module will be the single source of truth for all external market data.

#### 2.1. Architectural Plan

The `data_provider.py` will contain a `DataProvider` class that encapsulates the logic for fetching data from Polygon.io.

#### 2.2. Interfaces and Classes

```python
# data_provider.py

import os
import requests
from cachetools import cached, TTLCache

class DataProvider:
    """
    Provides financial data from Polygon.io API.
    """
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("Polygon.io API key not provided.")
        self.base_url = "https://api.polygon.io"

    @cached(cache=TTLCache(maxsize=100, ttl=300))
    def get_historical_stock_prices(self, ticker, from_date, to_date):
        """
        Fetches historical stock prices.
        """
        # Implementation to call Polygon.io aggregates endpoint
        pass

    @cached(cache=TTLCache(maxsize=100, ttl=300))
    def get_options_chain(self, ticker):
        """
        Fetches the options chain for a given ticker.
        """
        # Implementation to call Polygon.io options chain endpoint
        pass

    def get_implied_volatility(self, option_ticker):
        """
        Fetches implied volatility for a specific option contract.
        """
        # Implementation to call Polygon.io option details endpoint
        pass
```

#### 2.3. API Keys, Rate Limiting, and Caching

*   **API Keys:** The API key will be managed via an environment variable (`POLYGON_API_KEY`) for security.
*   **Rate Limiting:** While `requests` doesn't handle rate limiting out-of-the-box, we can add a simple decorator using `time.sleep` if we encounter rate limit issues. For a more robust solution, a library like `ratelimit` could be used.
*   **Caching:** `cachetools.TTLCache` will be used to cache API responses for a short period (e.g., 5 minutes) to reduce redundant API calls and improve performance.

### 3. Advanced Model Integration Plan

To ensure consistency and interchangeability, we will introduce a base class for all pricing models.

#### 3.1. Strategy for Model Integration

A new file, `base_model.py`, will be created to define the abstract base class `BaseOptionModel`. All models in `models.py` and `advanced_models.py` will be refactored to inherit from this class.

#### 3.2. Common Interface (`BaseOptionModel`)

```python
# base_model.py

from abc import ABC, abstractmethod

class BaseOptionModel(ABC):
    """
    Abstract base class for all option pricing models.
    """
    @abstractmethod
    def price(self):
        """
        Calculate the option price.
        """
        pass

    @abstractmethod
    def calculate_greeks(self):
        """
        Calculate the option Greeks.
        """
        pass
```

The `BlackScholesOption`, `HestonModel`, etc., will then be updated:

```python
# In models.py or advanced_models.py
from base_model import BaseOptionModel

class BlackScholesOption(BaseOptionModel):
    # ... existing implementation ...
    def price(self):
        # ...
    
    def calculate_greeks(self):
        # ...
```

#### 3.3. `option_pricing.py` Updates

The `calculate_option_price` and `calculate_option_greeks` functions will remain largely the same, but they will now be guaranteed a consistent interface from any model they instantiate.

### 4. Volatility Calculation Engine (`volatility_engine.py`)

This new module will centralize all volatility calculations.

#### 4.1. Component Design

The `volatility_engine.py` will contain a `VolatilityEngine` class.

```python
# volatility_engine.py

import numpy as np

class VolatilityEngine:
    """
    Calculates historical and implied volatility.
    """
    def __init__(self, data_provider):
        self.data_provider = data_provider

    def calculate_historical_volatility(self, ticker, window=252):
        """
        Calculates annualized historical volatility.
        """
        # 1. Fetch historical prices from data_provider
        # 2. Calculate daily log returns
        # 3. Calculate standard deviation of returns and annualize it
        pass

    def get_implied_volatility_from_chain(self, ticker, strike, expiry):
        """
        Finds the implied volatility for a specific option in the chain.
        """
        # 1. Fetch options chain from data_provider
        # 2. Find the matching option contract
        # 3. Return its implied volatility
        pass
```

#### 4.2. Triggering and Consumption

The `streamlit_app.py` will instantiate the `VolatilityEngine`. When a user wants to use calculated volatility, the UI will call the appropriate method on the engine, and the result will be passed as the `volatility` parameter to the selected pricing model.

### 5. User Interface (UI) Integration

The `streamlit_app.py` will be updated to incorporate these new capabilities.

#### 5.1. Plan for UI Updates

*   Add a new section in the sidebar to fetch real-time data. This will include a text input for a stock ticker and a button to "Fetch Data".
*   Once data is fetched, display the current stock price.
*   Add a dropdown to the volatility input section, allowing the user to choose between "Manual Input", "Calculate Historical Volatility", or "Fetch Implied Volatility".
*   When a calculation option is selected, the UI will call the `VolatilityEngine` and populate the volatility input with the result.
*   The UI for model-specific parameters will remain as it is.

### 6. Advanced Model Implementation Details

This section provides further details on the newly integrated advanced models.

#### 6.1. Heston Model

The Heston model is a stochastic volatility model that addresses one of the key limitations of the Black-Scholes model by allowing volatility to be a random process.

**Mathematical Formulation:**

The Heston model is defined by two stochastic differential equations:

1.  **Asset Price Process:**
    ```
    dS_t = r * S_t * dt + sqrt(v_t) * S_t * dW_t^1
    ```
2.  **Volatility Process:**
    ```
    dv_t = k * (θ - v_t) * dt + σ * sqrt(v_t) * dW_t^2
    ```

Where:
- `S_t`: Asset price at time `t`
- `v_t`: Variance of the asset price at time `t`
- `r`: Risk-free interest rate
- `k`: Rate of mean reversion for variance
- `θ`: Long-term mean of variance
- `σ`: Volatility of variance
- `dW_t^1`, `dW_t^2`: Wiener processes with correlation `ρ`

**Implementation Notes:**

- The `HestonModel` class in `advanced_models.py` implements a Monte Carlo simulation to price options.
- It simulates both the asset price and its variance over time, capturing the complex dynamics of stochastic volatility.
- **TODO:** The current implementation of the Heston model has a known issue where it can sometimes produce inaccurate prices, particularly for far-out-of-the-money or deep-in-the-money options. This is under investigation and will be addressed in a future update.

#### 6.2. Merton Jump-Diffusion Model

The Merton Jump-Diffusion model extends the Black-Scholes model to account for the possibility of sudden, large price movements (jumps) in the underlying asset.

**Mathematical Formulation:**

The asset price process is given by:
```
dS_t = (r - λ * k) * S_t * dt + σ * S_t * dW_t + dJ_t
```
Where:
- `J_t`: A compound Poisson process representing the jumps.
- `λ`: The intensity of the Poisson process (average number of jumps per year).
- `k`: The expected jump size.

**Implementation Notes:**

- The `MertonJumpModel` in `advanced_models.py` uses a Monte Carlo simulation to incorporate jump risk.
- The simulation models both the standard diffusion process and the random jump events, providing a more realistic price for assets prone to sudden shocks.
