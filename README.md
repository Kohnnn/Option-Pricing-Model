# Option Pricing Model Dashboard

## Overview

This project provides an advanced, interactive dashboard for pricing financial options using a variety of mathematical models. It integrates real-time financial data and sophisticated volatility analysis to deliver accurate and comprehensive pricing information. The application is built with Streamlit and features a modular architecture that is both scalable and easy to maintain.

## Features

- **Multiple Pricing Models:**
  - **Black-Scholes:** The industry standard for European options.
  - **Binomial Tree:** A flexible model for both European and American options.
  - **Monte Carlo Simulation:** For pricing complex and path-dependent options.
  - **Heston Model:** A stochastic volatility model that captures the volatility smile.
  - **Merton Jump-Diffusion Model:** Accounts for sudden, significant price jumps.
- **Real-Time Data Integration:**
  - Fetches the latest stock prices using the **Polygon.io API**.
- **Advanced Volatility Analysis:**
  - **Historical Volatility:** Calculated from historical price data.
  - **Implied Volatility:** Extracted from market option prices.
- **Interactive Dashboard:**
  - A user-friendly interface built with **Streamlit** for easy input of option parameters.
  - Detailed visualizations of pricing results and model comparisons.
  - In-depth analysis of **Greeks** (Delta, Gamma, Vega, Theta, Rho).
- **Model Benchmarking:**
  - Tools to compare the accuracy and performance of different models.

## Architecture

The application is designed with a modular and extensible architecture, consisting of the following key components:

- **`streamlit_app.py`:** The main entry point for the user interface.
- **`data_provider.py`:** Handles all interactions with the Polygon.io API for fetching financial data.
- **`volatility_engine.py`:** Centralizes the calculation of historical and implied volatility.
- **`models.py` & `advanced_models.py`:** Contain the implementations of the various option pricing models.
- **`base_model.py`:** Defines a common interface for all pricing models to ensure consistency.

For a more detailed architectural overview, please see [`ADVANCED_MODELS_README.md`](ADVANCED_MODELS_README.md).

## Prerequisites

- Python 3.9+
- A Polygon.io API key

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Kohnnn/option-pricing-model.git
    cd option-pricing-model
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API key:**
    You will need a Polygon.io API key to fetch market data. You can enter it directly in the application's sidebar.

## Running the Application

To launch the Streamlit dashboard, run the following command in your terminal:

```bash
streamlit run streamlit_app.py
```

## How to Use the Dashboard

1.  **Enter Your API Key:**
    - In the sidebar, enter your Polygon.io API key to enable market data fetching.

2.  **Fetch Market Data (Optional):**
    - Enter a stock ticker (e.g., `AAPL`) and click "Fetch Market Data" to get the latest price.

3.  **Select a Pricing Model:**
    - Choose from the available models in the sidebar, such as Black-Scholes, Heston, or Merton Jump.

4.  **Configure Option Parameters:**
    - Adjust the **Spot Price**, **Strike Price**, **Risk-Free Rate**, and other relevant parameters.

5.  **Choose a Volatility Source:**
    - **Manual Input:** Enter a volatility value directly.
    - **Historical Volatility:** Calculate volatility based on historical data.
    - **Implied Volatility:** Fetch implied volatility from the options chain.

6.  **Calculate the Option Price:**
    - Click the "Calculate Option Price" button to see the results, including the option price and a detailed analysis of the Greeks.

7.  **Analyze a Portfolio:**
    - **Navigate to the "Portfolio Analysis" Tab:** Select the "Portfolio Analysis" tab from the main menu to access the portfolio management and risk analysis tools.
    - **Build Your Portfolio:**
        - In the sidebar, enter a stock ticker (e.g., `AAPL`), the quantity of shares, and the purchase price.
        - Click "Add to Portfolio" to include the asset in your portfolio.
        - Repeat this process to add multiple assets. Your portfolio will be displayed in a table.
    - **Run Risk Calculations:**
        - Once your portfolio is built, you can perform risk analysis.
        - Select the desired risk model (e.g., Historical VaR/CVaR, Monte Carlo VaR/CVaR, Parametric VaR).
        - Configure the parameters, such as the confidence level and time horizon.
        - The calculated Value at Risk (VaR) and Conditional Value at Risk (CVaR) will be displayed, giving you insight into potential losses.
    - **Define and Run Stress Tests:**
        - Create custom stress test scenarios to see how your portfolio would perform under specific market conditions.
        - Define a scenario by giving it a name, selecting an asset, and specifying the price shock (either as a percentage or an absolute value).
        - Run the stress test to see the potential profit and loss (P&L) impact on your portfolio.

## Contributing

Contributions are welcome! If you would like to improve the application or add new features, please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
