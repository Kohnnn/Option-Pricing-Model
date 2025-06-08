import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import json

# Import our custom models and accuracy tools
from models import BlackScholesOption, AdvancedOptionPricing, BinomialTreeOption
from advanced_models import HestonModel, MertonJumpModel
from data_provider import FinancialDataAPI
from volatility_engine import VolatilityEngine
from portfolio import Portfolio
from risk_management import ValueAtRisk, StressTester
import datetime

# Set page configuration
st.set_page_config(
    page_title="Option Pricing Model",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(to right, #f0f2f6, #e6e9ef);
    }
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
    }
    .metric-container {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 15px;
        margin-bottom: 15px;
    }
    .explanation-text {
        font-size: 0.8rem;
        color: #666;
        margin-top: 5px;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted black;
        cursor: help;
        margin-bottom: 50px;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 250px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -125px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

def add_tooltip(text, explanation):
    """
    Create a tooltip with text and explanation
    """
    return f"""
    <div class="tooltip">{text}
        <span class="tooltiptext">{explanation}</span>
    </div>
    """

def add_explanation(text):
    """
    Add a small explanatory text with a muted style
    """
    st.markdown(f'<div class="explanation-text">{text}</div>', unsafe_allow_html=True)

def render_option_pricing_page(data_provider, volatility_engine):
    st.header("Single Option Pricing")
    st.markdown("This section allows you to price a single option using various models.")

    model_explanations = {
        "Black-Scholes": {
            "methodology": "The Black-Scholes model provides a theoretical estimate of the price of European-style options. It is based on the idea of creating a riskless portfolio by combining the option and its underlying asset.",
            "assumptions": """
            - **Lognormal Distribution:** Stock prices follow a geometric Brownian motion, resulting in a lognormal distribution.
            - **Constant Volatility & Interest Rates:** The model assumes that the volatility of the underlying asset and the risk-free interest rate are constant over the option's life.
            - **No Dividends:** The underlying stock does not pay dividends.
            - **European-Style Option:** The option can only be exercised at expiration.
            - **No Transaction Costs or Taxes:** The model ignores the impact of commissions and taxes.
            - **Frictionless Markets:** It is possible to borrow and lend money at the risk-free rate.
            """,
            "formula": r'''
            C(S, t) = S_0 N(d_1) - K e^{-rT} N(d_2) \\
            d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}} \\
            d_2 = d_1 - \sigma\sqrt{T}
            ''',
            "use_cases": "Best suited for pricing European options on non-dividend-paying stocks in relatively stable markets. It is a fundamental building block in finance and serves as a benchmark."
        },
        "Binomial Tree": {
            "methodology": "The Binomial Tree model is a numerical method that discretizes the time to expiration into a number of steps. At each step, the asset price can move up or down by a specific amount, creating a lattice of possible future prices.",
            "assumptions": """
            - **Discrete Time Steps:** The model assumes that price movements occur at discrete time intervals.
            - **No-Arbitrage:** The up and down movement probabilities are calculated to prevent arbitrage opportunities.
            """,
            "use_cases": "Highly versatile for pricing both American and European options. It can easily be adapted to handle dividends, changing interest rates, and other complex features."
        },
        "Monte Carlo": {
            "methodology": "Monte Carlo simulation is a computational technique that models the price of an option by generating thousands of random price paths for the underlying asset. The option's payoff is calculated for each path, and the average payoff is discounted to its present value.",
            "assumptions": """
            - **Assumed Price Process:** Relies on the same price process assumptions as the analytical model it is simulating (e.g., geometric Brownian motion for Black-Scholes).
            - **Large Number of Simulations:** Accuracy improves with the number of simulated paths.
            """,
            "use_cases": "Extremely powerful for pricing exotic options with complex payoffs (e.g., Asian, barrier, or lookback options) and for valuing options on multiple underlying assets."
        },
        "Heston": {
            "methodology": "The Heston model is a stochastic volatility model, meaning it assumes that the volatility of the underlying asset is not constant but follows a random process. This allows it to better capture real-world market phenomena like the volatility smile and skew.",
            "assumptions": """
            - **Stochastic Volatility:** Volatility follows a mean-reverting process (the Cox-Ingersoll-Ross, or CIR, process).
            - **Correlation:** The model allows for a correlation between the asset's price and its volatility.
            """,
            "formula": r'''
            dS_t = rS_t dt + \sqrt{v_t}S_t dW_t^S \\
            dv_t = \kappa(\theta - v_t)dt + \sigma_v\sqrt{v_t}dW_t^v
            ''',
            "use_cases": "Provides more accurate pricing for options where volatility is a key factor, such as long-term options, options on assets with fluctuating volatility, and currency options."
        },
        "Merton Jump": {
            "methodology": "The Merton Jump-Diffusion model extends Black-Scholes by incorporating the possibility of sudden, large price movements (jumps) in the underlying asset. It combines a standard geometric Brownian motion with a compound Poisson process for the jumps.",
            "assumptions": """
            - **Jumps:** Asset prices can experience sudden, discontinuous jumps.
            - **Jump Characteristics:** The timing of jumps is random (Poisson process), and the size of the jumps follows a normal distribution.
            """,
            "formula": r"dS_t = (r - \lambda k)S_t dt + \sigma S_t dW_t + dJ_t",
            "use_cases": "Useful for pricing options on assets prone to sudden price shocks from major news events, such as earnings announcements, clinical trial results, or regulatory decisions."
        }
    }

    st.sidebar.header("Option Parameters")
    ticker = st.sidebar.text_input("Stock Ticker", "AAPL", key="option_ticker")
    if st.sidebar.button("Fetch Market Data"):
        # Check if the ticker is already in the cache
        if ticker in st.session_state.market_data_cache:
            st.sidebar.info(f"Loading '{ticker}' data from cache.")
            cached_data = st.session_state.market_data_cache[ticker]
            st.session_state.spot_price = cached_data['spot_price']
            st.session_state.volatility = cached_data['volatility']
        else:
            # If not in cache, fetch new data
            with st.spinner("Fetching market data..."):
                try:
                    historical_data = data_provider.get_historical_stock_prices(ticker)
                    if not historical_data:
                        st.sidebar.error("Could not fetch historical data.")
                    else:
                        latest_price = historical_data[-1]['c']
                        volatility = volatility_engine.calculate_historical_volatility(ticker, historical_data=historical_data)

                        # Update session state
                        st.session_state.spot_price = latest_price
                        st.session_state.volatility = volatility

                        # Store newly fetched data in the cache
                        st.session_state.market_data_cache[ticker] = {
                            'spot_price': latest_price,
                            'volatility': volatility,
                            'raw_data': historical_data
                        }
                        st.sidebar.success(f"Fetched and cached data for '{ticker}'.")

                except Exception as e:
                    st.sidebar.error(f"Error fetching data: {e}")

        # Handle risk-free rate caching separately
        if st.session_state.risk_free_rate_cache is None:
            try:
                with st.spinner("Fetching risk-free rate..."):
                    risk_free_rate = data_provider.get_risk_free_rate()
                    st.session_state.risk_free_rate_cache = risk_free_rate
                    st.session_state.risk_free_rate = risk_free_rate
                    st.sidebar.success("Fetched and cached risk-free rate.")
            except Exception as e:
                st.sidebar.error(f"Error fetching risk-free rate: {e}")
        else:
            st.session_state.risk_free_rate = st.session_state.risk_free_rate_cache

    pricing_model = st.sidebar.selectbox("Pricing Model", ["Black-Scholes", "Binomial Tree", "Monte Carlo", "Heston", "Merton Jump"])
    
    with st.expander("Model Explanation"):
        explanation = model_explanations[pricing_model]
        st.markdown(f"**Methodology:** {explanation['methodology']}")
        if "assumptions" in explanation:
            st.markdown("**Assumptions:**")
            st.markdown(explanation["assumptions"])
        if "formula" in explanation:
            st.latex(explanation["formula"])
        st.markdown(f"**Use Cases:** {explanation['use_cases']}")

    col1, col2 = st.sidebar.columns(2)
    spot_price = col1.number_input("Spot Price ($)", value=st.session_state.get('spot_price', 100.0), help="The current market price of the underlying asset.")
    strike_price = col2.number_input("Strike Price ($)", value=100.0, help="The price at which the option can be exercised.")
    option_type = col1.selectbox("Option Type", ["Call", "Put"], help="A 'Call' option gives the right to buy, while a 'Put' option gives the right to sell.")
    time_to_expiry = col2.number_input("Time to Expiry (Years)", value=1.0, help="The remaining life of the option, expressed in years.")
    risk_free_rate = col1.number_input("Risk-Free Rate (%)", value=st.session_state.get('risk_free_rate', 0.05) * 100, help="The theoretical rate of return of an investment with zero risk (e.g., government bond yield).") / 100
    volatility = col2.number_input("Volatility (%)", value=st.session_state.get('volatility', 0.20) * 100, help="The annualized standard deviation of the asset's returns, representing its price fluctuation.") / 100

    # Advanced parameters for applicable models
    if pricing_model in ["Heston", "Merton Jump"]:
        st.sidebar.subheader("Advanced Model Parameters")
        if pricing_model == "Heston":
            with st.sidebar.expander("Heston Parameters", expanded=True):
                kappa = st.slider("Kappa (Mean Reversion Speed)", 0.1, 10.0, 2.0, 0.1, help="Controls how quickly the volatility reverts to its long-term average. A high kappa means volatility is expected to stabilize faster.", key="heston_kappa")
                theta = st.slider("Theta (Long-Term Variance)", 0.01, 0.5, 0.05, 0.01, help="The long-term average level of the asset's variance. Volatility will tend to drift towards this level.", key="heston_theta")
                sigma_v = st.slider("Sigma (Volatility of Variance)", 0.01, 1.0, 0.1, 0.01, help="Determines the magnitude of the volatility's own fluctuations. Higher values mean volatility is more erratic.", key="heston_sigma_v")
                rho = st.slider("Rho (Correlation)", -0.99, 0.99, -0.7, 0.01, help="Measures the correlation between the asset's price and its volatility. A negative rho (common in equity markets) means volatility tends to rise as the asset price falls.", key="heston_rho")
        if pricing_model == "Merton Jump":
            with st.sidebar.expander("Merton Jump Parameters", expanded=True):
                lambda_jump = st.slider("Lambda (Jump Intensity)", 0.0, 2.0, 0.1, 0.05, help="The expected number of large price jumps per year. A higher lambda suggests more frequent shocks.", key="merton_lambda")
                mu_jump = st.slider("Mu (Mean Jump Size)", -0.5, 0.5, 0.0, 0.05, help="The average size of the price jump, expressed as a percentage. A negative value indicates an expected downward jump.", key="merton_mu")
                sigma_jump = st.slider("Sigma (Jump Volatility)", 0.0, 0.5, 0.1, 0.01, help="The standard deviation of the jump size, representing the uncertainty or variability in the magnitude of the jumps.", key="merton_sigma")

    if st.sidebar.button("Calculate Option Price", type="primary"):
        option_price = None
        greeks = None
        try:
            if pricing_model == "Black-Scholes":
                option = BlackScholesOption(spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, option_type.lower())
                option_price = option.price()
                greeks = option.calculate_greeks()
            elif pricing_model == "Binomial Tree":
                steps = st.sidebar.number_input("Binomial Tree Steps", 50, 1000, 100, key="single_binomial_steps")
                option = BinomialTreeOption(spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, steps, option_type.lower())
                option_price = option.price()
            elif pricing_model == "Monte Carlo":
                simulations = st.sidebar.number_input("Monte Carlo Simulations", 1000, 50000, 10000, key="single_mc_simulations")
                option = AdvancedOptionPricing(spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, option_type.lower())
                mc_results = option.monte_carlo_pricing(simulations=simulations)
                option_price = mc_results['price']
            elif pricing_model == "Heston":
                option = HestonModel(spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, kappa, theta, sigma_v, rho, option_type.lower())
                option_price = option.price()
            elif pricing_model == "Merton Jump":
                option = MertonJumpModel(spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, lambda_jump, mu_jump, sigma_jump, option_type.lower())
                option_price = option.price()
        except Exception as e:
            st.error(f"An error occurred: {e}")

        if option_price is not None:
            st.metric(f"{pricing_model} Price", f"${option_price:.4f}")

            # --- Sensitivity and Greeks Analysis Section ---
            st.subheader("Price Sensitivity Analysis")
            
            # 1. Spot Price Sensitivity
            spot_range = np.linspace(spot_price * 0.8, spot_price * 1.2, 21)
            price_sensitivity = []
            for s in spot_range:
                if pricing_model == "Black-Scholes":
                    opt = BlackScholesOption(s, strike_price, risk_free_rate, volatility, time_to_expiry, option_type.lower())
                    price_sensitivity.append(opt.price())
                # Add other models if their calculation is fast enough
            
            if price_sensitivity:
                fig = go.Figure(data=go.Scatter(x=spot_range, y=price_sensitivity, mode='lines+markers'))
                fig.update_layout(title='Option Price vs. Spot Price', xaxis_title='Spot Price ($)', yaxis_title='Option Price ($)')
                st.plotly_chart(fig, use_container_width=True)

        if greeks is not None:
            st.subheader("Greeks Analysis")
            st.markdown("The 'Greeks' measure the sensitivity of an option's price to various factors. Understanding them is crucial for risk management.")
            
            cols = st.columns(len(greeks))
            for i, (greek, value) in enumerate(greeks.items()):
                with cols[i]:
                    st.metric(label=greek.capitalize(), value=f"{value:.4f}")

def render_model_comparison_page(data_provider, volatility_engine):
    st.header("Model Comparison")
    st.markdown("Select multiple models to compare their option price calculations side-by-side.")

    # Restore advanced parameters to the sidebar
    st.sidebar.header("Advanced Model Parameters")
    
    # Heston Parameters
    st.sidebar.subheader("Heston Model")
    with st.sidebar.expander("Heston Parameters", expanded=False):
        st.markdown("Configure the parameters for the Heston stochastic volatility model.")
        kappa = st.slider("Kappa (Mean Reversion Speed)", 0.1, 10.0, 2.0, 0.1, help="The rate at which the variance reverts to its long-term mean. Higher kappa means faster reversion.")
        theta = st.slider("Theta (Long-Term Variance)", 0.01, 0.5, 0.05, 0.01, help="The long-term average level of the asset's variance. This is the level that the variance process tends to revert to.")
        sigma_v = st.slider("Sigma (Volatility of Variance)", 0.01, 1.0, 0.1, 0.01, help="The volatility of the variance process itself. It determines the amplitude of the variance's fluctuations.")
        rho = st.slider("Rho (Correlation)", -0.99, 0.99, -0.7, 0.01, help="The correlation between the asset's price shock and its volatility shock. A negative rho is common (the 'leverage effect').")

    # Merton Jump Parameters
    st.sidebar.subheader("Merton Jump Model")
    with st.sidebar.expander("Merton Jump Parameters", expanded=False):
        st.markdown("Configure the parameters for the Merton jump-diffusion model.")
        lambda_jump = st.slider("Lambda (Jump Intensity)", 0.0, 2.0, 0.1, 0.05, help="The average number of jumps per year. A higher lambda means more frequent jumps.")
        mu_jump = st.slider("Mu (Mean Jump Size)", -0.5, 0.5, 0.0, 0.05, help="The average size of the price jumps. It is expressed as a percentage of the asset price.")
        sigma_jump = st.slider("Sigma (Jump Volatility)", 0.0, 0.5, 0.1, 0.01, help="The standard deviation of the jump size, representing the uncertainty in the magnitude of the jumps.")

    # Numerical Method Parameters
    st.sidebar.subheader("Numerical Methods")
    steps = st.sidebar.number_input("Binomial Tree Steps", 50, 1000, 100)
    simulations = st.sidebar.number_input("Monte Carlo Simulations", 1000, 50000, 10000)

    available_models = ["Black-Scholes", "Binomial Tree", "Monte Carlo", "Heston", "Merton Jump"]
    selected_models = st.multiselect("Select Models to Compare", options=available_models, default=["Black-Scholes", "Heston", "Merton Jump"])

    if st.button("Compare Models", type="primary"):
        results = []
        with st.spinner("Running model comparisons..."):
            # Use parameters from the main UI
            spot_price = st.session_state.get('spot_price', 100.0)
            strike_price = 100.0 # Or get from a dedicated input
            risk_free_rate = st.session_state.get('risk_free_rate', 0.05)
            volatility = st.session_state.get('volatility', 0.20)
            time_to_expiry = 1.0 # Or get from a dedicated input
            option_type = 'call' # Or get from a dedicated input

            for model_name in selected_models:
                option_price = None
                try:
                    if model_name == "Black-Scholes":
                        option = BlackScholesOption(spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, option_type)
                        option_price = option.price()
                    elif model_name == "Binomial Tree":
                        option = BinomialTreeOption(spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, steps, option_type)
                        option_price = option.price()
                    elif model_name == "Monte Carlo":
                        option = AdvancedOptionPricing(spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, option_type)
                        mc_results = option.monte_carlo_pricing(simulations=simulations)
                        option_price = mc_results['price']
                    elif model_name == "Heston":
                        # Pass the correct parameters from the sidebar
                        option = HestonModel(spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, kappa, theta, sigma_v, rho, option_type)
                        option_price = option.price()
                    elif model_name == "Merton Jump":
                        # Pass the correct parameters from the sidebar
                        option = MertonJumpModel(spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, lambda_jump, mu_jump, sigma_jump, option_type)
                        option_price = option.price()
                    
                    if option_price is not None:
                        results.append({"Model": model_name, "Option Price": option_price})
                except Exception as e:
                    st.error(f"Error running {model_name}: {e}")
        
        if results:
            st.subheader("Comparison Results")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df.style.format({"Option Price": "${:.4f}"}), use_container_width=True)
            
            fig = px.bar(results_df, x="Model", y="Option Price", title="Model Price Comparison", color="Model", text_auto='.4f')
            fig.update_layout(yaxis_title="Option Price ($)")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Why Do Model Prices Differ?")
            st.markdown("""
            The variation in option prices across different models stems from their unique underlying assumptions and methodologies:

            - **Black-Scholes:** This model's rigid assumptions of constant volatility and no market jumps mean it often serves as a baseline. It may underprice options in volatile markets or overprice them in very stable ones because it cannot adapt to changing conditions.

            - **Heston:** By treating volatility as a random, mean-reverting process, the Heston model can capture the **volatility smile**—a key market phenomenon where options with the same expiration but different strike prices have different implied volatilities. This typically leads to more realistic prices, especially for longer-term options or in markets where volatility is known to fluctuate.

            - **Merton Jump-Diffusion:** This model accounts for the market's ability to make sudden, sharp movements (jumps) that cannot be explained by normal diffusion. The resulting option price includes a premium for this jump risk. This is why the Merton price may be significantly higher than the Black-Scholes price, especially for options on assets prone to unexpected news (e.g., biotech stocks awaiting clinical trial results).

            - **Binomial Tree & Monte Carlo:** These are numerical methods that approximate the option price by simulating the asset's price path.
                - The **Binomial Tree** is ideal for American options as it allows for early exercise decisions at each step. Its accuracy is highly dependent on the number of time steps used.
                - **Monte Carlo** simulation is highly flexible and can price complex, path-dependent options. Its accuracy is a function of the number of simulations; more paths lead to a more accurate result but require more computation time.
            """)

def render_portfolio_analysis_page(data_provider):
    st.header("Portfolio Analysis")
    st.markdown("Build and manage your investment portfolio, and analyze its risk profile.")

    portfolio = st.session_state.portfolio

    st.subheader("Add New Asset")
    with st.form(key="add_asset_form"):
        col1, col2, col3 = st.columns(3)
        ticker = col1.text_input("Ticker Symbol", "AAPL")
        quantity = col2.number_input("Quantity", min_value=0.01, value=10.0)
        price = col3.number_input("Purchase Price", min_value=0.01, value=150.0)
        submitted = st.form_submit_button("Add Asset")
        if submitted:
            portfolio.add_position(ticker, quantity, price)
            st.success(f"Added {quantity} of {ticker} to the portfolio.")

    st.subheader("Current Portfolio")
    if not portfolio.positions:
        st.info("Your portfolio is empty.")
    else:
        positions_df = pd.DataFrame([{"Symbol": symbol, "Quantity": pos['quantity'], "Price": f"${pos['price']:.2f}", "Total Value": f"${pos['quantity'] * pos['price']:.2f}"} for symbol, pos in portfolio.positions.items()])
        st.dataframe(positions_df, use_container_width=True)
        st.metric("Total Portfolio Value", f"${portfolio.get_total_value():,.2f}")

    st.subheader("Risk Analysis")
    if not portfolio.positions:
        st.warning("Add assets to the portfolio to perform risk analysis.")
    else:
        st.markdown("""
        **Value at Risk (VaR)** is a statistical measure that estimates the potential loss in value of a portfolio over a defined period for a given confidence interval. For example, a VaR of $1,000 at a 95% confidence level means there is a 5% chance of losing more than $1,000.
        
        **Conditional Value at Risk (CVaR)**, also known as Expected Shortfall, goes a step further to quantify the average loss that would be incurred if the VaR threshold is breached.
        """)
        risk_analysis_form = st.form("risk_analysis_form")
        risk_analysis_form.subheader("Value at Risk (VaR) and CVaR")
        var_confidence = risk_analysis_form.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01, help="The probability level at which to calculate VaR and CVaR.")
        var_horizon = risk_analysis_form.number_input("Time Horizon (Days)", 1, 30, 10, help="The time period over which to estimate potential losses.")
        num_simulations = risk_analysis_form.number_input("MC Simulations", 100, 10000, 1000, help="The number of simulations to run for the Monte Carlo VaR calculation.")
        run_var = risk_analysis_form.form_submit_button("Calculate VaR & CVaR")

        if run_var:
            with st.spinner("Running Monte Carlo simulation..."):
                var_calculator = ValueAtRisk(portfolio, confidence_level=var_confidence)
                pnl_dist = var_calculator.monte_carlo_simulation(num_simulations, var_horizon)
                if pnl_dist.size > 0:
                    var = var_calculator.get_var_from_pnl(pnl_dist)
                    cvar = var_calculator.get_cvar_from_pnl(pnl_dist)
                    st.metric("Value at Risk (VaR)", f"${var:,.2f}", help=f"Estimates a {var_confidence:.0%} probability of not losing more than this amount over {var_horizon} days.")
                    st.metric("Conditional VaR (CVaR)", f"${cvar:,.2f}", help="Represents the average loss you could expect if your losses exceed the VaR estimate.")
                    fig = px.histogram(pnl_dist, nbins=50, title="Portfolio P&L Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("""
                    **Interpreting the P&L Distribution:**
                    This histogram visualizes the simulated profit and loss (P&L) outcomes for your portfolio over the selected time horizon.
                    - The **x-axis** represents the potential P&L in dollars.
                    - The **y-axis** shows the frequency of each outcome in the simulation.
                    - The shape of the distribution helps you understand the likelihood of different gains and losses. A distribution skewed to the left, for example, indicates a higher probability of smaller losses and a lower probability of larger gains.
                    """)
                else:
                    st.error("Could not run simulation.")
def render_data_browser_page():
    """
    Renders a page to display the cached market data.
    """
    st.header("Market Data Browser")
    st.markdown("This page shows the most recently fetched and cached market data for each ticker.")

    if not st.session_state.market_data_cache:
        st.info("No market data has been cached yet. Fetch data from the 'Option Pricing' page to see it here.")
        return

    st.subheader("Cached Ticker Data")
    for ticker, data in st.session_state.market_data_cache.items():
        with st.expander(f"Data for {ticker}"):
            st.metric("Latest Spot Price", f"${data['spot_price']:.2f}")
            st.metric("Calculated Historical Volatility", f"{data['volatility']:.2%}")
            
            if 'raw_data' in data and data['raw_data']:
                st.subheader("Raw Historical Data")
                df = pd.DataFrame(data['raw_data'])
                st.dataframe(df)
            else:
                st.warning("No raw historical data is stored for this ticker.")

    if st.session_state.risk_free_rate_cache is not None:
        st.subheader("Cached Global Data")
        st.metric("Risk-Free Rate", f"{st.session_state.risk_free_rate_cache:.2%}")

def main():
    """
    Main function to run the Streamlit application.
    This function sets up the page configuration, loads custom CSS,
    initializes session state, and renders the appropriate page based on user selection.
    """
    st.title("📊 Comprehensive Option Pricing & Analysis Suite")
    st.markdown("An educational tool for pricing options, comparing advanced models, and analyzing portfolio risk. Select a page from the navigation panel to begin.")

    # Initialize session state
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = Portfolio()
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'api_valid' not in st.session_state:
        st.session_state.api_valid = False
    if 'market_data_cache' not in st.session_state:
        st.session_state.market_data_cache = {}
    if 'risk_free_rate_cache' not in st.session_state:
        st.session_state.risk_free_rate_cache = None

    # Sidebar for navigation and global settings
    st.sidebar.title("Main Navigation")
    page = st.sidebar.radio("Select a Page", ["Option Pricing", "Model Comparison", "Portfolio Analysis", "Data Browser"], label_visibility="collapsed")

    st.sidebar.header("Global Parameters")
    st.sidebar.subheader("API Configuration")
    api_key = st.sidebar.text_input("Alpha Vantage API Key (Optional)", type="password", value=st.session_state.api_key)

    data_provider = FinancialDataAPI(api_key=api_key)
    volatility_engine = VolatilityEngine(data_provider)

    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        if api_key:
            with st.spinner("Validating API key..."):
                if data_provider.validate_api_key():
                    st.session_state.api_valid = True
                    st.sidebar.success("API key is valid and connected.")
                else:
                    st.session_state.api_valid = False
                    st.sidebar.error("Invalid or expired API key.")
        else:
            st.session_state.api_valid = False
    
    if not st.session_state.api_valid:
        st.sidebar.info("Provide a valid API key to enable automated data fetching.")


    # Page routing
    if page == "Option Pricing":
        render_option_pricing_page(data_provider, volatility_engine)
    elif page == "Model Comparison":
        render_model_comparison_page(data_provider, volatility_engine)
    elif page == "Portfolio Analysis":
        render_portfolio_analysis_page(data_provider)
    elif page == "Data Browser":
        render_data_browser_page()
        render_portfolio_analysis_page(data_provider)

def add_footer():
    st.markdown("---")
    st.markdown("""
    **Disclaimer:** This tool is for educational purposes only. The calculated prices are based on mathematical models and may not reflect real market conditions.
    """)

if __name__ == "__main__":
    main()
    add_footer()
