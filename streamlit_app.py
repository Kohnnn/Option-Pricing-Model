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
            "methodology": "The Black-Scholes model provides a theoretical estimate of the price of European-style options.",
            "formula": r"C(S, t) = S_0 N(d_1) - K e^{-rT} N(d_2)",
            "use_cases": "Best for pricing European options on non-dividend-paying stocks."
        },
        "Binomial Tree": {
            "methodology": "The Binomial Tree model breaks down the time to expiration into discrete intervals.",
            "use_cases": "Flexible for pricing American and European options, and can handle dividends."
        },
        "Monte Carlo": {
            "methodology": "The Monte Carlo simulation generates random price paths for the underlying asset.",
            "use_cases": "Useful for pricing complex and path-dependent options."
        },
        "Heston": {
            "methodology": "The Heston model is a stochastic volatility model, assuming volatility is not constant.",
            "use_cases": "More realistic for pricing options where volatility fluctuates over time."
        },
        "Merton Jump": {
            "methodology": "The Merton Jump-Diffusion model incorporates sudden, large price movements (jumps).",
            "use_cases": "Suitable for pricing options on assets subject to sudden price shocks."
        }
    }

    st.sidebar.header("Option Parameters")
    ticker = st.sidebar.text_input("Stock Ticker", "AAPL", key="option_ticker")
    if st.sidebar.button("Fetch Market Data"):
        with st.spinner("Fetching market data..."):
            try:
                price = data_provider.get_latest_stock_price(ticker)
                if price:
                    st.session_state.spot_price = price
                volatility = volatility_engine.calculate_historical_volatility(ticker)
                st.session_state.volatility = volatility
                risk_free_rate = data_provider.get_risk_free_rate()
                st.session_state.risk_free_rate = risk_free_rate
            except Exception as e:
                st.sidebar.error(f"Error fetching data: {e}")

    pricing_model = st.sidebar.selectbox("Pricing Model", list(model_explanations.keys()))
    
    with st.expander("Model Explanation"):
        explanation = model_explanations[pricing_model]
        st.markdown(f"**Methodology:** {explanation['methodology']}")
        if "formula" in explanation:
            st.latex(explanation["formula"])
        st.markdown(f"**Use Cases:** {explanation['use_cases']}")

    col1, col2 = st.sidebar.columns(2)
    spot_price = col1.number_input("Spot Price ($)", value=st.session_state.get('spot_price', 100.0))
    strike_price = col2.number_input("Strike Price ($)", value=100.0)
    option_type = col1.selectbox("Option Type", ["Call", "Put"])
    time_to_expiry = col2.number_input("Time to Expiry (Years)", value=1.0)
    risk_free_rate = col1.number_input("Risk-Free Rate (%)", value=st.session_state.get('risk_free_rate', 0.05) * 100) / 100
    volatility = col2.number_input("Volatility (%)", value=st.session_state.get('volatility', 0.20) * 100) / 100

    if st.sidebar.button("Calculate Option Price", type="primary"):
        option_price = None
        greeks = None
        try:
            if pricing_model == "Black-Scholes":
                option = BlackScholesOption(spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, option_type.lower())
                option_price = option.price()
                greeks = option.calculate_greeks()
            # Add other models here
        except Exception as e:
            st.error(f"An error occurred: {e}")

        if option_price is not None:
            st.metric(f"{pricing_model} Price", f"${option_price:.4f}")
        if greeks is not None:
            st.subheader("Greeks Analysis")
            st.json(greeks)

def render_model_comparison_page(data_provider, volatility_engine):
    st.header("Model Comparison")
    st.markdown("Select multiple models to compare their option price calculations side-by-side.")

    available_models = ["Black-Scholes", "Binomial Tree", "Monte Carlo", "Merton Jump"]
    selected_models = st.multiselect("Select Models to Compare", options=available_models, default=["Black-Scholes", "Binomial Tree"])

    st.sidebar.header("Advanced Model Parameters")
    steps = st.sidebar.slider("Steps (Binomial Tree)", 50, 500, 100, 10)
    simulations = st.sidebar.slider("Simulations (Monte Carlo)", 1000, 100000, 10000, 1000)
    lambda_jump = st.sidebar.slider("Jump Intensity (λ)", 0.0, 2.0, 0.5, 0.1)
    mu_jump = st.sidebar.slider("Avg. Jump Size (μ)", -0.5, 0.5, 0.0, 0.05)
    sigma_jump = st.sidebar.slider("Jump Volatility (σ)", 0.0, 0.5, 0.1, 0.01)

    if st.button("Compare Models", type="primary"):
        results = []
        with st.spinner("Running model comparisons..."):
            # These parameters should be defined in the main app layout
            spot_price = 100.0
            strike_price = 100.0
            risk_free_rate = 0.05
            volatility = 0.20
            time_to_expiry = 1.0
            option_type = 'call'

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
                    elif model_name == "Merton Jump":
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
        risk_analysis_form = st.form("risk_analysis_form")
        risk_analysis_form.subheader("Value at Risk (VaR) and CVaR")
        var_confidence = risk_analysis_form.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
        var_horizon = risk_analysis_form.number_input("Time Horizon (Days)", 1, 30, 10)
        num_simulations = risk_analysis_form.number_input("MC Simulations", 100, 10000, 1000)
        run_var = risk_analysis_form.form_submit_button("Calculate VaR & CVaR")

        if run_var:
            with st.spinner("Running Monte Carlo simulation..."):
                var_calculator = ValueAtRisk(portfolio, confidence_level=var_confidence)
                pnl_dist = var_calculator.monte_carlo_simulation(num_simulations, var_horizon)
                if pnl_dist.size > 0:
                    var = var_calculator.get_var_from_pnl(pnl_dist)
                    cvar = var_calculator.get_cvar_from_pnl(pnl_dist)
                    st.metric("Value at Risk (VaR)", f"${var:,.2f}")
                    st.metric("Conditional VaR (CVaR)", f"${cvar:,.2f}")
                    fig = px.histogram(pnl_dist, nbins=50, title="Portfolio P&L Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Could not run simulation.")
def main():
    """
    Main function to run the Streamlit application.
    This function sets up the page configuration, loads custom CSS,
    initializes session state, and renders the appropriate page based on user selection.
    """
    st.title("📊 Option Pricing Model Dashboard")
    st.markdown("Welcome to the advanced option pricing and risk analysis dashboard. "
                "Select a tool from the sidebar to get started.")

    # Initialize session state
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = Portfolio()
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""

    # Sidebar for navigation and global settings
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Option Pricing", "Model Comparison", "Portfolio Analysis"])

    st.sidebar.header("Global Parameters")
    st.sidebar.subheader("API Configuration")
    api_key = st.sidebar.text_input("Alpha Vantage API Key", type="password", value=st.session_state.api_key)
    
    if api_key:
        st.session_state.api_key = api_key
        try:
            data_provider = FinancialDataAPI(api_key=api_key)
            # Perform a quick check to validate the API key
            if data_provider.get_latest_stock_price("IBM") is None:
                st.sidebar.error("Invalid API key or no data for test ticker.")
                return
            volatility_engine = VolatilityEngine(data_provider)
            st.sidebar.success("API key is valid.")
        except (ValueError, IOError) as e:
            st.sidebar.error(f"API Error: {e}")
            return
    else:
        st.sidebar.warning("Please enter your Alpha Vantage API key to fetch market data.")
        return

    # Page routing
    if page == "Option Pricing":
        render_option_pricing_page(data_provider, volatility_engine)
    elif page == "Model Comparison":
        render_model_comparison_page(data_provider, volatility_engine)
    elif page == "Portfolio Analysis":
        render_portfolio_analysis_page(data_provider)

def add_footer():
    st.markdown("---")
    st.markdown("""
    **Disclaimer:** This tool is for educational purposes only. The calculated prices are based on mathematical models and may not reflect real market conditions.
    """)

if __name__ == "__main__":
    main()
    add_footer()
