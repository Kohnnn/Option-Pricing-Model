import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

# Import our custom models and accuracy tools
from models import BlackScholesOption, AdvancedOptionPricing, BinomialTreeOption
from model_accuracy import calculate_model_accuracy, visualize_pricing_accuracy

# Set page configuration
st.set_page_config(
    page_title="Option Pricing Model",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Dark theme specific styles */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #1c2333;
    }
    .stTextInput>div>div>input {
        color: #ffffff;
        background-color: #262730;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #4a4a4a;
    }
    .stSelectbox>div>div>select {
        color: #ffffff;
        background-color: #262730;
    }
    .metric-container {
        background-color: #262730;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        color: #ffffff;
    }
    .explanation-text {
        color: #aaaaaa;
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

def main():
    st.title("📊 Option Pricing Model Dashboard")
    st.markdown("""
    Welcome to the Option Pricing Model! This tool helps you analyze and price financial options 
    using various sophisticated mathematical models.
    """)
    
    # Sidebar for input parameters
    st.sidebar.header("Option Parameters")
    st.sidebar.markdown("""
    Set the key parameters that define your option's characteristics.
    """)
    
    # Input parameters with improved ranges and defaults
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        spot_price = st.number_input("Spot Price ($)", min_value=1.0, max_value=1000.0, value=100.0, step=1.0)
        st.markdown(add_tooltip("ℹ️ Spot Price", "Current market price of the underlying asset"), unsafe_allow_html=True)
        
        strike_price = st.number_input("Strike Price ($)", min_value=1.0, max_value=1000.0, value=100.0, step=1.0)
        st.markdown(add_tooltip("ℹ️ Strike Price", "Price at which the option can be exercised"), unsafe_allow_html=True)
        
        option_type = st.selectbox("Option Type", ["Call", "Put"])
        st.markdown(add_tooltip("ℹ️ Option Type", "Call: Right to buy, Put: Right to sell"), unsafe_allow_html=True)
    
    with col2:
        risk_free_rate = st.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.1) / 100
        st.markdown(add_tooltip("ℹ️ Risk-Free Rate", "Theoretical return of a risk-free investment"), unsafe_allow_html=True)
        
        volatility = st.number_input("Volatility (%)", min_value=0.1, max_value=100.0, value=20.0, step=0.5) / 100
        st.markdown(add_tooltip("ℹ️ Volatility", "Measure of how much the asset price can change"), unsafe_allow_html=True)
        
        time_to_expiry = st.number_input("Time to Expiry (Years)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
        st.markdown(add_tooltip("ℹ️ Time to Expiry", "Remaining time until the option expires"), unsafe_allow_html=True)
    
    # Simulation parameters
    st.sidebar.header("Simulation Settings")
    st.sidebar.markdown("Adjust computational parameters for pricing models")
    
    simulations = st.sidebar.slider("Number of Simulations", min_value=1000, max_value=100000, value=50000, step=1000)
    st.sidebar.markdown(add_tooltip("ℹ️ Simulations", "More simulations increase pricing accuracy"), unsafe_allow_html=True)
    
    confidence_level = st.sidebar.slider("Confidence Level (%)", min_value=80, max_value=99, value=95, step=1) / 100
    st.sidebar.markdown(add_tooltip("ℹ️ Confidence Level", "Statistical confidence for price estimation"), unsafe_allow_html=True)
    
    # Pricing button
    if st.sidebar.button("Calculate Option Price", type="primary"):
        # Initialize option pricing models
        advanced_option = AdvancedOptionPricing(
            spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, option_type.lower()
        )
        
        # Create columns for different pricing methods
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pricing Results")
            
            # Black-Scholes Pricing
            bs_price = advanced_option.price()
            st.metric("Black-Scholes Price", f"${bs_price:.2f}")
            st.markdown(add_tooltip("ℹ️ Black-Scholes", "Standard theoretical option pricing model"), unsafe_allow_html=True)
            
            # Monte Carlo Simulation
            mc_results = advanced_option.monte_carlo_pricing(simulations=simulations, confidence_level=confidence_level)
            st.markdown("#### Monte Carlo Simulation")
            st.metric("Monte Carlo Price", f"${mc_results['price']:.2f}")
            st.markdown(add_tooltip("ℹ️ Monte Carlo", "Price estimated through random sampling"), unsafe_allow_html=True)
            
            st.metric("Median Monte Carlo Price", f"${mc_results['median_price']:.2f}")
            st.markdown(add_tooltip("ℹ️ Median Price", "Middle value of simulated option prices"), unsafe_allow_html=True)
            
            # Confidence Interval
            st.markdown("**Monte Carlo Confidence Interval:**")
            st.text(f"({mc_results['confidence_interval'][0]:.2f}, {mc_results['confidence_interval'][1]:.2f})")
            st.markdown(add_tooltip("ℹ️ Confidence Interval", "Range likely containing the true option price"), unsafe_allow_html=True)
            
            st.metric("Standard Error", f"{mc_results['standard_error']:.4f}")
            st.markdown(add_tooltip("ℹ️ Standard Error", "Measure of price estimation precision"), unsafe_allow_html=True)
            # Simulation Details
            st.markdown("#### Simulation Details")
            st.metric("Drift", f"{mc_results['simulation_details']['drift']:.4f}")
            st.markdown(add_tooltip("ℹ️ Drift", "Expected return of the underlying asset"), unsafe_allow_html=True)
            
            st.metric("Diffusion", f"{mc_results['simulation_details']['diffusion']:.4f}")
            st.markdown(add_tooltip("ℹ️ Diffusion", "Measure of price volatility spread"), unsafe_allow_html=True)
        
        with col2:
            st.subheader("Advanced Analysis")
            
            # Quasi-Monte Carlo Pricing
            qmc_price = advanced_option.quasi_monte_carlo_pricing(simulations=simulations)
            st.metric("Quasi-Monte Carlo Price", f"${qmc_price:.2f}")
            st.markdown(add_tooltip("ℹ️ Quasi-Monte Carlo", "More precise sampling technique"), unsafe_allow_html=True)
            
            # Option Greeks with Expandable Sections
            st.markdown("#### Option Greeks")
            
            # Simplified Greeks Explanations
            greeks_explanations = {
                'delta': {
                    'name': 'Delta',
                    'simple_explanation': 'How much the option price changes when the stock price moves',
                    'detailed_explanation': 'Delta measures the rate of change in the option\'s price relative to changes in the underlying asset\'s price. It ranges from 0 to 1 for call options and -1 to 0 for put options.'
                },
                'gamma': {
                    'name': 'Gamma',
                    'simple_explanation': 'Rate of change in delta',
                    'detailed_explanation': 'Gamma represents the curvature of the option\'s price change. It shows how quickly delta changes as the underlying asset\'s price moves.'
                },
                'theta': {
                    'name': 'Theta',
                    'simple_explanation': 'How much value the option loses each day',
                    'detailed_explanation': 'Theta measures the rate of decline in the option\'s value over time, also known as time decay. It represents how much value an option loses as it approaches expiration.'
                },
                'vega': {
                    'name': 'Vega',
                    'simple_explanation': 'How option price changes with volatility',
                    'detailed_explanation': 'Vega indicates how much an option\'s price might change when the volatility of the underlying asset changes. Higher vega means the option is more sensitive to volatility.'
                },
                'rho': {
                    'name': 'Rho',
                    'simple_explanation': 'How option price changes with interest rates',
                    'detailed_explanation': 'Rho measures the sensitivity of the option\'s price to changes in the risk-free interest rate. It shows how much the option\'s value might change if interest rates shift.'
                }
            }
            
            # Calculate Greeks
            greeks = advanced_option.calculate_greeks()
            
            # Create expandable sections for each Greek
            for greek, value in greeks.items():
                greek_info = greeks_explanations.get(greek, {})
                
                # Expandable section for each Greek, expanded by default
                with st.expander(f"{greek_info.get('name', greek.capitalize())} Greek", expanded=True):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.metric(f"{greek_info.get('name', greek.capitalize())} Value", f"{value:.4f}")
                    
                    with col_b:
                        st.write("**Simple Explanation:**")
                        st.write(greek_info.get('simple_explanation', 'Sensitivity measure'))
                    
                    st.write("**Detailed Explanation:**")
                    st.write(greek_info.get('detailed_explanation', 'Additional details about this Greek'))
        
        # Model Accuracy Comparison
        st.subheader("Model Accuracy Comparison")
        add_explanation("Comparing different option pricing models")
        
        # Calculate accuracy metrics
        accuracy_df = calculate_model_accuracy(
            spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, option_type.lower()
        )
        
        # Create interactive Plotly bar chart for pricing comparison
        fig_prices = go.Figure(data=[
            go.Bar(
                x=accuracy_df.index, 
                y=accuracy_df['Price'], 
                text=[f'${price:.2f}' for price in accuracy_df['Price']],
                textposition='auto',
                marker_color='rgba(58, 71, 80, 0.6)',
                marker_line_color='rgba(58, 71, 80, 1.0)',
                marker_line_width=1.5
            )
        ])
        fig_prices.update_layout(
            title='Option Pricing Comparison',
            xaxis_title='Pricing Method',
            yaxis_title='Option Price ($)',
            template='plotly_white'
        )
        st.plotly_chart(fig_prices, use_container_width=True)
        add_explanation("Visual comparison of prices from different models")
        
        # Create interactive Plotly bar chart for relative differences
        fig_diff = go.Figure(data=[
            go.Bar(
                x=accuracy_df.index, 
                y=accuracy_df['Relative Difference (%)'], 
                text=[f'{diff:.2f}%' for diff in accuracy_df['Relative Difference (%)']],
                textposition='auto',
                marker_color='rgba(255, 64, 129, 0.6)',
                marker_line_color='rgba(255, 64, 129, 1.0)',
                marker_line_width=1.5
            )
        ])
        fig_diff.update_layout(
            title='Relative Difference from Black-Scholes',
            xaxis_title='Pricing Method',
            yaxis_title='Relative Difference (%)',
            template='plotly_white'
        )
        st.plotly_chart(fig_diff, use_container_width=True)
        add_explanation("Percentage difference between models compared to Black-Scholes")

def add_footer():
    st.markdown("---")
    st.markdown("""
    ### About This Tool
    - **Option Pricing Model Dashboard** provides comprehensive option valuation techniques
    - Uses Black-Scholes, Monte Carlo, and Quasi-Monte Carlo methods
    - Compares accuracy of different pricing models
    - Calculates option Greeks and provides detailed statistical analysis
    """)
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Created by:** [Kiet Vo](linkedin.com/in/kiet-vo-097)

    Disclaimer - This is my first take on quant and option pricing. 
    The model is far from having any real application. 
    I am no expert in the financial and future market and still learning while doing.
    """)

if __name__ == "__main__":
    main()
    add_footer()
