from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from option_pricing import calculate_option_price, calculate_option_greeks
import streamlit as st
import numpy as np
import plotly.graph_objs as go
from models import BlackScholesOption, BinomialTreeOption

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    """
    Render the main option pricing interface
    """
    return render_template('index.html')

@app.route('/calculate_option', methods=['POST'])
def calculate_option():
    """
    API endpoint to calculate option prices and Greeks
    """
    try:
        data = request.json
        
        # Extract parameters
        pricing_model = data.get('pricing_model', 'black_scholes')
        spot_price = float(data['spot_price'])
        strike_price = float(data['strike_price'])
        risk_free_rate = float(data['risk_free_rate'])
        volatility = float(data['volatility'])
        time_to_expiry = float(data['time_to_expiry'])
        option_type = data.get('option_type', 'call')
        
        # Calculate option price
        price = calculate_option_price(
            pricing_model=pricing_model,
            spot_price=spot_price,
            strike_price=strike_price,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            time_to_expiry=time_to_expiry,
            option_type=option_type
        )
        
        # Calculate Greeks (only for Black-Scholes)
        greeks = {}
        if pricing_model == 'black_scholes':
            greeks = calculate_option_greeks(
                pricing_model=pricing_model,
                spot_price=spot_price,
                strike_price=strike_price,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                time_to_expiry=time_to_expiry,
                option_type=option_type
            )
        
        return jsonify({
            'option_price': price,
            'greeks': greeks
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def main():
    st.title('Advanced Option Pricing Model')
    
    # Sidebar for option parameters
    st.sidebar.header('Option Parameters')
    
    # Basic Option Parameters
    spot_price = st.sidebar.number_input('Spot Price', min_value=0.01, value=100.0, step=0.1)
    strike_price = st.sidebar.number_input('Strike Price', min_value=0.01, value=100.0, step=0.1)
    risk_free_rate = st.sidebar.number_input('Risk-Free Rate (%)', min_value=0.0, max_value=100.0, value=5.0, step=0.1) / 100
    volatility = st.sidebar.number_input('Volatility (%)', min_value=0.0, max_value=100.0, value=20.0, step=0.1) / 100
    time_to_expiry = st.sidebar.number_input('Time to Expiry (Years)', min_value=0.01, value=1.0, step=0.1)
    option_type = st.sidebar.selectbox('Option Type', ['Call', 'Put'])
    
    # Create option object
    option = BlackScholesOption(
        spot_price=spot_price, 
        strike_price=strike_price, 
        risk_free_rate=risk_free_rate, 
        volatility=volatility, 
        time_to_expiry=time_to_expiry,
        option_type=option_type.lower()
    )
    
    # Main content area
    st.header('Option Pricing Results')
    
    # Pricing and Greeks
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Option Pricing')
        st.write(f'Option Price: ${option.price():.2f}')
        st.write(f'Delta: {option.delta():.4f}')
        st.write(f'Gamma: {option.gamma():.4f}')
        st.write(f'Vega: {option.vega():.4f}')
        st.write(f'Theta: {option.theta():.4f}')
    
    # Implied Volatility
    with col2:
        st.subheader('Implied Volatility')
        market_price = st.number_input('Market Price', min_value=0.01, value=option.price(), step=0.1)
        try:
            implied_vol = option.implied_volatility(market_price)
            st.write(f'Implied Volatility: {implied_vol * 100:.2f}%')
        except ValueError as e:
            st.error(str(e))
    
    # Scenario Analysis
    st.header('Scenario Analysis')
    
    # Perform scenario analysis
    scenario_results = option.scenario_analysis()
    
    # Create heatmap for option prices
    fig_prices = go.Figure(data=go.Heatmap(
        z=scenario_results['prices'],
        x=scenario_results['volatilities'],
        y=scenario_results['spot_prices'],
        colorscale='Viridis',
        colorbar=dict(title='Option Price')
    ))
    fig_prices.update_layout(
        title='Option Price Sensitivity',
        xaxis_title='Volatility',
        yaxis_title='Spot Price'
    )
    st.plotly_chart(fig_prices)
    
    # Create heatmap for deltas
    fig_deltas = go.Figure(data=go.Heatmap(
        z=scenario_results['deltas'],
        x=scenario_results['volatilities'],
        y=scenario_results['spot_prices'],
        colorscale='RdBu',
        colorbar=dict(title='Delta')
    ))
    fig_deltas.update_layout(
        title='Delta Sensitivity',
        xaxis_title='Volatility',
        yaxis_title='Spot Price'
    )
    st.plotly_chart(fig_deltas)

if __name__ == '__main__':
    main()
    app.run(debug=True, port=5000)
