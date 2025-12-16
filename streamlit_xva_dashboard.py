import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from xva_engine import (
    XVAEngine, Trade, MarketData,
    generate_synthetic_portfolio,
    ExposureSimulator
)

# Page configuration
st.set_page_config(
    page_title="XVA Framework Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #6865F2;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #6865F2;
    }
    .stProgress > div > div > div {
        background-color: #6865F2;
    }
    </style>
""", unsafe_allow_html=True)

# Main header
st.markdown(
    '<h1 class="main-header">📊 XVA Framework for Central Clearing</h1>',
    unsafe_allow_html=True
)

st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
        Funding, Margin, and Capital Valuation Adjustments (FVA, MVA, KVA) 
        for centrally cleared derivative portfolios under Basel III/IV and EMIR.
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = None
if 'market_data' not in st.session_state:
    st.session_state.market_data = None
if 'xva_results' not in st.session_state:
    st.session_state.xva_results = None

# Sidebar for inputs
with st.sidebar:
    st.header("📋 Portfolio Configuration")
    
    num_trades = st.slider(
        "Number of Trades",
        min_value=10,
        max_value=100,
        value=50,
        step=10,
        help="Number of trades in the portfolio"
    )
    
    total_notional = st.number_input(
        "Total Notional (Million)",
        min_value=100.0,
        max_value=1000.0,
        value=250.0,
        step=50.0,
        format="%.0f"
    )
    
    if st.button("🔄 Generate Portfolio", type="primary"):
        with st.spinner("Generating synthetic portfolio..."):
            portfolio = generate_synthetic_portfolio(
                num_trades, total_notional * 1e6, random_seed=42
            )
            st.session_state.portfolio = portfolio
            st.success(f"Generated {len(portfolio)} trades")
    
    st.markdown("---")
    st.header("⚙️ Market Parameters")
    
    risk_free_rate = st.slider(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=5.0,
        value=2.5,
        step=0.1,
        help="Risk-free interest rate"
    ) / 100
    
    funding_spread = st.slider(
        "Funding Spread (bps)",
        min_value=10,
        max_value=200,
        value=50,
        step=5,
        help="Funding spread over risk-free rate"
    ) / 10000
    
    avg_cds_spread = st.slider(
        "Average CDS Spread (bps)",
        min_value=80,
        max_value=200,
        value=120,
        step=10,
        help="Average counterparty CDS spread"
    ) / 10000
    
    st.markdown("---")
    st.header("📊 Regulatory Parameters")
    
    cva_multiplier = st.slider(
        "CVA Multiplier (α)",
        min_value=1.0,
        max_value=2.0,
        value=1.25,
        step=0.05,
        help="Basel III CVA multiplier"
    )
    
    risk_weight = st.slider(
        "Risk Weight (RW)",
        min_value=0.1,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="Risk weight for OTC derivatives"
    )
    
    cost_of_capital = st.slider(
        "Cost of Capital (%)",
        min_value=5.0,
        max_value=15.0,
        value=10.0,
        step=0.5,
        help="Cost of capital for KVA calculation"
    ) / 100
    
    st.markdown("---")
    st.header("⚙️ Simulation Settings")
    
    num_paths = st.slider(
        "Monte Carlo Paths",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Number of simulation paths (more = slower but more accurate). Start with 200 for faster results."
    )
    
    time_steps_per_year = st.slider(
        "Time Steps per Year",
        min_value=12,
        max_value=52,
        value=24,
        step=12,
        help="Number of time steps per year (fewer = faster). 24 = bi-weekly, 52 = weekly"
    )

# Main content area
if st.session_state.portfolio is None:
    st.info("👆 Please generate a portfolio using the sidebar controls.")
    st.markdown("""
    ### Getting Started:
    1. Configure portfolio size and notional
    2. Click "Generate Portfolio" to create synthetic trades
    3. Adjust market and regulatory parameters
    4. Explore XVA calculations and visualisations
    """)
else:
    portfolio = st.session_state.portfolio
    
    # Create market data
    counterparties = list(set(t.counterparty for t in portfolio))
    cds_spreads = {cp: avg_cds_spread * np.random.uniform(0.8, 1.2) 
                   for cp in counterparties}
    
    market_data = MarketData(
        risk_free_rate=risk_free_rate,
        funding_spread=funding_spread,
        cds_spread=cds_spreads,
        volatility={'USD': 0.12, 'EUR': 0.15, 'GBP': 0.13},
        correlations=np.array([[1.0, 0.5, 0.4], [0.5, 1.0, 0.6], [0.4, 0.6, 1.0]])
    )
    
    st.session_state.market_data = market_data
    
    # Calculate XVA
    with st.spinner("Calculating XVA components..."):
        engine = XVAEngine(
            num_paths=num_paths,
            num_steps=time_steps_per_year,  # Use configurable time steps
            cva_multiplier=cva_multiplier,
            risk_weight=risk_weight,
            cost_of_capital=cost_of_capital
        )
        
        xva_results = engine.calculate_all_xva(portfolio, market_data)
        st.session_state.xva_results = xva_results
    
    # Key metrics
    st.subheader("💰 XVA Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fva_pct = xva_results['FVA']/xva_results['Total XVA']*100
        st.metric(
            label="FVA",
            value=f"£{xva_results['FVA']:,.2f}",
            delta=f"{fva_pct:.1f}% of Total XVA"
        )
    
    with col2:
        mva_pct = xva_results['MVA']/xva_results['Total XVA']*100
        st.metric(
            label="MVA",
            value=f"£{xva_results['MVA']:,.2f}",
            delta=f"{mva_pct:.1f}% of Total XVA"
        )
    
    with col3:
        kva_pct = xva_results['KVA']/xva_results['Total XVA']*100
        st.metric(
            label="KVA",
            value=f"£{xva_results['KVA']:,.2f}",
            delta=f"{kva_pct:.1f}% of Total XVA"
        )
    
    with col4:
        total_notional_value = total_notional * 1e6
        xva_pct_notional = (xva_results['Total XVA']/total_notional_value)*100
        st.metric(
            label="Total XVA",
            value=f"£{xva_results['Total XVA']:,.2f}",
            delta=f"{xva_pct_notional:.3f}% of notional"
        )
    
    st.markdown("---")
    
    # XVA Breakdown Chart
    st.subheader("📊 XVA Component Breakdown")
    
    fig_breakdown = go.Figure(data=[
        go.Bar(
            x=['FVA', 'MVA', 'KVA'],
            y=[xva_results['FVA'], xva_results['MVA'], xva_results['KVA']],
            marker_color=['#6865F2', '#5DFFBC', '#FF6B6B'],
            text=[f"£{x:.2f}" for x in [xva_results['FVA'], xva_results['MVA'], xva_results['KVA']]],
            textposition='outside'
        )
    ])
    
    fig_breakdown.update_layout(
        title="XVA Components",
        xaxis_title="Component",
        yaxis_title="Amount (£)",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig_breakdown, use_container_width=True)
    
    # Exposure Profile - Optimized for Performance
    st.subheader("📈 Exposure Profile Analysis")
    
    if xva_results['FVA_details'] is not None:
        time_grid = xva_results['FVA_details']['time_grid']
        epe = xva_results['FVA_details']['epe']
        exposures = xva_results['FVA_details']['exposures']
        
        # Static exposure profile - optimized for performance
        fig_exposure = go.Figure()
        
        if exposures is not None:
            # Calculate confidence bands instead of showing many paths
            # This is much more performant
            percentile_5 = np.percentile(exposures, 5, axis=0)
            percentile_95 = np.percentile(exposures, 95, axis=0)
            
            # Add confidence band (filled area)
            fig_exposure.add_trace(go.Scatter(
                x=np.concatenate([time_grid, time_grid[::-1]]),
                y=np.concatenate([percentile_5, percentile_95[::-1]]),
                fill='toself',
                fillcolor='rgba(104, 101, 242, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                name='5th-95th Percentile'
            ))
            
            # Show just a few sample paths for context
            sample_size = min(5, exposures.shape[0])  # Only 5 sample paths
            if sample_size > 0:
                sample_indices = np.random.choice(exposures.shape[0], sample_size, replace=False)
                
                for idx in sample_indices:
                    fig_exposure.add_trace(go.Scatter(
                        x=time_grid,
                        y=exposures[idx, :],
                        mode='lines',
                        line=dict(color='rgba(104, 101, 242, 0.2)', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # Add EPE (main line)
        fig_exposure.add_trace(go.Scatter(
            x=time_grid,
            y=epe,
            mode='lines',
            name='Expected Positive Exposure',
            line=dict(color='#6865F2', width=3),
            hovertemplate='Time: %{x:.2f} years<br>EPE: £%{y:,.2f}<extra></extra>'
        ))
        
        # Add zero line
        fig_exposure.add_hline(
            y=0, line_dash="dash", line_color="gray", opacity=0.5
        )
        
        fig_exposure.update_layout(
            title="Exposure Profile",
            xaxis_title="Time (Years)",
            yaxis_title="Exposure (£)",
            height=500,
            hovermode='x unified'  # Better hover performance
        )
        
        # Make plot non-interactive for better performance
        st.plotly_chart(
            fig_exposure, 
            use_container_width=True,
            config={
                'staticPlot': True,  # Disable interactivity
                'displayModeBar': False  # Hide toolbar
            }
        )
    
    # Initial Margin Profile
    st.subheader("💳 Initial Margin Profile (MVA)")
    
    if xva_results['MVA_details'] is not None:
        im_profile = xva_results['MVA_details']['im_profile']
        time_grid_im = xva_results['MVA_details']['time_grid']
        
        fig_im = go.Figure()
        fig_im.add_trace(go.Scatter(
            x=time_grid_im,
            y=im_profile,
            mode='lines+markers',
            name='Initial Margin',
            line=dict(color='#5DFFBC', width=3),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(93, 255, 188, 0.2)',
            hovertemplate='Time: %{x:.2f} years<br>IM: £%{y:,.2f}<extra></extra>'
        ))
        
        fig_im.update_layout(
            title="Initial Margin Over Time",
            xaxis_title="Time (Years)",
            yaxis_title="Initial Margin (£)",
            height=400
        )
        
        st.plotly_chart(fig_im, use_container_width=True)
    
    # Capital Profile
    st.subheader("🏦 Regulatory Capital Profile (KVA)")
    
    if xva_results['KVA_details'] is not None:
        capital_profile = xva_results['KVA_details']['capital_profile']
        time_grid_kva = xva_results['KVA_details']['time_grid']
        
        fig_capital = go.Figure()
        fig_capital.add_trace(go.Scatter(
            x=time_grid_kva,
            y=capital_profile,
            mode='lines+markers',
            name='CVA Capital',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.2)',
            hovertemplate='Time: %{x:.2f} years<br>Capital: £%{y:,.2f}<extra></extra>'
        ))
        
        fig_capital.update_layout(
            title="Regulatory Capital Requirement Over Time",
            xaxis_title="Time (Years)",
            yaxis_title="Capital Requirement (£)",
            height=400
        )
        
        st.plotly_chart(fig_capital, use_container_width=True)
    
    # Portfolio Statistics
    st.markdown("---")
    st.subheader("📋 Portfolio Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Portfolio composition by counterparty
        cp_counts = pd.Series([t.counterparty for t in portfolio]).value_counts()
        
        fig_cp = go.Figure(data=[
            go.Bar(
                x=cp_counts.index,
                y=cp_counts.values,
                marker_color='#6865F2',
                text=cp_counts.values,
                textposition='outside'
            )
        ])
        
        fig_cp.update_layout(
            title="Trades by Counterparty",
            xaxis_title="Counterparty",
            yaxis_title="Number of Trades",
            height=300
        )
        
        st.plotly_chart(fig_cp, use_container_width=True)
    
    with col2:
        # Portfolio composition by currency
        currency_counts = pd.Series([t.currency for t in portfolio]).value_counts()
        
        fig_currency = go.Figure(data=[
            go.Pie(
                labels=currency_counts.index,
                values=currency_counts.values,
                hole=0.4,
                marker_colors=['#6865F2', '#5DFFBC', '#FF6B6B']
            )
        ])
        
        fig_currency.update_layout(
            title="Portfolio by Currency",
            height=300
        )
        
        st.plotly_chart(fig_currency, use_container_width=True)
    
    # Detailed Results Table
    st.markdown("---")
    st.subheader("📊 Detailed Results")
    
    results_df = pd.DataFrame({
        'Component': ['FVA', 'MVA', 'KVA', 'Total XVA'],
        'Amount (£)': [
            xva_results['FVA'],
            xva_results['MVA'],
            xva_results['KVA'],
            xva_results['Total XVA']
        ],
        'Percentage of Total': [
            xva_results['FVA'] / xva_results['Total XVA'] * 100,
            xva_results['MVA'] / xva_results['Total XVA'] * 100,
            xva_results['KVA'] / xva_results['Total XVA'] * 100,
            100.0
        ]
    })
    
    st.dataframe(
        results_df.style.format({
            'Amount (£)': '{:,.2f}',
            'Percentage of Total': '{:.2f}%'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Download button
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="xva_results.csv",
        mime="text/csv"
    )

