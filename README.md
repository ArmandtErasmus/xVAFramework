# Advanced XVA Framework: FVA, MVA, KVA for Central Clearing

A comprehensive, industry-grade XVA (eXposure Valuation Adjustment) framework implementing Funding Valuation Adjustment (FVA), Margin Valuation Adjustment (MVA), and Capital Valuation Adjustment (KVA) calculations for centrally cleared derivative portfolios.

## Features

- **FVA (Funding Valuation Adjustment)**: Captures treasury funding costs for uncollateralised exposures using Monte Carlo simulation
- **MVA (Margin Valuation Adjustment)**: Prices initial margin costs over trade lifecycle using ISDA SIMM methodology
- **KVA (Capital Valuation Adjustment)**: Quantifies regulatory capital costs under Basel III/IV frameworks
- **Exposure Simulation**: Monte Carlo simulation of portfolio exposure profiles
- **COA Analysis**: Close-Out Amount simulation for Margin Period of Risk (MPoR) analysis
- **Interactive Dashboard**: Professional Streamlit interface with real-time simulations and animations
- **Object-Oriented Design**: Clean, extensible code architecture

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Dashboard

To launch the interactive Streamlit dashboard:

```bash
streamlit run streamlit_xva_dashboard.py
```

The dashboard will open in your default web browser, typically at `http://localhost:8501`.

### Features of the Dashboard

- **Real-Time Simulations**: Watch exposure profiles being constructed in real-time
- **Interactive Parameters**: Adjust market and regulatory parameters and see immediate updates
- **Comprehensive Visualisations**: Exposure profiles, initial margin evolution, capital requirements
- **Portfolio Analysis**: Counterparty and currency breakdowns
- **Export Capabilities**: Download results as CSV

### Using the Python API

```python
from xva_engine import (
    XVAEngine, Trade, MarketData,
    generate_synthetic_portfolio
)

# Generate synthetic portfolio
portfolio = generate_synthetic_portfolio(
    num_trades=50, total_notional=250e6
)

# Create market data
market_data = MarketData(
    risk_free_rate=0.025,
    funding_spread=0.005,
    cds_spread={'CP1': 0.008, 'CP2': 0.012},
    volatility={'USD': 0.12, 'EUR': 0.15},
    correlations=np.array([[1.0, 0.5], [0.5, 1.0]])
)

# Calculate XVA
engine = XVAEngine()
results = engine.calculate_all_xva(portfolio, market_data)

print(f"FVA: £{results['FVA']:,.2f}")
print(f"MVA: £{results['MVA']:,.2f}")
print(f"KVA: £{results['KVA']:,.2f}")
print(f"Total XVA: £{results['Total XVA']:,.2f}")
```

## Project Structure

- `xva_engine.py`: Core OOP implementation of XVA calculations
- `streamlit_xva_dashboard.py`: Interactive Streamlit dashboard with real-time simulations
- `xva_framework.tex`: LaTeX documentation with comprehensive theory and mathematics
- `requirements.txt`: Python package dependencies

## Mathematical Framework

The framework implements:

- **FVA**: Integration of funding spread over expected positive exposure
- **MVA**: Present value of funding costs for initial margin using ISDA SIMM
- **KVA**: Cost of regulatory capital under Basel III CVA capital framework
- **MPoR**: Margin Period of Risk modelling for Close-Out Amount simulation

## Regulatory Context

This tool addresses post-crisis regulatory requirements:

- **EMIR**: European Market Infrastructure Regulation for central clearing
- **Basel III/IV**: Capital requirements including CVA multiplier (α = 1.25) and risk weights (RW = 0.50)
- **ISDA SIMM**: Standard Initial Margin Model for margin calculations
- **CCP Clearing**: Central Counterparty margin algorithms and default waterfall analysis

## Performance Considerations

- Monte Carlo simulations can be computationally intensive
- Adjust `num_paths` parameter to balance accuracy against speed
- For production use, consider implementing parallel processing or GPU acceleration

