import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from scipy.stats import norm
import warnings


@dataclass
class Trade:
    """Represents a single derivative trade."""
    trade_id: str
    counterparty: str
    notional: float
    maturity: float  # Years
    currency: str
    trade_type: str  # 'IRS', 'CCS', 'FX', etc.
    fixed_rate: Optional[float] = None
    float_index: Optional[str] = None
    start_date: float = 0.0  # Years from today
    
    def __post_init__(self):
        """Validate trade parameters."""
        if self.notional <= 0:
            raise ValueError("Notional must be positive")
        if self.maturity <= 0:
            raise ValueError("Maturity must be positive")


@dataclass
class MarketData:
    """Market data container for XVA calculations."""
    risk_free_rate: float
    funding_spread: float  # Funding spread over risk-free rate
    cds_spread: Dict[str, float]  # Counterparty CDS spreads (as decimals)
    volatility: Dict[str, float]  # Volatility by currency/asset
    correlations: np.ndarray  # Correlation matrix
    initial_margin_risk_weight: float = 0.02  # ISDA SIMM risk weight
    mpor: float = 5.0 / 252.0  # Margin Period of Risk (5 days)
    
    def __post_init__(self):
        """Validate market data."""
        if self.risk_free_rate < 0:
            warnings.warn("Negative risk-free rate")
        if self.funding_spread < 0:
            warnings.warn("Negative funding spread")


class XVACalculator(ABC):
    """Abstract base class for XVA calculations."""
    
    @abstractmethod
    def calculate(self, portfolio: List[Trade], 
                  market_data: MarketData,
                  time_grid: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate XVA adjustment.
        
        Args:
            portfolio: List of Trade objects
            market_data: MarketData object
            time_grid: Optional time grid for calculations
        
        Returns:
            Dictionary containing XVA results and intermediate calculations
        """
        pass


class ExposureSimulator:
    """Simulates portfolio exposure paths using Monte Carlo."""
    
    def __init__(self, num_paths: int = 10000, num_steps: int = 252):
        """
        Initialise exposure simulator.
        
        Args:
            num_paths: Number of Monte Carlo paths
            num_steps: Number of time steps per year
        """
        self.num_paths = num_paths
        self.num_steps = num_steps
    
    def simulate_exposure(self, 
                         portfolio: List[Trade],
                         market_data: MarketData,
                         time_grid: np.ndarray,
                         random_seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate exposure paths using Monte Carlo.
        
        Args:
            portfolio: List of Trade objects
            market_data: MarketData object
            time_grid: Time grid for simulation
            random_seed: Random seed for reproducibility
        
        Returns:
            Array of shape (num_paths, len(time_grid)) with exposure paths
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        exposures = np.zeros((self.num_paths, len(time_grid)))
        
        # Get portfolio parameters
        max_maturity = max(trade.maturity for trade in portfolio)
        currencies = list(set(trade.currency for trade in portfolio))
        
        # Generate correlated random shocks
        num_factors = len(currencies)
        if num_factors == 0:
            return exposures
        
        # Simplified: use single factor model
        for path in range(self.num_paths):
            # Generate random shocks
            dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 0.01
            shocks = np.random.normal(0, np.sqrt(dt), len(time_grid))
            
            for i, t in enumerate(time_grid):
                exposure = 0.0
                
                for trade in portfolio:
                    if trade.start_date <= t < trade.maturity:
                        # Simplified trade valuation
                        time_to_maturity = trade.maturity - t
                        
                        if trade.trade_type == 'IRS':
                            # Interest rate swap valuation
                            vol = market_data.volatility.get(trade.currency, 0.15)
                            # Simplified: value depends on rate movement
                            rate_shock = shocks[i] * vol
                            value = (trade.notional * 
                                   (trade.fixed_rate - market_data.risk_free_rate - rate_shock) * 
                                   time_to_maturity)
                            exposure += value
                        
                        elif trade.trade_type == 'CCS':
                            # Cross-currency swap
                            vol = market_data.volatility.get(trade.currency, 0.15)
                            fx_shock = shocks[i] * vol
                            value = trade.notional * fx_shock * time_to_maturity
                            exposure += value
                        
                        else:
                            # Generic trade
                            vol = market_data.volatility.get(trade.currency, 0.15)
                            value = trade.notional * shocks[i] * vol * np.sqrt(time_to_maturity)
                            exposure += value
                
                exposures[path, i] = exposure
        
        return exposures
    
    def calculate_expected_exposure(self, exposures: np.ndarray) -> np.ndarray:
        """Calculate expected exposure profile."""
        return np.mean(exposures, axis=0)
    
    def calculate_positive_exposure(self, exposures: np.ndarray) -> np.ndarray:
        """Calculate expected positive exposure (EPE)."""
        positive_exposures = np.maximum(exposures, 0)
        return np.mean(positive_exposures, axis=0)
    
    def calculate_effective_expected_exposure(self, epe: np.ndarray) -> np.ndarray:
        """Calculate Effective Expected Exposure (running maximum of EPE)."""
        eee = np.zeros_like(epe)
        current_max = 0.0
        for i, val in enumerate(epe):
            current_max = max(current_max, val)
            eee[i] = current_max
        return eee


class FVACalculator(XVACalculator):
    """Funding Valuation Adjustment calculator."""
    
    def __init__(self, num_paths: int = 10000, num_steps: int = 252):
        """
        Initialise FVA calculator.
        
        Args:
            num_paths: Number of Monte Carlo paths
            num_steps: Number of time steps per year
        """
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.exposure_simulator = ExposureSimulator(num_paths, num_steps)
    
    def calculate(self, portfolio: List[Trade],
                 market_data: MarketData,
                 time_grid: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate FVA.
        
        Returns:
            Dictionary with FVA value and intermediate calculations
        """
        if not portfolio:
            return {'FVA': 0.0, 'exposures': None, 'epe': None}
        
        # Create time grid
        max_maturity = max(trade.maturity for trade in portfolio)
        if time_grid is None:
            num_points = int(max_maturity * self.num_steps) + 1
            time_grid = np.linspace(0, max_maturity, num_points)
        
        dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 0.01
        
        # Simulate exposures
        exposures = self.exposure_simulator.simulate_exposure(
            portfolio, market_data, time_grid
        )
        
        # Calculate Expected Positive Exposure (EPE)
        epe = self.exposure_simulator.calculate_positive_exposure(exposures)
        
        # Calculate FVA
        fva = 0.0
        for i, t in enumerate(time_grid[:-1]):
            discount_factor = np.exp(-market_data.risk_free_rate * t)
            fva += market_data.funding_spread * epe[i] * discount_factor * dt
        
        return {
            'FVA': fva,
            'exposures': exposures,
            'epe': epe,
            'time_grid': time_grid
        }


class MVACalculator(XVACalculator):
    """Margin Valuation Adjustment calculator."""
    
    def __init__(self, mpor: float = 5.0 / 252.0):
        """
        Initialise MVA calculator.
        
        Args:
            mpor: Margin Period of Risk in years (default 5 days)
        """
        self.mpor = mpor
    
    def calculate_initial_margin_simm(self,
                                     portfolio: List[Trade],
                                     market_data: MarketData,
                                     t: float) -> float:
        """
        Calculate initial margin using simplified ISDA SIMM methodology.
        
        Args:
            portfolio: List of Trade objects
            market_data: MarketData object
            t: Time point
        
        Returns:
            Initial margin amount
        """
        # Simplified SIMM calculation
        # In practice, this would involve full risk sensitivity aggregation
        
        total_delta_risk = 0.0
        total_vega_risk = 0.0
        
        for trade in portfolio:
            if trade.start_date <= t < trade.maturity:
                time_to_maturity = trade.maturity - t
                
                # Delta risk (simplified)
                vol = market_data.volatility.get(trade.currency, 0.15)
                delta_risk = trade.notional * vol * np.sqrt(time_to_maturity)
                total_delta_risk += delta_risk
                
                # Vega risk (simplified)
                vega_risk = trade.notional * vol * time_to_maturity
                total_vega_risk += vega_risk
        
        # Aggregate risks with correlation (simplified)
        # In practice, use full SIMM correlation matrix
        correlation = 0.5  # Simplified correlation
        aggregated_risk = np.sqrt(
            total_delta_risk**2 + 
            total_vega_risk**2 + 
            2 * correlation * total_delta_risk * total_vega_risk
        )
        
        # Apply risk weight
        im = aggregated_risk * market_data.initial_margin_risk_weight
        
        # Add concentration and curvature adjustments (simplified)
        im *= 1.1  # 10% adjustment
        
        return max(im, 0.0)
    
    def calculate(self, portfolio: List[Trade],
                 market_data: MarketData,
                 time_grid: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate MVA.
        
        Returns:
            Dictionary with MVA value and initial margin profile
        """
        if not portfolio:
            return {'MVA': 0.0, 'im_profile': None}
        
        # Create time grid
        max_maturity = max(trade.maturity for trade in portfolio)
        if time_grid is None:
            num_points = int(max_maturity * 252) + 1
            time_grid = np.linspace(0, max_maturity, num_points)
        
        dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 0.01
        
        # Calculate initial margin over time
        im_profile = np.zeros(len(time_grid))
        for i, t in enumerate(time_grid):
            im_profile[i] = self.calculate_initial_margin_simm(
                portfolio, market_data, t
            )
        
        # Calculate MVA
        mva = 0.0
        for i, t in enumerate(time_grid):
            discount_factor = np.exp(-market_data.risk_free_rate * t)
            mva += market_data.funding_spread * im_profile[i] * discount_factor * dt
        
        return {
            'MVA': mva,
            'im_profile': im_profile,
            'time_grid': time_grid
        }


class KVACalculator(XVACalculator):
    """Capital Valuation Adjustment calculator."""
    
    def __init__(self, cva_multiplier: float = 1.25,
                 risk_weight: float = 0.50,
                 cost_of_capital: float = 0.10):
        """
        Initialise KVA calculator.
        
        Args:
            cva_multiplier: CVA multiplier (Basel III: 1.25)
            risk_weight: Risk weight for OTC derivatives (0.50)
            cost_of_capital: Cost of capital (e.g., 10%)
        """
        self.cva_multiplier = cva_multiplier
        self.risk_weight = risk_weight
        self.cost_of_capital = cost_of_capital
    
    def calculate_cva_capital(self,
                              portfolio: List[Trade],
                              market_data: MarketData,
                              time_grid: np.ndarray,
                              exposures: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate CVA capital requirement over time.
        
        Args:
            portfolio: List of Trade objects
            market_data: MarketData object
            time_grid: Time grid
            exposures: Optional exposure array (if None, uses simplified calculation)
        
        Returns:
            Array of capital requirements over time
        """
        capital = np.zeros(len(time_grid))
        
        # Group trades by counterparty
        counterparties = list(set(trade.counterparty for trade in portfolio))
        
        for i, t in enumerate(time_grid):
            # Calculate EAD (Exposure at Default) by counterparty
            ead_by_cp = {}
            
            if exposures is not None:
                # Use simulated exposures
                epe = np.mean(np.maximum(exposures[:, i], 0))
                # Distribute EPE across counterparties (simplified)
                for cp in counterparties:
                    cp_trades = [trade for trade in portfolio if trade.counterparty == cp 
                               and trade.start_date <= t < trade.maturity]
                    total_notional = sum(
                        trade.notional for trade in portfolio if trade.start_date <= t < trade.maturity
                    )
                    cp_weight = (sum(trade.notional for trade in cp_trades) / total_notional 
                               if total_notional > 0 else 0)
                    ead_by_cp[cp] = epe * cp_weight
            else:
                # Simplified EAD calculation
                for trade in portfolio:
                    if trade.start_date <= t < trade.maturity:
                        cp = trade.counterparty
                        if cp not in ead_by_cp:
                            ead_by_cp[cp] = 0.0
                        # Simplified: EAD as percentage of notional
                        ead_by_cp[cp] += abs(trade.notional * 0.1)
            
            # Calculate capital with correlation
            # Basel III formula: K_CVA = alpha * sqrt(sum_i sum_j rho_ij * RW_i * EAD_i * RW_j * EAD_j)
            total_capital_squared = 0.0
            
            for cp1 in counterparties:
                for cp2 in counterparties:
                    ead1 = ead_by_cp.get(cp1, 0.0)
                    ead2 = ead_by_cp.get(cp2, 0.0)
                    
                    # Correlation: 1.0 for same counterparty, 0.3-0.5 for different
                    if cp1 == cp2:
                        correlation = 1.0
                    else:
                        correlation = 0.3
                    
                    contribution = (self.risk_weight * ead1 * 
                                  self.risk_weight * ead2 * correlation)
                    total_capital_squared += contribution
            
            capital[i] = self.cva_multiplier * np.sqrt(max(total_capital_squared, 0))
        
        return capital
    
    def calculate(self, portfolio: List[Trade],
                 market_data: MarketData,
                 time_grid: Optional[np.ndarray] = None,
                 exposures: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate KVA.
        
        Args:
            portfolio: List of Trade objects
            market_data: MarketData object
            time_grid: Optional time grid
            exposures: Optional exposure array from FVA calculation
        
        Returns:
            Dictionary with KVA value and capital profile
        """
        if not portfolio:
            return {'KVA': 0.0, 'capital_profile': None}
        
        # Create time grid
        max_maturity = max(trade.maturity for trade in portfolio)
        if time_grid is None:
            num_points = int(max_maturity * 252) + 1
            time_grid = np.linspace(0, max_maturity, num_points)
        
        dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 0.01
        
        # Calculate capital over time
        capital = self.calculate_cva_capital(
            portfolio, market_data, time_grid, exposures
        )
        
        # Calculate KVA
        kva = 0.0
        for i, t in enumerate(time_grid):
            discount_factor = np.exp(-market_data.risk_free_rate * t)
            kva += self.cost_of_capital * capital[i] * discount_factor * dt
        
        return {
            'KVA': kva,
            'capital_profile': capital,
            'time_grid': time_grid
        }


class COASimulator:
    """Close-Out Amount simulator for MPoR analysis."""
    
    def __init__(self, mpor: float = 5.0 / 252.0, num_paths: int = 10000):
        """
        Initialise COA simulator.
        
        Args:
            mpor: Margin Period of Risk in years
            num_paths: Number of simulation paths
        """
        self.mpor = mpor
        self.num_paths = num_paths
    
    def simulate_coa(self,
                    portfolio: List[Trade],
                    market_data: MarketData,
                    t: float,
                    initial_margin: float,
                    variation_margin: float = 0.0) -> np.ndarray:
        """
        Simulate Close-Out Amount at time t.
        
        Args:
            portfolio: List of Trade objects
            market_data: MarketData object
            t: Current time
            initial_margin: Initial margin posted
            variation_margin: Variation margin at time t
        
        Returns:
            Array of COA values across simulation paths
        """
        # Simulate portfolio value at t + MPoR
        coa_values = np.zeros(self.num_paths)
        
        for path in range(self.num_paths):
            # Simulate portfolio value change over MPoR
            portfolio_value_change = 0.0
            
            for trade in portfolio:
                if trade.start_date <= t < trade.maturity:
                    vol = market_data.volatility.get(trade.currency, 0.15)
                    shock = np.random.normal(0, vol * np.sqrt(self.mpor))
                    
                    if trade.trade_type == 'IRS':
                        value_change = trade.notional * shock * (trade.maturity - t)
                    else:
                        value_change = trade.notional * shock * np.sqrt(trade.maturity - t)
                    
                    portfolio_value_change += value_change
            
            # COA = max(portfolio_value_change - IM - VM, 0)
            coa_values[path] = max(
                portfolio_value_change - initial_margin - variation_margin, 0
            )
        
        return coa_values


class XVAEngine:
    """Main XVA calculation engine."""
    
    def __init__(self,
                 num_paths: int = 10000,
                 num_steps: int = 252,
                 cva_multiplier: float = 1.25,
                 risk_weight: float = 0.50,
                 cost_of_capital: float = 0.10):
        """
        Initialise XVA engine.
        
        Args:
            num_paths: Number of Monte Carlo paths
            num_steps: Number of time steps per year
            cva_multiplier: CVA multiplier for capital calculation
            risk_weight: Risk weight for capital calculation
            cost_of_capital: Cost of capital
        """
        self.fva_calculator = FVACalculator(num_paths, num_steps)
        self.mva_calculator = MVACalculator()
        self.kva_calculator = KVACalculator(
            cva_multiplier, risk_weight, cost_of_capital
        )
        self.coa_simulator = COASimulator()
    
    def calculate_all_xva(self,
                         portfolio: List[Trade],
                         market_data: MarketData) -> Dict:
        """
        Calculate all XVA components.
        
        Args:
            portfolio: List of Trade objects
            market_data: MarketData object
        
        Returns:
            Dictionary containing all XVA results
        """
        if not portfolio:
            return {
                'FVA': 0.0,
                'MVA': 0.0,
                'KVA': 0.0,
                'Total XVA': 0.0
            }
        
        # Create common time grid using configured num_steps
        max_maturity = max(trade.maturity for trade in portfolio)
        num_points = int(max_maturity * self.fva_calculator.num_steps) + 1
        time_grid = np.linspace(0, max_maturity, num_points)
        
        # Calculate FVA (includes exposure simulation)
        fva_results = self.fva_calculator.calculate(portfolio, market_data, time_grid)
        
        # Calculate MVA
        mva_results = self.mva_calculator.calculate(portfolio, market_data, time_grid)
        
        # Calculate KVA (reuse exposures from FVA)
        kva_results = self.kva_calculator.calculate(
            portfolio, market_data, time_grid, fva_results['exposures']
        )
        
        # Aggregate results
        total_xva = (fva_results['FVA'] + 
                    mva_results['MVA'] + 
                    kva_results['KVA'])
        
        return {
            'FVA': fva_results['FVA'],
            'MVA': mva_results['MVA'],
            'KVA': kva_results['KVA'],
            'Total XVA': total_xva,
            'FVA_details': fva_results,
            'MVA_details': mva_results,
            'KVA_details': kva_results,
            'time_grid': time_grid
        }


def generate_synthetic_portfolio(num_trades: int = 50,
                                 total_notional: float = 250e6,
                                 counterparties: Optional[List[str]] = None,
                                 random_seed: Optional[int] = 42) -> List[Trade]:
    """
    Generate synthetic derivative portfolio for testing.
    
    Args:
        num_trades: Number of trades to generate
        total_notional: Total notional amount
        counterparties: List of counterparty names (default: CP1-CP8)
        random_seed: Random seed for reproducibility
    
    Returns:
        List of Trade objects
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if counterparties is None:
        counterparties = [f'CP{i+1}' for i in range(8)]
    
    trades = []
    currencies = ['USD', 'EUR', 'GBP']
    trade_types = ['IRS', 'CCS', 'FX']
    
    # Distribute notional across trades
    notional_per_trade = total_notional / num_trades
    
    for i in range(num_trades):
        # Random trade parameters
        notional = notional_per_trade * np.random.uniform(0.5, 1.5)
        maturity = np.random.uniform(1.0, 30.0)  # 1-30 years
        currency = np.random.choice(currencies)
        trade_type = np.random.choice(trade_types)
        counterparty = np.random.choice(counterparties)
        
        # Fixed rate for IRS
        fixed_rate = None
        if trade_type == 'IRS':
            fixed_rate = np.random.uniform(0.01, 0.05)  # 1-5%
        
        trade = Trade(
            trade_id=f'Trade_{i+1:04d}',
            counterparty=counterparty,
            notional=notional,
            maturity=maturity,
            currency=currency,
            trade_type=trade_type,
            fixed_rate=fixed_rate,
            start_date=0.0
        )
        trades.append(trade)
    
    return trades

