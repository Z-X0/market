# analysis/covered_call.py
"""
Covered Call Analysis Module.

This module provides tools for analyzing and optimizing covered call strategies,
including strike selection, premium calculation, and risk-adjusted metrics.
"""

import logging
import math
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis.volatility import calculate_vol_forecast

logger = logging.getLogger(__name__)

# Constants
RISK_FREE_RATE = 0.02  # 2% risk-free rate


class CoveredCallStrategy:
    """
    Covered call strategy implementation.
    
    This class implements the analysis and optimization of covered call strategies,
    including strike selection, premium calculation, and risk metrics.
    """
    
    def __init__(
        self,
        symbol: str,
        shares: int,
        risk_level: str = 'conservative',
        slippage: float = 0.01,
        commission: float = 0.65,
        market_regime: Optional[int] = None,
        vol_forecast: Optional[float] = None,
        stock_data: Optional[Dict[str, pd.DataFrame]] = None,
        option_chains: Optional[Dict[int, List[Dict[str, Any]]]] = None
    ):
        """
        Initialize the strategy.
        
        Parameters:
        symbol (str): Stock symbol
        shares (int): Number of shares held
        risk_level (str): Risk level for strategy ('conservative', 'moderate', 'aggressive')
        slippage (float): Estimated slippage per trade
        commission (float): Commission per contract
        market_regime (int, optional): Market regime override
        vol_forecast (float, optional): Volatility forecast override
        stock_data (dict, optional): Pre-fetched stock data
        option_chains (dict, optional): Pre-fetched option chains
        """
        self.symbol = symbol
        self.shares = shares
        self.risk_level = risk_level
        self.slippage = slippage
        self.commission = commission
        
        # Calculate the number of contracts (each contract is 100 shares)
        self.contracts = shares // 100
        
        # Store pre-fetched data if provided
        self.stock_data = stock_data or {}
        self.option_chains = option_chains or {}
        
        # Set market regime and volatility if provided
        self.market_regime = market_regime
        self.vol_forecast = vol_forecast
        
        # Initialize attributes to be calculated later
        self.weekly_data = self.stock_data.get('weekly_data', pd.DataFrame())
        self.daily_data = self.stock_data.get('daily_data', pd.DataFrame())
        self.current_price = 0
        self.market_analyzer = None
        self.volatility_analysis = None
        
        # Load data and initialize analysis
        self._load_data()
        self._initialize_analysis()
    
    def _load_data(self) -> None:
        """Load required data if not already loaded."""
        if self.weekly_data.empty or self.daily_data.empty:
            try:
                # Import fetcher within method to avoid circular imports
                from data.fetcher import get_stock_history_weekly, get_stock_history_daily
                
                if self.weekly_data.empty:
                    self.weekly_data = get_stock_history_weekly(self.symbol)
                
                if self.daily_data.empty:
                    self.daily_data = get_stock_history_daily(self.symbol)
            
            except Exception as e:
                logger.error(f"Error loading data for {self.symbol}: {e}")
        
        # Set current price
        if not self.weekly_data.empty:
            self.current_price = self.weekly_data['Close'].iloc[-1]
        elif not self.daily_data.empty:
            self.current_price = self.daily_data['Close'].iloc[-1]
        else:
            logger.warning(f"No price data for {self.symbol}, using zero")
            self.current_price = 0
    
    def _initialize_analysis(self) -> None:
        """Initialize market regime and volatility analysis."""
        # Market regime analysis
        if self.market_regime is None and not self.weekly_data.empty:
            try:
                from analysis.market_regime import MarketRegimeAnalyzer
                self.market_analyzer = MarketRegimeAnalyzer(self.weekly_data)
                self.market_regime = self.market_analyzer.get_current_regime()
            except Exception as e:
                logger.error(f"Error initializing market regime analysis: {e}")
                self.market_regime = 3  # Default to neutral
        
        # Volatility forecast
        if self.vol_forecast is None and not self.daily_data.empty:
            try:
                returns = self.daily_data['Close'].pct_change().dropna()
                self.volatility_analysis = calculate_vol_forecast(returns, self.market_regime)
                self.vol_forecast = self.volatility_analysis['forecast']
            except Exception as e:
                logger.error(f"Error calculating volatility forecast: {e}")
                self.vol_forecast = 0.2  # Default to 20%
    
    def get_recommendations(self) -> Dict[str, Any]:
        """
        Generate covered call recommendations.
        
        Returns:
        dict: Strategy recommendations
        """
        # Check if we have valid data
        if self.current_price <= 0:
            return {'error': 'No valid price data'}
        
        # Delta targets based on risk level
        delta_targets = {
            'conservative': [0.20, 0.25, 0.30],
            'moderate': [0.25, 0.30, 0.35],
            'aggressive': [0.30, 0.35, 0.40]
        }
        
        # DTE targets based on risk level
        dte_targets = {
            'conservative': [30, 45],
            'moderate': [30, 45, 60],
            'aggressive': [45, 60, 90]
        }
        
        # Get targets for current risk level
        deltas = delta_targets.get(self.risk_level, delta_targets['moderate'])
        dtes = dte_targets.get(self.risk_level, dte_targets['moderate'])
        
        # Create optimizer
        optimizer = CoveredCallOptimizer(
            self.symbol, 
            self.shares, 
            self.current_price, 
            self.vol_forecast, 
            self.market_regime,
            option_chains=self.option_chains
        )
        
        # Run optimization
        results = optimizer.optimize_parameter_grid(dtes=dtes, deltas=deltas)
        
        # Get optimal strikes for each DTE
        optimal_strikes = {}
        for dte in dtes:
            best_delta = None
            best_score = -float('inf')
            
            for delta in deltas:
                if dte in results['parameter_grid'] and delta in results['parameter_grid'][dte]:
                    score = results['parameter_grid'][dte][delta].get('risk_adjusted_score', -float('inf'))
                    if score > best_score:
                        best_score = score
                        best_delta = delta
            
            if best_delta is not None:
                params = results['parameter_grid'][dte][best_delta]
                
                # Calculate probabilities if not provided
                if 'prob_itm' not in params:
                    t = dte / 365
                    
                    d1 = (np.log(self.current_price/params['strike']) + 
                          (RISK_FREE_RATE + self.vol_forecast**2/2) * t) / (self.vol_forecast * np.sqrt(t))
                    
                    prob_itm = norm.cdf(d1)
                    prob_otm = 1 - prob_itm
                    prob_touch = np.exp((-2 * np.log(self.current_price/params['strike']) * 
                                        np.log(self.current_price/params['strike'])) / 
                                       (self.vol_forecast**2 * t))
                else:
                    prob_itm = params.get('prob_itm', 0.5)
                    prob_otm = params.get('prob_otm', 0.5)
                    prob_touch = params.get('prob_touch', 0.5)
                
                # Add to optimal strikes
                optimal_strikes[dte] = {
                    'strike': params['strike'],
                    'call_price': params['premium'],
                    'delta': params.get('delta', best_delta),
                    'annualized_return': params['annualized_return'] * 100,  # Convert to percentage
                    'upside_potential': params['upside_cap_pct'] * 100,  # Convert to percentage
                    'greeks': params.get('greeks', {
                        'delta': best_delta,
                        'gamma': 0,
                        'theta': 0,
                        'vega': 0
                    }),
                    'probabilities': {
                        'prob_itm': prob_itm,
                        'prob_otm': prob_otm,
                        'prob_touch': prob_touch
                    }
                }
        
        # Calculate position risk metrics
        var_99 = self.current_price * self.vol_forecast * abs(norm.ppf(0.01)) / np.sqrt(52)
        es_99 = self.current_price * self.vol_forecast * norm.pdf(norm.ppf(0.01)) / 0.01 / np.sqrt(52)
        
        position_value = self.shares * self.current_price
        
        # Compile final recommendations
        recommendations = {
            'symbol': self.symbol,
            'shares': self.shares,
            'contracts': self.contracts,
            'current_price': self.current_price,
            'position_value': position_value,
            'risk_level': self.risk_level,
            'market_regime': {
                'current': self.market_regime,
                'name': self._get_regime_name(self.market_regime),
                'transition_probs': self.market_analyzer.get_regime_transitions().tolist() if self.market_analyzer else None,
                'stats': self.market_analyzer.get_regime_stats() if self.market_analyzer else {}
            },
            'volatility': self.volatility_analysis or {
                'forecast': self.vol_forecast,
                'model_breakdown': {
                    'historical': 0,
                    'ewma': 0,
                    'garch': 0,
                    'har': 0,
                    'ensemble': self.vol_forecast
                },
                'term_structure': {
                    'short': self.vol_forecast * 0.9,
                    'medium': self.vol_forecast,
                    'long': self.vol_forecast * 1.1
                }
            },
            'optimal_strikes': optimal_strikes,
            'risk_metrics': {
                'position_value': position_value,
                'value_at_risk': position_value * (var_99 / self.current_price),
                'expected_shortfall': position_value * (es_99 / self.current_price),
                'max_loss': position_value  # Theoretical max loss is the entire position value
            },
            'optimization_results': results
        }
        
        return recommendations
    
    def _get_regime_name(self, regime: int) -> str:
        """Get the name of a regime."""
        regime_names = {
            0: "Severe Bearish",
            1: "Bearish",
            2: "Weak Bearish",
            3: "Neutral",
            4: "Weak Bullish",
            5: "Bullish",
            6: "Strong Bullish"
        }
        
        return regime_names.get(regime, f"Regime {regime}")


class CoveredCallOptimizer:
    """
    Optimizer for covered call parameters.
    
    This class implements optimization algorithms for covered call parameters,
    including strike selection and expiration.
    """
    
    def __init__(
        self,
        symbol: str,
        position_size: int,
        current_price: float,
        vol_forecast: float,
        market_regime: int,
        option_chains: Optional[Dict[int, List[Dict[str, Any]]]] = None
    ):
        """
        Initialize the optimizer.
        
        Parameters:
        symbol (str): Symbol
        position_size (int): Number of shares
        current_price (float): Current price
        vol_forecast (float): Volatility forecast
        market_regime (int): Current market regime
        option_chains (dict, optional): Pre-fetched option chains
        """
        self.symbol = symbol
        self.position_size = position_size
        self.current_price = current_price
        self.vol_forecast = vol_forecast
        self.market_regime = market_regime
        self.position_value = current_price * position_size
        self.contracts = position_size // 100
        self.option_chains = option_chains or {}
    
    def optimize_parameter_grid(
        self, 
        dtes: List[int] = [30, 45, 60, 90], 
        deltas: List[float] = [0.20, 0.25, 0.30, 0.35, 0.40]
    ) -> Dict[str, Any]:
        """
        Generate a parameter grid for optimization.
        
        Parameters:
        dtes (list): Days to expiry to test
        deltas (list): Target deltas to test
        
        Returns:
        dict: Optimized results for each parameter combination
        """
        results = {}
        
        for dte in dtes:
            results[dte] = {}
            t = dte / 365  # Years
            
            # Check if we have option chain data for this DTE
            chain = self.option_chains.get(dte, [])
            
            for delta_target in deltas:
                # Try to find option in chain first if available
                option_found = False
                
                if chain:
                    # Find closest option to target delta
                    closest_option = None
                    min_delta_diff = float('inf')
                    
                    for option in chain:
                        # Skip if not a call or missing key data
                        if option.get('option_type') != 'call' or 'strike' not in option or 'mid' not in option:
                            continue
                            
                        # Calculate delta if not provided
                        if 'delta' not in option:
                            strike = option['strike']
                            
                            d1 = (np.log(self.current_price/strike) + 
                                  (RISK_FREE_RATE + self.vol_forecast**2/2) * t) / (self.vol_forecast * np.sqrt(t))
                            
                            option['delta'] = norm.cdf(d1)
                        
                        # Check if this is closer to target delta
                        delta_diff = abs(option['delta'] - delta_target)
                        if delta_diff < min_delta_diff:
                            min_delta_diff = delta_diff
                            closest_option = option
                    
                    # If found a suitable option, use it
                    if closest_option and min_delta_diff < 0.1:  # Within reasonable range
                        strike = closest_option['strike']
                        premium = closest_option['mid']
                        actual_delta = closest_option['delta']
                        
                        # Calculate other metrics
                        upside_cap = strike - self.current_price
                        upside_cap_pct = upside_cap / self.current_price
                        
                        premium_yield = premium / self.current_price
                        annualized_return = premium_yield * (365 / dte)
                        
                        # Get greeks if available
                        greeks = {
                            'delta': actual_delta,
                            'gamma': closest_option.get('gamma', 0),
                            'theta': closest_option.get('theta', 0),
                            'vega': closest_option.get('vega', 0)
                        }
                        
                        # Get probabilities if available
                        prob_itm = closest_option.get('prob_itm', None)
                        prob_otm = closest_option.get('prob_otm', None)
                        prob_touch = closest_option.get('prob_touch', None)
                        
                        # Calculate if not available
                        if prob_itm is None:
                            prob_itm = actual_delta  # Delta approximates ITM probability
                            prob_otm = 1 - prob_itm
                        
                        if prob_touch is None:
                            prob_touch = np.exp((-2 * np.log(self.current_price/strike) * 
                                                np.log(self.current_price/strike)) / 
                                               (self.vol_forecast**2 * t))
                        
                        # Calculate regime adjustments
                        regime_risk_adjustments = {
                            0: -0.5,  # Severe Bearish - heavily penalize upside capping
                            1: -0.4,  # Bearish
                            2: -0.2,  # Weak Bearish
                            3: 0.0,   # Neutral - no adjustment
                            4: 0.2,   # Weak Bullish
                            5: 0.4,   # Bullish
                            6: 0.5    # Strong Bullish - reward upside potential
                        }
                        
                        regime_adj = regime_risk_adjustments.get(self.market_regime, 0.0)
                        
                        # Calculate risk-adjusted score
                        risk_adjusted_score = (
                            annualized_return * 100 +  # Annualized return as percentage
                            upside_cap_pct * 50 * (1 + regime_adj) +  # Upside cap weighted by regime
                            (1 - prob_itm) * 20 +  # Lower assignment probability is better
                            abs(greeks['theta'] / premium) * 30  # Higher theta decay rate is better
                        )
                        
                        # Store results
                        results[dte][delta_target] = {
                            'strike': strike,
                            'premium': premium,
                            'delta': actual_delta,
                            'upside_cap': upside_cap,
                            'upside_cap_pct': upside_cap_pct,
                            'premium_yield': premium_yield,
                            'annualized_return': annualized_return,
                            'prob_itm': prob_itm,
                            'prob_otm': prob_otm,
                            'prob_touch': prob_touch,
                            'greeks': greeks,
                            'risk_adjusted_score': risk_adjusted_score,
                            'from_option_chain': True
                        }
                        
                        option_found = True
                
                # If no suitable option found in chain, calculate theoretical values
                if not option_found:
                    # Approximate strike price for target delta
                    # Using Black-Scholes to approximate: For call options, delta = N(d1)
                    d1_for_delta = norm.ppf(delta_target)
                    log_moneyness = d1_for_delta * self.vol_forecast * math.sqrt(t) - (RISK_FREE_RATE + self.vol_forecast**2/2) * t
                    strike = self.current_price * math.exp(-log_moneyness)
                    
                    # Calculate option premium using Black-Scholes
                    d1 = (math.log(self.current_price/strike) + (RISK_FREE_RATE + self.vol_forecast**2/2) * t) / (self.vol_forecast * math.sqrt(t))
                    d2 = d1 - self.vol_forecast * math.sqrt(t)
                    
                    premium = self.current_price * norm.cdf(d1) - strike * math.exp(-RISK_FREE_RATE * t) * norm.cdf(d2)
                    
                    # Calculate option greeks
                    delta = norm.cdf(d1)
                    gamma = norm.pdf(d1) / (self.current_price * self.vol_forecast * math.sqrt(t))
                    theta = -(self.current_price * norm.pdf(d1) * self.vol_forecast) / (2 * math.sqrt(t)) - RISK_FREE_RATE * strike * math.exp(-RISK_FREE_RATE * t) * norm.cdf(d2)
                    theta = theta / 365  # Convert to daily theta
                    vega = self.current_price * math.sqrt(t) * norm.pdf(d1) / 100  # For 1% vol change
                    
                    # Calculate covered call metrics
                    upside_cap = strike - self.current_price
                    upside_cap_pct = upside_cap / self.current_price
                    
                    premium_yield = premium / self.current_price
                    annualized_return = premium_yield * (365 / dte)
                    
                    # Calculate probability metrics
                    prob_itm = delta
                    prob_otm = 1 - prob_itm
                    prob_touch = 2 * (1 - norm.cdf((math.log(strike/self.current_price) - (RISK_FREE_RATE - self.vol_forecast**2/2) * t) / (self.vol_forecast * math.sqrt(t))))
                    
                    # Calculate risk-adjusted metrics based on regime
                    # Adjust for different market regimes
                    regime_risk_adjustments = {
                        0: -0.5,  # Severe Bearish - heavily penalize upside capping
                        1: -0.4,  # Bearish
                        2: -0.2,  # Weak Bearish
                        3: 0.0,   # Neutral - no adjustment
                        4: 0.2,   # Weak Bullish
                        5: 0.4,   # Bullish
                        6: 0.5    # Strong Bullish - reward upside potential
                    }
                    
                    regime_adj = regime_risk_adjustments.get(self.market_regime, 0.0)
                    
                    # Calculate risk-adjusted score
                    risk_adjusted_score = (
                        annualized_return * 100 +  # Annualized return as percentage
                        upside_cap_pct * 50 * (1 + regime_adj) +  # Upside cap weighted by regime
                        (1 - prob_itm) * 20 +  # Lower assignment probability is better
                        abs(theta / premium) * 30  # Higher theta decay rate is better
                    )
                    
                    # Store results
                    results[dte][delta_target] = {
                        'strike': strike,
                        'premium': premium,
                        'upside_cap': upside_cap,
                        'upside_cap_pct': upside_cap_pct,
                        'premium_yield': premium_yield,
                        'annualized_return': annualized_return,
                        'prob_itm': prob_itm,
                        'prob_otm': prob_otm,
                        'prob_touch': prob_touch,
                        'greeks': {
                            'delta': delta,
                            'gamma': gamma,
                            'theta': theta,
                            'vega': vega
                        },
                        'risk_adjusted_score': risk_adjusted_score,
                        'from_option_chain': False
                    }
        
        # Find optimal parameters
        best_params = None
        best_score = -float('inf')
        
        for dte in results:
            for delta in results[dte]:
                score = results[dte][delta]['risk_adjusted_score']
                if score > best_score:
                    best_score = score
                    best_params = (dte, delta)
        
        if best_params:
            best_dte, best_delta = best_params
            optimal_result = results[best_dte][best_delta].copy()
            optimal_result['optimal_dte'] = best_dte
            optimal_result['optimal_delta'] = best_delta
        else:
            optimal_result = None
        
        return {
            'parameter_grid': results,
            'optimal_parameters': optimal_result
        }
    
    def calculate_expected_outcomes(
        self, 
        strike: float, 
        premium: float, 
        dte: int, 
        num_scenarios: int = 1000
    ) -> Dict[str, Any]:
        """
        Calculate expected outcomes using Monte Carlo simulation.
        
        Parameters:
        strike (float): Strike price
        premium (float): Option premium
        dte (int): Days to expiry
        num_scenarios (int): Number of scenarios to simulate
        
        Returns:
        dict: Expected outcome metrics
        """
        # Convert DTE to years
        t = dte / 365
        
        # Generate price scenarios at expiration
        # Using log-normal price distribution
        annual_return = 0.08  # Expected annual return
        drift = annual_return / 365 * dte  # Expected drift over period
        
        # Generate random price paths
        np.random.seed(42)  # For reproducibility
        z = np.random.normal(0, 1, num_scenarios)
        price_scenarios = self.current_price * np.exp((drift - 0.5 * self.vol_forecast**2) * t + self.vol_forecast * np.sqrt(t) * z)
        
        # Calculate covered call payoffs
        stock_return = price_scenarios / self.current_price - 1
        
        # Covered call payoff is limited by strike
        covered_call_values = np.minimum(price_scenarios, strike) + premium
        covered_call_return = covered_call_values / self.current_price - 1
        
        # Calculate various metrics
        mean_stock_return = np.mean(stock_return)
        mean_cc_return = np.mean(covered_call_return)
        
        std_stock_return = np.std(stock_return)
        std_cc_return = np.std(covered_call_return)
        
        # Calculate probability of outperformance
        outperform_prob = np.mean(covered_call_return > stock_return)
        
        # Calculate probability of max profit (stock price >= strike)
        max_profit_prob = np.mean(price_scenarios >= strike)
        
        # Calculate VaR for both strategies
        stock_var_95 = np.percentile(stock_return, 5)
        cc_var_95 = np.percentile(covered_call_return, 5)
        
        # Calculate maximum possible loss
        max_loss_stock = np.min(stock_return)
        max_loss_cc = np.min(covered_call_return)
        
        # Return comprehensive statistics
        return {
            'scenarios': {
                'price_scenarios': price_scenarios.tolist(),
                'stock_returns': stock_return.tolist(),
                'cc_returns': covered_call_return.tolist()
            },
            'summary_metrics': {
                'mean_stock_return': mean_stock_return,
                'mean_cc_return': mean_cc_return,
                'std_stock_return': std_stock_return,
                'std_cc_return': std_cc_return,
                'sharpe_stock': (mean_stock_return - RISK_FREE_RATE * t) / std_stock_return if std_stock_return > 0 else 0,
                'sharpe_cc': (mean_cc_return - RISK_FREE_RATE * t) / std_cc_return if std_cc_return > 0 else 0
            },
            'probability_metrics': {
                'outperform_prob': outperform_prob,
                'max_profit_prob': max_profit_prob,
                'assignment_prob': max_profit_prob
            },
            'risk_metrics': {
                'stock_var_95': stock_var_95,
                'cc_var_95': cc_var_95,
                'max_loss_stock': max_loss_stock,
                'max_loss_cc': max_loss_cc,
                'var_reduction': (stock_var_95 - cc_var_95) / abs(stock_var_95) if stock_var_95 < 0 else 0
            }
        }