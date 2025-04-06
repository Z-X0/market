# analysis/portfolio.py
"""
Portfolio Analysis Module.

This module provides tools for analyzing portfolio-level metrics, correlations,
and optimizing portfolio covered call strategies.
"""

import logging
import math
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis.covered_call import CoveredCallOptimizer

logger = logging.getLogger(__name__)

# Constants
RISK_FREE_RATE = 0.02  # 2% risk-free rate


def build_portfolio_correlation(symbols_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    Build correlation matrices for portfolio.
    
    Parameters:
    symbols_data (dict): Data for each symbol
    
    Returns:
    dict: Correlation matrices
    """
    correlations = {}
    
    # Build price correlation matrix
    price_df = pd.DataFrame()
    
    for symbol, data in symbols_data.items():
        if 'weekly_data' in data and not data['weekly_data'].empty:
            price_df[symbol] = data['weekly_data']['Close']
    
    if not price_df.empty and len(price_df.columns) > 1:
        # Calculate return correlations
        ret_df = price_df.pct_change().dropna()
        correlations['price'] = ret_df.corr()
    
    # Build volatility correlation matrix
    vol_df = pd.DataFrame()
    
    for symbol, data in symbols_data.items():
        if 'weekly_data' in data and not data['weekly_data'].empty:
            returns = data['weekly_data']['Close'].pct_change().dropna()
            vol_df[symbol] = returns.rolling(12).std()
    
    if not vol_df.empty and len(vol_df.columns) > 1:
        # Calculate volatility correlations
        vol_df = vol_df.dropna()
        correlations['volatility'] = vol_df.corr()
    
    return correlations


def calculate_portfolio_volatility(
    weights: Dict[str, float], 
    volatilities: Dict[str, float], 
    correlation_matrix: pd.DataFrame
) -> float:
    """
    Calculate portfolio volatility using correlation matrix.
    
    Parameters:
    weights (dict): Portfolio weights by symbol
    volatilities (dict): Volatility forecasts by symbol
    correlation_matrix (pandas.DataFrame): Correlation matrix
    
    Returns:
    float: Portfolio volatility
    """
    if correlation_matrix.empty or not weights:
        # Simple weighted average if no correlation data
        return sum(weights[s] * volatilities[s] for s in weights if s in volatilities)
    
    # Create lists of symbols, weights, and vols in the same order
    symbols = []
    w_list = []
    v_list = []
    
    for symbol in correlation_matrix.columns:
        if symbol in weights and symbol in volatilities:
            symbols.append(symbol)
            w_list.append(weights[symbol])
            v_list.append(volatilities[symbol])
    
    if not symbols:
        return 0
    
    # Create weight and volatility vectors
    w_vec = np.array(w_list)
    v_vec = np.array(v_list)
    
    # Create correlation matrix for these symbols
    corr_mat = correlation_matrix.loc[symbols, symbols].values
    
    # Create covariance matrix
    cov_mat = np.outer(v_vec, v_vec) * corr_mat
    
    # Calculate portfolio variance
    port_var = w_vec.T @ cov_mat @ w_vec
    
    # Return portfolio volatility
    return math.sqrt(max(0, port_var))


def generate_market_analysis(
    all_results: Dict[str, Any], 
    current_prices: Dict[str, float]
) -> Dict[str, Any]:
    """
    Generate market analysis for implementation plan.
    
    Parameters:
    all_results (dict): Analysis results
    current_prices (dict): Current prices
    
    Returns:
    dict: Market analysis
    """
    from analysis.market_regime import MarketRegimeAnalyzer
    from optimization.strategy_optimizer import ImplementationPlan
    
    # Use conservative results
    cons_data = all_results.get('conservative', {})
    pos_data = cons_data.get('position_results', {})
    
    # Extract key metrics
    market_regime = None
    vol_forecast = 0.20  # Default
    portfolio_value = 0
    
    # Find a valid symbol to extract regime
    for symbol in pos_data:
        if 'market_regime' in pos_data[symbol]:
            market_regime = pos_data[symbol]['market_regime'].get('current')
            break
    
    # Calculate average volatility forecast
    vol_forecasts = []
    
    for symbol in pos_data:
        if 'volatility' in pos_data[symbol] and 'forecast' in pos_data[symbol]['volatility']:
            vol_forecasts.append(pos_data[symbol]['volatility']['forecast'])
    
    if vol_forecasts:
        vol_forecast = sum(vol_forecasts) / len(vol_forecasts)
    
    # Create implementation plan
    implementation_plan = {}
    
    # Create execution timing strategy
    if market_regime is not None:
        execution_planner = ImplementationPlan({}, {}, current_prices)
        implementation_plan['execution_timing'] = execution_planner.execution_timing_strategy(market_regime)
    
    # Create risk management guidelines
    if market_regime is not None:
        risk_manager = ImplementationPlan({}, {}, current_prices)
        implementation_plan['risk_management'] = risk_manager.risk_management_guidelines(market_regime, vol_forecast)
    
    return {
        'market_regime': market_regime,
        'volatility_forecast': vol_forecast,
        'implementation_plan': implementation_plan
    }


class OptionsPortfolioBuilder:
    """
    Portfolio builder for options strategies.
    
    This class helps optimize and build a portfolio of options strategies,
    particularly focused on covered calls.
    """
    
    def __init__(
        self, 
        portfolio: Dict[str, int], 
        prices: Dict[str, float], 
        vol_forecasts: Dict[str, float], 
        market_regimes: Dict[str, int]
    ):
        """
        Initialize the portfolio builder.
        
        Parameters:
        portfolio (dict): Portfolio positions {symbol: shares}
        prices (dict): Current prices {symbol: price}
        vol_forecasts (dict): Volatility forecasts {symbol: forecast}
        market_regimes (dict): Market regimes {symbol: regime}
        """
        self.portfolio = portfolio
        self.prices = prices
        self.vol_forecasts = vol_forecasts
        self.market_regimes = market_regimes
        
        # Calculate portfolio value
        self.position_values = {
            symbol: qty * prices.get(symbol, 0) 
            for symbol, qty in portfolio.items()
        }
        self.total_value = sum(self.position_values.values())
        
        # Calculate position weights
        self.weights = {
            symbol: value / self.total_value 
            for symbol, value in self.position_values.items()
        } if self.total_value > 0 else {}
    
    def optimize_all_positions(self) -> Dict[str, Any]:
        """
        Optimize covered call strategies for all positions.
        
        Returns:
        dict: Optimized strategies for each position
        """
        position_strategies = {}
        
        for symbol, shares in self.portfolio.items():
            if shares < 100:  # Skip if less than 1 contract
                continue
                
            price = self.prices.get(symbol, 0)
            vol = self.vol_forecasts.get(symbol, 0.2)  # Default to 20% vol
            regime = self.market_regimes.get(symbol, 3)  # Default to neutral regime
            
            if price <= 0:
                continue
                
            # Create optimizer
            optimizer = CoveredCallOptimizer(symbol, shares, price, vol, regime)
            
            # Run full optimization
            try:
                strategy = optimizer.optimize_parameter_grid()
                position_strategies[symbol] = strategy
            except Exception as e:
                logger.error(f"Error optimizing {symbol}: {e}")
                position_strategies[symbol] = {'error': str(e)}
        
        return position_strategies
    
    def calculate_portfolio_metrics(self, position_strategies: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate portfolio-level metrics for covered call strategy.
        
        Parameters:
        position_strategies (dict): Optimized strategies by symbol
        
        Returns:
        dict: Portfolio-level metrics
        """
        if not position_strategies:
            return {'error': 'No position strategies provided'}
        
        # Calculate aggregate premiums and yields
        total_premium = 0
        premium_weighted_delta = 0
        total_premium_value = 0
        
        for symbol, strategy in position_strategies.items():
            if 'error' in strategy:
                continue
                
            # Get optimal parameters
            optimal = strategy.get('optimal_parameters', {})
            if not optimal:
                continue
                
            premium = optimal.get('premium', 0)
            delta = optimal.get('delta', 0)
            total_premium_position = premium * (self.portfolio[symbol] // 100) * 100
            
            total_premium += total_premium_position
            premium_weighted_delta += delta * total_premium_position
            total_premium_value += total_premium_position
        
        # Calculate weighted average delta
        avg_delta = premium_weighted_delta / total_premium_value if total_premium_value > 0 else 0
        
        # Calculate yield metrics
        portfolio_yield = total_premium / self.total_value if self.total_value > 0 else 0
        
        # Average DTE and annualized yield
        weighted_dte = 0
        weighted_annual_yield = 0
        
        for symbol, strategy in position_strategies.items():
            if 'error' in strategy:
                continue
                
            optimal = strategy.get('optimal_parameters', {})
            if not optimal:
                continue
                
            dte = optimal.get('optimal_dte', 30)
            annual_yield = optimal.get('annualized_return', 0)
            position_value = self.position_values.get(symbol, 0)
            weight = position_value / self.total_value if self.total_value > 0 else 0
            
            weighted_dte += dte * weight
            weighted_annual_yield += annual_yield * weight
        
        # Calculate exposure metrics
        covered_value = sum(
            self.position_values.get(symbol, 0)
            for symbol in position_strategies
            if 'error' not in position_strategies[symbol]
        )
        
        coverage_ratio = covered_value / self.total_value if self.total_value > 0 else 0
        
        # Calculate risk metrics
        total_upside_cap = 0
        for symbol, strategy in position_strategies.items():
            if 'error' in strategy:
                continue
                
            optimal = strategy.get('optimal_parameters', {})
            if not optimal:
                continue
                
            upside_cap = optimal.get('upside_cap_pct', 0)
            position_value = self.position_values.get(symbol, 0)
            
            total_upside_cap += upside_cap * position_value
        
        upside_cap_ratio = total_upside_cap / self.total_value if self.total_value > 0 else 0
        
        # Return portfolio metrics
        return {
            'total_portfolio_value': self.total_value,
            'covered_call_metrics': {
                'total_premium': total_premium,
                'portfolio_yield': portfolio_yield * 100,  # Convert to percentage
                'weighted_average_dte': weighted_dte,
                'weighted_annual_yield': weighted_annual_yield * 100,  # Convert to percentage
                'weighted_average_delta': avg_delta,
                'coverage_ratio': coverage_ratio * 100,  # Convert to percentage
                'upside_cap_ratio': upside_cap_ratio * 100  # Convert to percentage
            },
            'portfolio_allocation': self.weights,
            'position_values': self.position_values
        }
    
    def generate_portfolio_recommendations(self) -> Dict[str, Any]:
        """
        Generate portfolio-level recommendations.
        
        Returns:
        dict: Portfolio recommendations
        """
        # Optimize individual positions
        position_strategies = self.optimize_all_positions()
        
        # Calculate portfolio metrics
        portfolio_metrics = self.calculate_portfolio_metrics(position_strategies)
        
        # Determine overall portfolio allocation strategy
        allocation_strategy = self._determine_allocation_strategy(position_strategies)
        
        # Generate execution calendar
        execution_calendar = self._generate_execution_calendar(position_strategies)
        
        # Return comprehensive portfolio recommendations
        return {
            'portfolio_metrics': portfolio_metrics,
            'position_strategies': position_strategies,
            'allocation_strategy': allocation_strategy,
            'execution_calendar': execution_calendar
        }
    
    def _determine_allocation_strategy(self, position_strategies: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine optimal allocation strategy for covered calls.
        
        Parameters:
        position_strategies (dict): Optimized strategies by symbol
        
        Returns:
        dict: Allocation strategy
        """
        # Count regimes in portfolio
        regime_counts = {}
        for symbol, strategy in position_strategies.items():
            if 'error' in strategy:
                continue
                
            # Get regime from parameter grid results
            if 'parameter_grid' in strategy:
                for dte in strategy['parameter_grid']:
                    for delta in strategy['parameter_grid'][dte]:
                        if 'regime' in strategy['parameter_grid'][dte][delta]:
                            regime = strategy['parameter_grid'][dte][delta]['regime']
                            regime_counts[regime] = regime_counts.get(regime, 0) + 1
                            break
                    if regime_counts:
                        break
        
        # If no regimes found, use market regimes directly
        if not regime_counts:
            for symbol, regime in self.market_regimes.items():
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Determine dominant regime
        dominant_regime = max(regime_counts.items(), key=lambda x: x[1])[0] if regime_counts else 3
        
        # Define allocation strategies based on regime
        allocation_strategies = {
            0: {  # Severe Bearish
                'max_portfolio_coverage': 0.5,  # Max 50% of portfolio with covered calls
                'max_position_coverage': 0.5,   # Max 50% of any position
                'focus': 'Capital preservation and downside protection'
            },
            1: {  # Bearish
                'max_portfolio_coverage': 0.6,
                'max_position_coverage': 0.6,
                'focus': 'Defensive income generation'
            },
            2: {  # Weak Bearish
                'max_portfolio_coverage': 0.7,
                'max_position_coverage': 0.7,
                'focus': 'Income generation with moderate protection'
            },
            3: {  # Neutral
                'max_portfolio_coverage': 0.8,
                'max_position_coverage': 0.8, 
                'focus': 'Balanced income and upside potential'
            },
            4: {  # Weak Bullish
                'max_portfolio_coverage': 0.7,
                'max_position_coverage': 0.7,
                'focus': 'Growth with supplemental income'
            },
            5: {  # Bullish
                'max_portfolio_coverage': 0.6,
                'max_position_coverage': 0.6,
                'focus': 'Prioritize upside potential with selective income'
            },
            6: {  # Strong Bullish
                'max_portfolio_coverage': 0.5,
                'max_position_coverage': 0.5,
                'focus': 'Maximum upside participation with minimal call writing'
            }
        }
        
        # Get allocation strategy for dominant regime
        strategy = allocation_strategies.get(dominant_regime, allocation_strategies[3])
        
        # Calculate total covered value based on recommended coverage
        total_portfolio_value = self.total_value
        recommended_coverage = total_portfolio_value * strategy['max_portfolio_coverage']
        
        # Calculate recommended allocation for each position
        position_allocations = {}
        for symbol, pos_value in self.position_values.items():
            if symbol not in position_strategies or 'error' in position_strategies[symbol]:
                position_allocations[symbol] = 0
                continue
                
            # Recommended coverage for this position
            recommended_position_coverage = min(
                pos_value * strategy['max_position_coverage'],
                pos_value
            )
            
            position_allocations[symbol] = recommended_position_coverage
        
        # Calculate target number of contracts for each position
        target_contracts = {}
        for symbol, allocation in position_allocations.items():
            if allocation <= 0 or symbol not in self.prices or self.prices[symbol] <= 0:
                target_contracts[symbol] = 0
                continue
                
            # Calculate target contracts
            price = self.prices[symbol]
            target_value = allocation
            target_shares = int(target_value / price)
            
            target_contracts[symbol] = target_shares // 100
        
        return {
            'dominant_regime': dominant_regime,
            'strategy_focus': strategy['focus'],
            'max_portfolio_coverage': strategy['max_portfolio_coverage'] * 100,  # Convert to percentage
            'max_position_coverage': strategy['max_position_coverage'] * 100,  # Convert to percentage
            'recommended_coverage_value': recommended_coverage,
            'position_allocations': position_allocations,
            'target_contracts': target_contracts
        }
    
    def _generate_execution_calendar(self, position_strategies: Dict[str, Any]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Generate execution calendar for implementing covered calls.
        
        Parameters:
        position_strategies (dict): Optimized strategies by symbol
        
        Returns:
        dict: Execution calendar
        """
        from datetime import datetime, timedelta
        
        # Get current date
        today = datetime.now().date()
        
        # Initialize calendar
        calendar = {}
        
        # Add execution dates for each position
        for symbol, strategy in position_strategies.items():
            if 'error' in strategy:
                continue
                
            # Get optimal parameters
            optimal = strategy.get('optimal_parameters', {})
            if not optimal:
                continue
                
            # Get parameters for execution
            strike = optimal.get('strike', 0)
            premium = optimal.get('premium', 0)
            dte = optimal.get('optimal_dte', 30)
            contracts = self.portfolio[symbol] // 100
            
            # Skip if no contracts
            if contracts <= 0:
                continue
                
            # Calculate optimal execution date (next trading day)
            # For simplicity, assume tomorrow is a trading day
            execution_date = today + timedelta(days=1)
            
            # Format date as string
            date_key = execution_date.strftime('%Y-%m-%d')
            
            # Create execution entry
            execution_entry = {
                'symbol': symbol,
                'strike': strike,
                'premium': premium,
                'dte': dte,
                'contracts': contracts,
                'execution_time': '10:00 AM - 11:00 AM',  # Example time window
                'order_type': 'Limit',
                'target_price': premium
            }
            
            # Add to calendar
            if date_key not in calendar:
                calendar[date_key] = {
                    'execution': [],
                    'monitoring': []
                }
                
            calendar[date_key]['execution'].append(execution_entry)
            
            # Add monitoring entries
            # Mid-point check
            mid_check_date = today + timedelta(days=dte//2)
            mid_date_key = mid_check_date.strftime('%Y-%m-%d')
            
            if mid_date_key not in calendar:
                calendar[mid_date_key] = {
                    'execution': [],
                    'monitoring': []
                }
                
            calendar[mid_date_key]['monitoring'].append({
                'symbol': symbol,
                'action': 'Mid-point check',
                'description': f"Check position status for {symbol} at {dte//2} days into trade"
            })
            
            # Pre-expiration check
            pre_exp_date = today + timedelta(days=dte-5)  # 5 days before expiration
            pre_exp_key = pre_exp_date.strftime('%Y-%m-%d')
            
            if pre_exp_key not in calendar:
                calendar[pre_exp_key] = {
                    'execution': [],
                    'monitoring': []
                }
                
            calendar[pre_exp_key]['monitoring'].append({
                'symbol': symbol,
                'action': 'Pre-expiration check',
                'description': f"Evaluate whether to roll or let expire for {symbol}"
            })
        
        return calendar