# optimization/strategy_optimizer.py
"""
Strategy Optimizer Module.

This module provides functions and classes for optimizing options strategies,
particularly covered calls, across different market regimes and risk levels.
"""

import logging
import math
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis.market_regime import MarketRegimeAnalyzer
from analysis.volatility import calculate_vol_forecast
from analysis.covered_call import CoveredCallStrategy

logger = logging.getLogger(__name__)


def run_enhanced_quant_analysis(
    portfolio: Dict[str, int],
    symbols_data: Dict[str, Dict[str, Any]],
    risk_levels: List[str] = ['conservative', 'moderate', 'aggressive'],
    run_backtest: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run enhanced quantitative analysis for covered call strategy.
    
    Parameters:
    portfolio (dict): Portfolio positions {symbol: quantity}
    symbols_data (dict): Pre-fetched data for symbols
    risk_levels (list): Risk levels to analyze
    run_backtest (bool): Whether to run backtest simulations
    
    Returns:
    tuple: (analysis results, backtests)
    """
    results = {}
    backtests = {}
    
    # Build correlation matrix for portfolio analysis
    try:
        from analysis.portfolio import build_portfolio_correlation
        portfolio_corr = build_portfolio_correlation(symbols_data)
    except Exception as e:
        logger.error(f"Error building correlation matrix: {e}")
        portfolio_corr = {}
    
    # Run analysis for each risk level
    for risk_level in risk_levels:
        logger.info(f"\n--- Analysis at {risk_level} risk level ---")
        
        position_results = {}
        portfolio_data = {
            'positions': {},
            'correlations': portfolio_corr,
            'risk_metrics': {}
        }
        
        total_value = 0
        
        # Analyze each symbol
        for symbol, shares in portfolio.items():
            if symbol not in symbols_data:
                logger.warning(f"No data for {symbol}, skipping")
                continue
                
            symbol_data = symbols_data[symbol]
            
            logger.info(f"Analyzing {symbol} => {shares} shares")
            
            try:
                # Initialize covered call strategy
                current_price = symbol_data.get('current_price', 0)
                position_value = shares * current_price
                
                # Run market regime analysis if not precomputed
                if 'market_analyzer' not in symbol_data and 'weekly_data' in symbol_data:
                    market_analyzer = MarketRegimeAnalyzer(symbol_data['weekly_data'])
                    symbol_data['market_analyzer'] = market_analyzer
                else:
                    market_analyzer = symbol_data.get('market_analyzer')
                
                # Get market regime
                if market_analyzer:
                    market_regime = market_analyzer.get_current_regime()
                    symbol_data['market_regime'] = market_regime
                else:
                    market_regime = symbol_data.get('market_regime', 3)  # Default to neutral
                
                # Calculate volatility forecast if not precomputed
                if 'vol_forecast' not in symbol_data and 'daily_data' in symbol_data:
                    daily_returns = symbol_data['daily_data']['Close'].pct_change().dropna()
                    vol_analysis = calculate_vol_forecast(daily_returns, market_regime)
                    symbol_data['vol_forecast'] = vol_analysis['forecast']
                    symbol_data['volatility_analysis'] = vol_analysis
                
                vol_forecast = symbol_data.get('vol_forecast', 0.2)  # Default to 20%
                
                # Create covered call strategy
                ccs = CoveredCallStrategy(
                    symbol, 
                    shares, 
                    risk_level=risk_level, 
                    slippage=0.001, 
                    commission=0.65,
                    market_regime=market_regime,
                    vol_forecast=vol_forecast,
                    stock_data=symbol_data,
                    option_chains=symbol_data.get('option_chains')
                )
                
                # Get recommendations
                recommendations = ccs.get_recommendations()
                position_results[symbol] = recommendations
                
                # Add to portfolio data
                portfolio_data['positions'][symbol] = {
                    'value': position_value,
                    'weight': position_value / sum(symbols_data[s].get('current_price', 0) * portfolio[s] for s in portfolio),
                    'regime': market_regime,
                    'regime_name': _get_regime_name(market_regime),
                    'volatility': vol_forecast,
                    'risk_metrics': recommendations.get('risk_metrics', {})
                }
                
                total_value += position_value
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                position_results[symbol] = {}  # Empty dict to avoid KeyError
        
        # Calculate portfolio-level risk metrics
        if total_value > 0:
            # Calculate weights
            weights = {
                symbol: portfolio_data['positions'][symbol]['value'] / total_value
                for symbol in portfolio_data['positions']
            }
            
            # Calculate portfolio volatility
            try:
                from analysis.portfolio import calculate_portfolio_volatility
                port_vol = calculate_portfolio_volatility(
                    weights,
                    {symbol: data['volatility'] for symbol, data in portfolio_data['positions'].items()},
                    portfolio_corr.get('price', pd.DataFrame())
                )
            except Exception as e:
                logger.error(f"Error calculating portfolio volatility: {e}")
                # Fallback to weighted average
                port_vol = sum(
                    weights[symbol] * portfolio_data['positions'][symbol]['volatility']
                    for symbol in weights
                )
            
            # Calculate VaR and Expected Shortfall
            var_99 = total_value * port_vol * abs(norm.ppf(0.01)) / math.sqrt(52)
            es_99 = total_value * port_vol / math.sqrt(52) * norm.pdf(norm.ppf(0.01)) / 0.01
            
            # Calculate diversification ratio
            weighted_vol_sum = sum(
                weights[symbol] * portfolio_data['positions'][symbol]['volatility']
                for symbol in weights
            )
            div_ratio = 1 - (port_vol / weighted_vol_sum if weighted_vol_sum > 0 else 0)
            
            # Store portfolio risk metrics
            portfolio_data['risk_metrics'] = {
                'total_value': total_value,
                'portfolio_volatility': port_vol,
                'value_at_risk': var_99,
                'expected_shortfall': es_99,
                'diversification_ratio': div_ratio
            }
        
        # Store results for this risk level
        results[risk_level] = {
            'position_results': position_results,
            'portfolio_data': portfolio_data
        }
        
        # Run backtests if requested
        if run_backtest and risk_level == risk_levels[0]:  # Only for first risk level
            for symbol, shares in portfolio.items():
                if symbol not in symbols_data:
                    continue
                    
                logger.info(f"Running backtest for {symbol}")
                
                try:
                    # Import backtest function within the loop to avoid circular imports
                    from backtest.covered_call_backtest import backtest_covered_call
                    from backtest.performance_metrics import calculate_risk_adjusted_metrics
                    
                    bt = backtest_covered_call(symbol, shares, rolling_days=30)
                    
                    # Calculate additional performance metrics
                    if 'daily_cumulative_pnl' in bt and bt['daily_index'] is not None:
                        daily_returns = pd.Series(bt['daily_cumulative_pnl']).pct_change().dropna()
                        
                        if not daily_returns.empty:
                            metrics = calculate_risk_adjusted_metrics(daily_returns)
                            
                            bt['sharpe_ratio'] = metrics['sharpe_ratio']
                            bt['sortino_ratio'] = metrics['sortino_ratio']
                            bt['max_drawdown'] = metrics['max_drawdown']
                            bt['calmar_ratio'] = metrics['calmar_ratio']
                    
                    backtests[symbol] = bt
                except Exception as e:
                    logger.error(f"Backtest for {symbol} failed: {e}")
                    backtests[symbol] = {}
    
    return results, backtests


def _get_regime_name(regime: int) -> str:
    """Get the name for a market regime."""
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


class ImplementationPlan:
    """
    Implementation plan for options strategies.
    
    This class provides tools to generate execution plans, risk management
    guidelines, and implementation details for options strategies.
    """
    
    def __init__(
        self, 
        portfolio: Dict[str, int], 
        recommendations: Dict[str, Any], 
        current_prices: Dict[str, float]
    ):
        """
        Initialize the implementation plan.
        
        Parameters:
        portfolio (dict): Current portfolio positions {symbol: quantity}
        recommendations (dict): Strategy recommendations
        current_prices (dict): Current prices {symbol: price}
        """
        self.portfolio = portfolio
        self.recommendations = recommendations
        self.current_prices = current_prices
    
    def generate_execution_plan(self) -> List[Dict[str, Any]]:
        """
        Generate execution plan for implementing covered call strategy.
        
        Returns:
        list: Execution plan
        """
        execution_plan = []
        
        for symbol, rec in self.recommendations.items():
            if symbol not in self.portfolio or symbol not in self.current_prices:
                continue
                
            shares = self.portfolio[symbol]
            current_price = self.current_prices[symbol]
            contracts = shares // 100
            
            if contracts == 0:
                continue
                
            # Get recommendation details
            optimal_strikes = rec.get('optimal_strikes', {})
            if not optimal_strikes:
                continue
                
            # Process each DTE option
            for dte, details in optimal_strikes.items():
                strike = details['strike']
                premium = details['call_price']
                greeks = details.get('greeks', {})
                
                delta = greeks.get('delta', 0.30)
                theta = greeks.get('theta', 0)
                vega = greeks.get('vega', 0)
                gamma = greeks.get('gamma', 0)
                
                prob_otm = details.get('probabilities', {}).get('prob_otm', 0)
                
                # Calculate execution parameters
                upside_potential = (strike - current_price) / current_price
                premium_yield = premium / current_price
                annualized_return = premium_yield * (365 / int(dte)) * 100
                
                # Determine execution strategy
                if premium < 0.50:
                    # For low premium options, use limit orders slightly below mid
                    execution_strategy = f"Limit order at {premium:.2f}, decrease by 0.05 every 15 mins until filled"
                    slippage_est = 0.01  # 1 cent
                elif premium >= 5.00:
                    # For high premium options, can be more aggressive
                    execution_strategy = f"Limit order at mid price, adjust every 5 mins until filled"
                    slippage_est = 0.05  # 5 cents
                else:
                    # For medium premium options
                    execution_strategy = f"Limit order at mid price, adjust every 10 mins until filled"
                    slippage_est = 0.03  # 3 cents
                
                # Calculate trading costs
                commission = 0.65 * contracts  # $0.65 per contract
                slippage_cost = slippage_est * contracts
                total_cost = commission + slippage_cost
                cost_pct = total_cost / (premium * contracts * 100) if premium > 0 else 0
                
                # Add to execution plan
                execution_plan.append({
                    'symbol': symbol,
                    'action': 'Sell to Open',
                    'contracts': contracts,
                    'option_type': 'Call',
                    'strike': strike,
                    'expiry': f"DTE: {dte}",
                    'premium': premium,
                    'total_premium': premium * contracts * 100,
                    'upside_potential': upside_potential * 100,  # Convert to percentage
                    'premium_yield': premium_yield * 100,  # Convert to percentage
                    'annualized_return': annualized_return,
                    'probability_otm': prob_otm * 100,  # Convert to percentage
                    'delta': delta,
                    'theta': theta,
                    'vega': vega,
                    'gamma': gamma,
                    'execution_strategy': execution_strategy,
                    'trading_costs': {
                        'commission': commission,
                        'slippage_est': slippage_cost,
                        'total_cost': total_cost,
                        'cost_percentage': cost_pct * 100  # Convert to percentage
                    }
                })
        
        return sorted(execution_plan, key=lambda x: x['annualized_return'], reverse=True)
    
    def risk_management_guidelines(
        self, 
        market_regime: int, 
        vol_forecast: float
    ) -> Dict[str, Any]:
        """
        Generate risk management guidelines.
        
        Parameters:
        market_regime (int): Current market regime
        vol_forecast (float): Volatility forecast
        
        Returns:
        dict: Risk management guidelines
        """
        # Calculate aggregate position metrics
        total_portfolio_value = sum(self.portfolio.get(symbol, 0) * self.current_prices.get(symbol, 0) 
                                  for symbol in self.portfolio)
        
        # Define risk parameters by regime
        regime_risk = {
            0: {'max_allocation': 0.50, 'max_delta': 0.20, 'stop_loss_mult': 2.0},  # Severe Bearish
            1: {'max_allocation': 0.60, 'max_delta': 0.25, 'stop_loss_mult': 1.8},  # Bearish
            2: {'max_allocation': 0.70, 'max_delta': 0.30, 'stop_loss_mult': 1.6},  # Weak Bearish
            3: {'max_allocation': 0.80, 'max_delta': 0.40, 'stop_loss_mult': 1.5},  # Neutral
            4: {'max_allocation': 0.90, 'max_delta': 0.45, 'stop_loss_mult': 1.4},  # Weak Bullish
            5: {'max_allocation': 1.00, 'max_delta': 0.50, 'stop_loss_mult': 1.3},  # Bullish
            6: {'max_allocation': 1.00, 'max_delta': 0.55, 'stop_loss_mult': 1.2}   # Strong Bullish
        }
        
        # Get risk parameters for current regime
        risk_params = regime_risk.get(market_regime, regime_risk[3])  # Default to neutral
        
        # Adjust for volatility
        vol_adj_factor = vol_forecast / 0.20  # Normalized to "typical" vol of 20%
        
        # Calculate risk budget
        max_portfolio_loss = total_portfolio_value * 0.05  # 5% max loss
        
        # Calculate position-level risk limits
        position_risk_limits = {}
        for symbol, shares in self.portfolio.items():
            if symbol not in self.current_prices:
                continue
                
            current_price = self.current_prices[symbol]
            position_value = shares * current_price
            position_weight = position_value / total_portfolio_value if total_portfolio_value > 0 else 0
            
            position_risk_limits[symbol] = {
                'max_loss_amount': max_portfolio_loss * position_weight,
                'max_loss_pct': 5 * position_weight,  # 5% * weight
                'max_delta': risk_params['max_delta'] / vol_adj_factor,
                'gamma_limit': 0.01 * position_value / vol_adj_factor,
                'vega_limit': 0.02 * position_value * vol_adj_factor,  # More vega allowed in high vol
                'max_contracts_pct': min(0.75, 0.5 + 0.25 * position_weight)  # % of shares to write calls against
            }
        
        # Define exit criteria
        exit_criteria = {
            'profit_target_pct': 75 - (vol_adj_factor - 1) * 10,  # Lower profit target in high vol
            'loss_limit_pct': 200 * risk_params['stop_loss_mult'],
            'dte_threshold': 5,  # Consider closing when DTE <= 5
            'delta_threshold': 0.85,  # Consider closing when delta >= 0.85
            'gamma_threshold': 0.05 * vol_adj_factor  # Higher gamma threshold in high vol
        }
        
        # Define adjustment triggers
        adjustment_triggers = {
            'underlying_moves': f"+/-{max(3, min(7, int(vol_forecast * 100)))}% in a single day",
            'volatility_spike': f"VIX increases by {max(15, min(25, int(vol_forecast * 100)))}% or more in a week",
            'delta_threshold_reached': f"Position delta exceeds {risk_params['max_delta']:.2f}",
            'earnings_announcement': "Close or roll positions before earnings"
        }
        
        return {
            'portfolio_level': {
                'total_value': total_portfolio_value,
                'max_loss': max_portfolio_loss,
                'max_delta_exposure': risk_params['max_delta'] * total_portfolio_value / vol_adj_factor,
                'cash_reserve_min': max(0.05, (vol_forecast - 0.15) * 2) * total_portfolio_value,  # Higher reserves in high vol
                'max_vega_exposure': 0.02 * total_portfolio_value * vol_adj_factor
            },
            'position_risk_limits': position_risk_limits,
            'exit_criteria': exit_criteria,
            'monitoring_frequency': 'Daily' if vol_forecast > 0.25 else 'Weekly',
            'adjustment_triggers': adjustment_triggers,
            'market_regime': market_regime,
            'volatility_forecast': vol_forecast,
            'volatility_adjustment_factor': vol_adj_factor
        }
    
    def execution_timing_strategy(
        self, 
        market_regime: int
    ) -> Dict[str, str]:
        """
        Generate execution timing strategy.
        
        Parameters:
        market_regime (int): Current market regime
        
        Returns:
        dict: Execution timing strategy
        """
        # Define timing parameters by regime
        regime_timing = {
            0: {'time_of_day': 'Morning', 'day_of_week': 'Tuesday-Thursday', 'avoid_days': 'Monday, Friday'},  # Severe Bearish
            1: {'time_of_day': 'Morning', 'day_of_week': 'Tuesday-Thursday', 'avoid_days': 'Monday'},  # Bearish
            2: {'time_of_day': 'Mid-day', 'day_of_week': 'Tuesday-Thursday', 'avoid_days': None},  # Weak Bearish
            3: {'time_of_day': 'Any', 'day_of_week': 'Any', 'avoid_days': None},  # Neutral
            4: {'time_of_day': 'Mid-day', 'day_of_week': 'Any', 'avoid_days': None},  # Weak Bullish
            5: {'time_of_day': 'Afternoon', 'day_of_week': 'Monday-Thursday', 'avoid_days': 'Friday'},  # Bullish
            6: {'time_of_day': 'Afternoon', 'day_of_week': 'Monday-Wednesday', 'avoid_days': 'Thursday, Friday'}  # Strong Bullish
        }
        
        # Get timing parameters for current regime
        timing_params = regime_timing.get(market_regime, regime_timing[3])  # Default to neutral
        
        # Generate IV dynamics by time of day
        iv_dynamics = {
            'Morning': 'IV typically highest at market open, especially in high-volatility regimes',
            'Mid-day': 'IV typically stabilizes during mid-day, good for fair pricing',
            'Afternoon': 'IV often decreases into market close, especially in low-volatility regimes',
            'Any': 'IV dynamics less predictable in current regime'
        }
        
        # Generate day of week effects
        dow_effects = {
            'Monday': 'Often shows weekend gap effects and higher uncertainty',
            'Tuesday-Thursday': 'Typically most stable trading days',
            'Friday': 'Can see option premium compression due to weekend theta decay',
            'Any': 'Day of week less important in current regime'
        }
        
        # Generate other considerations
        other_considerations = []
        
        if market_regime in [0, 1, 2]:  # Bearish regimes
            other_considerations.append("Consider breaking up large orders to minimize market impact")
            other_considerations.append("Avoid selling calls immediately after significant down days")
        elif market_regime in [4, 5, 6]:  # Bullish regimes
            other_considerations.append("Consider selling on strength (intraday rallies)")
            other_considerations.append("Watch for potential gamma squeezes in high-volume names")
        
        if 'avoid_days' in timing_params and timing_params['avoid_days']:
            other_considerations.append(f"Avoid executing on {timing_params['avoid_days']} if possible")
        
        return {
            'optimal_time_of_day': timing_params['time_of_day'],
            'optimal_days_of_week': timing_params['day_of_week'],
            'days_to_avoid': timing_params['avoid_days'],
            'iv_dynamics': iv_dynamics.get(timing_params['time_of_day'], ''),
            'day_of_week_effects': dow_effects.get(timing_params['day_of_week'].split('-')[0], ''),
            'other_considerations': other_considerations
        }
    
    def liquidity_analysis(self) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Analyze option liquidity for execution planning.
        
        Returns:
        dict: Liquidity analysis
        """
        liquidity_analysis = {}
        
        for symbol, rec in self.recommendations.items():
            if symbol not in self.portfolio or symbol not in self.current_prices:
                continue
                
            # Get recommendation details
            optimal_strikes = rec.get('optimal_strikes', {})
            if not optimal_strikes:
                continue
                
            symbol_liquidity = {}
            
            for dte, details in optimal_strikes.items():
                strike = details['strike']
                bid = details.get('bid', 0)
                ask = details.get('ask', 0)
                
                # If bid/ask not available, estimate from premium
                if bid == 0 and ask == 0:
                    premium = details['call_price']
                    bid = premium * 0.95  # Estimate bid as 95% of mid
                    ask = premium * 1.05  # Estimate ask as 105% of mid
                
                # Calculate liquidity metrics
                spread = ask - bid
                spread_pct = spread / ((bid + ask) / 2) if (bid + ask) > 0 else float('inf')
                
                # Assess liquidity
                if spread_pct < 0.05:
                    liquidity_rating = "Excellent"
                    max_contracts_per_order = 50
                elif spread_pct < 0.10:
                    liquidity_rating = "Good"
                    max_contracts_per_order = 20
                elif spread_pct < 0.20:
                    liquidity_rating = "Fair"
                    max_contracts_per_order = 10
                else:
                    liquidity_rating = "Poor"
                    max_contracts_per_order = 5
                
                symbol_liquidity[int(dte)] = {
                    'strike': strike,
                    'bid': bid,
                    'ask': ask,
                    'spread': spread,
                    'spread_percentage': spread_pct * 100,  # Convert to percentage
                    'liquidity_rating': liquidity_rating,
                    'max_contracts_per_order': max_contracts_per_order,
                    'execution_recommendation': self._get_liquidity_recommendation(liquidity_rating)
                }
            
            liquidity_analysis[symbol] = symbol_liquidity
        
        return liquidity_analysis
    
    def _get_liquidity_recommendation(self, liquidity_rating: str) -> str:
        """Generate execution recommendation based on liquidity."""
        if liquidity_rating == "Excellent":
            return "Market orders acceptable; limit orders at mid-price should fill quickly"
        elif liquidity_rating == "Good":
            return "Use limit orders at or near mid-price; adjust every 5-10 minutes if needed"
        elif liquidity_rating == "Fair":
            return "Use limit orders slightly below mid-price; smaller order sizes; be patient"
        else:  # Poor
            return "Use limit orders on bid side or slightly higher; consider multiple small orders"