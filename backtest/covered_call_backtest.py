# backtest/covered_call_backtest.py
"""
Covered Call Backtest Module.

This module provides functions for backtesting covered call strategies
on historical data, with various parameter settings and configuration options.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from data.fetcher import get_stock_history_daily

logger = logging.getLogger(__name__)


def backtest_covered_call(
    symbol: str,
    shares: int,
    risk_level: str = 'conservative',
    rolling_days: int = 30,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    commission: float = 0.65,
    slippage: float = 0.01
) -> Dict[str, Any]:
    """
    Backtest a covered call strategy on daily data.

    Parameters:
        symbol (str): Stock symbol.
        shares (int): Number of shares held.
        risk_level (str): 'conservative', 'moderate', 'aggressive'.
        rolling_days (int): Days until rolling the option to the next expiry.
        start_date (str, optional): Start date for backtest in YYYY-MM-DD format.
        end_date (str, optional): End date for backtest in YYYY-MM-DD format.
        commission (float): Commission per contract in dollars.
        slippage (float): Slippage in dollars per contract.

    Returns:
        dict: Backtest summary data, including daily P&L arrays and trade list.
    """
    # Parameter mappings based on risk level
    risk_params = {
        'conservative': {'delta_target': 0.25, 'dte_range': (21, 45)},
        'moderate': {'delta_target': 0.35, 'dte_range': (30, 60)},
        'aggressive': {'delta_target': 0.45, 'dte_range': (45, 90)}
    }
    
    # Get parameters for selected risk level
    params = risk_params.get(risk_level, risk_params['conservative'])
    delta_target = params['delta_target']
    dte_min, dte_max = params['dte_range']

    # 1) Fetch historical data (daily)
    try:
        daily_data = get_stock_history_daily(symbol, period="3y")
        if daily_data.empty:
            return {'error': f"No historical data found for {symbol}"}
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return {'error': f"Failed to fetch data: {e}"}

    # Apply date filters if provided
    if start_date:
        start_date = pd.to_datetime(start_date)
        daily_data = daily_data[daily_data.index >= start_date]
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        daily_data = daily_data[daily_data.index <= end_date]

    if daily_data.empty:
        return {'error': "No data available for the specified date range"}

    # 2) Calculate how many contracts we can write (1 contract = 100 shares)
    contracts = shares // 100
    if contracts == 0:
        return {'error': 'Not enough shares for at least one option contract'}

    # 3) Initialize tracking variables
    position_value = daily_data['Close'].iloc[0] * shares
    cash = 0.0
    active_position = False
    position_details = {}
    trades = []
    daily_pnl = []
    daily_index = []

    # Calculate rolling volatility for option pricing
    daily_data['returns'] = daily_data['Close'].pct_change()
    daily_data['rolling_vol'] = daily_data['returns'].rolling(21).std() * np.sqrt(252)
    daily_data['rolling_vol'].fillna(0.2, inplace=True)  # Default to 20% vol

    # 4) Main backtest loop from day=30 (for minimal lookback) up to len(daily_data)
    for i in range(30, len(daily_data)):
        date = daily_data.index[i]
        price = daily_data['Close'].iloc[i]
        daily_index.append(date)  # Always append the current date

        # If we don't have enough shares to write even 1 covered call, we can break
        if shares < 100:
            break

        # Current volatility measure
        current_vol = daily_data['rolling_vol'].iloc[i]

        # Check if we need to open or roll the covered call
        if not active_position:
            # Determine strike price based on delta target
            # For simplicity, use a rule-based approach rather than full BSM
            dte = rolling_days
            strike_multiplier = 1.0
            
            # Adjust strike based on delta target (higher delta = closer to the money)
            if delta_target <= 0.3:  # Conservative
                strike_multiplier = 1.05
            elif delta_target <= 0.4:  # Moderate
                strike_multiplier = 1.03
            else:  # Aggressive
                strike_multiplier = 1.02
                
            strike = price * strike_multiplier
            
            # Calculate approximate option premium
            t = dte / 365  # Time to expiry in years
            option_std = price * current_vol * np.sqrt(t)  # Expected standard deviation move
            
            # Simple model: premium decreases as strike increases above current price
            # Higher vol increases premium
            moneyness = (strike - price) / price
            premium = price * current_vol * t * max(0.1, (1 - 2 * moneyness))
            
            # Ensure reasonable premium (at least 0.5% of stock price)
            premium = max(premium, price * 0.005)

            # Record trade
            new_trade = {
                'open_date': date,
                'type': 'sell_call',
                'strike': strike,
                'premium': premium,
                'price_at_entry': price,
                'volatility': current_vol,
                'delta': delta_target
            }
            
            trades.append(new_trade)

            # Update cash with the premium (less transaction costs)
            transaction_cost = commission * contracts + slippage * contracts
            cash += (premium * contracts * 100) - transaction_cost
            
            active_position = True
            position_details = {
                'open_date': date,
                'strike': strike,
                'premium': premium,
                'days_held': 0,
                'price_at_entry': price,
                'dte': dte
            }

        else:
            # We have an active covered call, increment days_held
            position_details['days_held'] += 1
            days_held = position_details['days_held']
            days_remaining = position_details['dte'] - days_held

            # Check if time to roll or if stock is above strike (assignment scenario)
            strike = position_details['strike']
            price_at_entry = position_details.get('price_at_entry', price)
            original_premium = position_details['premium']
            
            # Calculate current option value (simple time decay model)
            time_passed_ratio = days_held / position_details['dte']
            intrinsic = max(0, price - strike)
            time_value = original_premium * (1 - time_passed_ratio) * (1 - intrinsic / original_premium if original_premium > 0 else 0)
            current_option_value = intrinsic + time_value
            
            # Check if we should close or roll
            if days_remaining <= 5 or price > strike:
                # Close out the existing option
                call_pnl = 0.0
                
                if price > strike:
                    # Assigned - we lose the shares at 'strike'
                    # But we keep the premium
                    call_pnl = (strike - price_at_entry + original_premium) * contracts * 100
                    # Realistically, you'd have to "sell" your shares at strike
                    # so your final position in the stock is zero:
                    shares -= (contracts * 100)
                else:
                    # Close position - buy back option
                    transaction_cost = commission * contracts + slippage * contracts
                    call_pnl = (original_premium - current_option_value) * contracts * 100 - transaction_cost

                trades.append({
                    'close_date': date,
                    'type': 'close_call',
                    'strike': strike,
                    'price_at_close': price,
                    'option_value_at_close': current_option_value,
                    'days_held': days_held,
                    'pnl': call_pnl,
                    'price_at_entry': price_at_entry
                })

                active_position = False

        # 5) Calculate daily portfolio value: cash + value of shares
        stock_value = price * shares
        daily_pnl.append(stock_value + cash)

    # ---------- Ensure daily_index and daily_pnl match length -----------
    min_len = min(len(daily_index), len(daily_pnl))
    daily_index = daily_index[:min_len]
    daily_pnl = daily_pnl[:min_len]

    # 6) Compute final stats, total return, etc.
    if len(daily_pnl) > 1:
        start_value = daily_pnl[0]
        end_value = daily_pnl[-1]
        total_return = (end_value / start_value) - 1

        days = (daily_index[-1] - daily_index[0]).days
        if days > 0:
            annualized_return = (1 + total_return) ** (365 / days) - 1
        else:
            annualized_return = 0

        # Basic max drawdown
        cum_pnl = pd.Series(daily_pnl, index=daily_index)
        running_max = cum_pnl.cummax()
        drawdowns = (cum_pnl / running_max) - 1
        max_drawdown = drawdowns.min()
    else:
        total_return = 0
        annualized_return = 0
        max_drawdown = 0

    return {
        'symbol': symbol,
        'initial_shares': shares,
        'risk_level': risk_level,
        'start_date': daily_index[0] if len(daily_index) else None,
        'end_date': daily_index[-1] if len(daily_index) else None,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'trades': trades,
        'daily_cumulative_pnl': daily_pnl,
        'daily_index': daily_index
    }


def backtest_multiple_strategies(
    symbol: str,
    shares: int,
    risk_levels: List[str] = ['conservative', 'moderate', 'aggressive'],
    rolling_days_options: List[int] = [21, 30, 45],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Backtest multiple covered call strategies with different parameters.
    
    Parameters:
        symbol (str): Stock symbol.
        shares (int): Number of shares held.
        risk_levels (list): Risk levels to test.
        rolling_days_options (list): Rolling days options to test.
        start_date (str, optional): Start date for backtest.
        end_date (str, optional): End date for backtest.
    
    Returns:
        dict: Dictionary of backtest results by strategy.
    """
    results = {}
    
    # Run backtests for each combination of parameters
    for risk_level in risk_levels:
        for rolling_days in rolling_days_options:
            strategy_name = f"{risk_level}_{rolling_days}d"
            
            try:
                bt_result = backtest_covered_call(
                    symbol=symbol,
                    shares=shares,
                    risk_level=risk_level,
                    rolling_days=rolling_days,
                    start_date=start_date,
                    end_date=end_date
                )
                
                results[strategy_name] = bt_result
            except Exception as e:
                logger.error(f"Error running backtest {strategy_name}: {e}")
                results[strategy_name] = {'error': str(e)}
    
    # Add a buy and hold baseline for comparison
    try:
        daily_data = get_stock_history_daily(symbol, period="3y")
        
        if not daily_data.empty:
            # Apply date filters if provided
            if start_date:
                start_date = pd.to_datetime(start_date)
                daily_data = daily_data[daily_data.index >= start_date]
            
            if end_date:
                end_date = pd.to_datetime(end_date)
                daily_data = daily_data[daily_data.index <= end_date]
            
            # Calculate buy and hold performance
            start_price = daily_data['Close'].iloc[0]
            end_price = daily_data['Close'].iloc[-1]
            total_return = (end_price / start_price) - 1
            
            days = (daily_data.index[-1] - daily_data.index[0]).days
            if days > 0:
                annualized_return = (1 + total_return) ** (365 / days) - 1
            else:
                annualized_return = 0
            
            # Calculate max drawdown
            prices = daily_data['Close']
            drawdowns = prices / prices.cummax() - 1
            max_drawdown = drawdowns.min()
            
            results['buy_and_hold'] = {
                'symbol': symbol,
                'initial_shares': shares,
                'start_date': daily_data.index[0],
                'end_date': daily_data.index[-1],
                'total_return': total_return,
                'annualized_return': annualized_return,
                'max_drawdown': max_drawdown
            }
    except Exception as e:
        logger.error(f"Error calculating buy and hold baseline: {e}")
        results['buy_and_hold'] = {'error': str(e)}
    
    return results


def compare_backtest_results(
    results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compare different backtest strategy results.
    
    Parameters:
    results (dict): Dictionary of backtest results by strategy.
    
    Returns:
    dict: Comparison metrics.
    """
    comparison = {
        'strategies': [],
        'total_returns': [],
        'annualized_returns': [],
        'max_drawdowns': [],
        'sharpe_ratios': [],
        'sortino_ratios': [],
        'calmar_ratios': []
    }
    
    # Collect results for valid strategies
    for strategy, result in results.items():
        if 'error' in result:
            continue
        
        comparison['strategies'].append(strategy)
        comparison['total_returns'].append(result.get('total_return', 0) * 100)  # Convert to percentage
        comparison['annualized_returns'].append(result.get('annualized_return', 0) * 100)  # Convert to percentage
        comparison['max_drawdowns'].append(result.get('max_drawdown', 0) * 100)  # Convert to percentage
        
        # Add risk-adjusted metrics if available
        comparison['sharpe_ratios'].append(result.get('sharpe_ratio', 0))
        comparison['sortino_ratios'].append(result.get('sortino_ratio', 0))
        
        # Calculate Calmar ratio if not available
        if 'calmar_ratio' in result:
            calmar = result['calmar_ratio']
        else:
            ann_return = result.get('annualized_return', 0)
            max_dd = result.get('max_drawdown', 0)
            calmar = -ann_return / max_dd if max_dd < 0 else 0
            
        comparison['calmar_ratios'].append(calmar)
    
    # Find best strategy by different metrics
    if comparison['strategies']:
        # Best by annualized return
        best_return_idx = np.argmax(comparison['annualized_returns'])
        best_return_strategy = comparison['strategies'][best_return_idx]
        
        # Best by Sharpe ratio
        best_sharpe_idx = np.argmax(comparison['sharpe_ratios'])
        best_sharpe_strategy = comparison['strategies'][best_sharpe_idx]
        
        # Best by Calmar ratio (risk-adjusted)
        best_calmar_idx = np.argmax(comparison['calmar_ratios'])
        best_calmar_strategy = comparison['strategies'][best_calmar_idx]
        
        comparison['best_strategies'] = {
            'best_return': best_return_strategy,
            'best_sharpe': best_sharpe_strategy,
            'best_calmar': best_calmar_strategy
        }
    
    return comparison