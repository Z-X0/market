# backtest/performance_metrics.py
"""
Performance Metrics Module.

This module provides functions for calculating and analyzing performance metrics
for options strategies and portfolio returns.
"""

import logging
import math
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Constants
RISK_FREE_RATE = 0.02  # 2% risk-free rate


def calculate_risk_adjusted_metrics(
    returns: pd.Series, 
    risk_free_rate: float = RISK_FREE_RATE, 
    period: str = 'D'
) -> Dict[str, float]:
    """
    Calculate comprehensive risk-adjusted performance metrics.
    
    Parameters:
    returns (pandas.Series): Strategy returns
    risk_free_rate (float): Annualized risk-free rate
    period (str): 'D' for daily, 'W' for weekly, 'M' for monthly
    
    Returns:
    dict: Dictionary of risk-adjusted metrics
    """
    # Check for empty or invalid returns
    if returns.empty:
        return {
            'annualized_return': 0,
            'annualized_volatility': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'calmar_ratio': 0,
            'omega_ratio': 0,
            'skewness': 0,
            'kurtosis': 0,
            'var_95': 0,
            'cvar_95': 0,
            'profit_factor': 0,
            'gain_to_pain': 0,
            'ulcer_index': 0
        }
    
    # Determine annualization factor based on period
    ann_factor = {'D': 252, 'W': 52, 'M': 12}.get(period, 252)
    rf_period = risk_free_rate / ann_factor
    
    # Calculate excess returns
    excess_returns = returns - rf_period
    
    # Basic metrics
    avg_return = returns.mean() * ann_factor
    volatility = returns.std() * np.sqrt(ann_factor)
    sharpe = (avg_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Downside metrics
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(ann_factor) if not downside_returns.empty else 0
    sortino = (avg_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
    
    # Maximum drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns / running_max) - 1
    max_drawdown = drawdowns.min()
    calmar = (avg_return - risk_free_rate) / abs(max_drawdown) if max_drawdown < 0 else float('inf')
    
    # Omega ratio (threshold = risk-free rate)
    threshold = rf_period
    omega_numerator = np.sum(returns[returns > threshold] - threshold)
    omega_denominator = np.sum(threshold - returns[returns < threshold])
    omega = omega_numerator / omega_denominator if omega_denominator > 0 else float('inf')
    
    # Higher moments
    returns_skewness = stats.skew(returns.dropna())
    returns_kurtosis = stats.kurtosis(returns.dropna())
    
    # VaR and CVaR (Expected Shortfall)
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean() if any(returns <= var_95) else var_95
    
    # Additional metrics
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    
    profit_factor = (positive_returns.sum() / abs(negative_returns.sum()) 
                      if not negative_returns.empty and negative_returns.sum() < 0 
                      else float('inf'))
    
    gain_to_pain = (positive_returns.mean() / abs(negative_returns.mean()) 
                     if not negative_returns.empty and negative_returns.mean() < 0 
                     else float('inf'))
    
    # Ulcer Index (quadratic underwater area)
    ulcer_index = np.sqrt(np.mean(np.square(drawdowns)))
    
    return {
        'annualized_return': avg_return,
        'annualized_volatility': volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar,
        'omega_ratio': omega,
        'skewness': returns_skewness,
        'kurtosis': returns_kurtosis,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'profit_factor': profit_factor,
        'gain_to_pain': gain_to_pain,
        'ulcer_index': ulcer_index
    }


def calculate_metrics_by_regime(
    returns: pd.Series, 
    regimes: pd.Series, 
    risk_free_rate: float = RISK_FREE_RATE, 
    period: str = 'D'
) -> Dict[int, Dict[str, float]]:
    """
    Calculate performance metrics for each regime.
    
    Parameters:
    returns (pandas.Series): Strategy returns
    regimes (pandas.Series): Regime classifications with same index as returns
    risk_free_rate (float): Annualized risk-free rate
    period (str): 'D' for daily, 'W' for weekly, 'M' for monthly
    
    Returns:
    dict: Dictionary of metrics by regime
    """
    # Align data
    aligned = pd.concat([returns, regimes], axis=1).dropna()
    if aligned.empty:
        return {}
        
    aligned.columns = ['returns', 'regime']
    
    metrics_by_regime = {}
    
    # Calculate for each regime
    for regime in aligned['regime'].unique():
        regime_returns = aligned[aligned['regime'] == regime]['returns']
        if len(regime_returns) > 5:  # Need minimum sample size
            metrics_by_regime[int(regime)] = calculate_risk_adjusted_metrics(
                regime_returns, risk_free_rate, period
            )
            # Add sample size
            metrics_by_regime[int(regime)]['sample_size'] = len(regime_returns)
            
    return metrics_by_regime


def calculate_turnover(
    positions: pd.DataFrame, 
    prices: Optional[pd.DataFrame] = None
) -> float:
    """
    Calculate annualized turnover rate for portfolio or strategy.
    
    Parameters:
    positions (pandas.DataFrame): Daily position sizes
    prices (pandas.DataFrame, optional): Price data if positions are in units
    
    Returns:
    float: Annualized turnover rate
    """
    if positions.empty:
        return 0.0
        
    # If prices provided, convert positions to monetary values
    if prices is not None:
        # Align data
        aligned_prices = prices.reindex(positions.index)
        position_values = positions * aligned_prices
    else:
        position_values = positions  # Assume already in monetary values
    
    # Calculate daily changes
    daily_turnover = position_values.diff().abs().sum(axis=1)
    
    # Calculate portfolio value
    portfolio_value = position_values.abs().sum(axis=1)
    
    # Calculate turnover ratio
    turnover_ratio = daily_turnover / portfolio_value.shift(1)
    
    # Annualize (assuming daily data)
    annual_turnover = turnover_ratio.mean() * 252
    
    return annual_turnover


def calculate_drawdowns(returns: pd.Series) -> Tuple[pd.Series, float, int]:
    """
    Calculate drawdowns from returns series.
    
    Parameters:
    returns (pandas.Series): Returns series
    
    Returns:
    tuple: (drawdowns, max_drawdown, max_drawdown_duration)
    """
    # Cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate drawdowns
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns / running_max) - 1
    
    # Find max drawdown
    max_drawdown = drawdowns.min()
    
    # Calculate drawdown durations
    is_in_drawdown = cum_returns < running_max
    drawdown_start = is_in_drawdown.diff().fillna(False)
    drawdown_start = drawdown_start[drawdown_start].index
    
    if len(drawdown_start) > 0:
        drawdown_periods = []
        for start in drawdown_start:
            # Find end of this drawdown period
            try:
                end = is_in_drawdown.loc[start:][~is_in_drawdown].index[0]
            except IndexError:
                # Drawdown hasn't ended yet
                end = is_in_drawdown.index[-1]
            
            # Calculate duration
            duration = len(is_in_drawdown.loc[start:end])
            drawdown_periods.append(duration)
        
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
    else:
        max_drawdown_duration = 0
    
    return drawdowns, max_drawdown, max_drawdown_duration


def calculate_rolling_performance(
    returns: pd.Series, 
    window: int = 252, 
    risk_free_rate: float = RISK_FREE_RATE
) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.
    
    Parameters:
    returns (pandas.Series): Returns series
    window (int): Rolling window size
    risk_free_rate (float): Annualized risk-free rate
    
    Returns:
    pandas.DataFrame: DataFrame with rolling metrics
    """
    # Initialize result DataFrame
    result = pd.DataFrame(index=returns.index)
    
    # Rolling return
    result['rolling_return'] = returns.rolling(window).apply(
        lambda x: (1 + x).prod() - 1
    )
    
    # Rolling volatility
    result['rolling_vol'] = returns.rolling(window).std() * np.sqrt(252)
    
    # Rolling Sharpe Ratio
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    
    result['rolling_sharpe'] = returns.rolling(window).apply(
        lambda x: (x.mean() * 252 - risk_free_rate) / (x.std() * np.sqrt(252)) 
        if x.std() > 0 else 0
    )
    
    # Rolling Sortino Ratio
    result['rolling_sortino'] = returns.rolling(window).apply(
        lambda x: (x.mean() * 252 - risk_free_rate) / (x[x < 0].std() * np.sqrt(252)) 
        if not x[x < 0].empty and x[x < 0].std() > 0 else 0
    )
    
    # Rolling Max Drawdown
    result['rolling_max_dd'] = returns.rolling(window).apply(
        lambda x: (((1 + x).cumprod()) / ((1 + x).cumprod()).cummax() - 1).min()
    )
    
    return result


def analyze_option_strategy_greeks(
    strategy_returns: pd.Series,
    underlying_returns: pd.Series,
    vix_changes: Optional[pd.Series] = None
) -> Dict[str, float]:
    """
    Analyze sensitivity of option strategy returns to various factors.
    
    Parameters:
    strategy_returns (pandas.Series): Strategy returns
    underlying_returns (pandas.Series): Underlying asset returns
    vix_changes (pandas.Series, optional): Changes in VIX index
    
    Returns:
    dict: Strategy sensitivities
    """
    # Align data
    data = pd.concat([strategy_returns, underlying_returns], axis=1).dropna()
    if data.empty:
        return {}
        
    data.columns = ['strategy', 'underlying']
    
    # Add VIX changes if available
    if vix_changes is not None:
        data = pd.concat([data, vix_changes], axis=1).dropna()
        data.columns = ['strategy', 'underlying', 'vix_change']
    
    # Calculate beta (delta equivalent)
    model = np.polyfit(data['underlying'], data['strategy'], 1)
    beta = model[0]  # Slope coefficient
    
    # Calculate gamma (curvature)
    # Add squared returns for quadratic model
    data['underlying_squared'] = data['underlying'] ** 2
    model_quad = np.polyfit(data['underlying'], data['strategy'], 2)
    gamma = model_quad[0] * 2  # Second derivative coefficient
    
    # Calculate vega equivalent if VIX data available
    vega_equivalent = None
    if 'vix_change' in data.columns:
        vega_model = np.polyfit(data['vix_change'], data['strategy'], 1)
        vega_equivalent = vega_model[0]
    
    # Calculate R-squared of the linear model
    strategy_pred = model[0] * data['underlying'] + model[1]
    ss_total = np.sum((data['strategy'] - data['strategy'].mean()) ** 2)
    ss_residual = np.sum((data['strategy'] - strategy_pred) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    
    # Return all sensitivities
    result = {
        'beta': beta,
        'gamma': gamma,
        'r_squared': r_squared
    }
    
    if vega_equivalent is not None:
        result['vega_equivalent'] = vega_equivalent
    
    return result


def calculate_win_rate(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate win rate and related metrics from trades.
    
    Parameters:
    trades (list): List of trade dictionaries
    
    Returns:
    dict: Win rate metrics
    """
    if not trades:
        return {
            'win_rate': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'win_loss_ratio': 0,
            'largest_win': 0,
            'largest_loss': 0
        }
    
    # Filter to close trades only
    close_trades = [t for t in trades if 'close_date' in t or 'pnl' in t]
    
    # Get trade PnLs
    pnls = []
    for trade in close_trades:
        if 'pnl' in trade:
            pnls.append(trade['pnl'])
        elif 'close_price' in trade and 'open_price' in trade:
            # Calculate PnL based on prices
            direction = 1 if trade.get('type', '').startswith('buy') else -1
            pnl = direction * (trade['close_price'] - trade['open_price'])
            pnls.append(pnl)
    
    if not pnls:
        return {
            'win_rate': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'win_loss_ratio': 0,
            'largest_win': 0,
            'largest_loss': 0
        }
    
    # Calculate win rate
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    
    win_rate = len(wins) / len(pnls) if pnls else 0
    
    # Calculate other metrics
    total_wins = sum(wins)
    total_losses = sum(losses)
    
    profit_factor = abs(total_wins / total_losses) if total_losses else float('inf')
    
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss else float('inf')
    
    largest_win = max(wins) if wins else 0
    largest_loss = min(losses) if losses else 0
    
    return {
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'largest_win': largest_win,
        'largest_loss': largest_loss
    }