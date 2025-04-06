# reporting/metrics_formatter.py
"""
Metrics Formatter Module.

This module provides utility functions for formatting and presenting
metrics and results in a clear and readable way for reports and dashboards.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def format_currency(value: float, decimals: int = 2, include_sign: bool = False) -> str:
    """
    Format a value as currency.
    
    Parameters:
    value (float): Value to format
    decimals (int): Number of decimal places
    include_sign (bool): Whether to include + sign for positive values
    
    Returns:
    str: Formatted currency string
    """
    if np.isnan(value):
        return "$—"
    
    sign = "+" if value > 0 and include_sign else ""
    return f"{sign}${abs(value):,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2, include_sign: bool = False) -> str:
    """
    Format a value as percentage.
    
    Parameters:
    value (float): Value to format
    decimals (int): Number of decimal places
    include_sign (bool): Whether to include + sign for positive values
    
    Returns:
    str: Formatted percentage string
    """
    if np.isnan(value):
        return "—%"
    
    sign = "+" if value > 0 and include_sign else ""
    return f"{sign}{value * 100:.{decimals}f}%"


def format_option_metrics(
    metrics: Dict[str, Any]
) -> Dict[str, str]:
    """
    Format option metrics for display.
    
    Parameters:
    metrics (dict): Option metrics
    
    Returns:
    dict: Formatted metrics
    """
    formatted = {}
    
    for key, value in metrics.items():
        if key in ['strike', 'call_price', 'premium', 'upside_cap']:
            formatted[key] = format_currency(value)
        elif key in ['delta', 'gamma', 'vega', 'theta', 'rho']:
            formatted[key] = f"{value:.4f}"
        elif key in ['annualized_return', 'upside_potential']:
            formatted[key] = format_percentage(value / 100)  # Already percentage
        elif key in ['probabilities']:
            if isinstance(value, dict):
                formatted[key] = {k: format_percentage(v) for k, v in value.items()}
            else:
                formatted[key] = value
        else:
            formatted[key] = value
    
    return formatted


def format_backtest_metrics(
    backtest: Dict[str, Any]
) -> Dict[str, str]:
    """
    Format backtest metrics for display.
    
    Parameters:
    backtest (dict): Backtest results
    
    Returns:
    dict: Formatted metrics
    """
    formatted = {}
    
    # Format basic metrics
    for key, value in backtest.items():
        if key in ['total_return', 'annualized_return', 'max_drawdown']:
            formatted[key] = format_percentage(value, include_sign=True)
        elif key in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']:
            formatted[key] = f"{value:.2f}"
        elif key in ['start_date', 'end_date']:
            if value:
                try:
                    formatted[key] = value.strftime('%Y-%m-%d')
                except:
                    formatted[key] = str(value)
            else:
                formatted[key] = "N/A"
        elif key in ['trades', 'daily_cumulative_pnl', 'daily_index']:
            # Skip arrays
            pass
        else:
            formatted[key] = value
    
    return formatted


def format_regime_metrics(
    metrics_by_regime: Dict[int, Dict[str, float]]
) -> Dict[int, Dict[str, str]]:
    """
    Format regime-specific metrics for display.
    
    Parameters:
    metrics_by_regime (dict): Metrics by regime
    
    Returns:
    dict: Formatted metrics by regime
    """
    formatted = {}
    
    # Get regime names
    regime_names = {
        0: "Severe Bearish",
        1: "Bearish",
        2: "Weak Bearish",
        3: "Neutral",
        4: "Weak Bullish",
        5: "Bullish",
        6: "Strong Bullish"
    }
    
    for regime, metrics in metrics_by_regime.items():
        formatted[regime] = {
            'name': regime_names.get(regime, f"Regime {regime}")
        }
        
        for key, value in metrics.items():
            if key in ['mean_return', 'volatility', 'frequency']:
                formatted[regime][key] = format_percentage(value)
            elif key in ['sharpe', 'skewness', 'kurtosis']:
                formatted[regime][key] = f"{value:.2f}"
            elif key in ['avg_duration', 'sample_size']:
                formatted[regime][key] = f"{value:.1f}"
            else:
                formatted[regime][key] = value
    
    return formatted


def format_vol_forecast(
    vol_forecast: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Format volatility forecast metrics for display.
    
    Parameters:
    vol_forecast (dict): Volatility forecast
    
    Returns:
    dict: Formatted forecast
    """
    formatted = {
        'forecast': format_percentage(vol_forecast.get('forecast', 0)),
        'model_breakdown': {}
    }
    
    # Format model breakdown
    model_breakdown = vol_forecast.get('model_breakdown', {})
    for model, value in model_breakdown.items():
        formatted['model_breakdown'][model] = format_percentage(value)
    
    # Format term structure
    term_structure = vol_forecast.get('term_structure', {})
    formatted['term_structure'] = {
        period: format_percentage(value)
        for period, value in term_structure.items()
    }
    
    return formatted


def format_risk_metrics(
    risk_metrics: Dict[str, Any]
) -> Dict[str, str]:
    """
    Format risk metrics for display.
    
    Parameters:
    risk_metrics (dict): Risk metrics
    
    Returns:
    dict: Formatted metrics
    """
    formatted = {}
    
    for key, value in risk_metrics.items():
        if key in ['position_value', 'value_at_risk', 'expected_shortfall', 'max_loss']:
            formatted[key] = format_currency(value)
        elif key in ['portfolio_volatility', 'diversification_ratio']:
            formatted[key] = format_percentage(value)
        elif key in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']:
            formatted[key] = f"{value:.2f}"
        else:
            formatted[key] = value
    
    return formatted


def create_metrics_table(
    metrics: Dict[str, Any], 
    categories: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create a metrics table for reporting.
    
    Parameters:
    metrics (dict): Metrics to include
    categories (list, optional): Categories to include
    
    Returns:
    pandas.DataFrame: Metrics table
    """
    # Define default categories
    if categories is None:
        categories = [
            'Performance', 'Risk', 'Return', 'Volatility', 
            'Options', 'Strategy', 'Portfolio'
        ]
    
    # Define metrics by category
    category_metrics = {
        'Performance': [
            'total_return', 'annualized_return', 'alpha', 
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio'
        ],
        'Risk': [
            'max_drawdown', 'volatility', 'var_95', 'cvar_95', 
            'value_at_risk', 'expected_shortfall', 'beta'
        ],
        'Return': [
            'mean_return', 'median_return', 'best_return', 
            'worst_return', 'skewness', 'kurtosis'
        ],
        'Volatility': [
            'forecast', 'historical', 'ewma', 'garch', 'ensemble'
        ],
        'Options': [
            'premium', 'delta', 'gamma', 'vega', 'theta', 
            'annualized_return', 'upside_potential'
        ],
        'Strategy': [
            'win_rate', 'profit_factor', 'avg_win', 'avg_loss', 
            'avg_duration', 'turnover'
        ],
        'Portfolio': [
            'position_value', 'weight', 'contracts', 'shares'
        ]
    }
    
    # Create table data
    table_data = []
    
    for category in categories:
        category_items = category_metrics.get(category, [])
        
        for metric in category_items:
            if metric in metrics:
                value = metrics[metric]
                
                # Format value if it's a number
                if isinstance(value, (int, float)):
                    if metric in ['total_return', 'annualized_return', 'alpha', 'max_drawdown']:
                        value = format_percentage(value, include_sign=True)
                    elif metric in ['volatility', 'var_95', 'cvar_95', 'forecast']:
                        value = format_percentage(value)
                    elif metric in ['position_value', 'value_at_risk', 'expected_shortfall']:
                        value = format_currency(value)
                    elif metric in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'beta']:
                        value = f"{value:.2f}"
                
                table_data.append([category, metric, value])
    
    # Create DataFrame
    if table_data:
        return pd.DataFrame(table_data, columns=['Category', 'Metric', 'Value'])
    else:
        return pd.DataFrame(columns=['Category', 'Metric', 'Value'])


def create_comparison_table(
    results: Dict[str, Dict[str, Any]], 
    metrics: List[str] = None
) -> pd.DataFrame:
    """
    Create a comparison table for multiple strategies or scenarios.
    
    Parameters:
    results (dict): Results by strategy/scenario
    metrics (list, optional): Metrics to include
    
    Returns:
    pandas.DataFrame: Comparison table
    """
    if metrics is None:
        metrics = [
            'total_return', 'annualized_return', 'max_drawdown', 
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'win_rate', 'profit_factor'
        ]
    
    # Create table data
    table_data = []
    
    for strategy, result in results.items():
        row = [strategy]
        
        for metric in metrics:
            if metric in result:
                value = result[metric]
                
                # Format value if it's a number
                if isinstance(value, (int, float)):
                    if metric in ['total_return', 'annualized_return', 'max_drawdown']:
                        value = format_percentage(value, include_sign=True)
                    elif metric in ['win_rate']:
                        value = format_percentage(value)
                    elif metric in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']:
                        value = f"{value:.2f}"
                    elif metric in ['profit_factor']:
                        value = f"{value:.2f}"
                
                row.append(value)
            else:
                row.append("N/A")
        
        table_data.append(row)
    
    # Create DataFrame
    if table_data:
        return pd.DataFrame(table_data, columns=['Strategy'] + metrics)
    else:
        return pd.DataFrame(columns=['Strategy'] + metrics)