# data/processor.py
"""
Data processing module for options strategy analysis.

This module provides functions for processing and transforming stock price data
and options data for further analysis.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Constants
RISK_FREE_RATE = 0.02  # Default risk-free rate (2%)


def calculate_returns(
    prices: pd.Series, 
    method: str = 'simple'
) -> pd.Series:
    """
    Calculate returns from a price series.
    
    Parameters:
    prices (pandas.Series): Price time series
    method (str): 'simple' or 'log' for return calculation method
    
    Returns:
    pandas.Series: Returns series
    """
    if method == 'log':
        return np.log(prices / prices.shift(1)).dropna()
    else:  # Default to simple returns
        return (prices / prices.shift(1) - 1).dropna()


def calculate_rolling_statistics(
    returns: pd.Series, 
    windows: List[int] = [21, 63, 252]
) -> pd.DataFrame:
    """
    Calculate rolling statistics for returns.
    
    Parameters:
    returns (pandas.Series): Returns series
    windows (list): List of rolling windows to calculate
    
    Returns:
    pandas.DataFrame: DataFrame with rolling statistics
    """
    result = pd.DataFrame(index=returns.index)
    
    for window in windows:
        # Rolling mean (annualized)
        result[f'mean_{window}d'] = returns.rolling(window).mean() * 252
        
        # Rolling volatility (annualized)
        result[f'vol_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Rolling Sharpe ratio
        result[f'sharpe_{window}d'] = (
            (result[f'mean_{window}d'] - RISK_FREE_RATE) /
            result[f'vol_{window}d']
        )
        
        # Rolling skewness
        result[f'skew_{window}d'] = returns.rolling(window).skew()
        
        # Rolling kurtosis
        result[f'kurt_{window}d'] = returns.rolling(window).kurt()
    
    return result


def prepare_option_chain_for_analysis(
    chain: List[Dict[str, Any]], 
    current_price: float
) -> pd.DataFrame:
    """
    Prepare option chain data for analysis.
    
    Parameters:
    chain (list): List of option dictionaries from API
    current_price (float): Current price of the underlying
    
    Returns:
    pandas.DataFrame: Processed option chain data
    """
    if not chain:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(chain)
    
    # Calculate additional metrics
    if 'strike' in df.columns:
        # Moneyness
        df['moneyness'] = df['strike'] / current_price
        df['log_moneyness'] = np.log(df['moneyness'])
        
        # ITM/OTM classification
        if 'option_type' in df.columns:
            df['itm'] = (df['option_type'] == 'call') & (current_price > df['strike']) | \
                        (df['option_type'] == 'put') & (current_price < df['strike'])
        
        # Premium as percentage
        if 'mid' in df.columns:
            df['premium_pct'] = df['mid'] / current_price * 100
    
    return df


def align_price_data(
    symbols_data: Dict[str, Dict[str, pd.DataFrame]], 
    freq: str = 'daily'
) -> pd.DataFrame:
    """
    Align price data for multiple symbols.
    
    Parameters:
    symbols_data (dict): Dictionary of symbol data from fetcher
    freq (str): 'daily' or 'weekly' frequency
    
    Returns:
    pandas.DataFrame: DataFrame with aligned price data
    """
    aligned_prices = pd.DataFrame()
    
    for symbol, data in symbols_data.items():
        key = f"{freq}_data"
        if key in data and not data[key].empty:
            # Extract closing prices
            prices = data[key]['Close']
            aligned_prices[symbol] = prices
    
    # Align on dates (forward fill missing values up to 5 periods)
    aligned_prices = aligned_prices.ffill(limit=5)
    
    return aligned_prices


def calculate_correlation_matrix(
    returns: pd.DataFrame, 
    min_periods: int = 30
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate correlation and covariance matrices from returns.
    
    Parameters:
    returns (pandas.DataFrame): Returns for multiple symbols
    min_periods (int): Minimum number of overlapping periods required
    
    Returns:
    tuple: (correlation_matrix, covariance_matrix)
    """
    # Calculate correlations
    correlation_matrix = returns.corr(min_periods=min_periods)
    
    # Calculate covariance
    covariance_matrix = returns.cov(min_periods=min_periods)
    
    return correlation_matrix, covariance_matrix


def apply_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply technical indicators to price data.
    
    Parameters:
    df (pandas.DataFrame): Price DataFrame with OHLC data
    
    Returns:
    pandas.DataFrame: DataFrame with added indicators
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Price-based indicators
    if all(col in result.columns for col in ['Open', 'High', 'Low', 'Close']):
        # Daily returns
        result['return_1d'] = result['Close'].pct_change(1)
        
        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            result[f'ma_{window}'] = result['Close'].rolling(window).mean()
        
        # Relative to moving averages
        result['rel_ma_50'] = result['Close'] / result['ma_50'] - 1
        result['rel_ma_200'] = result['Close'] / result['ma_200'] - 1
        
        # Bollinger Bands (20-day, 2 standard deviations)
        result['bb_middle'] = result['Close'].rolling(20).mean()
        result['bb_std'] = result['Close'].rolling(20).std()
        result['bb_upper'] = result['bb_middle'] + 2 * result['bb_std']
        result['bb_lower'] = result['bb_middle'] - 2 * result['bb_std']
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        
        # True Range and ATR
        result['tr'] = np.maximum(
            result['High'] - result['Low'],
            np.maximum(
                abs(result['High'] - result['Close'].shift(1)),
                abs(result['Low'] - result['Close'].shift(1))
            )
        )
        result['atr_14'] = result['tr'].rolling(14).mean()
        
        # Price volatility
        for window in [5, 10, 20, 60]:
            result[f'volatility_{window}d'] = result['return_1d'].rolling(window).std() * np.sqrt(252)
    
    # Volume-based indicators (if available)
    if 'Volume' in result.columns:
        # Volume moving average
        result['vol_ma_20'] = result['Volume'].rolling(20).mean()
        
        # Relative volume
        result['rel_volume'] = result['Volume'] / result['vol_ma_20']
        
        # On-balance volume (OBV)
        result['obv'] = (result['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * result['Volume']).cumsum()
    
    return result


def calculate_implied_metrics(
    option_chain: pd.DataFrame,
    current_price: float,
    days_to_expiry: int,
    risk_free_rate: float = RISK_FREE_RATE
) -> pd.DataFrame:
    """
    Calculate implied metrics for option chain.
    
    Parameters:
    option_chain (pandas.DataFrame): Option chain data
    current_price (float): Current price of the underlying
    days_to_expiry (int): Days to expiry
    risk_free_rate (float): Risk-free rate
    
    Returns:
    pandas.DataFrame: DataFrame with added implied metrics
    """
    if option_chain.empty:
        return pd.DataFrame()
    
    # Make a copy
    chain = option_chain.copy()
    
    # Time to expiry in years
    t = days_to_expiry / 365
    
    # For each option, calculate Black-Scholes implied metrics
    for i, row in chain.iterrows():
        try:
            strike = row['strike']
            vol = row['implied_vol']
            
            # Calculate d1 and d2
            d1 = (np.log(current_price/strike) + (risk_free_rate + 0.5 * vol**2) * t) / (vol * np.sqrt(t))
            d2 = d1 - vol * np.sqrt(t)
            
            # Calculate option greeks
            if row['option_type'] == 'call':
                # Delta
                chain.at[i, 'delta'] = norm.cdf(d1)
                
                # Gamma (same for calls and puts)
                chain.at[i, 'gamma'] = norm.pdf(d1) / (current_price * vol * np.sqrt(t))
                
                # Theta (time decay)
                chain.at[i, 'theta'] = (
                    -(current_price * norm.pdf(d1) * vol) / (2 * np.sqrt(t)) -
                    risk_free_rate * strike * np.exp(-risk_free_rate * t) * norm.cdf(d2)
                ) / 365  # Daily theta
                
                # Vega (sensitivity to volatility)
                chain.at[i, 'vega'] = current_price * np.sqrt(t) * norm.pdf(d1) / 100  # For 1% vol change
                
                # Probability metrics
                chain.at[i, 'prob_itm'] = norm.cdf(d1)
                chain.at[i, 'prob_otm'] = 1 - chain.at[i, 'prob_itm']
                
                # Probability of touch (reaches strike before expiry)
                chain.at[i, 'prob_touch'] = np.exp(
                    (-2 * np.log(current_price/strike) * np.log(current_price/strike)) / 
                    (vol**2 * t)
                )
                
            else:  # Put
                # Delta
                chain.at[i, 'delta'] = norm.cdf(d1) - 1
                
                # Gamma (same for calls and puts)
                chain.at[i, 'gamma'] = norm.pdf(d1) / (current_price * vol * np.sqrt(t))
                
                # Theta (time decay)
                chain.at[i, 'theta'] = (
                    -(current_price * norm.pdf(d1) * vol) / (2 * np.sqrt(t)) +
                    risk_free_rate * strike * np.exp(-risk_free_rate * t) * norm.cdf(-d2)
                ) / 365  # Daily theta
                
                # Vega (sensitivity to volatility)
                chain.at[i, 'vega'] = current_price * np.sqrt(t) * norm.pdf(d1) / 100  # For 1% vol change
                
                # Probability metrics
                chain.at[i, 'prob_itm'] = norm.cdf(-d2)
                chain.at[i, 'prob_otm'] = 1 - chain.at[i, 'prob_itm']
                
                # Probability of touch (reaches strike before expiry)
                chain.at[i, 'prob_touch'] = np.exp(
                    (-2 * np.log(current_price/strike) * np.log(current_price/strike)) / 
                    (vol**2 * t)
                )
        
        except Exception as e:
            logger.warning(f"Error calculating implied metrics for strike {row.get('strike', 'N/A')}: {e}")
    
    return chain


def find_optimal_strikes(
    option_chain: pd.DataFrame,
    current_price: float,
    volatility: float,
    target_deltas: List[float] = [0.20, 0.25, 0.30, 0.35, 0.40],
    days_to_expiry: int = 30
) -> Dict[float, Dict[str, Any]]:
    """
    Find optimal strikes for covered call strategy based on target deltas.
    
    Parameters:
    option_chain (pandas.DataFrame): Option chain data
    current_price (float): Current price of the underlying
    volatility (float): Volatility forecast
    target_deltas (list): List of target deltas
    days_to_expiry (int): Days to expiry
    
    Returns:
    dict: Optimal strikes for each target delta
    """
    result = {}
    
    # Check if option chain has the necessary data
    if option_chain.empty or 'delta' not in option_chain.columns:
        # Generate theoretical options instead
        t = days_to_expiry / 365
        for delta_target in target_deltas:
            try:
                # Approximate strike price for target delta using Black-Scholes
                d1_for_delta = norm.ppf(delta_target)
                log_moneyness = d1_for_delta * volatility * np.sqrt(t) - (RISK_FREE_RATE + volatility**2/2) * t
                strike = current_price * np.exp(-log_moneyness)
                
                # Calculate option premium
                d1 = (np.log(current_price/strike) + (RISK_FREE_RATE + volatility**2/2) * t) / (volatility * np.sqrt(t))
                d2 = d1 - volatility * np.sqrt(t)
                premium = current_price * norm.cdf(d1) - strike * np.exp(-RISK_FREE_RATE * t) * norm.cdf(d2)
                
                # Calculate greeks and other metrics
                upside_cap = (strike - current_price) / current_price
                premium_yield = premium / current_price
                annualized_return = premium_yield * (365 / days_to_expiry)
                
                result[delta_target] = {
                    'strike': strike,
                    'premium': premium,
                    'delta': delta_target,
                    'upside_cap': upside_cap,
                    'premium_yield': premium_yield,
                    'annualized_return': annualized_return,
                    'is_theoretical': True
                }
            except Exception as e:
                logger.warning(f"Error calculating theoretical option for delta {delta_target}: {e}")
    
    else:
        # Use real option chain data
        for delta_target in target_deltas:
            # Find closest option to target delta
            option_chain['delta_diff'] = abs(option_chain['delta'] - delta_target)
            closest_option = option_chain.loc[option_chain['delta_diff'].idxmin()]
            
            # Extract key metrics
            strike = closest_option['strike']
            premium = closest_option['mid']
            delta = closest_option['delta']
            
            # Calculate derived metrics
            upside_cap = (strike - current_price) / current_price
            premium_yield = premium / current_price
            annualized_return = premium_yield * (365 / days_to_expiry)
            
            result[delta_target] = {
                'strike': strike,
                'premium': premium,
                'delta': delta,
                'gamma': closest_option.get('gamma', None),
                'theta': closest_option.get('theta', None),
                'vega': closest_option.get('vega', None),
                'prob_itm': closest_option.get('prob_itm', None),
                'prob_otm': closest_option.get('prob_otm', None),
                'upside_cap': upside_cap,
                'premium_yield': premium_yield,
                'annualized_return': annualized_return,
                'is_theoretical': False
            }
    
    return result