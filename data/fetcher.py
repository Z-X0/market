# data/fetcher.py
"""
Data fetching module for options strategy analysis.

This module provides functions for fetching stock price data and options chain data
from various sources, with built-in retry logic and error handling.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, Union

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_RETRY_COUNT = 3
DEFAULT_INITIAL_DELAY = 1
DEFAULT_BACKOFF_FACTOR = 2


def fetch_data_with_backoff(
    fetch_function: callable, 
    max_retries: int = DEFAULT_RETRY_COUNT,
    initial_delay: int = DEFAULT_INITIAL_DELAY, 
    backoff_factor: int = DEFAULT_BACKOFF_FACTOR,
    *args, **kwargs
) -> Any:
    """
    Fetch data with exponential backoff retry strategy.
    
    Parameters:
    fetch_function (callable): Function to fetch data
    max_retries (int): Maximum number of retries
    initial_delay (int): Initial delay in seconds
    backoff_factor (int): Factor to increase delay on each retry
    *args, **kwargs: Arguments to pass to fetch_function
    
    Returns:
    Any: Result of fetch_function or None if all retries fail
    
    Raises:
    Exception: The last exception encountered after all retries
    """
    delay = initial_delay
    last_exception = None
    
    for retry in range(max_retries):
        try:
            return fetch_function(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if retry == max_retries - 1:
                logger.error(f"Max retries reached: {e}")
                raise
            
            # Log warning and retry
            logger.warning(f"Retry {retry+1}/{max_retries} after error: {e}")
            time.sleep(delay)
            delay *= backoff_factor  # Exponential backoff

    # This should not be reached due to the final raise in the loop
    return None


def get_stock_history_weekly(
    symbol: str, 
    period: str = "3y"
) -> pd.DataFrame:
    """
    Fetch weekly historical price data for a symbol.
    
    Parameters:
    symbol (str): Stock symbol
    period (str): Timeframe to fetch data for
    
    Returns:
    pandas.DataFrame: Weekly stock data
    
    Raises:
    Exception: If fetching data fails
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval="1wk")
        
        # Add some basic indicators
        if not hist.empty:
            hist['returns'] = hist['Close'].pct_change()
            hist['volatility'] = hist['returns'].rolling(12).std()
        
        return hist
    except Exception as e:
        logger.error(f"Error fetching weekly data for {symbol}: {e}")
        raise


def get_stock_history_daily(
    symbol: str, 
    period: str = "1y"
) -> pd.DataFrame:
    """
    Fetch daily historical price data for a symbol.
    
    Parameters:
    symbol (str): Stock symbol
    period (str): Timeframe to fetch data for
    
    Returns:
    pandas.DataFrame: Daily stock data
    
    Raises:
    Exception: If fetching data fails
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval="1d")
        
        # Add some basic indicators
        if not hist.empty:
            hist['returns'] = hist['Close'].pct_change()
            hist['volatility'] = hist['returns'].rolling(21).std()
        
        return hist
    except Exception as e:
        logger.error(f"Error fetching daily data for {symbol}: {e}")
        raise


def fetch_stock_data(
    symbol: str, 
    daily_period: str = "1y", 
    weekly_period: str = "3y"
) -> Dict[str, pd.DataFrame]:
    """
    Fetch all necessary stock data for a symbol (both daily and weekly).
    
    Parameters:
    symbol (str): Stock symbol
    daily_period (str): Period for daily data
    weekly_period (str): Period for weekly data
    
    Returns:
    dict: Dictionary containing daily and weekly data
    """
    result = {
        "daily_data": pd.DataFrame(),
        "weekly_data": pd.DataFrame()
    }
    
    try:
        # Fetch with retry mechanism
        result["weekly_data"] = fetch_data_with_backoff(
            get_stock_history_weekly,
            symbol=symbol,
            period=weekly_period
        )
        
        result["daily_data"] = fetch_data_with_backoff(
            get_stock_history_daily,
            symbol=symbol,
            period=daily_period
        )
        
    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")
    
    return result


def fetch_option_chains(
    symbol: str, 
    dtes: List[int] = [30, 45, 60, 90]
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Fetch option chains for a symbol for various days-to-expiry.
    
    Parameters:
    symbol (str): Stock symbol
    dtes (list): List of days to expiry to fetch
    
    Returns:
    dict: Dictionary mapping DTE to option chain data
    """
    option_chains = {}
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Get all available expirations first
        all_expirations = []
        try:
            all_expirations = ticker.options
            if not all_expirations:
                logger.warning(f"No options data available for {symbol}")
                return option_chains
                
            logger.debug(f"Available expirations for {symbol}: {all_expirations}")
        except Exception as e:
            logger.error(f"Error getting options expirations for {symbol}: {e}")
            return option_chains
            
        # Current date for DTE calculation
        current_date = datetime.now().date()
        
        # Match requested DTEs to available expirations
        for dte_target in dtes:
            target_date = current_date + timedelta(days=dte_target)
            
            # Find closest available expiration to the target date
            best_exp = None
            min_diff = float('inf')
            
            for exp_str in all_expirations:
                try:
                    exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                    diff = abs((exp_date - target_date).days)
                    
                    if diff < min_diff:
                        min_diff = diff
                        best_exp = exp_str
                except Exception as e:
                    logger.warning(f"Error parsing expiration date {exp_str}: {e}")
                    continue
            
            # If found a reasonable match (within 10 days of target)
            if best_exp and min_diff <= 10:
                try:
                    # Calculate actual DTE
                    actual_dte = (datetime.strptime(best_exp, '%Y-%m-%d').date() - current_date).days
                    
                    # Fetch option chain
                    chain = ticker.option_chain(best_exp)
                    
                    if chain and hasattr(chain, 'calls'):
                        # Process calls
                        calls_data = []
                        
                        for _, row in chain.calls.iterrows():
                            # Calculate mid price
                            bid = row.get('bid', 0)
                            ask = row.get('ask', 0)
                            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
                            
                            calls_data.append({
                                'strike': row.get('strike', 0),
                                'bid': bid,
                                'ask': ask,
                                'mid': mid,
                                'implied_vol': row.get('impliedVolatility', 0),
                                'volume': row.get('volume', 0),
                                'open_interest': row.get('openInterest', 0),
                                'option_type': 'call'
                            })
                        
                        if calls_data:
                            option_chains[actual_dte] = calls_data
                            logger.debug(f"Fetched {len(calls_data)} call options for {symbol} with DTE={actual_dte} (requested {dte_target})")
                except Exception as e:
                    logger.warning(f"Error fetching options for {symbol} with expiration {best_exp}: {e}")
        
    except Exception as e:
        logger.error(f"Error fetching option chains for {symbol}: {e}")
    
    return option_chains
    return option_chains


def fetch_market_factors_data(
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch market factors data for factor analysis.
    
    Parameters:
    start_date (str): Start date in YYYY-MM-DD format
    end_date (str): End date in YYYY-MM-DD format
    
    Returns:
    pandas.DataFrame: Market factors data
    """
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # List of ETFs to use as factors
    factor_etfs = {
        "SPY": "Market",
        "IWM": "Size", 
        "VTV": "Value",
        "VUG": "Growth",
        "QUAL": "Quality",
        "MTUM": "Momentum",
        "USMV": "LowVol"
    }
    
    factors_data = pd.DataFrame()
    
    try:
        # Fetch data for each factor ETF
        for etf, factor_name in factor_etfs.items():
            try:
                etf_data = yf.download(etf, start=start_date, end=end_date, progress=False)
                if not etf_data.empty:
                    # Calculate returns
                    returns = etf_data['Adj Close'].pct_change().dropna()
                    factors_data[factor_name] = returns
            except Exception as e:
                logger.warning(f"Error fetching factor data for {etf}: {e}")
    
    except Exception as e:
        logger.error(f"Error fetching market factors data: {e}")
    
    return factors_data


def fetch_dividend_schedule(
    symbol: str, 
    lookahead_days: int = 90
) -> Dict[str, float]:
    """
    Fetch upcoming dividend schedule for a symbol.
    
    Parameters:
    symbol (str): Stock symbol
    lookahead_days (int): Number of days to look ahead
    
    Returns:
    dict: Dictionary mapping dividend dates to amounts
    """
    dividend_schedule = {}
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Get dividend information
        try:
            # Get historical dividends
            dividends = ticker.dividends
            
            if len(dividends) > 0:  # Changed from direct emptiness check to length check
                # Get most recent dividend
                last_div = dividends.iloc[-1]
                last_div_date = dividends.index[-1]
                
                # Estimate next dividend date based on historical frequency
                if len(dividends) > 1:
                    # Calculate average time between dividends
                    div_dates = dividends.index
                    intervals = []
                    
                    for i in range(1, len(div_dates)):
                        interval = (div_dates[i] - div_dates[i-1]).days
                        intervals.append(interval)
                    
                    avg_interval = sum(intervals) / len(intervals)
                    next_div_date = last_div_date + timedelta(days=int(avg_interval))
                    
                    # Only include if it's within the lookahead period
                    if (next_div_date - datetime.now()).days <= lookahead_days:
                        dividend_schedule[next_div_date.strftime('%Y-%m-%d')] = last_div
                        
                        # If quarterly, add one more
                        if 80 <= avg_interval <= 100:
                            second_div_date = next_div_date + timedelta(days=int(avg_interval))
                            if (second_div_date - datetime.now()).days <= lookahead_days:
                                dividend_schedule[second_div_date.strftime('%Y-%m-%d')] = last_div
        
        except Exception as e:
            logger.warning(f"Error estimating dividend schedule for {symbol}: {e}")
    
    except Exception as e:
        logger.error(f"Error fetching dividend data for {symbol}: {e}")
    
    return dividend_schedule