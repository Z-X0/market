# data/utils.py
"""
Utility functions for data processing and transformation.
"""

import os
import json
import logging
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def ensure_dir_exists(directory_path: str) -> None:
    """
    Ensure directory exists, create it if it doesn't.
    
    Parameters:
    directory_path (str): Path to directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.debug(f"Created directory: {directory_path}")


def save_to_json(data: Any, file_path: str) -> None:
    """
    Save data to JSON file.
    
    Parameters:
    data (Any): Data to save
    file_path (str): Path to save file
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory:
            ensure_dir_exists(directory)
        
        # Convert special types to serializable format
        serializable_data = json_serialize(data)
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        logger.debug(f"Data saved to {file_path}")
    
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")


def load_from_json(file_path: str) -> Any:
    """
    Load data from JSON file.
    
    Parameters:
    file_path (str): Path to JSON file
    
    Returns:
    Any: Loaded data
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        logger.debug(f"Data loaded from {file_path}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return None


def json_serialize(obj: Any) -> Any:
    """
    Convert special types to JSON serializable format.
    
    Parameters:
    obj (Any): Object to convert
    
    Returns:
    Any: JSON serializable object
    """
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_serialize(item) for item in obj]
    else:
        return obj


def cache_data(data: Any, cache_dir: str, cache_key: str) -> None:
    """
    Cache data to file.
    
    Parameters:
    data (Any): Data to cache
    cache_dir (str): Cache directory
    cache_key (str): Cache key (filename)
    """
    ensure_dir_exists(cache_dir)
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")
    save_to_json(data, cache_path)


def get_cached_data(cache_dir: str, cache_key: str, max_age_hours: int = 24) -> Optional[Any]:
    """
    Get cached data if it exists and is not too old.
    
    Parameters:
    cache_dir (str): Cache directory
    cache_key (str): Cache key (filename)
    max_age_hours (int): Maximum age of cache in hours
    
    Returns:
    Any: Cached data or None if not found or too old
    """
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")
    
    if not os.path.exists(cache_path):
        return None
    
    # Check if cache is too old
    if max_age_hours > 0:
        file_age_hours = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))).total_seconds() / 3600
        if file_age_hours > max_age_hours:
            logger.debug(f"Cache for {cache_key} is too old ({file_age_hours:.1f} hours)")
            return None
    
    return load_from_json(cache_path)


def interpolate_missing_values(df: pd.DataFrame, method: str = 'linear', limit: int = 5) -> pd.DataFrame:
    """
    Interpolate missing values in a DataFrame.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with missing values
    method (str): Interpolation method
    limit (int): Maximum number of consecutive NaNs to fill
    
    Returns:
    pandas.DataFrame: DataFrame with interpolated values
    """
    return df.interpolate(method=method, limit=limit, limit_direction='both')


def resample_price_data(
    df: pd.DataFrame, 
    freq: str = 'D', 
    price_col: str = 'Close', 
    volume_col: Optional[str] = 'Volume'
) -> pd.DataFrame:
    """
    Resample price data to a different frequency.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with price data
    freq (str): Target frequency ('D' for daily, 'W' for weekly, etc.)
    price_col (str): Column name for price data
    volume_col (str, optional): Column name for volume data
    
    Returns:
    pandas.DataFrame: Resampled DataFrame
    """
    # Check if index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    # Create resampler
    resampler = df.resample(freq)
    
    # Define aggregation functions
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }
    
    # Add volume if available
    if volume_col in df.columns:
        agg_dict[volume_col] = 'sum'
    
    # Apply resampling
    resampled = resampler.agg(agg_dict)
    
    return resampled


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
        
        max_drawdown_duration = max(drawdown_periods)
    else:
        max_drawdown_duration = 0
    
    return drawdowns, max_drawdown, max_drawdown_duration


def normalize_symbol(symbol: str) -> str:
    """
    Normalize stock symbol.
    
    Parameters:
    symbol (str): Stock symbol
    
    Returns:
    str: Normalized symbol
    """
    # Remove whitespace and convert to uppercase
    normalized = symbol.strip().upper()
    
    # Handle special cases (like BRK.B -> BRK-B for some APIs)
    normalized = normalized.replace('.', '-')
    
    return normalized