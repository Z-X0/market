# analysis/volatility.py
"""
Volatility Analysis Module.

This module provides tools for analyzing and forecasting volatility
using various models including EWMA, GARCH, and ensemble methods.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Try to import optional ARCH package
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    logger.warning("arch package not available, GARCH models disabled")


def safe_sqrt(x):
    """
    Safe square root function that handles negative numbers and NaN values.
    
    Parameters:
    x: Input value or array
    
    Returns:
    Square root of x, with safety checks
    """
    if isinstance(x, np.ndarray):
        # Replace negative values with 0 before sqrt
        return np.sqrt(np.maximum(x, 0))
    elif x is None or np.isnan(x):
        return np.nan
    else:
        return np.sqrt(max(0, x))


def preprocess_returns_for_modeling(returns: pd.Series) -> pd.Series:
    """
    Preprocess returns data for statistical modeling, applying appropriate scaling.
    
    Parameters:
    returns (pandas.Series): Returns series
    
    Returns:
    pandas.Series: Processed returns suitable for modeling
    """
    if returns.empty:
        return returns
    
    # Drop NaN values
    clean_returns = returns.dropna()
    
    # Calculate standard deviation to check scale
    std_dev = clean_returns.std()
    
    # Apply appropriate scaling based on magnitude
    if std_dev < 0.01:
        # Very small values (common for daily returns) - scale up by 100
        logger.debug(f"Scaling up returns by factor of 100 (original std: {std_dev:.6f})")
        return clean_returns * 100
    elif std_dev > 10:
        # Very large values - scale down by 100
        logger.debug(f"Scaling down returns by factor of 100 (original std: {std_dev:.6f})")
        return clean_returns / 100
    else:
        # Values in a reasonable range
        return clean_returns


def calculate_historical_volatility(
    returns: pd.Series, 
    window: int = 21, 
    annualize: bool = True
) -> pd.Series:
    """
    Calculate historical volatility using rolling standard deviation.
    
    Parameters:
    returns (pandas.Series): Returns series
    window (int): Window size for rolling calculation
    annualize (bool): Whether to annualize the volatility
    
    Returns:
    pandas.Series: Historical volatility
    """
    # Handle any NaN values
    clean_returns = returns.ffill()  # Use ffill() instead of fillna(method='ffill')
    
    # Use rolling std with minimum number of observations
    vol = clean_returns.rolling(window=window, min_periods=max(5, window//4)).std()
    
    if annualize:
        # Annualization factor based on trading days
        if isinstance(returns.index, pd.DatetimeIndex):
            # Try to determine frequency
            freq = pd.infer_freq(returns.index)
            if freq:
                if 'D' in freq:  # Daily
                    factor = np.sqrt(252)
                elif 'W' in freq:  # Weekly
                    factor = np.sqrt(52)
                elif 'M' in freq:  # Monthly
                    factor = np.sqrt(12)
                else:
                    factor = np.sqrt(252)  # Default to daily
            else:
                # Calculate average days between observations
                days_diff = returns.index.to_series().diff().dt.days.mean()
                if days_diff < 3:  # Daily
                    factor = np.sqrt(252)
                elif days_diff < 10:  # Weekly
                    factor = np.sqrt(52)
                else:  # Monthly or longer
                    factor = np.sqrt(12)
        else:
            # Default to daily
            factor = np.sqrt(252)
        
        vol = vol * factor
    
    return vol


def calculate_ewma_volatility(
    returns: pd.Series, 
    span: int = 30, 
    annualize: bool = True
) -> pd.Series:
    """
    Calculate EWMA (Exponentially Weighted Moving Average) volatility.
    
    Parameters:
    returns (pandas.Series): Returns series
    span (int): Span parameter for ewm
    annualize (bool): Whether to annualize the volatility
    
    Returns:
    pandas.Series: EWMA volatility
    """
    # Handle any NaN values
    clean_returns = returns.ffill()  # Use ffill() instead of fillna(method='ffill')
    
    # Ensure we have a minimum number of observations
    min_periods = max(5, span//4)
    vol = clean_returns.ewm(span=span, min_periods=min_periods).std()
    
    if annualize:
        # Use same logic as historical_volatility
        if isinstance(returns.index, pd.DatetimeIndex):
            freq = pd.infer_freq(returns.index)
            if freq:
                if 'D' in freq:  # Daily
                    factor = np.sqrt(252)
                elif 'W' in freq:  # Weekly
                    factor = np.sqrt(52)
                elif 'M' in freq:  # Monthly
                    factor = np.sqrt(12)
                else:
                    factor = np.sqrt(252)  # Default to daily
            else:
                days_diff = returns.index.to_series().diff().dt.days.mean()
                if days_diff < 3:  # Daily
                    factor = np.sqrt(252)
                elif days_diff < 10:  # Weekly
                    factor = np.sqrt(52)
                else:  # Monthly or longer
                    factor = np.sqrt(12)
        else:
            # Default to daily
            factor = np.sqrt(252)
        
        vol = vol * factor
    
    return vol


def calculate_garch_volatility(
    returns: pd.Series, 
    p: int = 1, 
    q: int = 1, 
    forecast_horizon: int = 1
) -> Tuple[pd.Series, float]:
    """
    Calculate GARCH volatility and forecast.
    
    Parameters:
    returns (pandas.Series): Returns series
    p (int): GARCH lag order
    q (int): ARCH lag order
    forecast_horizon (int): Number of periods to forecast
    
    Returns:
    tuple: (GARCH volatility series, forecast volatility)
    """
    if not HAS_ARCH:
        logger.warning("arch package not available, returning zero")
        return pd.Series(index=returns.index), 0.0
    
    try:
        # Preprocess returns
        processed_returns = preprocess_returns_for_modeling(returns)
        
        # Fit GARCH(p,q) model
        model = arch_model(processed_returns, p=p, q=q, mean='Zero', vol='GARCH', dist='normal', rescale=True)
        results = model.fit(disp='off')
        
        # Get conditional volatility
        # Scale back to original scale if we preprocessed
        scale_factor = 1.0
        if processed_returns.std() > 0 and returns.std() > 0:
            scale_factor = returns.std() / processed_returns.std()
        
        vol = pd.Series(np.sqrt(results.conditional_volatility) * scale_factor, 
                      index=processed_returns.index)
        
        # Forecast volatility
        forecast = results.forecast(horizon=forecast_horizon)
        forecast_vol = safe_sqrt(forecast.variance.iloc[-1, 0]) * scale_factor
        
        # Annualize if data frequency is inferred to be daily
        if isinstance(returns.index, pd.DatetimeIndex):
            freq = pd.infer_freq(returns.index)
            if freq and 'D' in freq:
                vol = vol * np.sqrt(252)
                forecast_vol = forecast_vol * np.sqrt(252)
        
        return vol, forecast_vol
    
    except Exception as e:
        logger.error(f"Error calculating GARCH volatility: {e}")
        return pd.Series(index=returns.index), 0.0


def calculate_har_volatility(
    returns: pd.Series, 
    daily_window: int = 1, 
    weekly_window: int = 5, 
    monthly_window: int = 22
) -> Tuple[pd.Series, float]:
    """
    Calculate HAR (Heterogeneous Autoregressive) volatility.
    
    Parameters:
    returns (pandas.Series): Returns series
    daily_window (int): Window for daily component
    weekly_window (int): Window for weekly component
    monthly_window (int): Window for monthly component
    
    Returns:
    tuple: (HAR volatility series, forecast volatility)
    """
    try:
        # First check if we have enough data points
        if returns is None or len(returns) < 30:
            logger.warning("Not enough data points for HAR volatility calculation")
            return pd.Series(index=returns.index if returns is not None else None), 0.0
        
        # Drop NaN values from the series
        clean_returns = returns.dropna()
        
        # Calculate realized variance (squared returns)
        realized_var = clean_returns**2
        
        # Calculate components
        daily_var = realized_var.rolling(window=daily_window, min_periods=1).mean()
        weekly_var = realized_var.rolling(window=weekly_window, min_periods=2).mean()
        monthly_var = realized_var.rolling(window=monthly_window, min_periods=5).mean()
        
        # Create features DataFrame - shift features by 1 for prediction
        X_df = pd.DataFrame({
            'daily': daily_var.shift(1),
            'weekly': weekly_var.shift(1),
            'monthly': monthly_var.shift(1)
        })
        
        # Drop any rows with NaN values
        X_df.dropna(inplace=True)
        
        # Check if we have enough data left after dropping NaNs
        if len(X_df) < 10:
            logger.warning("Not enough valid data points after NaN handling for HAR volatility")
            return pd.Series(index=returns.index), 0.0
        
        # Get corresponding y values for regression (align by index)
        valid_indices = X_df.index
        y_values = realized_var.loc[valid_indices].values
        
        # Fit linear model
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_df.values, y_values)
        
        # Calculate fitted values
        fitted_values = model.predict(X_df.values)
        fitted_series = pd.Series(fitted_values, index=valid_indices)
        
        # Calculate volatility (sqrt of variance)
        vol = pd.Series(np.sqrt(np.maximum(fitted_values, 0)), index=valid_indices)
        
        # Forecast next period
        if len(X_df) > 0:
            # Get the most recent observation
            last_values = X_df.iloc[-1:].values
            forecast_var = float(model.predict(last_values)[0])
            forecast_vol = np.sqrt(max(0, forecast_var))
        else:
            forecast_vol = 0.0
        
        # Annualize if data frequency is inferred to be daily
        if isinstance(returns.index, pd.DatetimeIndex):
            freq = pd.infer_freq(returns.index)
            if freq and 'D' in freq:
                vol = vol * np.sqrt(252)
                forecast_vol = forecast_vol * np.sqrt(252)
        
        return vol, forecast_vol
    
    except Exception as e:
        logger.error(f"Error calculating HAR volatility: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.Series(index=returns.index if returns is not None else None), 0.0


def calculate_vol_forecast(
    returns: pd.Series,
    market_regime: int = 3
) -> Dict[str, Any]:
    """
    Calculate volatility forecast using multiple models.
    
    Parameters:
    returns (pandas.Series): Returns series
    market_regime (int): Current market regime
    
    Returns:
    dict: Volatility forecast results
    """
    # Check for empty returns
    if returns.empty:
        logger.warning("Empty returns series, returning default volatility")
        return {
            'forecast': 0.2,  # Default 20% volatility
            'model_breakdown': {
                'historical': 0.2,
                'ewma': 0.2,
                'garch': 0.2,
                'har': 0.2,
                'ensemble': 0.2
            },
            'weights': {'hist': 0.25, 'ewma': 0.25, 'garch': 0.25, 'har': 0.25},
            'term_structure': {
                'short': 0.18,
                'medium': 0.2,
                'long': 0.22
            }
        }
    
    # Calculate historical volatility
    hist_vol = calculate_historical_volatility(returns)
    
    # Calculate EWMA volatility
    ewma_vol = calculate_ewma_volatility(returns)
    
    # Calculate GARCH volatility if available
    if HAS_ARCH:
        garch_vol, garch_forecast = calculate_garch_volatility(returns)
    else:
        garch_vol = pd.Series(index=returns.index)
        garch_forecast = 0.0
    
    # Calculate HAR volatility
    har_vol, har_forecast = calculate_har_volatility(returns)
    
    # Get last values for each model
    last_hist_vol = hist_vol.iloc[-1] if not hist_vol.empty else 0.0
    last_ewma_vol = ewma_vol.iloc[-1] if not ewma_vol.empty else 0.0
    last_garch_vol = garch_vol.iloc[-1] if not garch_vol.empty and not np.isnan(garch_vol.iloc[-1]) else 0.0
    last_har_vol = har_vol.iloc[-1] if not har_vol.empty and not np.isnan(har_vol.iloc[-1]) else 0.0
    
    # Handle any invalid values
    if np.isnan(last_hist_vol) or last_hist_vol <= 0:
        last_hist_vol = 0.2  # Default to 20%
    if np.isnan(last_ewma_vol) or last_ewma_vol <= 0:
        last_ewma_vol = 0.2
    if np.isnan(last_garch_vol) or last_garch_vol <= 0:
        last_garch_vol = 0.2
    if np.isnan(last_har_vol) or last_har_vol <= 0:
        last_har_vol = 0.2
    
    # Define regime-specific weights
    regime_weights = {
        0: {'hist': 0.2, 'ewma': 0.3, 'garch': 0.3, 'har': 0.2},  # Severe Bearish
        1: {'hist': 0.2, 'ewma': 0.3, 'garch': 0.3, 'har': 0.2},  # Bearish
        2: {'hist': 0.3, 'ewma': 0.3, 'garch': 0.2, 'har': 0.2},  # Weak Bearish
        3: {'hist': 0.3, 'ewma': 0.3, 'garch': 0.2, 'har': 0.2},  # Neutral
        4: {'hist': 0.4, 'ewma': 0.3, 'garch': 0.2, 'har': 0.1},  # Weak Bullish
        5: {'hist': 0.4, 'ewma': 0.3, 'garch': 0.2, 'har': 0.1},  # Bullish
        6: {'hist': 0.5, 'ewma': 0.3, 'garch': 0.1, 'har': 0.1}   # Strong Bullish
    }
    
    # Get weights for current regime
    weights = regime_weights.get(market_regime, regime_weights[3])
    
    # Calculate ensemble volatility with safety checks
    ensemble_vol = (
        weights['hist'] * last_hist_vol +
        weights['ewma'] * last_ewma_vol +
        weights['garch'] * last_garch_vol +
        weights['har'] * last_har_vol
    )
    
    # Sanity check on final volatility
    if np.isnan(ensemble_vol) or ensemble_vol <= 0.05:
        ensemble_vol = 0.2  # Default to 20% if calculation fails
    elif ensemble_vol > 1.0:
        ensemble_vol = 1.0  # Cap at 100% for extreme cases
    
    # Calculate term structure
    # Short term (1 month)
    short_term_vol = ensemble_vol * 0.9  # Typically lower
    
    # Medium term (3 months)
    medium_term_vol = ensemble_vol
    
    # Long term (6 months)
    long_term_vol = ensemble_vol * 1.1  # Typically higher
    
    # Return complete results
    return {
        'forecast': ensemble_vol,
        'model_breakdown': {
            'historical': last_hist_vol,
            'ewma': last_ewma_vol,
            'garch': last_garch_vol,
            'har': last_har_vol,
            'ensemble': ensemble_vol
        },
        'weights': weights,
        'term_structure': {
            'short': short_term_vol,
            'medium': medium_term_vol,
            'long': long_term_vol
        }
    }


def analyze_volatility_surface(
    option_chain: Dict[int, List[Dict[str, Any]]], 
    current_price: float
) -> Dict[str, Any]:
    """
    Analyze the volatility surface from option chain data.
    
    Parameters:
    option_chain (dict): Option chain by expiry
    current_price (float): Current price of the underlying
    
    Returns:
    dict: Volatility surface analysis
    """
    if not option_chain:
        return {'error': 'No option chain data provided'}
    
    # Extract implied volatilities
    surface_data = []
    
    for dte, chain in option_chain.items():
        for option in chain:
            # Skip if no implied vol or invalid
            iv = option.get('implied_vol', None)
            if iv is None or iv <= 0:
                continue
            
            # Get option details
            strike = option.get('strike', 0)
            option_type = option.get('option_type', 'call')
            
            # Skip if no strike
            if strike <= 0:
                continue
            
            # Calculate moneyness
            moneyness = strike / current_price
            
            surface_data.append({
                'dte': dte,
                't': dte / 365,  # Time in years
                'strike': strike,
                'moneyness': moneyness,
                'log_moneyness': np.log(moneyness),
                'implied_vol': iv,
                'option_type': option_type
            })
    
    if not surface_data:
        return {'error': 'No valid volatility data in option chain'}
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(surface_data)
        
        # Calculate ATM volatility term structure
        atm_options = df[(df['moneyness'] >= 0.95) & (df['moneyness'] <= 1.05)]
        term_structure = atm_options.groupby('dte')['implied_vol'].mean().to_dict()
        
        # Calculate skew metrics by expiry
        skew_metrics = {}
        
        for dte, group in df.groupby('dte'):
            # Need sufficient data points
            if len(group) < 5:
                continue
            
            # Calculate skew
            X = group['log_moneyness'].values.reshape(-1, 1)
            y = group['implied_vol'].values
            
            # Linear regression for slope
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            slope = model.coef_[0]
            r2 = model.score(X, y)
            
            # Calculate smile components
            put_wing = group[group['moneyness'] < 0.95]['implied_vol'].mean()
            atm_vol = group[(group['moneyness'] >= 0.95) & (group['moneyness'] <= 1.05)]['implied_vol'].mean()
            call_wing = group[group['moneyness'] > 1.05]['implied_vol'].mean()
            
            # Risk reversal and butterfly
            risk_reversal = call_wing - put_wing if not (np.isnan(call_wing) or np.isnan(put_wing)) else 0
            butterfly = (call_wing + put_wing) / 2 - atm_vol if not np.isnan(atm_vol) else 0
            
            skew_metrics[int(dte)] = {
                'slope': slope,
                'r_squared': r2,
                'risk_reversal': risk_reversal,
                'butterfly': butterfly,
                'put_wing_vol': put_wing,
                'atm_vol': atm_vol,
                'call_wing_vol': call_wing,
                'num_points': len(group)
            }
        
        # Calculate forward volatilities between terms
        forward_vols = {}
        terms = sorted(term_structure.keys())
        
        for i in range(len(terms) - 1):
            t1, t2 = terms[i] / 365, terms[i+1] / 365
            v1, v2 = term_structure[terms[i]], term_structure[terms[i+1]]
            
            # Forward variance formula: (v2²·t2 - v1²·t1) / (t2 - t1)
            # Use safe calculation to avoid negative values
            forward_var = ((v2**2 * t2) - (v1**2 * t1)) / (t2 - t1)
            forward_vol = safe_sqrt(forward_var)
            
            period = f"{terms[i]}_{terms[i+1]}"
            forward_vols[period] = forward_vol
        
        # Return complete analysis
        return {
            'term_structure': {
                'spot_volatilities': term_structure,
                'forward_volatilities': forward_vols
            },
            'skew_metrics': skew_metrics,
            'surface_data': df.to_dict(orient='records')
        }
    except Exception as e:
        logger.error(f"Error analyzing volatility surface: {e}")
        return {'error': f'Analysis failed: {str(e)}'}