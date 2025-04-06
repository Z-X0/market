# analysis/market_regime.py
"""
Market Regime Analysis Module.

This module implements the Hidden Markov Model (HMM) for market regime detection
and provides tools for analyzing market regimes.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats

logger = logging.getLogger(__name__)

# Try to import optional HMM module
try:
    from hmmlearn import hmm
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False
    logger.warning("hmmlearn not available, using simplified regime detection")


class MarketRegimeAnalyzer:
    """
    Analyze market regimes using Hidden Markov Models.
    
    This class implements a regime detection algorithm based on Hidden Markov Models 
    to identify distinct market states with different risk/return characteristics.
    """
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        n_regimes: int = 7, 
        feature_columns: Optional[List[str]] = None,
        use_hmm: bool = True
    ):
        """
        Initialize the analyzer.
        
        Parameters:
        data (pandas.DataFrame): Price data with OHLC and optional indicators
        n_regimes (int): Number of regimes to detect
        feature_columns (list, optional): Columns to use as features
        use_hmm (bool): Whether to use HMM or simplified detection
        """
        self.data = data
        self.n_regimes = n_regimes
        self.use_hmm = use_hmm and HAS_HMMLEARN
        
        # Select features or calculate default ones
        if feature_columns is not None:
            self.feature_columns = [col for col in feature_columns if col in data.columns]
            if not self.feature_columns:
                logger.warning("None of the specified feature columns found in data")
                self._calculate_default_features()
        else:
            self._calculate_default_features()
        
        # Initialize regime and valid index data
        self.regimes = np.array([])
        self.valid_idx = pd.DatetimeIndex([])
        
        # Calculate regime states
        self._detect_regimes()
    
    def _calculate_default_features(self) -> None:
        """Calculate default features if none are provided."""
        self.feature_columns = []
        
        # Add returns feature
        if 'Close' in self.data.columns:
            self.data['returns'] = self.data['Close'].pct_change()
            self.feature_columns.append('returns')
        
            # Add volatility feature
            self.data['rolling_vol_12w'] = self.data['returns'].rolling(12).std()
            self.feature_columns.append('rolling_vol_12w')
            
            # Add longer-term returns
            self.data['ret_12w'] = self.data['Close'].pct_change(12)
            self.feature_columns.append('ret_12w')
        
        # Add volume feature if available
        if 'Volume' in self.data.columns:
            self.data['volume_change'] = self.data['Volume'].pct_change()
            self.feature_columns.append('volume_change')
    
    def _detect_regimes(self) -> None:
        """
        Detect market regimes using the selected method.
        """
        # Prepare the data
        features_df = self.data[self.feature_columns].copy()
        
        # Drop NaN values
        features_df = features_df.dropna()
        
        if features_df.empty:
            logger.warning("No valid data after dropping NaN values")
            return
        
        # Store valid indices
        self.valid_idx = features_df.index
        
        # Detect regimes
        if self.use_hmm:
            self._detect_regimes_hmm(features_df)
        else:
            self._detect_regimes_simple(features_df)
    
    def _detect_regimes_hmm(self, features_df: pd.DataFrame) -> None:
        """
        Detect regimes using Hidden Markov Model.
        
        Parameters:
        features_df (pandas.DataFrame): Features for regime detection
        """
        try:
            # Standardize features
            scaler = StandardScaler()
            features_standardized = scaler.fit_transform(features_df.values)
            
            # Initialize HMM
            model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                random_state=42,
                n_iter=100
            )
            
            # Fit HMM
            model.fit(features_standardized)
            
            # Predict regimes
            self.regimes = model.predict(features_standardized)
            
            # Reorder regimes from bearish to bullish
            self._reorder_regimes(features_df)
            
            logger.debug(f"Successfully detected {self.n_regimes} regimes using HMM")
        
        except Exception as e:
            logger.error(f"Error in HMM regime detection: {e}")
            # Fall back to simple detection
            self._detect_regimes_simple(features_df)
    
    def _detect_regimes_simple(self, features_df: pd.DataFrame) -> None:
        """
        Simple rule-based regime detection as fallback.
        
        Parameters:
        features_df (pandas.DataFrame): Features for regime detection
        """
        # Ensure we have the necessary features
        if 'returns' not in features_df.columns:
            if 'Close' in self.data.columns:
                features_df['returns'] = self.data.loc[features_df.index, 'Close'].pct_change()
            else:
                logger.error("Cannot perform simple regime detection without returns data")
                return
        
        if 'rolling_vol_12w' not in features_df.columns:
            features_df['rolling_vol_12w'] = features_df['returns'].rolling(12).std()
        
        # Fill any NaN values
        features_df = features_df.fillna(method='ffill')
        
        # Simple classification based on returns and volatility
        returns = features_df['returns'].values
        volatility = features_df['rolling_vol_12w'].values
        
        regimes = []
        for i in range(len(returns)):
            ret = returns[i]
            vol = volatility[i] if not np.isnan(vol) else 0.02  # Default vol
            
            # Determine regime based on returns and volatility
            if ret < -0.03 and vol > 0.03:  # Severe bearish
                regime = 0
            elif ret < -0.02 and vol > 0.025:  # Bearish
                regime = 1
            elif ret < -0.01:  # Weak bearish
                regime = 2
            elif abs(ret) <= 0.01 and vol < 0.02:  # Neutral
                regime = 3
            elif ret > 0.01 and ret <= 0.02:  # Weak bullish
                regime = 4
            elif ret > 0.02 and ret <= 0.03:  # Bullish
                regime = 5
            elif ret > 0.03:  # Strong bullish
                regime = 6
            else:
                regime = 3  # Default to neutral
            
            regimes.append(regime)
        
        self.regimes = np.array(regimes)
        logger.debug("Used simple rule-based regime detection")
    
    def _reorder_regimes(self, features_df: pd.DataFrame) -> None:
        """
        Reorder regimes from bearish to bullish.
        
        Parameters:
        features_df (pandas.DataFrame): Features for regime detection
        """
        if 'returns' in features_df.columns:
            # Calculate average return for each regime
            regime_returns = {}
            for i in range(self.n_regimes):
                mask = (self.regimes == i)
                regime_returns[i] = features_df.loc[mask, 'returns'].mean()
            
            # Sort regimes by average return
            sorted_regimes = sorted(regime_returns.items(), key=lambda x: x[1])
            regime_map = {old_regime: new_regime for new_regime, (old_regime, _) in enumerate(sorted_regimes)}
            
            # Remap regimes
            self.regimes = np.array([regime_map[r] for r in self.regimes])
    
    def get_current_regime(self) -> int:
        """
        Get the most recent detected regime.
        
        Returns:
        int: Current regime (0-6)
        """
        if not self.regimes.size:
            return 3  # Default to neutral
        
        return self.regimes[-1]
    
    def get_regime_transitions(self) -> np.ndarray:
        """
        Calculate regime transition probabilities.
        
        Returns:
        numpy.ndarray: Transition matrix
        """
        if not self.regimes.size:
            return np.eye(self.n_regimes)  # Identity matrix if no transitions
        
        # Calculate transition counts
        transitions = np.zeros((self.n_regimes, self.n_regimes))
        
        for i in range(len(self.regimes) - 1):
            from_regime = self.regimes[i]
            to_regime = self.regimes[i + 1]
            transitions[from_regime, to_regime] += 1
        
        # Convert to probabilities
        row_sums = transitions.sum(axis=1)
        transition_probs = np.divide(
            transitions, 
            row_sums[:, np.newaxis], 
            out=np.zeros_like(transitions), 
            where=row_sums[:, np.newaxis] != 0
        )
        
        return transition_probs
    
    def get_regime_stats(self) -> Dict[int, Dict[str, float]]:
        """
        Calculate statistics for each regime.
        
        Returns:
        dict: Statistics by regime
        """
        if not self.regimes.size or len(self.valid_idx) != len(self.regimes):
            return {}
        
        stats = {}
        
        # Extract returns if available
        if 'Close' in self.data.columns:
            returns = self.data.loc[self.valid_idx, 'Close'].pct_change().values
        elif 'returns' in self.data.columns:
            returns = self.data.loc[self.valid_idx, 'returns'].values
        else:
            # Can't calculate return stats without returns
            logger.warning("Cannot calculate regime stats without returns data")
            return {}
        
        for regime in range(self.n_regimes):
            regime_mask = (self.regimes == regime)
            regime_returns = returns[regime_mask]
            
            if regime_returns.size:
                # Remove NaN values
                regime_returns = regime_returns[~np.isnan(regime_returns)]
                
                if regime_returns.size:
                    mean_return = np.mean(regime_returns)
                    volatility = np.std(regime_returns)
                    sharpe = mean_return / volatility if volatility > 0 else 0
                    frequency = regime_returns.size / len(returns)
                    
                    # Calculate average duration
                    durations = []
                    current_duration = 0
                    
                    for r in self.regimes:
                        if r == regime:
                            current_duration += 1
                        elif current_duration > 0:
                            durations.append(current_duration)
                            current_duration = 0
                    
                    # Don't forget the last duration
                    if current_duration > 0:
                        durations.append(current_duration)
                    
                    avg_duration = np.mean(durations) if durations else 0
                    
                    stats[regime] = {
                        'mean_return': mean_return,
                        'volatility': volatility,
                        'sharpe': sharpe,
                        'frequency': frequency,
                        'avg_duration': avg_duration,
                        'skewness': stats.skew(regime_returns) if regime_returns.size > 2 else 0,
                        'kurtosis': stats.kurtosis(regime_returns) if regime_returns.size > 3 else 0,
                        'sample_size': regime_returns.size
                    }
        
        return stats
    
    def get_regime_series(self) -> pd.Series:
        """
        Get regime classifications as a time series.
        
        Returns:
        pandas.Series: Regime series
        """
        if not self.regimes.size or len(self.valid_idx) != len(self.regimes):
            return pd.Series()
        
        return pd.Series(self.regimes, index=self.valid_idx)
    
    def predict_next_regime(self) -> Tuple[np.ndarray, int]:
        """
        Predict the probability of the next regime.
        
        Returns:
        tuple: (regime_probabilities, most_likely_regime)
        """
        if not self.regimes.size:
            # Default to equal probabilities
            probs = np.ones(self.n_regimes) / self.n_regimes
            return probs, 3  # Default to neutral
        
        # Get transition matrix
        transition_matrix = self.get_regime_transitions()
        
        # Get current regime
        current_regime = self.get_current_regime()
        
        # Get probabilities for next regime
        next_regime_probs = transition_matrix[current_regime, :]
        
        # Most likely next regime
        most_likely_next_regime = np.argmax(next_regime_probs)
        
        return next_regime_probs, most_likely_next_regime
    
    def get_regime_name(self, regime: int) -> str:
        """
        Get the name of a specific regime.
        
        Parameters:
        regime (int): Regime number
        
        Returns:
        str: Regime name
        """
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
    
    def generate_regime_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the current market regime.
        
        Returns:
        dict: Summary of regime analysis
        """
        current_regime = self.get_current_regime()
        regime_name = self.get_regime_name(current_regime)
        regime_stats = self.get_regime_stats()
        
        # Get transition probabilities
        transition_matrix = self.get_regime_transitions()
        
        # Predict next regime
        next_regime_probs, most_likely_next = self.predict_next_regime()
        next_regime_name = self.get_regime_name(most_likely_next)
        
        # Create summary
        summary = {
            'current_regime': current_regime,
            'regime_name': regime_name,
            'regime_stats': regime_stats.get(current_regime, {}),
            'transition_matrix': transition_matrix.tolist(),
            'next_regime_probabilities': next_regime_probs.tolist(),
            'most_likely_next_regime': {
                'regime': most_likely_next,
                'name': next_regime_name,
                'probability': next_regime_probs[most_likely_next]
            }
        }
        
        return summary