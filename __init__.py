# enhanced_options_strategy - A comprehensive framework for options strategy analysis
"""
Enhanced Options Strategy

A modular package for analyzing and optimizing options strategies,
particularly covered calls, across different market regimes and risk levels.

This package provides tools for market regime detection, volatility forecasting,
options analysis, and strategy optimization with backtest capabilities.
"""

__version__ = '0.1.0'
__author__ = 'Enhanced Quant Framework'

# Import main components for easier access
from analysis.market_regime import MarketRegimeAnalyzer
from analysis.volatility import calculate_vol_forecast
from analysis.covered_call import CoveredCallStrategy, CoveredCallOptimizer
from analysis.roll_strategy import RollStrategyOptimizer
from optimization.strategy_optimizer import run_enhanced_quant_analysis
from backtest.covered_call_backtest import backtest_covered_call

# Package metadata
PACKAGE_METADATA = {
    'name': 'enhanced_options_strategy',
    'version': __version__,
    'description': 'Enhanced Options Strategy Analysis Framework',
    'requires': [
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'scikit-learn',
        'statsmodels',
        'yfinance',
        'fpdf'
    ],
    'recommended': [
        'arch',      # For GARCH models
        'hmmlearn',  # For HMM market regime analysis
        'tensorflow',  # For advanced ML models
        'pytorch',   # For advanced ML models
        'plotly',    # For interactive visualizations
    ]
}