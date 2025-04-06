#!/usr/bin/env python3
# main.py
"""
Enhanced Options Strategy - Main execution script

This script provides the entry point for the enhanced options strategy analysis framework.
It orchestrates the workflow for analyzing covered call strategies across a portfolio of stocks.
"""

import os
import json
import logging
import argparse
import pandas as pd

from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("main")

# Import necessary modules
try:
    from data.fetcher import fetch_stock_data, fetch_option_chains
    from analysis.market_regime import MarketRegimeAnalyzer
    from analysis.volatility import calculate_vol_forecast
    from analysis.portfolio import build_portfolio_correlation, calculate_portfolio_volatility
    from optimization.strategy_optimizer import run_enhanced_quant_analysis
    from reporting.pdf_generator import generate_enhanced_pdf_report
    from backtest.covered_call_backtest import backtest_covered_call
    from reporting.chart_generator import convert_keys
except ImportError as e:
    logger.error(f"Error importing modules: {e}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced Options Strategy Analysis Framework"
    )
    parser.add_argument(
        '--portfolio', '-p', type=str, 
        help='Path to JSON file containing portfolio (format: {"symbol": shares, ...})'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='./output',
        help='Output directory for reports and data'
    )
    parser.add_argument(
        '--risk-level', '-r', type=str, default='conservative',
        choices=['conservative', 'moderate', 'aggressive'],
        help='Risk level for strategy optimization'
    )
    parser.add_argument(
        '--backtest', '-b', action='store_true',
        help='Run backtest simulation'
    )
    
    return parser.parse_args()


def load_portfolio_from_file(file_path):
    """Load portfolio from JSON file."""
    try:
        with open(file_path, 'r') as f:
            portfolio = json.load(f)
        return portfolio
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading portfolio from {file_path}: {e}")
        # Default portfolio if file loading fails
        return {
            "AAPL": 2000,
            "MSFT": 1500,
            "AMZN": 1000,
            "NVDA": 1400,
            "GOOGL": 1200
        }


def create_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Create timestamped subdirectory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir)
    
    return run_dir


def run_full_analysis(portfolio, output_dir, risk_level='conservative', run_backtest=False):
    """
    Run full covered call analysis workflow.
    
    Parameters:
    portfolio (dict): Portfolio positions {symbol: quantity}
    output_dir (str): Directory for output files
    risk_level (str): Risk level for strategy optimization
    run_backtest (bool): Whether to run backtest simulations
    
    Returns:
    tuple: (all_results, backtests, pdf_report)
    """
    # 1. Fetch data for all symbols
    logger.info("Fetching market data for portfolio...")
    symbols_data = {}
    current_prices = {}
    
    for symbol in portfolio:
        try:
            # Fetch stock data
            stock_data = fetch_stock_data(symbol)
            
            # Verify data existence with proper checks
            has_weekly_data = isinstance(stock_data.get("weekly_data"), pd.DataFrame) and not stock_data["weekly_data"].empty 
            has_daily_data = isinstance(stock_data.get("daily_data"), pd.DataFrame) and not stock_data["daily_data"].empty
                
            if not has_weekly_data or not has_daily_data:
                logger.warning(f"Empty data for {symbol}")
                continue
            
            # Get current price safely
            current_price = 0
            if has_weekly_data:
                current_price = stock_data["weekly_data"]['Close'].iloc[-1]
            
            current_prices[symbol] = current_price
            
            # Try to fetch option chains
            option_chains = fetch_option_chains(symbol)
            
            # Store all data
            symbols_data[symbol] = {
                **stock_data,
                "current_price": current_price,
                "option_chains": option_chains
            }
            
            logger.info(f"Fetched data for {symbol}, current price: {current_price:.2f}")
        
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
    
    # 2. Run enhanced quant analysis
    logger.info(f"Running strategy optimization with risk level: {risk_level}...")
    
    try:
        all_results, backtests = run_enhanced_quant_analysis(
            portfolio, symbols_data, risk_levels=[risk_level], run_backtest=run_backtest
        )
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        all_results = {risk_level: {}}
        backtests = {}
    
    # 3. Generate market analysis
    logger.info("Generating market analysis...")
    try:
        market_analysis = generate_market_analysis(all_results, current_prices)
    except Exception as e:
        logger.error(f"Error generating market analysis: {e}")
        market_analysis = {}
    
    # 4. Generate PDF report
    logger.info("Generating PDF report...")
    pdf_report = None
    try:
        # Need to import datetime to avoid error
        from datetime import datetime
        pdf_report = generate_enhanced_pdf_report(portfolio, all_results, backtests, market_analysis)
        logger.info("PDF report generation successful")
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        pdf_report = None
    
    # 5. Save results
    save_results(all_results, backtests, pdf_report, output_dir)
    
    return all_results, backtests, pdf_report


def generate_market_analysis(all_results, current_prices):
    """
    Generate market analysis for implementation plan.
    
    Parameters:
    all_results (dict): Analysis results
    current_prices (dict): Current prices
    
    Returns:
    dict: Market analysis
    """
    try:
        from analysis.portfolio import generate_market_analysis as gen_analysis
        return gen_analysis(all_results, current_prices)
    except Exception as e:
        logger.error(f"Error generating market analysis: {e}")
        # Return default market analysis data
        return {
            'market_regime': 3,  # Neutral default
            'volatility_forecast': 0.20,  # 20% default vol
            'implementation_plan': {
                'execution_timing': {},
                'risk_management': {}
            }
        }

import os
import json
import logging
from reporting.chart_generator import convert_keys  # make sure this import is present

logger = logging.getLogger(__name__)

def save_results(all_results, backtests, pdf_report, output_dir):
    """
    Save JSON, backtest JSON, and PDF report — letting FPDF write the PDF bytes directly.
    """
    # 1. JSON results
    json_fname = os.path.join(output_dir, "Enhanced_Covered_Call_Analysis.json")
    with open(json_fname, "w") as f:
        json.dump(convert_keys(all_results), f, indent=4)
    logger.info(f"Saved results to {json_fname}")

    # 2. Backtest JSON
    backtest_fname = os.path.join(output_dir, "Backtest_Results.json")
    with open(backtest_fname, "w") as f:
        json.dump(convert_keys(backtests), f, indent=4)
    logger.info(f"Saved backtest results to {backtest_fname}")

    # 3. PDF report
    if pdf_report:
        pdf_fname = os.path.join(output_dir, "Enhanced_Covered_Call_Analysis.pdf")
        try:
            # **This** is the key: write straight to file, no intermediate string.
            pdf_report.output(name=pdf_fname, dest='F')
            logger.info(f"Saved PDF report to {pdf_fname}")
        except Exception as e:
            logger.error(f"Error saving PDF report: {e}")








def main():
    """Main function to run the analysis."""
        
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Create output directory
        output_dir = create_output_directory(args.output)
        
        # Load portfolio
        if args.portfolio:
            portfolio = load_portfolio_from_file(args.portfolio)
        else:
            # Default portfolio
            portfolio = {
                "AAPL": 2000,
                "MSFT": 1500,
                "AMZN": 1000,
                "NVDA": 1400,
                "GOOGL": 1200
            }
        
        logger.info(f"Starting enhanced covered call analysis with portfolio of {len(portfolio)} symbols")
        
        # Run full analysis
        all_results, backtests, pdf_report = run_full_analysis(
            portfolio, output_dir, args.risk_level, args.backtest
        )
        
        logger.info(f"Analysis complete. Results saved to {output_dir}")
    
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()