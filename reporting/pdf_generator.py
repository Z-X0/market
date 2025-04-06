# reporting/pdf_generator.py
"""
PDF Report Generator Module.

This module provides functions for generating comprehensive PDF reports
with visualizations, tables, and analysis of options strategies.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D

# Import local modules
from reporting.chart_generator import (
    generate_regime_chart,
    generate_vol_comparison_chart,
    generate_option_chain_analysis,
    generate_backtest_pnl_chart,
    generate_regime_performance_chart,
    convert_keys
)

# Try to import FPDF
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False
    logging.warning("FPDF not installed, PDF generation disabled")

logger = logging.getLogger(__name__)

# Helper functions
def get_strategy_recommendation(regime: int) -> str:
    """Get strategy recommendation based on market regime."""
    regime_recommendations = {
        0: "very conservative",  # Severe Bearish
        1: "conservative",       # Bearish
        2: "cautious",           # Weak Bearish
        3: "balanced",           # Neutral
        4: "moderately aggressive",  # Weak Bullish
        5: "aggressive",         # Bullish
        6: "very aggressive"     # Strong Bullish
    }
    
    return regime_recommendations.get(regime, "balanced")


def get_target_yield(risk_level: str) -> float:
    """Get target annual premium yield based on risk level."""
    yields = {
        'conservative': 0.12,  # 12%
        'moderate': 0.16,      # 16%
        'aggressive': 0.20     # 20%
    }
    
    return yields.get(risk_level, 0.12)


def get_delta_range(risk_level: str) -> str:
    """Get target delta range based on risk level."""
    delta_ranges = {
        'conservative': "0.15 - 0.25",
        'moderate': "0.25 - 0.35",
        'aggressive': "0.30 - 0.45"
    }
    
    return delta_ranges.get(risk_level, "0.20 - 0.30")


def get_dte_range(risk_level: str) -> str:
    """Get target DTE range based on risk level."""
    dte_ranges = {
        'conservative': "30 - 45 days",
        'moderate': "21 - 45 days",
        'aggressive': "14 - 30 days"
    }
    
    return dte_ranges.get(risk_level, "30 - 45 days")


def get_allocation(regime: int, risk_level: str) -> float:
    """Get portfolio allocation percentage based on regime and risk level."""
    # Base allocation by risk level
    base_allocation = {
        'conservative': 0.5,
        'moderate': 0.7,
        'aggressive': 0.9
    }.get(risk_level, 0.6)
    
    # Regime adjustment
    regime_adjustment = {
        0: -0.2,  # Severe Bearish
        1: -0.1,  # Bearish
        2: -0.05, # Weak Bearish
        3: 0.0,   # Neutral
        4: 0.05,  # Weak Bullish
        5: 0.1,   # Bullish
        6: 0.2    # Strong Bullish
    }.get(regime, 0.0)
    
    # Calculate final allocation (constrained between 0.1 and 1.0)
    allocation = max(0.1, min(1.0, base_allocation + regime_adjustment))
    
    return allocation


class EnhancedQuantPDF(FPDF):
    """Enhanced PDF class with better styling and visualization capabilities."""
    
    def __init__(self, orientation='L', unit='mm', format='A4'):
        """Initialize PDF with custom settings."""
        super().__init__(orientation, unit, format)
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(10, 10, 10)
        
        # Try to add a standard font - if this fails, it will use the default font
        try:
            self.add_font('Arial', '', 'arial.ttf', uni=True)
        except Exception:
            logger.warning("Arial font not found, using default font")
    
    def create_section_header(self, title, level=1):
        """Create formatted section header with better styling."""
        if level == 1:
            self.set_font("Arial", "B", 16)
            self.set_fill_color(25, 65, 120)  # Dark blue
            self.set_text_color(255, 255, 255)  # White text
            self.cell(0, 10, title, 0, 1, "L", True)
            self.set_text_color(0, 0, 0)  # Reset text color
        elif level == 2:
            self.set_font("Arial", "B", 14)
            self.set_fill_color(65, 105, 225)  # Royal blue
            self.set_text_color(255, 255, 255)  # White text
            self.cell(0, 8, title, 0, 1, "L", True)
            self.set_text_color(0, 0, 0)  # Reset text color
        else:
            self.set_font("Arial", "I", 12)  
            self.set_fill_color(135, 206, 250)  # Light blue
            self.cell(0, 6, title, 0, 1, "L", True)
        self.ln(4)
    
    def insert_chart(self, fig, caption="", width=180):
        """Insert matplotlib figure with better styling."""
        import uuid
        fname = f"temp_{uuid.uuid4().hex}.png"
        fig.savefig(fname, dpi=300, bbox_inches="tight", facecolor='#F8F9FA')
        plt.close(fig)
        
        self.image(fname, w=width)
        os.remove(fname)
        
        if caption:
            self.set_font("Arial", "I", 9)
            self.set_fill_color(240, 240, 240)
            self.cell(width, 5, caption, 0, 1, 'C', True)
        self.ln(5)
    
    def create_info_box(self, title, content, width=180):
        """Create highlighted information box with better styling."""
        self.set_fill_color(230, 240, 250)  # Light blue background
        self.set_draw_color(25, 65, 120)    # Dark blue border
        self.set_line_width(0.3)
        self.set_font("Arial", "B", 10)
        self.cell(width, 7, title, 1, 1, "L", True)
        self.set_font("Arial", "", 9)
        self.multi_cell(width, 5, content, 1, "L", True)
        self.ln(4)
    
    def create_data_table(self, headers, data, width=180, col_widths=None):
        """Create a nicely formatted data table."""
        if col_widths is None:
            col_widths = [width / len(headers)] * len(headers)
            
        # Table header
        self.set_font("Arial", "B", 9)
        self.set_fill_color(25, 65, 120)  # Dark blue
        self.set_text_color(255, 255, 255)  # White
        
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, self.sanitize_text(header), 1, 0, "C", True)
        self.ln()
        
        # Reset text color
        self.set_text_color(0, 0, 0)
        
        # Table data
        self.set_font("Arial", "", 9)
        row_fill = False  # Start with no fill
        
        for row in data:
            if row_fill:
                self.set_fill_color(240, 240, 245)  # Light gray
            else:
                self.set_fill_color(255, 255, 255)  # White
                
            for i, cell in enumerate(row):
                # Align numeric values to right
                align = "R" if isinstance(cell, (int, float)) else "L"
                # Format percentages and numbers
                if isinstance(cell, float):
                    if abs(cell) < 0.1:
                        cell_text = f"{cell:.4f}"
                    elif "%" in str(headers[i]) or (i > 0 and "%" in str(row[0])):
                        cell_text = f"{cell:.2f}%"
                    else:
                        cell_text = f"{cell:.2f}"
                else:
                    cell_text = str(cell)
                
                self.cell(col_widths[i], 6, self.sanitize_text(cell_text), 1, 0, align, True)
            
            self.ln()
            row_fill = not row_fill  # Alternate row colors
    
    def sanitize_text(self, text):
        """Sanitize text for PDF output."""
        if isinstance(text, (int, float)):
            return str(text)
        return str(text).encode('latin-1', 'replace').decode('latin-1')


def generate_enhanced_pdf_report(
    portfolio: Dict[str, int],
    analysis_results: Dict[str, Dict[str, Any]],
    backtests: Dict[str, Dict[str, Any]],
    market_analysis: Dict[str, Any]
) -> Optional[EnhancedQuantPDF]:
    """
    Generate enhanced quantitative analysis PDF report.
    
    Parameters:
    portfolio (dict): Portfolio positions
    analysis_results (dict): Analysis results by risk level
    backtests (dict): Backtest results by symbol
    market_analysis (dict): Market analysis data
    
    Returns:
    EnhancedQuantPDF: Generated PDF report or None if FPDF not available
    """
    if not HAS_FPDF:
        logger.warning("FPDF not available, cannot generate PDF report")
        return None
    
    # Create PDF object
    pdf = EnhancedQuantPDF('L', 'mm', 'A4')
    pdf.add_page()
    
    # Cover page
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 20, "Enhanced Options Strategy Analysis", 0, 1, 'C')
    
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 15, "Covered Call Strategy Optimization", 0, 1, 'C')
    
    # Date and portfolio summary
    pdf.set_font('Arial', '', 12)
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
    pdf.cell(0, 10, f"Generated on: {current_date}", 0, 1, 'C')
    
    num_symbols = len(portfolio)
    total_shares = sum(portfolio.values())
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 15, f"Portfolio: {num_symbols} symbols, {total_shares:,} shares", 0, 1, 'C')
    
    # Add a portfolio table
    pdf.set_font('Arial', 'B', 10)
    pdf.ln(5)
    pdf.cell(0, 8, "Portfolio Composition:", 0, 1, 'L')
    
    # Create table data
    headers = ['Symbol', 'Shares', 'Weight']
    data = []
    
    # Get total portfolio value
    total_value = 0
    for symbol, shares in portfolio.items():
        # Try to get current price
        current_price = 0
        for risk_level, results in analysis_results.items():
            if 'position_results' in results and symbol in results['position_results']:
                current_price = results['position_results'][symbol].get('current_price', 0)
                break
        
        position_value = shares * current_price
        total_value += position_value
    
    # Generate portfolio data
    for symbol, shares in portfolio.items():
        # Try to get current price
        current_price = 0
        for risk_level, results in analysis_results.items():
            if 'position_results' in results and symbol in results['position_results']:
                current_price = results['position_results'][symbol].get('current_price', 0)
                break
        
        position_value = shares * current_price
        weight = position_value / total_value if total_value > 0 else 0
        
        data.append([symbol, f"{shares:,}", f"{weight:.2%}"])
    
    # Create portfolio table
    pdf.create_data_table(headers, data, width=150, col_widths=[50, 50, 50])
    
    # Table of contents
    pdf.add_page()
    pdf.create_section_header("Table of Contents", 1)
    
    toc_items = [
        "1. Executive Summary",
        "2. Market Regime Analysis",
        "3. Volatility Analysis",
        "4. Strategy Recommendations",
        "5. Position Details",
        "6. Backtest Results",
        "7. Implementation Plan"
    ]
    
    for item in toc_items:
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 8, item, 0, 1, 'L')
    
    # Executive Summary
    pdf.add_page()
    pdf.create_section_header("1. Executive Summary", 1)
    
    # Summary text
    risk_level = list(analysis_results.keys())[0] if analysis_results else "conservative"
    
    summary_text = (
        f"This report provides a comprehensive analysis of covered call strategies for "
        f"your portfolio of {num_symbols} stocks. The analysis is conducted at a "
        f"{risk_level} risk level, with considerations for current market regimes and "
        f"volatility forecasts.\n\n"
        f"Based on our analysis, we recommend implementing a covered call strategy "
        f"with the parameters outlined in this report. The strategy is designed to "
        f"generate income while managing risk through careful strike selection and "
        f"expiration timing."
    )
    
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, summary_text)
    pdf.ln(5)
    
    # Key metrics summary
    pdf.create_section_header("Key Portfolio Metrics", 2)
    
    # Create a key metrics box
    metrics_content = ""
    
    for risk_level, results in analysis_results.items():
        if 'portfolio_data' in results and 'risk_metrics' in results['portfolio_data']:
            risk_metrics = results['portfolio_data']['risk_metrics']
            metrics_content += f"Risk Level: {risk_level.capitalize()}\n"
            metrics_content += f"Total Portfolio Value: ${risk_metrics.get('total_value', 0):,.2f}\n"
            metrics_content += f"Portfolio Volatility: {risk_metrics.get('portfolio_volatility', 0):.2%}\n"
            metrics_content += f"Value at Risk (99%): ${risk_metrics.get('value_at_risk', 0):,.2f}\n"
            metrics_content += f"Expected Shortfall: ${risk_metrics.get('expected_shortfall', 0):,.2f}\n"
            metrics_content += f"Diversification Ratio: {risk_metrics.get('diversification_ratio', 0):.2%}\n"
            break
    
    if metrics_content:
        pdf.create_info_box("Portfolio Risk Metrics", metrics_content)
    
    # Market Regime Analysis
    pdf.add_page()
    pdf.create_section_header("2. Market Regime Analysis", 1)
    
    # Get first symbol to show regime chart
    if portfolio and analysis_results:
        first_symbol = list(portfolio.keys())[0]
        
        # Try to get regime analyzer
        market_analyzer = None
        weekly_data = None
        
        for risk_level, results in analysis_results.items():
            if 'position_results' in results and first_symbol in results['position_results']:
                position_results = results['position_results'][first_symbol]
                
                # Get market analyzer from the first symbol
                if 'market_analyzer' in position_results:
                    market_analyzer = position_results['market_analyzer']
                
                # Get weekly data for the first symbol
                if 'weekly_data' in position_results:
                    weekly_data = position_results['weekly_data']
                
                break
        
        # Generate and add regime chart
        fig = generate_regime_chart(first_symbol, market_analyzer, weekly_data)
        pdf.insert_chart(fig, f"Market Regime Analysis for {first_symbol}")
        
        # Add regime distribution information
        pdf.create_section_header("Market Regime Distribution", 2)
        
        # Try to get regime stats
        regime_stats = {}
        regime_names = {
            0: "Severe Bearish", 1: "Bearish", 2: "Weak Bearish", 
            3: "Neutral", 4: "Weak Bullish", 5: "Bullish", 6: "Strong Bullish"
        }
        
        for risk_level, results in analysis_results.items():
            if 'position_results' in results and first_symbol in results['position_results']:
                position_results = results['position_results'][first_symbol]
                
                if 'market_regime' in position_results and 'stats' in position_results['market_regime']:
                    regime_stats = position_results['market_regime']['stats']
                    break
        
        # Create regime stats table
        if regime_stats:
            headers = ['Regime', 'Return', 'Volatility', 'Sharpe', 'Frequency']
            data = []
            
            for regime, stats in regime_stats.items():
                regime_name = regime_names.get(int(regime), f"Regime {regime}")
                mean_return = stats.get('mean_return', 0) * 100  # Convert to percentage
                volatility = stats.get('volatility', 0) * 100  # Convert to percentage
                sharpe = stats.get('sharpe', 0)
                frequency = stats.get('frequency', 0) * 100  # Convert to percentage
                
                data.append([regime_name, f"{mean_return:.2f}%", f"{volatility:.2f}%", f"{sharpe:.2f}", f"{frequency:.2f}%"])
            
            pdf.create_data_table(headers, data)
    
    # Current Market Conditions
    pdf.create_section_header("Current Market Conditions", 2)
    
    # Get market regime and forecast from market_analysis
    current_regime = market_analysis.get('market_regime', 3)
    regime_name = regime_names.get(int(current_regime), f"Regime {current_regime}")
    vol_forecast = market_analysis.get('volatility_forecast', 0.2)
    
    conditions_text = (
        f"Current Market Regime: {regime_name}\n"
        f"Volatility Forecast: {vol_forecast:.2%}\n\n"
        f"This market environment suggests a {get_strategy_recommendation(current_regime)} "
        f"approach to covered call writing."
    )
    
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, conditions_text)
    
    # Volatility Analysis
    pdf.add_page()
    pdf.create_section_header("3. Volatility Analysis", 1)
    
    # Generate volatility comparison chart
    if analysis_results:
        risk_level = list(analysis_results.keys())[0]
        if 'position_results' in analysis_results[risk_level]:
            pos_data = analysis_results[risk_level]['position_results']
            fig = generate_vol_comparison_chart(pos_data, portfolio)
            pdf.insert_chart(fig, "Volatility Model Comparison")
    
    # Volatility Forecasts
    pdf.create_section_header("Volatility Forecasts by Symbol", 2)
    
    # Create volatility forecast table
    if analysis_results:
        headers = ['Symbol', 'Historical', 'EWMA', 'GARCH', 'Ensemble', 'Term Structure']
        data = []
        
        risk_level = list(analysis_results.keys())[0]
        if 'position_results' in analysis_results[risk_level]:
            pos_data = analysis_results[risk_level]['position_results']
            
            for symbol in portfolio:
                if symbol in pos_data and 'volatility' in pos_data[symbol]:
                    vol_data = pos_data[symbol]['volatility']
                    model_breakdown = vol_data.get('model_breakdown', {})
                    term_structure = vol_data.get('term_structure', {})
                    
                    historical = model_breakdown.get('historical', 0) * 100
                    ewma = model_breakdown.get('ewma', 0) * 100
                    garch = model_breakdown.get('garch', 0) * 100
                    ensemble = model_breakdown.get('ensemble', 0) * 100
                    
                    # Get term structure description
                    term_desc = "Short: {:.1f}%, Med: {:.1f}%, Long: {:.1f}%".format(
                        term_structure.get('short', 0) * 100,
                        term_structure.get('medium', 0) * 100,
                        term_structure.get('long', 0) * 100
                    )
                    
                    data.append([
                        symbol, 
                        f"{historical:.1f}%", 
                        f"{ewma:.1f}%", 
                        f"{garch:.1f}%", 
                        f"{ensemble:.1f}%",
                        term_desc
                    ])
            
            if data:
                pdf.create_data_table(headers, data)
    
    # Strategy Recommendations
    pdf.add_page()
    pdf.create_section_header("4. Strategy Recommendations", 1)
    
    # Overall strategy recommendations
    if analysis_results:
        risk_level = list(analysis_results.keys())[0]
        if 'portfolio_data' in analysis_results[risk_level]:
            portfolio_data = analysis_results[risk_level]['portfolio_data']
            risk_metrics = portfolio_data.get('risk_metrics', {})
            
            # Strategy summary
            strategy_text = (
                f"Based on our analysis at a {risk_level} risk level, we recommend the following overall strategy:\n\n"
                f"• Target annual premium yield: {get_target_yield(risk_level):.1%}\n"
                f"• Target delta range: {get_delta_range(risk_level)}\n"
                f"• Typical days to expiry: {get_dte_range(risk_level)}\n"
                f"• Position allocation: Cover up to {get_allocation(current_regime, risk_level):.0%} of portfolio value\n\n"
                f"This strategy is designed to optimize income generation while managing risk given "
                f"the current market regime ({regime_name}) and volatility forecast ({vol_forecast:.1%})."
            )
            
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 5, strategy_text)
            pdf.ln(5)
    
    # Strategy recommendations by symbol
    pdf.create_section_header("Recommendations by Position", 2)
    
    # Create strategy recommendations table
    if analysis_results:
        headers = ['Symbol', 'Best Strike', 'DTE', 'Annualized Return', 'Upside Potential', 'Delta']
        data = []
        
        risk_level = list(analysis_results.keys())[0]
        if 'position_results' in analysis_results[risk_level]:
            pos_data = analysis_results[risk_level]['position_results']
            
            for symbol in portfolio:
                if symbol in pos_data and 'optimal_strikes' in pos_data[symbol]:
                    optimal_strikes = pos_data[symbol]['optimal_strikes']
                    
                    # Find best DTE option
                    best_dte = None
                    best_annualized = 0
                    
                    for dte, strike_data in optimal_strikes.items():
                        annualized = strike_data.get('annualized_return', 0)
                        if annualized > best_annualized:
                            best_annualized = annualized
                            best_dte = dte
                    
                    if best_dte is not None:
                        strike_data = optimal_strikes[best_dte]
                        strike = strike_data.get('strike', 0)
                        annualized = strike_data.get('annualized_return', 0)
                        upside = strike_data.get('upside_potential', 0)
                        
                        # Try to get delta
                        delta = 0
                        if 'greeks' in strike_data:
                            delta = strike_data['greeks'].get('delta', 0)
                        elif 'delta' in strike_data:
                            delta = strike_data['delta']
                        
                        data.append([
                            symbol, 
                            f"${strike:.2f}", 
                            f"{int(best_dte)}", 
                            f"{annualized:.2f}%", 
                            f"{upside:.2f}%",
                            f"{delta:.2f}"
                        ])
            
            if data:
                pdf.create_data_table(headers, data)
    
    # Position Details
    pdf.add_page()
    pdf.create_section_header("5. Position Details", 1)
    
    # Show detailed analysis for each position
    if analysis_results and portfolio:
        risk_level = list(analysis_results.keys())[0]
        if 'position_results' in analysis_results[risk_level]:
            pos_data = analysis_results[risk_level]['position_results']
            
            for symbol in portfolio:
                if symbol in pos_data:
                    position = pos_data[symbol]
                    current_price = position.get('current_price', 0)
                    
                    # Position header
                    pdf.create_section_header(f"{symbol} - ${current_price:.2f}", 2)
                    
                    # Position summary
                    shares = portfolio[symbol]
                    contracts = shares // 100
                    position_value = shares * current_price
                    
                    summary_text = (
                        f"Shares: {shares:,}\n"
                        f"Contracts available: {contracts}\n"
                        f"Position value: ${position_value:,.2f}\n"
                    )
                    
                    if 'volatility' in position and 'forecast' in position['volatility']:
                        vol = position['volatility']['forecast']
                        summary_text += f"Volatility forecast: {vol:.2%}\n"
                    
                    if 'market_regime' in position and 'current' in position['market_regime']:
                        regime = position['market_regime']['current']
                        regime_name = position['market_regime'].get('name', f"Regime {regime}")
                        summary_text += f"Market regime: {regime_name}\n"
                    
                    pdf.create_info_box(f"{symbol} Position Summary", summary_text)
                    
                    # Option chain analysis
                    if 'option_chains' in position:
                        for dte, chain in position['option_chains'].items():
                            if chain:
                                fig = generate_option_chain_analysis(chain, current_price)
                                pdf.insert_chart(fig, f"{symbol} Option Chain - {dte} DTE")
                                break  # Just show one chain for brevity
                    
                    # Strike recommendations
                    if 'optimal_strikes' in position:
                        pdf.create_section_header(f"{symbol} Strike Recommendations", 3)
                        
                        headers = ['DTE', 'Strike', 'Premium', 'Ann. Return', 'Upside %', 'Delta', 'Prob. OTM']
                        data = []
                        
                        for dte, strike_data in position['optimal_strikes'].items():
                            strike = strike_data.get('strike', 0)
                            premium = strike_data.get('call_price', 0)
                            annualized = strike_data.get('annualized_return', 0)
                            upside = strike_data.get('upside_potential', 0)
                            
                            # Try to get delta and probability metrics
                            delta = 0
                            prob_otm = 0
                            
                            if 'greeks' in strike_data:
                                delta = strike_data['greeks'].get('delta', 0)
                            elif 'delta' in strike_data:
                                delta = strike_data['delta']
                            
                            if 'probabilities' in strike_data:
                                probs = strike_data['probabilities']
                                prob_otm = probs.get('prob_otm', 0)
                            
                            data.append([
                                f"{int(dte)}", 
                                f"${strike:.2f}", 
                                f"${premium:.2f}", 
                                f"{annualized:.2f}%", 
                                f"{upside:.2f}%",
                                f"{delta:.2f}",
                                f"{prob_otm:.2%}"
                            ])
                        
                        if data:
                            pdf.create_data_table(headers, data)
    
    # Backtest Results
    pdf.add_page()
    pdf.create_section_header("6. Backtest Results", 1)
    
    # Show backtest results for each position
    if backtests and portfolio:
        for symbol in portfolio:
            if symbol in backtests:
                bt = backtests[symbol]
                
                if 'error' in bt:
                    continue
                
                # Backtest summary
                pdf.create_section_header(f"{symbol} Backtest Results", 2)
                
                total_return = bt.get('total_return', 0)
                annualized_return = bt.get('annualized_return', 0)
                max_drawdown = bt.get('max_drawdown', 0)
                
                summary_text = (
                    f"Period: {bt.get('start_date', 'N/A')} to {bt.get('end_date', 'N/A')}\n"
                    f"Total return: {total_return:.2%}\n"
                    f"Annualized return: {annualized_return:.2%}\n"
                    f"Maximum drawdown: {max_drawdown:.2%}\n"
                )
                
                if 'sharpe_ratio' in bt:
                    summary_text += f"Sharpe ratio: {bt['sharpe_ratio']:.2f}\n"
                
                if 'sortino_ratio' in bt:
                    summary_text += f"Sortino ratio: {bt['sortino_ratio']:.2f}\n"
                
                pdf.create_info_box(f"{symbol} Backtest Metrics", summary_text)
                
                # Backtest chart
                fig = generate_backtest_pnl_chart(bt)
                pdf.insert_chart(fig, f"{symbol} Covered Call Strategy Performance")
                
                # Trade summary if available
                if 'trades' in bt and bt['trades']:
                    pdf.create_section_header(f"{symbol} Trade Summary", 3)
                    
                    # Calculate trade metrics
                    trades = bt['trades']
                    num_trades = len(trades)
                    
                    # Filter to closed trades
                    closed_trades = [t for t in trades if 'close_date' in t]
                    num_closed = len(closed_trades)
                    
                    # Calculate win rate if PnL available
                    win_rate = 0
                    avg_profit = 0
                    avg_loss = 0
                    
                    if any('pnl' in t for t in closed_trades):
                        wins = [t for t in closed_trades if t.get('pnl', 0) > 0]
                        losses = [t for t in closed_trades if t.get('pnl', 0) <= 0]
                        
                        win_rate = len(wins) / num_closed if num_closed > 0 else 0
                        
                        if wins:
                            avg_profit = sum(t.get('pnl', 0) for t in wins) / len(wins)
                        
                        if losses:
                            avg_loss = sum(t.get('pnl', 0) for t in losses) / len(losses)
                    
                    trade_summary = (
                        f"Total trades: {num_trades}\n"
                        f"Closed trades: {num_closed}\n"
                        f"Win rate: {win_rate:.2%}\n"
                        f"Average profit: ${avg_profit:.2f}\n"
                        f"Average loss: ${avg_loss:.2f}\n"
                    )
                    
                    pdf.set_font('Arial', '', 10)
                    pdf.multi_cell(0, 5, trade_summary)
    
    # Implementation Plan
    pdf.add_page()
    pdf.create_section_header("7. Implementation Plan", 1)
    
    # Implementation plan from market analysis
    if market_analysis and 'implementation_plan' in market_analysis:
        impl_plan = market_analysis['implementation_plan']
        
        # Execution timing
        if 'execution_timing' in impl_plan:
            pdf.create_section_header("Execution Timing Strategy", 2)
            
            timing = impl_plan['execution_timing']
            timing_text = (
                f"Optimal time of day: {timing.get('optimal_time_of_day', 'Any')}\n"
                f"Optimal days of week: {timing.get('optimal_days_of_week', 'Any')}\n"
                f"Days to avoid: {timing.get('days_to_avoid', 'None')}\n\n"
                f"IV dynamics: {timing.get('iv_dynamics', '')}\n"
                f"Day of week effects: {timing.get('day_of_week_effects', '')}\n"
            )
            
            # Add other considerations
            other = timing.get('other_considerations', [])
            if other:
                timing_text += "\nOther considerations:\n"
                for item in other:
                    timing_text += f"• {item}\n"
            
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 5, timing_text)
        
        # Risk management
        if 'risk_management' in impl_plan:
            pdf.create_section_header("Risk Management Guidelines", 2)
            
            risk_mgmt = impl_plan['risk_management']
            
            # Portfolio level risk management
            if 'portfolio_level' in risk_mgmt:
                port_risk = risk_mgmt['portfolio_level']
                
                port_risk_text = (
                    f"Maximum loss target: ${port_risk.get('max_loss', 0):,.2f}\n"
                    f"Maximum delta exposure: ${port_risk.get('max_delta_exposure', 0):,.2f}\n"
                    f"Minimum cash reserve: ${port_risk.get('cash_reserve_min', 0):,.2f}\n"
                    f"Maximum vega exposure: ${port_risk.get('max_vega_exposure', 0):,.2f}\n"
                )
                
                pdf.create_info_box("Portfolio Risk Limits", port_risk_text)
            
            # Exit criteria
            if 'exit_criteria' in risk_mgmt:
                exit_criteria = risk_mgmt['exit_criteria']
                
                exit_text = (
                    f"Profit target: {exit_criteria.get('profit_target_pct', 0):.1f}% of premium\n"
                    f"Loss limit: {exit_criteria.get('loss_limit_pct', 0):.1f}% of premium\n"
                    f"DTE threshold: {exit_criteria.get('dte_threshold', 0)} days\n"
                    f"Delta threshold: {exit_criteria.get('delta_threshold', 0):.2f}\n"
                    f"Gamma threshold: {exit_criteria.get('gamma_threshold', 0):.4f}\n"
                )
                
                pdf.create_info_box("Exit Criteria", exit_text)
            
            # Adjustment triggers
            if 'adjustment_triggers' in risk_mgmt:
                triggers = risk_mgmt['adjustment_triggers']
                
                trigger_text = ""
                for trigger, description in triggers.items():
                    trigger_text += f"• {trigger.replace('_', ' ').title()}: {description}\n"
                
                pdf.create_info_box("Adjustment Triggers", trigger_text)
    
    # Implementation calendar
    pdf.create_section_header("Implementation Calendar", 2)
    
    # Create calendar table
    headers = ['Date', 'Symbol', 'Action', 'Details']
    data = []
    
    # Generate artificial implementation calendar
    curr_date = datetime.now()
    for i, symbol in enumerate(portfolio):
        execution_date = curr_date + timedelta(days=i%3)  # Spread over 3 days
        date_str = execution_date.strftime('%Y-%m-%d')
        
        # Get position details if available
        strike = "Best strike"
        premium = "Market price"
        
        if analysis_results:
            risk_level = list(analysis_results.keys())[0]
            if ('position_results' in analysis_results[risk_level] and 
                symbol in analysis_results[risk_level]['position_results'] and
                'optimal_strikes' in analysis_results[risk_level]['position_results'][symbol]):
                
                pos_data = analysis_results[risk_level]['position_results'][symbol]
                optimal_strikes = pos_data['optimal_strikes']
                
                # Get first optimal strike
                if optimal_strikes:
                    first_dte = list(optimal_strikes.keys())[0]
                    strike_data = optimal_strikes[first_dte]
                    
                    strike = f"${strike_data.get('strike', 0):.2f}"
                    premium = f"${strike_data.get('call_price', 0):.2f}"
        
        data.append([
            date_str,
            symbol,
            "Sell to open",
            f"{strike} call, premium {premium}"
        ])
    
    if data:
        pdf.create_data_table(headers, data)
    
    return pdf