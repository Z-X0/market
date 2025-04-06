# reporting/pdf_generator.py

"""
PDF Report Generator Module.

This module provides functions for generating comprehensive PDF reports
with visualizations, tables, and analysis of options strategies.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

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
    base_allocation = {
        'conservative': 0.5,
        'moderate': 0.7,
        'aggressive': 0.9
    }.get(risk_level, 0.6)
    regime_adjustment = {
        0: -0.2, 1: -0.1, 2: -0.05,
        3: 0.0, 4: 0.05, 5: 0.1, 6: 0.2
    }.get(regime, 0.0)
    return max(0.1, min(1.0, base_allocation + regime_adjustment))


class EnhancedQuantPDF(FPDF):
    """Enhanced PDF class with better styling and visualization capabilities."""

    def __init__(self, orientation='L', unit='mm', format='A4'):
        super().__init__(orientation, unit, format)
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(10, 10, 10)
        try:
            self.add_font('Arial', '', 'arial.ttf', uni=True)
        except Exception:
            logger.warning("Arial font not found, using default font")

    def sanitize_text(self, text):
        """Sanitize text for PDF output, replacing non-latin1 chars."""
        if not isinstance(text, str):
            text = str(text)
        return text.encode('latin-1', 'replace').decode('latin-1')

    # Override cell and multi_cell to sanitize every string
    def cell(self, w, h=0, txt='', border=0, ln=0, align='', fill=False, link=''):
        safe_txt = self.sanitize_text(txt)
        return super().cell(w, h, safe_txt, border, ln, align, fill, link)

    def multi_cell(self, w, h, txt, border=0, align='J', fill=False):
        safe_txt = self.sanitize_text(txt)
        return super().multi_cell(w, h, safe_txt, border, align, fill)

    def create_section_header(self, title, level=1):
        """Create formatted section header with better styling."""
        if level == 1:
            self.set_font("Arial", "B", 16)
            self.set_fill_color(25, 65, 120)
            self.set_text_color(255, 255, 255)
            self.cell(0, 10, title, 0, 1, "L", True)
            self.set_text_color(0, 0, 0)
        elif level == 2:
            self.set_font("Arial", "B", 14)
            self.set_fill_color(65, 105, 225)
            self.set_text_color(255, 255, 255)
            self.cell(0, 8, title, 0, 1, "L", True)
            self.set_text_color(0, 0, 0)
        else:
            self.set_font("Arial", "I", 12)
            self.set_fill_color(135, 206, 250)
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
        self.set_fill_color(230, 240, 250)
        self.set_draw_color(25, 65, 120)
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
        self.set_fill_color(25, 65, 120)
        self.set_text_color(255, 255, 255)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, self.sanitize_text(header), 1, 0, "C", True)
        self.ln()
        # Table data
        self.set_text_color(0, 0, 0)
        self.set_font("Arial", "", 9)
        fill = False
        for row in data:
            self.set_fill_color(240, 240, 245 if fill else 255)
            for i, cell in enumerate(row):
                if isinstance(cell, float):
                    # Format numbers
                    cell_text = f"{cell:.2f}" if abs(cell) >= 0.1 else f"{cell:.4f}"
                else:
                    cell_text = str(cell)
                align = "R" if isinstance(cell, (int, float)) else "L"
                self.cell(col_widths[i], 6, self.sanitize_text(cell_text), 1, 0, align, True)
            self.ln()
            fill = not fill


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

    pdf = EnhancedQuantPDF('L', 'mm', 'A4')
    pdf.add_page()

    # Cover page
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 20, "Enhanced Options Strategy Analysis", 0, 1, 'C')
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 15, "Covered Call Strategy Optimization", 0, 1, 'C')
    pdf.set_font('Arial', '', 12)
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
    pdf.cell(0, 10, f"Generated on: {current_date}", 0, 1, 'C')

    # Portfolio summary
    num_symbols = len(portfolio)
    total_shares = sum(portfolio.values())
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 15, f"Portfolio: {num_symbols} symbols, {total_shares:,} shares", 0, 1, 'C')

    # Portfolio table
    pdf.set_font('Arial', 'B', 10)
    pdf.ln(5)
    pdf.cell(0, 8, "Portfolio Composition:", 0, 1, 'L')
    headers = ['Symbol', 'Shares', 'Weight']
    data = []
    total_value = 0
    for symbol, shares in portfolio.items():
        price = next(
            (res['position_results'][symbol]['current_price']
             for res in analysis_results.values()
             if symbol in res.get('position_results', {})),
            0
        )
        total_value += shares * price
    for symbol, shares in portfolio.items():
        price = next(
            (res['position_results'][symbol]['current_price']
             for res in analysis_results.values()
             if symbol in res.get('position_results', {})),
            0
        )
        weight = (shares * price) / total_value if total_value else 0
        data.append([symbol, f"{shares:,}", f"{weight:.2%}"])
    pdf.create_data_table(headers, data, width=150, col_widths=[50, 50, 50])

    # Table of Contents
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

    # 1. Executive Summary
    pdf.add_page()
    pdf.create_section_header("1. Executive Summary", 1)
    risk_level = next(iter(analysis_results), 'conservative')
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

    # Key Portfolio Metrics
    pdf.create_section_header("Key Portfolio Metrics", 2)
    metrics_content = ""
    for res in analysis_results.values():
        rm = res.get('portfolio_data', {}).get('risk_metrics')
        if rm:
            metrics_content = (
                f"Total Portfolio Value: ${rm.get('total_value',0):,.2f}\n"
                f"Portfolio Volatility: {rm.get('portfolio_volatility',0):.2%}\n"
                f"Value at Risk (99%): ${rm.get('value_at_risk',0):,.2f}\n"
                f"Expected Shortfall: ${rm.get('expected_shortfall',0):,.2f}\n"
                f"Diversification Ratio: {rm.get('diversification_ratio',0):.2%}\n"
            )
            break
    if metrics_content:
        pdf.create_info_box("Portfolio Risk Metrics", metrics_content)

    # 2. Market Regime Analysis
    pdf.add_page()
    pdf.create_section_header("2. Market Regime Analysis", 1)
    first_symbol = next(iter(portfolio))
    pos = next(
        (res['position_results'][first_symbol] for res in analysis_results.values()
         if first_symbol in res.get('position_results', {})),
        {}
    )
    fig = generate_regime_chart(first_symbol, pos.get('market_analyzer'), pos.get('weekly_data'))
    pdf.insert_chart(fig, f"Market Regime Analysis for {first_symbol}")

    # 3. Volatility Analysis
    pdf.add_page()
    pdf.create_section_header("3. Volatility Analysis", 1)
    pos_data = next(iter(analysis_results.values())).get('position_results', {})
    fig = generate_vol_comparison_chart(pos_data, portfolio)
    pdf.insert_chart(fig, "Volatility Model Comparison")

    # 4. Strategy Recommendations
    pdf.add_page()
    pdf.create_section_header("4. Strategy Recommendations", 1)
    strat_text = (
        f"• Target annual premium yield: {get_target_yield(risk_level):.1%}\n"
        f"• Target delta range: {get_delta_range(risk_level)}\n"
        f"• Typical days to expiry: {get_dte_range(risk_level)}\n"
        f"• Position allocation: Cover up to {get_allocation(market_analysis.get('market_regime',3), risk_level):.0%} of portfolio value\n"
    )
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, strat_text)

    # 5. Position Details
    pdf.add_page()
    pdf.create_section_header("5. Position Details", 1)
    for symbol in portfolio:
        p = pos_data.get(symbol)
        if not p: continue
        price = p.get('current_price',0)
        pdf.create_section_header(f"{symbol} - ${price:.2f}",2)
        shares = portfolio[symbol]
        summary = f"Shares: {shares}\nContracts: {shares//100}\nValue: ${shares*price:,.2f}"
        pdf.create_info_box(f"{symbol} Summary", summary)
        for dte, chain in p.get('option_chains',{}).items():
            if chain:
                fig = generate_option_chain_analysis(chain, price)
                pdf.insert_chart(fig, f"{symbol} Option Chain - {dte} DTE")
                break

    # 6. Backtest Results
    pdf.add_page()
    pdf.create_section_header("6. Backtest Results", 1)
    for symbol, bt in backtests.items():
        if 'error' in bt: continue
        pdf.create_section_header(f"{symbol} Backtest",2)
        metrics = (
            f"Total return: {bt.get('total_return',0):.2%}\n"
            f"Ann. return: {bt.get('annualized_return',0):.2%}\n"
            f"Max drawdown: {bt.get('max_drawdown',0):.2%}"
        )
        pdf.create_info_box(f"{symbol} Metrics", metrics)
        fig = generate_backtest_pnl_chart(bt)
        pdf.insert_chart(fig, f"{symbol} Performance")

    # 7. Implementation Plan
    pdf.add_page()
    pdf.create_section_header("7. Implementation Plan", 1)
    timing = market_analysis.get('implementation_plan',{}).get('execution_timing',{})
    ip_text = (
        f"Optimal time: {timing.get('optimal_time_of_day','Any')}\n"
        f"Days to avoid: {timing.get('days_to_avoid','None')}"
    )
    pdf.multi_cell(0,5,ip_text)

    return pdf
