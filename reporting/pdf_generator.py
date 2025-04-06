# reporting/pdf_generator.py
"""
PDF Report Generator Module.

This module provides functions for generating comprehensive PDF reports
with visualizations, tables, and analysis of options strategies.
"""

import os
import logging
from datetime import datetime
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
                    elif "%" in headers[i] or i > 0 and "%" in str(row[0]):
                        cell_text = f"{cell:.2f}%"
                    else:
                        cell_text = f"{cell:.2f}"
                else:
                    cell_text = str(cell)