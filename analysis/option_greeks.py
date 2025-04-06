# analysis/option_greeks.py
"""
Option Greeks Analysis Module.

This module provides tools for analyzing option greeks and managing
the greek exposure of option portfolios.
"""

import logging
import math
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Constants
RISK_FREE_RATE = 0.02  # 2% risk-free rate


class OptionGreeksManager:
    """
    Manager for option greeks and risk exposures.
    
    This class provides tools to calculate, analyze, and manage option greeks
    for individual positions and portfolios.
    """
    
    def __init__(
        self, 
        portfolio: Dict[str, float],
        option_positions: Dict[str, Dict[str, Any]],
        underlying_prices: Dict[str, float]
    ):
        """
        Initialize with portfolio data.
        
        Parameters:
        portfolio (dict): Underlying positions {symbol: quantity}
        option_positions (dict): Option positions with details
        underlying_prices (dict): Current prices for underlyings
        """
        self.portfolio = portfolio
        self.option_positions = option_positions
        self.underlying_prices = underlying_prices
    
    def calculate_portfolio_greeks(self) -> Dict[str, float]:
        """
        Calculate aggregate portfolio-level option greeks.
        
        Returns:
        dict: Aggregated greeks
        """
        portfolio_greeks = {
            'delta': 0,
            'gamma': 0,
            'theta': 0,
            'vega': 0, 
            'rho': 0
        }
        
        # Add option greeks
        for option_id, option in self.option_positions.items():
            symbol = option['symbol']
            quantity = option['quantity']
            option_type = option.get('option_type', 'call')
            
            # For short positions, negate the quantity
            if option.get('position_type', 'long') == 'short':
                quantity = -quantity
                
            # Skip if no greek data available
            if 'greeks' not in option:
                continue
                
            for greek in portfolio_greeks:
                if greek in option['greeks']:
                    # For puts, delta is negative by convention
                    if greek == 'delta' and option_type == 'put':
                        portfolio_greeks[greek] -= option['greeks'][greek] * quantity
                    else:
                        portfolio_greeks[greek] += option['greeks'][greek] * quantity
        
        # Add delta from underlying positions (delta = 1)
        for symbol, quantity in self.portfolio.items():
            if symbol in self.underlying_prices:
                portfolio_greeks['delta'] += quantity
        
        return portfolio_greeks
    
    def get_greek_exposures_by_expiry(self) -> Dict[str, Dict[str, float]]:
        """
        Group greek exposures by expiration date.
        
        Returns:
        dict: Greek exposures by expiry
        """
        exposures = {}
        
        for option_id, option in self.option_positions.items():
            expiry = option.get('expiry', 'unknown')
            option_type = option.get('option_type', 'call')
            position_type = option.get('position_type', 'long')
            quantity = option['quantity']
            
            # For short positions, negate the quantity
            if position_type == 'short':
                quantity = -quantity
            
            if expiry not in exposures:
                exposures[expiry] = {
                    'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0,
                    'call_delta': 0, 'put_delta': 0,
                    'call_positions': 0, 'put_positions': 0
                }
            
            if 'greeks' in option:
                for greek in ['gamma', 'theta', 'vega', 'rho']:
                    if greek in option['greeks']:
                        exposures[expiry][greek] += option['greeks'][greek] * quantity
                
                # Handle delta separately for puts and calls
                if 'delta' in option['greeks']:
                    if option_type == 'call':
                        exposures[expiry]['delta'] += option['greeks']['delta'] * quantity
                        exposures[expiry]['call_delta'] += option['greeks']['delta'] * quantity
                        exposures[expiry]['call_positions'] += quantity
                    elif option_type == 'put':
                        exposures[expiry]['delta'] -= option['greeks']['delta'] * quantity  # Put delta is negative
                        exposures[expiry]['put_delta'] -= option['greeks']['delta'] * quantity
                        exposures[expiry]['put_positions'] += quantity
        
        return exposures
    
    def calculate_delta_exposure_by_strike(self) -> Dict[str, Dict[float, Dict[str, float]]]:
        """
        Calculate delta exposure bucketed by strike price.
        
        Returns:
        dict: Delta exposure by strike
        """
        delta_exposure = {}
        
        for option_id, option in self.option_positions.items():
            symbol = option['symbol']
            strike = option.get('strike', 0)
            option_type = option.get('option_type', 'call')
            position_type = option.get('position_type', 'long')
            quantity = option['quantity']
            
            # Skip if no underlying price
            if symbol not in self.underlying_prices:
                continue
                
            # Calculate moneyness
            underlying_price = self.underlying_prices[symbol]
            moneyness = strike / underlying_price
            
            # Round moneyness to buckets (e.g., 0.90, 0.95, 1.00, etc.)
            bucket = round(moneyness * 20) / 20
            
            if symbol not in delta_exposure:
                delta_exposure[symbol] = {}
                
            if bucket not in delta_exposure[symbol]:
                delta_exposure[symbol][bucket] = {'call_delta': 0, 'put_delta': 0, 'total_delta': 0}
            
            # For short positions, negate the quantity
            if position_type == 'short':
                quantity = -quantity
                
            # Add delta exposure
            if 'greeks' in option and 'delta' in option['greeks']:
                delta = option['greeks']['delta'] * quantity
                
                if option_type == 'call':
                    delta_exposure[symbol][bucket]['call_delta'] += delta
                    delta_exposure[symbol][bucket]['total_delta'] += delta
                elif option_type == 'put':
                    delta_exposure[symbol][bucket]['put_delta'] -= delta  # Put delta is negative
                    delta_exposure[symbol][bucket]['total_delta'] -= delta
        
        return delta_exposure
    
    def calculate_gamma_profile_by_price(
        self, 
        price_moves: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate gamma exposure profile for different price levels.
        
        Parameters:
        price_moves (dict): Percentage price moves to consider by symbol
        
        Returns:
        dict: Gamma profile
        """
        if price_moves is None:
            # Default: evaluate range of price levels
            price_moves = {symbol: np.linspace(-0.10, 0.10, 21) for symbol in self.underlying_prices}
        
        gamma_profile = {}
        
        for symbol in self.underlying_prices:
            if symbol not in price_moves:
                continue
                
            current_price = self.underlying_prices[symbol]
            
            # Calculate total vega for this symbol
            total_vega = 0
            for option_id, option in self.option_positions.items():
                if option['symbol'] != symbol or 'greeks' not in option or 'vega' not in option['greeks']:
                    continue
                    
                quantity = option['quantity']
                if option.get('position_type', 'long') == 'short':
                    quantity = -quantity
                    
                total_vega += option['greeks']['vega'] * quantity
            
            # Calculate impact for each vol change
            impact = {}
            for change in vol_changes[symbol]:
                # Vega is P&L for 1% change in vol, so scale by vol change * 100
                pnl = total_vega * (change * 100)
                
                impact[change] = {
                    'vol_change': change,
                    'pnl': pnl,
                    'pnl_pct': pnl / current_price if current_price > 0 else 0
                }
            
            vol_impact[symbol] = {
                'total_vega': total_vega,
                'impact': impact
            }
        
        return vol_impact
    
    def evaluate_delta_hedging_requirements(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate delta hedging requirements.
        
        Returns:
        dict: Hedging requirements by symbol
        """
        # Calculate net delta by symbol
        net_delta = {}
        
        # Add delta from options
        for option_id, option in self.option_positions.items():
            symbol = option['symbol']
            option_type = option.get('option_type', 'call')
            position_type = option.get('position_type', 'long')
            
            if symbol not in net_delta:
                net_delta[symbol] = 0
                
            if 'greeks' in option and 'delta' in option['greeks']:
                quantity = option['quantity']
                if position_type == 'short':
                    quantity = -quantity
                    
                if option_type == 'call':
                    net_delta[symbol] += option['greeks']['delta'] * quantity
                else:  # put
                    net_delta[symbol] -= option['greeks']['delta'] * quantity
        
        # Add delta from underlyings (delta = 1)
        for symbol, quantity in self.portfolio.items():
            if symbol not in net_delta:
                net_delta[symbol] = 0
                
            net_delta[symbol] += quantity
        
        # Calculate hedging requirements
        hedging_requirements = {}
        for symbol, delta in net_delta.items():
            target_delta = 0  # Delta-neutral by default
            
            # Convert delta to shares
            delta_shares = int(delta * 100)  # Assuming delta is per 100 shares
            
            hedging_requirements[symbol] = {
                'current_delta': delta,
                'delta_shares': delta_shares,
                'target_delta': target_delta,
                'adjustment_needed': target_delta - delta,
                'adjustment_shares': int((target_delta - delta) * 100)
            }
        
        return hedging_requirements
    
    def calculate_gamma_scalping_opportunity(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate gamma scalping opportunity.
        
        Returns:
        dict: Gamma scalping metrics
        """
        opportunities = {}
        
        for symbol in self.underlying_prices:
            current_price = self.underlying_prices[symbol]
            
            # Calculate total gamma for this symbol
            total_gamma = 0
            for option_id, option in self.option_positions.items():
                if option['symbol'] != symbol or 'greeks' not in option or 'gamma' not in option['greeks']:
                    continue
                    
                quantity = option['quantity']
                if option.get('position_type', 'long') == 'short':
                    quantity = -quantity
                    
                total_gamma += option['greeks']['gamma'] * quantity
            
            # Skip if no gamma exposure
            if abs(total_gamma) < 0.001:
                continue
                
            # Calculate breakeven move for gamma scalping
            # Formula: Breakeven move = 2 * (hedging cost per share) / (gamma * current price)
            hedging_cost = 0.01  # 1 cent per share, can be adjusted
            breakeven_move = 2 * hedging_cost / (abs(total_gamma) * current_price)
            
            # Calculate expected P&L from gamma scalping for various volatility scenarios
            vol_scenarios = {
                'low_vol': 0.10,
                'med_vol': 0.20,
                'high_vol': 0.30
            }
            
            scalping_pnl = {}
            for scenario, vol in vol_scenarios.items():
                # Daily expected move based on volatility
                daily_expected_move = current_price * vol / math.sqrt(252)
                
                # Daily P&L = 0.5 * gamma * expected_move^2 - hedging_cost
                daily_pnl = 0.5 * abs(total_gamma) * daily_expected_move**2 - hedging_cost
                
                scalping_pnl[scenario] = {
                    'volatility': vol,
                    'daily_expected_move': daily_expected_move,
                    'daily_pnl': daily_pnl,
                    'daily_pnl_pct': daily_pnl / current_price
                }
            
            opportunities[symbol] = {
                'total_gamma': total_gamma,
                'gamma_per_price': total_gamma / current_price,
                'breakeven_move': breakeven_move,
                'breakeven_move_pct': breakeven_move / current_price,
                'hedging_cost': hedging_cost,
                'scalping_pnl': scalping_pnl
            }
        
        return opportunities


def calculate_option_greeks(
    option_type: str,
    strike: float,
    underlying_price: float,
    days_to_expiry: int,
    volatility: float,
    risk_free_rate: float = RISK_FREE_RATE
) -> Dict[str, float]:
    """
    Calculate option greeks using Black-Scholes model.
    
    Parameters:
    option_type (str): 'call' or 'put'
    strike (float): Strike price
    underlying_price (float): Price of the underlying
    days_to_expiry (int): Days to expiry
    volatility (float): Implied volatility
    risk_free_rate (float): Risk-free rate
    
    Returns:
    dict: Calculated option greeks
    """
    # Time to expiry in years
    t = days_to_expiry / 365
    
    # Check for valid inputs
    if t <= 0 or volatility <= 0 or underlying_price <= 0 or strike <= 0:
        return {
            'delta': 0,
            'gamma': 0,
            'theta': 0,
            'vega': 0,
            'rho': 0
        }
    
    # Calculate d1 and d2
    d1 = (math.log(underlying_price/strike) + (risk_free_rate + 0.5 * volatility**2) * t) / (volatility * math.sqrt(t))
    d2 = d1 - volatility * math.sqrt(t)
    
    # Calculate option greeks
    if option_type.lower() == 'call':
        # Delta
        delta = norm.cdf(d1)
        
        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (underlying_price * volatility * math.sqrt(t))
        
        # Theta (time decay)
        theta = (
            -(underlying_price * norm.pdf(d1) * volatility) / (2 * math.sqrt(t)) -
            risk_free_rate * strike * math.exp(-risk_free_rate * t) * norm.cdf(d2)
        ) / 365  # Daily theta
        
        # Vega (sensitivity to volatility)
        vega = underlying_price * math.sqrt(t) * norm.pdf(d1) / 100  # For 1% vol change
        
        # Rho (sensitivity to interest rate)
        rho = strike * t * math.exp(-risk_free_rate * t) * norm.cdf(d2) / 100  # For 1% rate change
    
    else:  # Put
        # Delta
        delta = norm.cdf(d1) - 1
        
        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (underlying_price * volatility * math.sqrt(t))
        
        # Theta (time decay)
        theta = (
            -(underlying_price * norm.pdf(d1) * volatility) / (2 * math.sqrt(t)) +
            risk_free_rate * strike * math.exp(-risk_free_rate * t) * norm.cdf(-d2)
        ) / 365  # Daily theta
        
        # Vega (sensitivity to volatility)
        vega = underlying_price * math.sqrt(t) * norm.pdf(d1) / 100  # For 1% vol change
        
        # Rho (sensitivity to interest rate)
        rho = -strike * t * math.exp(-risk_free_rate * t) * norm.cdf(-d2) / 100  # For 1% rate change
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }
 self.underlying_prices[symbol]
            moves = price_moves[symbol]
            
            profile = {}
            for move in moves:
                new_price = current_price * (1 + move)
                price_level = round(new_price, 2)
                
                # Calculate gamma at this price level
                total_gamma = 0
                total_delta = 0
                
                for option_id, option in self.option_positions.items():
                    if option['symbol'] != symbol or 'greeks' not in option:
                        continue
                        
                    quantity = option['quantity']
                    if option.get('position_type', 'long') == 'short':
                        quantity = -quantity
                    
                    # Use BSM to calculate greeks at new price level
                    if all(k in option for k in ['strike', 'days_to_expiry', 'implied_vol']):
                        strike = option['strike']
                        t = option['days_to_expiry'] / 365
                        sigma = option['implied_vol']
                        
                        if t > 0 and sigma > 0:
                            option_type = option.get('option_type', 'call')
                            
                            # Calculate d1, d2
                            d1 = (math.log(new_price/strike) + (RISK_FREE_RATE + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
                            d2 = d1 - sigma * math.sqrt(t)
                            
                            # Calculate gamma
                            gamma = norm.pdf(d1) / (new_price * sigma * math.sqrt(t))
                            
                            # Calculate delta
                            if option_type == 'call':
                                delta = norm.cdf(d1)
                            else:  # put
                                delta = norm.cdf(d1) - 1
                            
                            total_gamma += gamma * quantity
                            total_delta += delta * quantity
                
                # Store results
                profile[price_level] = {
                    'price_move': move,
                    'price': price_level,
                    'gamma': total_gamma,
                    'delta': total_delta,
                    # Gamma P&L = 0.5 * Gamma * (ΔS)^2
                    'gamma_pnl': 0.5 * total_gamma * (price_level - current_price)**2
                }
            
            gamma_profile[symbol] = {
                'current_price': current_price,
                'profile': profile
            }
        
        return gamma_profile
    
    def calculate_theta_decay_projection(
        self, 
        days_forward: List[int] = [1, 7, 30]
    ) -> Dict[str, float]:
        """
        Project theta decay over specified days.
        
        Parameters:
        days_forward (list): List of days to project
        
        Returns:
        dict: Projected theta decay
        """
        theta_decay = {}
        
        for day in days_forward:
            daily_theta = 0
            
            for option_id, option in self.option_positions.items():
                if 'greeks' not in option or 'theta' not in option['greeks']:
                    continue
                    
                quantity = option['quantity']
                if option.get('position_type', 'long') == 'short':
                    quantity = -quantity
                    
                daily_theta += option['greeks']['theta'] * quantity
            
            theta_decay[f'day_{day}'] = daily_theta * day
        
        return theta_decay
    
    def calculate_vega_exposure_by_expiry(self) -> Dict[str, float]:
        """
        Calculate vega exposure by expiry.
        
        Returns:
        dict: Vega exposure by expiry
        """
        vega_exposure = {}
        
        exposures = self.get_greek_exposures_by_expiry()
        
        for expiry, greeks in exposures.items():
            vega_exposure[expiry] = greeks.get('vega', 0)
        
        return vega_exposure
    
    def calculate_volatility_impact(
        self, 
        vol_changes: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate impact of volatility changes on portfolio.
        
        Parameters:
        vol_changes (dict): Volatility changes by symbol
        
        Returns:
        dict: Volatility impact analysis
        """
        if vol_changes is None:
            # Default: evaluate range of vol changes
            vol_changes = {symbol: [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05] for symbol in self.underlying_prices}
        
        vol_impact = {}
        
        for symbol in self.underlying_prices:
            if symbol not in vol_changes:
                continue
                
            current_price =