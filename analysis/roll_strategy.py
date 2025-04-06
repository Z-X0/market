# analysis/roll_strategy.py
"""
Roll Strategy Analysis Module.

This module provides tools for analyzing and optimizing option roll strategies,
including timing, strike selection, and execution guidelines.
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


class RollStrategyOptimizer:
    """
    Optimizer for option roll strategies.
    
    This class provides tools to analyze and optimize roll strategies
    for existing option positions.
    """
    
    def __init__(
        self, 
        symbol: str, 
        current_option: Optional[Dict[str, Any]] = None, 
        option_chain_provider: Optional[callable] = None
    ):
        """
        Initialize with symbol and option data.
        
        Parameters:
        symbol (str): Underlying symbol
        current_option (dict): Current option position details
        option_chain_provider (callable): Function to fetch option chains
        """
        self.symbol = symbol
        self.current_option = current_option
        self.get_option_chain = option_chain_provider
    
    def evaluate_roll_candidates(
        self, 
        current_price: float, 
        risk_preference: str = 'neutral'
    ) -> Dict[str, Any]:
        """
        Evaluate potential roll candidates for a covered call.
        
        Parameters:
        current_price (float): Current price of underlying
        risk_preference (str): 'defensive', 'neutral', or 'aggressive'
        
        Returns:
        dict: Ranked roll candidates
        """
        if self.current_option is None or self.get_option_chain is None:
            return {"error": "Missing current option or chain provider"}
            
        # Extract current option details
        current_strike = self.current_option.get('strike', 0)
        current_dte = self.current_option.get('days_to_expiry', 0)
        current_premium = self.current_option.get('premium', 0)
        current_delta = self.current_option.get('delta', 0.3)
        
        # Skip if invalid current option
        if current_strike <= 0 or current_dte <= 0:
            return {"error": "Invalid current option details"}
            
        # Define target DTE ranges based on risk preference
        dte_ranges = {
            'defensive': [14, 21, 30],
            'neutral': [30, 45, 60],
            'aggressive': [45, 60, 90]
        }
        
        # Define target deltas based on risk preference
        target_deltas = {
            'defensive': 0.20,
            'neutral': 0.30,
            'aggressive': 0.40
        }
        
        # Get potential roll targets
        roll_targets = {}
        for dte in dte_ranges.get(risk_preference, [30, 45, 60]):
            if dte <= current_dte and current_dte > 7:  # Don't roll to shorter expiries unless current is near expiry
                continue
                
            try:
                chain = self.get_option_chain(self.symbol, dte)
                if chain:
                    roll_targets[dte] = chain
            except Exception as e:
                logger.warning(f"Error fetching chain for {dte} DTE: {e}")
        
        if not roll_targets:
            return {"error": "No valid roll targets found"}
            
        # Evaluate roll candidates
        candidates = []
        target_delta = target_deltas.get(risk_preference, 0.30)
        
        for dte, chain in roll_targets.items():
            for new_option in chain:
                # Skip if not a call
                if new_option.get('option_type', 'call') != 'call':
                    continue
                
                # Extract details
                new_strike = new_option.get('strike', 0)
                new_premium = new_option.get('premium', 0)
                new_delta = new_option.get('delta', 0.3)
                
                # Skip if invalid
                if new_strike <= 0 or new_premium <= 0:
                    continue
                    
                # Calculate roll metrics
                roll_credit = new_premium - current_premium
                days_extended = dte - current_dte
                
                # Calculate annualized metrics
                current_annual_return = (current_premium / current_strike) * (365 / current_dte)
                new_annual_return = (new_premium / new_strike) * (365 / dte)
                
                # Calculate probability metrics
                prob_otm_current = 1 - current_delta
                prob_otm_new = 1 - new_delta
                
                # Strike selection metrics
                upside_protection = (new_strike - current_price) / current_price
                
                # Roll efficiency metrics
                time_value_ratio = new_premium / (dte / 365) / (current_premium / (current_dte / 365))
                
                theta_efficiency = 0
                if days_extended > 0:
                    theta_efficiency = roll_credit / days_extended
                
                # Composite score (can be customized based on risk preference)
                if risk_preference == 'defensive':
                    score = (prob_otm_new * 0.4) + (upside_protection * 100 * 0.4) + (theta_efficiency * 50 * 0.2)
                elif risk_preference == 'aggressive':
                    score = (new_annual_return * 0.5) + (theta_efficiency * 50 * 0.3) + (time_value_ratio * 0.2)
                else:  # neutral
                    score = (prob_otm_new * 0.3) + (new_annual_return * 0.3) + (theta_efficiency * 50 * 0.2) + (upside_protection * 100 * 0.2)
                
                candidates.append({
                    'dte': dte,
                    'strike': new_strike,
                    'premium': new_premium,
                    'delta': new_delta,
                    'roll_credit': roll_credit,
                    'days_extended': days_extended,
                    'theta_efficiency': theta_efficiency,
                    'prob_otm': prob_otm_new,
                    'upside_protection': upside_protection,
                    'annual_return': new_annual_return,
                    'time_value_ratio': time_value_ratio,
                    'score': score
                })
        
        # Sort candidates by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'current_option': {
                'strike': current_strike,
                'dte': current_dte,
                'premium': current_premium,
                'delta': current_delta,
                'prob_otm': 1 - current_delta,
                'annual_return': (current_premium / current_strike) * (365 / current_dte)
            },
            'roll_candidates': candidates[:10],  # Top 10 candidates
            'risk_preference': risk_preference
        }
    
    def defensive_roll_evaluation(
        self, 
        current_price: float, 
        vol_forecast: float, 
        max_roll_debit: float = -0.005
    ) -> Dict[str, Any]:
        """
        Evaluate defensive roll options when delta is high.
        
        Parameters:
        current_price (float): Current price of underlying
        vol_forecast (float): Volatility forecast
        max_roll_debit (float): Maximum acceptable roll debit as % of current strike
        
        Returns:
        dict: Defensive roll recommendations
        """
        if self.current_option is None or self.get_option_chain is None:
            return {"error": "Missing current option or chain provider"}
            
        # Extract current option details
        current_strike = self.current_option.get('strike', 0)
        current_dte = self.current_option.get('days_to_expiry', 0)
        current_premium = self.current_option.get('premium', 0)
        current_delta = self.current_option.get('delta', 0.3)
        
        # Check if defensive roll is needed
        if current_delta < 0.7:
            return {
                "recommendation": "No defensive action needed",
                "current_delta": current_delta,
                "threshold": 0.7
            }
        
        # Calculate new target strike with buffer
        days_to_expiry_years = current_dte / 365
        implied_move = current_price * vol_forecast * math.sqrt(days_to_expiry_years)
        
        # Get potential roll expirations
        roll_recommendations = []
        
        for roll_dte in [current_dte, current_dte + 30]:
            try:
                roll_chain = self.get_option_chain(self.symbol, roll_dte)
                if not roll_chain:
                    continue
            except Exception as e:
                logger.warning(f"Error fetching chain for {roll_dte} DTE: {e}")
                continue
            
            # Filter to call options only
            call_options = [opt for opt in roll_chain if opt.get('option_type', 'call') == 'call']
            
            # Find higher strikes
            higher_strikes = [opt for opt in call_options if opt.get('strike', 0) > current_strike]
            if not higher_strikes:
                continue
                
            # Sort by strike
            higher_strikes.sort(key=lambda x: x.get('strike', 0))
            
            # Calculate cost to roll up
            for i, new_opt in enumerate(higher_strikes):
                new_strike = new_opt.get('strike', 0)
                new_premium = new_opt.get('premium', 0)
                new_delta = new_opt.get('delta', 0.5)
                
                # Calculate roll metrics
                roll_debit = new_premium - current_premium
                roll_debit_pct = roll_debit / current_strike
                
                # Skip if roll debit is too large
                if roll_debit_pct < max_roll_debit:
                    continue
                
                # Calculate new buffer and probability metrics
                new_buffer = (new_strike - current_price) / current_price
                prob_otm_new = 1 - new_delta
                
                roll_recommendations.append({
                    'dte': roll_dte,
                    'new_strike': new_strike,
                    'roll_debit': roll_debit,
                    'roll_debit_pct': roll_debit_pct * 100,  # Convert to percentage
                    'new_delta': new_delta,
                    'new_buffer_pct': new_buffer * 100,  # Convert to percentage
                    'prob_otm': prob_otm_new,
                    'days_added': roll_dte - current_dte,
                    # Safety score: lower is better (50% prob_otm improvement, 50% buffer improvement)
                    'safety_score': ((current_delta - new_delta) * 0.5) + (new_buffer * 0.5)
                })
                
                # Limit to first few higher strikes
                if i >= 3:
                    break
        
        # Sort by safety score (higher is better)
        roll_recommendations.sort(key=lambda x: x['safety_score'], reverse=True)
        
        # Calculate assignment risk
        days_to_earnings = self.current_option.get('days_to_earnings', 999)
        earnings_risk = "High" if days_to_earnings < current_dte else "Low"
        
        # Calculate expected move to expiration
        expected_move = current_price * vol_forecast * math.sqrt(days_to_expiry_years)
        expected_move_pct = expected_move / current_price
        
        # Calculate probability of touch before expiry
        # Formula from "The Complete Guide to Option Pricing Formulas" by Espen Gaarder Haug
        prob_touch = 2 * (1 - norm.cdf((math.log(current_strike/current_price)) / (vol_forecast * math.sqrt(days_to_expiry_years))))
        
        return {
            "recommendation": "Defensive roll recommended" if roll_recommendations else "No viable defensive roll found",
            "current_option": {
                "strike": current_strike,
                "dte": current_dte,
                "premium": current_premium,
                "delta": current_delta
            },
            "current_price": current_price,
            "assignment_risk": {
                "earnings_risk": earnings_risk,
                "expected_move": expected_move,
                "expected_move_pct": expected_move_pct * 100,  # Convert to percentage
                "probability_of_touch": prob_touch
            },
            "roll_options": roll_recommendations[:3]  # Top 3 recommendations
        }
    
    def early_assignment_risk_analysis(
        self, 
        dividend_schedule: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze early assignment risk, especially around dividends.
        
        Parameters:
        dividend_schedule (dict): Upcoming dividends {date: amount}
        
        Returns:
        dict: Assignment risk analysis
        """
        if self.current_option is None:
            return {"error": "Missing current option details"}
            
        # Extract current option details
        current_strike = self.current_option.get('strike', 0)
        current_dte = self.current_option.get('days_to_expiry', 0)
        current_premium = self.current_option.get('premium', 0)
        current_delta = self.current_option.get('delta', 0.3)
        current_type = self.current_option.get('option_type', 'call')
        
        # Skip if not a call option
        if current_type != 'call':
            return {"risk": "N/A - Not a call option"}
            
        # Basic risk metrics
        risk_factors = {
            'delta_risk': "High" if current_delta > 0.8 else "Medium" if current_delta > 0.6 else "Low",
            'time_value': current_premium - max(0, current_strike - self.current_option.get('underlying_price', 0)),
            'time_value_pct': (current_premium - max(0, current_strike - self.current_option.get('underlying_price', 0))) / current_strike * 100
        }
        
        # Calculate risk around dividends
        dividend_risks = []
        if dividend_schedule:
            for div_date, div_amount in dividend_schedule.items():
                # Calculate days to dividend
                try:
                    from datetime import datetime
                    div_date_dt = datetime.strptime(div_date, '%Y-%m-%d')
                    days_to_div = (div_date_dt - datetime.now()).days
                except Exception as e:
                    logger.warning(f"Error calculating days to dividend: {e}")
                    days_to_div = 0
                
                # Skip if dividend is after expiration
                if days_to_div > current_dte:
                    continue
                    
                # Calculate relevant metrics
                time_value_at_div = risk_factors['time_value'] * (1 - days_to_div/current_dte)  # Simple decay model
                
                dividend_risks.append({
                    'date': div_date,
                    'days_to_dividend': days_to_div,
                    'amount': div_amount,
                    'amount_pct': div_amount / current_strike * 100,
                    'time_value_at_div': time_value_at_div,
                    'risk_level': "High" if div_amount > time_value_at_div else "Medium" if div_amount * 1.5 > time_value_at_div else "Low"
                })
        
        # Overall risk assessment
        if dividend_risks and any(r['risk_level'] == "High" for r in dividend_risks):
            overall_risk = "High"
        elif current_delta > 0.8 and risk_factors['time_value_pct'] < 0.5:
            overall_risk = "High"
        elif current_delta > 0.6 or (dividend_risks and any(r['risk_level'] == "Medium" for r in dividend_risks)):
            overall_risk = "Medium"
        else:
            overall_risk = "Low"
        
        return {
            'current_option': {
                'strike': current_strike,
                'dte': current_dte,
                'delta': current_delta,
                'time_value': risk_factors['time_value'],
                'time_value_pct': risk_factors['time_value_pct']
            },
            'risk_factors': risk_factors,
            'dividend_risks': dividend_risks,
            'overall_risk': overall_risk,
            'recommendation': "Consider rolling" if overall_risk == "High" else "Monitor closely" if overall_risk == "Medium" else "No action needed"
        }
    
    def optimal_roll_timing(
        self, 
        vol_forecast: float, 
        market_regime: int
    ) -> Dict[str, Any]:
        """
        Determine optimal timing for rolling covered calls.
        
        Parameters:
        vol_forecast (float): Volatility forecast
        market_regime (int): Current market regime
        
        Returns:
        dict: Roll timing recommendations
        """
        if self.current_option is None:
            return {"error": "Missing current option details"}
            
        # Extract current option details
        current_dte = self.current_option.get('days_to_expiry', 0)
        current_delta = self.current_option.get('delta', 0.3)
        current_theta = self.current_option.get('theta', 0)
        
        # Define timing thresholds based on market regime and volatility
        regime_thresholds = {
            0: {'dte': 14, 'delta': 0.85},  # Severe Bearish
            1: {'dte': 12, 'delta': 0.85},  # Bearish
            2: {'dte': 10, 'delta': 0.80},  # Weak Bearish
            3: {'dte': 7, 'delta': 0.80},   # Neutral
            4: {'dte': 7, 'delta': 0.75},   # Weak Bullish
            5: {'dte': 5, 'delta': 0.75},   # Bullish
            6: {'dte': 5, 'delta': 0.70}    # Strong Bullish
        }
        
        # Get thresholds for current regime
        thresholds = regime_thresholds.get(market_regime, {'dte': 7, 'delta': 0.80})
        
        # Adjust for volatility
        vol_adj_factor = vol_forecast / 0.20  # Normalized to "typical" vol of 20%
        
        if vol_forecast > 0.30:  # High volatility
            thresholds['dte'] += 3
            thresholds['delta'] += 0.05
        elif vol_forecast < 0.15:  # Low volatility
            thresholds['dte'] -= 2
            thresholds['delta'] -= 0.05
        
        # Calculate timing metrics
        theta_decay_rate = abs(current_theta) / self.current_option.get('premium', 1)
        time_decay_inflection = current_dte <= 21 and current_dte >= 7  # Gamma region
        
        # Determine roll trigger
        if current_delta >= thresholds['delta']:
            trigger = "delta"
        elif current_dte <= thresholds['dte']:
            trigger = "dte"
        elif time_decay_inflection and theta_decay_rate > 0.05:  # 5% daily decay
            trigger = "theta"
        else:
            trigger = None
        
        # Calculate optimal roll day
        if trigger:
            roll_day = 0  # Roll immediately
        else:
            # Calculate days until hitting delta threshold based on vol forecast
            # Simple model: delta increases by ~0.05 for each standard deviation move
            std_move_per_day = vol_forecast / math.sqrt(252)
            days_to_delta_threshold = math.floor((thresholds['delta'] - current_delta) / (0.05 * std_move_per_day))
            
            # Don't exceed DTE threshold
            days_to_delta_threshold = min(days_to_delta_threshold, current_dte - thresholds['dte'])
            
            # Ensure positive
            roll_day = max(0, days_to_delta_threshold)
        
        return {
            'current_dte': current_dte,
            'current_delta': current_delta,
            'theta_decay_rate': theta_decay_rate,
            'dte_threshold': thresholds['dte'],
            'delta_threshold': thresholds['delta'],
            'roll_trigger': trigger,
            'optimal_roll_day': roll_day,
            'recommendation': f"Roll {trigger}-triggered" if trigger else f"Roll in {roll_day} days"
        }


def covered_call_roll_analysis(
    position: Dict[str, Any], 
    current_price: float, 
    vol_forecast: float
) -> Dict[str, Any]:
    """
    Analyze roll opportunities for an existing covered call position.
    
    Parameters:
    position (dict): Current option position details
    current_price (float): Current price of underlying
    vol_forecast (float): Volatility forecast
        
    Returns:
    dict: Roll analysis and recommendations
    """
    # Extract position details
    strike = position.get('strike', 0)
    days_to_expiry = position.get('days_to_expiry', 0)
    premium = position.get('premium', 0)
    
    # Default roll targets
    roll_targets = [
        {'dte': 30, 'strikes_higher': 1},
        {'dte': 45, 'strikes_higher': 2},
        {'dte': 60, 'strikes_higher': 3}
    ]
    
    # Generate roll options
    roll_options = []
    
    for target in roll_targets:
        dte = target['dte']
        strikes_higher = target['strikes_higher']
        
        # Calculate new strike (current strike + (strikes_higher * some increment))
        # In a real scenario, you'd use the option chain to find actual available strikes
        strike_increment = current_price * 0.02  # 2% of current price as example increment
        new_strike = strike + (strikes_higher * strike_increment)
        
        # Estimate new premium using Black-Scholes approximation
        t = dte / 365
        d1 = (np.log(current_price/new_strike) + (RISK_FREE_RATE + vol_forecast**2/2) * t) / (vol_forecast * np.sqrt(t))
        d2 = d1 - vol_forecast * np.sqrt(t)
        new_premium = current_price * norm.cdf(d1) - new_strike * np.exp(-RISK_FREE_RATE * t) * norm.cdf(d2)
        
        # Calculate roll credit/debit
        roll_credit = new_premium - premium
        
        # Add to roll options
        roll_options.append({
            'dte': dte,
            'new_strike': new_strike,
            'new_premium': new_premium,
            'roll_credit': roll_credit,
            'days_added': dte - days_to_expiry
        })
    
    # Sort options by roll credit
    roll_options.sort(key=lambda x: x['roll_credit'], reverse=True)
    
    return {
        'current_position': {
            'strike': strike,
            'days_to_expiry': days_to_expiry,
            'premium': premium
        },
        'roll_options': roll_options,
        'recommendation': roll_options[0] if roll_options else None
    }