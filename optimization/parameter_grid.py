# optimization/parameter_grid.py
"""
Parameter Grid Optimization Module.

This module provides utilities for parameter grid optimization for options strategies,
including grid search, random search, and Bayesian optimization approaches.
"""

import logging
import math
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Constants
RISK_FREE_RATE = 0.02  # 2% risk-free rate


class ParameterGrid:
    """
    Parameter grid optimization for options strategies.
    
    This class provides tools to optimize strategy parameters using various
    search methods, including grid search, random search, and Bayesian optimization.
    """
    
    def __init__(
        self, 
        param_ranges: Dict[str, List[Any]], 
        objective_function: Callable,
        maximize: bool = True
    ):
        """
        Initialize the parameter grid.
        
        Parameters:
        param_ranges (dict): Dictionary of parameter names and possible values
        objective_function (callable): Function to evaluate parameters
        maximize (bool): Whether to maximize (True) or minimize (False) the objective
        """
        self.param_ranges = param_ranges
        self.objective_function = objective_function
        self.maximize = maximize
        
        # Initialize results storage
        self.results = []
        self.best_params = None
        self.best_score = -float('inf') if maximize else float('inf')
    
    def grid_search(self) -> Dict[str, Any]:
        """
        Perform grid search over parameter ranges.
        
        Returns:
        dict: Best parameters and score
        """
        # Generate parameter combinations
        param_keys = list(self.param_ranges.keys())
        param_values = list(self.param_ranges.values())
        
        # Clear previous results
        self.results = []
        self.best_params = None
        self.best_score = -float('inf') if self.maximize else float('inf')
        
        # Iterate through all combinations
        for params_tuple in product(*param_values):
            params = dict(zip(param_keys, params_tuple))
            
            # Evaluate objective function
            try:
                score = self.objective_function(**params)
                
                # Store result
                result = {
                    'params': params,
                    'score': score
                }
                self.results.append(result)
                
                # Update best parameters if better
                if (self.maximize and score > self.best_score) or \
                   (not self.maximize and score < self.best_score):
                    self.best_score = score
                    self.best_params = params.copy()
            
            except Exception as e:
                logger.warning(f"Error evaluating parameters {params}: {e}")
        
        # Return best parameters
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results
        }
    
    def random_search(self, n_iterations: int = 100) -> Dict[str, Any]:
        """
        Perform random search over parameter ranges.
        
        Parameters:
        n_iterations (int): Number of random samples to try
        
        Returns:
        dict: Best parameters and score
        """
        # Clear previous results
        self.results = []
        self.best_params = None
        self.best_score = -float('inf') if self.maximize else float('inf')
        
        # Generate random parameters
        for _ in range(n_iterations):
            params = {}
            
            # Sample each parameter
            for param, values in self.param_ranges.items():
                if isinstance(values, list):
                    # Discrete parameter
                    params[param] = np.random.choice(values)
                elif isinstance(values, tuple) and len(values) == 2:
                    # Continuous parameter (min, max)
                    params[param] = np.random.uniform(values[0], values[1])
                else:
                    raise ValueError(f"Unsupported parameter range: {values}")
            
            # Evaluate objective function
            try:
                score = self.objective_function(**params)
                
                # Store result
                result = {
                    'params': params,
                    'score': score
                }
                self.results.append(result)
                
                # Update best parameters if better
                if (self.maximize and score > self.best_score) or \
                   (not self.maximize and score < self.best_score):
                    self.best_score = score
                    self.best_params = params.copy()
            
            except Exception as e:
                logger.warning(f"Error evaluating parameters {params}: {e}")
        
        # Return best parameters
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results
        }
    
    def optimize(
        self, 
        method: str = 'grid', 
        n_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Optimize parameters using the specified method.
        
        Parameters:
        method (str): Optimization method ('grid', 'random', 'bayesian')
        n_iterations (int): Number of iterations for random or Bayesian search
        
        Returns:
        dict: Optimization results
        """
        if method == 'grid':
            return self.grid_search()
        elif method == 'random':
            return self.random_search(n_iterations=n_iterations)
        elif method == 'bayesian':
            try:
                return self.bayesian_optimization(n_iterations=n_iterations)
            except ImportError:
                logger.warning("Scikit-optimize not available, falling back to random search")
                return self.random_search(n_iterations=n_iterations)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def bayesian_optimization(
        self, 
        n_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Perform Bayesian optimization over parameter ranges.
        
        Parameters:
        n_iterations (int): Number of iterations
        
        Returns:
        dict: Best parameters and score
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
        except ImportError:
            raise ImportError("Scikit-optimize is required for Bayesian optimization")
        
        # Clear previous results
        self.results = []
        self.best_params = None
        self.best_score = -float('inf') if self.maximize else float('inf')
        
        # Define search space
        space = []
        param_keys = []
        
        for param, values in self.param_ranges.items():
            param_keys.append(param)
            
            if isinstance(values, list):
                if all(isinstance(v, int) for v in values):
                    # Discrete integer parameters
                    space.append(Integer(min(values), max(values)))
                elif all(isinstance(v, (int, float)) for v in values):
                    # Continuous parameters
                    space.append(Real(min(values), max(values)))
                else:
                    # Categorical parameters
                    space.append(Categorical(values))
            elif isinstance(values, tuple) and len(values) == 2:
                # Continuous range (min, max)
                if all(isinstance(v, int) for v in values):
                    space.append(Integer(values[0], values[1]))
                else:
                    space.append(Real(values[0], values[1]))
            else:
                raise ValueError(f"Unsupported parameter range: {values}")
        
        # Define objective function wrapper
        def objective(x):
            params = dict(zip(param_keys, x))
            try:
                score = self.objective_function(**params)
                
                # Store result
                result = {
                    'params': params,
                    'score': score
                }
                self.results.append(result)
                
                # Update best parameters if better
                if (self.maximize and score > self.best_score) or \
                   (not self.maximize and score < self.best_score):
                    self.best_score = score
                    self.best_params = params.copy()
                
                return -score if self.maximize else score
            
            except Exception as e:
                logger.warning(f"Error evaluating parameters {params}: {e}")
                return 0 if self.maximize else float('inf')
        
        # Run optimization
        result = gp_minimize(
            objective,
            space,
            n_calls=n_iterations,
            random_state=42
        )
        
        # Return best parameters
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results,
            'skopt_result': result
        }