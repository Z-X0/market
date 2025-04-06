# Create market analyzer if not provided
        if market_analyzer is None and weekly_data is not None and not weekly_data.empty:
            try:
                from analysis.market_regime import MarketRegimeAnalyzer
                market_analyzer = MarketRegimeAnalyzer(weekly_data)
            except ImportError:
                logger.warning("Could not import MarketRegimeAnalyzer")
        
        # Check if we have valid data
        if weekly_data is None or weekly_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F9FA')
            ax.text(0.5, 0.5, f"No weekly data available for {symbol}", 
                   ha='center', va='center', fontsize=12)
            return fig
        
        if market_analyzer is None:
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F9FA')
            ax.text(0.5, 0.5, f"No market regime analyzer available for {symbol}", 
                   ha='center', va='center', fontsize=12)
            return fig
        
        # Create a figure with better styling
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True, 
                                gridspec_kw={'height_ratios': [3, 1]},
                                facecolor='#F8F9FA')
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Price chart with moving averages
        axs[0].plot(weekly_data.index, weekly_data['Close'], color='#0066CC', linewidth=2, label='Close Price')
        
        # Add moving averages
        weekly_data['MA50'] = weekly_data['Close'].rolling(window=50).mean()
        weekly_data['MA200'] = weekly_data['Close'].rolling(window=200).mean()
        axs[0].plot(weekly_data.index, weekly_data['MA50'], color='#FF9900', linewidth=1.5, 
                   linestyle='--', label='50-week MA')
        axs[0].plot(weekly_data.index, weekly_data['MA200'], color='#CC0000', linewidth=1.5, 
                   linestyle='--', label='200-week MA')
        
        # Configure price chart
        axs[0].set_title(f"{symbol} - Price Action and Regime Detection", 
                        fontsize=14, fontweight='bold')
        axs[0].legend(loc='upper left', frameon=True)
        axs[0].set_ylabel('Price ($)', fontsize=12)
        axs[0].grid(True, alpha=0.3)
        
        # Format y-axis as currency
        axs[0].yaxis.set_major_formatter('${x:,.2f}')
        
        # Regime chart
        reg_ser = market_analyzer.get_regime_series()
        
        if reg_ser.empty:
            axs[1].text(0.5, 0.5, "No regime data available", 
                      ha='center', va='center', fontsize=10)
        else:
            # Map regimes to colors
            regime_colors = {
                0: '#8B0000',  # Severe Bearish - dark red
                1: '#FF4136',  # Bearish - red
                2: '#FF851B',  # Weak Bearish - orange
                3: '#FFDC00',  # Neutral - yellow
                4: '#2ECC40',  # Weak Bullish - light green
                5: '#008000',  # Bullish - green
                6: '#006400'   # Strong Bullish - dark green
            }
            
            # Create colorful regime visualization
            for regime in sorted(set(market_analyzer.regimes)):
                mask = reg_ser == regime
                if not any(mask):
                    continue
                regime_periods = mask.index[mask]
                axs[1].fill_between(regime_periods, [regime] * len(regime_periods), 
                                  [regime+0.8] * len(regime_periods),
                                  color=regime_colors.get(regime, '#CCCCCC'), 
                                  alpha=0.8, label=f"Regime {regime}")
            
            # Configure regime chart
            regime_names = {
                0: "Severe Bearish", 1: "Bearish", 2: "Weak Bearish", 
                3: "Neutral", 4: "Weak Bullish", 5: "Bullish", 6: "Strong Bullish"
            }
            
            axs[1].set_yticks(np.arange(0.4, 7, 1))
            axs[1].set_yticklabels([regime_names.get(i, f"Regime {i}") for i in range(7)])
            axs[1].set_ylabel('Market Regime', fontsize=12)
            axs[1].set_xlabel('Date', fontsize=12)
            axs[1].grid(False)
            
            # Format x-axis dates
            fig.autofmt_xdate()
            
            # Add current regime highlight
            current_regime = market_analyzer.get_current_regime()
            axs[1].text(0.02, 0.5, f"Current: {regime_names.get(current_regime, f'Regime {current_regime}')}",
                     transform=axs[1].transAxes, fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
            plt.tight_layout()
            return fig
    
        except Exception as e:
            logger.error(f"Error generating regime chart: {e}")
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F9FA')
            ax.text(0.5, 0.5, f"Unable to generate regime chart: {e}", 
                ha='center', va='center', fontsize=12)
        return fig


def generate_vol_comparison_chart(
    pos_data: Dict[str, Dict[str, Any]], 
    portfolio: Dict[str, int]
) -> plt.Figure:
    """
    Generate a visually appealing chart comparing different volatility models.
    
    Parameters:
    pos_data (dict): Position data with volatility information
    portfolio (dict): Portfolio positions
    
    Returns:
    matplotlib.figure.Figure: Generated figure
    """
    try:
        syms = list(portfolio.keys())
        hv = []
        ewma = []
        garch = []
        ensemble = []
        
        for s in syms:
            if s not in pos_data or not pos_data[s] or 'volatility' not in pos_data[s]:
                hv.append(0)
                ewma.append(0)
                garch.append(0)
                ensemble.append(0)
                continue
            
            vdict = pos_data[s]['volatility']['model_breakdown']
            hv.append(vdict.get('historical', 0))
            ewma.append(vdict.get('ewma', 0))
            garch.append(vdict.get('garch', 0))
            ensemble.append(vdict.get('ensemble', 0))
        
        x = np.arange(len(syms))
        width = 0.2
        
        # Create a more visually appealing chart
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F9FA')
        
        # Use a better color palette
        colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']
        
        rects1 = ax.bar(x - width*1.5, hv, width, label='Historical', color=colors[0], alpha=0.8, 
                      edgecolor='white', linewidth=1)
        rects2 = ax.bar(x - width/2, ewma, width, label='EWMA', color=colors[1], alpha=0.8, 
                      edgecolor='white', linewidth=1)
        rects3 = ax.bar(x + width/2, garch, width, label='GARCH', color=colors[2], alpha=0.8, 
                      edgecolor='white', linewidth=1)
        rects4 = ax.bar(x + width*1.5, ensemble, width, label='Ensemble', color=colors[3], alpha=0.8, 
                      edgecolor='white', linewidth=1)
        
        # Add data labels on top of bars
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0.01:  # Only add labels for visible bars
                    ax.annotate(f'{height:.0%}',
                              xy=(rect.get_x() + rect.get_width() / 2, height),
                              xytext=(0, 3),  # 3 points vertical offset
                              textcoords="offset points",
                              ha='center', va='bottom',
                              fontsize=8)
        
        add_labels(rects1)
        add_labels(rects2)
        add_labels(rects3)
        add_labels(rects4)
        
        # Improve chart styling
        ax.set_title("Volatility Model Comparison", fontsize=14, fontweight='bold')
        ax.set_xlabel("Symbol", fontsize=12)
        ax.set_ylabel("Annualized Volatility", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(syms, fontweight='bold')
        
        # Add subtle horizontal grid lines
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Enhance legend
        ax.legend(title="Volatility Models", frameon=True, fancybox=True, framealpha=0.8, 
                fontsize=10, title_fontsize=12, loc='upper right')
        
        # Add a subtle border
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(0.8)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Error generating vol comparison chart: {e}")
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F9FA')
        ax.text(0.5, 0.5, f"Unable to generate volatility comparison chart: {e}", 
               ha='center', va='center', fontsize=12)
        return fig


def generate_option_chain_analysis(
    option_chain: List[Dict[str, Any]], 
    current_price: float, 
    mode: str = 'covered_call'
) -> plt.Figure:
    """
    Generate a visually appealing analysis chart for option chain.
    
    Parameters:
    option_chain (list): Option chain data
    current_price (float): Current price of the underlying
    mode (str): 'covered_call' or 'cash_secured_put'
    
    Returns:
    matplotlib.figure.Figure: Generated figure
    """
    if not option_chain:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F9FA')
        ax.text(0.5, 0.5, "No option chain data available", 
               ha='center', va='center', fontsize=12)
        return fig
    
    try:
        # Convert to DataFrame
        chain_df = pd.DataFrame(option_chain)
        
        # Calculate additional metrics
        chain_df['moneyness'] = chain_df['strike'] / current_price
        chain_df['premium_pct'] = chain_df['mid'] / chain_df['strike'] * 100
        
        # For covered calls, calculate upside potential
        if mode == 'covered_call':
            chain_df['upside_potential'] = (chain_df['strike'] - current_price) / current_price * 100
            chain_df['static_return'] = chain_df['mid'] / current_price * 100
        else:  # cash secured puts
            chain_df['downside_protection'] = (current_price - chain_df['strike']) / current_price * 100
            chain_df['static_return'] = chain_df['mid'] / chain_df['strike'] * 100
        
        # Create figure with improved styling
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax1 = plt.subplots(figsize=(10, 6), facecolor='#F8F9FA')
        ax2 = ax1.twinx()
        
        # Sort by strike
        chain_df = chain_df.sort_values('strike')
        
        # Define color palette based on moneyness
        def get_bar_color(row):
            if mode == 'covered_call':
                if row['moneyness'] < 0.98:  # ITM
                    return '#27AE60'  # Green
                elif row['moneyness'] > 1.02:  # OTM
                    return '#3498DB'  # Blue
                else:  # ATM
                    return '#F39C12'  # Orange
            else:  # cash secured puts
                if row['moneyness'] < 0.98:  # OTM for puts
                    return '#3498DB'  # Blue
                elif row['moneyness'] > 1.02:  # ITM for puts
                    return '#27AE60'  # Green
                else:  # ATM
                    return '#F39C12'  # Orange
        
        # Apply colors
        chain_df['color'] = chain_df.apply(get_bar_color, axis=1)
        
        # Plot premium and IV
        if mode == 'covered_call':
            bars = ax1.bar(chain_df['strike'], chain_df['static_return'], 
                         color=chain_df['color'], alpha=0.7)
            ax2.plot(chain_df['strike'], chain_df['implied_vol'] * 100, 'r-', marker='o', 
                   markersize=6, linewidth=2, label='Implied Vol')
            
            # Add vertical line at current price
            ax1.axvline(current_price, color='black', linestyle='--', alpha=0.7)
            ax1.annotate(f"Current: ${current_price:.2f}", 
                       xy=(current_price, 0),
                       xytext=(current_price, ax1.get_ylim()[1] * 0.2),
                       arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
        else:  # cash secured puts
            bars = ax1.bar(chain_df['strike'], chain_df['static_return'], 
                         color=chain_df['color'], alpha=0.7)
            ax2.plot(chain_df['strike'], chain_df['implied_vol'] * 100, 'r-', marker='o', 
                   markersize=6, linewidth=2, label='Implied Vol')
            
            # Add vertical line at current price
            ax1.axvline(current_price, color='black', linestyle='--', alpha=0.7)
            ax1.annotate(f"Current: ${current_price:.2f}", 
                       xy=(current_price, 0),
                       xytext=(current_price, ax1.get_ylim()[1] * 0.2),
                       arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Add data labels to bars
        for bar in bars:
            height = bar.get_height()
            if height > 0.5:  # Only label significant bars
                ax1.annotate(f"{height:.1f}%",
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        # Enhance chart styling
        strategy_name = "Covered Call" if mode == 'covered_call' else "Cash Secured Put"
        ax1.set_title(f"Option Chain Analysis: {strategy_name} Strategy", 
                    fontsize=14, fontweight='bold')
        ax1.set_xlabel("Strike Price ($)", fontsize=12)
        ax1.set_ylabel(f"Premium % of {'Current Price' if mode == 'covered_call' else 'Strike'}", 
                     color='b', fontsize=12)
        ax2.set_ylabel("Implied Volatility %", color='r', fontsize=12)
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#27AE60', alpha=0.7, label='ITM'),
            Patch(facecolor='#F39C12', alpha=0.7, label='ATM'),
            Patch(facecolor='#3498DB', alpha=0.7, label='OTM'),
            Patch(facecolor='red', alpha=0.0, label='Implied Vol', hatch='/')
        ]
        ax1.legend(handles=legend_elements, loc='upper left', frameon=True)
        
        # Format axes
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}%'.format(y)))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}%'.format(y)))
        
        # Add a subtle grid
        ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
        ax1.set_axisbelow(True)
        
        # Add a subtle border
        for spine in ax1.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(0.8)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Error generating option chain analysis: {e}")
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F9FA')
        ax.text(0.5, 0.5, f"Unable to generate option chain analysis: {e}", 
               ha='center', va='center', fontsize=12)
        return fig


def generate_backtest_pnl_chart(
    bt: Dict[str, Any]
) -> plt.Figure:
    """
    Generate a visually appealing chart showing backtest P&L.
    
    Parameters:
    bt (dict): Backtest results
    
    Returns:
    matplotlib.figure.Figure: Generated figure
    """
    if not bt or 'daily_index' not in bt or 'daily_cumulative_pnl' not in bt:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F9FA')
        ax.text(0.5, 0.5, "No backtest data available", 
               ha='center', va='center', fontsize=12)
        return fig
    
    try:
        x = bt['daily_index']
        y = bt['daily_cumulative_pnl']
        
        # Fix for mismatched array lengths
        if len(x) != len(y):
            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]
        
        # Create a more visually appealing chart
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F9FA')
        
        # Plot with improved styling
        ax.plot(x, y, color='#2980B9', linewidth=2.5, alpha=0.8)
        
        # Add shaded area below the line
        ax.fill_between(x, 0, y, color='#2980B9', alpha=0.2)
        
        # Add markers at key points
        # First, last, max, min points
        first_idx, last_idx = 0, len(y) - 1
        max_idx = np.argmax(y)
        min_idx = np.argmin(y)
        
        key_points = [(first_idx, 'First'), (last_idx, 'Last'), 
                     (max_idx, 'Max'), (min_idx, 'Min')]
        
        for idx, label in key_points:
            ax.plot(x[idx], y[idx], 'o', markersize=8, 
                  markerfacecolor='white', markeredgecolor='#E74C3C', markeredgewidth=2)
            
            # Add annotations with formatted values
            ax.annotate(f"{label}: ${y[idx]:,.0f}", 
                      xy=(x[idx], y[idx]),
                      xytext=(0, 10) if label in ['Max', 'Last'] else (0, -25),
                      textcoords="offset points",
                      ha='center',
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Calculate and display key metrics
        if 'total_return' in bt:
            total_ret = bt['total_return']
            ann_ret = bt.get('annualized_return', 0)
            max_dd = bt.get('max_drawdown', 0) if 'max_drawdown' in bt else None
            
            metrics_text = (f"Total Return: {total_ret:.2%}  •  "
                          f"Ann. Return: {ann_ret:.2%}")
            
            if max_dd is not None:
                metrics_text += f"  •  Max Drawdown: {max_dd:.2%}"
                
            # Add metrics box
            ax.text(0.5, 0.02, metrics_text, 
                  transform=ax.transAxes,
                  ha='center', va='bottom',
                  bbox=dict(boxstyle="round,pad=0.5", fc="#F8F9FA", ec="#CCCCCC", alpha=0.9))
        
        # Enhance chart styling
        symbol = bt.get('symbol', 'Portfolio')
        ax.set_title(f"{symbol} Covered Call Strategy Performance", 
                   fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Cumulative P&L ($)", fontsize=12)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '${:,.0f}'.format(y)))
        
        # Add subtle grid lines
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Add a subtle border
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(0.8)
        
        # Auto-format date axis
        fig.autofmt_xdate()
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Error generating backtest PnL chart: {e}")
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F9FA')
        ax.text(0.5, 0.5, f"Unable to generate backtest chart: {e}", 
               ha='center', va='center', fontsize=12)
        return fig


def generate_regime_performance_chart(
    metrics_by_regime: Dict[int, Dict[str, float]], 
    title: str = "Performance by Regime"
) -> plt.Figure:
    """
    Generate a visually appealing chart showing performance metrics by regime.
    
    Parameters:
    metrics_by_regime (dict): Performance metrics by regime
    title (str): Chart title
    
    Returns:
    matplotlib.figure.Figure: Generated figure
    """
    if not metrics_by_regime:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F9FA')
        ax.text(0.5, 0.5, "No regime performance data available", 
               ha='center', va='center', fontsize=12)
        return fig
    
    try:
        # Extract regime numbers and metrics
        regimes = sorted(metrics_by_regime.keys())
        
        # Get regime names
        regime_names = {
            0: "Severe Bearish", 1: "Bearish", 2: "Weak Bearish", 
            3: "Neutral", 4: "Weak Bullish", 5: "Bullish", 6: "Strong Bullish"
        }
        
        # Create a more visually appealing chart
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='#F8F9FA')
        
        # Extract metrics
        metrics = ['annualized_return', 'annualized_volatility', 'sharpe_ratio']
        metric_labels = ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio']
        
        # Prepare data in a more visual format
        data = []
        for r in regimes:
            name = regime_names.get(int(r), f"Regime {r}")
            values = [metrics_by_regime[r].get(m, 0) for m in metrics]
            data.append([name] + values)
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(data, columns=['Regime'] + metric_labels)
        
        # Prepare for grouped bar chart
        x = np.arange(len(df))
        width = 0.25
        
        # Define color palette based on metric type
        colors = ['#3498DB', '#E74C3C', '#2ECC71']
        
        # Create grouped bar chart
        for i, (metric, label) in enumerate(zip(metric_labels, metric_labels)):
            values = df[label].values
            
            # For returns and volatility, scale percentages
            if metric in ['Annualized Return', 'Annualized Volatility']:
                values = [v * 100 for v in values]  # Convert to percentage
            
            bars = ax.bar(x + (i - 1) * width, values, width, label=label, color=colors[i], alpha=0.8)
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                if abs(height) > 0.1:  # Only label significant bars
                    if metric in ['Annualized Return', 'Annualized Volatility']:
                        label_text = f"{height:.1f}%"
                    else:  # Sharpe ratio
                        label_text = f"{height:.2f}"
                        
                    ax.annotate(label_text,
                              xy=(bar.get_x() + bar.get_width()/2, height),
                              xytext=(0, 3 if height > 0 else -10),  # offset
                              textcoords="offset points",
                              ha='center', va='bottom' if height > 0 else 'top',
                              fontsize=9)
        
        # Enhance chart styling
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Regime'], fontsize=10, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12)
        
        # Add a horizontal line at 0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add a subtle grid
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        # Add a subtle border
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(0.8)
        
        # Enhance legend
        ax.legend(title="Metrics", frameon=True, fancybox=True, 
                fontsize=10, title_fontsize=12,
                loc='upper right')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Error generating regime performance chart: {e}")
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F9FA')
        ax.text(0.5, 0.5, f"Unable to generate regime performance chart: {e}", 
               ha='center', va='center', fontsize=12)
        return fig


def generate_volatility_surface_chart(
    volatility_data: Dict[str, Any]
) -> plt.Figure:
    """
    Generate a 3D volatility surface visualization.
    
    Parameters:
    volatility_data (dict): Volatility surface data
    
    Returns:
    matplotlib.figure.Figure: Generated figure
    """
    if not volatility_data or 'x' not in volatility_data or 'y' not in volatility_data or 'z' not in volatility_data:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F9FA')
        ax.text(0.5, 0.5, "No volatility surface data available", 
               ha='center', va='center', fontsize=12)
        return fig
    
    try:
        # Extract data
        X = np.array(volatility_data['x'])
        Y = np.array(volatility_data['y'])
        Z = np.array(volatility_data['z'])
        
        # Create 3D figure
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(12, 8), facecolor='#F8F9FA')
        ax = fig.add_subplot(111, projection='3d')
        
        # Create surface plot
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.8)
        
        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Implied Volatility')
        
        # Enhance chart styling
        ax.set_title("Implied Volatility Surface", fontsize=16, fontweight='bold')
        ax.set_xlabel(volatility_data.get('x_label', 'Days to Expiration'), fontsize=12)
        ax.set_ylabel(volatility_data.get('y_label', 'Moneyness'), fontsize=12)
        ax.set_zlabel(volatility_data.get('z_label', 'Implied Volatility'), fontsize=12)
        
        # Customize view angle
        ax.view_init(30, 35)
        
        # Adjust grid and background
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Error generating volatility surface chart: {e}")
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F9FA')
        ax.text(0.5, 0.5, f"Unable to generate volatility surface chart: {e}", 
               ha='center', va='center', fontsize=12)
        return fig


def generate_roll_strategy_chart(
    roll_options: List[Dict[str, Any]],
    current_strike: float,
    current_days: int,
    current_price: float
) -> plt.Figure:
    """
    Generate a chart showing roll strategy options.
    
    Parameters:
    roll_options (list): List of roll options
    current_strike (float): Current strike price
    current_days (int): Current days to expiry
    current_price (float): Current price
    
    Returns:
    matplotlib.figure.Figure: Generated figure
    """
    if not roll_options:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F9FA')
        ax.text(0.5, 0.5, "No roll strategy options available", 
               ha='center', va='center', fontsize=12)
        return fig
    
    try:
        # Extract data
        strikes = [opt.get('strike', 0) for opt in roll_options]
        if not strikes:
            strikes = [opt.get('new_strike', 0) for opt in roll_options]
            
        dtes = [opt.get('dte', 0) for opt in roll_options]
        if not dtes:
            dtes = [current_days + opt.get('days_added', 0) for opt in roll_options]
            
        scores = [opt.get('score', 0) for opt in roll_options]
        roll_credits = [opt.get('roll_credit', 0) for opt in roll_options]
        
        # Create a scatter plot with color gradient by score
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F9FA')
        
        # Create scatter plot
        scatter = ax.scatter(dtes, strikes, c=scores, cmap='viridis', 
                           s=abs(np.array(roll_credits)) * 500, alpha=0.7)
        
        # Add color bar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Score', rotation=270, labelpad=20)
        
        # Highlight current position
        ax.scatter([current_days], [current_strike], c='red', s=200, marker='*', 
                 label='Current Position')
        
        # Add current price line
        ax.axhline(current_price, color='red', linestyle='--', alpha=0.5, 
                 label=f'Current Price: ${current_price:.2f}')
        
        # Add labels for top options
        top_indices = np.argsort(scores)[-3:]  # Top 3 scores
        for idx in top_indices:
            ax.annotate(f"Roll Credit: ${roll_credits[idx]:.2f}",
                      xy=(dtes[idx], strikes[idx]),
                      xytext=(10, 10),
                      textcoords="offset points",
                      fontsize=9,
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Enhance chart styling
        ax.set_title("Roll Strategy Options", fontsize=16, fontweight='bold')
        ax.set_xlabel("Days to Expiration", fontsize=12)
        ax.set_ylabel("Strike Price ($)", fontsize=12)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '${:,.2f}'.format(y)))
        
        # Add a grid
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Add a legend
        ax.legend(loc='best')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Error generating roll strategy chart: {e}")
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F9FA')
        ax.text(0.5, 0.5, f"Unable to generate roll strategy chart: {e}", 
               ha='center', va='center', fontsize=12)
        return fig
# reporting/chart_generator.py
"""
Chart Generator Module.

This module provides functions for generating various charts and visualizations
for options strategy analysis and reporting.
"""

import logging
import math
from datetime import datetime, date
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)


def convert_keys(obj: Any) -> Any:
    """
    Convert keys for JSON serialization.
    
    Parameters:
    obj: Object to convert
    
    Returns:
    Object with converted keys
    """
    if isinstance(obj, dict):
        return {str(k): convert_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys(i) for i in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, pd.Series, pd.DataFrame)):
        return obj.tolist() if hasattr(obj, 'tolist') else obj.to_dict()
    elif isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d')
    elif isinstance(obj, (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(obj).strftime('%Y-%m-%d')
    return obj


def generate_regime_chart(
    symbol: str, 
    market_analyzer: Optional[Any] = None, 
    weekly_data: Optional[pd.DataFrame] = None
) -> plt.Figure:
    """
    Generate a visually appealing chart showing price and regime classification.
    
    Parameters:
    symbol (str): Symbol to chart
    market_analyzer (MarketRegimeAnalyzer, optional): Market regime analyzer
    weekly_data (pandas.DataFrame, optional): Weekly price data
    
    Returns:
    matplotlib.figure.Figure: Generated figure
    """
    try:
        # Get data if not provided
        if weekly_data is None or weekly_data.empty:
            try:
                from data.fetcher import get_stock_history_weekly
                weekly_data = get_stock_history_weekly(symbol)
            except ImportError:
                logger.warning("Could not import fetcher module")
                
        # Create market analyzer if not provided