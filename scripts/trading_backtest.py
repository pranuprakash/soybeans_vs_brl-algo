import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from itertools import product
warnings.filterwarnings('ignore')

def calculate_portfolio_returns(df, initial_cash=10000, position_size=0.5, transaction_cost=0.001):
    """
    Calculate portfolio returns with proper position management
    
    Position interpretation:
    - signal = +1: Long USD position (expect USD to strengthen vs BRL) 
    - signal = -1: Long BRL position (expect BRL to strengthen vs USD)
    - signal = 0: Flat/Exit position
    
    Returns are calculated based on USD/BRL price movements
    """
    df = df.copy()
    
    # Initialize portfolio tracking
    df['portfolio_value'] = initial_cash
    df['position'] = 0.0
    df['returns'] = 0.0
    df['trade_pnl'] = 0.0
    
    current_position = 0.0
    current_value = initial_cash
    entry_price = 0.0
    
    for i in range(1, len(df)):
        current_price = df.iloc[i]['usdbrl']
        prev_price = df.iloc[i-1]['usdbrl']
        signal = df.iloc[i]['signal']
        
        if pd.isna(current_price) or pd.isna(prev_price):
            df.iloc[i, df.columns.get_loc('portfolio_value')] = current_value
            df.iloc[i, df.columns.get_loc('position')] = current_position
            continue
        
        # Calculate daily P&L from existing position
        daily_pnl = 0.0
        if current_position != 0:
            price_change = (current_price - prev_price) / prev_price
            # Long USD position profits when USD strengthens (BRL weakens, usdbrl decreases)
            # Long BRL position profits when BRL strengthens (USD weakens, usdbrl increases)
            position_pnl = -current_position * price_change * current_value * position_size
            daily_pnl += position_pnl
        
        # Handle new signals
        if signal != 0:
            # Close existing position first
            if current_position != 0:
                # Transaction cost for closing
                transaction_cost_amount = abs(current_position) * current_value * position_size * transaction_cost
                daily_pnl -= transaction_cost_amount
            
            # Open new position
            current_position = signal
            entry_price = current_price
            # Transaction cost for opening
            transaction_cost_amount = abs(current_position) * current_value * position_size * transaction_cost
            daily_pnl -= transaction_cost_amount
        
        # Update portfolio value
        current_value += daily_pnl
        daily_return = daily_pnl / df.iloc[i-1]['portfolio_value'] if df.iloc[i-1]['portfolio_value'] > 0 else 0
        
        df.iloc[i, df.columns.get_loc('portfolio_value')] = current_value
        df.iloc[i, df.columns.get_loc('position')] = current_position
        df.iloc[i, df.columns.get_loc('returns')] = daily_return
        df.iloc[i, df.columns.get_loc('trade_pnl')] = daily_pnl
    
    # Calculate cumulative returns
    df['cumulative_returns'] = ((df['portfolio_value'] / initial_cash) - 1) * 100
    
    return df

def calculate_risk_metrics(portfolio_df, trading_days_per_year=252):
    """Calculate comprehensive risk metrics for the portfolio"""
    returns = portfolio_df['returns'].dropna()
    
    if len(returns) == 0 or portfolio_df['portfolio_value'].iloc[-1] <= 0:
        return {
            'Total Return (%)': -100,
            'Annualized Return (%)': -100,
            'Annualized Volatility (%)': 0,
            'Sharpe Ratio': -10,
            'Max Drawdown (%)': -100,
            'Calmar Ratio': 0,
            'Win Rate (%)': 0,
            'Profit Factor': 0,
            'Trading Days': len(returns)
        }
    
    # Basic return metrics
    total_return = (portfolio_df['portfolio_value'].iloc[-1] / portfolio_df['portfolio_value'].iloc[0]) - 1
    annualized_return = (1 + total_return) ** (trading_days_per_year / len(returns)) - 1
    annualized_vol = returns.std() * np.sqrt(trading_days_per_year)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
    
    # Drawdown calculation
    portfolio_values = portfolio_df['portfolio_value']
    rolling_max = portfolio_values.expanding().max()
    drawdowns = (portfolio_values - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Trading statistics
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]
    win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
    
    # Profit factor
    gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
    gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
    
    return {
        'Total Return (%)': total_return * 100,
        'Annualized Return (%)': annualized_return * 100,
        'Annualized Volatility (%)': annualized_vol * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown * 100,
        'Calmar Ratio': calmar_ratio,
        'Win Rate (%)': win_rate * 100,
        'Profit Factor': profit_factor,
        'Trading Days': len(returns)
    }

def analyze_trades(df):
    """Analyze trading activity and patterns"""
    signals = df['signal'][df['signal'] != 0]
    
    if len(signals) == 0:
        return {
            'Total Trades': 0,
            'Long USD Trades': 0,
            'Long BRL Trades': 0,
            'Avg Holding Period': 0,
            'Market Exposure (%)': 0
        }
    
    trade_count = len(signals)
    long_usd_trades = len(signals[signals > 0])
    long_brl_trades = len(signals[signals < 0])
    
    # Calculate exposure
    in_position = df['position'] != 0
    exposure_days = in_position.sum()
    total_days = len(df)
    market_exposure = (exposure_days / total_days) * 100 if total_days > 0 else 0
    
    # Average holding period
    avg_holding = exposure_days / trade_count if trade_count > 0 else 0
    
    return {
        'Total Trades': trade_count,
        'Long USD Trades': long_usd_trades,
        'Long BRL Trades': long_brl_trades,
        'Avg Holding Period': avg_holding,
        'Market Exposure (%)': market_exposure
    }

def generate_signals_for_optimization(df, z_threshold=2.0, max_holding_days=10):
    """Local implementation of signal generation for parameter optimization"""
    df = df.copy()
    df['signal'] = 0
    df['position'] = 0
    df['days_held'] = 0
    
    position = 0
    days_held = 0
    
    for i in range(1, len(df)):
        current_z = df.iloc[i]['z_score'] if 'z_score' in df.columns else 0
        
        if pd.isna(current_z):
            df.iloc[i, df.columns.get_loc('signal')] = 0
            df.iloc[i, df.columns.get_loc('position')] = position
            df.iloc[i, df.columns.get_loc('days_held')] = days_held
            continue
        
        # Exit conditions
        if position != 0:
            days_held += 1
            if abs(current_z) > 3.5 or days_held >= max_holding_days:
                df.iloc[i, df.columns.get_loc('signal')] = -position
                position = 0
                days_held = 0
            elif (position == 1 and current_z > -0.5) or (position == -1 and current_z < 0.5):
                df.iloc[i, df.columns.get_loc('signal')] = -position
                position = 0
                days_held = 0
        
        # Entry conditions
        if position == 0:
            if current_z > z_threshold:
                position = -1
                days_held = 0
                df.iloc[i, df.columns.get_loc('signal')] = position
            elif current_z < -z_threshold:
                position = 1
                days_held = 0
                df.iloc[i, df.columns.get_loc('signal')] = position
        
        df.iloc[i, df.columns.get_loc('position')] = position
        df.iloc[i, df.columns.get_loc('days_held')] = days_held
    
    return df

def backtest_strategy(df_full, z_threshold=2.0, max_holding_days=10, target_year=2024):
    """
    Backtest strategy with given parameters on target year only
    Returns risk metrics for optimization
    """
    # Generate signals with parameters
    df_signals = generate_signals_for_optimization(df_full, z_threshold=z_threshold, max_holding_days=max_holding_days)
    
    # Filter to target year
    df_2024 = df_signals[f"{target_year}-01-02":f"{target_year}-12-31"].copy()
    
    if len(df_2024) == 0:
        return {'Total Return (%)': -100, 'Sharpe Ratio': -10, 'Max Drawdown (%)': -100}
    
    # Run portfolio simulation
    df_portfolio = calculate_portfolio_returns(df_2024, initial_cash=10000, position_size=0.3)
    
    # Calculate metrics
    risk_metrics = calculate_risk_metrics(df_portfolio)
    trade_metrics = analyze_trades(df_2024)
    
    # Combine metrics
    all_metrics = {**risk_metrics, **trade_metrics}
    
    return all_metrics

def parameter_optimization_heatmap(df_full):
    """
    Section 7: Heat-map parameter grid
    Sweep: band ∈ {1.5, 2, 2.5} × holding_max ∈ 5‥20
    Score only on 2024 data
    """
    print("Running parameter optimization heat-map...")
    
    # Parameter ranges
    z_thresholds = [1.5, 2.0, 2.5]
    holding_periods = list(range(5, 21))  # 5 to 20 days
    
    # Results storage
    results = []
    
    # Progress tracking
    total_combinations = len(z_thresholds) * len(holding_periods)
    current = 0
    
    for z_threshold in z_thresholds:
        for holding_days in holding_periods:
            current += 1
            print(f"Testing {current}/{total_combinations}: z={z_threshold}, holding={holding_days}")
            
            try:
                metrics = backtest_strategy(df_full, z_threshold=z_threshold, 
                                          max_holding_days=holding_days, target_year=2024)
                
                results.append({
                    'z_threshold': z_threshold,
                    'max_holding_days': holding_days,
                    'total_return': metrics.get('Total Return (%)', -100),
                    'sharpe_ratio': metrics.get('Sharpe Ratio', -10),
                    'max_drawdown': metrics.get('Max Drawdown (%)', -100),
                    'total_trades': metrics.get('Total Trades', 0),
                    'win_rate': metrics.get('Win Rate (%)', 0),
                    'profit_factor': metrics.get('Profit Factor', 0)
                })
            except Exception as e:
                print(f"Error with z={z_threshold}, holding={holding_days}: {e}")
                results.append({
                    'z_threshold': z_threshold,
                    'max_holding_days': holding_days,
                    'total_return': -100,
                    'sharpe_ratio': -10,
                    'max_drawdown': -100,
                    'total_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0
                })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv("results/parameter_optimization.csv", index=False)
    
    # Create heat-map
    create_heatmap_visualizations(results_df)
    
    # Find best parameters
    best_return = results_df.loc[results_df['total_return'].idxmax()]
    best_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
    
    print(f"\nBest Total Return: {best_return['total_return']:.2f}% (z={best_return['z_threshold']}, holding={best_return['max_holding_days']})")
    print(f"Best Sharpe Ratio: {best_sharpe['sharpe_ratio']:.3f} (z={best_sharpe['z_threshold']}, holding={best_sharpe['max_holding_days']})")
    
    return results_df, best_return, best_sharpe

def create_heatmap_visualizations(results_df):
    """Create heat-map visualizations for parameter optimization"""
    
    # Prepare data for heat-maps
    metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'total_trades']
    titles = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Total Trades']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        # Pivot data for heatmap
        pivot_data = results_df.pivot(index='max_holding_days', columns='z_threshold', values=metric)
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=0 if metric != 'total_trades' else None,
                   ax=axes[i], cbar_kws={'label': title})
        
        axes[i].set_title(f'Parameter Optimization: {title}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Z-Score Threshold')
        axes[i].set_ylabel('Max Holding Days')
    
    plt.tight_layout()
    plt.savefig("results/parameter_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Heat-map saved to results/parameter_heatmap.png")

def create_visualizations(df, df_2024, strategy_name, output_dir="results"):
    """Create comprehensive visualizations"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Rolling Beta and Z-Score Analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Rolling beta plot
    if 'beta' in df.columns:
        beta_data = df['beta'].dropna()
        ax1.plot(beta_data.index, beta_data.values, linewidth=1, alpha=0.7, color='darkblue')
        ax1.axhline(y=beta_data.mean(), color='red', linestyle='--', alpha=0.7, 
                   label=f'Mean β = {beta_data.mean():.3f}')
        ax1.set_title('Rolling 250-Day Beta: USD/BRL vs Soybean Prices', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Beta Coefficient')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Z-score evolution
    if 'z_score' in df.columns:
        zscore_data = df['z_score'].dropna()
        ax2.plot(zscore_data.index, zscore_data.values, linewidth=1, alpha=0.7, color='navy')
        ax2.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='±2σ Bands')
        ax2.axhline(y=-2, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=1.5, color='orange', linestyle='--', alpha=0.7, label='±1.5σ Fallback')
        ax2.axhline(y=-1.5, color='orange', linestyle='--', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Z-Score Evolution (Regression Residuals)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Z-Score')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'rolling_beta_zscore.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Z-Score Histogram
    if 'z_score' in df.columns:
        plt.figure(figsize=(10, 6))
        zscore_clean = df['z_score'].dropna()
        plt.hist(zscore_clean, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        plt.axvline(x=2, color='red', linestyle='--', label='2σ Threshold')
        plt.axvline(x=-2, color='red', linestyle='--')
        plt.axvline(x=1.5, color='orange', linestyle='--', label='1.5σ Fallback')
        plt.axvline(x=-1.5, color='orange', linestyle='--')
        plt.title('Distribution of Z-Scores (Regression Residuals)', fontsize=14, fontweight='bold')
        plt.xlabel('Z-Score')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'zscore_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 2024 Trading Activity
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Price and signals
    ax1.plot(df_2024.index, df_2024['usdbrl'], linewidth=2, label='USD/BRL', color='blue')
    signal_dates = df_2024[df_2024['signal'] != 0]
    long_usd_signals = signal_dates[signal_dates['signal'] > 0]
    long_brl_signals = signal_dates[signal_dates['signal'] < 0]
    
    if len(long_usd_signals) > 0:
        ax1.scatter(long_usd_signals.index, long_usd_signals['usdbrl'], color='green', s=100, marker='^', 
                   label=f'Long USD ({len(long_usd_signals)} trades)', zorder=5)
    if len(long_brl_signals) > 0:
        ax1.scatter(long_brl_signals.index, long_brl_signals['usdbrl'], color='red', s=100, marker='v', 
                   label=f'Long BRL ({len(long_brl_signals)} trades)', zorder=5)
    
    ax1.set_title(f'2024 Trading Signals - {strategy_name.upper()}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('USD/BRL Exchange Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Z-score with signals
    if 'z_score' in df_2024.columns:
        ax2.plot(df_2024.index, df_2024['z_score'], linewidth=1, color='navy', alpha=0.8)
        ax2.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='±2σ Primary')
        ax2.axhline(y=-2, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=1.5, color='orange', linestyle='--', alpha=0.7, label='±1.5σ Fallback')
        ax2.axhline(y=-1.5, color='orange', linestyle='--', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        if len(long_usd_signals) > 0:
            ax2.scatter(long_usd_signals.index, long_usd_signals['z_score'], color='green', s=80, marker='^', zorder=5)
        if len(long_brl_signals) > 0:
            ax2.scatter(long_brl_signals.index, long_brl_signals['z_score'], color='red', s=80, marker='v', zorder=5)
    
    ax2.set_title('Z-Score and Trading Signals', fontsize=12)
    ax2.set_ylabel('Z-Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Portfolio performance
    if 'cumulative_returns' in df_2024.columns:
        ax3.plot(df_2024.index, df_2024['cumulative_returns'], linewidth=2, color='darkgreen', label='Portfolio')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('2024 Portfolio Performance', fontsize=12)
        ax3.set_ylabel('Cumulative Returns (%)')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'trading_analysis_2024.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Portfolio Performance with Drawdowns
    if 'portfolio_value' in df_2024.columns:
        plt.figure(figsize=(12, 8))
        
        portfolio_values = df_2024['portfolio_value']
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max * 100
        
        plt.subplot(2, 1, 1)
        plt.plot(df_2024.index, portfolio_values, linewidth=2, color='darkgreen', label='Portfolio Value')
        plt.plot(df_2024.index, rolling_max, linewidth=1, color='red', alpha=0.7, linestyle='--', label='Peak Value')
        plt.title('2024 Portfolio Performance', fontsize=14, fontweight='bold')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.fill_between(df_2024.index, drawdowns, 0, color='red', alpha=0.3, label='Drawdown')
        plt.plot(df_2024.index, drawdowns, linewidth=1, color='darkred')
        plt.title('Portfolio Drawdowns', fontsize=12)
        plt.ylabel('Drawdown (%)')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'portfolio_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Visualizations saved to {output_path}/")

def run_backtest_2024(df_full, strategy_name, initial_cash=10000):
    """Run comprehensive backtest for 2024"""
    
    # Filter to 2024
    df_2024 = df_full["2024-01-02":"2024-12-31"].copy()
    
    if len(df_2024) == 0:
        print("❌ No 2024 data available")
        return None, None, None
    
    print(f"📊 Backtesting {len(df_2024)} trading days in 2024...")
    
    # Calculate portfolio returns  
    df_portfolio = calculate_portfolio_returns(df_2024, initial_cash, position_size=0.3)
    
    # Calculate metrics
    risk_metrics = calculate_risk_metrics(df_portfolio)
    trade_metrics = analyze_trades(df_2024)
    
    # Create visualizations
    create_visualizations(df_full, df_portfolio, strategy_name)
    
    # Combined metrics
    all_metrics = {**risk_metrics, **trade_metrics}
    
    # Print results
    print(f"\n📈 2024 PORTFOLIO RESULTS ({strategy_name.upper()})")
    print("=" * 50)
    for key, value in all_metrics.items():
        if isinstance(value, float):
            print(f"{key:<25}: {value:>8.2f}")
        else:
            print(f"{key:<25}: {value:>8}")
    
    return df_portfolio, all_metrics, trade_metrics

def main():
    """Main execution function"""
    try:
        # Load analysis results
        df_full = pd.read_csv("results/analysis_full.csv", index_col=0, parse_dates=True)
        
        # Determine strategy used
        if 'z_score' in df_full.columns:
            strategy_name = "z_score_bands"
        elif 'sma_short' in df_full.columns:
            strategy_name = "sma_crossover"
        else:
            strategy_name = "unknown"
        
        # Run main backtest
        df_portfolio, metrics, trades = run_backtest_2024(df_full, strategy_name)
        
        if df_portfolio is not None:
            # Save results
            df_portfolio.to_csv("results/portfolio_2024.csv")
            
            metrics_df = pd.DataFrame([metrics]).T
            metrics_df.columns = ['Value']
            metrics_df.to_csv("results/risk_metrics_2024.csv")
            
            print(f"\n✅ Main results saved to results/ directory")
        
        # Run parameter optimization heat-map (Section 7)
        print(f"\n🔥 Running parameter optimization heat-map...")
        results_df, best_return, best_sharpe = parameter_optimization_heatmap(df_full)
        
        print(f"\n✅ Heat-map analysis complete!")
        
        return df_portfolio, metrics, results_df
        
    except FileNotFoundError:
        print("❌ Analysis results not found. Please run soy_vs_brl.py first.")
        return None, None, None

if __name__ == "__main__":
    main() 