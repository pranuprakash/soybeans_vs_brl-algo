import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def calculate_portfolio_returns(df, initial_cash=10000, transaction_cost=0.001):
    """
    Calculate portfolio returns based on signals
    
    Args:
        df: DataFrame with signal, usdbrl columns
        initial_cash: Starting portfolio value
        transaction_cost: Transaction cost as percentage of trade value
        
    Returns:
        DataFrame with portfolio value, returns, positions
    """
    df = df.copy()
    df['portfolio_value'] = initial_cash
    df['position_size'] = 0.0
    df['cash'] = initial_cash
    df['returns'] = 0.0
    df['cumulative_returns'] = 0.0
    
    cash = initial_cash
    position = 0.0  # USD position in BRL terms
    
    for i in range(1, len(df)):
        current_price = df.iloc[i]['usdbrl']
        prev_price = df.iloc[i-1]['usdbrl']
        signal = df.iloc[i]['signal']
        
        if pd.isna(current_price) or pd.isna(prev_price):
            df.iloc[i, df.columns.get_loc('portfolio_value')] = df.iloc[i-1]['portfolio_value']
            df.iloc[i, df.columns.get_loc('position_size')] = position
            df.iloc[i, df.columns.get_loc('cash')] = cash
            continue
        
        # Calculate PnL from existing position
        if position != 0:
            price_return = (current_price - prev_price) / prev_price
            position_pnl = position * price_return * initial_cash
        else:
            position_pnl = 0
        
        # Handle new signals
        if signal != 0:
            # Close existing position if any
            if position != 0:
                transaction_cost_amount = abs(position) * initial_cash * transaction_cost
                cash -= transaction_cost_amount
            
            # Open new position
            position = signal * 0.95  # Use 95% of portfolio for position
            transaction_cost_amount = abs(position) * initial_cash * transaction_cost
            cash -= transaction_cost_amount
        
        # Update portfolio value
        portfolio_value = cash + (position * initial_cash * (1 + ((current_price - prev_price) / prev_price)))
        daily_return = (portfolio_value - df.iloc[i-1]['portfolio_value']) / df.iloc[i-1]['portfolio_value']
        
        df.iloc[i, df.columns.get_loc('portfolio_value')] = portfolio_value
        df.iloc[i, df.columns.get_loc('position_size')] = position
        df.iloc[i, df.columns.get_loc('cash')] = cash
        df.iloc[i, df.columns.get_loc('returns')] = daily_return
    
    # Calculate cumulative returns
    df['cumulative_returns'] = (df['portfolio_value'] / initial_cash - 1) * 100
    
    return df

def calculate_risk_metrics(returns_series, trading_days_per_year=252):
    """Calculate comprehensive risk metrics"""
    returns = returns_series.dropna()
    
    if len(returns) == 0:
        return {}
    
    # Basic metrics
    total_return = (returns + 1).prod() - 1
    annualized_return = (1 + total_return) ** (trading_days_per_year / len(returns)) - 1
    annualized_vol = returns.std() * np.sqrt(trading_days_per_year)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
    
    # Drawdown calculation
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win rate and trade stats
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
    
    avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
    avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
    
    return {
        'Total Return (%)': total_return * 100,
        'Annualized Return (%)': annualized_return * 100,
        'Annualized Volatility (%)': annualized_vol * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown * 100,
        'Calmar Ratio': calmar_ratio,
        'Win Rate (%)': win_rate * 100,
        'Average Win (%)': avg_win * 100,
        'Average Loss (%)': avg_loss * 100,
        'Trading Days': len(returns)
    }

def analyze_trades(df):
    """Analyze trading activity"""
    signals = df['signal'][df['signal'] != 0]
    
    if len(signals) == 0:
        return {
            'Total Trades': 0,
            'Long Trades': 0,
            'Short Trades': 0,
            'Avg Holding Period': 0,
            'Market Exposure (%)': 0
        }
    
    trade_count = len(signals)
    long_trades = len(signals[signals > 0])
    short_trades = len(signals[signals < 0])
    
    # Calculate holding periods and exposure
    if 'position_size' in df.columns:
        in_position = df['position_size'] != 0
        exposure_days = in_position.sum()
    else:
        # Estimate exposure from signal activity
        exposure_days = trade_count * 5  # Assume average 5-day holding
    
    total_days = len(df)
    market_exposure = (exposure_days / total_days) * 100 if total_days > 0 else 0
    
    # Estimate average holding period
    avg_holding = exposure_days / trade_count if trade_count > 0 else 0
    
    return {
        'Total Trades': trade_count,
        'Long Trades': long_trades,
        'Short Trades': short_trades,
        'Avg Holding Period': avg_holding,
        'Market Exposure (%)': market_exposure
    }

def create_visualizations(df, df_2024, strategy_name, output_dir="results"):
    """Create comprehensive visualizations"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Rolling Beta Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Beta evolution
    beta_data = df['beta'].dropna()
    ax1.plot(beta_data.index, beta_data.values, linewidth=1, alpha=0.7)
    ax1.axhline(y=beta_data.mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean β = {beta_data.mean():.3f}')
    ax1.set_title('Rolling 250-Day Beta: USD/BRL vs Soy Prices', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Beta Coefficient')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Z-score over time
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
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    
    # Price and signals
    ax1.plot(df_2024.index, df_2024['usdbrl'], linewidth=2, label='USD/BRL', color='blue')
    signal_dates = df_2024[df_2024['signal'] != 0]
    long_signals = signal_dates[signal_dates['signal'] > 0]
    short_signals = signal_dates[signal_dates['signal'] < 0]
    
    if len(long_signals) > 0:
        ax1.scatter(long_signals.index, long_signals['usdbrl'], color='green', s=100, marker='^', 
                   label=f'Long USD ({len(long_signals)} trades)', zorder=5)
    if len(short_signals) > 0:
        ax1.scatter(short_signals.index, short_signals['usdbrl'], color='red', s=100, marker='v', 
                   label=f'Long BRL ({len(short_signals)} trades)', zorder=5)
    
    ax1.set_title(f'2024 Trading Signals - {strategy_name.upper()}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('USD/BRL Exchange Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Z-score with trading bands
    if 'z_score' in df_2024.columns:
        ax2.plot(df_2024.index, df_2024['z_score'], linewidth=1, color='navy', alpha=0.8)
        ax2.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='±2σ Primary')
        ax2.axhline(y=-2, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=1.5, color='orange', linestyle='--', alpha=0.7, label='±1.5σ Fallback')
        ax2.axhline(y=-1.5, color='orange', linestyle='--', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Mark signal points
        if len(long_signals) > 0:
            ax2.scatter(long_signals.index, long_signals['z_score'], color='green', s=80, marker='^', zorder=5)
        if len(short_signals) > 0:
            ax2.scatter(short_signals.index, short_signals['z_score'], color='red', s=80, marker='v', zorder=5)
    
    ax2.set_title('Z-Score and Trading Signals', fontsize=12)
    ax2.set_ylabel('Z-Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Portfolio performance
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
    
    # 4. Portfolio Curve with Drawdowns
    plt.figure(figsize=(12, 8))
    
    # Calculate drawdowns for 2024
    portfolio_values = df_2024['portfolio_value']
    rolling_max = portfolio_values.expanding().max()
    drawdowns = (portfolio_values - rolling_max) / rolling_max * 100
    
    # Main plot
    plt.subplot(2, 1, 1)
    plt.plot(df_2024.index, portfolio_values, linewidth=2, color='darkgreen', label='Portfolio Value')
    plt.plot(df_2024.index, rolling_max, linewidth=1, color='red', alpha=0.7, linestyle='--', label='Peak Value')
    plt.title('2024 Portfolio Performance', fontsize=14, fontweight='bold')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Drawdown plot
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
    """Run portfolio backtest focused on 2024"""
    
    # Filter to 2024 trading days
    df_2024 = df_full["2024-01-02":"2024-12-31"].copy()
    
    if len(df_2024) == 0:
        print("❌ No 2024 data available")
        return None, None, None
    
    print(f"📊 Backtesting {len(df_2024)} trading days in 2024...")
    
    # Calculate portfolio returns
    df_portfolio = calculate_portfolio_returns(df_2024, initial_cash)
    
    # Calculate metrics
    risk_metrics = calculate_risk_metrics(df_portfolio['returns'])
    trade_metrics = analyze_trades(df_2024)
    
    # Create visualizations
    create_visualizations(df_full, df_portfolio, strategy_name)
    
    # Combine all metrics
    all_metrics = {**risk_metrics, **trade_metrics}
    
    # Print summary
    print(f"\n📈 2024 PORTFOLIO RESULTS ({strategy_name.upper()})")
    print("=" * 50)
    for key, value in all_metrics.items():
        if isinstance(value, float):
            print(f"{key:<25}: {value:>8.2f}")
        else:
            print(f"{key:<25}: {value:>8}")
    
    return df_portfolio, all_metrics, trade_metrics

def main():
    """Main execution for backtesting"""
    try:
        # Load analysis results
        df_full = pd.read_csv("results/analysis_full.csv", index_col=0, parse_dates=True)
        
        # Determine strategy used (check which columns exist)
        if 'z_score' in df_full.columns:
            strategy_name = "z_score_bands"
        elif 'sma_short' in df_full.columns:
            strategy_name = "sma_crossover"
        else:
            strategy_name = "unknown"
        
        # Run 2024 backtest
        df_portfolio, metrics, trades = run_backtest_2024(df_full, strategy_name)
        
        if df_portfolio is not None:
            # Save detailed results
            df_portfolio.to_csv("results/portfolio_2024.csv")
            
            # Save metrics
            metrics_df = pd.DataFrame([metrics]).T
            metrics_df.columns = ['Value']
            metrics_df.to_csv("results/risk_metrics_2024.csv")
            
            print(f"\n✅ Results saved to results/ directory")
            
            return df_portfolio, metrics
        
    except FileNotFoundError:
        print("❌ Analysis results not found. Please run soy_vs_brl.py first.")
        return None, None

if __name__ == "__main__":
    main() 