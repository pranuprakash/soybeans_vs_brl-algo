import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the merged data"""
    try:
        df = pd.read_csv("data/raw_merged.csv", index_col=0, parse_dates=True)
        return df
    except FileNotFoundError:
        print("Data file not found. Please run download_data.py first.")
        return None

def rolling_ols_zscore(df, window=250):
    """
    Implement rolling OLS regression with z-score calculation
    
    Args:
        df: DataFrame with soy_usd and usdbrl columns
        window: Rolling window size (250 trading days ≈ 1 year)
        
    Returns:
        DataFrame with beta, predicted, residuals, z_score columns
    """
    results = []
    
    for i in range(window, len(df)):
        # Get window data
        window_data = df.iloc[i-window:i]
        
        # Fit regression: usdbrl ~ soy_usd
        X = window_data['soy_usd'].values.reshape(-1, 1)
        y = window_data['usdbrl'].values
        
        # Remove any NaN values
        mask = ~(np.isnan(X.flatten()) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) < window * 0.8:  # Need at least 80% of window data
            results.append({
                'beta': np.nan,
                'predicted': np.nan,
                'residual': np.nan,
                'z_score': np.nan
            })
            continue
            
        # Fit linear regression
        reg = LinearRegression().fit(X_clean.reshape(-1, 1), y_clean)
        beta = reg.coef_[0]
        intercept = reg.intercept_
        
        # Current observation prediction
        current_soy = df.iloc[i]['soy_usd']
        predicted = intercept + beta * current_soy
        actual = df.iloc[i]['usdbrl']
        residual = actual - predicted
        
        # Calculate z-score using residuals from the window
        window_residuals = y_clean - (intercept + beta * X_clean.flatten())
        residual_std = np.std(window_residuals)
        z_score = residual / residual_std if residual_std > 0 else 0
        
        results.append({
            'beta': beta,
            'predicted': predicted,
            'residual': residual,
            'z_score': z_score
        })
    
    # Create result DataFrame
    result_df = pd.DataFrame(results, index=df.index[window:])
    
    # Align with original DataFrame
    full_result = pd.DataFrame(index=df.index)
    full_result = full_result.join(result_df)
    full_result = full_result.join(df)
    
    return full_result

def generate_signals(df, z_threshold=2.0, fallback_threshold=1.5, max_holding_days=10):
    """
    Generate trading signals with adaptive fallback mechanism
    
    Args:
        df: DataFrame with z_score column
        z_threshold: Primary z-score threshold (2.0)
        fallback_threshold: Fallback threshold (1.5)
        max_holding_days: Maximum holding period
        
    Returns:
        DataFrame with signals column
    """
    df = df.copy()
    df['signal'] = 0
    df['position'] = 0
    df['days_held'] = 0
    
    position = 0
    days_held = 0
    
    for i in range(1, len(df)):
        current_z = df.iloc[i]['z_score']
        
        if pd.isna(current_z):
            df.iloc[i, df.columns.get_loc('signal')] = 0
            df.iloc[i, df.columns.get_loc('position')] = position
            df.iloc[i, df.columns.get_loc('days_held')] = days_held
            continue
        
        # Exit conditions
        if position != 0:
            days_held += 1
            
            # Hard stop conditions
            if abs(current_z) > 3.5 or days_held >= max_holding_days:
                position = 0
                days_held = 0
                df.iloc[i, df.columns.get_loc('signal')] = -position  # Exit signal
            # Mean reversion exit
            elif (position > 0 and current_z > -0.5) or (position < 0 and current_z < 0.5):
                position = 0
                days_held = 0
                df.iloc[i, df.columns.get_loc('signal')] = -position  # Exit signal
        
        # Entry conditions (only if not in position)
        if position == 0:
            if current_z > z_threshold:
                position = -1  # Long BRL (sell USD)
                days_held = 0
                df.iloc[i, df.columns.get_loc('signal')] = position
            elif current_z < -z_threshold:
                position = 1   # Long USD
                days_held = 0
                df.iloc[i, df.columns.get_loc('signal')] = position
        
        df.iloc[i, df.columns.get_loc('position')] = position
        df.iloc[i, df.columns.get_loc('days_held')] = days_held
    
    return df

def sma_crossover_signals(df, short_window=30, long_window=90):
    """
    Generate SMA crossover signals as ultimate fallback
    """
    df = df.copy()
    df['sma_short'] = df['usdbrl'].rolling(short_window).mean()
    df['sma_long'] = df['usdbrl'].rolling(long_window).mean()
    
    df['signal'] = 0
    df.loc[df['sma_short'] > df['sma_long'], 'signal'] = -1  # Long BRL
    df.loc[df['sma_short'] < df['sma_long'], 'signal'] = 1   # Long USD
    
    # Only take signal changes
    df['signal_change'] = df['signal'].diff()
    signal_final = df['signal'].copy()
    signal_final[df['signal_change'] == 0] = 0
    df['signal'] = signal_final
    
    return df

def adaptive_signal_generation(df, target_year=2024):
    """
    Implement adaptive signal generation with fallback mechanisms
    """
    print("Testing primary z-score bands (2.0)...")
    
    # Primary strategy: 2.0 z-score
    df_signals = generate_signals(df, z_threshold=2.0)
    signals_2024 = df_signals[f"{target_year}-01-02":f"{target_year}-12-31"]
    trade_count = abs(signals_2024['signal']).sum()
    
    print(f"Primary strategy trades in {target_year}: {trade_count}")
    
    if trade_count >= 3:
        print("✓ Primary strategy sufficient")
        return df_signals, "primary_2.0"
    
    # Fallback 1: 1.5 z-score
    print("Switching to fallback z-score bands (1.5)...")
    df_signals = generate_signals(df, z_threshold=1.5)
    signals_2024 = df_signals[f"{target_year}-01-02":f"{target_year}-12-31"]
    trade_count = abs(signals_2024['signal']).sum()
    
    print(f"Fallback strategy trades in {target_year}: {trade_count}")
    
    if trade_count >= 1:
        print("✓ Fallback strategy sufficient")
        return df_signals, "fallback_1.5"
    
    # Ultimate fallback: SMA crossover
    print("Switching to SMA crossover strategy...")
    df_signals = sma_crossover_signals(df)
    signals_2024 = df_signals[f"{target_year}-01-02":f"{target_year}-12-31"]
    trade_count = abs(signals_2024['signal']).sum()
    
    print(f"SMA crossover trades in {target_year}: {trade_count}")
    print("✓ Using SMA crossover as final fallback")
    
    return df_signals, "sma_crossover"

def main():
    """Main execution function"""
    # Load data
    df = load_data()
    if df is None:
        return None
    
    print(f"Loaded data: {len(df)} rows from {df.index.min()} to {df.index.max()}")
    
    # Calculate rolling regression and z-scores
    print("Calculating rolling 250-day regression...")
    df_analysis = rolling_ols_zscore(df)
    
    # Generate adaptive signals
    df_final, strategy_used = adaptive_signal_generation(df_analysis)
    
    print(f"Final strategy used: {strategy_used}")
    
    # Save results
    df_final.to_csv("results/analysis_full.csv")
    
    return df_final, strategy_used

if __name__ == "__main__":
    main() 