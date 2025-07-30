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
    
    For each day t, fit: usdbrl ~ soy_usd on prior 250 observations
    Store β(t), ŷ(t), ε(t), and z-score
    """
    results = []
    
    for i in range(window, len(df)):
        # Get window data (prior 250 observations)
        window_data = df.iloc[i-window:i]
        
        # Clean data
        clean_data = window_data[['soy_usd', 'usdbrl']].dropna()
        
        if len(clean_data) < window * 0.8:  # Need at least 80% of window data
            results.append({
                'beta': np.nan,
                'predicted': np.nan,
                'residual': np.nan,
                'z_score': np.nan
            })
            continue
            
        # Fit regression: usdbrl ~ soy_usd
        X = clean_data['soy_usd'].values.reshape(-1, 1)
        y = clean_data['usdbrl'].values
        
        reg = LinearRegression().fit(X, y)
        beta = reg.coef_[0]
        intercept = reg.intercept_
        
        # Current observation (day t)
        current_soy = df.iloc[i]['soy_usd']
        current_brl = df.iloc[i]['usdbrl']
        
        if pd.isna(current_soy) or pd.isna(current_brl):
            results.append({
                'beta': beta,
                'predicted': np.nan,
                'residual': np.nan,
                'z_score': np.nan
            })
            continue
        
        # Prediction and residual for current observation
        predicted = intercept + beta * current_soy
        residual = current_brl - predicted
        
        # Calculate z-score using rolling standard deviation of residuals
        window_predictions = reg.predict(X)
        window_residuals = y - window_predictions
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
    
    # Join with original data
    full_result = df.join(result_df, how='left')
    
    return full_result

def generate_signals(df, z_threshold=2.0, max_holding_days=10):
    """
    Generate trading signals based on z-score bands
    
    Position logic:
    - LONG BRL (sell USD) when z > +threshold → signal = -1  
    - LONG USD when z < -threshold → signal = +1
    
    Exit conditions:
    - Mean reversion (z crosses back towards 0)
    - Hard stop at |z| > 3.5 or after max_holding_days
    """
    df = df.copy()
    df['signal'] = 0
    df['position'] = 0
    df['days_held'] = 0
    
    position = 0  # Current position: +1 = Long USD, -1 = Long BRL, 0 = Flat
    days_held = 0
    entry_z = 0
    
    for i in range(1, len(df)):
        current_z = df.iloc[i]['z_score']
        
        if pd.isna(current_z):
            df.iloc[i, df.columns.get_loc('signal')] = 0
            df.iloc[i, df.columns.get_loc('position')] = position
            df.iloc[i, df.columns.get_loc('days_held')] = days_held
            continue
        
        # Check exit conditions if in position
        if position != 0:
            days_held += 1
            
            # Hard stops
            if abs(current_z) > 3.5 or days_held >= max_holding_days:
                df.iloc[i, df.columns.get_loc('signal')] = -position  # Exit signal
                position = 0
                days_held = 0
                entry_z = 0
            # Mean reversion exit
            elif (position == 1 and current_z > -0.5) or (position == -1 and current_z < 0.5):
                df.iloc[i, df.columns.get_loc('signal')] = -position  # Exit signal
                position = 0
                days_held = 0
                entry_z = 0
        
        # Check entry conditions if flat
        if position == 0:
            if current_z > z_threshold:
                # Z-score too high → BRL overvalued → LONG BRL (signal = -1)
                position = -1
                days_held = 0
                entry_z = current_z
                df.iloc[i, df.columns.get_loc('signal')] = position
            elif current_z < -z_threshold:
                # Z-score too low → BRL undervalued → LONG USD (signal = +1)  
                position = 1
                days_held = 0
                entry_z = current_z
                df.iloc[i, df.columns.get_loc('signal')] = position
        
        df.iloc[i, df.columns.get_loc('position')] = position
        df.iloc[i, df.columns.get_loc('days_held')] = days_held
    
    return df

def sma_crossover_signals(df, short_window=30, long_window=90):
    """
    Generate SMA crossover signals as ultimate fallback
    Long BRL when 30-day SMA > 90-day SMA (BRL strengthening)
    Long USD when 30-day SMA < 90-day SMA (BRL weakening)
    """
    df = df.copy()
    df['sma_short'] = df['usdbrl'].rolling(short_window).mean()
    df['sma_long'] = df['usdbrl'].rolling(long_window).mean()
    
    df['signal'] = 0
    df['position'] = 0
    
    # Generate signals
    prev_signal = 0
    for i in range(long_window, len(df)):
        if pd.isna(df.iloc[i]['sma_short']) or pd.isna(df.iloc[i]['sma_long']):
            continue
            
        if df.iloc[i]['sma_short'] > df.iloc[i]['sma_long']:
            # Short MA > Long MA → BRL strengthening → Long BRL
            current_signal = -1
        else:
            # Short MA < Long MA → BRL weakening → Long USD
            current_signal = 1
        
        # Only signal on changes
        if current_signal != prev_signal:
            df.iloc[i, df.columns.get_loc('signal')] = current_signal
            df.iloc[i, df.columns.get_loc('position')] = current_signal
        else:
            df.iloc[i, df.columns.get_loc('position')] = current_signal
            
        prev_signal = current_signal
    
    return df

def adaptive_signal_generation(df, target_year=2024):
    """
    Implement three-tier adaptive signal generation:
    1. Primary: 2σ z-score bands
    2. Fallback: 1.5σ z-score bands  
    3. Ultimate: SMA crossover
    """
    print("Testing primary z-score bands (2.0σ)...")
    
    # Primary strategy: 2σ z-score
    df_signals = generate_signals(df, z_threshold=2.0)
    signals_2024 = df_signals[f"{target_year}-01-02":f"{target_year}-12-31"]
    trade_count = (signals_2024['signal'] != 0).sum()
    
    print(f"Primary strategy trades in {target_year}: {trade_count}")
    
    if trade_count >= 3:
        print("✓ Primary strategy sufficient")
        return df_signals, "primary_2.0"
    
    # Fallback 1: 1.5σ z-score
    print("Switching to fallback z-score bands (1.5σ)...")
    df_signals = generate_signals(df, z_threshold=1.5)
    signals_2024 = df_signals[f"{target_year}-01-02":f"{target_year}-12-31"]
    trade_count = (signals_2024['signal'] != 0).sum()
    
    print(f"Fallback strategy trades in {target_year}: {trade_count}")
    
    if trade_count >= 1:
        print("✓ Fallback strategy sufficient")
        return df_signals, "fallback_1.5"
    
    # Ultimate fallback: SMA crossover
    print("Switching to SMA crossover strategy...")
    df_signals = sma_crossover_signals(df)
    signals_2024 = df_signals[f"{target_year}-01-02":f"{target_year}-12-31"]
    trade_count = (signals_2024['signal'] != 0).sum()
    
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