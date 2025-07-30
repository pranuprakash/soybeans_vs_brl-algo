import yfinance as yf
import pandas as pd
import datetime as dt
import pathlib

def download_data():
    """Download soybean and Brazilian Real data from Yahoo Finance"""
    p = pathlib.Path("data")
    p.mkdir(exist_ok=True)
    
    start, end = "2010-01-01", dt.date.today().isoformat()
    
    print(f"Downloading data from {start} to {end}")
    
    # Download soybean futures (ZS=F) and Brazilian Real (BRL=X)
    soy_data = yf.download("ZS=F", start, end)
    brl_data = yf.download("BRL=X", start, end)
    
    # Extract the appropriate price column
    if isinstance(soy_data.columns, pd.MultiIndex):
        soy = soy_data["Close"].iloc[:, 0].rename("soy_usd")
    else:
        soy = soy_data["Close"].rename("soy_usd")
        
    if isinstance(brl_data.columns, pd.MultiIndex):
        brl = brl_data["Close"].iloc[:, 0].rename("usdbrl")
    else:
        brl = brl_data["Close"].rename("usdbrl")
    
    # Merge and save
    merged_data = soy.to_frame().join(brl, how="inner").dropna()
    merged_data.to_csv(p / "raw_merged.csv")
    
    print(f"Downloaded {len(merged_data)} rows of data")
    print(f"Date range: {merged_data.index.min()} to {merged_data.index.max()}")
    
    return merged_data

if __name__ == "__main__":
    download_data() 