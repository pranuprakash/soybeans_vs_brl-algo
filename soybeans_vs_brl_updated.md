# Soybean‑Price vs Brazilian‑Real (USD/BRL) Quant Strategy  
_Adaptive rolling‑beta remake of the “Oil Money” framework_

---

## 0 · Motivation  
Brazil is the world’s #1 soybean exporter.  USD/BRL often tracks Chicago‑soybean
futures (ticker **ZS=F**).  We model this linkage and trade mean‑reverting
deviations.

---

## 1 · Directory layout  

```

data/
soy\_usd.csv          # ZS=F daily settle (USD/bu)
usdb rl.csv          # BRL=X daily close (USD per BRL)
notebooks/
EDA.ipynb
scripts/
download\_data.py
soy\_vs\_brl.py
trading\_backtest.py
results/
docs/
README.md
requirements.txt

````

---

## 2 · Data acquisition (`scripts/download_data.py`)

```python
import yfinance as yf, pandas as pd, datetime as dt, pathlib
p = pathlib.Path("data"); p.mkdir(exist_ok=True)
start, end = "2010-01-01", dt.date.today().isoformat()

soy  = yf.download("ZS=F",  start, end)["Adj Close"].rename("soy_usd")
brl  = yf.download("BRL=X", start, end)["Adj Close"].rename("usbrl")

(soy.to_frame().join(brl, how="inner").dropna()
 ).to_csv(p / "raw_merged.csv")
````

*(Ten‑plus years ensures enough history for rolling regression.)*

---

## 3 · Exploratory analysis (`notebooks/EDA.ipynb`)

* Scatter & Pearson ρ
* ADF on log returns; Engle‑Granger for cointegration
* **Rolling 250‑day β** of USD/BRL on soy price
* Regime plot: z‑score of residuals 2010‑2023

---

## 4 · Model & adaptive signal generation (`scripts/soy_vs_brl.py`)

| Component            | Detail                                                                                                                 |   |                           |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------- | - | ------------------------- |
| **Rolling OLS**      | Window = 250 trading days (≈ 1 yr).  For each day *t*, fit `usbrl ~ soy_usd` on prior 250 obs; store β(t), ŷ(t), ε(t). |   |                           |
| **z‑score**          | `z = ε / σ`, where σ = rolling std of ε over same 250‑day window.                                                      |   |                           |
| **Primary bands**    | Enter when \`                                                                                                          | z | ≥ 2\` (≈ 2 σ).            |
| **Fallback bands**   | If **< 3 trades triggered in 2024**, auto‑switch to \`                                                                 | z | ≥ 1.5\`.                  |
| **Momentum overlay** | If fallback still yields < 1 trade: use 30‑/90‑day SMA crossover on USD/BRL (long when 30 < 90, short reverse).        |   |                           |
| **Positions**        | LONG BRL (sell USD) when *z* > band → `signal = -1`;  LONG USD when *z* < −band → `signal = +1`.                       |   |                           |
| **Stops / flips**    | Hard stop at \`                                                                                                        | z | > 3.5\` or after 10 days. |

---

## 5 · Portfolio simulation focused on **calendar‑year 2024**

```python
signals_full = generate_signals(df)                    # full history
signals_2024 = signals_full["2024-01-02":"2024-12-31"] # trading days only
pnl_2024      = portfolio(signals_2024, cash0=10_000)
```

---

## 6 · Risk & performance metrics (on 2024 window)

* Total & annualised return, volatility, Sharpe
* Max drawdown, Calmar
* Trade count, hit rate, avg holding period
* Exposure % (days in market / 252)

---

## 7 · Heat‑map parameter grid

Sweep:
`band ∈ {1.5, 2, 2.5}` × `holding_max ∈ 5‥20`
…but still score **only 2024**.  Save CSV + heat‑map PNG.

---

## 8 · Requirements

```
pandas numpy statsmodels yfinance matplotlib seaborn scikit-learn
```

---

## 9 · README.md must include

1. 120‑word abstract
2. Data sources & date‑range
3. Rolling‑β plot, z‑score histogram, entry/exit chart (2024 only)
4. Portfolio‑curve + risk table
5. Explanation of adaptive bands & fallback logic
6. Heat‑map (if run)
7. Limitations & next steps (Kalman‑β, options overlay)

---

## 10 · Quality checklist

* [ ] Download script runs without edit
* [ ] At least **3** trades executed in 2024 (else adjust band automatically)
* [ ] README embeds 3+ figures and a risk‑metric table
* [ ] Code PEP‑8; all paths relative
* [ ] Return **`DONE ✅`** when everything passes

```

---
