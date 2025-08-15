# Multi-Asset Strategy Backtesting with Risk Management

## 📌 Overview
This repository contains a **custom-built backtesting framework** and a **single trading strategy** applied across multiple instruments.  
The goal: robust performance across diverse market regimes while enforcing realistic execution, dynamic position sizing, and strict risk management.

**Bootcamp Task:** Quant Bootcamp — Multi-Asset Strategy Backtesting with Risk Management  
**Initial Capital:** ₹1,000,000  
**Brokerage:** 0.03% per side  
**Instruments:** 4 separate datasets (equity/commodity-like, anonymized)  

---

## 📂 Project Structure
.
├── quant_backtester.py # Main backtesting script
├── config.yaml # Strategy & risk parameters
├── REPORT_TEMPLATE.md # Short report template
├── data/ # Place your 4 datasets here
├── out/ # Output folder for metrics & trades
└── README.md # This file


---

## ⚙️ Strategy Logic
The core strategy is a **trend-following system with volatility scaling**:
- **Signal generation**: 50-period EMA crossover with volatility filter (ATR-based)
- **Execution**: Next bar open, no lookahead bias
- **Stop-loss**: 2 × ATR initial stop + Chandelier trailing stop
- **Position sizing**: Volatility targeting + fixed fractional sizing
- **Portfolio risk control**: Max 20–25% drawdown halts new entries for N days

This design ensures a **uniform logic** across all datasets while adapting position size to prevailing volatility conditions.

---

## 🛡️ Risk Management
- **Trade-level controls**: ATR-based stops to cut losses quickly
- **Portfolio-level controls**: Max drawdown cap; trading pause after breach
- **Transaction costs**: 0.03% per side, applied to every trade
- **Slippage model**: Base bps + liquidity impact from participation rate

---

## 📏 Position Sizing
The position sizing methodology is based on **dynamic volatility targeting**:
- Target portfolio volatility (default: 15% annualized)
- Fixed fractional allocation per trade (default: 1% risk of equity)
- Cap: 10% of initial capital per instrument
- Sizing adjusts automatically to changing market conditions

---

## 📊 Example Performance Metrics
| Dataset   | Net Return | CAGR   | Sharpe | Sortino | Max DD  | Win Rate | Profit Factor | Avg Holding (days) | Trades |
|-----------|-----------:|-------:|-------:|--------:|--------:|---------:|--------------:|--------------------:|-------:|
| Data1     | 12.5%      | 6.1%   | 0.91   | 1.24    | -18.3%  | 54%      | 1.34          | 7.2                 | 145    |
| Data2     | 9.7%       | 4.8%   | 0.85   | 1.10    | -19.2%  | 51%      | 1.28          | 8.0                 | 138    |
| Data3     | ...        | ...    | ...    | ...     | ...     | ...      | ...           | ...                 | ...    |
| Data4     | ...        | ...    | ...    | ...     | ...     | ...      | ...           | ...                 | ...    |

> These are **placeholder results**. Run the backtester to generate your actual metrics.

---

## 🚀 How to Run
1. Place your datasets (`.csv`) inside `./data/` with columns:
   Date, Open, High, Low, Close, Volume
2. (Optional) Edit `config.yaml` to tweak parameters.
3. Run the backtester:
   
'''bash
python quant_backtester.py --data_dir ./data --out_dir ./out --config ./config.yaml
  
4. View results in:

out/metrics_summary.csv (summary table)

out/*_trades.csv (trade logs)

out/*_equity.csv (equity curve data)
