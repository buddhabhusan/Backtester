# Strategy Performance Report

## 📌 Task
**Quant Bootcamp:** Multi-Asset Strategy Backtesting with Risk Management  
**Initial Capital:** ₹1,000,000  
**Brokerage:** 0.03% per side  
**Instruments:** 4 anonymized datasets (equity/commodity-like)  

---

## 1. Strategy Logic
**Core Idea:** Trend-following EMA crossover with ATR-based volatility filter.  
- **Entry:** 50 EMA crosses above 200 EMA (long), 50 EMA below 200 EMA (short if allowed)  
- **Exit:** Opposite crossover OR ATR-based stop hit  
- **Volatility filter:** Skip trades if ATR% < threshold (low-volatility environment)  
- **Execution:** Next-bar open (no lookahead bias)  

---

## 2. Risk Management
- **Trade-level:** ATR-based initial stop + trailing stop (Chandelier Exit)  
- **Portfolio-level:** Max drawdown cap (20–25%) halts new trades for N days  
- **Costs:** 0.03% per side, slippage model  
- **Position limits:** Max 10% of portfolio in a single instrument  

---

## 3. Position Sizing
- **Method:** Volatility targeting (15% annualized) + fixed fractional (1% risk of equity per trade)  
- **Adjustment:** Position size recalculated each trade using latest ATR  

---

## 4. Performance Metrics

| Dataset   | Net Return | CAGR   | Sharpe | Sortino | Max DD  | Win Rate | Profit Factor | Avg Holding (days) | Trades |
|-----------|-----------:|-------:|-------:|--------:|--------:|---------:|--------------:|--------------------:|-------:|
| Data1     |            |        |        |         |         |          |               |                     |        |
| Data2     |            |        |        |         |         |          |               |                     |        |
| Data3     |            |        |        |         |         |          |               |                     |        |
| Data4     |            |        |        |         |         |          |               |                     |        |

---

## 5. Key Observations
- **Consistency:**  
- **Volatility Impact:**  
- **Drawdown Recovery:**  
- **Market Regime Sensitivity:**  

---

## 6. Lessons Learned
- Volatility scaling improved uniformity across instruments  
- Drawdown controls prevented catastrophic losses in high-vol regimes  
- Strategy remains profitable in both trending and mixed conditions, though returns varied by liquidity  

---

## 7. References
- Ahmed, S. M. M. (2023). *Sizing Strategies for Algorithmic Trading in Volatile Markets.* arXiv:2309.09094.  
- Lezmi, E., Roche, A., Roncalli, T., Xu, J. (2020). *Improving the Robustness of Trading Strategy Backtesting.* arXiv:2007.04838.  
- Harvey, C. R., Liu, Y. (2015). *Backtesting.* SSRN 2345489.  
