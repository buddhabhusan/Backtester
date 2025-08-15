#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Asset Strategy Backtesting with Risk Management — Single-File Backtester

Author: (you)
Date: 2025-08-15

How to run
----------
1) With your 4 CSV datasets (OHLCV):
   python multi_asset_backtester.py --data path/to/dataset1.csv path/to/dataset2.csv path/to/dataset3.csv path/to/dataset4.csv

   Expected CSV columns (case-insensitive accepted, auto-normalized):
     - Date or datetime (parsed as pandas-like string or YYYY-MM-DD)
     - Open, High, Low, Close
     - Volume (optional; if missing, volume modeling is disabled)
   The script auto-sorts by date ascending.

2) With no files present (demo on synthetic data):
   python multi_asset_backtester.py

Key modeling choices
--------------------
- Single strategy applied to each dataset independently:
  EMA(50)/EMA(200) crossover with ATR filter and regime throttle.
- Execution: signals formed on close[t-1], executed at open[t] (no lookahead).
- Transaction costs: brokerage 0.03% per side, plus configurable slippage model.
- Shorting: allowed.
- Risk management:
  * Trade-level: ATR-based stop-loss & take-profit; time-stop/maximum holding days.
  * Portfolio-level: equity peak-to-trough drawdown halt; daily loss limit.
- Position sizing:
  * Volatility scaling + fixed-fractional (risk per trade = risk_fraction * equity).
  * Shares = risk_per_trade / (ATR_mult_stop * ATR). Capped by notional and shares constraints.
- Result artifacts per dataset:
  * metrics_<name>.json
  * trades_<name>.csv
  * equity_<name>.csv
  * equity_<name>.png

You may tune parameters in CONFIG, but the core rules remain identical across datasets.

No external backtesting libraries are used.
"""
import argparse
import math
import statistics
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime as dt

# -----------------------------
# Global constants / CONFIG
# -----------------------------

CONFIG = {
    "initial_capital": 1_000_000.0,       # ₹ or unit currency
    "brokerage_bps_per_side": 3.0,        # 0.03% per side => 3 bps
    "slippage_mode": "atr_bps",           # 'fixed_bps' or 'atr_bps'
    "slippage_fixed_bps": 2.0,            # if mode=='fixed_bps', apply per transaction
    "slippage_atr_bps": 5.0,              # if mode=='atr_bps', effective bps = min(slippage_atr_bps, 0.1*ATR%)

    "allow_short": True,

    # Strategy parameters (applied equally to all datasets)
    "ema_fast": 50,
    "ema_slow": 200,
    "atr_lookback": 14,
    "atr_filter_min": 0.002,              # require ATR% > 0.2% to trade
    "regime_vol_cap": 0.08,               # if realized 20d vol > 8%, reduce target risk scaling
    "signal_cooldown_bars": 1,            # min bars between flips to avoid whipsaw

    # Trade-level risk
    "stop_atr_mult": 2.0,                 # initial stop k * ATR
    "tp_atr_mult": 4.0,                   # take-profit k * ATR (optional; set None to disable)
    "max_holding_days": 60,               # time stop

    # Portfolio-level risk
    "max_drawdown_halt": 0.15,            # halt new entries if equity DD from peak exceeds 15%
    "daily_loss_limit": 0.04,             # if single day loss exceeds 4% equity, close positions and halt next day

    # Position sizing
    "risk_fraction": 0.01,                # risk per trade = 1% of equity
    "max_leverage": 1.0,                  # notional / equity <= 1
    "per_instrument_notional_cap": 0.5,   # max 50% of equity per instrument
    "min_shares": 1,

    # Reporting
    "risk_free_rate_annual": 0.0,         # Sharpe RF
    "trading_days_per_year": 252,

    # Backtest mechanics
    "execution": "next_open",             # execute at bar open following the signal bar
    "bar_timezone": "UTC",
}

# -----------------------------
# Utilities
# -----------------------------

def to_bps(x: float) -> float:
    return x * 1e4

def from_bps(bps: float) -> float:
    return bps / 1e4

def annualize_return(daily_returns: np.ndarray, days_per_year: int) -> float:
    if len(daily_returns) == 0:
        return 0.0
    cum = np.prod(1.0 + daily_returns) - 1.0
    years = len(daily_returns) / days_per_year
    if years <= 0:
        return 0.0
    return (1.0 + cum) ** (1.0 / years) - 1.0

def sharpe_ratio(daily_returns: np.ndarray, rf_annual: float, days_per_year: int) -> float:
    if len(daily_returns) < 2:
        return 0.0
    rf_daily = (1 + rf_annual) ** (1/days_per_year) - 1
    excess = daily_returns - rf_daily
    mu = np.mean(excess)
    sigma = np.std(excess, ddof=1)
    return 0.0 if sigma == 0 else (mu / sigma) * math.sqrt(days_per_year)

def sortino_ratio(daily_returns: np.ndarray, rf_annual: float, days_per_year: int) -> float:
    if len(daily_returns) < 2:
        return 0.0
    rf_daily = (1 + rf_annual) ** (1/days_per_year) - 1
    excess = daily_returns - rf_daily
    downside = excess[excess < 0]
    denom = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
    mu = np.mean(excess)
    return 0.0 if denom == 0 else (mu / denom) * math.sqrt(days_per_year)

def max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peaks) / peaks
    i = int(np.argmin(drawdowns))
    j = int(np.argmax(equity_curve[:i+1]))
    return float(-drawdowns[i]), j, i  # magnitude, peak idx, trough idx

def ema(series: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(series).ewm(span=span, adjust=False).mean().values

def true_range(h: np.ndarray, l: np.ndarray, c_prev: np.ndarray) -> np.ndarray:
    tr1 = h - l
    tr2 = np.abs(h - c_prev)
    tr3 = np.abs(l - c_prev)
    return np.maximum.reduce([tr1, tr2, tr3])

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, lookback: int) -> np.ndarray:
    c_prev = np.concatenate([[close[0]], close[:-1]])
    tr = true_range(high, low, c_prev)
    return pd.Series(tr).ewm(span=lookback, adjust=False).mean().values

def realized_vol(close: np.ndarray, window: int = 20) -> np.ndarray:
    ret = np.diff(np.log(close), prepend=np.log(close[0]))
    vol = pd.Series(ret).rolling(window).std().values
    return vol

# -----------------------------
# Data loading / normalization
# -----------------------------

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    # Normalize expected columns
    mapping = {}
    for key in ["date", "datetime", "timestamp"]:
        if key in cols:
            mapping[cols[key]] = "date"
            break
    for key in ["open"]:
        if key in cols: mapping[cols[key]] = "open"
    for key in ["high"]:
        if key in cols: mapping[cols[key]] = "high"
    for key in ["low"]:
        if key in cols: mapping[cols[key]] = "low"
    for key in ["close", "adj close", "adj_close"]:
        if key in cols: mapping[cols[key]] = "close"
    if "volume" in cols:
        mapping[cols["volume"]] = "volume"
    df = df.rename(columns=mapping)
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing '{col}' in {path}")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    else:
        df = df.reset_index().rename(columns={"index": "date"})
        df["date"] = pd.to_datetime(df["date"])
    df = df.reset_index(drop=True)
    return df

# -----------------------------
# Order & Position structures
# -----------------------------

@dataclass
class Position:
    side: int                   # +1 long, -1 short
    entry_price: float
    size: int                   # number of shares/contracts
    entry_idx: int
    stop_price: float
    tp_price: Optional[float]
    max_adverse: float = 0.0    # track MAE
    max_favorable: float = 0.0  # track MFE

@dataclass
class TradeRecord:
    entry_date: str
    exit_date: str
    side: str
    entry_price: float
    exit_price: float
    size: int
    pnl: float
    pnl_pct: float
    bars_held: int
    mae_pct: float
    mfe_pct: float

# -----------------------------
# Backtester
# -----------------------------

class SingleAssetBacktester:
    def __init__(self, df: pd.DataFrame, name: str, config: Dict):
        self.df = df.copy()
        self.name = name
        self.cfg = config
        self._prepare_indicators()

        self.equity = self.cfg["initial_capital"]
        self.equity_curve = []
        self.daily_returns = []
        self.peak_equity = self.equity
        self.halt_new_entries = False
        self.open_position: Optional[Position] = None
        self.last_signal_idx = -999

        self.trades: List[TradeRecord] = []
        self.notional_cap = self.cfg["per_instrument_notional_cap"]
        self.max_leverage = self.cfg["max_leverage"]
        self.daily_pnl_tracker = {}

    def _prepare_indicators(self):
        price = self.df["close"].values
        high = self.df["high"].values
        low = self.df["low"].values

        self.df["ema_fast"] = ema(price, self.cfg["ema_fast"])
        self.df["ema_slow"] = ema(price, self.cfg["ema_slow"])
        self.df["atr"] = atr(high, low, price, self.cfg["atr_lookback"])
        self.df["atr_pct"] = self.df["atr"] / self.df["close"]
        self.df["rv20"] = realized_vol(price, 20)
        self.df["signal_raw"] = 0
        self.df.loc[self.df["ema_fast"] > self.df["ema_slow"], "signal_raw"] = 1
        self.df.loc[self.df["ema_fast"] < self.df["ema_slow"], "signal_raw"] = -1

        # ATR filter & regime throttle
        self.df["can_trade"] = (self.df["atr_pct"] > self.cfg["atr_filter_min"]).astype(int)

    def _slippage_bps(self, i: int) -> float:
        mode = self.cfg["slippage_mode"]
        if mode == "fixed_bps":
            return self.cfg["slippage_fixed_bps"]
        # atr_bps
        atr_pct = float(self.df.loc[i, "atr_pct"])
        cap_bps = self.cfg["slippage_atr_bps"]
        return min(cap_bps, max(0.0, atr_pct * 1e4 * 0.1))  # <= 0.1 * ATR% in bps

    def _price_with_costs(self, raw_price: float, i: int, side: int, is_entry: bool) -> float:
        # Apply brokerage and slippage: entries buy => pay up; sell => get less; exits opposite sign
        bps_broker = self.cfg["brokerage_bps_per_side"]
        bps_slip = self._slippage_bps(i)
        total_bps = bps_broker + bps_slip

        if is_entry:
            adj = 1.0 + from_bps(total_bps) if side > 0 else 1.0 - from_bps(total_bps)
        else:
            # on exit, for long selling we lose bps; for short cover we pay up bps
            adj = 1.0 - from_bps(total_bps) if side > 0 else 1.0 + from_bps(total_bps)
        return raw_price * adj

    def _risk_scaler(self, i: int) -> float:
        # Scale down target risk when realized vol is high
        rv20 = float(self.df.loc[i, "rv20"]) if not np.isnan(self.df.loc[i, "rv20"]) else 0.0
        cap = self.cfg["regime_vol_cap"]
        if rv20 <= 0 or np.isnan(rv20):
            return 1.0
        # Rough daily vol -> annual ~ rv20*sqrt(252); compare to cap
        ann = rv20 * math.sqrt(self.cfg["trading_days_per_year"])
        return max(0.3, min(1.0, cap / max(1e-8, ann)))  # between 0.3 and 1.0

    def _target_position(self, i: int, signal: int) -> int:
        # Compute shares using volatility scaling & fixed fractional risk
        if signal == 0:
            return 0
        atr_val = float(self.df.loc[i, "atr"])
        px = float(self.df.loc[i, "open"])  # we will execute at open[i] for signal from i-1
        if atr_val <= 0 or px <= 0:
            return 0

        equity = self.equity
        risk_per_trade = self.cfg["risk_fraction"] * equity * self._risk_scaler(i)
        stop_dist = self.cfg["stop_atr_mult"] * atr_val
        if stop_dist <= 0:
            return 0
        shares = int(risk_per_trade / stop_dist)
        shares = max(self.cfg["min_shares"], shares)

        # Enforce notional & leverage caps
        notional = shares * px
        max_inst_notional = self.notional_cap * equity
        max_total_notional = self.max_leverage * equity
        cap_shares_inst = int(max_inst_notional // px)
        cap_shares_total = int(max_total_notional // px)
        cap_shares = max(self.cfg["min_shares"], min(cap_shares_inst, cap_shares_total))
        shares = min(shares, cap_shares)

        return shares * int(math.copysign(1, signal))

    def _update_drawdown_halt(self):
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        dd = 1.0 - (self.equity / self.peak_equity if self.peak_equity > 0 else 1.0)
        if dd >= self.cfg["max_drawdown_halt"]:
            self.halt_new_entries = True

    def _record_trade(self, pos: Position, exit_price: float, exit_idx: int):
        side = pos.side
        entry_price_net = pos.entry_price
        exit_price_net = exit_price
        pnl_per_share = (exit_price_net - entry_price_net) * side * -1.0  # because entry was cost-adjusted
        pnl = pnl_per_share * pos.size
        pnl_pct = pnl / self.equity if self.equity != 0 else 0.0  # equity at exit
        bars_held = exit_idx - pos.entry_idx

        mae_pct = pos.max_adverse / pos.entry_price if pos.entry_price else 0.0
        mfe_pct = pos.max_favorable / pos.entry_price if pos.entry_price else 0.0

        self.trades.append(TradeRecord(
            entry_date=str(self.df.loc[pos.entry_idx, "date"].date()),
            exit_date=str(self.df.loc[exit_idx, "date"].date()),
            side="LONG" if side > 0 else "SHORT",
            entry_price=pos.entry_price,
            exit_price=exit_price_net,
            size=pos.size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            bars_held=bars_held,
            mae_pct=mae_pct,
            mfe_pct=mfe_pct,
        ))
        self.equity += pnl

    def _maybe_exit(self, i: int, raw_open: float):
        if self.open_position is None:
            return False
        pos = self.open_position
        side = pos.side
        atr_val = float(self.df.loc[i, "atr"])
        # Check stops against prior close to avoid peeking at current bar extremes for next-open exec model
        prev_close = float(self.df.loc[i-1, "close"]) if i > 0 else raw_open

        should_exit = False
        reason = ""

        # Time stop
        if (i - pos.entry_idx) >= self.cfg["max_holding_days"]:
            should_exit, reason = True, "time"

        # Stop-loss / Take-profit by reference to prev_close
        if not should_exit:
            if side > 0:
                if prev_close <= pos.stop_price:
                    should_exit, reason = True, "stop"
                elif pos.tp_price is not None and prev_close >= pos.tp_price:
                    should_exit, reason = True, "tp"
            else:
                if prev_close >= pos.stop_price:
                    should_exit, reason = True, "stop"
                elif pos.tp_price is not None and prev_close <= pos.tp_price:
                    should_exit, reason = True, "tp"

        if should_exit:
            # Execute at current open with exit costs
            exit_price_net = self._price_with_costs(raw_open, i, pos.side, is_entry=False)
            self._record_trade(pos, exit_price_net, i)
            self.open_position = None
            return True
        else:
            # Update MAE/MFE using prev_close vs entry
            delta = (prev_close - pos.entry_price) * (1 if side > 0 else -1)
            if delta < pos.max_adverse:
                pos.max_adverse = delta
            if delta > pos.max_favorable:
                pos.max_favorable = delta
            return False

    def _maybe_enter(self, i: int, signal: int, raw_open: float):
        if self.halt_new_entries:
            return
        if self.open_position is not None:
            return
        if signal == 0:
            return
        if self.df.loc[i, "can_trade"] == 0:
            return
        if (i - self.last_signal_idx) < self.cfg["signal_cooldown_bars"]:
            return
        if signal < 0 and not self.cfg["allow_short"]:
            return

        target_shares_signed = self._target_position(i, signal)
        if target_shares_signed == 0:
            return

        side = 1 if target_shares_signed > 0 else -1
        size = abs(int(target_shares_signed))
        entry_price_net = self._price_with_costs(raw_open, i, side, is_entry=True)

        atr_val = float(self.df.loc[i, "atr"])
        stop_dist = self.cfg["stop_atr_mult"] * atr_val
        tp_dist = (self.cfg["tp_atr_mult"] * atr_val) if self.cfg["tp_atr_mult"] is not None else None

        stop_price = entry_price_net - stop_dist if side > 0 else entry_price_net + stop_dist
        tp_price = (entry_price_net + tp_dist) if (tp_dist is not None and side > 0) else (
                   entry_price_net - tp_dist if tp_dist is not None else None)

        self.open_position = Position(side=side, entry_price=entry_price_net, size=size,
                                      entry_idx=i, stop_price=stop_price, tp_price=tp_price)
        self.last_signal_idx = i

    def _portfolio_daily_controls(self, date_key: str):
        # Daily loss limit: if exceeded, close and halt for next day
        daily = self.daily_pnl_tracker.get(date_key, 0.0)
        if self.equity > 0 and daily < -self.cfg["daily_loss_limit"] * self.equity:
            # Close any open position at current day's close (approximation), and halt.
            if self.open_position is not None:
                i = self._current_index
                raw_close = float(self.df.loc[i, "close"])
                exit_price_net = self._price_with_costs(raw_close, i, self.open_position.side, is_entry=False)
                self._record_trade(self.open_position, exit_price_net, i)
                self.open_position = None
            self.halt_new_entries = True

    def run(self) -> Dict:
        df = self.df
        n = len(df)
        if n < max(self.cfg["ema_slow"] + 5, 300):
            print(f"[WARN] {self.name}: dataset is short ({n} bars). Results may be unreliable.")

        prev_equity = self.equity

        for i in range(1, n):
            self._current_index = i
            date = df.loc[i, "date"]
            date_key = str(date.date())

            # Signal is computed from prior bar close
            signal = int(df.loc[i-1, "signal_raw"])

            # Portfolio daily controls from yesterday realized PnL
            self._portfolio_daily_controls(date_key)

            raw_open = float(df.loc[i, "open"])

            # 1) Exit checks (time, stop, tp)
            exited = self._maybe_exit(i, raw_open)

            # 2) Entry checks (if flat)
            self._maybe_enter(i, signal, raw_open)

            # Mark-to-market equity at close[i]
            mtm_px = float(df.loc[i, "close"])
            if self.open_position is not None:
                side = self.open_position.side
                # unrealized PnL approximated w/o extra costs (costs applied at entry/exit)
                pnl_unrl = (mtm_px - self.open_position.entry_price) * (1 if side > 0 else -1) * self.open_position.size
            else:
                pnl_unrl = 0.0

            equity_today = self.equity + pnl_unrl
            self.equity_curve.append(equity_today)

            # Daily return
            if prev_equity != 0:
                ret = (equity_today - prev_equity) / prev_equity
            else:
                ret = 0.0
            self.daily_returns.append(ret)
            self.daily_pnl_tracker[date_key] = self.daily_pnl_tracker.get(date_key, 0.0) + (equity_today - prev_equity)
            prev_equity = equity_today

            # Update portfolio-level halts from DD
            self._update_drawdown_halt()

        # Close any open position on final bar at close with exit costs
        if self.open_position is not None and len(df) > 0:
            i = len(df) - 1
            raw_close = float(df.loc[i, "close"])
            exit_price_net = self._price_with_costs(raw_close, i, self.open_position.side, is_entry=False)
            self._record_trade(self.open_position, exit_price_net, i)
            self.open_position = None

        # Metrics
        eq = np.array(self.equity_curve, dtype=float)
        rets = np.array(self.daily_returns, dtype=float)
        mdd, peak_i, trough_i = max_drawdown(eq) if len(eq) else (0.0, 0, 0)
        ann_ret = annualize_return(rets, self.cfg["trading_days_per_year"])
        shrp = sharpe_ratio(rets, self.cfg["risk_free_rate_annual"], self.cfg["trading_days_per_year"])
        sort = sortino_ratio(rets, self.cfg["risk_free_rate_annual"], self.cfg["trading_days_per_year"])

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        win_rate = (len(wins) / len(self.trades)) if self.trades else 0.0
        avg_hold = float(np.mean([t.bars_held for t in self.trades])) if self.trades else 0.0

        # Maximum Dip (largest intra-trade adverse move % from entry, i.e., worst MAE)
        max_dip = 0.0
        if self.trades:
            max_dip = max([-t.mae_pct for t in self.trades] + [0.0])

        metrics = {
            "dataset": self.name,
            "final_equity": float(self.equity),
            "net_return_pct": float((self.equity / self.cfg['initial_capital'] - 1.0) * 100.0),
            "annualized_return_pct": float(ann_ret * 100.0),
            "annualized_sharpe": float(shrp),
            "sortino": float(sort),
            "max_drawdown_pct": float(mdd * 100.0),
            "win_rate_pct": float(win_rate * 100.0),
            "avg_holding_period_bars": float(avg_hold),
            "num_trades": int(len(self.trades)),
            "max_dip_pct": float(max_dip * 100.0),
            "peak_index": int(peak_i),
            "trough_index": int(trough_i),
        }
        return metrics

# -----------------------------
# Strategy wrapper (same logic for all assets)
# -----------------------------

def backtest_dataset(df: pd.DataFrame, name: str, outdir: Path, config: Dict) -> Dict:
    bt = SingleAssetBacktester(df, name, config)
    metrics = bt.run()

    # Save artifacts
    trades = pd.DataFrame([asdict(t) for t in bt.trades])
    equity = pd.DataFrame({
        "date": df["date"].iloc[-len(bt.equity_curve):].values if len(df) >= len(bt.equity_curve) else pd.date_range(periods=len(bt.equity_curve), start=0),
        "equity": bt.equity_curve
    })
    outdir.mkdir(parents=True, exist_ok=True)
    trades_path = outdir / f"trades_{name}.csv"
    equity_path = outdir / f"equity_{name}.csv"
    metrics_path = outdir / f"metrics_{name}.json"
    trades.to_csv(trades_path, index=False)
    equity.to_csv(equity_path, index=False)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot equity curve
    plt.figure()
    plt.plot(equity["date"], equity["equity"])
    plt.title(f"Equity Curve — {name}")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    fig_path = outdir / f"equity_{name}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    return metrics

# -----------------------------
# Synthetic data generator (for demo if files not provided)
# -----------------------------

def synth_dataset(seed: int, n: int = 1500) -> pd.DataFrame:
    rnd = np.random.RandomState(seed)
    dates = pd.bdate_range(end=dt.date.today(), periods=n)
    price = 100.0
    prices = []
    vols = []
    for i in range(n):
        # Regime switching: trending vs mean-reverting vs high vol vs illiquid-like (gappy)
        if i % 400 < 120:
            drift = 0.0006
            vol = 0.01
        elif i % 400 < 200:
            drift = -0.0004
            vol = 0.012
        elif i % 400 < 320:
            drift = 0.0
            vol = 0.02
        else:
            drift = 0.0001
            vol = 0.03
        ret = rnd.normal(drift, vol)
        price *= math.exp(ret)
        prices.append(price)
        vols.append(vol)
    prices = np.array(prices)
    # Create OHLC with realistic ranges
    close = prices
    open_ = np.concatenate([[prices[0]], prices[:-1]])
    hl_spread = np.maximum(0.002, np.array(vols)) * prices
    high = np.maximum.reduce([open_, close]) + 0.5 * hl_spread
    low = np.minimum.reduce([open_, close]) - 0.5 * hl_spread
    volume = (1e6 * (1 + 0.1 * np.sin(np.linspace(0, 10, n))) * (1 + 0.5 * np.random.rand(n))).astype(int)
    df = pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })
    return df

# -----------------------------
# CLI
# -----------------------------

import json  # needed for metrics dump above

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="*", help="Paths to 4 CSV datasets (OHLCV)")
    parser.add_argument("--outdir", default="bt_results", help="Output directory for artifacts")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    if args.data and len(args.data) == 4:
        datasets = []
        for i, p in enumerate(args.data, 1):
            path = Path(p)
            if not path.exists():
                raise FileNotFoundError(f"{path} not found")
            df = load_csv(path)
            datasets.append((df, f"Dataset{i}"))
    else:
        # Synthesize 4 datasets if not provided
        datasets = [(synth_dataset(7), "Dataset1"),
                    (synth_dataset(13), "Dataset2"),
                    (synth_dataset(29), "Dataset3"),
                    (synth_dataset(43), "Dataset4")]

    all_metrics = []
    for df, name in datasets:
        m = backtest_dataset(df, name, outdir, CONFIG)
        all_metrics.append(m)

    # Save aggregated metrics
    metrics_df = pd.DataFrame(all_metrics).set_index("dataset")
    metrics_df.to_csv(Path(outdir) / "metrics_summary.csv")
    print(metrics_df.round(3))

    # Produce short report template
    rows = []
    for m in all_metrics:
        rows.append("| {dataset} | {net_return_pct:.2f} | {annualized_sharpe:.2f} | {sortino:.2f} | {max_drawdown_pct:.2f} | {win_rate_pct:.1f} | {avg_holding_period_bars:.1f} | {max_dip_pct:.2f} | {num_trades} |".format(**m))
    table = "\n".join(rows)

    report_path = Path(outdir) / "short_report_template.md"
    report = (
        "# Short Report — Multi-Asset Strategy Backtesting (max 3 pages)\n\n"
        "**Date:** " + dt.date.today().isoformat() + "\n\n"
        "## Strategy Logic & Indicators\n"
        "- Core: EMA(" + str(CONFIG["ema_fast"]) + ")/EMA(" + str(CONFIG["ema_slow"]) + ") crossover\n"
        "- Filter: ATR(" + str(CONFIG["atr_lookback"]) + ") with min ATR% " + "{:.2f}".format(CONFIG["atr_filter_min"]*100.0) + "%\n"
        "- Execution: Signals from close[t-1], executed at open[t]; no lookahead.\n"
        "- Shorting: " + ("Enabled" if CONFIG["allow_short"] else "Disabled") + "\n\n"
        "## Risk Management\n"
        "- Trade-level: Stop " + str(CONFIG["stop_atr_mult"]) + "×ATR, Take-Profit " + str(CONFIG["tp_atr_mult"] if CONFIG["tp_atr_mult"] is not None else 0) + "×ATR, Time stop " + str(CONFIG["max_holding_days"]) + " bars.\n"
        "- Portfolio-level: Max Drawdown Halt " + "{:.1f}".format(CONFIG["max_drawdown_halt"]*100.0) + "%, Daily loss limit " + "{:.1f}".format(CONFIG["daily_loss_limit"]*100.0) + "%.\n"
        "- Slippage & Costs: Brokerage " + "{:.3f}".format(CONFIG["brokerage_bps_per_side"]/100.0) + "%/side + Slippage mode `" + CONFIG["slippage_mode"] + "`.\n\n"
        "## Position Sizing\n"
        "- Risk per trade: " + "{:.1f}".format(CONFIG["risk_fraction"]*100.0) + "% of equity with volatility scaling.\n"
        "- Caps: Max leverage " + str(CONFIG["max_leverage"]) + "×, per-instrument notional cap " + "{:.0f}".format(CONFIG["per_instrument_notional_cap"]*100.0) + "%.\n\n"
        "## Performance Summary\n"
        "(Replace with actual results when run on provided datasets.)\n\n"
        "| Dataset | Net Return % | Ann. Sharpe | Sortino | Max DD % | Win Rate % | Avg Hold (bars) | Max Dip % | Trades |\n"
        "|--------|---------------|-------------|---------|----------|------------|-----------------|-----------|--------|\n"
        + table + "\n\n"
        "## Key Observations Across Datasets\n"
        "- Regime sensitivity (trend vs mean-revert vs high-vol): ...\n"
        "- Impact of ATR filter and cooldown on whipsaw: ...\n"
        "- Cost drag under illiquidity proxy (slippage scaling): ...\n\n"
        "## Robustness & Lessons\n"
        "- Same logic, parameter stability across assets.\n"
        "- Effect of percent-costs and realistic execution.\n"
        "- Potential improvements: alternative filters (ADX), dynamic cooldown, volatility targeting at portfolio level, ML overlay (optional).\n\n"
        "*Appendix (optional):* equity curves and top trades per dataset in `bt_results/`.\n"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\nArtifacts saved to:", outdir.resolve())
    print("Files: metrics_*.json, trades_*.csv, equity_*.csv/.png, metrics_summary.csv, short_report_template.md")

if __name__ == "__main__":
    main()
