"""
src/backtest/runner.py
Backtesting engine — Stage 4.

Bridges our pre-computed signals (Stage 3) into backtesting.py's
Strategy pattern, extracts standardised metrics, and saves results.

Pipeline:
  OHLCV DataFrame
    → generate_signals()       (Stage 3)
    → prepare_df()             (column rename + tz strip)
    → Backtest(SignalStrategy) (backtesting.py 0.6.x)
    → extract_metrics()        (standardised dict)
    → save_results()           (JSON + CSV in results/)

Usage:
    from src.backtest.runner import run_backtest, compare_strategies

    result = run_backtest(df, "ema_cross", config=config)
    print(result["metrics"])

    comparison = compare_strategies(df, ["ema_cross", "rsi_mean_reversion"], config=config)
"""

import io
import json
import logging
import contextlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Strategy Adapter ──────────────────────────────────────────────────────────

def _make_signal_strategy(signal_arr: np.ndarray) -> type:
    """
    Factory: returns a backtesting.py Strategy class that drives trades
    from a pre-computed signal array captured via closure.

    Signal values:
      1  → BUY  — open long if no position
     -1  → SELL — close long if in position
      0  → HOLD — no action

    Args:
        signal_arr: 1-D numpy int array, same length as the OHLCV DataFrame.

    Returns:
        A Strategy subclass ready to pass to Backtest().
    """
    from backtesting import Strategy
    _sig = np.array(signal_arr, dtype=float)   # float required by backtesting.py I()

    class _SignalStrategy(Strategy):
        def init(self) -> None:
            self._signal = self.I(lambda: _sig, name="Signal", plot=False)

        def next(self) -> None:
            s = self._signal[-1]
            if s == 1 and not self.position:
                self.buy()
            elif s == -1 and self.position:
                self.position.close()

    return _SignalStrategy


# ── DataFrame Preparation ─────────────────────────────────────────────────────

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert our internal OHLCV DataFrame to backtesting.py format:

    - Rename lowercase ohlcv columns → title-case (Open/High/Low/Close/Volume)
    - Strip timezone from DatetimeIndex (backtesting.py requires tz-naive)
    - Drop non-OHLCV columns (signals, indicators) to keep the engine clean;
      signals are injected via the closure instead.

    Args:
        df: Internal OHLCV DataFrame with lowercase columns and UTC index.

    Returns:
        Clean DataFrame ready for Backtest().
    """
    rename_map = {
        "open":   "Open",
        "high":   "High",
        "low":    "Low",
        "close":  "Close",
        "volume": "Volume",
    }
    bt_df = df.copy()
    bt_df = bt_df.rename(columns={k: v for k, v in rename_map.items() if k in bt_df.columns})

    # Keep only OHLCV — all extras (signal, strategy, indicators) are dropped.
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in bt_df.columns]
    bt_df = bt_df[keep]

    # backtesting.py requires a tz-naive DatetimeIndex
    if bt_df.index.tz is not None:
        bt_df.index = bt_df.index.tz_localize(None)

    return bt_df


# ── Metrics Extraction ────────────────────────────────────────────────────────

def extract_metrics(stats: pd.Series) -> dict:
    """
    Extract key performance metrics from a backtesting.py Stats Series
    into a plain, JSON-serialisable dict.

    Handles NaN / missing keys gracefully — returns 0.0 / 0 as defaults.

    Args:
        stats: pd.Series returned by Backtest.run().

    Returns:
        Dict with standardised metric names and numeric values.
    """
    def _f(key: str, default: float = 0.0) -> float:
        """Safe float extraction — converts NaN to default."""
        try:
            val = stats[key]
            if pd.isna(val):
                return default
            return round(float(val), 4)
        except (KeyError, TypeError, ValueError):
            return default

    def _i(key: str, default: int = 0) -> int:
        try:
            val = stats[key]
            if pd.isna(val):
                return default
            return int(val)
        except (KeyError, TypeError, ValueError):
            return default

    return {
        "total_return_pct":     round(_f("Return [%]"),              2),
        "buy_hold_return_pct":  round(_f("Buy & Hold Return [%]"),   2),
        "annualised_return_pct":round(_f("Return (Ann.) [%]"),       2),
        "annualised_vol_pct":   round(_f("Volatility (Ann.) [%]"),   2),
        "sharpe_ratio":         round(_f("Sharpe Ratio"),             3),
        "sortino_ratio":        round(_f("Sortino Ratio"),            3),
        "calmar_ratio":         round(_f("Calmar Ratio"),             3),
        "max_drawdown_pct":     round(_f("Max. Drawdown [%]"),        2),
        "avg_drawdown_pct":     round(_f("Avg. Drawdown [%]"),        2),
        "win_rate_pct":         round(_f("Win Rate [%]"),             2),
        "profit_factor":        round(_f("Profit Factor"),            3),
        "expectancy_pct":       round(_f("Expectancy [%]"),           3),
        "total_trades":         _i("# Trades"),
        "exposure_time_pct":    round(_f("Exposure Time [%]"),        2),
        "equity_final":         round(_f("Equity Final [$]"),         2),
        "equity_peak":          round(_f("Equity Peak [$]"),          2),
        "commissions":          round(_f("Commissions [$]"),          2),
    }


# ── Results Persistence ───────────────────────────────────────────────────────

def save_results(
    metrics:       dict,
    strategy_name: str,
    ticker:        str,
    df_sig:        pd.DataFrame,
    results_dir:   str = "results",
) -> tuple[Path, Path]:
    """
    Persist backtest output to the results/ directory (git-ignored).

    Saves two files:
      {ticker}_{strategy}_{timestamp}_metrics.json   ← performance dict
      {ticker}_{strategy}_{timestamp}_signals.csv    ← full signal DataFrame

    Returns:
        (json_path, csv_path) as Path objects.
    """
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)

    stamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{ticker.upper()}_{strategy_name}_{stamp}"

    json_path = out / f"{base_name}_metrics.json"
    csv_path  = out / f"{base_name}_signals.csv"

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    df_sig.to_csv(csv_path)

    logger.info(f"[backtest] Results saved → {json_path.name}, {csv_path.name}")
    return json_path, csv_path


# ── Main Backtest Runner ──────────────────────────────────────────────────────

def run_backtest(
    df:            pd.DataFrame,
    strategy_name: str,
    config:        Optional[dict] = None,
    cash:          float = 10_000.0,
    commission:    float = 0.001,
    results_dir:   str   = "results",
    save:          bool  = True,
) -> dict:
    """
    Full backtesting pipeline for one strategy.

    Steps:
      1. Generate signals via Stage 3 signal_generator
      2. Prepare OHLCV DataFrame for backtesting.py
      3. Build a SignalStrategy class capturing signals via closure
      4. Run Backtest and extract metrics
      5. (Optional) Save results to results/

    Args:
        df:            Raw OHLCV DataFrame from DataManager.
        strategy_name: Strategy slug, e.g. "ema_cross".
        config:        Full settings.yaml dict (or None for defaults).
        cash:          Starting portfolio cash.
        commission:    Per-trade commission fraction (0.001 = 0.1%).
        results_dir:   Directory for JSON/CSV output.
        save:          Whether to persist results to disk.

    Returns:
        dict with keys:
          "metrics"  : standardised metrics dict
          "strategy" : strategy name
          "ticker"   : ticker symbol
          "stats"    : raw backtesting.py Stats Series
          "json_path": Path to saved JSON (or None if save=False)
          "csv_path" : Path to saved CSV  (or None if save=False)
    """
    from backtesting import Backtest
    from src.signals.signal_generator import generate_signals

    cfg    = config or {}
    ticker = df.attrs.get("ticker", "UNKNOWN")

    logger.info(
        f"[backtest] Running | ticker={ticker} strategy={strategy_name} "
        f"cash={cash:,.0f} commission={commission}"
    )

    # ── 1. Generate signals ───────────────────────────────────────────────────
    df_sig     = generate_signals(df, strategy_name=strategy_name, config=cfg)
    signal_arr = df_sig["signal"].fillna(0).to_numpy()

    # ── 2. Prepare OHLCV for backtesting.py ──────────────────────────────────
    bt_df = prepare_df(df_sig)

    # ── 3. Build strategy with signals captured in closure ────────────────────
    StrategyClass = _make_signal_strategy(signal_arr)

    # ── 4. Run (suppress tqdm progress bar that goes to stderr) ───────────────
    bt = Backtest(
        bt_df,
        StrategyClass,
        cash=cash,
        commission=commission,
        exclusive_orders=True,
        finalize_trades=True,   # close any open position at period end
    )

    with contextlib.redirect_stderr(io.StringIO()):
        stats = bt.run()

    metrics = extract_metrics(stats)

    n_trades = metrics["total_trades"]
    logger.info(
        f"[backtest] Done | {n_trades} trades | "
        f"return={metrics['total_return_pct']}% | "
        f"sharpe={metrics['sharpe_ratio']}"
    )

    # ── 5. Save results ───────────────────────────────────────────────────────
    json_path, csv_path = None, None
    if save:
        json_path, csv_path = save_results(
            metrics, strategy_name, ticker, df_sig, results_dir
        )

    return {
        "metrics":   metrics,
        "strategy":  strategy_name,
        "ticker":    ticker,
        "stats":     stats,
        "json_path": json_path,
        "csv_path":  csv_path,
    }


# ── Multi-Strategy Comparison ─────────────────────────────────────────────────

def compare_strategies(
    df:             pd.DataFrame,
    strategy_names: list[str],
    config:         Optional[dict] = None,
    cash:           float = 10_000.0,
    commission:     float = 0.001,
    results_dir:    str   = "results",
    save:           bool  = True,
) -> dict[str, dict]:
    """
    Run the same backtest period for multiple strategies and collect metrics.

    Args:
        df:              Raw OHLCV DataFrame (same data used for all strategies).
        strategy_names:  List of strategy slugs.
        config:          Settings dict.
        cash, commission, results_dir: passed through to run_backtest().

    Returns:
        Dict mapping strategy_name → metrics dict (or {"error": msg} on failure).
    """
    results: dict[str, dict] = {}

    for name in strategy_names:
        try:
            result = run_backtest(
                df.copy(),
                strategy_name=name,
                config=config,
                cash=cash,
                commission=commission,
                results_dir=results_dir,
                save=save,
            )
            results[name] = result["metrics"]
            logger.info(f"[compare] {name} → return={results[name]['total_return_pct']}%")
        except Exception as e:
            logger.error(f"[compare] {name} failed: {e}")
            results[name] = {"error": str(e)}

    return results
