"""
src/signals/signal_generator.py
Strategy registry and signal dispatch layer.

Usage:
    from src.signals.signal_generator import generate_signals, list_strategies

    df = generate_signals(df, strategy_name="ema_cross", config=config)
    # df now has a 'signal' column: 1=BUY, -1=SELL, 0=HOLD

The generator auto-computes whatever indicators the chosen strategy
declares in required_indicators before calling generate_signals() —
callers don't need to run compute() separately.
"""

import logging
from typing import Optional

import pandas as pd

from strategies.base_strategy          import BaseStrategy
from strategies.ema_cross              import EMACrossStrategy
from strategies.rsi_mean_reversion     import RSIMeanReversionStrategy

logger = logging.getLogger(__name__)

# ── Strategy Registry ─────────────────────────────────────────────────────────
# Maps strategy slug → strategy class.
# Add new strategies here and they'll be available in the CLI immediately.

STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "ema_cross":           EMACrossStrategy,
    "rsi_mean_reversion":  RSIMeanReversionStrategy,
}


# ── Public API ────────────────────────────────────────────────────────────────

def list_strategies() -> list[str]:
    """Return a sorted list of available strategy names."""
    return sorted(STRATEGY_REGISTRY.keys())


def get_strategy(name: str, config: Optional[dict] = None) -> BaseStrategy:
    """
    Instantiate a strategy by name, reading parameters from config.

    Args:
        name:   Strategy slug, e.g. "ema_cross".
        config: Full settings.yaml dict (or None for defaults).

    Returns:
        Strategy instance ready to call generate_signals() on.

    Raises:
        ValueError: Unknown strategy name.
    """
    key = name.strip().lower()
    if key not in STRATEGY_REGISTRY:
        available = list_strategies()
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {available}"
        )

    cls = STRATEGY_REGISTRY[key]
    cfg = config or {}

    # Use from_config() if the class defines it, otherwise use defaults
    if hasattr(cls, "from_config"):
        return cls.from_config(cfg)
    return cls()


def generate_signals(
    df:            pd.DataFrame,
    strategy_name: str,
    config:        Optional[dict] = None,
) -> pd.DataFrame:
    """
    Full pipeline: auto-compute required indicators → produce signal column.

    Args:
        df:            OHLCV DataFrame (from DataManager).
        strategy_name: Strategy slug, e.g. "ema_cross".
        config:        Full settings.yaml dict (or None for defaults).

    Returns:
        DataFrame with 'signal' column (1/−1/0) and 'strategy' column added.

    Raises:
        ValueError: Unknown strategy or missing required columns.
    """
    from src.indicators.indicators import compute  # local import to avoid circularity

    strategy = get_strategy(strategy_name, config)
    logger.info(
        f"[signal_generator] strategy={strategy.name} "
        f"required_indicators={strategy.required_indicators}"
    )

    # Auto-compute whatever indicators the strategy needs
    if strategy.required_indicators:
        df = compute(df, strategy.required_indicators, config=config)

    # Let the strategy do its work
    return strategy.generate_signals(df)


# ── Signal Statistics ─────────────────────────────────────────────────────────

def signal_summary(df: pd.DataFrame) -> dict:
    """
    Return a summary dict for a DataFrame that has a 'signal' column.

    Keys: total_bars, buy_signals, sell_signals, hold_bars,
          signal_rate_pct, first_signal_date, last_signal_date
    """
    if "signal" not in df.columns:
        raise ValueError("DataFrame has no 'signal' column — run generate_signals() first.")

    sig        = df["signal"]
    active     = sig[sig != 0]
    total      = len(sig)
    n_buy      = int((sig == 1).sum())
    n_sell     = int((sig == -1).sum())
    n_hold     = int((sig == 0).sum())
    rate_pct   = round(len(active) / total * 100, 2) if total else 0.0

    return {
        "total_bars":        total,
        "buy_signals":       n_buy,
        "sell_signals":      n_sell,
        "hold_bars":         n_hold,
        "signal_rate_pct":   rate_pct,
        "first_signal_date": str(active.index[0].date())  if not active.empty else "—",
        "last_signal_date":  str(active.index[-1].date()) if not active.empty else "—",
    }
