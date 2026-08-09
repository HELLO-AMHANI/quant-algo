"""
strategies/ema_cross.py
EMA Crossover Strategy

Logic:
  BUY  → fast EMA crosses above slow EMA  (bullish momentum)
  SELL → fast EMA crosses below slow EMA  (bearish momentum)
  HOLD → all other bars

Configuration (settings.yaml → signals.ema_cross):
  fast_period: 9    (default)
  slow_period: 21   (default)

The required indicators (EMA fast + slow) are auto-computed by the
signal_generator before generate_signals() is called.
"""

import logging

import pandas as pd

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class EMACrossStrategy(BaseStrategy):

    name                 = "ema_cross"
    description          = "EMA crossover — BUY on fast/slow bullish cross, SELL on bearish"
    required_indicators  = ["EMA"]

    def __init__(self, fast_period: int = 9, slow_period: int = 21) -> None:
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be less than slow_period ({slow_period})"
            )
        self.fast_period = fast_period
        self.slow_period = slow_period

    # ── Core ──────────────────────────────────────────────────────────────────

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scan for EMA crossover events and emit BUY / SELL signals.

        Expects columns: ema_{fast_period}, ema_{slow_period}
        (auto-computed by signal_generator before this is called)
        """
        self.validate_input(df)
        df = df.copy()

        fast_col = f"ema_{self.fast_period}"
        slow_col = f"ema_{self.slow_period}"

        if fast_col not in df.columns or slow_col not in df.columns:
            raise ValueError(
                f"[{self.name}] Required columns missing: "
                f"expected '{fast_col}' and '{slow_col}'. "
                f"Run compute(['EMA']) first."
            )

        # Initialise all bars as HOLD (0)
        signal = pd.Series(0, index=df.index, dtype=int)

        # BUY: fast crosses above slow
        buy_mask = self._crossover(df[fast_col], df[slow_col])
        signal[buy_mask] = 1

        # SELL: fast crosses below slow
        sell_mask = self._crossunder(df[fast_col], df[slow_col])
        signal[sell_mask] = -1

        df["signal"]   = signal
        df["strategy"] = self.name

        n_buy  = int(buy_mask.sum())
        n_sell = int(sell_mask.sum())
        logger.info(
            f"[{self.name}] EMA({self.fast_period}/{self.slow_period}) "
            f"→ {n_buy} BUY, {n_sell} SELL signals over {len(df)} bars"
        )
        return df

    # ── Helpers ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: dict) -> "EMACrossStrategy":
        """Instantiate from settings.yaml config dict."""
        cfg = config.get("signals", {}).get("ema_cross", {})
        ema = config.get("indicators", {}).get("ema", {})
        return cls(
            fast_period=cfg.get("fast_period", ema.get("fast", 9)),
            slow_period=cfg.get("slow_period", ema.get("slow", 21)),
        )
