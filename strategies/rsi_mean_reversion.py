"""
strategies/rsi_mean_reversion.py
RSI Mean-Reversion Strategy

Logic:
  BUY  → RSI crosses DOWN through oversold threshold  (e.g. < 30)
  SELL → RSI crosses UP through overbought threshold  (e.g. > 70)
  HOLD → all other bars

The crossover approach avoids spamming signals while RSI stays in the
extreme zone — we only fire once, on entry into that zone.

Configuration (settings.yaml → signals.rsi_mean_reversion):
  rsi_period:  14   (default)
  oversold:    30   (default)
  overbought:  70   (default)
"""

import logging

import pandas as pd

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class RSIMeanReversionStrategy(BaseStrategy):

    name                 = "rsi_mean_reversion"
    description          = "RSI mean-reversion — BUY on oversold entry, SELL on overbought entry"
    required_indicators  = ["RSI"]

    def __init__(
        self,
        rsi_period:  int = 14,
        oversold:    int = 30,
        overbought:  int = 70,
    ) -> None:
        if not (0 < oversold < overbought < 100):
            raise ValueError(
                f"Thresholds must satisfy 0 < oversold < overbought < 100. "
                f"Got oversold={oversold}, overbought={overbought}"
            )
        self.rsi_period  = rsi_period
        self.oversold    = oversold
        self.overbought  = overbought

    # ── Core ──────────────────────────────────────────────────────────────────

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Emit BUY when RSI enters oversold zone, SELL when it enters overbought.

        Expects column: rsi_{rsi_period}
        (auto-computed by signal_generator before this is called)
        """
        self.validate_input(df)
        df = df.copy()

        rsi_col = f"rsi_{self.rsi_period}"
        if rsi_col not in df.columns:
            raise ValueError(
                f"[{self.name}] Required column '{rsi_col}' missing. "
                f"Run compute(['RSI']) first."
            )

        rsi    = df[rsi_col]
        signal = pd.Series(0, index=df.index, dtype=int)

        # BUY: RSI crosses below oversold (enters oversold zone)
        in_oversold      = rsi < self.oversold
        was_in_oversold  = in_oversold.shift(1).fillna(False)
        buy_mask         = in_oversold & ~was_in_oversold
        signal[buy_mask] = 1

        # SELL: RSI crosses above overbought (enters overbought zone)
        in_overbought     = rsi > self.overbought
        was_in_overbought = in_overbought.shift(1).fillna(False)
        sell_mask         = in_overbought & ~was_in_overbought
        signal[sell_mask] = -1

        df["signal"]   = signal
        df["strategy"] = self.name

        n_buy  = int(buy_mask.sum())
        n_sell = int(sell_mask.sum())
        logger.info(
            f"[{self.name}] RSI({self.rsi_period}) "
            f"oversold<{self.oversold} overbought>{self.overbought} "
            f"→ {n_buy} BUY, {n_sell} SELL signals over {len(df)} bars"
        )
        return df

    # ── Helpers ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: dict) -> "RSIMeanReversionStrategy":
        """Instantiate from settings.yaml config dict."""
        cfg = config.get("signals", {}).get("rsi_mean_reversion", {})
        rsi = config.get("indicators", {}).get("rsi", {})
        return cls(
            rsi_period=cfg.get("rsi_period", rsi.get("period", 14)),
            oversold=cfg.get(
                "oversold", config.get("signals", {}).get("rsi_oversold", 30)
            ),
            overbought=cfg.get(
                "overbought", config.get("signals", {}).get("rsi_overbought", 70)
            ),
        )
