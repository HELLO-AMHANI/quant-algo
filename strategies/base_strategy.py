"""
strategies/base_strategy.py
Abstract base class every strategy must inherit from.

Contract:
  - generate_signals(df) receives a clean OHLCV + indicator DataFrame
  - Returns the same DataFrame with a 'signal' column added:
        1  → BUY  (enter long)
       -1  → SELL (exit long / short)
        0  → HOLD (no action)
  - Must not mutate the input DataFrame
  - Must never return NaN in the signal column

Each subclass declares:
  - name              : unique lowercase slug used in CLI and registry
  - description       : one-line human description
  - required_indicators: list of indicator names (from REGISTRY) the
                         signal_generator will auto-compute before calling
                         generate_signals()
"""

from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd


class BaseStrategy(ABC):

    name:                ClassVar[str]       = "base"
    description:         ClassVar[str]       = "Abstract base strategy"
    required_indicators: ClassVar[list[str]] = []

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Produce trade signals from OHLCV + indicator data.

        Args:
            df: DataFrame with OHLCV columns and any indicator columns this
                strategy declared in required_indicators (already computed).

        Returns:
            New DataFrame (copy of df) with 'signal' column added.
            Signal values: 1 (BUY), -1 (SELL), 0 (HOLD). Never NaN.
        """

    # ── Shared helpers ────────────────────────────────────────────────────────

    def validate_input(self, df: pd.DataFrame) -> None:
        """
        Check that df is non-empty and has the required OHLCV columns.
        Raises ValueError if anything is missing.
        """
        if df.empty:
            raise ValueError(f"[{self.name}] DataFrame is empty — nothing to signal.")
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"[{self.name}] Missing required OHLCV columns: {missing}"
            )

    @staticmethod
    def _crossover(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
        """
        Returns True on bars where series_a crosses ABOVE series_b.
        (i.e. was below or equal on previous bar, is above now)
        """
        above      = series_a > series_b
        # .fillna() can return numpy scalars — cast to bool Series before ~
        # to avoid deprecated bitwise inversion on numpy booleans (~np.bool_)
        prev_above = above.shift(1).fillna(False).astype(bool)
        return above & ~prev_above

    @staticmethod
    def _crossunder(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
        """
        Returns True on bars where series_a crosses BELOW series_b.
        (i.e. was above on previous bar, is at or below now)
        """
        below      = series_a < series_b
        prev_below = below.shift(1).fillna(False).astype(bool)
        return below & ~prev_below

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
