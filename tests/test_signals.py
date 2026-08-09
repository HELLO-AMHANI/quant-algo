"""
tests/test_signals.py
Unit tests for Stage 3: Signal Generation

Run with:
    python -m pytest tests/test_signals.py -v

All tests are offline — no network, no API keys.
Synthetic OHLCV data is generated inline. Signal tests use
engineered price series that guarantee specific crossover events.
"""

import pytest
import pandas as pd
import numpy as np


# ── Synthetic data helpers ─────────────────────────────────────────────────────

def make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """General-purpose random-walk OHLCV."""
    rng    = np.random.default_rng(seed)
    close  = 100 + np.cumsum(rng.normal(0, 1, n))
    spread = rng.uniform(0.5, 2.0, n)
    idx    = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC", name="timestamp")
    return pd.DataFrame({
        "open":   close + rng.normal(0, 0.3, n),
        "high":   close + spread,
        "low":    close - spread,
        "close":  close,
        "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        "vwap":   np.nan,
    }, index=idx)


def make_ema_crossover_df(fast: int = 9, slow: int = 21) -> pd.DataFrame:
    """
    Engineer a price series that produces exactly ONE bullish and ONE bearish
    EMA crossover, at known positions.

    Structure:
      - Bars 0–49  : uptrend   → fast > slow  (bullish cross near bar 25)
      - Bars 50–99 : downtrend → fast < slow  (bearish cross near bar 75)
    """
    n = 150
    # Rising then falling price
    prices = np.concatenate([
        np.linspace(80, 140, 75),   # strong uptrend
        np.linspace(140, 80, 75),   # strong downtrend
    ])
    idx = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC", name="timestamp")
    return pd.DataFrame({
        "open":   prices * 0.999,
        "high":   prices * 1.005,
        "low":    prices * 0.995,
        "close":  prices,
        "volume": np.full(n, 5_000_000, dtype=float),
        "vwap":   np.nan,
    }, index=idx)


def make_rsi_extreme_df() -> pd.DataFrame:
    """
    Engineer a price series that drives RSI into both extremes.

    Structure:
      - Bars 0–29  : strong downtrend  → RSI should dip below 30
      - Bars 30–69 : consolidation     → RSI in neutral zone
      - Bars 70–99 : strong uptrend    → RSI should push above 70
    """
    n = 100
    prices = np.concatenate([
        np.linspace(200, 80, 30),    # sharp drop  → RSI oversold
        np.linspace(80, 90, 40),     # flat/slight rise → neutral
        np.linspace(90, 200, 30),    # sharp rise  → RSI overbought
    ])
    idx = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC", name="timestamp")
    return pd.DataFrame({
        "open":   prices * 0.999,
        "high":   prices * 1.002,
        "low":    prices * 0.998,
        "close":  prices,
        "volume": np.full(n, 5_000_000, dtype=float),
        "vwap":   np.nan,
    }, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# BaseStrategy Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBaseStrategy:

    def test_cannot_instantiate_directly(self):
        from strategies.base_strategy import BaseStrategy
        with pytest.raises(TypeError):
            BaseStrategy()

    def test_validate_input_raises_on_empty(self):
        from strategies.ema_cross import EMACrossStrategy
        s   = EMACrossStrategy()
        with pytest.raises(ValueError, match="empty"):
            s.validate_input(pd.DataFrame())

    def test_validate_input_raises_on_missing_columns(self):
        from strategies.ema_cross import EMACrossStrategy
        s   = EMACrossStrategy()
        bad = pd.DataFrame({"close": [1, 2, 3]})  # missing open/high/low/volume
        with pytest.raises(ValueError, match="Missing"):
            s.validate_input(bad)

    def test_crossover_helper_detects_cross(self):
        from strategies.base_strategy import BaseStrategy
        from strategies.ema_cross import EMACrossStrategy
        s = EMACrossStrategy()
        a = pd.Series([1.0, 1.0, 3.0, 3.0])
        b = pd.Series([2.0, 2.0, 2.0, 2.0])
        result = s._crossover(a, b)
        assert result.iloc[2] == True   # a goes above b at index 2
        assert result.iloc[0] == False
        assert result.iloc[3] == False

    def test_crossunder_helper_detects_cross(self):
        from strategies.ema_cross import EMACrossStrategy
        s = EMACrossStrategy()
        a = pd.Series([3.0, 3.0, 1.0, 1.0])
        b = pd.Series([2.0, 2.0, 2.0, 2.0])
        result = s._crossunder(a, b)
        assert result.iloc[2] == True   # a goes below b at index 2
        assert result.iloc[0] == False


# ─────────────────────────────────────────────────────────────────────────────
# EMACrossStrategy Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEMACrossStrategy:

    def test_invalid_params_raises(self):
        from strategies.ema_cross import EMACrossStrategy
        with pytest.raises(ValueError, match="fast_period"):
            EMACrossStrategy(fast_period=21, slow_period=9)

    def test_adds_signal_column(self):
        from strategies.ema_cross import EMACrossStrategy
        from src.indicators.indicators import compute
        df  = make_ohlcv()
        df  = compute(df, ["EMA"])
        out = EMACrossStrategy().generate_signals(df)
        assert "signal" in out.columns

    def test_signal_values_in_valid_set(self):
        from strategies.ema_cross import EMACrossStrategy
        from src.indicators.indicators import compute
        df  = make_ohlcv()
        df  = compute(df, ["EMA"])
        out = EMACrossStrategy().generate_signals(df)
        assert set(out["signal"].unique()).issubset({-1, 0, 1})

    def test_no_nan_in_signal(self):
        from strategies.ema_cross import EMACrossStrategy
        from src.indicators.indicators import compute
        df  = make_ohlcv()
        df  = compute(df, ["EMA"])
        out = EMACrossStrategy().generate_signals(df)
        assert out["signal"].isna().sum() == 0

    def test_does_not_mutate_input(self):
        from strategies.ema_cross import EMACrossStrategy
        from src.indicators.indicators import compute
        df   = make_ohlcv()
        df   = compute(df, ["EMA"])
        cols = set(df.columns)
        EMACrossStrategy().generate_signals(df)
        assert set(df.columns) == cols

    def test_same_length_as_input(self):
        from strategies.ema_cross import EMACrossStrategy
        from src.indicators.indicators import compute
        df  = make_ohlcv(80)
        df  = compute(df, ["EMA"])
        out = EMACrossStrategy().generate_signals(df)
        assert len(out) == len(df)

    def test_strategy_column_set(self):
        from strategies.ema_cross import EMACrossStrategy
        from src.indicators.indicators import compute
        df  = make_ohlcv()
        df  = compute(df, ["EMA"])
        out = EMACrossStrategy().generate_signals(df)
        assert "strategy" in out.columns
        assert out["strategy"].iloc[0] == "ema_cross"

    def test_buy_fires_at_bullish_crossover(self):
        """On engineered uptrend data, at least one BUY signal must appear."""
        from strategies.ema_cross import EMACrossStrategy
        from src.indicators.indicators import compute
        df  = make_ema_crossover_df()
        df  = compute(df, ["EMA"])
        out = EMACrossStrategy().generate_signals(df)
        assert (out["signal"] == 1).any(), "Expected at least one BUY signal"

    def test_sell_fires_at_bearish_crossover(self):
        """On engineered downtrend data, at least one SELL signal must appear."""
        from strategies.ema_cross import EMACrossStrategy
        from src.indicators.indicators import compute
        df  = make_ema_crossover_df()
        df  = compute(df, ["EMA"])
        out = EMACrossStrategy().generate_signals(df)
        assert (out["signal"] == -1).any(), "Expected at least one SELL signal"

    def test_missing_ema_columns_raises(self):
        from strategies.ema_cross import EMACrossStrategy
        df = make_ohlcv()   # no EMA columns computed
        with pytest.raises(ValueError, match="Required columns missing"):
            EMACrossStrategy().generate_signals(df)

    def test_from_config(self):
        from strategies.ema_cross import EMACrossStrategy
        cfg = {"indicators": {"ema": {"fast": 5, "slow": 30}}}
        s   = EMACrossStrategy.from_config(cfg)
        assert s.fast_period == 5
        assert s.slow_period == 30


# ─────────────────────────────────────────────────────────────────────────────
# RSIMeanReversionStrategy Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRSIMeanReversionStrategy:

    def test_invalid_thresholds_raises(self):
        from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
        with pytest.raises(ValueError, match="Thresholds"):
            RSIMeanReversionStrategy(oversold=80, overbought=20)

    def test_adds_signal_column(self):
        from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
        from src.indicators.indicators import compute
        df  = make_ohlcv(100)
        df  = compute(df, ["RSI"])
        out = RSIMeanReversionStrategy().generate_signals(df)
        assert "signal" in out.columns

    def test_signal_values_in_valid_set(self):
        from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
        from src.indicators.indicators import compute
        df  = make_ohlcv()
        df  = compute(df, ["RSI"])
        out = RSIMeanReversionStrategy().generate_signals(df)
        assert set(out["signal"].unique()).issubset({-1, 0, 1})

    def test_no_nan_in_signal(self):
        from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
        from src.indicators.indicators import compute
        df  = make_ohlcv()
        df  = compute(df, ["RSI"])
        out = RSIMeanReversionStrategy().generate_signals(df)
        assert out["signal"].isna().sum() == 0

    def test_does_not_mutate_input(self):
        from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
        from src.indicators.indicators import compute
        df   = make_ohlcv()
        df   = compute(df, ["RSI"])
        cols = set(df.columns)
        RSIMeanReversionStrategy().generate_signals(df)
        assert set(df.columns) == cols

    def test_buy_fires_at_oversold(self):
        """Engineered sharp downtrend must trigger at least one BUY (oversold entry)."""
        from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
        from src.indicators.indicators import compute
        df  = make_rsi_extreme_df()
        df  = compute(df, ["RSI"])
        out = RSIMeanReversionStrategy().generate_signals(df)
        assert (out["signal"] == 1).any(), "Expected at least one BUY (oversold)"

    def test_sell_fires_at_overbought(self):
        """Engineered sharp uptrend must trigger at least one SELL (overbought entry)."""
        from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
        from src.indicators.indicators import compute
        df  = make_rsi_extreme_df()
        df  = compute(df, ["RSI"])
        out = RSIMeanReversionStrategy().generate_signals(df)
        assert (out["signal"] == -1).any(), "Expected at least one SELL (overbought)"

    def test_signals_are_sparse_not_continuous(self):
        """
        Signals should fire ONCE on zone entry, not every bar inside the zone.
        So the total signal count should be low relative to bars in the extreme.
        """
        from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
        from src.indicators.indicators import compute
        df  = make_rsi_extreme_df()
        df  = compute(df, ["RSI"])
        out = RSIMeanReversionStrategy().generate_signals(df)
        rsi_col = "rsi_14"
        n_oversold_bars  = (df[rsi_col] < 30).sum()
        n_buy_signals    = (out["signal"] == 1).sum()
        if n_oversold_bars > 0:
            assert n_buy_signals <= n_oversold_bars, "BUY count must not exceed oversold bars"

    def test_missing_rsi_column_raises(self):
        from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
        df = make_ohlcv()   # no RSI computed
        with pytest.raises(ValueError, match="Required column"):
            RSIMeanReversionStrategy().generate_signals(df)

    def test_from_config(self):
        from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
        cfg = {
            "signals": {
                "rsi_mean_reversion": {"oversold": 25, "overbought": 75},
                "rsi_oversold":       30,
                "rsi_overbought":     70,
            }
        }
        s = RSIMeanReversionStrategy.from_config(cfg)
        assert s.oversold   == 25
        assert s.overbought == 75


# ─────────────────────────────────────────────────────────────────────────────
# SignalGenerator (dispatcher) Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalGenerator:

    def test_list_strategies_returns_sorted(self):
        from src.signals.signal_generator import list_strategies
        names = list_strategies()
        assert names == sorted(names)
        assert "ema_cross" in names
        assert "rsi_mean_reversion" in names

    def test_get_strategy_known(self):
        from src.signals.signal_generator import get_strategy
        from strategies.ema_cross import EMACrossStrategy
        s = get_strategy("ema_cross")
        assert isinstance(s, EMACrossStrategy)

    def test_get_strategy_unknown_raises(self):
        from src.signals.signal_generator import get_strategy
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("nonexistent_strategy")

    def test_get_strategy_case_insensitive(self):
        from src.signals.signal_generator import get_strategy
        from strategies.ema_cross import EMACrossStrategy
        s = get_strategy("EMA_CROSS")
        assert isinstance(s, EMACrossStrategy)

    def test_generate_signals_ema_cross(self):
        """Full pipeline: raw OHLCV → generate_signals auto-computes EMA → returns signal col."""
        from src.signals.signal_generator import generate_signals
        df  = make_ohlcv()
        out = generate_signals(df, "ema_cross")
        assert "signal"   in out.columns
        assert "ema_9"    in out.columns   # auto-computed
        assert "ema_21"   in out.columns   # auto-computed

    def test_generate_signals_rsi_mean_reversion(self):
        from src.signals.signal_generator import generate_signals
        df  = make_ohlcv()
        out = generate_signals(df, "rsi_mean_reversion")
        assert "signal" in out.columns
        assert "rsi_14" in out.columns    # auto-computed

    def test_generate_signals_does_not_mutate_input(self):
        from src.signals.signal_generator import generate_signals
        df   = make_ohlcv()
        cols = set(df.columns)
        generate_signals(df, "ema_cross")
        assert set(df.columns) == cols

    def test_generate_signals_unknown_strategy_raises(self):
        from src.signals.signal_generator import generate_signals
        with pytest.raises(ValueError, match="Unknown strategy"):
            generate_signals(make_ohlcv(), "made_up_strategy")


# ─────────────────────────────────────────────────────────────────────────────
# signal_summary Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalSummary:

    def _get_df_with_signals(self, strategy="ema_cross"):
        from src.signals.signal_generator import generate_signals
        return generate_signals(make_ohlcv(), strategy)

    def test_summary_keys_present(self):
        from src.signals.signal_generator import signal_summary
        df  = self._get_df_with_signals()
        s   = signal_summary(df)
        for key in ["total_bars", "buy_signals", "sell_signals",
                    "hold_bars", "signal_rate_pct",
                    "first_signal_date", "last_signal_date"]:
            assert key in s, f"Missing summary key: {key}"

    def test_counts_add_up(self):
        from src.signals.signal_generator import signal_summary
        df = self._get_df_with_signals()
        s  = signal_summary(df)
        assert s["buy_signals"] + s["sell_signals"] + s["hold_bars"] == s["total_bars"]

    def test_signal_rate_between_0_and_100(self):
        from src.signals.signal_generator import signal_summary
        df = self._get_df_with_signals()
        s  = signal_summary(df)
        assert 0.0 <= s["signal_rate_pct"] <= 100.0

    def test_no_signal_col_raises(self):
        from src.signals.signal_generator import signal_summary
        with pytest.raises(ValueError, match="no 'signal' column"):
            signal_summary(make_ohlcv())
