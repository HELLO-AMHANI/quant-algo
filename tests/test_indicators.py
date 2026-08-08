"""
tests/test_indicators.py
Unit tests for Stage 2: Indicators Engine

Run with:
    python -m pytest tests/test_indicators.py -v

All tests are offline — no network, no API keys required.
Synthetic OHLCV data is generated inline for each test.
"""

import pytest
import pandas as pd
import numpy as np

# ── Fixture: synthetic OHLCV DataFrame ────────────────────────────────────────

def make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Build a deterministic synthetic OHLCV DataFrame with enough rows for all
    indicators to warm up (RSI needs 14+, MACD needs 35+, BB needs 20+, etc.).
    Uses a seeded random walk so tests are reproducible.
    """
    rng   = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))          # random walk around 100
    spread = rng.uniform(0.5, 2.0, n)
    high   = close + spread
    low    = close - spread
    open_  = close + rng.normal(0, 0.3, n)
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)

    idx = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC", name="timestamp")
    df = pd.DataFrame({
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
        "vwap":   np.nan,        # yfinance default — NaN until VWAP indicator runs
    }, index=idx)
    df.attrs["ticker"] = "TEST"
    df.attrs["source"] = "synthetic"
    return df


# ── Individual indicator tests ─────────────────────────────────────────────────

class TestComputeRSI:

    def test_adds_rsi_column(self):
        from src.indicators.indicators import compute_rsi
        df = make_ohlcv()
        out = compute_rsi(df.copy(), period=14)
        assert "rsi_14" in out.columns

    def test_rsi_range(self):
        from src.indicators.indicators import compute_rsi
        df = make_ohlcv()
        out = compute_rsi(df.copy(), period=14)
        valid = out["rsi_14"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all(), "RSI must be 0–100"

    def test_rsi_custom_period(self):
        from src.indicators.indicators import compute_rsi
        df = make_ohlcv()
        out = compute_rsi(df.copy(), period=7)
        assert "rsi_7" in out.columns
        assert "rsi_14" not in out.columns

    def test_does_not_mutate_input(self):
        from src.indicators.indicators import compute_rsi
        df = make_ohlcv()
        cols_before = set(df.columns)
        compute_rsi(df, period=14)
        assert set(df.columns) == cols_before, "compute_rsi must not mutate the input"


class TestComputeEMA:

    def test_adds_both_ema_columns(self):
        from src.indicators.indicators import compute_ema
        df = make_ohlcv()
        out = compute_ema(df.copy(), fast=9, slow=21)
        assert "ema_9" in out.columns
        assert "ema_21" in out.columns

    def test_fast_ema_reacts_faster(self):
        """Fast EMA should have higher variance than slow EMA."""
        from src.indicators.indicators import compute_ema
        df = make_ohlcv(200)
        out = compute_ema(df.copy(), fast=5, slow=50)
        fast_std = out["ema_5"].dropna().std()
        slow_std = out["ema_50"].dropna().std()
        assert fast_std > slow_std, "Fast EMA must have higher variance than slow EMA"

    def test_ema_tracks_close(self):
        """EMA values should be broadly in the same range as close."""
        from src.indicators.indicators import compute_ema
        df = make_ohlcv()
        out = compute_ema(df.copy(), fast=9, slow=21)
        close_mean = df["close"].mean()
        ema_mean   = out["ema_9"].dropna().mean()
        assert abs(close_mean - ema_mean) < close_mean * 0.2, "EMA should track close price"


class TestComputeMACD:

    def test_adds_three_macd_columns(self):
        from src.indicators.indicators import compute_macd
        df = make_ohlcv()
        out = compute_macd(df.copy())
        assert "macd_line"   in out.columns
        assert "macd_signal" in out.columns
        assert "macd_hist"   in out.columns

    def test_histogram_equals_line_minus_signal(self):
        """macd_hist = macd_line - macd_signal (within float tolerance)."""
        from src.indicators.indicators import compute_macd
        df = make_ohlcv(100)
        out = compute_macd(df.copy())
        valid = out.dropna(subset=["macd_line", "macd_signal", "macd_hist"])
        diff  = (valid["macd_line"] - valid["macd_signal"] - valid["macd_hist"]).abs()
        assert (diff < 1e-6).all(), "MACD histogram must equal line − signal"

    def test_returns_df_when_too_few_rows(self):
        """MACD needs 26+ rows; fewer should return unchanged df, not raise."""
        from src.indicators.indicators import compute_macd
        df = make_ohlcv(10)
        out = compute_macd(df.copy())
        # Should either add the columns (all NaN) or return unchanged — not raise
        assert isinstance(out, pd.DataFrame)


class TestComputeBollinger:

    def test_adds_five_bb_columns(self):
        from src.indicators.indicators import compute_bollinger
        df = make_ohlcv()
        out = compute_bollinger(df.copy(), period=20, std_dev=2.0)
        for col in ["bb_upper", "bb_mid", "bb_lower", "bb_bandwidth", "bb_percent"]:
            assert col in out.columns, f"Missing Bollinger column: {col}"

    def test_upper_above_mid_above_lower(self):
        """upper ≥ mid ≥ lower for all valid rows."""
        from src.indicators.indicators import compute_bollinger
        df = make_ohlcv()
        out = compute_bollinger(df.copy(), period=20, std_dev=2.0)
        valid = out.dropna(subset=["bb_upper", "bb_mid", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_mid"]).all()
        assert (valid["bb_mid"]   >= valid["bb_lower"]).all()

    def test_bandwidth_is_positive(self):
        from src.indicators.indicators import compute_bollinger
        df = make_ohlcv()
        out = compute_bollinger(df.copy())
        valid = out["bb_bandwidth"].dropna()
        assert (valid >= 0).all(), "Bollinger bandwidth must be non-negative"


class TestComputeATR:

    def test_adds_atr_column(self):
        from src.indicators.indicators import compute_atr
        df = make_ohlcv()
        out = compute_atr(df.copy(), period=14)
        assert "atr_14" in out.columns

    def test_atr_positive(self):
        from src.indicators.indicators import compute_atr
        df = make_ohlcv()
        out = compute_atr(df.copy(), period=14)
        valid = out["atr_14"].dropna()
        assert (valid > 0).all(), "ATR must be positive"

    def test_atr_custom_period(self):
        from src.indicators.indicators import compute_atr
        df = make_ohlcv()
        out = compute_atr(df.copy(), period=7)
        assert "atr_7" in out.columns


class TestComputeVWAP:

    def test_adds_vwap_column(self):
        from src.indicators.indicators import compute_vwap
        df = make_ohlcv()
        out = compute_vwap(df.copy())
        assert "vwap" in out.columns

    def test_vwap_values_populated(self):
        from src.indicators.indicators import compute_vwap
        df = make_ohlcv()
        out = compute_vwap(df.copy())
        assert out["vwap"].notna().any(), "VWAP should have non-NaN values"

    def test_vwap_skipped_when_volume_missing(self):
        """If volume is all NaN, VWAP should log a warning and return df unchanged."""
        from src.indicators.indicators import compute_vwap
        df = make_ohlcv()
        df["volume"] = np.nan
        out = compute_vwap(df.copy())
        # Either skipped (all NaN) or not added at all — either is acceptable
        if "vwap" in out.columns:
            assert out["vwap"].isna().all(), "VWAP must be NaN when volume is NaN"

    def test_vwap_broadly_in_price_range(self):
        """VWAP should be between the day's low and high."""
        from src.indicators.indicators import compute_vwap
        df = make_ohlcv()
        out = compute_vwap(df.copy())
        valid = out.dropna(subset=["vwap"])
        price_min = df["low"].min()
        price_max = df["high"].max()
        assert valid["vwap"].between(price_min * 0.8, price_max * 1.2).all()


# ── Master compute() function tests ───────────────────────────────────────────

class TestComputeMaster:

    def test_single_indicator(self):
        from src.indicators.indicators import compute
        df = make_ohlcv()
        out = compute(df, ["RSI"])
        assert "rsi_14" in out.columns

    def test_multiple_indicators(self):
        from src.indicators.indicators import compute
        df = make_ohlcv()
        out = compute(df, ["RSI", "EMA", "ATR"])
        assert "rsi_14"  in out.columns
        assert "ema_9"   in out.columns
        assert "atr_14"  in out.columns

    def test_all_keyword(self):
        from src.indicators.indicators import compute
        df = make_ohlcv()
        out = compute(df, ["ALL"])
        for expected in ["rsi_14", "ema_9", "ema_21", "macd_line",
                         "bb_upper", "atr_14"]:
            assert expected in out.columns, f"ALL should compute {expected}"

    def test_unknown_indicator_raises(self):
        from src.indicators.indicators import compute
        df = make_ohlcv()
        with pytest.raises(ValueError, match="Unknown indicator"):
            compute(df, ["FOOBAR"])

    def test_case_insensitive(self):
        """Indicator names should be case-insensitive."""
        from src.indicators.indicators import compute
        df = make_ohlcv()
        out = compute(df, ["rsi", "ema"])
        assert "rsi_14" in out.columns
        assert "ema_9"  in out.columns

    def test_input_not_mutated(self):
        from src.indicators.indicators import compute
        df = make_ohlcv()
        cols_before = set(df.columns)
        compute(df, ["RSI", "EMA"])
        assert set(df.columns) == cols_before, "compute() must not mutate the input"

    def test_config_override_period(self):
        """period_override should change the primary period for RSI."""
        from src.indicators.indicators import compute
        df = make_ohlcv()
        out = compute(df, ["RSI"], period_override=7)
        assert "rsi_7"  in out.columns
        assert "rsi_14" not in out.columns

    def test_config_driven_params(self):
        """Params from config dict should override defaults."""
        from src.indicators.indicators import compute
        df = make_ohlcv()
        config = {"indicators": {"rsi": {"period": 21}}}
        out = compute(df, ["RSI"], config=config)
        assert "rsi_21" in out.columns

    def test_output_is_dataframe(self):
        from src.indicators.indicators import compute
        df = make_ohlcv()
        out = compute(df, ["RSI", "MACD", "BB"])
        assert isinstance(out, pd.DataFrame)

    def test_output_same_length_as_input(self):
        """compute() must not drop rows."""
        from src.indicators.indicators import compute
        df = make_ohlcv(100)
        out = compute(df, ["RSI", "EMA", "MACD", "BB", "ATR"])
        assert len(out) == len(df)


# ── Registry tests ─────────────────────────────────────────────────────────────

class TestRegistry:

    def test_registry_has_six_indicators(self):
        from src.indicators.indicators import REGISTRY
        assert len(REGISTRY) == 6

    def test_available_indicators_sorted(self):
        from src.indicators.indicators import available_indicators
        names = available_indicators()
        assert names == sorted(names)

    def test_all_registry_entries_have_fn(self):
        from src.indicators.indicators import REGISTRY
        for name, entry in REGISTRY.items():
            assert callable(entry["fn"]), f"REGISTRY[{name}]['fn'] must be callable"
