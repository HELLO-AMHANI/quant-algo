"""
tests/test_backtest.py
Unit tests for Stage 4: Backtesting Engine

Run with:
    python -m pytest tests/test_backtest.py -v

All tests are offline — no network, no API keys.
Synthetic OHLCV data is generated inline and signals are
injected directly to avoid any network dependency.
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch


# ── Shared Synthetic Data ─────────────────────────────────────────────────────

def make_ohlcv(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Random-walk OHLCV — enough bars for all indicators to warm up."""
    rng    = np.random.default_rng(seed)
    close  = 100 + np.cumsum(rng.normal(0, 1, n))
    spread = rng.uniform(0.5, 2.0, n)
    idx    = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC", name="timestamp")
    df = pd.DataFrame({
        "open":   close + rng.normal(0, 0.3, n),
        "high":   close + spread,
        "low":    close - spread,
        "close":  close,
        "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        "vwap":   np.nan,
    }, index=idx)
    df.attrs["ticker"] = "TEST"
    df.attrs["source"] = "synthetic"
    return df


def make_mock_stats(n_trades: int = 5) -> pd.Series:
    """Build a minimal backtesting.py-style Stats Series for unit testing."""
    return pd.Series({
        "Return [%]":              12.5,
        "Buy & Hold Return [%]":    8.2,
        "Return (Ann.) [%]":       10.1,
        "Volatility (Ann.) [%]":   18.3,
        "Sharpe Ratio":             1.23,
        "Sortino Ratio":            1.85,
        "Calmar Ratio":             0.92,
        "Max. Drawdown [%]":       -8.5,
        "Avg. Drawdown [%]":       -3.2,
        "Max. Drawdown Duration":   pd.Timedelta("30 days"),
        "Avg. Drawdown Duration":   pd.Timedelta("10 days"),
        "Win Rate [%]":            60.0,
        "Best Trade [%]":           6.2,
        "Worst Trade [%]":         -2.1,
        "Avg. Trade [%]":           1.8,
        "Max. Trade Duration":      pd.Timedelta("15 days"),
        "Avg. Trade Duration":      pd.Timedelta("7 days"),
        "Profit Factor":            1.8,
        "Expectancy [%]":           1.2,
        "SQN":                      2.1,
        "Kelly Criterion":          0.25,
        "# Trades":                 n_trades,
        "Exposure Time [%]":       35.2,
        "Equity Final [$]":     11_250.0,
        "Equity Peak [$]":      11_500.0,
        "Commissions [$]":         12.5,
        "_strategy":             None,
        "_equity_curve":         None,
        "_trades":               None,
    })


# ─────────────────────────────────────────────────────────────────────────────
# prepare_df Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPrepareDF:

    def test_renames_ohlcv_columns(self):
        from src.backtest.runner import prepare_df
        df  = make_ohlcv()
        out = prepare_df(df)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in out.columns, f"Missing titled column: {col}"

    def test_lowercase_columns_removed(self):
        from src.backtest.runner import prepare_df
        df  = make_ohlcv()
        out = prepare_df(df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col not in out.columns

    def test_strips_timezone(self):
        from src.backtest.runner import prepare_df
        df  = make_ohlcv()
        assert df.index.tz is not None, "Fixture should have UTC tz"
        out = prepare_df(df)
        assert out.index.tz is None, "Output index must be tz-naive"

    def test_strips_extra_columns(self):
        """Signal and indicator columns must NOT appear in the backtest df."""
        from src.backtest.runner import prepare_df
        df = make_ohlcv()
        df["signal"]   = 0
        df["ema_9"]    = df["close"].rolling(9).mean()
        df["strategy"] = "test"
        out = prepare_df(df)
        for extra in ["signal", "ema_9", "strategy", "vwap"]:
            assert extra not in out.columns, f"Extra column leaked: {extra}"

    def test_does_not_mutate_input(self):
        from src.backtest.runner import prepare_df
        df   = make_ohlcv()
        cols = set(df.columns)
        prepare_df(df)
        assert set(df.columns) == cols

    def test_output_has_same_row_count(self):
        from src.backtest.runner import prepare_df
        df  = make_ohlcv(80)
        out = prepare_df(df)
        assert len(out) == len(df)


# ─────────────────────────────────────────────────────────────────────────────
# extract_metrics Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractMetrics:

    def test_all_expected_keys_present(self):
        from src.backtest.runner import extract_metrics
        stats  = make_mock_stats()
        m      = extract_metrics(stats)
        expected_keys = [
            "total_return_pct", "buy_hold_return_pct", "annualised_return_pct",
            "sharpe_ratio", "sortino_ratio", "max_drawdown_pct",
            "win_rate_pct", "profit_factor", "total_trades",
            "exposure_time_pct", "equity_final", "equity_peak",
        ]
        for key in expected_keys:
            assert key in m, f"Missing metrics key: {key}"

    def test_values_match_stats(self):
        from src.backtest.runner import extract_metrics
        stats = make_mock_stats(n_trades=5)
        m     = extract_metrics(stats)
        assert m["total_return_pct"]   == 12.5
        assert m["sharpe_ratio"]       == 1.23
        assert m["win_rate_pct"]       == 60.0
        assert m["total_trades"]       == 5
        assert m["equity_final"]       == 11_250.0

    def test_nan_values_become_zero(self):
        """NaN metrics (e.g. no trades) should default to 0, not propagate NaN."""
        from src.backtest.runner import extract_metrics
        nan_stats = make_mock_stats()
        nan_stats["Sharpe Ratio"]   = float("nan")
        nan_stats["Win Rate [%]"]   = float("nan")
        nan_stats["Profit Factor"]  = float("nan")
        m = extract_metrics(nan_stats)
        assert m["sharpe_ratio"]  == 0.0
        assert m["win_rate_pct"]  == 0.0
        assert m["profit_factor"] == 0.0

    def test_missing_keys_default_gracefully(self):
        """A stats Series missing optional keys should not raise."""
        from src.backtest.runner import extract_metrics
        minimal = pd.Series({
            "Return [%]":    5.0,
            "# Trades":      2,
            "Equity Final [$]": 10_500.0,
        })
        m = extract_metrics(minimal)   # must not raise
        assert m["total_return_pct"] == 5.0
        assert m["sharpe_ratio"]     == 0.0   # missing → default

    def test_all_values_are_numeric(self):
        from src.backtest.runner import extract_metrics
        m = extract_metrics(make_mock_stats())
        for key, val in m.items():
            assert isinstance(val, (int, float)), \
                f"Metric '{key}' is not numeric: {type(val)}"

    def test_total_trades_is_int(self):
        from src.backtest.runner import extract_metrics
        m = extract_metrics(make_mock_stats(n_trades=7))
        assert isinstance(m["total_trades"], int)
        assert m["total_trades"] == 7


# ─────────────────────────────────────────────────────────────────────────────
# _make_signal_strategy Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeSignalStrategy:

    def test_returns_a_class(self):
        from src.backtest.runner import _make_signal_strategy
        cls = _make_signal_strategy(np.zeros(50))
        assert isinstance(cls, type)

    def test_each_call_returns_distinct_class(self):
        """Two calls with different arrays must return different classes."""
        from src.backtest.runner import _make_signal_strategy
        cls_a = _make_signal_strategy(np.ones(50))
        cls_b = _make_signal_strategy(np.zeros(50))
        assert cls_a is not cls_b

    def test_strategy_is_backtesting_subclass(self):
        from src.backtest.runner import _make_signal_strategy
        from backtesting import Strategy
        cls = _make_signal_strategy(np.zeros(50))
        assert issubclass(cls, Strategy)


# ─────────────────────────────────────────────────────────────────────────────
# run_backtest End-to-End Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunBacktest:

    @pytest.fixture
    def df(self):
        return make_ohlcv(120)

    def test_returns_expected_keys(self, df):
        from src.backtest.runner import run_backtest
        result = run_backtest(df, "ema_cross", save=False)
        for key in ["metrics", "strategy", "ticker", "stats"]:
            assert key in result, f"Missing result key: {key}"

    def test_metrics_dict_populated(self, df):
        from src.backtest.runner import run_backtest
        result = run_backtest(df, "ema_cross", save=False)
        m      = result["metrics"]
        assert isinstance(m["total_return_pct"], float)
        assert isinstance(m["total_trades"],     int)
        assert isinstance(m["sharpe_ratio"],     float)

    def test_ticker_preserved_in_result(self, df):
        from src.backtest.runner import run_backtest
        df.attrs["ticker"] = "TEST"
        result = run_backtest(df, "ema_cross", save=False)
        assert result["ticker"] == "TEST"

    def test_strategy_name_preserved(self, df):
        from src.backtest.runner import run_backtest
        result = run_backtest(df, "rsi_mean_reversion", save=False)
        assert result["strategy"] == "rsi_mean_reversion"

    def test_unknown_strategy_raises(self, df):
        from src.backtest.runner import run_backtest
        with pytest.raises(ValueError, match="Unknown strategy"):
            run_backtest(df, "nonexistent_strategy", save=False)

    def test_exposure_time_between_0_and_100(self, df):
        from src.backtest.runner import run_backtest
        m = run_backtest(df, "ema_cross", save=False)["metrics"]
        assert 0.0 <= m["exposure_time_pct"] <= 100.0

    def test_equity_final_positive(self, df):
        from src.backtest.runner import run_backtest
        m = run_backtest(df, "ema_cross", save=False)["metrics"]
        assert m["equity_final"] > 0

    def test_cash_parameter_reflected_in_equity(self, df):
        """Starting with more cash should produce proportionally higher final equity."""
        from src.backtest.runner import run_backtest
        r_10k  = run_backtest(df.copy(), "ema_cross", cash=10_000, save=False)
        r_50k  = run_backtest(df.copy(), "ema_cross", cash=50_000, save=False)
        # $50k start should produce higher absolute final equity than $10k
        assert r_50k["metrics"]["equity_final"] > r_10k["metrics"]["equity_final"]

    def test_no_save_skips_file_creation(self, df, tmp_path):
        from src.backtest.runner import run_backtest
        result = run_backtest(df, "ema_cross", results_dir=str(tmp_path), save=False)
        assert result["json_path"] is None
        assert result["csv_path"]  is None
        assert len(list(tmp_path.iterdir())) == 0

    def test_both_strategies_run_without_error(self, df):
        from src.backtest.runner import run_backtest
        for strategy in ["ema_cross", "rsi_mean_reversion"]:
            result = run_backtest(df.copy(), strategy, save=False)
            assert "metrics" in result


# ─────────────────────────────────────────────────────────────────────────────
# save_results Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveResults:

    def test_creates_json_and_csv(self, tmp_path):
        from src.backtest.runner import save_results
        df      = make_ohlcv()
        metrics = {"total_return_pct": 5.0, "sharpe_ratio": 1.1, "total_trades": 3}
        json_p, csv_p = save_results(metrics, "ema_cross", "AAPL", df, str(tmp_path))
        assert json_p.exists(), "JSON file must be created"
        assert csv_p.exists(),  "CSV file must be created"

    def test_json_is_valid_and_matches_metrics(self, tmp_path):
        from src.backtest.runner import save_results
        df      = make_ohlcv()
        metrics = {"total_return_pct": 7.5, "total_trades": 4}
        json_p, _ = save_results(metrics, "ema_cross", "AAPL", df, str(tmp_path))
        with open(json_p) as f:
            loaded = json.load(f)
        assert loaded["total_return_pct"] == 7.5
        assert loaded["total_trades"]     == 4

    def test_csv_contains_ohlcv_rows(self, tmp_path):
        from src.backtest.runner import save_results
        df  = make_ohlcv(50)
        _, csv_p = save_results({}, "ema_cross", "AAPL", df, str(tmp_path))
        loaded   = pd.read_csv(csv_p, index_col=0)
        assert len(loaded) == len(df)

    def test_creates_results_dir_if_missing(self, tmp_path):
        from src.backtest.runner import save_results
        new_dir = tmp_path / "deep" / "nested" / "results"
        assert not new_dir.exists()
        save_results({}, "ema_cross", "AAPL", make_ohlcv(), str(new_dir))
        assert new_dir.exists()

    def test_filename_contains_ticker_and_strategy(self, tmp_path):
        from src.backtest.runner import save_results
        json_p, csv_p = save_results({}, "ema_cross", "TSLA", make_ohlcv(), str(tmp_path))
        assert "TSLA"      in json_p.name
        assert "ema_cross" in json_p.name
        assert "TSLA"      in csv_p.name


# ─────────────────────────────────────────────────────────────────────────────
# compare_strategies Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCompareStrategies:

    @pytest.fixture
    def df(self):
        return make_ohlcv(120)

    def test_returns_all_strategy_keys(self, df):
        from src.backtest.runner import compare_strategies
        names  = ["ema_cross", "rsi_mean_reversion"]
        result = compare_strategies(df, names, save=False)
        for name in names:
            assert name in result

    def test_each_entry_has_metrics(self, df):
        from src.backtest.runner import compare_strategies
        result = compare_strategies(df, ["ema_cross", "rsi_mean_reversion"], save=False)
        for name, m in result.items():
            assert "error" not in m, f"Strategy {name} errored: {m}"
            assert "total_return_pct" in m

    def test_invalid_strategy_returns_error_entry(self, df):
        from src.backtest.runner import compare_strategies
        result = compare_strategies(df, ["ema_cross", "bad_strategy"], save=False)
        assert "error" in result["bad_strategy"]
        assert "total_return_pct" in result["ema_cross"]   # good one still ran

    def test_single_item_list_works(self, df):
        from src.backtest.runner import compare_strategies
        result = compare_strategies(df, ["ema_cross"], save=False)
        assert "ema_cross" in result
        assert "total_return_pct" in result["ema_cross"]
