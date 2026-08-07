"""
tests/test_data.py
Unit tests for Stage 1: Data Layer

Run with:
    python -m pytest tests/test_data.py -v
    python -m pytest tests/test_data.py -v -m "not network"   # skip live calls

Tests are split into:
  - Schema validation   : DataFrame shape and types (offline, mock-based)
  - YFinanceClient      : unsupported timeframe, mocked happy path
  - DataManager cache   : write → read round-trip, force_fetch, clear_cache (all mocked)
  - PolygonClient       : auth errors, schema (_to_dataframe) — no live call needed
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

# Helpers 

EXPECTED_COLUMNS = {"open", "high", "low", "close", "volume", "vwap"}


def assert_ohlcv_schema(df: pd.DataFrame) -> None:
    """Shared schema assertions used across all tests."""
    assert isinstance(df, pd.DataFrame), "Result must be a DataFrame"
    assert not df.empty, "DataFrame must not be empty"
    assert df.index.name == "timestamp",  "Index must be named 'timestamp'"
    assert pd.api.types.is_datetime64_any_dtype(df.index), "Index must be datetime"
    assert df.index.tz is not None, "Index must be timezone-aware (UTC)"
    missing = EXPECTED_COLUMNS - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    for col in ["open", "high", "low", "close"]:
        assert pd.api.types.is_numeric_dtype(df[col]), f"'{col}' must be numeric"
    assert df.index.is_monotonic_increasing, "Index must be sorted ascending"


def _mock_ohlcv(ticker: str = "AAPL", n: int = 10) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame that satisfies the schema for mocking."""
    idx = pd.date_range("2023-01-03", periods=n, freq="B", tz="UTC", name="timestamp")
    df = pd.DataFrame({
        "open":   np.linspace(130, 140, n),
        "high":   np.linspace(132, 142, n),
        "low":    np.linspace(128, 138, n),
        "close":  np.linspace(131, 141, n),
        "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
        "vwap":   np.full(n, np.nan),
    }, index=idx)
    df.attrs["ticker"] = ticker
    df.attrs["source"] = "yfinance"
    return df


# YFinanceClient Tests

class TestYFinanceClient:

    def test_unsupported_timeframe_raises(self):
        from src.data.yfinance_client import YFinanceClient
        client = YFinanceClient()
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            client.fetch_ohlcv("AAPL", "2023-01-01", "2023-06-01", timeframe="4Hour")

    def test_fetch_daily_ohlcv_schema(self):
        """Mocked fetch — verifies schema compliance without network."""
        from src.data.yfinance_client import YFinanceClient

        mock_raw = pd.DataFrame({
            "Open":   [130.0, 131.0, 132.0],
            "High":   [133.0, 134.0, 135.0],
            "Low":    [129.0, 130.0, 131.0],
            "Close":  [131.0, 132.0, 133.0],
            "Volume": [5_000_000, 6_000_000, 7_000_000],
        }, index=pd.date_range("2023-01-03", periods=3, freq="B", tz="UTC"))

        with patch("yfinance.download", return_value=mock_raw):
            client = YFinanceClient()
            df = client.fetch_ohlcv("AAPL", "2023-01-01", "2023-06-01", timeframe="1Day")

        assert_ohlcv_schema(df)

    def test_fetch_returns_correct_ticker_attr(self):
        """attrs should carry the ticker and source after fetch."""
        from src.data.yfinance_client import YFinanceClient

        mock_raw = pd.DataFrame({
            "Open":   [250.0], "High": [255.0], "Low": [248.0],
            "Close":  [252.0], "Volume": [3_000_000],
        }, index=pd.date_range("2023-06-01", periods=1, tz="UTC"))

        with patch("yfinance.download", return_value=mock_raw):
            client = YFinanceClient()
            df = client.fetch_ohlcv("MSFT", "2023-06-01", "2023-09-01")

        assert df.attrs.get("ticker") == "MSFT"
        assert df.attrs.get("source") == "yfinance"

    def test_fetch_empty_for_bad_ticker(self):
        """An empty yfinance result should return an empty DataFrame, not raise."""
        from src.data.yfinance_client import YFinanceClient

        with patch("yfinance.download", return_value=pd.DataFrame()):
            client = YFinanceClient()
            df = client.fetch_ohlcv("ZZZZNOTREAL", "2023-01-01", "2023-06-01")

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_vwap_column_present_but_nan(self):
        """yfinance doesn't supply VWAP — the column must exist but be NaN."""
        from src.data.yfinance_client import YFinanceClient

        mock_raw = pd.DataFrame({
            "Open":   [100.0], "High": [105.0], "Low": [99.0],
            "Close":  [103.0], "Volume": [1_000_000],
        }, index=pd.date_range("2023-01-03", periods=1, tz="UTC"))

        with patch("yfinance.download", return_value=mock_raw):
            client = YFinanceClient()
            df = client.fetch_ohlcv("SPY", "2023-01-01", "2023-04-01")

        assert "vwap" in df.columns
        assert df["vwap"].isna().all(), "vwap should be all NaN for yfinance"

    def test_multiindex_columns_flattened(self):
        """yfinance sometimes returns MultiIndex columns — must be flattened."""
        from src.data.yfinance_client import YFinanceClient

        idx = pd.date_range("2023-01-03", periods=2, tz="UTC")
        cols = pd.MultiIndex.from_tuples([
            ("Open","AAPL"), ("High","AAPL"), ("Low","AAPL"),
            ("Close","AAPL"), ("Volume","AAPL"),
        ])
        mock_raw = pd.DataFrame(
            [[130.0, 133.0, 129.0, 131.0, 5_000_000],
             [131.0, 134.0, 130.0, 132.0, 6_000_000]],
            index=idx, columns=cols,
        )

        with patch("yfinance.download", return_value=mock_raw):
            client = YFinanceClient()
            df = client.fetch_ohlcv("AAPL", "2023-01-01", "2023-04-01")

        assert not isinstance(df.columns, pd.MultiIndex), "Columns must be flat"
        assert_ohlcv_schema(df)


# DataManager Tests — fully mocked, no network

class TestDataManager:

    @pytest.fixture
    def tmp_dm(self, tmp_path):
        """DataManager with a temp cache dir — isolated per test."""
        from src.data.data_manager import DataManager
        return DataManager(cache_dir=tmp_path / "raw")

    @pytest.fixture
    def mock_fetch(self):
        """Patch _fetch to return a deterministic DataFrame without network calls."""
        def _fake_fetch(ticker, from_date, to_date, timeframe, source):
            return _mock_ohlcv(ticker=ticker, n=10)

        with patch("src.data.data_manager.DataManager._fetch", side_effect=_fake_fetch) as m:
            yield m

    # Cache behaviour 

    def test_cache_miss_then_hit(self, tmp_dm, mock_fetch):
        """First call fetches and caches; second call loads from CSV cache."""
        df1 = tmp_dm.get_ohlcv("AAPL", "2023-01-01", "2023-03-31", source="yfinance")
        assert not df1.empty
        assert len(tmp_dm.list_cache()) == 1
        assert mock_fetch.call_count == 1

        # Second call — cache hit, _fetch should NOT be called again
        df2 = tmp_dm.get_ohlcv("AAPL", "2023-01-01", "2023-03-31", source="yfinance")
        assert mock_fetch.call_count == 1, "_fetch called again on cache hit — bug"
        assert len(df1) == len(df2)

    def test_force_fetch_bypasses_cache(self, tmp_dm, mock_fetch):
        """force_fetch=True should re-fetch even when cache file exists."""
        tmp_dm.get_ohlcv("AAPL", "2023-01-01", "2023-03-31", source="yfinance")
        tmp_dm.get_ohlcv("AAPL", "2023-01-01", "2023-03-31",
                         source="yfinance", force_fetch=True)
        assert mock_fetch.call_count == 2, "force_fetch must call _fetch twice"
        assert len(tmp_dm.list_cache()) == 1  # file overwritten, not duplicated

    def test_invalid_source_raises(self, tmp_dm):
        with pytest.raises(ValueError, match="Unknown source"):
            tmp_dm.get_ohlcv("AAPL", "2023-01-01", "2023-03-31", source="bloomberg")

    def test_clear_cache_specific_ticker(self, tmp_dm, mock_fetch):
        """clear_cache(ticker) removes only that ticker's files."""
        tmp_dm.get_ohlcv("AAPL", "2023-01-01", "2023-03-31", source="yfinance")
        tmp_dm.get_ohlcv("MSFT", "2023-01-01", "2023-03-31", source="yfinance")
        assert len(tmp_dm.list_cache()) == 2

        deleted = tmp_dm.clear_cache(ticker="AAPL")
        assert deleted == 1
        remaining = tmp_dm.list_cache()
        assert len(remaining) == 1
        assert "MSFT" in remaining[0]

    def test_clear_cache_all(self, tmp_dm, mock_fetch):
        """clear_cache() with no ticker removes everything."""
        tmp_dm.get_ohlcv("AAPL", "2023-01-01", "2023-03-31", source="yfinance")
        tmp_dm.get_ohlcv("MSFT", "2023-01-01", "2023-03-31", source="yfinance")
        deleted = tmp_dm.clear_cache()
        assert deleted == 2
        assert tmp_dm.list_cache() == []

    def test_cache_roundtrip_preserves_schema(self, tmp_dm, mock_fetch):
        """Data loaded from CSV cache must pass the same schema checks as fresh data."""
        tmp_dm.get_ohlcv("SPY", "2023-01-01", "2023-06-01", source="yfinance")
        # Re-load from cache (mock won't be called this time)
        df = tmp_dm.get_ohlcv("SPY", "2023-01-01", "2023-06-01", source="yfinance")
        assert_ohlcv_schema(df)

    def test_cache_filename_is_deterministic(self, tmp_dm, mock_fetch):
        """Same params should always produce the same cache file name."""
        tmp_dm.get_ohlcv("AAPL", "2023-01-01", "2023-06-01", source="yfinance")
        files = tmp_dm.list_cache()
        assert len(files) == 1
        assert files[0] == "AAPL_2023-01-01_2023-06-01_1Day_yfinance.csv"


# PolygonClient Tests — auth/error handling + schema (_to_dataframe)

class TestPolygonClientErrors:

    def test_missing_api_key_raises(self, monkeypatch):
        """PolygonClient without POLYGON_API_KEY should raise EnvironmentError."""
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        from src.data import polygon_client as pc
        import importlib
        importlib.reload(pc)
        with pytest.raises(EnvironmentError, match="POLYGON_API_KEY"):
            pc.PolygonClient(api_key=None)

    def test_unsupported_timeframe_raises(self):
        from src.data.polygon_client import PolygonClient
        client = PolygonClient(api_key="dummy_key_for_test")
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            client.fetch_ohlcv("AAPL", "2023-01-01", "2023-06-01", timeframe="Weekly")

    def test_auth_error_403_raises_permission_error(self):
        """A 403 from Polygon must raise PermissionError — no retry."""
        from src.data.polygon_client import PolygonClient

        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.json.return_value = {"message": "Forbidden"}

        with patch("requests.Session.get", return_value=mock_resp):
            client = PolygonClient(api_key="fake_key")
            with pytest.raises(PermissionError, match="access denied"):
                client.fetch_ohlcv("AAPL", "2023-01-01", "2023-06-01")

    def test_auth_error_401_raises_permission_error(self):
        """A 401 from Polygon must also raise PermissionError immediately."""
        from src.data.polygon_client import PolygonClient

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"message": "Unauthorized"}

        with patch("requests.Session.get", return_value=mock_resp):
            client = PolygonClient(api_key="fake_key")
            with pytest.raises(PermissionError):
                client.fetch_ohlcv("AAPL", "2023-01-01", "2023-06-01")

    def test_to_dataframe_output_schema(self):
        """_to_dataframe with valid Polygon result list must pass schema checks."""
        from src.data.polygon_client import PolygonClient
        client = PolygonClient(api_key="dummy_key")
        sample = [
            {"t": 1672531200000, "o": 130.0, "h": 133.0, "l": 129.5,
             "c": 131.0, "v": 5_000_000, "vw": 130.5},
            {"t": 1672617600000, "o": 131.0, "h": 135.0, "l": 130.0,
             "c": 134.0, "v": 6_000_000, "vw": 132.1},
        ]
        df = client._to_dataframe(sample, "AAPL")
        assert_ohlcv_schema(df)
        assert df["vwap"].notna().all(), "Polygon VWAP must be populated"

    def test_to_dataframe_sorted_ascending(self):
        """_to_dataframe must sort bars by timestamp even if input is unordered."""
        from src.data.polygon_client import PolygonClient
        client = PolygonClient(api_key="dummy_key")
        unordered = [
            {"t": 1672617600000, "o": 131.0, "h": 135.0, "l": 130.0,
             "c": 134.0, "v": 6_000_000, "vw": 132.1},
            {"t": 1672531200000, "o": 130.0, "h": 133.0, "l": 129.5,
             "c": 131.0, "v": 5_000_000, "vw": 130.5},
        ]
        df = client._to_dataframe(unordered, "AAPL")
        assert df.index.is_monotonic_increasing, "Must be sorted ascending"
