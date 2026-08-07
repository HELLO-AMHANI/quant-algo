"""
src/indicators/indicators.py
pandas-ta indicator wrappers for the quant-algo pipeline.

Design:
  - Every function takes a DataFrame (with open/high/low/close/volume)
    and returns the SAME DataFrame with new columns appended.
  - Column names are ours — clean, lowercase, unambiguous — not pandas-ta internals.
  - Parameters come from settings.yaml via the config dict, with sensible defaults.
  - A REGISTRY maps indicator names (uppercase strings) to compute functions
    so the CLI and strategies can dispatch by name dynamically.

Supported indicators:
  RSI    — Relative Strength Index
  EMA    — Exponential Moving Average (fast + slow)
  MACD   — Moving Average Convergence Divergence
  BB     — Bollinger Bands
  ATR    — Average True Range
  VWAP   — Volume Weighted Average Price (most useful on intraday; included for completeness)
"""

import logging
from typing import Optional

import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Individual Indicator Functions
# Each returns the DataFrame in-place with new column(s) appended.
# ─────────────────────────────────────────────────────────────────────────────

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index.

    Adds column: rsi_{period}   e.g. rsi_14
    Range: 0–100. Overbought > 70, Oversold < 30.
    """
    col  = f"rsi_{period}"
    raw  = ta.rsi(df["close"], length=period)
    df[col] = raw
    logger.debug(f"RSI({period}) → '{col}' | last={df[col].iloc[-1]:.2f}")
    return df


def compute_ema(
    df: pd.DataFrame,
    fast: int = 9,
    slow: int = 21,
) -> pd.DataFrame:
    """
    Exponential Moving Average — fast and slow pair.

    Adds columns: ema_{fast}, ema_{slow}   e.g. ema_9, ema_21
    Crossover signal: ema_fast > ema_slow → bullish.
    """
    df[f"ema_{fast}"] = ta.ema(df["close"], length=fast)
    df[f"ema_{slow}"] = ta.ema(df["close"], length=slow)
    logger.debug(
        f"EMA({fast},{slow}) | "
        f"ema_{fast}={df[f'ema_{fast}'].iloc[-1]:.2f} "
        f"ema_{slow}={df[f'ema_{slow}'].iloc[-1]:.2f}"
    )
    return df


def compute_macd(
    df: pd.DataFrame,
    fast: int   = 12,
    slow: int   = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence.

    Adds columns:
      macd_line   — MACD line (fast EMA - slow EMA)
      macd_signal — Signal line (EMA of MACD line)
      macd_hist   — Histogram (macd_line - macd_signal)
    """
    raw = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
    if raw is None or raw.empty:
        logger.warning("MACD returned no data — not enough rows?")
        return df

    # pandas-ta column names for this version: MACD_f_s_sig, MACDh_f_s_sig, MACDs_f_s_sig
    src_line   = f"MACD_{fast}_{slow}_{signal}"
    src_hist   = f"MACDh_{fast}_{slow}_{signal}"
    src_signal = f"MACDs_{fast}_{slow}_{signal}"

    df["macd_line"]   = raw[src_line]
    df["macd_signal"] = raw[src_signal]
    df["macd_hist"]   = raw[src_hist]

    logger.debug(
        f"MACD({fast},{slow},{signal}) | "
        f"line={df['macd_line'].iloc[-1]:.4f} "
        f"signal={df['macd_signal'].iloc[-1]:.4f} "
        f"hist={df['macd_hist'].iloc[-1]:.4f}"
    )
    return df


def compute_bollinger(
    df: pd.DataFrame,
    period: int    = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands.

    Adds columns:
      bb_upper      — Upper band
      bb_mid        — Middle band (SMA)
      bb_lower      — Lower band
      bb_bandwidth  — Band width (volatility proxy)
      bb_percent    — %B: where price sits within the bands (0–1, can exceed)
    """
    raw = ta.bbands(df["close"], length=period, std=std_dev)
    if raw is None or raw.empty:
        logger.warning("Bollinger Bands returned no data.")
        return df

    # pandas-ta produces: BBL_20_2.0_2.0, BBM_..., BBU_..., BBB_..., BBP_...
    prefix = f"BB"
    cols   = raw.columns.tolist()

    def _pick(label: str) -> Optional[pd.Series]:
        """Find column starting with label."""
        match = [c for c in cols if c.startswith(label)]
        return raw[match[0]] if match else None

    for dest, label in [
        ("bb_upper",     "BBU"),
        ("bb_mid",       "BBM"),
        ("bb_lower",     "BBL"),
        ("bb_bandwidth", "BBB"),
        ("bb_percent",   "BBP"),
    ]:
        series = _pick(label)
        if series is not None:
            df[dest] = series

    logger.debug(
        f"BB({period},{std_dev}) | "
        f"upper={df['bb_upper'].iloc[-1]:.2f} "
        f"mid={df['bb_mid'].iloc[-1]:.2f} "
        f"lower={df['bb_lower'].iloc[-1]:.2f}"
    )
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average True Range — volatility / stop-loss sizing.

    Adds column: atr_{period}   e.g. atr_14
    Higher ATR = higher volatility.
    """
    col = f"atr_{period}"
    raw = ta.atr(df["high"], df["low"], df["close"], length=period)
    df[col] = raw
    logger.debug(f"ATR({period}) → '{col}' | last={df[col].iloc[-1]:.2f}")
    return df


def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volume Weighted Average Price.

    Adds column: vwap
    Note: Most meaningful on intraday data. On daily bars, pandas-ta computes
    a cumulative VWAP which still serves as a useful dynamic support/resistance level.
    Requires 'volume' column — skipped if absent.
    """
    if "volume" not in df.columns or df["volume"].isna().all():
        logger.warning("VWAP skipped — volume column missing or all NaN.")
        return df

    raw = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    if raw is None:
        logger.warning("VWAP returned None.")
        return df

    df["vwap"] = raw
    logger.debug(f"VWAP | last={df['vwap'].iloc[-1]:.2f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Indicator Registry
# Maps uppercase indicator name → (function, param_section_in_settings_yaml)
# ─────────────────────────────────────────────────────────────────────────────

REGISTRY: dict[str, dict] = {
    "RSI": {
        "fn":      compute_rsi,
        "config":  "rsi",
        "desc":    "Relative Strength Index",
        "outputs": ["rsi_{period}"],
    },
    "EMA": {
        "fn":      compute_ema,
        "config":  "ema",
        "desc":    "Exponential Moving Average (fast + slow pair)",
        "outputs": ["ema_{fast}", "ema_{slow}"],
    },
    "MACD": {
        "fn":      compute_macd,
        "config":  "macd",
        "desc":    "MACD line, signal, histogram",
        "outputs": ["macd_line", "macd_signal", "macd_hist"],
    },
    "BB": {
        "fn":      compute_bollinger,
        "config":  "bollinger",
        "desc":    "Bollinger Bands (upper, mid, lower, bandwidth, %B)",
        "outputs": ["bb_upper", "bb_mid", "bb_lower", "bb_bandwidth", "bb_percent"],
    },
    "ATR": {
        "fn":      compute_atr,
        "config":  "atr",
        "desc":    "Average True Range",
        "outputs": ["atr_{period}"],
    },
    "VWAP": {
        "fn":      compute_vwap,
        "config":  None,
        "desc":    "Volume Weighted Average Price",
        "outputs": ["vwap"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Master Compute Function
# ─────────────────────────────────────────────────────────────────────────────

def compute(
    df: pd.DataFrame,
    indicator_names: list[str],
    config: Optional[dict] = None,
    period_override: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute one or more indicators and append their columns to df.

    Args:
        df:               OHLCV DataFrame (from DataManager).
        indicator_names:  List of indicator names, e.g. ["RSI", "EMA", "MACD"].
                          Use "ALL" as the single element to run everything.
        config:           Full settings.yaml dict. Reads from 'indicators' section.
                          If None, uses hardcoded defaults.
        period_override:  If set, overrides the primary 'period' param for RSI/EMA/ATR.
                          Useful for CLI --period flag.

    Returns:
        Same DataFrame with new indicator columns appended.

    Raises:
        ValueError: Unknown indicator name.
    """
    cfg = (config or {}).get("indicators", {})

    # Expand "ALL"
    if indicator_names == ["ALL"]:
        indicator_names = list(REGISTRY.keys())

    # Normalise to uppercase
    requested = [n.strip().upper() for n in indicator_names]

    unknown = [n for n in requested if n not in REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown indicator(s): {unknown}. "
            f"Available: {list(REGISTRY.keys())}"
        )

    df = df.copy()  # don't mutate the caller's DataFrame

    for name in requested:
        fn_cfg   = cfg.get(REGISTRY[name]["config"] or "", {})
        logger.info(f"Computing {name}...")

        try:
            if name == "RSI":
                period = period_override or fn_cfg.get("period", 14)
                df = compute_rsi(df, period=period)

            elif name == "EMA":
                fast = period_override or fn_cfg.get("fast", 9)
                slow = fn_cfg.get("slow", 21)
                df = compute_ema(df, fast=fast, slow=slow)

            elif name == "MACD":
                df = compute_macd(
                    df,
                    fast=fn_cfg.get("fast", 12),
                    slow=fn_cfg.get("slow", 26),
                    signal=fn_cfg.get("signal", 9),
                )

            elif name == "BB":
                df = compute_bollinger(
                    df,
                    period=period_override or fn_cfg.get("period", 20),
                    std_dev=fn_cfg.get("std_dev", 2.0),
                )

            elif name == "ATR":
                period = period_override or fn_cfg.get("period", 14)
                df = compute_atr(df, period=period)

            elif name == "VWAP":
                df = compute_vwap(df)

        except Exception as e:
            logger.error(f"{name} failed: {e}")
            # Non-fatal — skip this indicator and continue
            continue

    return df


def available_indicators() -> list[str]:
    """Return sorted list of supported indicator names."""
    return sorted(REGISTRY.keys())
