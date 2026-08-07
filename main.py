#!/usr/bin/env python3
"""
quant-algo — CLI Entry Point
AMHANi Enterprise | Algorithmic Trading System

Usage:
    python main.py --help
    python main.py fetch --help
    python main.py indicators --help
    python main.py signals --help
    python main.py backtest --help
    python main.py trade --help
"""

import os
import sys
import logging
import yaml
from pathlib import Path
from dotenv import load_dotenv
import click
import pandas as pd

# ── Bootstrap ─────────────────────────────────────────────────────────────────

# Load .env from project root
load_dotenv()

# Config path
CONFIG_PATH = Path(__file__).parent / "config" / "settings.yaml"
CONFIG_EXAMPLE_PATH = Path(__file__).parent / "config" / "settings.example.yaml"


def load_config() -> dict:
    """Load settings.yaml. Falls back to example config with a warning."""
    path = CONFIG_PATH if CONFIG_PATH.exists() else CONFIG_EXAMPLE_PATH
    if not CONFIG_PATH.exists():
        click.echo(
            click.style(
                "[WARN] config/settings.yaml not found — using settings.example.yaml. "
                "Copy it and fill in your values.",
                fg="yellow",
            )
        )
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "quant-algo.log"),
        ],
    )


# ── CLI Root ──────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="0.1.0", prog_name="quant-algo")
@click.option("--log-level", default="INFO", help="Logging level (DEBUG/INFO/WARNING/ERROR)")
@click.pass_context
def cli(ctx: click.Context, log_level: str) -> None:
    """
    quant-algo — AMHANi Algorithmic Trading System

    A CLI-first quantitative trading pipeline:
    fetch → indicators → signals → backtest → trade

    Run any command with --help for usage details.
    """
    ctx.ensure_object(dict)
    setup_logging(log_level)
    ctx.obj["config"] = load_config()
    ctx.obj["logger"] = logging.getLogger("quant-algo")


# ── Stage 1: fetch ────────────────────────────────────────────────────────────

@cli.command()
@click.option("--ticker",      required=True,  help="Ticker symbol, e.g. AAPL")
@click.option("--from",        "date_from", required=True, help="Start date YYYY-MM-DD")
@click.option("--to",          "date_to",   required=True, help="End date   YYYY-MM-DD")
@click.option("--source",      default="polygon",
              help="Data source: polygon | yfinance",
              type=click.Choice(["polygon", "yfinance"], case_sensitive=False))
@click.option("--timeframe",   default=None,   help="Bar timeframe: 1Day | 1Hour | 5Min | 1Min")
@click.option("--force",       is_flag=True,   help="Skip cache and fetch fresh data")
@click.option("--preview",     is_flag=True,   help="Print first 5 rows after fetch")
@click.pass_context
def fetch(ctx: click.Context, ticker: str, date_from: str, date_to: str,
          source: str, timeframe: str, force: bool, preview: bool) -> None:
    """Fetch and cache OHLCV data for a ticker."""
    from src.data.data_manager import DataManager

    logger  = ctx.obj["logger"]
    config  = ctx.obj["config"]
    tf      = timeframe or config.get("data", {}).get("timeframe", "1Day")

    click.echo(f"\nFetching {ticker.upper()} | {date_from} → {date_to} | {source} | {tf}\n")

    try:
        dm = DataManager()
        df = dm.get_ohlcv(
            ticker=ticker,
            from_date=date_from,
            to_date=date_to,
            source=source,
            timeframe=tf,
            force_fetch=force,
        )
    except EnvironmentError as e:
        click.echo(click.style(f"[ERROR] {e}", fg="red"))
        raise SystemExit(1)
    except Exception as e:
        logger.exception(f"Fetch failed: {e}")
        click.echo(click.style(f"[ERROR] {e}", fg="red"))
        raise SystemExit(1)

    if df.empty:
        click.echo(click.style("No data returned. Check ticker and date range.", fg="yellow"))
        return

    # ── Summary ───────────────────────────────────────────────────────────────
    click.echo(click.style("✓ Fetch complete", fg="green"))
    click.echo(f"  Ticker    : {ticker.upper()}")
    click.echo(f"  Source    : {source}")
    click.echo(f"  Timeframe : {tf}")
    click.echo(f"  Rows      : {len(df):,}")
    click.echo(f"  Date range: {df.index[0].date()} → {df.index[-1].date()}")
    click.echo(f"  Columns   : {', '.join(df.columns.tolist())}")
    click.echo(f"  Close range: ${df['close'].min():.2f} – ${df['close'].max():.2f}")

    if preview:
        click.echo("\n── Preview (first 5 rows) ───────────────────────────────────")
        click.echo(df.head().to_string())


# ── Stage 2: indicators ───────────────────────────────────────────────────────

@cli.command()
@click.option("--ticker",     required=True, help="Ticker symbol, e.g. AAPL")
@click.option("--from",       "date_from", required=True,  help="Start date YYYY-MM-DD")
@click.option("--to",         "date_to",   required=True,  help="End date   YYYY-MM-DD")
@click.option("--source",     default="yfinance",
              help="Data source: polygon | yfinance",
              type=click.Choice(["polygon", "yfinance"], case_sensitive=False))
@click.option("--indicators", default="RSI,EMA",
              help="Comma-separated list or ALL. e.g. RSI,EMA,MACD,BB,ATR,VWAP")
@click.option("--period",     default=None, type=int,
              help="Override primary period for RSI / EMA / ATR / BB")
@click.option("--tail",       default=5,    type=int,
              help="Number of recent rows to display (default: 5)")
@click.pass_context
def indicators(ctx: click.Context, ticker: str, date_from: str, date_to: str,
               source: str, indicators: str, period: int, tail: int) -> None:
    """Compute indicators on OHLCV data and display recent values."""
    from src.data.data_manager    import DataManager
    from src.indicators.indicators import compute, available_indicators

    logger = ctx.obj["logger"]
    config = ctx.obj["config"]

    ind_list = (
        list(available_indicators())
        if indicators.upper() == "ALL"
        else [i.strip().upper() for i in indicators.split(",")]
    )

    click.echo(f"\nIndicators: {', '.join(ind_list)}  |  {ticker.upper()}  |  {date_from} → {date_to}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        dm = DataManager()
        df = dm.get_ohlcv(ticker, date_from, date_to, source=source)
    except Exception as e:
        click.echo(click.style(f"[ERROR] Data load failed: {e}", fg="red"))
        raise SystemExit(1)

    if df.empty:
        click.echo(click.style("No data. Run fetch first.", fg="yellow"))
        return

    # ── Compute ───────────────────────────────────────────────────────────────
    base_cols = set(df.columns)
    try:
        df_ind = compute(df, ind_list, config=config, period_override=period)
    except ValueError as e:
        click.echo(click.style(f"[ERROR] {e}", fg="red"))
        raise SystemExit(1)

    # ── Display ───────────────────────────────────────────────────────────────
    # New columns = anything added by compute() + vwap if it was filled in
    indicator_cols = [
        c for c in df_ind.columns
        if c not in base_cols                                    # genuinely new
        or (c == "vwap" and df_ind["vwap"].notna().any())       # vwap was computed
    ]

    click.echo(click.style("✓ Indicators computed", fg="green"))
    click.echo(f"  Ticker     : {ticker.upper()}")
    click.echo(f"  Bars       : {len(df_ind):,}")
    click.echo(f"  New columns: {', '.join(indicator_cols)}\n")

    # Show only indicator columns + close in the tail view
    display_cols = ["close"] + indicator_cols
    available    = [c for c in display_cols if c in df_ind.columns]
    click.echo(f"── Last {tail} rows ─────────────────────────────────────────────")
    click.echo(df_ind[available].tail(tail).round(4).to_string())
    click.echo("")

    # Latest values summary
    click.echo("── Latest values ────────────────────────────────────────────────")
    last = df_ind.iloc[-1]
    for col in indicator_cols:
        if col in df_ind.columns and not pd.isna(last[col]):
            click.echo(f"  {col:<20}: {last[col]:.4f}")


# ── Stage 3: signals ──────────────────────────────────────────────────────────

@cli.command()
@click.option("--ticker",   required=True, help="Ticker symbol")
@click.option("--strategy", required=True, help="Strategy key, e.g. ema_cross")
@click.option("--from",     "date_from",   default=None, help="Start date YYYY-MM-DD (optional)")
@click.option("--to",       "date_to",     default=None, help="End date   YYYY-MM-DD (optional)")
@click.pass_context
def signals(ctx: click.Context, ticker: str, strategy: str,
            date_from: str, date_to: str) -> None:
    """Generate BUY/SELL/HOLD signals for a strategy. (Stage 3)"""
    logger = ctx.obj["logger"]
    logger.info(f"[STUB] signals | ticker={ticker} strategy={strategy}")
    click.echo(click.style(f"[Stage 3 — STUB] signals not yet implemented.", fg="cyan"))
    click.echo(f"  Ticker   : {ticker}")
    click.echo(f"  Strategy : {strategy}")
    click.echo("  → Implement src/signals/ and strategies/ in Stage 3.")


# ── Stage 4: backtest ─────────────────────────────────────────────────────────

@cli.command()
@click.option("--ticker",   required=True,          help="Ticker symbol")
@click.option("--strategy", required=True,          help="Strategy key, e.g. ema_cross")
@click.option("--from",     "date_from", required=True, help="Start date YYYY-MM-DD")
@click.option("--to",       "date_to",   default=None,  help="End date   YYYY-MM-DD (optional)")
@click.option("--engine",   default=None,
              help="Backtest engine: backtesting | vectorbt (overrides settings.yaml)",
              type=click.Choice(["backtesting", "vectorbt"], case_sensitive=False))
@click.option("--cash",     default=None, type=float,   help="Starting cash (overrides settings.yaml)")
@click.pass_context
def backtest(ctx: click.Context, ticker: str, strategy: str, date_from: str,
             date_to: str, engine: str, cash: float) -> None:
    """Run a backtest and output performance metrics. (Stage 4)"""
    logger = ctx.obj["logger"]
    config  = ctx.obj["config"]
    engine  = engine or config.get("backtest", {}).get("engine", "backtesting")
    cash    = cash   or config.get("backtest", {}).get("initial_cash", 10000)
    logger.info(f"[STUB] backtest | ticker={ticker} strategy={strategy} engine={engine} cash={cash}")
    click.echo(click.style(f"[Stage 4 — STUB] backtest not yet implemented.", fg="cyan"))
    click.echo(f"  Ticker   : {ticker}")
    click.echo(f"  Strategy : {strategy}")
    click.echo(f"  Engine   : {engine}")
    click.echo(f"  Cash     : ${cash:,.2f}")
    click.echo("  → Implement src/backtest/ in Stage 4.")


# ── Stage 5: trade ────────────────────────────────────────────────────────────

@cli.command()
@click.option("--ticker",   required=True, help="Ticker symbol")
@click.option("--strategy", required=True, help="Strategy key, e.g. ema_cross")
@click.option("--mode",     default="paper",
              help="Execution mode: paper | live",
              type=click.Choice(["paper", "live"], case_sensitive=False))
@click.pass_context
def trade(ctx: click.Context, ticker: str, strategy: str, mode: str) -> None:
    """Execute live or paper trades via Alpaca. (Stage 5)"""
    logger = ctx.obj["logger"]

    if mode == "live":
        click.echo(click.style(
            "[WARN] Live mode selected. Ensure Stage 4 backtesting is complete "
            "and results are satisfactory before proceeding.", fg="red"
        ))
        click.confirm("Confirm you want to trade with REAL MONEY?", abort=True)

    logger.info(f"[STUB] trade | ticker={ticker} strategy={strategy} mode={mode}")
    click.echo(click.style(f"[Stage 5 — STUB] trade not yet implemented.", fg="cyan"))
    click.echo(f"  Ticker   : {ticker}")
    click.echo(f"  Strategy : {strategy}")
    click.echo(f"  Mode     : {mode}")
    click.echo("  → Implement src/execution/ in Stage 5.")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
