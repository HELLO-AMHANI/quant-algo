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
@click.option("--ticker",   required=True,  help="Ticker symbol, e.g. AAPL")
@click.option("--strategy", required=True,  help="Strategy key: ema_cross | rsi_mean_reversion")
@click.option("--from",     "date_from", required=True, help="Start date YYYY-MM-DD")
@click.option("--to",       "date_to",   required=True, help="End date   YYYY-MM-DD")
@click.option("--source",   default="yfinance",
              help="Data source: polygon | yfinance",
              type=click.Choice(["polygon", "yfinance"], case_sensitive=False))
@click.option("--tail",     default=10, type=int,
              help="Number of recent signal rows to display (default: 10)")
@click.option("--list-strategies", "list_strats", is_flag=True,
              help="List all available strategies and exit")
@click.pass_context
def signals(ctx: click.Context, ticker: str, strategy: str,
            date_from: str, date_to: str, source: str,
            tail: int, list_strats: bool) -> None:
    """Generate BUY/SELL/HOLD signals from a strategy. (Stage 3)"""
    from src.data.data_manager           import DataManager
    from src.signals.signal_generator   import generate_signals, list_strategies, signal_summary

    logger = ctx.obj["logger"]
    config = ctx.obj["config"]

    if list_strats:
        click.echo("Available strategies:")
        for s in list_strategies():
            click.echo(f"  {s}")
        return

    click.echo(
        f"\nSignals | {ticker.upper()} | {date_from} → {date_to} | {strategy}\n"
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        dm = DataManager()
        df = dm.get_ohlcv(ticker, date_from, date_to, source=source)
    except Exception as e:
        click.echo(click.style(f"[ERROR] Data load failed: {e}", fg="red"))
        raise SystemExit(1)

    if df.empty:
        click.echo(click.style("No data returned. Run fetch first.", fg="yellow"))
        return

    # ── Generate signals ──────────────────────────────────────────────────────
    try:
        df_sig = generate_signals(df, strategy_name=strategy, config=config)
    except ValueError as e:
        click.echo(click.style(f"[ERROR] {e}", fg="red"))
        raise SystemExit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = signal_summary(df_sig)

    click.echo(click.style("✓ Signals generated", fg="green"))
    click.echo(f"  Ticker       : {ticker.upper()}")
    click.echo(f"  Strategy     : {strategy}")
    click.echo(f"  Total bars   : {summary['total_bars']:,}")
    click.echo(f"  BUY signals  : {summary['buy_signals']}")
    click.echo(f"  SELL signals : {summary['sell_signals']}")
    click.echo(f"  Hold bars    : {summary['hold_bars']:,}")
    click.echo(f"  Signal rate  : {summary['signal_rate_pct']}%")
    click.echo(f"  First signal : {summary['first_signal_date']}")
    click.echo(f"  Last signal  : {summary['last_signal_date']}")

    # ── Show only bars where a signal fired ───────────────────────────────────
    active = df_sig[df_sig["signal"] != 0].copy()
    if active.empty:
        click.echo(click.style("\n  No signals fired in this date range.", fg="yellow"))
        return

    active["action"] = active["signal"].map({1: "BUY", -1: "SELL"})
    indicator_extras = [c for c in active.columns
                        if c not in {"open","high","low","close","volume","vwap",
                                     "signal","strategy","action"}]
    display_cols = ["close"] + indicator_extras[:4] + ["action"]
    available    = [c for c in display_cols if c in active.columns]

    click.echo(
        f"\n── Last {min(tail, len(active))} active signals "
        f"({len(active)} total) ──────────────────────────"
    )
    click.echo(active[available].tail(tail).round(4).to_string())
    click.echo("")


# ── Stage 4: backtest ─────────────────────────────────────────────────────────

@cli.command()
@click.option("--ticker",    required=True,  help="Ticker symbol, e.g. AAPL")
@click.option("--strategy",  required=True,
              help="Strategy slug or comma-separated list for comparison: ema_cross,rsi_mean_reversion")
@click.option("--from",      "date_from", required=True,  help="Start date YYYY-MM-DD")
@click.option("--to",        "date_to",   default=None,   help="End date YYYY-MM-DD (optional)")
@click.option("--source",    default="yfinance",
              help="Data source: polygon | yfinance",
              type=click.Choice(["polygon", "yfinance"], case_sensitive=False))
@click.option("--cash",      default=None, type=float,
              help="Starting cash — overrides settings.yaml (default: 10000)")
@click.option("--commission", default=None, type=float,
              help="Per-trade commission fraction — overrides settings.yaml (default: 0.001)")
@click.option("--no-save",   is_flag=True, default=False,
              help="Skip saving results to results/ directory")
@click.pass_context
def backtest(ctx: click.Context, ticker: str, strategy: str, date_from: str,
             date_to: str, source: str, cash: float,
             commission: float, no_save: bool) -> None:
    """Run a backtest and display performance metrics. (Stage 4)

    Single strategy:
      python main.py backtest --ticker AAPL --strategy ema_cross --from 2022-01-01

    Compare two strategies side-by-side:
      python main.py backtest --ticker AAPL --strategy ema_cross,rsi_mean_reversion --from 2022-01-01
    """
    from src.data.data_manager      import DataManager
    from src.backtest.runner        import run_backtest, compare_strategies

    logger = ctx.obj["logger"]
    config = ctx.obj["config"]

    cash       = cash       or config.get("backtest", {}).get("initial_cash", 10_000)
    commission = commission or config.get("backtest", {}).get("commission",   0.001)
    save       = not no_save
    results_dir = config.get("backtest", {}).get("results_dir", "results")

    strategy_names = [s.strip() for s in strategy.split(",")]
    multi          = len(strategy_names) > 1

    # ── Load data ─────────────────────────────────────────────────────────────
    click.echo(f"\nBacktest | {ticker.upper()} | {date_from} → {date_to or 'today'}\n")
    try:
        dm = DataManager()
        df = dm.get_ohlcv(ticker, date_from, date_to or "2099-01-01", source=source)
    except Exception as e:
        click.echo(click.style(f"[ERROR] Data load failed: {e}", fg="red"))
        raise SystemExit(1)

    if df.empty:
        click.echo(click.style("No data returned. Run fetch first.", fg="yellow"))
        return

    click.echo(f"  Loaded {len(df):,} bars | ${cash:,.0f} starting cash | {commission*100:.2f}% commission\n")

    # ── Single or multi-strategy ──────────────────────────────────────────────
    METRIC_LABELS = {
        "total_return_pct":      "Total Return %",
        "buy_hold_return_pct":   "Buy & Hold %",
        "annualised_return_pct": "Ann. Return %",
        "sharpe_ratio":          "Sharpe Ratio",
        "sortino_ratio":         "Sortino Ratio",
        "max_drawdown_pct":      "Max Drawdown %",
        "win_rate_pct":          "Win Rate %",
        "profit_factor":         "Profit Factor",
        "total_trades":          "# Trades",
        "exposure_time_pct":     "Exposure %",
        "equity_final":          "Final Equity $",
    }

    if not multi:
        # ── Single strategy ───────────────────────────────────────────────────
        try:
            result = run_backtest(
                df, strategy_names[0],
                config=config, cash=cash,
                commission=commission, results_dir=results_dir, save=save,
            )
        except ValueError as e:
            click.echo(click.style(f"[ERROR] {e}", fg="red"))
            raise SystemExit(1)

        m = result["metrics"]
        click.echo(click.style("✓ Backtest complete", fg="green"))
        click.echo(f"  Strategy : {strategy_names[0]}")
        click.echo("")

        for key, label in METRIC_LABELS.items():
            val = m.get(key, "—")
            if isinstance(val, float):
                line = f"  {label:<22}: {val:>10.2f}"
            else:
                line = f"  {label:<22}: {val:>10}"
            # Colour returns green/red
            if key == "total_return_pct":
                colour = "green" if val > 0 else "red"
                click.echo(click.style(line, fg=colour))
            elif key == "max_drawdown_pct":
                click.echo(click.style(line, fg="red"))
            else:
                click.echo(line)

        if save and result.get("json_path"):
            click.echo(f"\n  Results → {result['json_path'].name}")
            click.echo(f"           → {result['csv_path'].name}")

    else:
        # ── Multi-strategy comparison ─────────────────────────────────────────
        click.echo(f"  Comparing: {', '.join(strategy_names)}\n")
        try:
            comparison = compare_strategies(
                df, strategy_names,
                config=config, cash=cash,
                commission=commission, results_dir=results_dir,
            )
        except Exception as e:
            click.echo(click.style(f"[ERROR] {e}", fg="red"))
            raise SystemExit(1)

        # Print comparison table
        col_w = 18
        header = f"  {'Metric':<24}" + "".join(f"{n[:col_w]:>{col_w}}" for n in strategy_names)
        click.echo(click.style(header, bold=True))
        click.echo("  " + "─" * (24 + col_w * len(strategy_names)))

        for key, label in METRIC_LABELS.items():
            row = f"  {label:<24}"
            for name in strategy_names:
                m   = comparison.get(name, {})
                val = m.get(key, "ERR") if "error" not in m else "ERR"
                if isinstance(val, float):
                    row += f"{val:>{col_w}.2f}"
                else:
                    row += f"{str(val):>{col_w}}"
            if "return" in key:
                vals = [comparison.get(n, {}).get(key, 0) for n in strategy_names]
                click.echo(click.style(row, fg="green" if any(v > 0 for v in vals) else "red"))
            else:
                click.echo(row)

        click.echo("")
        if save:
            click.echo("  Results saved to results/")

    click.echo("")


# ── Stage 5: trade ────────────────────────────────────────────────────────────

@cli.command()
@click.option("--ticker",    required=True,  help="Ticker symbol, e.g. AAPL")
@click.option("--strategy",  required=True,  help="Strategy slug: ema_cross | rsi_mean_reversion")
@click.option("--mode",      default="paper",
              help="Execution mode: paper | live",
              type=click.Choice(["paper", "live"], case_sensitive=False))
@click.option("--lookback",  default=100, type=int,
              help="Number of recent bars to load for indicator warm-up (default: 100)")
@click.option("--source",    default="yfinance",
              help="Data source: polygon | yfinance",
              type=click.Choice(["polygon", "yfinance"], case_sensitive=False))
@click.option("--dry-run",   is_flag=True, default=False,
              help="Compute signal and show intended action without placing any order")
@click.pass_context
def trade(ctx: click.Context, ticker: str, strategy: str, mode: str,
          lookback: int, source: str, dry_run: bool) -> None:
    """Fetch latest data, generate a signal, and execute a paper or live order.

    Always start in paper mode:
      python main.py trade --ticker AAPL --strategy ema_cross --mode paper

    Check what would happen without placing an order:
      python main.py trade --ticker AAPL --strategy ema_cross --dry-run

    Only switch to live after extensive paper testing:
      python main.py trade --ticker AAPL --strategy ema_cross --mode live
    """
    import datetime
    from src.data.data_manager         import DataManager
    from src.signals.signal_generator  import generate_signals, signal_summary
    from src.execution.alpaca_client   import AlpacaClient
    from src.execution.order_manager   import OrderManager

    logger = ctx.obj["logger"]
    config = ctx.obj["config"]

    # ── Live-mode confirmation gate ───────────────────────────────────────────
    if mode == "live" and not dry_run:
        click.echo(click.style(
            "\n⚠  LIVE mode — real money will be used.", fg="red", bold=True
        ))
        click.echo("   Confirm you have reviewed Stage 4 backtest results.")
        click.confirm("   Proceed with LIVE trading?", abort=True)

    mode_label = click.style(f"[{mode.upper()}]", fg="yellow" if mode == "paper" else "red")
    dry_label  = click.style(" [DRY-RUN]", fg="cyan") if dry_run else ""
    click.echo(f"\nTrade {mode_label}{dry_label} | {ticker.upper()} | {strategy}\n")

    # ── Load recent data ──────────────────────────────────────────────────────
    date_to   = datetime.date.today().isoformat()
    date_from = (datetime.date.today() - datetime.timedelta(days=lookback * 2)).isoformat()

    try:
        dm = DataManager()
        df = dm.get_ohlcv(ticker, date_from, date_to, source=source)
    except Exception as e:
        click.echo(click.style(f"[ERROR] Data fetch failed: {e}", fg="red"))
        raise SystemExit(1)

    if df.empty or len(df) < 30:
        click.echo(click.style("Insufficient data — need at least 30 bars.", fg="yellow"))
        return

    # Use only the most recent `lookback` bars
    df = df.tail(lookback).copy()

    # ── Generate signal ───────────────────────────────────────────────────────
    try:
        df_sig = generate_signals(df, strategy_name=strategy, config=config)
    except ValueError as e:
        click.echo(click.style(f"[ERROR] Signal generation failed: {e}", fg="red"))
        raise SystemExit(1)

    latest          = df_sig.iloc[-1]
    latest_signal   = int(latest["signal"])
    latest_price    = float(latest["close"])
    latest_date     = str(df_sig.index[-1].date())
    signal_label    = {1: "BUY ▲", -1: "SELL ▼", 0: "HOLD —"}.get(latest_signal, "?")
    signal_colour   = {1: "green",  -1: "red",   0: "white"}.get(latest_signal, "white")

    summary = signal_summary(df_sig)

    click.echo(f"  Ticker        : {ticker.upper()}")
    click.echo(f"  Strategy      : {strategy}")
    click.echo(f"  Latest bar    : {latest_date}  close=${latest_price:,.2f}")
    click.echo(f"  Signal        : " + click.style(signal_label, fg=signal_colour, bold=True))
    click.echo(f"  Total signals : {summary['buy_signals']} BUY  "
               f"{summary['sell_signals']} SELL  "
               f"({summary['signal_rate_pct']}% signal rate)")
    click.echo("")

    if dry_run:
        click.echo(click.style(
            f"  [DRY-RUN] Would attempt to execute signal={latest_signal} "
            f"at ${latest_price:,.2f}. No order placed.", fg="cyan"
        ))
        click.echo("")
        return

    # ── Connect to Alpaca ─────────────────────────────────────────────────────
    try:
        client = AlpacaClient()
    except EnvironmentError as e:
        click.echo(click.style(f"[ERROR] Alpaca credentials missing: {e}", fg="red"))
        click.echo("  → Copy .env.example to .env and add your Alpaca keys.")
        raise SystemExit(1)

    # ── Account snapshot ──────────────────────────────────────────────────────
    try:
        account = client.get_account()
        clock   = client.get_clock()
    except Exception as e:
        click.echo(click.style(f"[ERROR] Alpaca connection failed: {e}", fg="red"))
        raise SystemExit(1)

    market_status = click.style(
        "OPEN ●" if clock["is_open"] else "CLOSED ○",
        fg="green" if clock["is_open"] else "yellow"
    )
    click.echo(f"  Portfolio     : ${account['portfolio_value']:>12,.2f}")
    click.echo(f"  Cash          : ${account['cash']:>12,.2f}")
    click.echo(f"  Buying power  : ${account['buying_power']:>12,.2f}")
    click.echo(f"  Market        : {market_status}")
    click.echo("")

    # ── Execute signal ────────────────────────────────────────────────────────
    manager = OrderManager(client, config=config)

    try:
        result = manager.execute_signal(
            ticker=ticker,
            signal=latest_signal,
            price=latest_price,
        )
    except Exception as e:
        click.echo(click.style(f"[ERROR] Order execution failed: {e}", fg="red"))
        logger.exception("Order execution error")
        raise SystemExit(1)

    # ── Display result ────────────────────────────────────────────────────────
    action = result["action"]

    if action == "BUY":
        click.echo(click.style(
            f"  ✓ BUY order submitted — {result['qty']}x {ticker.upper()} "
            f"@ ~${latest_price:,.2f}  |  stop=${result['stop_price']:,.2f}  "
            f"|  est. cost=${result['est_cost']:,.2f}",
            fg="green", bold=True
        ))
        click.echo(f"  Order ID: {result['order_id']}")

    elif action == "SELL":
        pl_colour = "green" if result.get("est_pl_pct", 0) >= 0 else "red"
        click.echo(click.style(
            f"  ✓ SELL order submitted — {result['qty']}x {ticker.upper()} "
            f"@ ~${latest_price:,.2f}  |  entry=${result['entry_price']:,.2f}  "
            f"|  P&L≈{result.get('est_pl_pct', 0):+.2f}%",
            fg=pl_colour, bold=True
        ))
        click.echo(f"  Order ID: {result['order_id']}")

    elif action == "HOLD":
        click.echo(click.style(
            f"  — HOLD — no action on {ticker.upper()}.", fg="white"
        ))

    elif action.startswith("SKIP"):
        reason_map = {
            "already_in_position":     "already holding a position",
            "pending_order_exists":    "open order already exists",
            "insufficient_buying_power": "not enough buying power",
            "no_position":             "no position to sell",
        }
        reason = reason_map.get(result.get("reason", ""), result.get("reason", ""))
        click.echo(click.style(
            f"  ⚠  Skipped — {reason}.", fg="yellow"
        ))

    click.echo("")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
