# quant-algo

**AMHANi Enterprise — Algorithmic Trading System**

A CLI-first quantitative trading pipeline built in Python. Fetches market data, computes indicators, generates signals, backtests strategies, and executes paper (or live) trades via Alpaca.

---

## Project Stages

| Stage | Focus | Status |
|-------|-------|--------|
| 0 | Repo & Environment | ✅ Complete |
| 1 | Data Layer | ✅ Complete |
| 2 | Indicators Engine | ✅ Complete |
| 3 | Signal Generation | ✅ Complete |
| 4 | Backtesting | ✅ Complete |
| 5 | Execution (Paper → Live) | ✅ Complete |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/QUANT-ALGO/quant-algo.git
cd quant-algo
```

### 2. Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure secrets

```bash
cp .env.example .env
# Open .env and fill in your API keys
```

### 5. Configure settings

```bash
cp config/settings.example.yaml config/settings.yaml
# Open config/settings.yaml and adjust parameters
```

---

## Running Tests

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
python -m pytest tests/ -v

# Run only Stage 1 data tests
python -m pytest tests/test_data.py -v
```

> Live fetch tests require a network connection. The PolygonClient tests mock HTTP
> calls so they run offline with no API key needed.

---

## CLI Commands

All commands support `--help` for full usage details.

```bash
# Check version and available commands
python main.py --help

# Fetch OHLCV data (Stage 1)
# Fetch and cache 2 years of AAPL daily bars
python main.py fetch --ticker AAPL --from 2023-01-01 --to 2024-12-31 --source yfinance

# Force re-fetch even if cached
python main.py fetch --ticker AAPL --from 2023-01-01 --to 2024-12-31 --force

python main.py fetch --ticker EURUSD=X --from 2023-01-01 --to 2024-01-01 --source yfinance
python main.py fetch --ticker BTC-USD --from 2023-01-01 --to 2024-01-01 --source yfinance

# Compute indicators (Stage 2)
python main.py indicators --ticker AAPL --from 2023-01-01 --to 2024-01-01 --indicators RSI,EMA --period 14

# Compute RSI + EMA and preview last 5 rows
python main.py indicators --ticker AAPL --from 2023-01-01 --to 2024-01-01 \
  --indicators RSI,EMA,MACD --tail 5

# All 6 indicators at once
python main.py indicators --ticker AAPL --from 2023-01-01 --to 2024-01-01 \
  --indicators ALL

# Generate signals (Stage 3)

# See all strategies available
python main.py signals --ticker AAPL --strategy ema_cross \
  --from 2023-01-01 --to 2024-01-01 --list-strategies

# EMA crossover signals — shows BUY/SELL dates and signal rate
python main.py signals --ticker AAPL --strategy ema_cross \
  --from 2023-01-01 --to 2024-01-01

# RSI mean-reversion signals
python main.py signals --ticker TSLA --strategy rsi_mean_reversion \
  --from 2022-01-01 --to 2024-01-01

# Run backtest (Stage 4)
python main.py backtest --ticker AAPL --strategy ema_cross --from 2022-01-01
python main.py backtest --ticker AAPL \
  --strategy ema_cross,rsi_mean_reversion --from 2022-01-01

# Single strategy — full metrics: return, Sharpe, drawdown, win rate
python main.py backtest --ticker AAPL --strategy ema_cross \
  --from 2022-01-01 --to 2024-01-01 --cash 10000

# Side-by-side comparison table of both strategies
python main.py backtest --ticker AAPL \
  --strategy ema_cross,rsi_mean_reversion \
  --from 2022-01-01 --to 2024-01-01

# Skip saving results to disk
python main.py backtest --ticker SPY --strategy ema_cross \
  --from 2021-01-01 --no-save

# Paper trade (Stage 5)

# Preview what would happen — no order placed
python main.py trade --ticker AAPL --strategy ema_cross --dry-run

# Paper trade — real Alpaca paper account, zero financial risk
python main.py trade --ticker AAPL --strategy ema_cross --mode paper

# Live trade — guarded by double confirmation prompt
python main.py trade --ticker AAPL --strategy ema_cross --mode live
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `POLYGON_API_KEY` | Polygon.io API key |
| `ALPACA_API_KEY` | Alpaca paper/live API key |
| `ALPACA_SECRET_KEY` | Alpaca secret key |
| `ALPACA_BASE_URL` | Alpaca endpoint (paper or live) |
| `ENV` | `development` or `production` |

---

## License

Private — AMHANi Enterprise
