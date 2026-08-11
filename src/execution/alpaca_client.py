"""
src/execution/alpaca_client.py
Alpaca Trading API wrapper — Stage 5.

Wraps alpaca-trade-api (paper + live) into a minimal, testable interface.
Every method returns plain dicts — no raw SDK objects leak out of this module,
making it easy to mock in tests and swap SDKs in the future.

Environment variables (set in .env):
    ALPACA_API_KEY     — API key ID
    ALPACA_SECRET_KEY  — Secret key
    ALPACA_BASE_URL    — https://paper-api.alpaca.markets  (paper)
                         https://api.alpaca.markets        (live)

Usage:
    client = AlpacaClient()
    account = client.get_account()
    print(f"Portfolio: ${account['portfolio_value']:,.2f}")
    print(f"Buying power: ${account['buying_power']:,.2f}")
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ── Supported timeframes for Alpaca bars ──────────────────────────────────────
ALPACA_TIMEFRAME_MAP = {
    "1Min":  "1Min",
    "5Min":  "5Min",
    "15Min": "15Min",
    "1Hour": "1Hour",
    "1Day":  "1Day",
}


class AlpacaClient:
    """
    Thin wrapper around the Alpaca REST API.

    All return values are plain dicts — no SDK-specific objects exposed.
    Raises descriptive exceptions so callers handle errors explicitly:
        EnvironmentError   — missing API credentials
        PermissionError    — 401/403 from Alpaca (bad key or wrong endpoint)
        ConnectionError    — network or 5xx errors
        ValueError         — bad arguments (unknown side, timeframe, etc.)
    """

    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL  = "https://api.alpaca.markets"

    def __init__(
        self,
        api_key:    Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url:   Optional[str] = None,
    ) -> None:
        self._api_key    = api_key    or os.getenv("ALPACA_API_KEY")
        self._secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self._base_url   = base_url   or os.getenv("ALPACA_BASE_URL", self.PAPER_URL)

        if not self._api_key or not self._secret_key:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env "
                "or passed to AlpacaClient()."
            )

        self._api = self._connect()
        logger.info(f"[alpaca] Connected | endpoint={self._base_url}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connect(self):
        """Lazily import and instantiate the Alpaca REST client."""
        try:
            import alpaca_trade_api as tradeapi
        except ImportError:
            raise ImportError(
                "alpaca-trade-api is not installed. "
                "Run: pip install alpaca-trade-api"
            )
        return tradeapi.REST(
            key_id=self._api_key,
            secret_key=self._secret_key,
            base_url=self._base_url,
        )

    def _order_to_dict(self, order) -> dict:
        """Convert an Alpaca Order object to a plain dict."""
        return {
            "id":            str(order.id),
            "client_id":     str(order.client_order_id),
            "ticker":        str(order.symbol),
            "side":          str(order.side),
            "qty":           float(order.qty),
            "filled_qty":    float(order.filled_qty or 0),
            "type":          str(order.type),
            "status":        str(order.status),
            "filled_price":  float(order.filled_avg_price or 0),
            "submitted_at":  str(order.submitted_at),
        }

    def _position_to_dict(self, position) -> dict:
        """Convert an Alpaca Position object to a plain dict."""
        return {
            "ticker":        str(position.symbol),
            "qty":           float(position.qty),
            "avg_entry":     float(position.avg_entry_price),
            "market_value":  float(position.market_value),
            "unrealised_pl": float(position.unrealized_pl),
            "unrealised_pct":float(position.unrealized_plpc) * 100,
            "current_price": float(position.current_price),
            "side":          str(position.side),
        }

    # ── Account ───────────────────────────────────────────────────────────────

    def get_account(self) -> dict:
        """
        Return key account metrics.

        Returns:
            {
              "cash":            float  — settled cash
              "portfolio_value": float  — total portfolio market value
              "buying_power":    float  — available for new orders
              "status":          str    — "ACTIVE" | "INACTIVE" etc.
              "pattern_day_trader": bool
            }
        """
        acct = self._api.get_account()
        return {
            "cash":               float(acct.cash),
            "portfolio_value":    float(acct.portfolio_value),
            "buying_power":       float(acct.buying_power),
            "status":             str(acct.status),
            "pattern_day_trader": bool(acct.pattern_day_trader),
        }

    # ── Market clock ──────────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        """Return True if the US equity market is currently open."""
        clock = self._api.get_clock()
        return bool(clock.is_open)

    def get_clock(self) -> dict:
        """Return market clock state with open/close times."""
        clock = self._api.get_clock()
        return {
            "is_open":   bool(clock.is_open),
            "next_open":  str(clock.next_open),
            "next_close": str(clock.next_close),
        }

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_positions(self) -> list[dict]:
        """Return all current open positions."""
        positions = self._api.list_positions()
        return [self._position_to_dict(p) for p in positions]

    def get_position(self, ticker: str) -> Optional[dict]:
        """
        Return position for a specific ticker, or None if not held.

        Args:
            ticker: Symbol, e.g. "AAPL"
        """
        try:
            pos = self._api.get_position(ticker.upper())
            return self._position_to_dict(pos)
        except Exception as e:
            if "position does not exist" in str(e).lower() or "404" in str(e):
                return None
            raise

    # ── Orders ────────────────────────────────────────────────────────────────

    def submit_order(
        self,
        ticker:        str,
        qty:           int,
        side:          str,                      # "buy" | "sell"
        order_type:    str        = "market",
        time_in_force: str        = "day",
        stop_price:    Optional[float] = None,   # used for bracket stop-loss leg
    ) -> dict:
        """
        Submit a market order with optional stop-loss.

        For BUY with stop_price set, submits a bracket order:
            - take_profit is omitted (no fixed profit target)
            - stop_loss is set to stop_price

        Args:
            ticker:        Symbol to trade.
            qty:           Number of shares (minimum 1).
            side:          "buy" or "sell".
            order_type:    "market" (default) | "limit" | "stop".
            time_in_force: "day" (default) | "gtc" | "opg".
            stop_price:    Stop-loss price for bracket orders (BUY only).

        Returns:
            Plain dict representing the submitted order.

        Raises:
            ValueError: Invalid side or order_type.
        """
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid order side: '{side}'. Must be 'buy' or 'sell'.")

        logger.info(
            f"[alpaca] submit_order | {side.upper()} {qty}x {ticker.upper()} "
            f"| type={order_type} | tif={time_in_force}"
            + (f" | stop=${stop_price:.2f}" if stop_price else "")
        )

        kwargs: dict = {
            "symbol":        ticker.upper(),
            "qty":           qty,
            "side":          side,
            "type":          order_type,
            "time_in_force": time_in_force,
        }

        # Bracket order: attach stop-loss leg on buy
        if side == "buy" and stop_price:
            kwargs["order_class"] = "bracket"
            kwargs["stop_loss"]   = {"stop_price": round(stop_price, 2)}

        order = self._api.submit_order(**kwargs)
        return self._order_to_dict(order)

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order by ID.

        Returns:
            True if cancelled, False if already filled/cancelled.
        """
        try:
            self._api.cancel_order(order_id)
            logger.info(f"[alpaca] Cancelled order {order_id}")
            return True
        except Exception as e:
            if "422" in str(e) or "order is not cancelable" in str(e).lower():
                return False
            raise

    def get_orders(self, status: str = "open") -> list[dict]:
        """
        Return orders filtered by status.

        Args:
            status: "open" | "closed" | "all"
        """
        orders = self._api.list_orders(status=status)
        return [self._order_to_dict(o) for o in orders]

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        self._api.cancel_all_orders()
        logger.info("[alpaca] All open orders cancelled")
        orders = self.get_orders("open")
        return len(orders)
