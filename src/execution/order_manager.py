"""
src/execution/order_manager.py
Order Manager — Stage 5.

Sits between signal generation (Stage 3) and the Alpaca API (alpaca_client.py).
Handles:
  - Position sizing:     max % of portfolio per trade (configurable)
  - Stop-loss:           per-position stop calculated from entry price
  - Deduplication:       won't double-enter or double-exit a position
  - Buying-power check:  won't place an order we can't afford
  - Market-hours check:  warns before placing orders when market is closed

Usage:
    from src.execution.alpaca_client import AlpacaClient
    from src.execution.order_manager import OrderManager

    client  = AlpacaClient()
    manager = OrderManager(client, config=config)

    result = manager.execute_signal(ticker="AAPL", signal=1, price=182.50)
    print(result)
    # {"action": "BUY", "qty": 5, "order_id": "...", "stop_price": 178.85}
"""

import logging
import math
from typing import Optional

from src.execution.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Translates signal integers (1 / -1 / 0) into real (paper) orders via
    AlpacaClient, applying position-sizing and safety checks along the way.

    Configuration (from settings.yaml → execution):
        max_position_pct: 0.10   ← max 10% of portfolio per position
        stop_loss_pct:    0.02   ← 2% stop-loss below entry

    All results are plain dicts — safe to log, serialise, and display.
    """

    def __init__(self, client: AlpacaClient, config: Optional[dict] = None) -> None:
        self.client    = client
        cfg            = (config or {}).get("execution", {})
        self.max_pct   = float(cfg.get("max_position_pct", 0.10))
        self.stop_pct  = float(cfg.get("stop_loss_pct",    0.02))
        logger.info(
            f"[order_manager] init | max_position={self.max_pct*100:.0f}% "
            f"| stop_loss={self.stop_pct*100:.0f}%"
        )

    # ── Sizing ────────────────────────────────────────────────────────────────

    def calculate_qty(self, price: float, portfolio_value: float) -> int:
        """
        Return the number of whole shares to buy, respecting max_position_pct.

        Minimum: 1 share (we never return 0 on a valid price).

        Args:
            price:           Current share price.
            portfolio_value: Total portfolio value (from account).

        Returns:
            Integer share count (≥ 1).
        """
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")
        max_dollars = portfolio_value * self.max_pct
        qty         = math.floor(max_dollars / price)
        return max(qty, 1)

    def stop_price_for(self, entry_price: float) -> float:
        """Calculate the stop-loss price for a given entry."""
        return round(entry_price * (1 - self.stop_pct), 2)

    # ── Position checks ───────────────────────────────────────────────────────

    def has_position(self, ticker: str) -> bool:
        """Return True if we currently hold shares of ticker."""
        pos = self.client.get_position(ticker)
        return pos is not None and float(pos.get("qty", 0)) > 0

    def has_pending_order(self, ticker: str) -> bool:
        """Return True if there's already an open order for ticker."""
        orders = self.client.get_orders(status="open")
        return any(o["ticker"].upper() == ticker.upper() for o in orders)

    # ── Core ─────────────────────────────────────────────────────────────────

    def execute_signal(
        self,
        ticker: str,
        signal: int,
        price:  float,
    ) -> dict:
        """
        Translate a signal integer into an order (or a skip with reason).

        Args:
            ticker: Symbol to trade, e.g. "AAPL".
            signal: 1 (BUY), -1 (SELL), 0 (HOLD).
            price:  Most recent close price used for sizing and stop-loss.

        Returns:
            Result dict — always has "action" key. Examples:
              {"action": "BUY",       "qty": 5,  "order_id": "...", "stop_price": 178.85}
              {"action": "SELL",      "qty": 5,  "order_id": "..."}
              {"action": "HOLD"}
              {"action": "SKIP_BUY",  "reason": "already_in_position"}
              {"action": "SKIP_BUY",  "reason": "pending_order_exists"}
              {"action": "SKIP_BUY",  "reason": "insufficient_buying_power"}
              {"action": "SKIP_SELL", "reason": "no_position"}
        """
        ticker = ticker.upper()

        if signal == 0:
            logger.info(f"[order_manager] HOLD {ticker} — signal=0")
            return {"action": "HOLD"}

        # ── Market hours warning ──────────────────────────────────────────────
        if not self.client.is_market_open():
            logger.warning(
                f"[order_manager] Market is CLOSED — order for {ticker} "
                "will be queued as a day order and executed at next open."
            )

        account = self.client.get_account()
        portfolio_value = float(account["portfolio_value"])
        buying_power    = float(account["buying_power"])

        # ── BUY ───────────────────────────────────────────────────────────────
        if signal == 1:
            if self.has_position(ticker):
                logger.info(f"[order_manager] SKIP BUY {ticker} — already in position")
                return {"action": "SKIP_BUY", "reason": "already_in_position"}

            if self.has_pending_order(ticker):
                logger.info(f"[order_manager] SKIP BUY {ticker} — pending order exists")
                return {"action": "SKIP_BUY", "reason": "pending_order_exists"}

            qty  = self.calculate_qty(price, portfolio_value)
            cost = qty * price

            if cost > buying_power:
                logger.warning(
                    f"[order_manager] SKIP BUY {ticker} — "
                    f"cost ${cost:,.2f} > buying power ${buying_power:,.2f}"
                )
                return {"action": "SKIP_BUY", "reason": "insufficient_buying_power"}

            stop  = self.stop_price_for(price)
            order = self.client.submit_order(
                ticker=ticker, qty=qty, side="buy", stop_price=stop
            )
            logger.info(
                f"[order_manager] BUY {qty}x {ticker} @ ~${price:.2f} "
                f"| stop=${stop:.2f} | order={order['id']}"
            )
            return {
                "action":     "BUY",
                "qty":        qty,
                "order_id":   order["id"],
                "stop_price": stop,
                "est_cost":   round(cost, 2),
            }

        # ── SELL ──────────────────────────────────────────────────────────────
        if signal == -1:
            pos = self.client.get_position(ticker)
            if pos is None or float(pos.get("qty", 0)) == 0:
                logger.info(f"[order_manager] SKIP SELL {ticker} — no position")
                return {"action": "SKIP_SELL", "reason": "no_position"}

            qty   = int(float(pos["qty"]))
            entry = float(pos["avg_entry"])
            pl    = round((price - entry) / entry * 100, 2)

            order = self.client.submit_order(ticker=ticker, qty=qty, side="sell")
            logger.info(
                f"[order_manager] SELL {qty}x {ticker} @ ~${price:.2f} "
                f"| entry=${entry:.2f} | P&L≈{pl:+.2f}% | order={order['id']}"
            )
            return {
                "action":       "SELL",
                "qty":          qty,
                "order_id":     order["id"],
                "entry_price":  entry,
                "est_pl_pct":   pl,
            }

        # Shouldn't reach here — guard against unexpected signal values
        logger.warning(f"[order_manager] Unknown signal value: {signal}")
        return {"action": "UNKNOWN", "signal": signal}

    # ── Convenience ───────────────────────────────────────────────────────────

    def close_all_positions(self) -> list[dict]:
        """
        Flatten the entire portfolio — sell everything.
        Useful for end-of-day or emergency exits.

        Returns list of result dicts from execute_signal().
        """
        results  = []
        positions = self.client.get_positions()
        if not positions:
            logger.info("[order_manager] close_all_positions — no open positions")
            return results

        for pos in positions:
            ticker = pos["ticker"]
            price  = float(pos["current_price"])
            result = self.execute_signal(ticker=ticker, signal=-1, price=price)
            results.append(result)

        return results
