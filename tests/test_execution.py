"""
tests/test_execution.py
Unit tests for Stage 5: Execution Layer

Run with:
    python -m pytest tests/test_execution.py -v

ALL Alpaca API calls are mocked — no real credentials needed,
no network traffic, no real orders placed.
"""

import pytest
import math
from unittest.mock import MagicMock, patch, PropertyMock


# ── Mock factories ─────────────────────────────────────────────────────────────

def _mock_alpaca_api(
    portfolio_value: float = 50_000.0,
    cash:            float = 30_000.0,
    buying_power:    float = 30_000.0,
    is_open:         bool  = True,
    positions:       list  = None,
    open_orders:     list  = None,
):
    """Build a fully mocked alpaca_trade_api.REST object."""
    api = MagicMock()

    # Account
    acct = MagicMock()
    acct.cash            = str(cash)
    acct.portfolio_value = str(portfolio_value)
    acct.buying_power    = str(buying_power)
    acct.status          = "ACTIVE"
    acct.pattern_day_trader = False
    api.get_account.return_value = acct

    # Clock
    clock = MagicMock()
    clock.is_open    = is_open
    clock.next_open  = "2024-01-02T09:30:00-05:00"
    clock.next_close = "2024-01-02T16:00:00-05:00"
    api.get_clock.return_value = clock

    # Positions
    api.list_positions.return_value = positions or []
    api.get_position.side_effect    = Exception("position does not exist")

    # Orders
    api.list_orders.return_value = open_orders or []

    # Submit order — returns a mock Order
    order = MagicMock()
    order.id                  = "mock-order-id-001"
    order.client_order_id     = "coid-001"
    order.symbol              = "AAPL"
    order.side                = "buy"
    order.qty                 = "5"
    order.filled_qty          = "0"
    order.type                = "market"
    order.status              = "accepted"
    order.filled_avg_price    = None
    order.submitted_at        = "2024-01-02T09:30:01Z"
    api.submit_order.return_value = order

    return api


def _make_client(api_mock=None, **kwargs):
    """Return an AlpacaClient with the internal _api replaced by a mock."""
    from src.execution.alpaca_client import AlpacaClient
    with patch("src.execution.alpaca_client.AlpacaClient._connect",
               return_value=api_mock or _mock_alpaca_api(**kwargs)):
        client = AlpacaClient(
            api_key="test_key",
            secret_key="test_secret",
            base_url="https://paper-api.alpaca.markets",
        )
    return client


def _make_manager(api_mock=None, config=None, **kwargs):
    """Return an OrderManager wired to a mocked AlpacaClient."""
    from src.execution.order_manager import OrderManager
    client = _make_client(api_mock=api_mock, **kwargs)
    return OrderManager(client, config=config)


# ─────────────────────────────────────────────────────────────────────────────
# AlpacaClient Construction
# ─────────────────────────────────────────────────────────────────────────────

class TestAlpacaClientInit:

    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY",    raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        from src.execution.alpaca_client import AlpacaClient
        with pytest.raises(EnvironmentError, match="ALPACA_API_KEY"):
            AlpacaClient(api_key=None, secret_key=None)

    def test_raises_without_secret_key(self, monkeypatch):
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        from src.execution.alpaca_client import AlpacaClient
        with pytest.raises(EnvironmentError, match="ALPACA_API_KEY"):
            AlpacaClient(api_key="key", secret_key=None)

    def test_connects_with_valid_credentials(self):
        client = _make_client()
        assert client is not None

    def test_default_url_is_paper(self, monkeypatch):
        monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
        from src.execution.alpaca_client import AlpacaClient
        with patch("src.execution.alpaca_client.AlpacaClient._connect",
                   return_value=_mock_alpaca_api()):
            client = AlpacaClient(api_key="k", secret_key="s")
        assert "paper" in client._base_url


# ─────────────────────────────────────────────────────────────────────────────
# AlpacaClient — Account & Clock
# ─────────────────────────────────────────────────────────────────────────────

class TestAlpacaClientAccount:

    def test_get_account_keys(self):
        client  = _make_client(portfolio_value=50_000, cash=30_000, buying_power=30_000)
        account = client.get_account()
        for key in ["cash", "portfolio_value", "buying_power", "status"]:
            assert key in account

    def test_get_account_values(self):
        client  = _make_client(portfolio_value=50_000, cash=20_000, buying_power=20_000)
        account = client.get_account()
        assert account["portfolio_value"] == 50_000.0
        assert account["cash"]            == 20_000.0
        assert account["status"]          == "ACTIVE"

    def test_all_account_values_numeric_or_str(self):
        account = _make_client().get_account()
        assert isinstance(account["portfolio_value"], float)
        assert isinstance(account["cash"],            float)
        assert isinstance(account["buying_power"],    float)
        assert isinstance(account["status"],          str)

    def test_is_market_open_true(self):
        client = _make_client(is_open=True)
        assert client.is_market_open() is True

    def test_is_market_open_false(self):
        client = _make_client(is_open=False)
        assert client.is_market_open() is False

    def test_get_clock_keys(self):
        clock = _make_client().get_clock()
        assert "is_open"    in clock
        assert "next_open"  in clock
        assert "next_close" in clock


# ─────────────────────────────────────────────────────────────────────────────
# AlpacaClient — Positions
# ─────────────────────────────────────────────────────────────────────────────

class TestAlpacaClientPositions:

    def _mock_position(self, ticker="AAPL", qty=5, entry=150.0, current=155.0):
        pos = MagicMock()
        pos.symbol          = ticker
        pos.qty             = str(qty)
        pos.avg_entry_price = str(entry)
        pos.market_value    = str(qty * current)
        pos.unrealized_pl   = str((current - entry) * qty)
        pos.unrealized_plpc = str((current - entry) / entry)
        pos.current_price   = str(current)
        pos.side            = "long"
        return pos

    def test_get_position_returns_none_when_not_held(self):
        client = _make_client()
        # api.get_position raises "position does not exist" by default in mock
        result = client.get_position("AAPL")
        assert result is None

    def test_get_position_returns_dict_when_held(self):
        api = _mock_alpaca_api()
        api.get_position.side_effect = None
        api.get_position.return_value = self._mock_position("AAPL", 5, 150.0, 160.0)

        client = _make_client(api_mock=api)
        pos    = client.get_position("AAPL")
        assert pos is not None
        assert pos["ticker"]    == "AAPL"
        assert pos["qty"]       == 5.0
        assert pos["avg_entry"] == 150.0

    def test_get_positions_empty(self):
        client = _make_client()
        assert client.get_positions() == []

    def test_get_positions_populated(self):
        api = _mock_alpaca_api()
        api.list_positions.return_value = [
            self._mock_position("AAPL", 5, 150.0, 155.0),
            self._mock_position("MSFT", 3, 300.0, 310.0),
        ]
        client    = _make_client(api_mock=api)
        positions = client.get_positions()
        assert len(positions) == 2
        tickers   = [p["ticker"] for p in positions]
        assert "AAPL" in tickers
        assert "MSFT" in tickers


# ─────────────────────────────────────────────────────────────────────────────
# AlpacaClient — Orders
# ─────────────────────────────────────────────────────────────────────────────

class TestAlpacaClientOrders:

    def test_submit_buy_order_returns_dict(self):
        client = _make_client()
        order  = client.submit_order("AAPL", qty=5, side="buy")
        assert order["id"]     == "mock-order-id-001"
        assert order["side"]   == "buy"
        assert order["status"] == "accepted"

    def test_submit_sell_order(self):
        api = _mock_alpaca_api()
        o   = api.submit_order.return_value
        o.side = "sell"
        client = _make_client(api_mock=api)
        order  = client.submit_order("AAPL", qty=5, side="sell")
        assert order["side"] == "sell"

    def test_invalid_side_raises(self):
        client = _make_client()
        with pytest.raises(ValueError, match="Invalid order side"):
            client.submit_order("AAPL", qty=5, side="hold")

    def test_submit_with_stop_price_passes_bracket(self):
        api    = _mock_alpaca_api()
        client = _make_client(api_mock=api)
        client.submit_order("AAPL", qty=5, side="buy", stop_price=145.0)
        call_kwargs = api.submit_order.call_args.kwargs
        assert call_kwargs.get("order_class") == "bracket"
        assert "stop_loss" in call_kwargs

    def test_cancel_order_returns_true(self):
        client = _make_client()
        result = client.cancel_order("mock-order-id-001")
        assert result is True

    def test_cancel_already_filled_returns_false(self):
        api = _mock_alpaca_api()
        api.cancel_order.side_effect = Exception("422 order is not cancelable")
        client = _make_client(api_mock=api)
        result = client.cancel_order("filled-order-id")
        assert result is False


# ─────────────────────────────────────────────────────────────────────────────
# OrderManager — Position Sizing
# ─────────────────────────────────────────────────────────────────────────────

class TestOrderManagerSizing:

    def test_calculate_qty_basic(self):
        manager = _make_manager(portfolio_value=50_000, buying_power=30_000)
        # 10% of $50k = $5k. At $100/share → 50 shares
        qty = manager.calculate_qty(price=100.0, portfolio_value=50_000)
        assert qty == 50

    def test_calculate_qty_rounds_down(self):
        manager = _make_manager()
        # 10% of $10k = $1k. At $333/share → floor(3.003) = 3
        qty = manager.calculate_qty(price=333.0, portfolio_value=10_000)
        assert qty == 3

    def test_calculate_qty_minimum_one(self):
        manager = _make_manager()
        # 10% of $100 = $10. At $500/share → floor(0.02) = 0 → clamped to 1
        qty = manager.calculate_qty(price=500.0, portfolio_value=100)
        assert qty == 1

    def test_calculate_qty_invalid_price_raises(self):
        manager = _make_manager()
        with pytest.raises(ValueError):
            manager.calculate_qty(price=0, portfolio_value=10_000)

    def test_stop_price_below_entry(self):
        manager = _make_manager()
        stop = manager.stop_price_for(entry_price=100.0)
        # Default stop_loss_pct = 0.02 → stop = 98.0
        assert stop == 98.0

    def test_stop_price_custom_pct(self):
        config  = {"execution": {"stop_loss_pct": 0.05}}
        manager = _make_manager(config=config)
        stop    = manager.stop_price_for(100.0)
        assert stop == 95.0

    def test_config_max_position_pct(self):
        config  = {"execution": {"max_position_pct": 0.20}}
        manager = _make_manager(config=config)
        # 20% of $50k = $10k. At $100 → 100 shares
        qty = manager.calculate_qty(price=100.0, portfolio_value=50_000)
        assert qty == 100


# ─────────────────────────────────────────────────────────────────────────────
# OrderManager — Deduplication Checks
# ─────────────────────────────────────────────────────────────────────────────

class TestOrderManagerDedup:

    def test_has_position_false_when_none(self):
        manager = _make_manager()
        assert manager.has_position("AAPL") is False

    def test_has_position_true_when_held(self):
        api = _mock_alpaca_api()
        pos = MagicMock()
        pos.symbol          = "AAPL"
        pos.qty             = "5"
        pos.avg_entry_price = "150"
        pos.market_value    = "775"
        pos.unrealized_pl   = "25"
        pos.unrealized_plpc = "0.033"
        pos.current_price   = "155"
        pos.side            = "long"
        api.get_position.side_effect = None
        api.get_position.return_value = pos
        manager = _make_manager(api_mock=api)
        assert manager.has_position("AAPL") is True

    def test_has_pending_order_false(self):
        manager = _make_manager()
        assert manager.has_pending_order("AAPL") is False

    def test_has_pending_order_true(self):
        api   = _mock_alpaca_api()
        order = MagicMock()
        order.id                = "oid-pending"
        order.client_order_id   = "coid"
        order.symbol            = "AAPL"
        order.side              = "buy"
        order.qty               = "5"
        order.filled_qty        = "0"
        order.type              = "market"
        order.status            = "pending_new"
        order.filled_avg_price  = None
        order.submitted_at      = "2024-01-02"
        api.list_orders.return_value = [order]
        manager = _make_manager(api_mock=api)
        assert manager.has_pending_order("AAPL") is True


# ─────────────────────────────────────────────────────────────────────────────
# OrderManager — execute_signal
# ─────────────────────────────────────────────────────────────────────────────

class TestOrderManagerExecuteSignal:

    def test_hold_signal_returns_hold(self):
        manager = _make_manager()
        result  = manager.execute_signal("AAPL", signal=0, price=150.0)
        assert result["action"] == "HOLD"

    def test_buy_signal_no_position_places_order(self):
        manager = _make_manager(portfolio_value=50_000, buying_power=30_000)
        result  = manager.execute_signal("AAPL", signal=1, price=150.0)
        assert result["action"]    == "BUY"
        assert result["qty"]       >= 1
        assert result["order_id"]  == "mock-order-id-001"
        assert result["stop_price"] < 150.0

    def test_buy_signal_already_in_position_skips(self):
        api = _mock_alpaca_api()
        pos = MagicMock()
        pos.symbol          = "AAPL"
        pos.qty             = "5"
        pos.avg_entry_price = "140"
        pos.market_value    = "750"
        pos.unrealized_pl   = "50"
        pos.unrealized_plpc = "0.07"
        pos.current_price   = "150"
        pos.side            = "long"
        api.get_position.side_effect  = None
        api.get_position.return_value = pos
        manager = _make_manager(api_mock=api)
        result  = manager.execute_signal("AAPL", signal=1, price=150.0)
        assert result["action"] == "SKIP_BUY"
        assert result["reason"] == "already_in_position"

    def test_buy_signal_insufficient_buying_power_skips(self):
        # 10% of $50k = $5k. $150 * 33 shares = $4,950. But buying power only $10
        manager = _make_manager(portfolio_value=50_000, buying_power=10)
        result  = manager.execute_signal("AAPL", signal=1, price=150.0)
        assert result["action"] == "SKIP_BUY"
        assert result["reason"] == "insufficient_buying_power"

    def test_buy_result_includes_est_cost(self):
        manager = _make_manager(portfolio_value=50_000, buying_power=30_000)
        result  = manager.execute_signal("AAPL", signal=1, price=100.0)
        assert "est_cost" in result
        assert result["est_cost"] == result["qty"] * 100.0

    def test_sell_signal_with_position_places_order(self):
        api = _mock_alpaca_api()
        pos = MagicMock()
        pos.symbol          = "AAPL"
        pos.qty             = "5"
        pos.avg_entry_price = "140"
        pos.market_value    = "750"
        pos.unrealized_pl   = "50"
        pos.unrealized_plpc = "0.07"
        pos.current_price   = "150"
        pos.side            = "long"
        api.get_position.side_effect  = None
        api.get_position.return_value = pos
        sell_order = MagicMock()
        sell_order.id               = "sell-order-001"
        sell_order.client_order_id  = "coid-sell"
        sell_order.symbol           = "AAPL"
        sell_order.side             = "sell"
        sell_order.qty              = "5"
        sell_order.filled_qty       = "0"
        sell_order.type             = "market"
        sell_order.status           = "accepted"
        sell_order.filled_avg_price = None
        sell_order.submitted_at     = "2024-01-02"
        api.submit_order.return_value = sell_order
        manager = _make_manager(api_mock=api)
        result  = manager.execute_signal("AAPL", signal=-1, price=150.0)
        assert result["action"]   == "SELL"
        assert result["qty"]      == 5
        assert result["order_id"] == "sell-order-001"

    def test_sell_signal_no_position_skips(self):
        manager = _make_manager()
        result  = manager.execute_signal("AAPL", signal=-1, price=150.0)
        assert result["action"] == "SKIP_SELL"
        assert result["reason"] == "no_position"

    def test_sell_result_includes_pl_pct(self):
        api = _mock_alpaca_api()
        pos = MagicMock()
        pos.symbol          = "AAPL"
        pos.qty             = "10"
        pos.avg_entry_price = "100"   # entry at 100
        pos.market_value    = "1500"
        pos.unrealized_pl   = "500"
        pos.unrealized_plpc = "0.5"
        pos.current_price   = "150"
        pos.side            = "long"
        api.get_position.side_effect  = None
        api.get_position.return_value = pos
        manager = _make_manager(api_mock=api)
        result  = manager.execute_signal("AAPL", signal=-1, price=150.0)
        assert "est_pl_pct" in result
        assert result["est_pl_pct"] == pytest.approx(50.0, abs=0.1)

    def test_result_always_has_action_key(self):
        manager = _make_manager()
        for signal in [1, -1, 0]:
            result = manager.execute_signal("AAPL", signal=signal, price=100.0)
            assert "action" in result


# ─────────────────────────────────────────────────────────────────────────────
# OrderManager — close_all_positions
# ─────────────────────────────────────────────────────────────────────────────

class TestCloseAllPositions:

    def test_close_all_empty_portfolio(self):
        manager = _make_manager()
        result  = manager.close_all_positions()
        assert result == []

    def test_close_all_returns_sell_results(self):
        api = _mock_alpaca_api()

        pos = MagicMock()
        pos.symbol          = "AAPL"
        pos.qty             = "5"
        pos.avg_entry_price = "140"
        pos.market_value    = "750"
        pos.unrealized_pl   = "50"
        pos.unrealized_plpc = "0.07"
        pos.current_price   = "150"
        pos.side            = "long"
        api.list_positions.return_value = [pos]
        api.get_position.side_effect    = None
        api.get_position.return_value   = pos

        sell_order = MagicMock()
        sell_order.id               = "sell-001"
        sell_order.client_order_id  = "coid"
        sell_order.symbol           = "AAPL"
        sell_order.side             = "sell"
        sell_order.qty              = "5"
        sell_order.filled_qty       = "0"
        sell_order.type             = "market"
        sell_order.status           = "accepted"
        sell_order.filled_avg_price = None
        sell_order.submitted_at     = "2024-01-02"
        api.submit_order.return_value = sell_order

        manager = _make_manager(api_mock=api)
        results = manager.close_all_positions()

        assert len(results) == 1
        assert results[0]["action"] == "SELL"
