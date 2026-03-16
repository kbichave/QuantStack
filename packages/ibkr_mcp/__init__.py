"""
ibkr_mcp — FastMCP server exposing Interactive Brokers tools to Claude.

Entry point: ``ibkr-mcp`` (configured in pyproject.toml scripts).

Requires IB Gateway (port 4001) or TWS (port 7497) to be running locally.
If the gateway is offline at startup, the server starts in a degraded state
and all tools return {\"success\": false, \"error\": \"IB Gateway not connected\"}.

Tools (11):
    connect_gateway         Explicitly connect to IB Gateway
    get_connection_status   Check gateway connectivity
    get_accounts            List accessible accounts
    get_balance             Account balance and margin
    get_positions           Current positions
    get_quote               Real-time snapshot prices
    get_historical_bars     OHLCV history from reqHistoricalData
    get_option_chains       Options chain snapshot
    place_order             Market / limit / stop equity order
    cancel_order            Cancel an open order
    get_orders              Open and filled order history
"""
