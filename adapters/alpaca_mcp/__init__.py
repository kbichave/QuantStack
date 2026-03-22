"""
alpaca_mcp — FastMCP server exposing Alpaca brokerage tools to Claude.

Entry point: ``alpaca-mcp`` (configured in pyproject.toml scripts).

Tools (11):
    get_account         Account info and buying power
    get_balance         Cash, buying power, portfolio value
    get_positions       Open positions with unrealised P&L
    get_quote           Real-time quotes (up to 50 symbols)
    get_bars            Historical OHLCV (delegates to AlpacaAdapter)
    preview_order       Estimate order cost and commission
    place_order         Market or limit equity order
    cancel_order        Cancel an open order by ID
    get_orders          Order history with optional status filter
    get_option_chains   Options chain snapshot (requires options endpoint)
    get_auth_status     Validate that API keys are active
"""
