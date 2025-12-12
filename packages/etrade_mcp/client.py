# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
eTrade API Client Module.

High-level wrapper for eTrade REST API operations including:
- Account management
- Market data (quotes, option chains)
- Order management (preview, place, cancel)

All API calls handle:
- OAuth authentication via ETradeAuthManager
- Rate limiting (2 requests/second)
- Error handling and retries
- Response parsing into Pydantic models
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode

import requests
from loguru import logger

from etrade_mcp.auth import ETradeAuthManager
from etrade_mcp.models import (
    Account,
    AccountBalance,
    APIResponse,
    Option,
    OptionChain,
    OptionExpiration,
    OptionQuote,
    Order,
    OrderAction,
    OrderDuration,
    OrderLeg,
    OrderPreview,
    OrderRequest,
    OrderType,
    Position,
    Quote,
    SecurityType,
    SpreadLeg,
    SpreadOrderRequest,
)


# =============================================================================
# CONSTANTS
# =============================================================================

# API versions
ACCOUNT_API_VERSION = "v1"
MARKET_API_VERSION = "v1"
ORDER_API_VERSION = "v1"

# Rate limiting
MIN_REQUEST_INTERVAL = 0.5  # 2 requests/second max

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1.0


# =============================================================================
# ETRADE CLIENT CLASS
# =============================================================================


class ETradeClient:
    """
    eTrade API client with full trading functionality.

    Features:
    - Account information and balances
    - Position tracking
    - Real-time quotes
    - Option chains and expiration dates
    - Order preview, placement, and cancellation
    - Multi-leg spread orders

    Usage:
        auth = ETradeAuthManager()
        client = ETradeClient(auth)

        # Get accounts
        accounts = client.get_accounts()

        # Get quote
        quote = client.get_quote("AAPL")

        # Place order
        order = client.place_order(account_id_key, order_request)
    """

    def __init__(self, auth_manager: ETradeAuthManager):
        """
        Initialize the client.

        Args:
            auth_manager: Authenticated ETradeAuthManager instance
        """
        self.auth = auth_manager
        self._last_request_time = 0.0
        self._session = requests.Session()

    @property
    def base_url(self) -> str:
        """Get base URL from auth manager."""
        return self.auth.base_url

    def _rate_limit(self) -> None:
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        retries: int = MAX_RETRIES,
    ) -> Dict[str, Any]:
        """
        Make an authenticated API request.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (without base URL)
            params: Query parameters
            json_data: JSON body for POST/PUT
            retries: Number of retries on failure

        Returns:
            Parsed JSON response

        Raises:
            requests.exceptions.RequestException: On API error
        """
        self._rate_limit()

        # Build full URL
        url = f"{self.base_url}{endpoint}"
        if params:
            url = f"{url}?{urlencode(params)}"

        # Get OAuth headers
        headers = self.auth.get_auth_headers(method, url.split("?")[0])
        headers["Accept"] = "application/json"

        if json_data:
            headers["Content-Type"] = "application/json"

        last_error = None
        for attempt in range(retries):
            try:
                response = self._session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json_data,
                    timeout=30,
                )

                # Handle specific error codes
                if response.status_code == 401:
                    logger.warning("401 Unauthorized - token may be expired")
                    if self.auth.refresh_token():
                        # Retry with refreshed token
                        headers = self.auth.get_auth_headers(method, url.split("?")[0])
                        continue
                    raise ValueError("Authentication failed - please re-authorize")

                if response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = float(
                        response.headers.get("Retry-After", RETRY_DELAY * 2)
                    )
                    logger.warning(f"Rate limited - waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()

                # Parse response
                if response.content:
                    return response.json()
                return {}

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < retries - 1:
                    logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"Request failed after {retries} attempts: {e}")
                    raise

        raise last_error or Exception("Request failed")

    # =========================================================================
    # ACCOUNT METHODS
    # =========================================================================

    def get_accounts(self) -> List[Account]:
        """
        Get list of accounts for the authenticated user.

        Returns:
            List of Account objects
        """
        endpoint = f"/{ACCOUNT_API_VERSION}/accounts/list"

        try:
            response = self._request("GET", endpoint)

            accounts = []
            accounts_data = (
                response.get("AccountListResponse", {})
                .get("Accounts", {})
                .get("Account", [])
            )

            if isinstance(accounts_data, dict):
                accounts_data = [accounts_data]

            for acc in accounts_data:
                accounts.append(
                    Account(
                        accountId=acc.get("accountId", ""),
                        accountIdKey=acc.get("accountIdKey", ""),
                        accountName=acc.get("accountName"),
                        accountType=acc.get("accountType"),
                        accountDesc=acc.get("accountDesc"),
                        institutionType=acc.get("institutionType"),
                    )
                )

            logger.info(f"Found {len(accounts)} accounts")
            return accounts

        except Exception as e:
            logger.error(f"Failed to get accounts: {e}")
            raise

    def get_account_balance(self, account_id_key: str) -> AccountBalance:
        """
        Get account balance and buying power.

        Args:
            account_id_key: Account identifier key

        Returns:
            AccountBalance object
        """
        endpoint = f"/{ACCOUNT_API_VERSION}/accounts/{account_id_key}/balance"
        params = {"instType": "BROKERAGE", "realTimeNAV": "true"}

        try:
            response = self._request("GET", endpoint, params=params)

            balance_data = response.get("BalanceResponse", {})
            computed = balance_data.get("Computed", {})

            return AccountBalance(
                accountId=balance_data.get("accountId", ""),
                totalAccountValue=computed.get("RealTimeValues", {}).get(
                    "totalAccountValue", 0
                ),
                netCash=computed.get("netCash", 0),
                cashAvailableForInvestment=computed.get(
                    "cashAvailableForInvestment", 0
                ),
                cashAvailableForWithdrawal=computed.get(
                    "cashAvailableForWithdrawal", 0
                ),
                marginBuyingPower=computed.get("marginBuyingPower", 0),
                optionBuyingPower=balance_data.get("Cash", {}).get(
                    "fundsForOpenOrdersCash", 0
                ),
                settledCash=computed.get("settledCash", 0),
                unsettledCash=computed.get("unsettledCash", 0),
            )

        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            raise

    def get_positions(
        self,
        account_id_key: str,
        symbol: Optional[str] = None,
    ) -> List[Position]:
        """
        Get positions for an account.

        Args:
            account_id_key: Account identifier key
            symbol: Optional symbol filter

        Returns:
            List of Position objects
        """
        endpoint = f"/{ACCOUNT_API_VERSION}/accounts/{account_id_key}/portfolio"
        params = {}
        if symbol:
            params["symbols"] = symbol

        try:
            response = self._request("GET", endpoint, params=params if params else None)

            positions = []
            portfolio_data = response.get("PortfolioResponse", {}).get(
                "AccountPortfolio", []
            )

            if isinstance(portfolio_data, dict):
                portfolio_data = [portfolio_data]

            for portfolio in portfolio_data:
                position_list = portfolio.get("Position", [])
                if isinstance(position_list, dict):
                    position_list = [position_list]

                for pos in position_list:
                    product = pos.get("Product", {})
                    quick = pos.get("Quick", {})

                    positions.append(
                        Position(
                            symbol=product.get("symbol", ""),
                            symbolDescription=pos.get("symbolDescription"),
                            securityType=product.get("securityType"),
                            quantity=pos.get("quantity", 0),
                            costBasis=pos.get("costPerShare", 0)
                            * pos.get("quantity", 0),
                            marketValue=quick.get("marketValue", 0),
                            currentPrice=quick.get("lastTrade", 0),
                            todayGainLoss=quick.get("change", 0)
                            * pos.get("quantity", 0),
                            totalGainLoss=quick.get("gainLoss", 0),
                            totalGainLossPct=quick.get("gainLossPct", 0),
                            pctOfPortfolio=pos.get("pctOfPortfolio", 0),
                            optionType=product.get("callPut"),
                            strikePrice=product.get("strikePrice"),
                            expirationDate=(
                                str(product.get("expiryYear", ""))
                                + "-"
                                + str(product.get("expiryMonth", "")).zfill(2)
                                + "-"
                                + str(product.get("expiryDay", "")).zfill(2)
                                if product.get("expiryYear")
                                else None
                            ),
                            underlyingSymbol=(
                                product.get("symbol")
                                if product.get("securityType") == "OPTN"
                                else None
                            ),
                        )
                    )

            logger.info(f"Found {len(positions)} positions")
            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise

    # =========================================================================
    # MARKET DATA METHODS
    # =========================================================================

    def get_quote(self, symbols: Union[str, List[str]]) -> List[Quote]:
        """
        Get real-time quotes for symbols.

        Args:
            symbols: Single symbol or list of symbols (max 25)

        Returns:
            List of Quote objects
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        if len(symbols) > 25:
            logger.warning("Max 25 symbols per request - truncating")
            symbols = symbols[:25]

        endpoint = f"/{MARKET_API_VERSION}/market/quote/{','.join(symbols)}"

        try:
            response = self._request("GET", endpoint)

            quotes = []
            quote_data = response.get("QuoteResponse", {}).get("QuoteData", [])

            if isinstance(quote_data, dict):
                quote_data = [quote_data]

            for q in quote_data:
                all_data = q.get("All", {})
                product = q.get("Product", {})

                quotes.append(
                    Quote(
                        symbol=product.get("symbol", ""),
                        lastPrice=all_data.get("lastTrade", 0),
                        change=all_data.get("changeClose", 0),
                        changePct=all_data.get("changeClosePercentage", 0),
                        bid=all_data.get("bid", 0),
                        ask=all_data.get("ask", 0),
                        bidSize=all_data.get("bidSize", 0),
                        askSize=all_data.get("askSize", 0),
                        volume=all_data.get("totalVolume", 0),
                        high=all_data.get("high", 0),
                        low=all_data.get("low", 0),
                        open=all_data.get("open", 0),
                        close=all_data.get("previousClose", 0),
                        week52High=all_data.get("high52", 0),
                        week52Low=all_data.get("low52", 0),
                        marketCap=all_data.get("marketCap"),
                        peRatio=all_data.get("pe"),
                        eps=all_data.get("eps"),
                        divYield=all_data.get("dividend"),
                        timestamp=q.get("dateTime"),
                    )
                )

            logger.info(f"Got quotes for {len(quotes)} symbols")
            return quotes

        except Exception as e:
            logger.error(f"Failed to get quotes: {e}")
            raise

    def get_option_expiry_dates(self, symbol: str) -> List[OptionExpiration]:
        """
        Get available option expiration dates for a symbol.

        Args:
            symbol: Underlying symbol

        Returns:
            List of OptionExpiration objects
        """
        endpoint = f"/{MARKET_API_VERSION}/market/optionexpiredate"
        params = {"symbol": symbol}

        try:
            response = self._request("GET", endpoint, params=params)

            expirations = []
            expiry_data = response.get("OptionExpireDateResponse", {}).get(
                "ExpirationDate", []
            )

            if isinstance(expiry_data, dict):
                expiry_data = [expiry_data]

            today = datetime.now().date()

            for exp in expiry_data:
                year = exp.get("year", 0)
                month = exp.get("month", 0)
                day = exp.get("day", 0)

                if year and month and day:
                    exp_date = datetime(year, month, day).date()
                    days_to_exp = (exp_date - today).days

                    expirations.append(
                        OptionExpiration(
                            expirationDate=exp_date.isoformat(),
                            daysToExpiration=days_to_exp,
                            expirationType=exp.get("expiryType", "STANDARD"),
                        )
                    )

            logger.info(f"Found {len(expirations)} expiration dates for {symbol}")
            return expirations

        except Exception as e:
            logger.error(f"Failed to get option expiry dates: {e}")
            raise

    def get_option_chains(
        self,
        symbol: str,
        expiration_date: Optional[str] = None,
        strike_price_near: Optional[float] = None,
        no_of_strikes: int = 10,
        option_type: Optional[str] = None,  # "CALL" or "PUT"
        include_weekly: bool = True,
    ) -> OptionChain:
        """
        Get option chain for a symbol.

        Args:
            symbol: Underlying symbol
            expiration_date: Specific expiration (YYYY-MM-DD) or None for nearest
            strike_price_near: Center strikes around this price
            no_of_strikes: Number of strikes to return
            option_type: Filter by CALL or PUT
            include_weekly: Include weekly options

        Returns:
            OptionChain with calls and puts
        """
        endpoint = f"/{MARKET_API_VERSION}/market/optionchains"
        params = {
            "symbol": symbol,
            "noOfStrikes": no_of_strikes,
            "includeWeekly": str(include_weekly).lower(),
            "priceType": "ALL",
        }

        if expiration_date:
            # Parse date
            exp_dt = datetime.fromisoformat(expiration_date)
            params["expiryYear"] = exp_dt.year
            params["expiryMonth"] = exp_dt.month
            params["expiryDay"] = exp_dt.day

        if strike_price_near:
            params["strikePriceNear"] = strike_price_near

        if option_type:
            params["optionCategory"] = option_type

        try:
            response = self._request("GET", endpoint, params=params)

            chain_data = response.get("OptionChainResponse", {})
            option_pairs = chain_data.get("OptionPair", [])

            if isinstance(option_pairs, dict):
                option_pairs = [option_pairs]

            calls = []
            puts = []
            chain_exp_date = expiration_date or ""

            for pair in option_pairs:
                # Process calls
                call_data = pair.get("Call")
                if call_data:
                    call = self._parse_option_quote(call_data, "CALL")
                    if call:
                        calls.append(call)
                        chain_exp_date = chain_exp_date or call.expiration_date

                # Process puts
                put_data = pair.get("Put")
                if put_data:
                    put = self._parse_option_quote(put_data, "PUT")
                    if put:
                        puts.append(put)
                        chain_exp_date = chain_exp_date or put.expiration_date

            logger.info(
                f"Got option chain for {symbol}: {len(calls)} calls, {len(puts)} puts"
            )

            return OptionChain(
                symbol=symbol,
                expirationDate=chain_exp_date,
                calls=calls,
                puts=puts,
            )

        except Exception as e:
            logger.error(f"Failed to get option chain: {e}")
            raise

    def _parse_option_quote(
        self, data: Dict[str, Any], opt_type: str
    ) -> Optional[OptionQuote]:
        """Parse option quote from API response."""
        try:
            exp_year = data.get("expirationYear", 0)
            exp_month = data.get("expirationMonth", 0)
            exp_day = data.get("expirationDay", 0)

            exp_date = ""
            if exp_year and exp_month and exp_day:
                exp_date = (
                    f"{exp_year}-{str(exp_month).zfill(2)}-{str(exp_day).zfill(2)}"
                )

            return OptionQuote(
                symbol=data.get("symbol", ""),
                rootSymbol=data.get("rootSymbol", ""),
                optionType=opt_type,
                strikePrice=data.get("strikePrice", 0),
                expirationDate=exp_date,
                bid=data.get("bid", 0),
                ask=data.get("ask", 0),
                lastPrice=data.get("lastPrice", 0),
                volume=data.get("volume", 0),
                openInterest=data.get("openInterest", 0),
                impliedVolatility=data.get("iv", 0),
                delta=data.get("OptionGreeks", {}).get("delta"),
                gamma=data.get("OptionGreeks", {}).get("gamma"),
                theta=data.get("OptionGreeks", {}).get("theta"),
                vega=data.get("OptionGreeks", {}).get("vega"),
                rho=data.get("OptionGreeks", {}).get("rho"),
                inTheMoney=data.get("inTheMoney", "n") == "y",
            )
        except Exception as e:
            logger.warning(f"Failed to parse option quote: {e}")
            return None

    # =========================================================================
    # ORDER METHODS
    # =========================================================================

    def preview_order(
        self,
        account_id_key: str,
        order_request: OrderRequest,
    ) -> OrderPreview:
        """
        Preview an order before placement.

        Args:
            account_id_key: Account identifier key
            order_request: Order details

        Returns:
            OrderPreview with estimated costs
        """
        endpoint = f"/{ORDER_API_VERSION}/accounts/{account_id_key}/orders/preview"

        # Build order payload
        payload = self._build_order_payload(order_request, preview=True)

        try:
            response = self._request("POST", endpoint, json_data=payload)

            preview_data = response.get("PreviewOrderResponse", {})

            return OrderPreview(
                previewId=str(
                    preview_data.get("PreviewIds", [{}])[0].get("previewId", "")
                ),
                estimatedCommission=preview_data.get("Order", [{}])[0].get(
                    "estimatedCommission", 0
                ),
                estimatedTotalAmount=preview_data.get("Order", [{}])[0].get(
                    "estimatedTotalAmount", 0
                ),
                orderValue=preview_data.get("Order", [{}])[0].get("orderValue", 0),
                messages=[
                    msg.get("description", "")
                    for msg in preview_data.get("Message", [])
                ],
            )

        except Exception as e:
            logger.error(f"Failed to preview order: {e}")
            raise

    def place_order(
        self,
        account_id_key: str,
        order_request: OrderRequest,
        preview_id: Optional[str] = None,
    ) -> Order:
        """
        Place an order.

        Args:
            account_id_key: Account identifier key
            order_request: Order details
            preview_id: Preview ID from preview_order (recommended)

        Returns:
            Order object with order ID
        """
        endpoint = f"/{ORDER_API_VERSION}/accounts/{account_id_key}/orders/place"

        # Build order payload
        payload = self._build_order_payload(order_request, preview_id=preview_id)

        try:
            response = self._request("POST", endpoint, json_data=payload)

            order_data = response.get("PlaceOrderResponse", {})
            order_info = order_data.get("Order", [{}])[0]

            return Order(
                orderId=str(order_data.get("OrderIds", [{}])[0].get("orderId", "")),
                accountId=account_id_key,
                status=order_info.get("status", "OPEN"),
                orderType=order_request.order_type,
                priceType=order_request.price_type,
                limitPrice=order_request.limit_price,
                stopPrice=order_request.stop_price,
                orderTerm=order_request.order_term.value,
                placedTime=order_info.get("placedTime"),
                quantityOrdered=sum(leg.quantity for leg in order_request.legs),
                legs=order_request.legs,
                messages=[
                    msg.get("description", "") for msg in order_data.get("Message", [])
                ],
            )

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise

    def place_spread_order(
        self,
        account_id_key: str,
        spread_request: SpreadOrderRequest,
        preview_id: Optional[str] = None,
    ) -> Order:
        """
        Place a multi-leg spread order.

        Args:
            account_id_key: Account identifier key
            spread_request: Spread order details
            preview_id: Preview ID from preview

        Returns:
            Order object
        """
        endpoint = f"/{ORDER_API_VERSION}/accounts/{account_id_key}/orders/place"

        # Build spread payload
        payload = self._build_spread_payload(spread_request, preview_id=preview_id)

        try:
            response = self._request("POST", endpoint, json_data=payload)

            order_data = response.get("PlaceOrderResponse", {})

            return Order(
                orderId=str(order_data.get("OrderIds", [{}])[0].get("orderId", "")),
                accountId=account_id_key,
                status="OPEN",
                orderType=spread_request.order_type,
                priceType=(
                    "NET_DEBIT"
                    if spread_request.limit_price and spread_request.limit_price > 0
                    else "NET_CREDIT"
                ),
                limitPrice=spread_request.limit_price,
                orderTerm=spread_request.order_term.value,
                placedTime=datetime.now().isoformat(),
                legs=[
                    OrderLeg(
                        symbol=leg.symbol,
                        securityType=SecurityType.OPTN,
                        orderAction=leg.order_action,
                        quantity=leg.quantity,
                        optionType=leg.option_type,
                        strikePrice=leg.strike_price,
                        expirationDate=leg.expiration_date,
                    )
                    for leg in spread_request.legs
                ],
            )

        except Exception as e:
            logger.error(f"Failed to place spread order: {e}")
            raise

    def cancel_order(
        self,
        account_id_key: str,
        order_id: str,
    ) -> bool:
        """
        Cancel an open order.

        Args:
            account_id_key: Account identifier key
            order_id: Order ID to cancel

        Returns:
            True if successful
        """
        endpoint = f"/{ORDER_API_VERSION}/accounts/{account_id_key}/orders/cancel"

        payload = {
            "CancelOrderRequest": {
                "orderId": order_id,
            }
        }

        try:
            response = self._request("PUT", endpoint, json_data=payload)

            cancel_data = response.get("CancelOrderResponse", {})
            logger.info(f"Order {order_id} cancelled")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise

    def get_orders(
        self,
        account_id_key: str,
        status: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> List[Order]:
        """
        Get orders for an account.

        Args:
            account_id_key: Account identifier key
            status: Filter by status (OPEN, EXECUTED, etc.)
            from_date: Start date (MMDDYYYY)
            to_date: End date (MMDDYYYY)

        Returns:
            List of Order objects
        """
        endpoint = f"/{ORDER_API_VERSION}/accounts/{account_id_key}/orders"
        params = {}

        if status:
            params["status"] = status
        if from_date:
            params["fromDate"] = from_date
        if to_date:
            params["toDate"] = to_date

        try:
            response = self._request("GET", endpoint, params=params if params else None)

            orders = []
            orders_data = response.get("OrdersResponse", {}).get("Order", [])

            if isinstance(orders_data, dict):
                orders_data = [orders_data]

            for order_data in orders_data:
                order_detail = order_data.get("OrderDetail", [{}])[0]

                legs = []
                for instrument in order_detail.get("Instrument", []):
                    product = instrument.get("Product", {})
                    legs.append(
                        OrderLeg(
                            symbol=product.get("symbol", ""),
                            securityType=SecurityType(
                                product.get("securityType", "EQ")
                            ),
                            orderAction=OrderAction(
                                instrument.get("orderAction", "BUY")
                            ),
                            quantity=instrument.get("orderedQuantity", 0),
                            optionType=product.get("callPut"),
                            strikePrice=product.get("strikePrice"),
                            expirationDate=(
                                f"{product.get('expiryYear', '')}-{str(product.get('expiryMonth', '')).zfill(2)}-{str(product.get('expiryDay', '')).zfill(2)}"
                                if product.get("expiryYear")
                                else None
                            ),
                        )
                    )

                orders.append(
                    Order(
                        orderId=str(order_data.get("orderId", "")),
                        accountId=account_id_key,
                        status=OrderStatus(order_detail.get("status", "OPEN")),
                        orderType=OrderType(order_detail.get("orderType", "LIMIT")),
                        priceType=order_detail.get("priceType", "LIMIT"),
                        limitPrice=order_detail.get("limitPrice"),
                        stopPrice=order_detail.get("stopPrice"),
                        orderTerm=order_detail.get("orderTerm", "DAY"),
                        placedTime=order_detail.get("placedTime"),
                        executedTime=order_detail.get("executedTime"),
                        quantityOrdered=order_detail.get("orderedQuantity", 0),
                        quantityFilled=order_detail.get("filledQuantity", 0),
                        executionPrice=order_detail.get("executionPrice"),
                        legs=legs,
                    )
                )

            logger.info(f"Found {len(orders)} orders")
            return orders

        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            raise

    def _build_order_payload(
        self,
        order: OrderRequest,
        preview: bool = False,
        preview_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build order payload for API request."""
        instruments = []

        for leg in order.legs:
            instrument = {
                "Product": {
                    "symbol": leg.symbol,
                    "securityType": leg.security_type.value,
                },
                "orderAction": leg.order_action.value,
                "orderedQuantity": leg.quantity,
                "quantityType": "QUANTITY",
            }

            # Add option details if applicable
            if leg.security_type == SecurityType.OPTN and leg.option_type:
                instrument["Product"]["callPut"] = leg.option_type.value

                if leg.strike_price:
                    instrument["Product"]["strikePrice"] = leg.strike_price

                if leg.expiration_date:
                    exp_dt = datetime.fromisoformat(leg.expiration_date)
                    instrument["Product"]["expiryYear"] = exp_dt.year
                    instrument["Product"]["expiryMonth"] = exp_dt.month
                    instrument["Product"]["expiryDay"] = exp_dt.day

            instruments.append(instrument)

        order_detail = {
            "orderTerm": order.order_term.value,
            "marketSession": order.market_session.value,
            "priceType": order.price_type,
            "Instrument": instruments,
        }

        if order.limit_price:
            order_detail["limitPrice"] = order.limit_price
        if order.stop_price:
            order_detail["stopPrice"] = order.stop_price
        if order.all_or_none:
            order_detail["allOrNone"] = True

        request_key = "PreviewOrderRequest" if preview else "PlaceOrderRequest"
        payload = {
            request_key: {
                "orderType": order.order_type.value,
                "clientOrderId": f"qp_{int(time.time() * 1000)}",
                "Order": [order_detail],
            }
        }

        if preview_id:
            payload[request_key]["PreviewIds"] = [{"previewId": preview_id}]

        return payload

    def _build_spread_payload(
        self,
        spread: SpreadOrderRequest,
        preview_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build spread order payload for API request."""
        instruments = []

        for leg in spread.legs:
            exp_dt = datetime.fromisoformat(leg.expiration_date)

            instrument = {
                "Product": {
                    "symbol": spread.underlying_symbol,
                    "securityType": "OPTN",
                    "callPut": leg.option_type.value,
                    "strikePrice": leg.strike_price,
                    "expiryYear": exp_dt.year,
                    "expiryMonth": exp_dt.month,
                    "expiryDay": exp_dt.day,
                },
                "orderAction": leg.order_action.value,
                "orderedQuantity": leg.quantity,
                "quantityType": "QUANTITY",
            }
            instruments.append(instrument)

        order_detail = {
            "orderTerm": spread.order_term.value,
            "marketSession": spread.market_session.value,
            "priceType": (
                "NET_DEBIT"
                if spread.limit_price and spread.limit_price > 0
                else "NET_CREDIT"
            ),
            "Instrument": instruments,
        }

        if spread.limit_price:
            order_detail["limitPrice"] = abs(spread.limit_price)

        payload = {
            "PlaceOrderRequest": {
                "orderType": spread.order_type.value,
                "clientOrderId": f"qp_spread_{int(time.time() * 1000)}",
                "Order": [order_detail],
            }
        }

        if preview_id:
            payload["PlaceOrderRequest"]["PreviewIds"] = [{"previewId": preview_id}]

        return payload
