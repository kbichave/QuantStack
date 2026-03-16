# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Pydantic models for eTrade API responses.

These models provide type-safe representations of eTrade API data structures.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# ENUMS
# =============================================================================


class OrderAction(str, Enum):
    """Order action type."""

    BUY = "BUY"
    SELL = "SELL"
    BUY_TO_OPEN = "BUY_TO_OPEN"
    BUY_TO_CLOSE = "BUY_TO_CLOSE"
    SELL_TO_OPEN = "SELL_TO_OPEN"
    SELL_TO_CLOSE = "SELL_TO_CLOSE"


class OrderType(str, Enum):
    """Order type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderDuration(str, Enum):
    """Order time in force."""

    DAY = "DAY"
    GTC = "GOOD_TILL_CANCEL"
    GTD = "GOOD_TILL_DATE"
    IOC = "IMMEDIATE_OR_CANCEL"
    FOK = "FILL_OR_KILL"


class OrderStatus(str, Enum):
    """Order status."""

    OPEN = "OPEN"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"
    PENDING = "PENDING"
    REJECTED = "REJECTED"
    PARTIAL = "PARTIAL"
    EXPIRED = "EXPIRED"


class OptionType(str, Enum):
    """Option contract type."""

    CALL = "CALL"
    PUT = "PUT"


class SecurityType(str, Enum):
    """Security type."""

    EQ = "EQ"  # Equity
    OPTN = "OPTN"  # Option
    MF = "MF"  # Mutual Fund
    BOND = "BOND"
    ETF = "ETF"


class AccountType(str, Enum):
    """Account type."""

    INDIVIDUAL = "INDIVIDUAL"
    JOINT = "JOINT"
    IRA = "IRA"
    TRUST = "TRUST"
    CORPORATE = "CORPORATE"


class MarketSession(str, Enum):
    """Market session."""

    REGULAR = "REGULAR"
    EXTENDED = "EXTENDED"


# =============================================================================
# ACCOUNT MODELS
# =============================================================================


class Account(BaseModel):
    """eTrade account information."""

    account_id: str = Field(..., alias="accountId")
    account_id_key: str = Field(..., alias="accountIdKey")
    account_name: Optional[str] = Field(None, alias="accountName")
    account_type: Optional[str] = Field(None, alias="accountType")
    account_desc: Optional[str] = Field(None, alias="accountDesc")
    institution_type: Optional[str] = Field(None, alias="institutionType")
    closed_date: Optional[str] = Field(None, alias="closedDate")

    class Config:
        populate_by_name = True


class AccountBalance(BaseModel):
    """Account balance information."""

    account_id: str = Field(..., alias="accountId")
    total_account_value: float = Field(0.0, alias="totalAccountValue")
    net_cash: float = Field(0.0, alias="netCash")
    cash_available_for_investment: float = Field(
        0.0, alias="cashAvailableForInvestment"
    )
    cash_available_for_withdrawal: float = Field(
        0.0, alias="cashAvailableForWithdrawal"
    )
    margin_buying_power: float = Field(0.0, alias="marginBuyingPower")
    option_buying_power: float = Field(0.0, alias="optionBuyingPower")
    day_trading_buying_power: float = Field(0.0, alias="dayTradingBuyingPower")
    settled_cash: float = Field(0.0, alias="settledCash")
    unsettled_cash: float = Field(0.0, alias="unsettledCash")

    class Config:
        populate_by_name = True


class Position(BaseModel):
    """Account position."""

    symbol: str
    symbol_description: Optional[str] = Field(None, alias="symbolDescription")
    security_type: Optional[str] = Field(None, alias="securityType")
    quantity: float = 0.0
    quantity_type: Optional[str] = Field(None, alias="quantityType")
    cost_basis: Optional[float] = Field(None, alias="costBasis")
    market_value: Optional[float] = Field(None, alias="marketValue")
    current_price: Optional[float] = Field(None, alias="currentPrice")
    today_gain_loss: Optional[float] = Field(None, alias="todayGainLoss")
    total_gain_loss: Optional[float] = Field(None, alias="totalGainLoss")
    total_gain_loss_pct: Optional[float] = Field(None, alias="totalGainLossPct")
    pct_of_portfolio: Optional[float] = Field(None, alias="pctOfPortfolio")

    # Option-specific fields
    option_type: Optional[str] = Field(None, alias="optionType")
    strike_price: Optional[float] = Field(None, alias="strikePrice")
    expiration_date: Optional[str] = Field(None, alias="expirationDate")
    underlying_symbol: Optional[str] = Field(None, alias="underlyingSymbol")

    class Config:
        populate_by_name = True


# =============================================================================
# MARKET DATA MODELS
# =============================================================================


class Quote(BaseModel):
    """Stock/ETF quote."""

    symbol: str
    last_price: float = Field(0.0, alias="lastPrice")
    change: float = 0.0
    change_pct: float = Field(0.0, alias="changePct")
    bid: float = 0.0
    ask: float = 0.0
    bid_size: int = Field(0, alias="bidSize")
    ask_size: int = Field(0, alias="askSize")
    volume: int = 0
    high: float = 0.0
    low: float = 0.0
    open: float = 0.0
    close: float = 0.0
    week_52_high: Optional[float] = Field(None, alias="week52High")
    week_52_low: Optional[float] = Field(None, alias="week52Low")
    total_volume: Optional[int] = Field(None, alias="totalVolume")
    market_cap: Optional[float] = Field(None, alias="marketCap")
    pe_ratio: Optional[float] = Field(None, alias="peRatio")
    eps: Optional[float] = None
    div_yield: Optional[float] = Field(None, alias="divYield")
    ex_div_date: Optional[str] = Field(None, alias="exDivDate")
    timestamp: Optional[str] = None

    class Config:
        populate_by_name = True


class OptionQuote(BaseModel):
    """Option contract quote."""

    symbol: str
    root_symbol: str = Field(..., alias="rootSymbol")
    option_type: str = Field(..., alias="optionType")  # CALL or PUT
    strike_price: float = Field(..., alias="strikePrice")
    expiration_date: str = Field(..., alias="expirationDate")
    bid: float = 0.0
    ask: float = 0.0
    last_price: float = Field(0.0, alias="lastPrice")
    volume: int = 0
    open_interest: int = Field(0, alias="openInterest")
    implied_volatility: Optional[float] = Field(None, alias="impliedVolatility")
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    in_the_money: bool = Field(False, alias="inTheMoney")

    class Config:
        populate_by_name = True


class OptionExpiration(BaseModel):
    """Option expiration date."""

    expiration_date: str = Field(..., alias="expirationDate")
    days_to_expiration: int = Field(..., alias="daysToExpiration")
    expiration_type: str = Field("STANDARD", alias="expirationType")

    class Config:
        populate_by_name = True


class OptionChain(BaseModel):
    """Option chain for a symbol."""

    symbol: str
    expiration_date: str = Field(..., alias="expirationDate")
    calls: List[OptionQuote] = []
    puts: List[OptionQuote] = []

    class Config:
        populate_by_name = True


# =============================================================================
# ORDER MODELS
# =============================================================================


class OrderLeg(BaseModel):
    """Single leg of an order."""

    symbol: str
    security_type: SecurityType = Field(SecurityType.EQ, alias="securityType")
    order_action: OrderAction = Field(..., alias="orderAction")
    quantity: int

    # Option-specific
    option_type: Optional[OptionType] = Field(None, alias="optionType")
    strike_price: Optional[float] = Field(None, alias="strikePrice")
    expiration_date: Optional[str] = Field(None, alias="expirationDate")

    class Config:
        populate_by_name = True


class OrderRequest(BaseModel):
    """Order request for preview/placement."""

    account_id_key: str = Field(..., alias="accountIdKey")
    order_type: OrderType = Field(OrderType.LIMIT, alias="orderType")
    price_type: str = Field("LIMIT", alias="priceType")
    limit_price: Optional[float] = Field(None, alias="limitPrice")
    stop_price: Optional[float] = Field(None, alias="stopPrice")
    order_term: OrderDuration = Field(OrderDuration.DAY, alias="orderTerm")
    market_session: MarketSession = Field(MarketSession.REGULAR, alias="marketSession")
    all_or_none: bool = Field(False, alias="allOrNone")
    legs: List[OrderLeg] = []

    class Config:
        populate_by_name = True


class OrderPreview(BaseModel):
    """Order preview response."""

    preview_id: str = Field(..., alias="previewId")
    estimated_commission: float = Field(0.0, alias="estimatedCommission")
    estimated_total_amount: float = Field(0.0, alias="estimatedTotalAmount")
    order_value: float = Field(0.0, alias="orderValue")
    margin_requirement: Optional[float] = Field(None, alias="marginRequirement")
    margin_level: Optional[str] = Field(None, alias="marginLevel")
    messages: List[str] = []

    class Config:
        populate_by_name = True


class Order(BaseModel):
    """Placed order information."""

    order_id: str = Field(..., alias="orderId")
    account_id: str = Field(..., alias="accountId")
    status: OrderStatus
    order_type: OrderType = Field(..., alias="orderType")
    price_type: str = Field(..., alias="priceType")
    limit_price: Optional[float] = Field(None, alias="limitPrice")
    stop_price: Optional[float] = Field(None, alias="stopPrice")
    order_term: str = Field(..., alias="orderTerm")
    placed_time: Optional[str] = Field(None, alias="placedTime")
    executed_time: Optional[str] = Field(None, alias="executedTime")
    quantity_ordered: int = Field(0, alias="quantityOrdered")
    quantity_filled: int = Field(0, alias="quantityFilled")
    execution_price: Optional[float] = Field(None, alias="executionPrice")
    commission: Optional[float] = None
    legs: List[OrderLeg] = []
    messages: List[str] = []

    class Config:
        populate_by_name = True


# =============================================================================
# SPREAD ORDER MODELS
# =============================================================================


class SpreadLeg(BaseModel):
    """Leg for a spread order."""

    symbol: str
    option_type: OptionType = Field(..., alias="optionType")
    strike_price: float = Field(..., alias="strikePrice")
    expiration_date: str = Field(..., alias="expirationDate")
    order_action: OrderAction = Field(..., alias="orderAction")
    quantity: int

    class Config:
        populate_by_name = True


class SpreadOrderRequest(BaseModel):
    """Multi-leg spread order request."""

    account_id_key: str = Field(..., alias="accountIdKey")
    underlying_symbol: str = Field(..., alias="underlyingSymbol")
    order_type: OrderType = Field(OrderType.LIMIT, alias="orderType")
    limit_price: Optional[float] = Field(None, alias="limitPrice")  # Net credit/debit
    order_term: OrderDuration = Field(OrderDuration.DAY, alias="orderTerm")
    market_session: MarketSession = Field(MarketSession.REGULAR, alias="marketSession")
    legs: List[SpreadLeg] = []

    class Config:
        populate_by_name = True


# =============================================================================
# AUTH MODELS
# =============================================================================


class TokenData(BaseModel):
    """OAuth token data for persistence."""

    oauth_token: str = Field(..., alias="oauthToken")
    oauth_token_secret: str = Field(..., alias="oauthTokenSecret")
    access_token: Optional[str] = Field(None, alias="accessToken")
    access_token_secret: Optional[str] = Field(None, alias="accessTokenSecret")
    expires_at: Optional[datetime] = Field(None, alias="expiresAt")
    created_at: datetime = Field(default_factory=datetime.now, alias="createdAt")

    class Config:
        populate_by_name = True


class AuthStatus(BaseModel):
    """Authentication status."""

    authenticated: bool = False
    expires_at: Optional[datetime] = Field(None, alias="expiresAt")
    expires_in_seconds: Optional[int] = Field(None, alias="expiresInSeconds")
    needs_refresh: bool = Field(False, alias="needsRefresh")
    sandbox_mode: bool = Field(False, alias="sandboxMode")
    message: str = ""

    class Config:
        populate_by_name = True


# =============================================================================
# RESPONSE WRAPPERS
# =============================================================================


class APIResponse(BaseModel):
    """Generic API response wrapper."""

    success: bool = True
    data: Optional[Any] = None
    error: Optional[str] = None
    error_code: Optional[str] = Field(None, alias="errorCode")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        populate_by_name = True


class RateLimitInfo(BaseModel):
    """Rate limit information."""

    requests_remaining: int = Field(..., alias="requestsRemaining")
    reset_time: datetime = Field(..., alias="resetTime")
    limit_per_second: float = Field(2.0, alias="limitPerSecond")

    class Config:
        populate_by_name = True
