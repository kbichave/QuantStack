# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
eTrade OAuth 1.0a Authentication Module.

Handles:
- OAuth 1.0a three-legged authentication flow
- Token persistence and refresh
- Sandbox vs production environment switching
- FastAPI endpoints for browser-based authorization
"""

from __future__ import annotations

import json
import os
import time
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlencode

import requests
from loguru import logger
from pydantic import BaseModel

from etrade_mcp.models import AuthStatus, TokenData

# =============================================================================
# CONSTANTS
# =============================================================================

# eTrade API endpoints
ETRADE_SANDBOX_BASE = "https://apisb.etrade.com"
ETRADE_PRODUCTION_BASE = "https://api.etrade.com"

OAUTH_REQUEST_TOKEN_URL = "/oauth/request_token"
OAUTH_ACCESS_TOKEN_URL = "/oauth/access_token"
OAUTH_RENEW_TOKEN_URL = "/oauth/renew_access_token"
OAUTH_REVOKE_TOKEN_URL = "/oauth/revoke_access_token"
OAUTH_AUTHORIZE_URL = "https://us.etrade.com/e/t/etws/authorize"

# Token expires at midnight Eastern
TOKEN_LIFETIME_HOURS = 24
TOKEN_REFRESH_THRESHOLD_HOURS = 2  # Refresh if less than 2 hours remaining

# Default token storage location
DEFAULT_TOKEN_PATH = Path.home() / ".etrade_tokens.json"


# =============================================================================
# AUTH MANAGER CLASS
# =============================================================================


class ETradeAuthManager:
    """
    Manages eTrade OAuth 1.0a authentication.

    Features:
    - Three-legged OAuth flow support
    - Automatic token persistence
    - Token refresh before expiration
    - Sandbox/production environment switching
    - Thread-safe token access

    Usage:
        auth = ETradeAuthManager()

        # Check if we have valid tokens
        if not auth.is_authenticated():
            # Start OAuth flow
            auth_url = auth.get_authorization_url()
            # User visits auth_url and gets verifier code
            auth.complete_authorization(verifier_code)

        # Get OAuth session for API calls
        session = auth.get_session()
    """

    def __init__(
        self,
        consumer_key: Optional[str] = None,
        consumer_secret: Optional[str] = None,
        sandbox: Optional[bool] = None,
        token_path: Optional[Path] = None,
    ):
        """
        Initialize the auth manager.

        Args:
            consumer_key: eTrade API consumer key (or ETRADE_CONSUMER_KEY env var)
            consumer_secret: eTrade API consumer secret (or ETRADE_CONSUMER_SECRET env var)
            sandbox: Use sandbox environment (or ETRADE_SANDBOX env var, default True)
            token_path: Path for token persistence (default ~/.etrade_tokens.json)
        """
        self.consumer_key = consumer_key or os.getenv("ETRADE_CONSUMER_KEY", "")
        self.consumer_secret = consumer_secret or os.getenv(
            "ETRADE_CONSUMER_SECRET", ""
        )

        # Sandbox mode from env or parameter (default to sandbox for safety)
        if sandbox is None:
            sandbox = os.getenv("ETRADE_SANDBOX", "true").lower() in (
                "true",
                "1",
                "yes",
            )
        self.sandbox = sandbox

        self.token_path = token_path or DEFAULT_TOKEN_PATH
        self._lock = Lock()

        # OAuth state
        self._request_token: Optional[str] = None
        self._request_token_secret: Optional[str] = None
        self._access_token: Optional[str] = None
        self._access_token_secret: Optional[str] = None
        self._expires_at: Optional[datetime] = None

        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 0.5  # 2 requests/second

        # Load persisted tokens
        self._load_tokens()

        logger.info(
            f"ETradeAuthManager initialized (sandbox={self.sandbox}, "
            f"authenticated={self.is_authenticated()})"
        )

    @property
    def base_url(self) -> str:
        """Get the base URL for the current environment."""
        return ETRADE_SANDBOX_BASE if self.sandbox else ETRADE_PRODUCTION_BASE

    def _rate_limit(self) -> None:
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _load_tokens(self) -> None:
        """Load tokens from persistent storage."""
        if not self.token_path.exists():
            logger.debug(f"No token file found at {self.token_path}")
            return

        try:
            with open(self.token_path, "r") as f:
                data = json.load(f)

            # Check environment match
            env_key = "sandbox" if self.sandbox else "production"
            if env_key not in data:
                logger.debug(f"No tokens for {env_key} environment")
                return

            tokens = data[env_key]
            self._access_token = tokens.get("access_token")
            self._access_token_secret = tokens.get("access_token_secret")

            if tokens.get("expires_at"):
                self._expires_at = datetime.fromisoformat(tokens["expires_at"])

            logger.info(f"Loaded tokens from {self.token_path}")
        except Exception as e:
            logger.warning(f"Failed to load tokens: {e}")

    def _save_tokens(self) -> None:
        """Save tokens to persistent storage."""
        try:
            # Load existing data
            data = {}
            if self.token_path.exists():
                with open(self.token_path, "r") as f:
                    data = json.load(f)

            # Update with current tokens
            env_key = "sandbox" if self.sandbox else "production"
            data[env_key] = {
                "access_token": self._access_token,
                "access_token_secret": self._access_token_secret,
                "expires_at": (
                    self._expires_at.isoformat() if self._expires_at else None
                ),
                "updated_at": datetime.now().isoformat(),
            }

            # Write atomically
            temp_path = self.token_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)
            temp_path.rename(self.token_path)

            # Secure permissions
            os.chmod(self.token_path, 0o600)

            logger.info(f"Saved tokens to {self.token_path}")
        except Exception as e:
            logger.error(f"Failed to save tokens: {e}")

    def _clear_tokens(self) -> None:
        """Clear all stored tokens."""
        self._request_token = None
        self._request_token_secret = None
        self._access_token = None
        self._access_token_secret = None
        self._expires_at = None

    def is_authenticated(self) -> bool:
        """Check if we have valid access tokens."""
        if not self._access_token or not self._access_token_secret:
            return False

        # Check expiration
        if self._expires_at and datetime.now() >= self._expires_at:
            logger.warning("Access token has expired")
            return False

        return True

    def needs_refresh(self) -> bool:
        """Check if tokens should be refreshed soon."""
        if not self.is_authenticated():
            return False

        if self._expires_at:
            threshold = self._expires_at - timedelta(
                hours=TOKEN_REFRESH_THRESHOLD_HOURS
            )
            return datetime.now() >= threshold

        return False

    def get_auth_status(self) -> AuthStatus:
        """Get current authentication status."""
        expires_in = None
        if self._expires_at:
            delta = self._expires_at - datetime.now()
            expires_in = max(0, int(delta.total_seconds()))

        return AuthStatus(
            authenticated=self.is_authenticated(),
            expires_at=self._expires_at,
            expires_in_seconds=expires_in,
            needs_refresh=self.needs_refresh(),
            sandbox_mode=self.sandbox,
            message="Authenticated" if self.is_authenticated() else "Not authenticated",
        )

    def _get_oauth_header(
        self,
        method: str,
        url: str,
        token: Optional[str] = None,
        token_secret: Optional[str] = None,
        verifier: Optional[str] = None,
        extra_params: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Generate OAuth 1.0a authorization header.

        Uses HMAC-SHA1 signature method as required by eTrade.
        """
        import base64
        import hashlib
        import hmac
        import uuid

        # OAuth parameters
        oauth_params = {
            "oauth_consumer_key": self.consumer_key,
            "oauth_nonce": uuid.uuid4().hex,
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_timestamp": str(int(time.time())),
            "oauth_version": "1.0",
        }

        if token:
            oauth_params["oauth_token"] = token
        if verifier:
            oauth_params["oauth_verifier"] = verifier

        # Add extra params for signature base
        all_params = {**oauth_params}
        if extra_params:
            all_params.update(extra_params)

        # Create signature base string
        sorted_params = sorted(all_params.items())
        param_string = urlencode(sorted_params)

        base_string = "&".join(
            [
                method.upper(),
                requests.utils.quote(url, safe=""),
                requests.utils.quote(param_string, safe=""),
            ]
        )

        # Create signing key
        signing_key = f"{requests.utils.quote(self.consumer_secret, safe='')}&"
        if token_secret:
            signing_key += requests.utils.quote(token_secret, safe="")

        # Generate signature
        signature = base64.b64encode(
            hmac.new(
                signing_key.encode("utf-8"),
                base_string.encode("utf-8"),
                hashlib.sha1,
            ).digest()
        ).decode("utf-8")

        oauth_params["oauth_signature"] = signature

        # Build Authorization header
        auth_header = "OAuth " + ", ".join(
            f'{k}="{requests.utils.quote(str(v), safe="")}"'
            for k, v in sorted(oauth_params.items())
        )

        return {"Authorization": auth_header}

    def get_request_token(self) -> Tuple[str, str]:
        """
        Step 1: Get OAuth request token.

        Returns:
            Tuple of (request_token, request_token_secret)
        """
        with self._lock:
            self._rate_limit()

            url = f"{self.base_url}{OAUTH_REQUEST_TOKEN_URL}"

            headers = self._get_oauth_header("GET", url)
            headers["Content-Type"] = "application/x-www-form-urlencoded"

            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()

                # Parse response
                params = parse_qs(response.text)
                self._request_token = params["oauth_token"][0]
                self._request_token_secret = params["oauth_token_secret"][0]

                logger.info("Obtained request token")
                return self._request_token, self._request_token_secret

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to get request token: {e}")
                raise

    def get_authorization_url(self, callback_url: Optional[str] = None) -> str:
        """
        Step 2: Get URL for user authorization.

        Args:
            callback_url: Optional OAuth callback URL

        Returns:
            URL for user to authorize the application
        """
        if not self._request_token:
            self.get_request_token()

        params = {
            "key": self.consumer_key,
            "token": self._request_token,
        }

        auth_url = f"{OAUTH_AUTHORIZE_URL}?{urlencode(params)}"
        logger.info(f"Authorization URL: {auth_url}")

        return auth_url

    def complete_authorization(self, verifier: str) -> bool:
        """
        Step 3: Exchange verifier for access token.

        Args:
            verifier: Verifier code from user authorization

        Returns:
            True if successful
        """
        with self._lock:
            self._rate_limit()

            if not self._request_token or not self._request_token_secret:
                raise ValueError("No request token - call get_authorization_url first")

            url = f"{self.base_url}{OAUTH_ACCESS_TOKEN_URL}"

            headers = self._get_oauth_header(
                "GET",
                url,
                token=self._request_token,
                token_secret=self._request_token_secret,
                verifier=verifier,
            )

            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()

                # Parse response
                params = parse_qs(response.text)
                self._access_token = params["oauth_token"][0]
                self._access_token_secret = params["oauth_token_secret"][0]

                # Set expiration to midnight Eastern (rough approximation)
                self._expires_at = datetime.now().replace(
                    hour=23, minute=59, second=59
                ) + timedelta(
                    hours=5
                )  # EST offset

                # Clear request tokens
                self._request_token = None
                self._request_token_secret = None

                # Persist tokens
                self._save_tokens()

                logger.info("Authorization complete - access token obtained")
                return True

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to complete authorization: {e}")
                raise

    def refresh_token(self) -> bool:
        """
        Renew access token before expiration.

        Returns:
            True if successful
        """
        with self._lock:
            self._rate_limit()

            if not self._access_token or not self._access_token_secret:
                logger.error("No access token to refresh")
                return False

            url = f"{self.base_url}{OAUTH_RENEW_TOKEN_URL}"

            headers = self._get_oauth_header(
                "GET",
                url,
                token=self._access_token,
                token_secret=self._access_token_secret,
            )

            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()

                # Update expiration
                self._expires_at = datetime.now().replace(
                    hour=23, minute=59, second=59
                ) + timedelta(hours=5)

                # Persist updated expiration
                self._save_tokens()

                logger.info("Access token renewed")
                return True

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to refresh token: {e}")
                return False

    def revoke_token(self) -> bool:
        """
        Revoke current access token.

        Returns:
            True if successful
        """
        with self._lock:
            self._rate_limit()

            if not self._access_token or not self._access_token_secret:
                logger.warning("No access token to revoke")
                return True

            url = f"{self.base_url}{OAUTH_REVOKE_TOKEN_URL}"

            headers = self._get_oauth_header(
                "GET",
                url,
                token=self._access_token,
                token_secret=self._access_token_secret,
            )

            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()

                # Clear tokens
                self._clear_tokens()

                logger.info("Access token revoked")
                return True

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to revoke token: {e}")
                return False

    def get_auth_headers(self, method: str, url: str) -> Dict[str, str]:
        """
        Get OAuth headers for an authenticated API request.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full URL of the request

        Returns:
            Headers dict with Authorization header
        """
        if not self.is_authenticated():
            raise ValueError("Not authenticated - complete OAuth flow first")

        if self.needs_refresh():
            self.refresh_token()

        return self._get_oauth_header(
            method,
            url,
            token=self._access_token,
            token_secret=self._access_token_secret,
        )

    def open_authorization_browser(self) -> str:
        """
        Open browser for user authorization.

        Returns:
            Authorization URL that was opened
        """
        auth_url = self.get_authorization_url()
        webbrowser.open(auth_url)
        logger.info(f"Opened browser for authorization: {auth_url}")
        return auth_url


# =============================================================================
# FASTAPI AUTH ENDPOINTS
# =============================================================================


def create_auth_router(auth_manager: ETradeAuthManager):
    """
    Create FastAPI router for OAuth endpoints.

    Endpoints:
    - GET /etrade/authorize - Start OAuth flow, redirects to eTrade
    - GET /etrade/callback - OAuth callback with verifier
    - GET /etrade/status - Check authentication status
    - POST /etrade/refresh - Refresh access token
    - POST /etrade/revoke - Revoke access token

    Args:
        auth_manager: ETradeAuthManager instance

    Returns:
        FastAPI APIRouter
    """
    try:
        from fastapi import APIRouter, Query, Request
        from fastapi.responses import HTMLResponse, RedirectResponse
    except ImportError:
        logger.warning("FastAPI not installed - auth endpoints not available")
        return None

    router = APIRouter(prefix="/etrade", tags=["eTrade Auth"])

    @router.get("/authorize")
    async def authorize():
        """Start OAuth authorization flow."""
        auth_url = auth_manager.get_authorization_url()
        return RedirectResponse(url=auth_url)

    @router.get("/callback", response_class=HTMLResponse)
    async def callback(oauth_verifier: str = Query(...)):
        """
        OAuth callback endpoint.

        eTrade redirects here with the verifier code.
        """
        try:
            auth_manager.complete_authorization(oauth_verifier)
            return """
            <html>
            <head><title>Authorization Successful</title></head>
            <body>
                <h1>✅ Authorization Successful!</h1>
                <p>You can close this window and return to the application.</p>
                <script>setTimeout(() => window.close(), 3000);</script>
            </body>
            </html>
            """
        except Exception as e:
            return f"""
            <html>
            <head><title>Authorization Failed</title></head>
            <body>
                <h1>❌ Authorization Failed</h1>
                <p>Error: {str(e)}</p>
            </body>
            </html>
            """

    @router.get("/status")
    async def status():
        """Get current authentication status."""
        return auth_manager.get_auth_status().model_dump()

    @router.post("/refresh")
    async def refresh():
        """Refresh access token."""
        success = auth_manager.refresh_token()
        return {
            "success": success,
            "status": auth_manager.get_auth_status().model_dump(),
        }

    @router.post("/revoke")
    async def revoke():
        """Revoke access token."""
        success = auth_manager.revoke_token()
        return {"success": success}

    return router
