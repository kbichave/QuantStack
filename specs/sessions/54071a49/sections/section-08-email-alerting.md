# Section 8: Email Alerting System

## Problem

No alerting mechanism exists in QuantStack. Critical events -- kill switch triggers, drawdown breaches, system errors, circuit breaker activations -- go completely unnoticed until someone manually checks logs or the dashboard. For a system designed to run unattended 24/7, silent failures are unacceptable. The existing `_send_alert()` in `src/quantstack/autonomous/watchdog.py` uses Discord webhooks (optional, best-effort), but Discord is not suitable for high-priority operational alerts that must reach the operator reliably.

## Design

A lightweight email alerting module using Gmail SMTP. Gmail was chosen because it is free, highly available, and supports app passwords for programmatic access without OAuth dance.

### Three Alert Levels

| Level | Use Cases | Rate Limiting |
|-------|-----------|---------------|
| **INFO** | Daily digest, strategy registrations, overnight research summary | Max 1 per event type per 15 minutes |
| **WARNING** | Threshold approaching (80% of daily loss limit), data staleness, failed retries, circuit breaker cooldown | Max 1 per event type per 15 minutes |
| **CRITICAL** | Kill switch triggered, circuit breaker activated (daily P&L or portfolio HWM), emergency liquidation, system errors | Bypasses rate limiting entirely |

The rate limiter prevents alert storms during cascading failures (e.g., data provider goes down, every collector fails, 22 WARNING emails fire simultaneously). CRITICAL events bypass rate limiting because they represent conditions that require immediate human attention regardless of how many have already been sent.

### Fallback Strategy

When SMTP fails (network issue, app password revoked, Gmail outage), alerts fall back to a local file at `/var/log/quantstack/alerts.log`. This ensures critical events are never silently lost. The file can be tailed manually or picked up by the log aggregation pipeline (Section 11). The fallback is transparent to callers -- `send_alert()` always succeeds from the caller's perspective.

### Configuration

All config comes from environment variables, consistent with the rest of the system:

| Env Var | Purpose | Default |
|---------|---------|---------|
| `GMAIL_APP_PASSWORD` | Gmail app password for SMTP auth | (required for email) |
| `ALERT_SENDER_EMAIL` | Gmail address to send from | (required for email) |
| `ALERT_RECIPIENT_EMAIL` | Where alerts are delivered | (required for email) |
| `ALERT_SMTP_HOST` | SMTP server hostname | `smtp.gmail.com` |
| `ALERT_SMTP_PORT` | SMTP server port | `587` |
| `ALERT_FALLBACK_PATH` | Local file fallback path | `/var/log/quantstack/alerts.log` |

When `GMAIL_APP_PASSWORD` is not set, the module operates in file-only mode (all alerts go to the fallback file). This is the safe default for development environments.

## Dependencies

- **Phase 1 complete** (sections 01-06): This section assumes baseline safety hardening is in place.
- **No external dependencies**: Uses Python's `smtplib` and `email` stdlib modules. No new packages.
- **Downstream: Section 11** (Kill Switch Auto-Recovery) depends on this module to send CRITICAL emails on kill switch trigger and escalation emails after 4-hour timeout.
- **Downstream: Section 15** (Layered Circuit Breaker) depends on this module to send CRITICAL emails on portfolio HWM breach and dead-man's switch activation.

## Tests First

All tests go in a single file. They mock SMTP to avoid real email sends. Tests verify the three core behaviors: sending works, rate limiting works, fallback works.

```python
# tests/alerting/test_email_sender.py

"""Tests for email alerting system.

All tests mock SMTP — no real emails sent. Tests verify:
1. CRITICAL alerts send email via SMTP
2. Rate limiting prevents INFO/WARNING storms
3. CRITICAL bypasses rate limiting
4. SMTP failure triggers file fallback
5. Config loads from environment
"""

import time
import pytest
from unittest.mock import patch, MagicMock

# ---- SMTP sending ----

# Test: send_alert sends email via SMTP for CRITICAL level
#   Mock smtplib.SMTP. Call send_alert with level=CRITICAL.
#   Assert SMTP.sendmail was called once with correct sender, recipient, and
#   a message body containing the alert text.
#   Assert SMTP.starttls() was called (TLS required for Gmail).
#   Assert SMTP.login() was called with configured credentials.

# ---- Rate limiting ----

# Test: send_alert rate-limits INFO/WARNING to 1 per event type per 15 min
#   Call send_alert(level=WARNING, event_type="data_stale") twice within 1 second.
#   Assert SMTP.sendmail called exactly once (second call suppressed).
#   Call again with a DIFFERENT event_type — assert it sends (different type, separate limit).

# Test: CRITICAL level bypasses rate limiting
#   Call send_alert(level=CRITICAL, event_type="kill_switch") 5 times in rapid succession.
#   Assert SMTP.sendmail called 5 times (no suppression).

# ---- SMTP failure fallback ----

# Test: SMTP failure falls back to local file logging
#   Mock SMTP to raise ConnectionRefusedError on connect.
#   Call send_alert with level=CRITICAL.
#   Assert the fallback file was written with the alert content.
#   Assert no exception propagates to the caller.

# Test: alert file fallback writes to configured path
#   Set ALERT_FALLBACK_PATH to a tmp file (use tmp_path fixture).
#   Trigger a fallback write. Read the file and verify it contains:
#   timestamp, alert level, event type, and the message body.
#   Verify the format is parseable (one alert per block, separated by newlines).

# ---- Configuration ----

# Test: AlertConfig loads from environment variables
#   Patch os.environ with GMAIL_APP_PASSWORD, ALERT_SENDER_EMAIL,
#   ALERT_RECIPIENT_EMAIL. Instantiate AlertConfig.
#   Assert all fields populated correctly.
#   Assert defaults applied for smtp_host (smtp.gmail.com) and smtp_port (587).

# Test: AlertConfig without GMAIL_APP_PASSWORD enables file-only mode
#   Patch os.environ without GMAIL_APP_PASSWORD.
#   Instantiate the alert manager. Call send_alert.
#   Assert SMTP was never attempted. Assert fallback file was written.
```

## Implementation Details

Two new files in a new `src/quantstack/alerting/` package. An `__init__.py` re-exports the public interface.

### File: `src/quantstack/alerting/__init__.py` (new)

```python
"""Email alerting with rate limiting and file fallback."""

from quantstack.alerting.email_sender import send_alert, AlertLevel
from quantstack.alerting.alert_manager import AlertManager, get_alert_manager

__all__ = ["send_alert", "AlertLevel", "AlertManager", "get_alert_manager"]
```

### File: `src/quantstack/alerting/email_sender.py` (new)

This is the core module. It contains the SMTP logic, rate limiter, and file fallback.

**`AlertLevel` enum** -- Three levels:

```python
class AlertLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
```

**`AlertConfig` dataclass** -- Loaded from environment:

```python
@dataclass(frozen=True)
class AlertConfig:
    smtp_host: str       # default: smtp.gmail.com
    smtp_port: int       # default: 587
    sender_email: str    # from ALERT_SENDER_EMAIL
    app_password: str    # from GMAIL_APP_PASSWORD
    recipient_email: str # from ALERT_RECIPIENT_EMAIL
    fallback_path: str   # default: /var/log/quantstack/alerts.log

    @classmethod
    def from_env(cls) -> "AlertConfig":
        """Load config from environment. Returns config with empty app_password
        if GMAIL_APP_PASSWORD is not set (file-only mode)."""
        ...
```

**`send_alert()` function** -- The primary public interface:

```python
def send_alert(
    level: AlertLevel,
    event_type: str,
    subject: str,
    body: str,
    *,
    config: AlertConfig | None = None,
) -> None:
    """Send an alert via email with rate limiting and file fallback.

    Args:
        level: Alert severity. CRITICAL bypasses rate limiting.
        event_type: Category string for rate-limit grouping (e.g., "kill_switch",
            "data_stale", "daily_digest"). Same event_type within 15 minutes
            is suppressed for INFO/WARNING.
        subject: Email subject line.
        body: Email body (plain text).
        config: Optional override. Defaults to config loaded from env.

    Behavior:
        1. Check rate limit (skip for CRITICAL).
        2. Attempt SMTP send if app_password is configured.
        3. On SMTP failure or missing credentials, write to fallback file.
        4. Never raises — alerting failure must not crash the caller.
    """
    ...
```

**Key implementation notes:**

- Rate limiter state is a module-level `dict[str, float]` mapping `event_type` to the monotonic timestamp of last send. Use `time.monotonic()` for immunity to wall clock changes. The 15-minute window is a constant `RATE_LIMIT_WINDOW = 900.0`.
- SMTP connection is opened and closed per send (not pooled). At the expected volume (a few emails per day in normal operation, dozens during incidents), connection pooling is unnecessary complexity.
- Use `smtplib.SMTP` with `starttls()` for Gmail. Build the email with `email.mime.text.MIMEText` for the body and `email.mime.multipart.MIMEMultipart` if HTML is ever needed (start with plain text only).
- The entire `send_alert()` body is wrapped in a top-level `try/except Exception` that logs to `logger.error()` and writes to the fallback file. Alerting must never crash the caller.
- Thread safety: The rate limiter dict is accessed from potentially concurrent graph nodes. Use a `threading.Lock` around reads and writes. The lock is module-level, same lifetime as the rate limiter dict.
- Fallback file: Create parent directories with `os.makedirs(exist_ok=True)` before writing. Use append mode (`"a"`). Each alert entry includes ISO timestamp, level, event_type, subject, and body, separated by a blank line for readability.

### File: `src/quantstack/alerting/alert_manager.py` (new)

A thin orchestration layer that provides a singleton pattern and convenience methods. This exists so callers don't need to import `AlertLevel` and construct arguments manually for common alert scenarios.

```python
class AlertManager:
    """Singleton alert manager with convenience methods for common alert patterns.

    Usage:
        mgr = get_alert_manager()
        mgr.critical("kill_switch", "Kill Switch Triggered", "Daily loss limit exceeded at -2.3%")
        mgr.warning("data_stale", "Data Staleness", "SPY last update 47 minutes ago")
        mgr.info("daily_digest", "Daily Summary", summary_text)
    """

    def __init__(self, config: AlertConfig | None = None):
        """Initialize with optional config override."""
        ...

    def critical(self, event_type: str, subject: str, body: str) -> None:
        """Send a CRITICAL alert. Bypasses rate limiting."""
        ...

    def warning(self, event_type: str, subject: str, body: str) -> None:
        """Send a WARNING alert. Rate-limited to 1 per event_type per 15 min."""
        ...

    def info(self, event_type: str, subject: str, body: str) -> None:
        """Send an INFO alert. Rate-limited to 1 per event_type per 15 min."""
        ...


def get_alert_manager() -> AlertManager:
    """Return the module-level singleton AlertManager instance."""
    ...
```

The singleton is lazily initialized on first call to `get_alert_manager()`. Config is loaded from env at that point.

### File: `.env.example` (modify)

Add the following block after the existing Discord webhook section:

```bash
# =============================================================================
# OPTIONAL: Email Alerting — Gmail SMTP
# =============================================================================

# Gmail app password (NOT your regular password).
# Generate at: https://myaccount.google.com/apppasswords
# Requires 2-Step Verification enabled on your Google account.
# When unset, alerts fall back to local file logging only.
# GMAIL_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx

# Sender Gmail address (must match the account that generated the app password)
# ALERT_SENDER_EMAIL=your.quantstack@gmail.com

# Where to deliver alerts (can be any email, not just Gmail)
# ALERT_RECIPIENT_EMAIL=you@example.com

# Override defaults (usually not needed)
# ALERT_SMTP_HOST=smtp.gmail.com
# ALERT_SMTP_PORT=587
# ALERT_FALLBACK_PATH=/var/log/quantstack/alerts.log
```

## Integration Points

Other sections consume the alerting module. These are documented here for context but implemented in their respective sections:

- **Section 11 (Kill Switch Recovery):** Calls `get_alert_manager().critical("kill_switch", ...)` immediately on trigger. Calls `get_alert_manager().critical("kill_switch_escalation", ...)` after 4-hour timeout if not reset.
- **Section 15 (Layered Circuit Breaker):** Calls `get_alert_manager().critical("portfolio_hwm_breach", ...)` on -5% HWM drawdown. Calls `get_alert_manager().critical("dead_man_switch", ...)` when emergency liquidation limit orders are unfilled after 60 seconds.
- **Future: Daily digest** (not in scope for this section): A scheduled job could call `get_alert_manager().info("daily_digest", ...)` at 16:30 ET with a summary of the day's trades, P&L, and research output.

## Verification

After implementation, verify by:

1. **Unit tests pass**: All 6 test cases in `tests/alerting/test_email_sender.py`.
2. **Manual smoke test**: Set `GMAIL_APP_PASSWORD`, `ALERT_SENDER_EMAIL`, and `ALERT_RECIPIENT_EMAIL` in `.env`. Run a quick script that calls `send_alert(AlertLevel.CRITICAL, "test", "Test Alert", "This is a test.")`. Confirm email arrives in inbox within 30 seconds.
3. **Fallback test**: Unset `GMAIL_APP_PASSWORD`. Call `send_alert()`. Confirm alert is written to the fallback file path.
4. **Rate limit test**: Call `send_alert(AlertLevel.WARNING, "test_rl", ...)` twice rapidly. Confirm only one email sent, second suppressed (check logs for "rate-limited" message).

## Rollback

Remove the `src/quantstack/alerting/` package. All callers in downstream sections (11, 15) should have a `try/except ImportError` or conditional import pattern so they degrade gracefully to logging-only if the alerting module is absent. The system returns to the current behavior: critical events logged but not pushed to any notification channel.

## Gmail SMTP Limits

Gmail allows 500 emails per day per account. At the expected alert volume (a handful of INFO/WARNING per day, rare CRITICAL events), this is more than sufficient. The rate limiter (1 per event type per 15 min for non-CRITICAL) provides an additional safety net. Even in a worst-case cascading failure with 50 distinct event types all firing, the system would send at most 50 emails in the first 15 minutes, then 50 more per 15-minute window -- well under Gmail's daily cap.
