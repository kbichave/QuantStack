# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared exception classes — zero intra-project dependencies."""


class BrokerError(Exception):
    """Base class for all broker-related errors."""


class BrokerConnectionError(BrokerError):
    """Raised when the broker connection cannot be established or is lost."""
