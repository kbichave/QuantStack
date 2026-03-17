# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
SignalEngine collectors — one per analysis domain.

Each collector is an async function that accepts (symbol, store) and returns
a plain dict. Failures are isolated: a timeout or exception in one collector
returns an empty dict and is recorded in collector_failures; it does not
prevent other collectors from completing.
"""
