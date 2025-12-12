from __future__ import annotations

from typing import Dict, List, Set

# Registry metadata for ICs and pod managers.
# Asset classes: equities, options, futures, fx_crypto

IC_REGISTRY: Dict[str, Dict[str, Set[str]]] = {
    "data_ingestion_ic": {
        "asset_classes": {"equities", "options", "futures", "fx_crypto"},
        "capabilities": {"data"},
    },
    "market_snapshot_ic": {
        "asset_classes": {"equities", "options", "futures", "fx_crypto"},
        "capabilities": {"market_state"},
    },
    "regime_detector_ic": {
        "asset_classes": {"equities", "options", "futures", "fx_crypto"},
        "capabilities": {"regime"},
    },
    "trend_momentum_ic": {
        "asset_classes": {"equities", "options", "futures", "fx_crypto"},
        "capabilities": {"technicals"},
    },
    "volatility_ic": {
        "asset_classes": {"equities", "options", "futures", "fx_crypto"},
        "capabilities": {"technicals"},
    },
    "structure_levels_ic": {
        "asset_classes": {"equities", "options", "futures", "fx_crypto"},
        "capabilities": {"technicals"},
    },
    "statarb_ic": {
        "asset_classes": {"equities", "options", "futures"},
        "capabilities": {"quant"},
    },
    "options_vol_ic": {
        "asset_classes": {"options"},
        "capabilities": {"options", "vol"},
    },
    "risk_limits_ic": {
        "asset_classes": {"equities", "options", "futures", "fx_crypto"},
        "capabilities": {"risk"},
    },
    "calendar_events_ic": {
        "asset_classes": {"equities", "options", "futures", "fx_crypto"},
        "capabilities": {"events"},
    },
}

POD_MANAGER_REGISTRY: Dict[str, Dict[str, Set[str]]] = {
    "data_pod_manager": {
        "asset_classes": {"equities", "options", "futures", "fx_crypto"},
        "capabilities": {"data"},
    },
    "market_monitor_pod_manager": {
        "asset_classes": {"equities", "options", "futures", "fx_crypto"},
        "capabilities": {"market_state"},
    },
    "technicals_pod_manager": {
        "asset_classes": {"equities", "options", "futures", "fx_crypto"},
        "capabilities": {"technicals"},
    },
    "quant_pod_manager": {
        "asset_classes": {"equities", "options", "futures"},
        "capabilities": {"quant", "options"},
    },
    "risk_pod_manager": {
        "asset_classes": {"equities", "options", "futures", "fx_crypto"},
        "capabilities": {"risk"},
    },
}

POD_DEPENDENCIES: Dict[str, List[str]] = {
    "data_pod_manager": ["data_ingestion_ic"],
    "market_monitor_pod_manager": ["market_snapshot_ic", "regime_detector_ic"],
    "technicals_pod_manager": [
        "trend_momentum_ic",
        "volatility_ic",
        "structure_levels_ic",
    ],
    "quant_pod_manager": ["statarb_ic", "options_vol_ic"],
    "risk_pod_manager": ["risk_limits_ic", "calendar_events_ic"],
}

# Default runtime profiles (ordered for determinism)
PROFILE_DEFAULTS: Dict[str, Dict[str, List[str]]] = {
    "equities": {
        # No options IC by default for equities
        "ics": [
            "data_ingestion_ic",
            "market_snapshot_ic",
            "regime_detector_ic",
            "trend_momentum_ic",
            "volatility_ic",
            "structure_levels_ic",
            "statarb_ic",
            "risk_limits_ic",
            "calendar_events_ic",
        ],
        "pod_managers": [
            "data_pod_manager",
            "market_monitor_pod_manager",
            "technicals_pod_manager",
            "quant_pod_manager",
            "risk_pod_manager",
        ],
    },
    "options": {
        "ics": [
            "data_ingestion_ic",
            "market_snapshot_ic",
            "regime_detector_ic",
            "trend_momentum_ic",
            "volatility_ic",
            "structure_levels_ic",
            "statarb_ic",
            "options_vol_ic",
            "risk_limits_ic",
            "calendar_events_ic",
        ],
        "pod_managers": [
            "data_pod_manager",
            "market_monitor_pod_manager",
            "technicals_pod_manager",
            "quant_pod_manager",
            "risk_pod_manager",
        ],
    },
    "futures": {
        "ics": [
            "data_ingestion_ic",
            "market_snapshot_ic",
            "regime_detector_ic",
            "trend_momentum_ic",
            "volatility_ic",
            "structure_levels_ic",
            "statarb_ic",
            "risk_limits_ic",
            "calendar_events_ic",
        ],
        "pod_managers": [
            "data_pod_manager",
            "market_monitor_pod_manager",
            "technicals_pod_manager",
            "quant_pod_manager",
            "risk_pod_manager",
        ],
    },
    "fx_crypto": {
        "ics": [
            "data_ingestion_ic",
            "market_snapshot_ic",
            "regime_detector_ic",
            "trend_momentum_ic",
            "volatility_ic",
            "structure_levels_ic",
            "risk_limits_ic",
            "calendar_events_ic",
        ],
        "pod_managers": [
            "data_pod_manager",
            "market_monitor_pod_manager",
            "technicals_pod_manager",
            "risk_pod_manager",
        ],
    },
}


def registry_snapshot() -> Dict[str, Dict[str, List[str]]]:
    """Return registry metadata for prompts/logging."""
    return {
        "ics": {
            name: sorted(meta["asset_classes"]) for name, meta in IC_REGISTRY.items()
        },
        "pod_managers": {
            name: sorted(meta["asset_classes"])
            for name, meta in POD_MANAGER_REGISTRY.items()
        },
        "profiles": PROFILE_DEFAULTS,
    }
