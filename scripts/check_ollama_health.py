#!/usr/bin/env python3
"""
Ollama health check for QuantPod.

Verifies:
  1. Ollama server is reachable
  2. qwen3.5:9b and qwen3.5:35b-a3b are pulled
  3. Both models are loaded in memory (resident in VRAM/unified memory)

If models are pulled but not loaded, preloads them automatically.

Exit codes:
  0 — healthy (both models pulled and resident in memory)
  1 — not healthy (server unreachable, model missing, or preload failed)

Usage:
  python scripts/check_ollama_health.py
  python scripts/check_ollama_health.py --no-preload   (skip auto-preload)

Dependencies: stdlib only (urllib, json, sys)
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

REQUIRED_MODELS = ["qwen3.5:9b", "qwen3.5:35b-a3b"]
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PRELOAD_TIMEOUT = 180  # seconds; 35b model can take ~30s to load cold


def _get(path: str, timeout: float = 5.0) -> Any:
    """GET request to Ollama API, returns parsed JSON or raises."""
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _post(path: str, body: dict, timeout: float = PRELOAD_TIMEOUT) -> Any:
    """POST request to Ollama API, returns parsed JSON or raises."""
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def check_server_reachable() -> bool:
    try:
        _get("/api/tags", timeout=3.0)
        return True
    except Exception:
        return False


def get_pulled_models() -> list[str]:
    data = _get("/api/tags")
    return [m["name"] for m in data.get("models", [])]


def get_loaded_models() -> dict[str, dict]:
    """Return {model_name: {size_vram, ...}} for models currently in memory."""
    try:
        data = _get("/api/ps")
        return {m["name"]: m for m in data.get("models", [])}
    except Exception:
        return {}


def preload_model(model: str) -> bool:
    """Send a minimal generate request to force the model into memory."""
    print(f"  Preloading {model} ... ", end="", flush=True)
    t0 = time.time()
    try:
        _post("/api/generate", {"model": model, "prompt": "test", "stream": False})
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")
        return True
    except Exception as exc:
        print(f"FAILED ({exc})")
        return False


def fmt_bytes(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024**2:.0f} MB"
    return f"{n} B"


def main() -> int:
    no_preload = "--no-preload" in sys.argv

    print(f"Ollama health check — {BASE_URL}")
    print("-" * 50)

    # 1. Server reachable?
    print("Server:  ", end="")
    if not check_server_reachable():
        print("NOT RUNNING")
        print()
        print("Fix: start Ollama with  `ollama serve`  or open the Ollama app.")
        return 1
    print("OK")

    # 2. Models pulled?
    pulled = get_pulled_models()
    missing: list[str] = []
    for model in REQUIRED_MODELS:
        tag = "OK" if model in pulled else "MISSING"
        print(f"Pulled   {model:<30s} {tag}")
        if model not in pulled:
            missing.append(model)

    if missing:
        print()
        print("Missing models. Pull them with:")
        for m in missing:
            print(f"  ollama pull {m}")
        return 1

    # 3. Models loaded in memory?
    loaded = get_loaded_models()
    needs_preload: list[str] = []

    print()
    print("Memory:")
    for model in REQUIRED_MODELS:
        if model in loaded:
            info = loaded[model]
            vram = info.get("size_vram", info.get("size", 0))
            print(f"  {model:<30s} LOADED   {fmt_bytes(vram)}")
        else:
            print(f"  {model:<30s} not loaded")
            needs_preload.append(model)

    if needs_preload:
        if no_preload:
            print()
            print("Models not in memory. Run without --no-preload to auto-load them.")
            return 1

        print()
        print("Preloading models into memory:")
        for model in needs_preload:
            if not preload_model(model):
                print(f"  ERROR: could not preload {model}")
                return 1

        # Re-verify both are loaded
        loaded = get_loaded_models()
        for model in needs_preload:
            if model not in loaded:
                print(f"  ERROR: {model} still not in memory after preload attempt")
                return 1

    print()
    print("Status: HEALTHY — both models loaded and ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
