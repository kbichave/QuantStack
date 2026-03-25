"""
Fetch Alpha Vantage data for a list of tickers and print as JSON.
Usage: python scripts/fetch_ticker_data.py
"""

import json
import sys
import time

from quantstack.data.fetcher import AlphaVantageClient
from quantstack.data.universe import SPECULATIVE_SYMBOLS

TICKERS = list(SPECULATIVE_SYMBOLS)

def main():
    client = AlphaVantageClient()
    results = {}

    # Bulk quotes — single call for current price/volume
    print("Fetching bulk quotes...", file=sys.stderr)
    try:
        quotes_df = client.fetch_bulk_quotes(TICKERS)
        quotes = quotes_df.set_index("symbol").to_dict(orient="index") if not quotes_df.empty else {}
    except Exception as e:
        print(f"Bulk quotes failed: {e}", file=sys.stderr)
        quotes = {}

    # Overview — one call per ticker
    for symbol in TICKERS:
        print(f"Fetching overview for {symbol}...", file=sys.stderr)
        try:
            overview = client.fetch_company_overview(symbol)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            overview = {}

        results[symbol] = {
            "overview": overview,
            "quote": quotes.get(symbol, {}),
        }
        time.sleep(0.5)  # gentle pacing

    print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    main()
