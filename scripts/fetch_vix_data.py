#!/usr/bin/env python3
"""Fetch all historical VIX data from FRED and store in DuckDB."""

from quantstack.data.fred_fetcher import FREDFetcher
from quantstack.data.economic_storage import EconomicStorage


def main():
    """Fetch VIX data from FRED and store in DuckDB."""
    # Create storage
    storage = EconomicStorage()

    # Create FRED fetcher
    fred_fetcher = FREDFetcher(storage=storage)

    print("Fetching VIX data from FRED (VIXCLS)...")
    # Fetch all available history (30+ years)
    df = fred_fetcher.fetch("vix", days=365 * 40, force_refresh=True)

    if df is not None and not df.empty:
        print(f"✓ Fetched {len(df)} VIX records")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Value range: {df['value'].min():.2f} to {df['value'].max():.2f}")

        # Show metadata
        metadata = storage.list_indicators()
        vix_meta = metadata[metadata['indicator'] == 'vix']
        if not vix_meta.empty:
            print(f"\nVIX Metadata:")
            print(f"  Records: {vix_meta['record_count'].values[0]}")
            print(f"  First date: {vix_meta['first_date'].values[0]}")
            print(f"  Last date: {vix_meta['last_date'].values[0]}")
    else:
        print("✗ Failed to fetch VIX data")
        print("  Make sure FRED_API_KEY is set in environment")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
