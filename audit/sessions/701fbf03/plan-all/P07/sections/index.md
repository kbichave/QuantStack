<!-- PROJECT_CONFIG
runtime: python-uv
test_command: uv run pytest
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-db-decomposition
section-02-migration-versioning  depends_on:section-01-db-decomposition
section-03-polygon-adapter
section-04-yahoo-adapter
section-05-fmp-adapter
section-06-provider-chain  depends_on:section-03-polygon-adapter,section-04-yahoo-adapter,section-05-fmp-adapter
section-07-point-in-time  depends_on:section-02-migration-versioning
section-08-staleness-tiering
section-09-ohlcv-partitioning  depends_on:section-02-migration-versioning
section-10-unit-tests  depends_on:section-06-provider-chain,section-07-point-in-time,section-08-staleness-tiering,section-09-ohlcv-partitioning
END_MANIFEST -->

# P07 Implementation Sections Index

## Dependency Graph

| Section | Depends On | Blocks | Parallelizable |
|---------|------------|--------|----------------|
| section-01-db-decomposition | - | 02 | Yes (with 03-05,08) |
| section-02-migration-versioning | 01 | 07, 09 | No |
| section-03-polygon-adapter | - | 06 | Yes (with 01,04,05,08) |
| section-04-yahoo-adapter | - | 06 | Yes (with 01,03,05,08) |
| section-05-fmp-adapter | - | 06 | Yes (with 01,03,04,08) |
| section-06-provider-chain | 03,04,05 | 10 | No |
| section-07-point-in-time | 02 | 10 | Yes (with 09) |
| section-08-staleness-tiering | - | 10 | Yes (with 01,03-05) |
| section-09-ohlcv-partitioning | 02 | 10 | Yes (with 07) |
| section-10-unit-tests | 06,07,08,09 | - | No |

## Execution Order

1. section-01-db-decomposition, section-03/04/05 adapters, section-08-staleness (parallel)
2. section-02-migration-versioning (after 01)
3. section-06-provider-chain (after 03-05), section-07-pit (after 02), section-09-partitioning (after 02)
4. section-10-unit-tests (final)

## Section Summaries

### section-01-db-decomposition
Extract connection.py and migrations.py from db.py. Backward-compatible re-exports via db/__init__.py.

### section-02-migration-versioning
schema_migrations table, migration runner, checksum tracking.

### section-03-polygon-adapter
Polygon.io REST adapter for OHLCV. Rate limiting, error handling.

### section-04-yahoo-adapter
yfinance wrapper with aggressive caching. Last-resort fallback only.

### section-05-fmp-adapter
Financial Modeling Prep adapter for fundamentals.

### section-06-provider-chain
ProviderChain class with ordered fallback. Integration into acquisition pipeline.

### section-07-point-in-time
Add available_date columns, backfill logic, pit_query() helper.

### section-08-staleness-tiering
Tiered thresholds (30min/8h/24h). Auto-disable collectors on stale data. Freshness report.

### section-09-ohlcv-partitioning
Partition by timeframe (list) then timestamp (range). Migration path with validation period.

### section-10-unit-tests
Tests for all new modules: adapters, chain, PIT, staleness, migration runner, partitioning.
