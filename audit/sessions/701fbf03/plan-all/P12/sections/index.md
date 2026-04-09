<!-- PROJECT_CONFIG
runtime: python-uv
test_command: uv run pytest
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-asset-class-base  
section-02-futures-adapter  depends_on:section-01-asset-class-base
section-03-crypto-adapter  depends_on:section-01-asset-class-base
section-04-cross-asset-signals  depends_on:section-02-futures-adapter,section-03-crypto-adapter
section-05-risk-gate-multi-asset  depends_on:section-01-asset-class-base
section-06-schema-migrations
section-07-integration  depends_on:section-04-cross-asset-signals,section-05-risk-gate-multi-asset,section-06-schema-migrations
section-08-unit-tests  depends_on:section-02-futures-adapter,section-03-crypto-adapter,section-04-cross-asset-signals,section-05-risk-gate-multi-asset
END_MANIFEST -->

# P12 Sections Index
## Execution Order
1. section-01, section-06 (parallel — base class + schema)
2. section-02, section-03, section-05 (parallel — after 01)
3. section-04, section-08 (after 02+03)
4. section-07 (after all)
