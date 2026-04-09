<!-- PROJECT_CONFIG
runtime: python-uv
test_command: uv run pytest
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-vol-arb-strategy
section-02-dispersion-trading
section-03-gamma-scalping  depends_on:section-01-vol-arb-strategy
section-04-condor-harvesting
section-05-hedging-extensions  depends_on:section-03-gamma-scalping
section-06-market-making-agent  depends_on:section-01-vol-arb-strategy,section-04-condor-harvesting
section-07-pnl-attribution-extensions  depends_on:section-06-market-making-agent
section-08-unit-tests  depends_on:section-01-vol-arb-strategy,section-02-dispersion-trading,section-03-gamma-scalping,section-04-condor-harvesting,section-05-hedging-extensions,section-06-market-making-agent
END_MANIFEST -->

# P08 Sections Index

## Execution Order
1. section-01, section-02, section-04 (parallel — independent strategies)
2. section-03 (after 01), section-05 (after 03)
3. section-06 (after 01, 04), section-07 (after 06)
4. section-08 (final)
