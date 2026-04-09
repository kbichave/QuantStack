<!-- PROJECT_CONFIG
runtime: python-uv
test_command: uv run pytest
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-conformal-prediction
section-02-position-sizing-ci  depends_on:section-01-conformal-prediction
section-03-transformer-forecaster  
section-04-transformer-signal-collector  depends_on:section-03-transformer-forecaster
section-05-gnn-market-graph  
section-06-gnn-contagion-collector  depends_on:section-05-gnn-market-graph
section-07-deep-hedging  
section-08-financial-nlp
section-09-schema-migrations
section-10-unit-tests  depends_on:section-01-conformal-prediction,section-03-transformer-forecaster,section-05-gnn-market-graph,section-07-deep-hedging,section-08-financial-nlp
END_MANIFEST -->

# P14 Sections Index
## Execution Order
1. section-01, section-03, section-05, section-07, section-08, section-09 (parallel — independent model types + schema)
2. section-02 (after 01), section-04 (after 03), section-06 (after 05)
3. section-10 (after 01, 03, 05, 07, 08)
