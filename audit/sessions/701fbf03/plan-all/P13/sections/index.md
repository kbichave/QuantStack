<!-- PROJECT_CONFIG
runtime: python-uv
test_command: uv run pytest
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-causal-discovery-engine
section-02-treatment-effects  depends_on:section-01-causal-discovery-engine
section-03-robustness-checks  depends_on:section-02-treatment-effects
section-04-causal-factor-library  depends_on:section-03-robustness-checks
section-05-causal-signal-collector  depends_on:section-04-causal-factor-library
section-06-counterfactual-analysis
section-07-research-graph-integration  depends_on:section-05-causal-signal-collector,section-06-counterfactual-analysis
section-08-unit-tests  depends_on:section-01-causal-discovery-engine,section-02-treatment-effects,section-03-robustness-checks,section-05-causal-signal-collector,section-06-counterfactual-analysis
END_MANIFEST -->

# P13 Sections Index
## Execution Order
1. section-01, section-06 (parallel — discovery + counterfactual are independent)
2. section-02 (after 01)
3. section-03 (after 02)
4. section-04 (after 03)
5. section-05, section-08 (after 04)
6. section-07 (after 05+06)
