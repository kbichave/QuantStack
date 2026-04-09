<!-- PROJECT_CONFIG
runtime: python-uv
test_command: uv run pytest
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-portfolio-opt-env
section-02-execution-env
section-03-strategy-select-env
section-04-training-infra  depends_on:section-01-portfolio-opt-env
section-05-finrl-tools  depends_on:section-04-training-infra
section-06-signal-integration  depends_on:section-05-finrl-tools
section-07-safety-gates  depends_on:section-06-signal-integration
section-08-unit-tests  depends_on:section-01-portfolio-opt-env,section-02-execution-env,section-03-strategy-select-env,section-05-finrl-tools,section-07-safety-gates
END_MANIFEST -->

# P09 Sections Index

## Execution Order
1. section-01, section-02, section-03 (parallel — independent environments)
2. section-04 (after 01), section-05 (after 04)
3. section-06 (after 05), section-07 (after 06)
4. section-08 (final)
