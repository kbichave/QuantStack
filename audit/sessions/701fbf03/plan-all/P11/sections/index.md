<!-- PROJECT_CONFIG
runtime: python-uv
test_command: uv run pytest
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-congressional-collector
section-02-web-traffic-collector
section-03-job-postings-collector
section-04-patent-collector
section-05-data-providers  depends_on:section-01-congressional-collector,section-02-web-traffic-collector,section-03-job-postings-collector,section-04-patent-collector
section-06-synthesis-integration  depends_on:section-05-data-providers
section-07-unit-tests  depends_on:section-01-congressional-collector,section-02-web-traffic-collector,section-03-job-postings-collector,section-04-patent-collector
END_MANIFEST -->

# P11 Sections Index
## Execution Order
1. section-01 through section-04 (parallel — independent collectors)
2. section-05, section-07 (after 01-04)
3. section-06 (after 05)
