# Implementation Findings

## Technical Decisions
| Decision | Rationale | Section |
|----------|-----------|---------|
| Used `?` placeholders in _upsert_metadata (matching existing pipeline code) vs `%s` in staleness.py (psycopg3 native) | acquisition_pipeline.py consistently uses `?`; staleness.py is new code using db_conn() directly which supports `%s` | section-01 |
| Used `datetime.now(UTC)` as last_ts for all non-OHLCV upserts | Represents "last successful fetch time" which is what staleness checks care about; avoids complex per-phase logic to extract max timestamps from varying dataframe schemas | section-01 |
| Boundary test uses 3d23h instead of exact 4d | Avoids sub-second timing race between test's now() and implementation's now() | section-01 |

## Issues Encountered
| Issue | Section | Resolution |
|-------|---------|------------|

## Resources
- Planning dir: CTO_ONBOARDING_AUDIT/specs/sessions/2fa6082b
- Test command: uv run pytest
