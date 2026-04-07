"""Prompt injection defense utilities.

Provides two primitives:
- safe_prompt(): parameterized prompt construction with XML-delimited,
  sanitized field values. Prevents injected content from being interpreted
  as instructions.
- detect_injection(): scans arbitrary text for known prompt injection
  patterns and returns structured findings.
"""

import re

from loguru import logger

# ---------------------------------------------------------------------------
# safe_prompt
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")
_XML_TAG_RE = re.compile(r"<[^>]+>")


def safe_prompt(template: str, **fields: str) -> str:
    """Build a prompt from *template* with XML-delimited, sanitized field values.

    Each ``{field_name}`` placeholder is replaced with
    ``<field_name>sanitized_value</field_name>`` where *sanitized_value* has
    all XML/HTML tags stripped.

    Raises ``KeyError`` if a placeholder in the template has no matching
    keyword argument -- we never silently produce empty tags.
    """
    placeholders = _PLACEHOLDER_RE.findall(template)

    # Validate: every placeholder must have a corresponding kwarg
    for name in placeholders:
        if name not in fields:
            raise KeyError(
                f"Placeholder '{{{name}}}' in template has no matching keyword argument. "
                f"Provided fields: {sorted(fields)}"
            )

    def _replace(match: re.Match) -> str:
        name = match.group(1)
        raw_value = fields[name]
        sanitized = _XML_TAG_RE.sub("", raw_value)
        return f"<{name}>{sanitized}</{name}>"

    return _PLACEHOLDER_RE.sub(_replace, template)


# ---------------------------------------------------------------------------
# detect_injection
# ---------------------------------------------------------------------------

# Each entry: (compiled_regex, severity, human-readable pattern name)
_INJECTION_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # High severity -- instruction override attempts
    (
        re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
        "high",
        "ignore_previous_instructions",
    ),
    (
        re.compile(r"disregard\s+(all\s+)?(prior|previous|above)\s+(instructions|context)", re.IGNORECASE),
        "high",
        "disregard_instructions",
    ),
    (
        re.compile(r"forget\s+(everything|all)\s+(above|before|previous)", re.IGNORECASE),
        "high",
        "forget_previous",
    ),
    # High severity -- role override prefixes at line start
    (
        re.compile(r"^system\s*:", re.IGNORECASE | re.MULTILINE),
        "high",
        "role_override_system",
    ),
    (
        re.compile(r"^assistant\s*:", re.IGNORECASE | re.MULTILINE),
        "high",
        "role_override_assistant",
    ),
    (
        re.compile(r"^human\s*:", re.IGNORECASE | re.MULTILINE),
        "high",
        "role_override_human",
    ),
    # Medium severity -- XML/HTML tags in data (potential delimiter escape)
    (
        re.compile(r"</?(?:system|instructions|prompt|assistant|human|user|context)\b[^>]*>", re.IGNORECASE),
        "medium",
        "xml_tag_in_data",
    ),
    # Low severity -- excessive delimiter patterns
    (
        re.compile(r"(?:^-{3,}\s*$.*){3,}", re.MULTILINE | re.DOTALL),
        "low",
        "excessive_dash_delimiters",
    ),
    (
        re.compile(r"(?:^#{3,}\s*$.*){3,}", re.MULTILINE | re.DOTALL),
        "low",
        "excessive_hash_delimiters",
    ),
]


def detect_injection(text: str, source: str = "") -> list[dict]:
    """Scan *text* for known prompt injection patterns.

    Returns a list of finding dicts, each with keys:
    ``pattern``, ``matched_text``, ``severity``, ``source``.

    Returns an empty list for clean data.
    """
    findings: list[dict] = []

    for pattern_re, severity, pattern_name in _INJECTION_PATTERNS:
        for match in pattern_re.finditer(text):
            finding = {
                "pattern": pattern_name,
                "matched_text": match.group(),
                "severity": severity,
                "source": source,
            }
            findings.append(finding)
            logger.warning(
                "Prompt injection detected: pattern={pattern} severity={severity} "
                "source={source} matched={matched!r}",
                pattern=pattern_name,
                severity=severity,
                source=source or "<unknown>",
                matched=match.group()[:100],
            )

    return findings
