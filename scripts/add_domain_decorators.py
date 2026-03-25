#!/usr/bin/env python3
"""Add @domain() decorators and imports to all MCP tool files.

This script reads each tool file, adds the domain import if missing,
and inserts @domain(Domain.XXX) above each @mcp.tool() decorator
based on a predefined file→domain mapping.

Safe to run multiple times — skips files/tools that already have @domain.
"""

import re
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "src" / "quantstack" / "mcp" / "tools"

# File → Domain(s) mapping.  Each tool in the file gets tagged with these domains.
# Cross-cutting tools (in analysis.py, signal.py) are handled separately below.
FILE_DOMAINS: dict[str, str] = {
    # Execution server
    "execution.py": "Domain.EXECUTION",
    "options_execution.py": "Domain.EXECUTION",
    "alerts.py": "Domain.EXECUTION",
    # coordination.py already done manually

    # Portfolio server
    "portfolio.py": "Domain.PORTFOLIO",
    "attribution.py": "Domain.PORTFOLIO",
    "feedback.py": "Domain.PORTFOLIO, Domain.SIGNALS",

    # Data server
    "qc_data.py": "Domain.DATA",
    "qc_indicators.py": "Domain.DATA",
    "qc_fundamentals.py": "Domain.DATA",
    "qc_fundamentals_av.py": "Domain.DATA",
    "qc_market.py": "Domain.DATA",
    "qc_acquisition.py": "Domain.DATA",

    # Research server
    "backtesting.py": "Domain.RESEARCH",
    "qc_backtesting.py": "Domain.RESEARCH",
    "qc_research.py": "Domain.RESEARCH",
    "strategy.py": "Domain.RESEARCH",
    "meta.py": "Domain.RESEARCH",
    "learning.py": "Domain.RESEARCH",
    "decoder.py": "Domain.RESEARCH",

    # Options server
    "qc_options.py": "Domain.OPTIONS",

    # ML server
    "ml.py": "Domain.ML",

    # FinRL server
    "finrl_tools.py": "Domain.FINRL",

    # Intel server
    "nlp.py": "Domain.INTEL",
    "cross_domain.py": "Domain.INTEL",
    # capitulation.py, institutional_accumulation.py, macro_signals.py already done

    # Risk server
    "qc_risk.py": "Domain.RISK",
}

# Files already handled (have @domain() on all tools)
SKIP_FILES = {
    "_registry.py",
    "__init__.py",
    "_impl.py",
    "_helpers.py",
    "analysis.py",     # done by agent
    "signal.py",       # done by agent
    "coordination.py", # done manually
    "capitulation.py", # done by agent
    "institutional_accumulation.py",  # done by agent
    "macro_signals.py",  # done by agent
}

IMPORT_LINES = (
    "from quantstack.mcp.domains import Domain\n"
    "from quantstack.mcp.tools._registry import domain\n"
)


def process_file(filepath: Path, domain_str: str) -> int:
    """Add @domain() + imports to a tool file.  Returns count of decorators added."""
    text = filepath.read_text()

    # Skip if already has domain import
    has_import = "from quantstack.mcp.tools._registry import domain" in text

    # Add imports after the last existing import block
    if not has_import:
        # Find last import line
        lines = text.split("\n")
        last_import_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and not stripped.startswith("from __future__"):
                last_import_idx = i
            # Stop at first function/class def or decorator
            if stripped.startswith(("def ", "async def ", "class ", "@")) and i > 10:
                break

        lines.insert(last_import_idx + 1, IMPORT_LINES)
        text = "\n".join(lines)

    # Add @domain() above each @mcp.tool() that doesn't already have one
    count = 0
    lines = text.split("\n")
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check if this line is @mcp.tool() and previous line is NOT @domain
        if stripped.startswith("@mcp.tool("):
            # Look back to see if @domain is already there
            prev_non_empty = ""
            for j in range(len(new_lines) - 1, -1, -1):
                if new_lines[j].strip():
                    prev_non_empty = new_lines[j].strip()
                    break

            if not prev_non_empty.startswith("@domain("):
                # Get indentation from the @mcp.tool line
                indent = line[: len(line) - len(line.lstrip())]
                new_lines.append(f"{indent}@domain({domain_str})")
                count += 1

        new_lines.append(line)
        i += 1

    filepath.write_text("\n".join(new_lines))
    return count


def main():
    total = 0
    for filename, domain_str in sorted(FILE_DOMAINS.items()):
        filepath = TOOLS_DIR / filename
        if not filepath.exists():
            print(f"  SKIP {filename} (not found)")
            continue
        if filename in SKIP_FILES:
            print(f"  SKIP {filename} (already done)")
            continue

        count = process_file(filepath, domain_str)
        total += count
        print(f"  {filename}: {count} @domain() decorators added")

    print(f"\nTotal: {total} decorators added")


if __name__ == "__main__":
    main()
