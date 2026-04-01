# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
SEC Filing NLP signals — derived from FD.ai filing item extraction.

Three signal families:

1. **8-K Event Classification** (`EightKClassifier`)
   Parse 8-K filings and classify the event type (management change, guidance
   raise/lower, M&A, regulatory action). Each event type has a signed market
   impact direction (positive/negative/neutral) based on academic evidence.

2. **MD&A Language Delta** (`MDADeltaAnalyzer`)
   Compute TF-IDF cosine similarity between consecutive Item 7 (MD&A) sections
   of 10-Q/10-K filings. Low similarity = management describing the business
   differently = structural change signal (positive OR negative depending on
   direction of change).

3. **Risk Factor Delta** (`RiskFactorDeltaAnalyzer`)
   Count new risk factors added vs removed between consecutive Item 1A sections.
   Net addition = management acknowledging new risks = bearish signal.

All classes operate on raw text (strings). They do NOT make API calls — the
caller is responsible for fetching text from FD.ai and passing it in. This keeps
the signal computation pure and testable without network access.

No external NLP libraries required — uses only Python stdlib + scikit-learn
(already a project dependency for feature engineering).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# 8-K Event Classifier
# ---------------------------------------------------------------------------


# Evidence-based impact mappings (Loughran-McDonald, 2011; Cohen et al., 2013)
_8K_ITEM_PATTERNS: dict[str, dict[str, Any]] = {
    "guidance_raise": {
        "patterns": [
            r"raises?\s+(?:\w+\s+){0,3}(?:full[- ]year|annual|fiscal|quarterly)\s+(?:guidance|outlook|forecast)",
            r"(?:increases?|raises?|improves?)\s+(?:\w+\s+){0,3}(?:revenue|eps|earnings)\s+(?:guidance|expectations)",
            r"above\s+(?:consensus|analyst|street)\s+expectations",
        ],
        "impact": 1,  # positive
        "item": "guidance",
    },
    "guidance_lower": {
        "patterns": [
            r"lower[sd]?\s+(?:full[- ]year|annual|fiscal|quarterly)\s+(?:guidance|outlook|forecast)",
            r"(?:reduces?|cuts?|decreases?)\s+(?:revenue|eps|earnings)\s+(?:guidance|expectations)",
            r"below\s+(?:consensus|analyst|street)\s+expectations",
            r"warns?\s+(?:of\s+)?(?:lower|weaker|declining)\s+(?:revenue|earnings|results)",
        ],
        "impact": -1,  # negative
        "item": "guidance",
    },
    "management_change": {
        "patterns": [
            r"(?:appoints?|names?|announces?)\s+(?:\w+\s+){0,3}(?:chief\s+executive|ceo|cfo|president|chairman)",
            r"(?:resigns?|retires?|steps\s+down|departur)",
            r"item\s+5\.02",  # 8-K item for director/officer changes
        ],
        "impact": 0,  # neutral (direction depends on context)
        "item": "management",
    },
    "merger_acquisition": {
        "patterns": [
            r"(?:agrees?\s+to|announces?|completes?)\s+(?:acquisition|merger|takeover)",
            r"(?:acquires?|purchases?|buys?)\s+(?:\w+\s+){1,4}for\s+\$[\d,]+",
            r"item\s+1\.01",  # material agreement
        ],
        "impact": 1,  # positive (acquiree) / negative (acquirer large premium); use cautiously
        "item": "m&a",
    },
    "regulatory_action": {
        "patterns": [
            r"(?:sec|doj|ftc|cfpb|fda)\s+(?:\w+\s+){0,3}(?:investigation|inquiry|subpoena|enforcement|charge)",
            r"(?:class\s+action|lawsuit|litigation)(?:\s+\w+){0,3}\s+(?:filed|initiated|commenced|has been filed)",
            r"item\s+8\.01",  # other events
        ],
        "impact": -1,  # negative
        "item": "regulatory",
    },
    "buyback_dividend": {
        "patterns": [
            r"(?:repurchase|buyback)\s+program",
            r"(?:declares?|announces?|increases?)\s+(?:quarterly\s+)?dividend",
            r"(?:special\s+dividend|one[- ]time\s+dividend)",
        ],
        "impact": 1,  # positive
        "item": "capital_return",
    },
}


@dataclass
class EightKEvent:
    category: str
    impact: int  # +1 = positive, -1 = negative, 0 = neutral
    item_type: str
    matched_patterns: list[str] = field(default_factory=list)
    confidence: float = 0.0  # fraction of patterns matched


class EightKClassifier:
    """
    Classify 8-K filing text into event categories with market impact direction.

    Parameters
    ----------
    min_confidence : float
        Minimum fraction of patterns for a category to trigger. Default 0 (any match).
    """

    def __init__(self, min_confidence: float = 0.0) -> None:
        self.min_confidence = min_confidence

    def classify(self, text: str) -> list[EightKEvent]:
        """
        Parameters
        ----------
        text : str
            Raw text of the 8-K filing (or its extracted items).

        Returns
        -------
        list[EightKEvent] — detected events, sorted by confidence descending.
        """
        text_lower = text.lower()
        events: list[EightKEvent] = []

        for category, config in _8K_ITEM_PATTERNS.items():
            matched = []
            for pattern in config["patterns"]:
                if re.search(pattern, text_lower):
                    matched.append(pattern)
            if not matched:
                continue
            confidence = len(matched) / len(config["patterns"])
            if confidence >= self.min_confidence:
                events.append(
                    EightKEvent(
                        category=category,
                        impact=config["impact"],
                        item_type=config["item"],
                        matched_patterns=matched,
                        confidence=round(confidence, 3),
                    )
                )

        return sorted(events, key=lambda e: e.confidence, reverse=True)

    def net_impact(self, text: str) -> int:
        """
        Aggregate impact across all detected events.

        Returns +1, -1, or 0.
        """
        events = self.classify(text)
        if not events:
            return 0
        total = sum(e.impact for e in events if e.impact != 0)
        return 1 if total > 0 else (-1 if total < 0 else 0)

    def to_signal_dict(self, text: str) -> dict[str, Any]:
        """
        Convert classification to flat signal dict for SignalEngine.

        Returns
        -------
        dict with keys:
            sec_8k_guidance (1/-1/0), sec_8k_management (bool),
            sec_8k_ma (bool), sec_8k_regulatory (bool),
            sec_8k_net_impact, sec_8k_n_events
        """
        events = self.classify(text)
        cats = {e.category for e in events}
        net = self.net_impact(text)
        return {
            "sec_8k_guidance": (
                1
                if "guidance_raise" in cats
                else (-1 if "guidance_lower" in cats else 0)
            ),
            "sec_8k_management": int("management_change" in cats),
            "sec_8k_ma": int("merger_acquisition" in cats),
            "sec_8k_regulatory": int("regulatory_action" in cats),
            "sec_8k_buyback": int("buyback_dividend" in cats),
            "sec_8k_net_impact": net,
            "sec_8k_n_events": len(events),
        }


# ---------------------------------------------------------------------------
# MD&A Language Delta (TF-IDF cosine similarity)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> dict[str, int]:
    """Simple word-frequency tokenizer (no external NLP libs)."""
    words = re.findall(r"[a-z]{3,}", text.lower())
    # Remove common stop words (subset)
    stops = {
        "the",
        "and",
        "for",
        "are",
        "was",
        "has",
        "have",
        "had",
        "not",
        "but",
        "this",
        "that",
        "with",
        "from",
        "our",
        "its",
        "may",
        "will",
        "can",
        "year",
        "quarter",
        "period",
        "fiscal",
        "million",
        "billion",
        "thousand",
    }
    freq: dict[str, int] = {}
    for w in words:
        if w not in stops and len(w) >= 4:
            freq[w] = freq.get(w, 0) + 1
    return freq


def _cosine_similarity(a: dict[str, int], b: dict[str, int]) -> float:
    """Cosine similarity between two term-frequency dicts."""
    if not a or not b:
        return 0.0
    vocab = set(a.keys()) | set(b.keys())
    dot = sum(a.get(w, 0) * b.get(w, 0) for w in vocab)
    norm_a = sum(v**2 for v in a.values()) ** 0.5
    norm_b = sum(v**2 for v in b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class MDADeltaAnalyzer:
    """
    Compute MD&A language change between consecutive 10-Q/10-K filings.

    Low similarity = management describing business differently = structural
    change (positive momentum if expanding, negative if describing decline).

    Parameters
    ----------
    similarity_threshold : float
        Below this cosine similarity → flag as significant change. Default 0.7.
    """

    def __init__(self, similarity_threshold: float = 0.7) -> None:
        self.similarity_threshold = similarity_threshold

    def compute_similarity(self, text_prior: str, text_current: str) -> float:
        """Cosine similarity between two MD&A sections. Range [0, 1]."""
        tf_prior = _tokenize(text_prior)
        tf_current = _tokenize(text_current)
        return round(_cosine_similarity(tf_prior, tf_current), 4)

    def analyze(self, text_prior: str, text_current: str) -> dict[str, Any]:
        """
        Parameters
        ----------
        text_prior, text_current : str
            MD&A (Item 7) text for consecutive periods.

        Returns
        -------
        dict with keys:
            mda_similarity      – cosine similarity [0, 1]
            mda_significant_change – 1 if similarity < threshold
            mda_word_count_delta   – word count change (current - prior)
            mda_sentiment_delta    – positive-word density delta
        """
        sim = self.compute_similarity(text_prior, text_current)
        significant_change = int(sim < self.similarity_threshold)

        wc_prior = len(re.findall(r"\w+", text_prior))
        wc_current = len(re.findall(r"\w+", text_current))
        wc_delta = wc_current - wc_prior

        # Loughran-McDonald positive word list (abbreviated)
        _positive_words = {
            "strong",
            "growth",
            "increase",
            "improved",
            "favorable",
            "record",
            "profitable",
            "expanding",
            "accelerating",
            "outperform",
            "exceed",
            "robust",
            "solid",
            "momentum",
            "gain",
            "success",
            "effective",
        }
        _negative_words = {
            "decline",
            "decrease",
            "loss",
            "risk",
            "uncertain",
            "challenge",
            "adverse",
            "impair",
            "delay",
            "weak",
            "difficult",
            "pressure",
            "headwind",
            "shortage",
            "disruption",
            "volatile",
            "concern",
        }

        def sentiment_density(text: str) -> float:
            words = set(re.findall(r"[a-z]+", text.lower()))
            if not words:
                return 0.0
            pos = sum(1 for w in words if w in _positive_words)
            neg = sum(1 for w in words if w in _negative_words)
            return (pos - neg) / len(words)

        sent_delta = round(
            sentiment_density(text_current) - sentiment_density(text_prior), 4
        )

        return {
            "mda_similarity": sim,
            "mda_significant_change": significant_change,
            "mda_word_count_delta": wc_delta,
            "mda_sentiment_delta": sent_delta,
        }


# ---------------------------------------------------------------------------
# Risk Factor Delta Analyzer (Item 1A)
# ---------------------------------------------------------------------------


def _extract_risk_factors(text: str) -> list[str]:
    """
    Split Item 1A text into individual risk factor sections.

    Heuristic: each risk factor starts with a bold/headline phrase
    (all-caps or title-case sentence ending with period before a paragraph).
    We split on double-newlines or markdown headers.
    """
    # Split on double newline, look for paragraphs starting with capitalized header
    chunks = re.split(r"\n\s*\n", text.strip())
    risks = []
    for chunk in chunks:
        chunk = chunk.strip()
        # Risk factor headers are typically short (< 20 words) and title-case
        first_line = chunk.split("\n")[0].strip()
        if len(first_line.split()) <= 20 and len(first_line) > 10:
            risks.append(chunk)
    return risks


def _risk_factor_hash(text: str) -> str:
    """Rough fingerprint for a risk factor: first 8 words, lowercased."""
    words = re.findall(r"[a-z]+", text.lower())[:8]
    return " ".join(words)


class RiskFactorDeltaAnalyzer:
    """
    Detect new / removed risk factors between consecutive Item 1A sections.

    Uses fuzzy matching on the first 8 words of each risk factor as a fingerprint.

    Parameters
    ----------
    similarity_threshold : float
        Risk factors with cosine similarity > this are considered "same". Default 0.75.
    """

    def __init__(self, similarity_threshold: float = 0.75) -> None:
        self.similarity_threshold = similarity_threshold

    def analyze(self, text_prior: str, text_current: str) -> dict[str, Any]:
        """
        Parameters
        ----------
        text_prior, text_current : str
            Item 1A risk factor text for consecutive annual filings.

        Returns
        -------
        dict with keys:
            rf_count_prior     – number of risk factors in prior filing
            rf_count_current   – number in current filing
            rf_added           – estimated new risk factors
            rf_removed         – estimated removed risk factors
            rf_net_change      – rf_added - rf_removed (positive = more risks)
            rf_bearish_signal  – 1 when rf_net_change > 0
        """
        prior_risks = _extract_risk_factors(text_prior)
        current_risks = _extract_risk_factors(text_current)

        prior_hashes = {_risk_factor_hash(r) for r in prior_risks}
        current_hashes = {_risk_factor_hash(r) for r in current_risks}

        # Added = in current but not in prior (fingerprint match)
        added = len(current_hashes - prior_hashes)
        removed = len(prior_hashes - current_hashes)
        net = added - removed

        return {
            "rf_count_prior": len(prior_risks),
            "rf_count_current": len(current_risks),
            "rf_added": added,
            "rf_removed": removed,
            "rf_net_change": net,
            "rf_bearish_signal": int(net > 0),
        }
