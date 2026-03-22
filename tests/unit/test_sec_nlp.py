# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for SEC NLP signals (EightKClassifier, MDADeltaAnalyzer, RiskFactorDeltaAnalyzer)."""

import pytest

from quantstack.core.features.sec_nlp import (
    EightKClassifier,
    MDADeltaAnalyzer,
    RiskFactorDeltaAnalyzer,
    _cosine_similarity,
    _tokenize,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_returns_dict(self):
        assert isinstance(_tokenize("hello world"), dict)

    def test_removes_short_words(self):
        result = _tokenize("the and are for hello")
        assert "the" not in result
        assert "and" not in result

    def test_word_frequency(self):
        result = _tokenize("profit profit revenue loss")
        assert result.get("profit", 0) == 2

    def test_empty_string(self):
        assert _tokenize("") == {}


class TestCosineSimilarity:
    def test_identical_dicts(self):
        d = {"revenue": 3, "profit": 2}
        assert abs(_cosine_similarity(d, d) - 1.0) < 1e-9

    def test_disjoint_dicts_zero(self):
        a = {"alpha": 1}
        b = {"beta": 1}
        assert _cosine_similarity(a, b) == 0.0

    def test_empty_dicts(self):
        assert _cosine_similarity({}, {}) == 0.0

    def test_range_zero_one(self):
        a = {"x": 3, "y": 1}
        b = {"x": 2, "z": 4}
        sim = _cosine_similarity(a, b)
        assert 0.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# EightKClassifier
# ---------------------------------------------------------------------------


class TestEightKClassifier:
    clf = EightKClassifier()

    def test_guidance_raise_detected(self):
        text = "The company raises full-year guidance and increases revenue expectations above consensus."
        events = self.clf.classify(text)
        cats = {e.category for e in events}
        assert "guidance_raise" in cats

    def test_guidance_lower_detected(self):
        text = (
            "The company lowered its fiscal year guidance below analyst expectations."
        )
        events = self.clf.classify(text)
        cats = {e.category for e in events}
        assert "guidance_lower" in cats

    def test_management_change_detected(self):
        text = "The board appoints a new Chief Executive Officer effective immediately."
        events = self.clf.classify(text)
        cats = {e.category for e in events}
        assert "management_change" in cats

    def test_regulatory_action_detected(self):
        text = "The SEC announced an enforcement investigation and class action lawsuit has been filed."
        events = self.clf.classify(text)
        cats = {e.category for e in events}
        assert "regulatory_action" in cats

    def test_buyback_detected(self):
        text = "The board declares a special dividend and announces a share repurchase program."
        events = self.clf.classify(text)
        cats = {e.category for e in events}
        assert "buyback_dividend" in cats

    def test_empty_text_no_events(self):
        events = self.clf.classify("")
        assert events == []

    def test_net_impact_positive_for_guidance_raise(self):
        text = "Raises full-year guidance above street expectations."
        assert self.clf.net_impact(text) == 1

    def test_net_impact_negative_for_guidance_lower(self):
        text = "Lowers fiscal guidance below analyst expectations."
        assert self.clf.net_impact(text) == -1

    def test_net_impact_negative_for_regulatory(self):
        text = "The SEC investigation and class action lawsuit have been initiated against the company."
        assert self.clf.net_impact(text) == -1

    def test_net_impact_zero_for_empty(self):
        assert self.clf.net_impact("") == 0

    def test_to_signal_dict_keys(self):
        text = "Raises full-year guidance. The CEO resigns."
        result = self.clf.to_signal_dict(text)
        expected_keys = {
            "sec_8k_guidance",
            "sec_8k_management",
            "sec_8k_ma",
            "sec_8k_regulatory",
            "sec_8k_buyback",
            "sec_8k_net_impact",
            "sec_8k_n_events",
        }
        assert expected_keys.issubset(result.keys())

    def test_to_signal_dict_guidance_value(self):
        text = "The company raises its full-year guidance."
        result = self.clf.to_signal_dict(text)
        assert result["sec_8k_guidance"] == 1

    def test_events_sorted_by_confidence_desc(self):
        text = "Raises full-year guidance. CEO resigns. SEC investigation."
        events = self.clf.classify(text)
        for i in range(len(events) - 1):
            assert events[i].confidence >= events[i + 1].confidence

    def test_confidence_in_zero_one(self):
        text = "Raises guidance. Lower guidance. Management change."
        events = self.clf.classify(text)
        for e in events:
            assert 0.0 <= e.confidence <= 1.0


# ---------------------------------------------------------------------------
# MDADeltaAnalyzer
# ---------------------------------------------------------------------------


class TestMDADeltaAnalyzer:
    analyzer = MDADeltaAnalyzer(similarity_threshold=0.7)

    def test_identical_texts_high_similarity(self):
        text = "Revenue growth improved significantly driven by strong demand in key markets."
        result = self.analyzer.analyze(text, text)
        assert result["mda_similarity"] > 0.95

    def test_different_texts_low_similarity(self):
        prior = "Revenue improved and growth accelerated across all segments."
        current = "Regulatory compliance and litigation risk increased significantly."
        result = self.analyzer.analyze(prior, current)
        assert result["mda_similarity"] < 0.5

    def test_significant_change_flagged(self):
        prior = "Revenue grew significantly across all product categories."
        current = "Litigation risk and compliance costs increased substantially."
        result = self.analyzer.analyze(prior, current)
        assert result["mda_significant_change"] == 1

    def test_no_significant_change_when_similar(self):
        text = "Revenue growth improved significantly driven by strong demand in key markets."
        result = self.analyzer.analyze(
            text, text + " Additionally, performance was solid."
        )
        assert result["mda_significant_change"] == 0

    def test_word_count_delta_positive_when_longer(self):
        prior = "Short text."
        current = "This is a much longer management discussion and analysis section with many more words."
        result = self.analyzer.analyze(prior, current)
        assert result["mda_word_count_delta"] > 0

    def test_sentiment_delta_positive_for_improving_tone(self):
        prior = "Revenue declined and challenges increased."
        current = "Strong growth momentum and robust expansion."
        result = self.analyzer.analyze(prior, current)
        assert result["mda_sentiment_delta"] > 0

    def test_sentiment_delta_negative_for_deteriorating_tone(self):
        prior = "Strong growth and solid performance."
        current = "Significant challenges and adverse conditions."
        result = self.analyzer.analyze(prior, current)
        assert result["mda_sentiment_delta"] < 0

    def test_returns_dict_with_expected_keys(self):
        result = self.analyzer.analyze("text a", "text b")
        for key in (
            "mda_similarity",
            "mda_significant_change",
            "mda_word_count_delta",
            "mda_sentiment_delta",
        ):
            assert key in result


# ---------------------------------------------------------------------------
# RiskFactorDeltaAnalyzer
# ---------------------------------------------------------------------------


_RISK_TEMPLATE = """Cybersecurity Risks
We face cybersecurity threats that could compromise our systems.

Regulatory Compliance Risks
Changes in regulations may increase our compliance costs significantly.

Competition Risks
Intense competition from established players may reduce our market share.
"""


class TestRiskFactorDeltaAnalyzer:
    analyzer = RiskFactorDeltaAnalyzer()

    def test_identical_filings_no_change(self):
        result = self.analyzer.analyze(_RISK_TEMPLATE, _RISK_TEMPLATE)
        assert result["rf_added"] == 0
        assert result["rf_removed"] == 0
        assert result["rf_net_change"] == 0
        assert result["rf_bearish_signal"] == 0

    def test_new_risk_factor_detected(self):
        added_risk = (
            _RISK_TEMPLATE
            + """

Supply Chain Risks
Disruptions in our supply chain could materially harm our operations.
"""
        )
        result = self.analyzer.analyze(_RISK_TEMPLATE, added_risk)
        assert result["rf_added"] >= 1
        assert result["rf_net_change"] > 0
        assert result["rf_bearish_signal"] == 1

    def test_removed_risk_factor_detected(self):
        shorter = """Cybersecurity Risks
We face cybersecurity threats that could compromise our systems.

Competition Risks
Intense competition from established players may reduce our market share.
"""
        result = self.analyzer.analyze(_RISK_TEMPLATE, shorter)
        assert result["rf_removed"] >= 1

    def test_count_fields_correct(self):
        result = self.analyzer.analyze(_RISK_TEMPLATE, _RISK_TEMPLATE)
        assert result["rf_count_prior"] == result["rf_count_current"]
        assert result["rf_count_prior"] > 0

    def test_bearish_signal_binary(self):
        result = self.analyzer.analyze(_RISK_TEMPLATE, _RISK_TEMPLATE)
        assert result["rf_bearish_signal"] in (0, 1)

    def test_returns_dict_with_expected_keys(self):
        result = self.analyzer.analyze(_RISK_TEMPLATE, _RISK_TEMPLATE)
        for key in (
            "rf_count_prior",
            "rf_count_current",
            "rf_added",
            "rf_removed",
            "rf_net_change",
            "rf_bearish_signal",
        ):
            assert key in result
