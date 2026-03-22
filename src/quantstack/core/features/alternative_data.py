# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Alternative Data Signal Frameworks.

These classes define the interface contracts for signals that require
data providers not yet integrated. Each class:
  1. Documents the exact data schema required.
  2. Raises `DataNotAvailableError` if used without real data.
  3. Provides an OHLCV-based approximation where possible.
  4. Is drop-in compatible once the real data feed is connected.

Data provider map
-----------------
Signal class            | Data source             | Availability
------------------------|-------------------------|--------------
DarkPoolSignals         | FINRA ADF / Polygon.io  | Pending subscription
BorrowRateSignals       | Ortex / Fintel          | Pending subscription
ShortInterestSignals    | Ortex / Fintel          | Pending subscription
EarningsTranscriptNLP   | FD.ai                   | April 2026

Integration pattern
-------------------
When the data feed goes live:
    1. Implement the `_fetch_real_data()` method in the relevant class.
    2. Remove the `_data_available = False` guard.
    3. The rest of the pipeline (normalisation, signal generation) is already built.

No code changes are needed outside these classes when the data arrives.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


class DataNotAvailableError(Exception):
    """Raised when a signal class requires data that is not yet integrated."""


# ---------------------------------------------------------------------------
# Dark Pool Signals
# ---------------------------------------------------------------------------


class DarkPoolSignals:
    """
    Dark pool and off-exchange volume signals.

    Dark pools are private exchanges where institutional orders execute away
    from lit markets. High dark pool volume relative to total volume signals
    large institutional participation — often predictive of near-term moves.

    Real data source: FINRA ADF (Alternative Display Facility) or Polygon.io
    off-exchange volume endpoint. Free via FINRA weekly; real-time via Polygon.

    Data contract (when available)
    --------------------------------
    Input DataFrame must contain:
        dark_pool_volume   – off-exchange volume for the symbol (shares)
        total_volume       – total traded volume (dark + lit)
        dark_pool_price    – volume-weighted average dark pool print price
        timestamp          – bar timestamp (DatetimeIndex)

    OHLCV approximation
    --------------------
    No reliable OHLCV proxy exists for dark pool activity. This class raises
    DataNotAvailableError unless real data is provided.

    Parameters
    ----------
    high_dark_threshold : float
        Fraction of total volume above which dark pool is "elevated". Default 0.4.
    window : int
        Rolling window for z-score computation. Default 20.
    """

    _data_available: bool = False

    def __init__(self, high_dark_threshold: float = 0.4, window: int = 20) -> None:
        self.high_dark_threshold = high_dark_threshold
        self.window = window

    def compute(self, dark_pool_data: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        dark_pool_data : pd.DataFrame
            Required columns: dark_pool_volume, total_volume, dark_pool_price.

        Returns
        -------
        pd.DataFrame with columns:
            dark_pct       – dark_pool_volume / total_volume [0, 1]
            dark_zscore    – z-score of dark_pct vs rolling history
            dark_elevated  – 1 when dark_pct > high_dark_threshold
            price_premium  – dark_pool_price vs prior close (positive = above market)
        """
        required = {"dark_pool_volume", "total_volume", "dark_pool_price"}
        if not required.issubset(dark_pool_data.columns):
            raise DataNotAvailableError(
                f"DarkPoolSignals requires columns {required}. "
                "Subscribe to FINRA ADF or Polygon.io off-exchange endpoint."
            )

        total = dark_pool_data["total_volume"].replace(0, np.nan)
        dark_pct = dark_pool_data["dark_pool_volume"] / total

        roll_mean = dark_pct.rolling(self.window).mean()
        roll_std = dark_pct.rolling(self.window).std().replace(0, np.nan)
        dark_zscore = (dark_pct - roll_mean) / roll_std

        dark_elevated = (dark_pct > self.high_dark_threshold).astype(int)

        # Price premium vs prior close (requires close column if available)
        if "close" in dark_pool_data.columns:
            price_premium = (
                dark_pool_data["dark_pool_price"] - dark_pool_data["close"].shift(1)
            ) / dark_pool_data["close"].shift(1)
        else:
            price_premium = pd.Series(np.nan, index=dark_pool_data.index)

        return pd.DataFrame(
            {
                "dark_pct": dark_pct,
                "dark_zscore": dark_zscore,
                "dark_elevated": dark_elevated,
                "price_premium": price_premium,
            },
            index=dark_pool_data.index,
        )


# ---------------------------------------------------------------------------
# Borrow Rate Signals
# ---------------------------------------------------------------------------


class BorrowRateSignals:
    """
    Securities borrow rate (HTB cost) signals for short interest analysis.

    Rising borrow rates signal increasing short interest and hard-to-borrow
    status — bearish for the stock in the short term (supply/demand), but
    can also signal squeeze potential when rates are extreme.

    Real data source: Ortex, Fintel, or FactSet short interest data.
    Updated daily; real-time borrow rate requires prime brokerage relationship.

    Data contract (when available)
    --------------------------------
    Input DataFrame must contain:
        borrow_rate_pct  – annualised borrow cost as percentage (e.g. 5.0 = 5%)
        short_interest   – total shares short
        shares_float     – shares available to trade
        days_to_cover    – short_interest / average_daily_volume
        timestamp        – bar timestamp (DatetimeIndex)

    Parameters
    ----------
    htb_threshold : float
        Borrow rate above which stock is "hard to borrow". Default 10.0 (%).
    squeeze_threshold : float
        Days-to-cover above which squeeze risk is flagged. Default 5.0.
    window : int
        Rolling window for trend detection. Default 20.
    """

    _data_available: bool = False

    def __init__(
        self,
        htb_threshold: float = 10.0,
        squeeze_threshold: float = 5.0,
        window: int = 20,
    ) -> None:
        self.htb_threshold = htb_threshold
        self.squeeze_threshold = squeeze_threshold
        self.window = window

    def compute(self, borrow_data: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        borrow_data : pd.DataFrame
            Required columns: borrow_rate_pct, short_interest, shares_float, days_to_cover.

        Returns
        -------
        pd.DataFrame with columns:
            short_float_pct   – short_interest / shares_float [0, 1]
            borrow_rate_pct   – annualised borrow rate
            borrow_rising     – 1 when borrow_rate trending up (5-day slope > 0)
            hard_to_borrow    – 1 when borrow_rate > htb_threshold
            squeeze_risk      – 1 when days_to_cover > squeeze_threshold
            borrow_zscore     – z-score of borrow_rate vs rolling history
        """
        required = {
            "borrow_rate_pct",
            "short_interest",
            "shares_float",
            "days_to_cover",
        }
        if not required.issubset(borrow_data.columns):
            raise DataNotAvailableError(
                f"BorrowRateSignals requires columns {required}. "
                "Subscribe to Ortex or Fintel short interest data."
            )

        rate = borrow_data["borrow_rate_pct"]
        float_ = borrow_data["shares_float"].replace(0, np.nan)
        short_float_pct = borrow_data["short_interest"] / float_

        roll_mean = rate.rolling(self.window).mean()
        roll_std = rate.rolling(self.window).std().replace(0, np.nan)
        borrow_zscore = (rate - roll_mean) / roll_std

        # Trend: 5-day linear slope
        borrow_rising = (
            rate.rolling(5).mean() > rate.rolling(5).mean().shift(5)
        ).astype(int)

        return pd.DataFrame(
            {
                "short_float_pct": short_float_pct,
                "borrow_rate_pct": rate,
                "borrow_rising": borrow_rising,
                "hard_to_borrow": (rate > self.htb_threshold).astype(int),
                "squeeze_risk": (
                    borrow_data["days_to_cover"] > self.squeeze_threshold
                ).astype(int),
                "borrow_zscore": borrow_zscore,
            },
            index=borrow_data.index,
        )


# ---------------------------------------------------------------------------
# Short Interest Signals
# ---------------------------------------------------------------------------


class ShortInterestSignals:
    """
    Short interest momentum and crowding signals.

    Short interest (SI) data is published bi-weekly by exchanges (FINRA/NASDAQ/NYSE).
    High SI + price rising = short squeeze potential.
    High SI + price falling = momentum fuel (shorts adding).

    Real data source: Ortex (daily estimate), Fintel, or FINRA bi-weekly.
    Free FINRA data: https://www.finra.org/investors/learn-to-invest/advanced-investing/short-sale-statistics

    Data contract (when available)
    --------------------------------
    Input DataFrame must contain:
        short_interest    – total shares short
        shares_float      – float shares
        avg_daily_volume  – 20-day average daily volume
        reporting_date    – date (DatetimeIndex)

    Parameters
    ----------
    high_si_threshold : float
        Short interest as % of float above which crowding is flagged. Default 0.2 (20%).
    squeeze_dtc_threshold : float
        Days-to-cover above which squeeze risk is high. Default 5.0.
    """

    _data_available: bool = False

    def __init__(
        self,
        high_si_threshold: float = 0.2,
        squeeze_dtc_threshold: float = 5.0,
    ) -> None:
        self.high_si_threshold = high_si_threshold
        self.squeeze_dtc_threshold = squeeze_dtc_threshold

    def compute(self, si_data: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        si_data : pd.DataFrame
            Required columns: short_interest, shares_float, avg_daily_volume.

        Returns
        -------
        pd.DataFrame with columns:
            si_pct            – short_interest / shares_float [0, 1]
            days_to_cover     – short_interest / avg_daily_volume
            si_crowded        – 1 when si_pct > high_si_threshold
            squeeze_setup     – 1 when days_to_cover > threshold AND si rising
            si_momentum       – change in si_pct vs prior period (positive = shorts adding)
        """
        required = {"short_interest", "shares_float", "avg_daily_volume"}
        if not required.issubset(si_data.columns):
            raise DataNotAvailableError(
                f"ShortInterestSignals requires columns {required}. "
                "Subscribe to Ortex, Fintel, or download FINRA bi-weekly data."
            )

        float_ = si_data["shares_float"].replace(0, np.nan)
        adv = si_data["avg_daily_volume"].replace(0, np.nan)

        si_pct = si_data["short_interest"] / float_
        days_to_cover = si_data["short_interest"] / adv
        si_momentum = si_pct.diff()
        si_crowded = (si_pct > self.high_si_threshold).astype(int)
        si_rising = (si_momentum > 0).astype(int)
        squeeze_setup = (
            (days_to_cover > self.squeeze_dtc_threshold) & si_rising.astype(bool)
        ).astype(int)

        return pd.DataFrame(
            {
                "si_pct": si_pct,
                "days_to_cover": days_to_cover,
                "si_crowded": si_crowded,
                "squeeze_setup": squeeze_setup,
                "si_momentum": si_momentum,
            },
            index=si_data.index,
        )


# ---------------------------------------------------------------------------
# Earnings Transcript NLP
# ---------------------------------------------------------------------------


# Loughran-McDonald word lists (abbreviated — expand when FD.ai delivers transcripts)
_LM_POSITIVE = {
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
    "confident",
    "outstanding",
    "exceptional",
    "excellent",
}

_LM_NEGATIVE = {
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
    "disappointing",
    "difficult",
    "cautious",
    "challenging",
}

_UNCERTAINTY = {
    "may",
    "might",
    "could",
    "possibly",
    "uncertain",
    "unclear",
    "expect",
    "anticipate",
    "believe",
    "appears",
    "seems",
    "approximately",
}


class EarningsTranscriptNLP:
    """
    Earnings call transcript NLP signals.

    Analyses CEO/CFO language for tone (positive/negative), uncertainty,
    forward guidance signals, and hedging language.

    **Data availability: FD.ai April 2026.**

    This class is fully implemented and ready to receive transcript text.
    When FD.ai's transcript endpoint goes live, pass the raw text to `analyze()`.

    Signals derived (Loughran-McDonald, 2011 word lists):
    - **tone_score**: (positive_words - negative_words) / total_words
    - **uncertainty_score**: fraction of uncertainty words
    - **guidance_polarity**: regex-based forward guidance direction (+1/-1/0)
    - **hedging_ratio**: fraction of sentences containing hedge words (may/might/could)
    - **qa_tone_delta**: tone change from prepared remarks to Q&A section
      (Q&A is less scripted → more informative)

    Parameters
    ----------
    min_confidence : float
        Minimum word density for tone to be considered meaningful. Default 0.001.
    """

    _data_available: bool = True  # Text processing doesn't need external data

    def __init__(self, min_confidence: float = 0.001) -> None:
        self.min_confidence = min_confidence

    @staticmethod
    def _word_density(text: str, word_set: set[str]) -> float:
        """Fraction of unique words in text that appear in word_set."""
        words = set(re.findall(r"[a-z]{3,}", text.lower()))
        if not words:
            return 0.0
        return sum(1 for w in words if w in word_set) / len(words)

    @staticmethod
    def _split_qa(transcript: str) -> tuple[str, str]:
        """
        Split transcript into prepared remarks and Q&A section.

        Heuristic: Q&A typically starts with 'Operator', 'Question-and-Answer',
        'Q&A', or 'Analyst:'.
        """
        qa_markers = [
            r"(?i)(?:operator|question[- ]and[- ]answer|q&a session|questions?\s+and\s+answers?)",
            r"(?i)(?:your\s+first\s+question|first\s+question\s+comes?\s+from)",
        ]
        for pattern in qa_markers:
            match = re.search(pattern, transcript)
            if match:
                split_pos = match.start()
                return transcript[:split_pos], transcript[split_pos:]
        # No Q&A marker found — treat all as prepared remarks
        return transcript, ""

    def analyze(self, transcript: str) -> dict[str, Any]:
        """
        Parameters
        ----------
        transcript : str
            Raw earnings call transcript text.
            When FD.ai API is live, retrieve via:
            `client.get_earnings_transcript(ticker, period_of_report)`

        Returns
        -------
        dict with keys:
            tone_score          – (pos - neg) / total_words (range: -1 to +1)
            uncertainty_score   – uncertainty word density
            guidance_polarity   – +1 raise, -1 lower, 0 neutral
            hedging_ratio       – fraction of sentences with hedge words
            qa_tone_delta       – Q&A tone minus prepared remarks tone
            word_count          – total word count
            positive_count      – count of positive word matches
            negative_count      – count of negative word matches
            has_qa_section      – 1 if Q&A detected
        """
        prepared, qa = self._split_qa(transcript)
        text_lower = transcript.lower()

        words_all = re.findall(r"[a-z]{3,}", text_lower)
        n_words = len(words_all) or 1

        pos_words = [w for w in words_all if w in _LM_POSITIVE]
        neg_words = [w for w in words_all if w in _LM_NEGATIVE]
        unc_words = [w for w in words_all if w in _UNCERTAINTY]

        tone_score = (len(pos_words) - len(neg_words)) / n_words
        uncertainty_score = len(unc_words) / n_words

        # Guidance polarity via regex
        guidance_pos = bool(
            re.search(
                r"(?:raise[sd]?|increas(?:e|ed|ing)|above(?:\s+prior)?)\s+(?:guidance|outlook|forecast|expectations?)",
                text_lower,
            )
        )
        guidance_neg = bool(
            re.search(
                r"(?:lower[sd]?|decreas(?:e|ed|ing)|below(?:\s+prior)?|reduc(?:e|ed|ing))\s+(?:guidance|outlook|forecast|expectations?)",
                text_lower,
            )
        )
        guidance_polarity = (
            1 if guidance_pos and not guidance_neg else (-1 if guidance_neg else 0)
        )

        # Hedging: sentences containing hedge words
        sentences = re.split(r"[.!?]+", transcript)
        n_sentences = max(len(sentences), 1)
        hedge_pattern = r"\b(?:may|might|could|possibly|uncertain|approximately|expect|anticipate)\b"
        hedged_count = sum(1 for s in sentences if re.search(hedge_pattern, s.lower()))
        hedging_ratio = hedged_count / n_sentences

        # Q&A tone delta
        qa_tone_delta = 0.0
        if qa:
            qa_words = re.findall(r"[a-z]{3,}", qa.lower())
            n_qa = len(qa_words) or 1
            qa_pos = sum(1 for w in qa_words if w in _LM_POSITIVE)
            qa_neg = sum(1 for w in qa_words if w in _LM_NEGATIVE)
            qa_tone = (qa_pos - qa_neg) / n_qa

            prep_words = re.findall(r"[a-z]{3,}", prepared.lower())
            n_prep = len(prep_words) or 1
            prep_pos = sum(1 for w in prep_words if w in _LM_POSITIVE)
            prep_neg = sum(1 for w in prep_words if w in _LM_NEGATIVE)
            prep_tone = (prep_pos - prep_neg) / n_prep

            qa_tone_delta = qa_tone - prep_tone

        return {
            "tone_score": round(tone_score, 6),
            "uncertainty_score": round(uncertainty_score, 6),
            "guidance_polarity": guidance_polarity,
            "hedging_ratio": round(hedging_ratio, 6),
            "qa_tone_delta": round(qa_tone_delta, 6),
            "word_count": n_words,
            "positive_count": len(pos_words),
            "negative_count": len(neg_words),
            "has_qa_section": int(bool(qa)),
        }

    def batch_analyze(self, transcripts: list[dict[str, Any]]) -> pd.DataFrame:
        """
        Analyse a list of transcript records and return a DataFrame.

        Parameters
        ----------
        transcripts : list of dicts, each with keys:
            text   – raw transcript string
            date   – earnings date (str or datetime)
            ticker – symbol (optional, for index labeling)

        Returns
        -------
        pd.DataFrame indexed by date, with all signal columns.
        """
        records = []
        for t in transcripts:
            signals = self.analyze(t["text"])
            signals["ticker"] = t.get("ticker", "")
            signals["date"] = pd.to_datetime(t.get("date", pd.NaT))
            records.append(signals)
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records).set_index("date").sort_index()
        return df
