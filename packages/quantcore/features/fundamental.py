"""
Accounting-based alpha factors (quantamental signals).

All factors operate on balance sheet / income statement / cash flow data
as returned by FD.ai's financial statement endpoints. No network calls here.

Factors
-------
SloanAccruals       – earnings quality; high accruals = mean reversion candidate
NovyMarxGP          – gross profitability factor (Novy-Marx 2013)
AssetGrowthAnomaly  – high asset growth → future underperformance
FCFYield            – free cash flow yield (harder to manipulate than P/E)
RevenueAcceleration – QoQ revenue acceleration = bullish EPS momentum signal
OperatingLeverage   – DOL = %Δ EBITDA / %Δ Revenue; earnings convexity proxy
PiotroskiFScore     – 9-point financial strength score (Piotroski 2000)
BeneishMScore       – 8-variable accounting manipulation detection (Beneish 1999)
"""

import numpy as np
import pandas as pd


class SloanAccruals:
    """
    Sloan (1996) accruals — earnings quality factor.

    Accruals = (Net_Income - Operating_CFO) / avg(Total_Assets)

    High accruals indicate that earnings are cash-poor and driven by
    accounting adjustments rather than real cash generation. High-accrual
    firms systematically underperform (mean reversion in earnings quality).
    """

    def compute(self, financials_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        financials_df : pd.DataFrame
            Must contain columns:
            - period_end (datetime)     — fiscal period end date
            - net_income (float)        — net income
            - operating_cash_flow (float) — CFO from cash flow statement
            - total_assets (float)      — total assets (balance sheet)

        Returns
        -------
        pd.DataFrame with columns:
            accruals          – Sloan accruals ratio (signed)
            accruals_high     – 1 if accruals > 0.1 (earnings quality concern)
            accruals_low      – 1 if accruals < -0.1 (strong cash earnings)
        """
        df = financials_df.copy()
        df["period_end"] = pd.to_datetime(df["period_end"])
        df = df.sort_values("period_end").reset_index(drop=True)

        avg_assets = (df["total_assets"] + df["total_assets"].shift(1)) / 2
        avg_assets_safe = avg_assets.replace(0, np.nan)

        df["accruals"] = (df["net_income"] - df["operating_cash_flow"]) / avg_assets_safe

        df["accruals_high"] = (df["accruals"] > 0.1).astype(int)
        df["accruals_low"] = (df["accruals"] < -0.1).astype(int)

        return df


class NovyMarxGP:
    """
    Novy-Marx (2013) gross profitability factor.

    GP = (Revenue - COGS) / Total_Assets

    One of the most robust quality factors: high gross profitability firms
    persistently outperform despite being expensive on traditional value metrics.
    Often used to augment value strategies (value × quality composite).
    """

    def compute(self, financials_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        financials_df : pd.DataFrame
            Must contain columns:
            - period_end (datetime)
            - revenue (float)
            - cost_of_revenue (float)   — COGS
            - total_assets (float)

        Returns
        -------
        pd.DataFrame with columns:
            gross_profit      – revenue - cost_of_revenue
            gp_ratio          – gross profit / total_assets (Novy-Marx factor)
            gp_high           – 1 if gp_ratio > top 30th percentile of history
            gp_zscore         – z-score vs rolling 8-quarter history
        """
        df = financials_df.copy()
        df["period_end"] = pd.to_datetime(df["period_end"])
        df = df.sort_values("period_end").reset_index(drop=True)

        df["gross_profit"] = df["revenue"] - df["cost_of_revenue"]
        assets_safe = df["total_assets"].replace(0, np.nan)
        df["gp_ratio"] = df["gross_profit"] / assets_safe

        rolling_mean = df["gp_ratio"].rolling(8, min_periods=2).mean()
        rolling_std = df["gp_ratio"].rolling(8, min_periods=2).std().replace(0, np.nan)
        df["gp_zscore"] = (df["gp_ratio"] - rolling_mean) / rolling_std

        threshold = df["gp_ratio"].expanding(min_periods=4).quantile(0.7)
        df["gp_high"] = (df["gp_ratio"] >= threshold).astype(int)

        return df


class AssetGrowthAnomaly:
    """
    Cooper, Gulen & Schill (2008) asset growth anomaly.

    Asset_Growth = (Total_Assets_t - Total_Assets_{t-1}) / Total_Assets_{t-1}

    High asset growth firms underperform over the next 3-5 years, likely
    because aggressive expansion is associated with overinvestment and
    poor capital allocation discipline.
    """

    def compute(self, financials_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        financials_df : pd.DataFrame
            Must contain columns:
            - period_end (datetime)
            - total_assets (float)

        Returns
        -------
        pd.DataFrame with columns:
            asset_growth         – YoY asset growth rate (as decimal)
            asset_growth_high    – 1 if asset_growth > 20% (aggressive expansion)
            asset_growth_negative – 1 if asset_growth < 0 (shrinkage)
        """
        df = financials_df.copy()
        df["period_end"] = pd.to_datetime(df["period_end"])
        df = df.sort_values("period_end").reset_index(drop=True)

        prior_assets = df["total_assets"].shift(4)  # 4 quarters = 1 year
        prior_safe = prior_assets.replace(0, np.nan)

        df["asset_growth"] = (df["total_assets"] - prior_assets) / prior_safe

        df["asset_growth_high"] = (df["asset_growth"] > 0.20).astype(int)
        df["asset_growth_negative"] = (df["asset_growth"] < 0).astype(int)

        return df


class FCFYield:
    """
    Free Cash Flow Yield — a more manipulation-resistant valuation metric.

    FCF = Operating_CFO - CapEx
    FCF_Yield = FCF / Market_Cap

    Higher FCF yield → cheaper on a cash-generation basis. More robust
    than earnings yield because FCF is harder to inflate via accruals.
    """

    def compute(self, financials_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        financials_df : pd.DataFrame
            Must contain columns:
            - period_end (datetime)
            - operating_cash_flow (float)
            - capital_expenditures (float)    — CapEx (positive value)
            - market_cap (float)              — market capitalization

        Returns
        -------
        pd.DataFrame with columns:
            fcf              – free cash flow (absolute)
            fcf_yield        – fcf / market_cap (as decimal)
            fcf_yield_pct    – fcf_yield * 100 (percentage)
            fcf_positive     – 1 if fcf > 0
            fcf_yield_high   – 1 if fcf_yield_pct > 5% (attractive valuation)
        """
        df = financials_df.copy()
        df["period_end"] = pd.to_datetime(df["period_end"])
        df = df.sort_values("period_end").reset_index(drop=True)

        df["fcf"] = df["operating_cash_flow"] - df["capital_expenditures"].abs()

        mktcap_safe = df["market_cap"].replace(0, np.nan)
        df["fcf_yield"] = df["fcf"] / mktcap_safe
        df["fcf_yield_pct"] = df["fcf_yield"] * 100

        df["fcf_positive"] = (df["fcf"] > 0).astype(int)
        df["fcf_yield_high"] = (df["fcf_yield_pct"] > 5).astype(int)

        return df


class RevenueAcceleration:
    """
    Revenue QoQ acceleration — leading earnings momentum signal.

    Acceleration = current_QoQ_growth - prior_QoQ_growth

    Even companies growing slowly can surprise if their growth rate is
    accelerating. Positive acceleration + positive growth = high conviction
    momentum setup.
    """

    def compute(self, financials_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        financials_df : pd.DataFrame
            Must contain columns:
            - period_end (datetime)
            - revenue (float)

        Returns
        -------
        pd.DataFrame with columns:
            revenue_qoq_growth    – quarter-over-quarter revenue growth (%)
            revenue_yoy_growth    – year-over-year revenue growth (%)
            revenue_acceleration  – delta of QoQ growth (positive = accelerating)
            accelerating          – 1 if acceleration > 0 and QoQ growth > 0
            decelerating          – 1 if acceleration < 0 and QoQ growth > 0
        """
        df = financials_df.copy()
        df["period_end"] = pd.to_datetime(df["period_end"])
        df = df.sort_values("period_end").reset_index(drop=True)

        prior_q = df["revenue"].shift(1)
        prior_y = df["revenue"].shift(4)

        prior_q_safe = prior_q.replace(0, np.nan)
        prior_y_safe = prior_y.replace(0, np.nan)

        df["revenue_qoq_growth"] = (df["revenue"] - prior_q) / prior_q_safe * 100
        df["revenue_yoy_growth"] = (df["revenue"] - prior_y) / prior_y_safe * 100

        df["revenue_acceleration"] = df["revenue_qoq_growth"].diff(1)

        df["accelerating"] = (
            (df["revenue_acceleration"] > 0) & (df["revenue_qoq_growth"] > 0)
        ).astype(int)
        df["decelerating"] = (
            (df["revenue_acceleration"] < 0) & (df["revenue_qoq_growth"] > 0)
        ).astype(int)

        return df


class OperatingLeverage:
    """
    Degree of Operating Leverage (DOL) — earnings convexity proxy.

    DOL = % Δ EBITDA / % Δ Revenue

    High DOL firms have large fixed costs: when revenue grows, EBITDA grows
    faster (convex upside). When revenue falls, EBITDA falls faster (convex
    downside). High DOL + revenue acceleration = earnings beat setup.
    """

    def compute(self, financials_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        financials_df : pd.DataFrame
            Must contain columns:
            - period_end (datetime)
            - revenue (float)
            - ebitda (float)

        Returns
        -------
        pd.DataFrame with columns:
            revenue_change_pct – QoQ revenue % change
            ebitda_change_pct  – QoQ EBITDA % change
            dol                – degree of operating leverage
            high_leverage      – 1 if dol > 3 (high earnings sensitivity)
        """
        df = financials_df.copy()
        df["period_end"] = pd.to_datetime(df["period_end"])
        df = df.sort_values("period_end").reset_index(drop=True)

        prior_rev = df["revenue"].shift(1).replace(0, np.nan)
        prior_ebitda = df["ebitda"].shift(1).replace(0, np.nan)

        df["revenue_change_pct"] = (df["revenue"] - df["revenue"].shift(1)) / prior_rev * 100
        df["ebitda_change_pct"] = (df["ebitda"] - df["ebitda"].shift(1)) / prior_ebitda.abs() * 100

        rev_safe = df["revenue_change_pct"].replace(0, np.nan)
        df["dol"] = df["ebitda_change_pct"] / rev_safe

        # Cap extreme values (near-zero revenue change creates huge DOL artifacts)
        df["dol"] = df["dol"].clip(-20, 20)
        df["high_leverage"] = (df["dol"].abs() > 3).astype(int)

        return df


class PiotroskiFScore:
    """
    Piotroski (2000) F-Score — 9-point financial strength composite.

    Combines 9 binary signals across three dimensions:
      Profitability (4): ROA, CFO, ΔROA, accruals
      Leverage/Liquidity (3): ΔLeverage, ΔLiquidity, share issuance
      Operating Efficiency (2): ΔGross Margin, ΔAsset Turnover

    Score 8-9 = strong long candidate; 0-2 = short candidate.
    Particularly effective for value screens (high B/M universe).
    """

    def compute(self, financials_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        financials_df : pd.DataFrame
            Must contain columns:
            - period_end (datetime)
            - total_assets (float)
            - net_income (float)
            - operating_cash_flow (float)
            - long_term_debt (float)
            - current_assets (float)
            - current_liabilities (float)
            - shares_outstanding (float)
            - revenue (float)
            - cost_of_revenue (float)

        Returns
        -------
        pd.DataFrame with columns:
            f_roa, f_cfo, f_delta_roa, f_accrual,         (profitability)
            f_delta_leverage, f_delta_liquidity, f_no_dilution, (leverage)
            f_delta_gross_margin, f_delta_asset_turnover,  (efficiency)
            f_score                                         (0-9 composite)
        """
        df = financials_df.copy()
        df["period_end"] = pd.to_datetime(df["period_end"])
        df = df.sort_values("period_end").reset_index(drop=True)

        assets_avg = (df["total_assets"] + df["total_assets"].shift(1)) / 2
        assets_avg_safe = assets_avg.replace(0, np.nan)

        roa = df["net_income"] / assets_avg_safe
        cfo_over_assets = df["operating_cash_flow"] / assets_avg_safe

        leverage = df["long_term_debt"] / assets_avg_safe
        current_ratio = df["current_assets"] / df["current_liabilities"].replace(0, np.nan)
        gross_margin = (df["revenue"] - df["cost_of_revenue"]) / df["revenue"].replace(0, np.nan)
        asset_turnover = df["revenue"] / assets_avg_safe

        # Profitability signals
        f_roa = (roa > 0).astype(int)
        f_cfo = (cfo_over_assets > 0).astype(int)
        f_delta_roa = (roa > roa.shift(4)).astype(int)      # vs same quarter prior year
        f_accrual = (cfo_over_assets > roa).astype(int)      # cash earnings > accrual earnings

        # Leverage/Liquidity signals
        f_delta_leverage = (leverage < leverage.shift(4)).astype(int)   # debt decreased
        f_delta_liquidity = (current_ratio > current_ratio.shift(4)).astype(int)
        f_no_dilution = (df["shares_outstanding"] <= df["shares_outstanding"].shift(4)).astype(int)

        # Efficiency signals
        f_delta_gm = (gross_margin > gross_margin.shift(4)).astype(int)
        f_delta_at = (asset_turnover > asset_turnover.shift(4)).astype(int)

        f_score = (
            f_roa + f_cfo + f_delta_roa + f_accrual
            + f_delta_leverage + f_delta_liquidity + f_no_dilution
            + f_delta_gm + f_delta_at
        )

        return df.assign(
            f_roa=f_roa,
            f_cfo=f_cfo,
            f_delta_roa=f_delta_roa,
            f_accrual=f_accrual,
            f_delta_leverage=f_delta_leverage,
            f_delta_liquidity=f_delta_liquidity,
            f_no_dilution=f_no_dilution,
            f_delta_gross_margin=f_delta_gm,
            f_delta_asset_turnover=f_delta_at,
            f_score=f_score,
        )


class BeneishMScore:
    """
    Beneish (1999) M-Score — earnings manipulation detection.

    8-variable model trained to detect earnings manipulation:
    M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
         + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

    M > -1.78: high manipulation probability (short signal)
    M < -2.22: low manipulation probability (clean)

    DSRI = Days Sales in Receivables Index
    GMI  = Gross Margin Index
    AQI  = Asset Quality Index
    SGI  = Sales Growth Index
    DEPI = Depreciation Index
    SGAI = Sales, General & Admin expenses Index
    TATA = Total Accruals to Total Assets
    LVGI = Leverage Index
    """

    def compute(self, financials_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        financials_df : pd.DataFrame
            Must contain columns:
            - period_end (datetime)
            - revenue (float)
            - accounts_receivable (float)
            - cost_of_revenue (float)
            - total_assets (float)
            - current_assets (float)
            - property_plant_equipment (float)   — net PP&E
            - depreciation (float)
            - sga_expenses (float)               — SG&A
            - net_income (float)
            - operating_cash_flow (float)
            - long_term_debt (float)
            - current_liabilities (float)

        Returns
        -------
        pd.DataFrame with columns:
            DSRI, GMI, AQI, SGI, DEPI, SGAI, TATA, LVGI,
            m_score          – Beneish M-Score
            manipulation_risk – 1 if m_score > -1.78
        """
        df = financials_df.copy()
        df["period_end"] = pd.to_datetime(df["period_end"])
        df = df.sort_values("period_end").reset_index(drop=True)

        rev = df["revenue"]
        rev_prior = rev.shift(4)
        rev_safe = rev.replace(0, np.nan)
        rev_prior_safe = rev_prior.replace(0, np.nan)

        ar = df["accounts_receivable"]
        ar_prior = ar.shift(4)

        cogs = df["cost_of_revenue"]
        cogs_prior = cogs.shift(4)

        assets = df["total_assets"]
        assets_prior = assets.shift(4)
        assets_safe = assets.replace(0, np.nan)
        assets_prior_safe = assets_prior.replace(0, np.nan)

        # DSRI: (AR_t / Sales_t) / (AR_{t-1} / Sales_{t-1})
        dsri = (ar / rev_safe) / ((ar_prior / rev_prior_safe).replace(0, np.nan))

        # GMI: ((Sales_{t-1} - COGS_{t-1}) / Sales_{t-1}) / ((Sales_t - COGS_t) / Sales_t)
        gm_t = (rev - cogs) / rev_safe
        gm_prior = (rev_prior - cogs_prior) / rev_prior_safe
        gmi = gm_prior / gm_t.replace(0, np.nan)

        # AQI: (1 - (CA_t + PP&E_t) / TA_t) / (1 - (CA_{t-1} + PP&E_{t-1}) / TA_{t-1})
        ca = df["current_assets"]
        ppe = df["property_plant_equipment"]
        aqi_t = 1 - (ca + ppe) / assets_safe
        aqi_prior = 1 - (ca.shift(4) + ppe.shift(4)) / assets_prior_safe
        aqi = aqi_t / aqi_prior.replace(0, np.nan)

        # SGI: Sales_t / Sales_{t-1}
        sgi = rev / rev_prior_safe

        # DEPI: (Depr_{t-1} / (PP&E_{t-1} + Depr_{t-1})) / (Depr_t / (PP&E_t + Depr_t))
        dep = df["depreciation"]
        dep_prior = dep.shift(4)
        depi_t = dep / (ppe + dep).replace(0, np.nan)
        depi_prior = dep_prior / (ppe.shift(4) + dep_prior).replace(0, np.nan)
        depi = depi_prior / depi_t.replace(0, np.nan)

        # SGAI: (SGA_t / Sales_t) / (SGA_{t-1} / Sales_{t-1})
        sga = df["sga_expenses"]
        sgai = (sga / rev_safe) / ((sga.shift(4) / rev_prior_safe).replace(0, np.nan))

        # TATA: (NI - CFO) / TA
        tata = (df["net_income"] - df["operating_cash_flow"]) / assets_safe

        # LVGI: ((LTD_t + CL_t) / TA_t) / ((LTD_{t-1} + CL_{t-1}) / TA_{t-1})
        ltd = df["long_term_debt"]
        cl = df["current_liabilities"]
        lvgi_t = (ltd + cl) / assets_safe
        lvgi_prior = (ltd.shift(4) + cl.shift(4)) / assets_prior_safe
        lvgi = lvgi_t / lvgi_prior.replace(0, np.nan)

        # Beneish regression coefficients
        m_score = (
            -4.84
            + 0.920 * dsri
            + 0.528 * gmi
            + 0.404 * aqi
            + 0.892 * sgi
            + 0.115 * depi
            - 0.172 * sgai
            + 4.679 * tata
            - 0.327 * lvgi
        )

        return df.assign(
            DSRI=dsri,
            GMI=gmi,
            AQI=aqi,
            SGI=sgi,
            DEPI=depi,
            SGAI=sgai,
            TATA=tata,
            LVGI=lvgi,
            m_score=m_score,
            manipulation_risk=(m_score > -1.78).astype(int),
        )
