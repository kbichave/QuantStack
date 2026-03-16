"""
Options-specific DuckDB storage for IV history, earnings, and fundamentals.

Separate database (data/options.duckdb) for options-specific data.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import duckdb
import pandas as pd
import numpy as np
from loguru import logger


class OptionsDataStore:
    """
    DuckDB-based storage for options-specific data.
    
    Tables:
    - iv_history: Historical ATM IV and realized volatility
    - earnings_calendar: Earnings announcement dates and surprises
    - company_fundamentals: Company overview data (P/E, beta, etc.)
    """
    
    DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "options.duckdb"
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the options data store.
        
        Args:
            db_path: Path to database file (default: data/options.duckdb)
        """
        self.db_path = db_path or str(self.DEFAULT_DB_PATH)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._init_schema()
    
    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path)
        return self._conn
    
    def _init_schema(self) -> None:
        """Initialize database schema."""
        # IV History table - stores both historical ATM IV and realized vol
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS iv_history (
                symbol VARCHAR NOT NULL,
                date DATE NOT NULL,
                atm_iv DOUBLE,
                realized_vol_20d DOUBLE,
                realized_vol_60d DOUBLE,
                source VARCHAR NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date)
            )
        """)
        
        # Earnings calendar with historical data
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS earnings_calendar (
                symbol VARCHAR NOT NULL,
                report_date DATE NOT NULL,
                fiscal_date_ending DATE,
                estimate DOUBLE,
                reported_eps DOUBLE,
                surprise DOUBLE,
                surprise_pct DOUBLE,
                is_historical BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, report_date)
            )
        """)
        
        # Company fundamentals cache
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS company_fundamentals (
                symbol VARCHAR NOT NULL PRIMARY KEY,
                name VARCHAR,
                sector VARCHAR,
                industry VARCHAR,
                market_cap DOUBLE,
                pe_ratio DOUBLE,
                forward_pe DOUBLE,
                peg_ratio DOUBLE,
                price_to_book DOUBLE,
                dividend_yield DOUBLE,
                ex_dividend_date DATE,
                beta DOUBLE,
                fifty_two_week_high DOUBLE,
                fifty_two_week_low DOUBLE,
                eps DOUBLE,
                revenue_ttm DOUBLE,
                profit_margin DOUBLE,
                shares_outstanding DOUBLE,
                analyst_target_price DOUBLE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # IV statistics cache (for quick IV rank lookup)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS iv_statistics (
                symbol VARCHAR NOT NULL PRIMARY KEY,
                lookback_days INTEGER NOT NULL,
                iv_min DOUBLE,
                iv_max DOUBLE,
                iv_mean DOUBLE,
                iv_median DOUBLE,
                iv_std DOUBLE,
                iv_percentile_25 DOUBLE,
                iv_percentile_75 DOUBLE,
                last_iv DOUBLE,
                last_iv_rank DOUBLE,
                last_iv_percentile DOUBLE,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_iv_history_symbol_date 
            ON iv_history (symbol, date DESC)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_earnings_date 
            ON earnings_calendar (report_date)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_earnings_symbol 
            ON earnings_calendar (symbol, report_date DESC)
        """)
        
        logger.debug("Options database schema initialized")
    
    # ========================================
    # IV History Methods
    # ========================================
    
    def save_iv_history(
        self,
        symbol: str,
        date: date,
        atm_iv: Optional[float] = None,
        realized_vol_20d: Optional[float] = None,
        realized_vol_60d: Optional[float] = None,
        source: str = "options_chain",
    ) -> None:
        """
        Save IV history for a symbol/date.
        
        Args:
            symbol: Stock symbol
            date: Date of the IV reading
            atm_iv: ATM implied volatility (from options chain)
            realized_vol_20d: 20-day realized volatility
            realized_vol_60d: 60-day realized volatility
            source: Data source ('options_chain' or 'realized_vol')
        """
        self.conn.execute("""
            INSERT OR REPLACE INTO iv_history 
            (symbol, date, atm_iv, realized_vol_20d, realized_vol_60d, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [symbol, date, atm_iv, realized_vol_20d, realized_vol_60d, source])
    
    def save_iv_history_bulk(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> int:
        """
        Bulk save IV history from DataFrame.
        
        Args:
            df: DataFrame with columns: date, atm_iv, realized_vol_20d, realized_vol_60d, source
            symbol: Stock symbol
            
        Returns:
            Number of rows saved
        """
        if df.empty:
            return 0
        
        data = df.copy()
        data["symbol"] = symbol
        
        # Ensure date column is date type
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"]).dt.date
        
        # Select valid columns
        valid_cols = ["symbol", "date", "atm_iv", "realized_vol_20d", "realized_vol_60d", "source"]
        available_cols = [c for c in valid_cols if c in data.columns]
        
        insert_data = data[available_cols]
        
        self.conn.execute(f"""
            INSERT OR REPLACE INTO iv_history 
            ({', '.join(available_cols)})
            SELECT {', '.join(available_cols)}
            FROM insert_data
        """)
        
        logger.info(f"Saved {len(data)} IV history records for {symbol}")
        return len(data)
    
    def load_iv_history(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        """
        Load IV history for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start of date range
            end_date: End of date range
            lookback_days: If no dates provided, load last N days
            
        Returns:
            DataFrame with IV history
        """
        query = """
            SELECT date, atm_iv, realized_vol_20d, realized_vol_60d, source
            FROM iv_history
            WHERE symbol = ?
        """
        params = [symbol]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        elif not end_date:
            # Use lookback from today
            query += " AND date >= CURRENT_DATE - INTERVAL ? DAY"
            params.append(lookback_days)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date"
        
        df = self.conn.execute(query, params).fetchdf()
        
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        
        return df
    
    def get_iv_for_rank(
        self,
        symbol: str,
        lookback_days: int = 252,
    ) -> pd.Series:
        """
        Get IV series for rank calculation (prefers ATM IV, falls back to realized vol).
        
        Args:
            symbol: Stock symbol
            lookback_days: Lookback period
            
        Returns:
            Series of IV values
        """
        df = self.load_iv_history(symbol, lookback_days=lookback_days)
        
        if df.empty:
            return pd.Series(dtype=float)
        
        # Use ATM IV if available, otherwise realized vol
        if "atm_iv" in df.columns and df["atm_iv"].notna().sum() > 0:
            iv = df["atm_iv"].dropna()
        elif "realized_vol_20d" in df.columns:
            iv = df["realized_vol_20d"].dropna()
        else:
            return pd.Series(dtype=float)
        
        return iv
    
    # ========================================
    # IV Statistics Methods
    # ========================================
    
    def update_iv_statistics(
        self,
        symbol: str,
        lookback_days: int = 252,
    ) -> Dict[str, float]:
        """
        Calculate and cache IV statistics for a symbol.
        
        Args:
            symbol: Stock symbol
            lookback_days: Lookback period for statistics
            
        Returns:
            Dictionary of IV statistics
        """
        iv_series = self.get_iv_for_rank(symbol, lookback_days)
        
        if iv_series.empty or len(iv_series) < 20:
            logger.warning(f"Insufficient IV data for {symbol}")
            return {}
        
        # Calculate statistics
        iv_min = float(iv_series.min())
        iv_max = float(iv_series.max())
        iv_mean = float(iv_series.mean())
        iv_median = float(iv_series.median())
        iv_std = float(iv_series.std())
        iv_p25 = float(iv_series.quantile(0.25))
        iv_p75 = float(iv_series.quantile(0.75))
        last_iv = float(iv_series.iloc[-1])
        
        # Calculate IV rank and percentile
        if iv_max > iv_min:
            iv_rank = (last_iv - iv_min) / (iv_max - iv_min) * 100
        else:
            iv_rank = 50.0
        
        iv_percentile = (iv_series < last_iv).mean() * 100
        
        stats = {
            "iv_min": iv_min,
            "iv_max": iv_max,
            "iv_mean": iv_mean,
            "iv_median": iv_median,
            "iv_std": iv_std,
            "iv_percentile_25": iv_p25,
            "iv_percentile_75": iv_p75,
            "last_iv": last_iv,
            "last_iv_rank": iv_rank,
            "last_iv_percentile": iv_percentile,
        }
        
        # Save to cache
        self.conn.execute("""
            INSERT OR REPLACE INTO iv_statistics 
            (symbol, lookback_days, iv_min, iv_max, iv_mean, iv_median, iv_std,
             iv_percentile_25, iv_percentile_75, last_iv, last_iv_rank, last_iv_percentile,
             calculated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [symbol, lookback_days, iv_min, iv_max, iv_mean, iv_median, iv_std,
              iv_p25, iv_p75, last_iv, iv_rank, iv_percentile])
        
        logger.debug(f"Updated IV stats for {symbol}: rank={iv_rank:.1f}, percentile={iv_percentile:.1f}")
        return stats
    
    def get_iv_statistics(
        self,
        symbol: str,
    ) -> Optional[Dict[str, float]]:
        """
        Get cached IV statistics for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary of IV statistics or None
        """
        result = self.conn.execute("""
            SELECT * FROM iv_statistics WHERE symbol = ?
        """, [symbol]).fetchdf()
        
        if result.empty:
            return None
        
        return result.iloc[0].to_dict()
    
    # ========================================
    # Earnings Calendar Methods
    # ========================================
    
    def save_earnings(
        self,
        df: pd.DataFrame,
        replace: bool = False,
    ) -> int:
        """
        Save earnings calendar data.
        
        Args:
            df: DataFrame with earnings data
            replace: If True, replace all existing data
            
        Returns:
            Number of rows saved
        """
        if df.empty:
            return 0
        
        data = df.copy()
        
        # Standardize column names
        column_mapping = {
            "reportdate": "report_date",
            "reportDate": "report_date",
            "fiscaldateending": "fiscal_date_ending",
            "fiscalDateEnding": "fiscal_date_ending",
            "reported_eps": "reported_eps",
            "reportedEPS": "reported_eps",
            "surprise_pct": "surprise_pct",
            "surprisePercentage": "surprise_pct",
        }
        data = data.rename(columns={k: v for k, v in column_mapping.items() if k in data.columns})
        
        if replace:
            self.conn.execute("DELETE FROM earnings_calendar")
        
        # Select valid columns
        valid_cols = [
            "symbol", "report_date", "fiscal_date_ending", "estimate",
            "reported_eps", "surprise", "surprise_pct", "is_historical"
        ]
        available_cols = [c for c in valid_cols if c in data.columns]
        
        if "symbol" not in available_cols:
            logger.warning("No symbol column in earnings data")
            return 0
        
        insert_data = data[available_cols]
        
        self.conn.execute(f"""
            INSERT OR REPLACE INTO earnings_calendar 
            ({', '.join(available_cols)})
            SELECT {', '.join(available_cols)}
            FROM insert_data
        """)
        
        logger.info(f"Saved {len(data)} earnings records")
        return len(data)
    
    def load_earnings(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        upcoming_only: bool = False,
    ) -> pd.DataFrame:
        """
        Load earnings calendar data.
        
        Args:
            symbol: Filter by symbol
            start_date: Start of date range
            end_date: End of date range
            upcoming_only: If True, only return future earnings
            
        Returns:
            DataFrame with earnings data
        """
        query = "SELECT * FROM earnings_calendar WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if upcoming_only:
            query += " AND report_date >= CURRENT_DATE"
        
        if start_date:
            query += " AND report_date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND report_date <= ?"
            params.append(end_date)
        
        query += " ORDER BY report_date"
        
        df = self.conn.execute(query, params).fetchdf()
        
        if not df.empty:
            if "report_date" in df.columns:
                df["report_date"] = pd.to_datetime(df["report_date"])
            if "fiscal_date_ending" in df.columns:
                df["fiscal_date_ending"] = pd.to_datetime(df["fiscal_date_ending"])
        
        return df
    
    def get_days_to_earnings(
        self,
        symbol: str,
        as_of_date: Optional[date] = None,
    ) -> Optional[int]:
        """
        Get days until next earnings for a symbol.
        
        Args:
            symbol: Stock symbol
            as_of_date: Reference date (default: today)
            
        Returns:
            Days to next earnings, or None if not found
        """
        if as_of_date is None:
            as_of_date = datetime.now().date()
        
        result = self.conn.execute("""
            SELECT MIN(report_date) as next_earnings
            FROM earnings_calendar
            WHERE symbol = ? AND report_date >= ?
        """, [symbol, as_of_date]).fetchone()
        
        if result and result[0]:
            next_date = pd.to_datetime(result[0]).date()
            return (next_date - as_of_date).days
        
        return None
    
    def get_days_since_earnings(
        self,
        symbol: str,
        as_of_date: Optional[date] = None,
    ) -> Optional[int]:
        """
        Get days since last earnings for a symbol.
        
        Args:
            symbol: Stock symbol
            as_of_date: Reference date (default: today)
            
        Returns:
            Days since last earnings, or None if not found
        """
        if as_of_date is None:
            as_of_date = datetime.now().date()
        
        result = self.conn.execute("""
            SELECT MAX(report_date) as last_earnings
            FROM earnings_calendar
            WHERE symbol = ? AND report_date < ?
        """, [symbol, as_of_date]).fetchone()
        
        if result and result[0]:
            last_date = pd.to_datetime(result[0]).date()
            return (as_of_date - last_date).days
        
        return None
    
    def get_last_earnings_surprise(
        self,
        symbol: str,
        as_of_date: Optional[date] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get last earnings surprise data for a symbol.
        
        Args:
            symbol: Stock symbol
            as_of_date: Reference date (default: today)
            
        Returns:
            Dict with surprise, surprise_pct, or None
        """
        if as_of_date is None:
            as_of_date = datetime.now().date()
        
        result = self.conn.execute("""
            SELECT report_date, estimate, reported_eps, surprise, surprise_pct
            FROM earnings_calendar
            WHERE symbol = ? AND report_date < ? AND reported_eps IS NOT NULL
            ORDER BY report_date DESC
            LIMIT 1
        """, [symbol, as_of_date]).fetchdf()
        
        if result.empty:
            return None
        
        return result.iloc[0].to_dict()
    
    # ========================================
    # Company Fundamentals Methods
    # ========================================
    
    def save_fundamentals(
        self,
        data: Dict[str, Any],
    ) -> None:
        """
        Save company fundamentals from overview API.
        
        Args:
            data: Dictionary from fetch_company_overview()
        """
        if not data or "Symbol" not in data:
            return
        
        self.conn.execute("""
            INSERT OR REPLACE INTO company_fundamentals
            (symbol, name, sector, industry, market_cap, pe_ratio, forward_pe,
             peg_ratio, price_to_book, dividend_yield, ex_dividend_date, beta,
             fifty_two_week_high, fifty_two_week_low, eps, revenue_ttm,
             profit_margin, shares_outstanding, analyst_target_price, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [
            data.get("Symbol"),
            data.get("Name"),
            data.get("Sector"),
            data.get("Industry"),
            data.get("MarketCapitalization"),
            data.get("PERatio"),
            data.get("ForwardPE"),
            data.get("PEGRatio"),
            data.get("PriceToBookRatio"),
            data.get("DividendYield"),
            data.get("ExDividendDate"),
            data.get("Beta"),
            data.get("52WeekHigh"),
            data.get("52WeekLow"),
            data.get("EPS"),
            data.get("RevenueTTM"),
            data.get("ProfitMargin"),
            data.get("SharesOutstanding"),
            data.get("AnalystTargetPrice"),
        ])
        
        logger.debug(f"Saved fundamentals for {data.get('Symbol')}")
    
    def load_fundamentals(
        self,
        symbol: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Load company fundamentals for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with fundamentals or None
        """
        result = self.conn.execute("""
            SELECT * FROM company_fundamentals WHERE symbol = ?
        """, [symbol]).fetchdf()
        
        if result.empty:
            return None
        
        return result.iloc[0].to_dict()
    
    def load_fundamentals_batch(
        self,
        symbols: List[str],
    ) -> pd.DataFrame:
        """
        Load fundamentals for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            DataFrame with fundamentals
        """
        if not symbols:
            return pd.DataFrame()
        
        placeholders = ",".join(["?" for _ in symbols])
        query = f"SELECT * FROM company_fundamentals WHERE symbol IN ({placeholders})"
        
        return self.conn.execute(query, symbols).fetchdf()
    
    # ========================================
    # Utility Methods
    # ========================================
    
    def has_iv_data(
        self,
        symbol: str,
        min_records: int = 100,
    ) -> bool:
        """Check if we have sufficient IV history for a symbol."""
        result = self.conn.execute("""
            SELECT COUNT(*) as cnt FROM iv_history WHERE symbol = ?
        """, [symbol]).fetchone()
        
        return result[0] >= min_records if result else False
    
    def get_symbols_with_iv_data(self) -> List[str]:
        """Get list of symbols with IV history."""
        result = self.conn.execute("""
            SELECT DISTINCT symbol FROM iv_history ORDER BY symbol
        """).fetchall()
        return [row[0] for row in result]
    
    def vacuum(self) -> None:
        """Optimize database storage."""
        self.conn.execute("VACUUM")
        logger.info("Options database vacuumed")
    
    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("Options database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Singleton instance
_options_store: Optional[OptionsDataStore] = None


def get_options_store() -> OptionsDataStore:
    """Get options data store singleton."""
    global _options_store
    if _options_store is None:
        _options_store = OptionsDataStore()
    return _options_store

