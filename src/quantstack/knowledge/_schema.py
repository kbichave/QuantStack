# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Schema mixin — DDL methods for KnowledgeStore."""

from quantstack.db import PgConnection


class SchemaMixin:
    """DDL methods: CREATE TABLE, CREATE INDEX, CREATE SEQUENCE."""

    conn: PgConnection

    def _init_schema(self) -> None:
        """Initialize database schema."""
        # Trade journal
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_trade_journal START 1")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_journal (
                id BIGINT PRIMARY KEY DEFAULT nextval('seq_trade_journal'),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR NOT NULL,
                direction VARCHAR NOT NULL,
                structure_type VARCHAR NOT NULL,
                status VARCHAR DEFAULT 'PENDING',
                legs JSON,
                entry_price DOUBLE,
                exit_price DOUBLE,
                quantity INTEGER DEFAULT 1,
                pnl DOUBLE,
                pnl_pct DOUBLE,
                max_profit_potential DOUBLE,
                max_loss_potential DOUBLE,
                wave_scenario_id VARCHAR,
                regime_at_entry VARCHAR,
                volatility_at_entry DOUBLE,
                confidence_score DOUBLE DEFAULT 0.5,
                agent_rationale TEXT,
                research_score DOUBLE,
                arena_rank INTEGER,
                outcome_correct BOOLEAN,
                lessons_learned TEXT,
                tags JSON,
                entry_order_id VARCHAR,
                exit_order_id VARCHAR
            )
        """
        )

        # Market observations
        self.conn.execute(
            "CREATE SEQUENCE IF NOT EXISTS seq_market_observations START 1"
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_observations (
                id BIGINT PRIMARY KEY DEFAULT nextval('seq_market_observations'),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR NOT NULL,
                observation_type VARCHAR NOT NULL,
                current_price DOUBLE,
                price_change_pct DOUBLE,
                volume BIGINT,
                volume_ratio DOUBLE,
                support_level DOUBLE,
                resistance_level DOUBLE,
                alert_message TEXT,
                severity VARCHAR DEFAULT 'INFO',
                source_agent VARCHAR,
                processed BOOLEAN DEFAULT FALSE
            )
        """
        )

        # Wave scenarios
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS wave_scenarios (
                id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                wave_position VARCHAR,
                wave_degree VARCHAR,
                confidence DOUBLE DEFAULT 0.5,
                primary_target DOUBLE,
                secondary_target DOUBLE,
                invalidation_level DOUBLE NOT NULL,
                scenario_type VARCHAR,
                description TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                invalidated_at TIMESTAMP,
                target_hit_at TIMESTAMP,
                source_agent VARCHAR
            )
        """
        )

        # Regime states
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_regime_states START 1")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS regime_states (
                id BIGINT PRIMARY KEY DEFAULT nextval('seq_regime_states'),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                trend_regime VARCHAR,
                volatility_regime VARCHAR,
                atr DOUBLE,
                atr_percentile DOUBLE,
                adx DOUBLE,
                correlation_to_spy DOUBLE,
                regime_changed BOOLEAN DEFAULT FALSE,
                previous_trend_regime VARCHAR,
                previous_volatility_regime VARCHAR,
                confidence DOUBLE DEFAULT 0.5,
                source_agent VARCHAR
            )
        """
        )

        # Agent messages
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_agent_messages START 1")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_messages (
                id BIGINT PRIMARY KEY DEFAULT nextval('seq_agent_messages'),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                from_agent VARCHAR NOT NULL,
                to_agent VARCHAR,
                message_type VARCHAR NOT NULL,
                subject VARCHAR,
                content TEXT,
                data JSON,
                priority INTEGER DEFAULT 5,
                requires_response BOOLEAN DEFAULT FALSE,
                response_deadline TIMESTAMP,
                acknowledged BOOLEAN DEFAULT FALSE,
                acknowledged_at TIMESTAMP
            )
        """
        )

        # Trading signals
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trading_signals (
                id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR NOT NULL,
                direction VARCHAR NOT NULL,
                signal_type VARCHAR NOT NULL,
                strength DOUBLE DEFAULT 0.5,
                confidence DOUBLE DEFAULT 0.5,
                entry_price DOUBLE,
                target_price DOUBLE,
                stop_loss DOUBLE,
                wave_scenario_id VARCHAR,
                regime_state_id INTEGER,
                observation_ids JSON,
                rationale TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                processed BOOLEAN DEFAULT FALSE,
                trade_id INTEGER,
                source_agent VARCHAR
            )
        """
        )

        # Performance metrics
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                entity_type VARCHAR NOT NULL,
                entity_name VARCHAR NOT NULL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                win_rate DOUBLE DEFAULT 0.0,
                total_pnl DOUBLE DEFAULT 0.0,
                avg_pnl_per_trade DOUBLE DEFAULT 0.0,
                largest_win DOUBLE DEFAULT 0.0,
                largest_loss DOUBLE DEFAULT 0.0,
                sharpe_ratio DOUBLE,
                sortino_ratio DOUBLE,
                max_drawdown DOUBLE,
                profit_factor DOUBLE,
                period_start TIMESTAMP,
                period_end TIMESTAMP
            )
        """
        )

        # Create indexes
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trade_journal(symbol)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_status ON trade_journal(status)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_obs_symbol ON market_observations(symbol)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_waves_symbol ON wave_scenarios(symbol)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_regime_symbol ON regime_states(symbol)"
        )

        # =====================================================================
        # HISTORICAL ALPHA ARENA TABLES
        # =====================================================================

        # Daily portfolio state (for equity curve)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_state (
                date DATE PRIMARY KEY,
                equity DOUBLE NOT NULL,
                cash DOUBLE NOT NULL,
                max_drawdown DOUBLE DEFAULT 0,
                exposures JSON,
                regime_summary VARCHAR
            )
        """
        )

        # Historical signals from strategy pods
        # Use SEQUENCE for auto-increment in DuckDB
        self.conn.execute(
            "CREATE SEQUENCE IF NOT EXISTS seq_historical_signals START 1"
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_signals (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_historical_signals'),
                date DATE NOT NULL,
                symbol VARCHAR NOT NULL,
                agent VARCHAR NOT NULL,
                signal_type VARCHAR NOT NULL,
                confidence DOUBLE DEFAULT 0.5,
                regime VARCHAR,
                structural_label VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Policy evolution snapshots
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS policy_snapshots (
                effective_date DATE PRIMARY KEY,
                pod_weights JSON NOT NULL,
                thresholds JSON,
                comment VARCHAR
            )
        """
        )

        # Agent logs for chat timeline (KEY TABLE)
        # Message field is TEXT to allow full reasoning chains
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_logs (
                log_id INTEGER,
                date DATE NOT NULL,
                agent_name VARCHAR NOT NULL,
                symbol VARCHAR,
                message TEXT NOT NULL,
                role VARCHAR DEFAULT 'analysis',
                context_id VARCHAR,
                created_at_sim_time VARCHAR
            )
        """
        )

        # Indexes for historical tables
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_daily_state_date ON daily_state(date)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hist_signals_date ON historical_signals(date)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hist_signals_symbol ON historical_signals(symbol)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_logs_date ON agent_logs(date)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_logs_symbol ON agent_logs(symbol)"
        )

        # =====================================================================
        # AGENTIC LEARNING SYSTEM TABLES
        # =====================================================================

        # Lessons learned from trade outcomes (ReflectionAgent)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_lessons (
                lesson_id TEXT PRIMARY KEY,
                lesson_text TEXT NOT NULL,
                applies_to JSON,
                confidence REAL DEFAULT 0.5,
                source_trade_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT DEFAULT 'ReflectionAgent',
                usage_count INT DEFAULT 0,
                effectiveness_score REAL
            )
        """
        )

        # Prompt modification proposals (PromptTunerAgent)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_proposals (
                proposal_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                section TEXT NOT NULL,
                old_text TEXT,
                new_text TEXT NOT NULL,
                reason TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT DEFAULT 'PromptTunerAgent'
            )
        """
        )

        # A/B tests for prompt changes
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ab_tests (
                test_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                control_version TEXT,
                treatment_version TEXT,
                traffic_pct REAL DEFAULT 0.2,
                status TEXT DEFAULT 'running',
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                control_trades INT DEFAULT 0,
                control_wins INT DEFAULT 0,
                treatment_trades INT DEFAULT 0,
                treatment_wins INT DEFAULT 0
            )
        """
        )

        # Active lesson injections
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lesson_injections (
                agent_id TEXT,
                lesson_id TEXT,
                injected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                PRIMARY KEY (agent_id, lesson_id)
            )
        """
        )

        # Learning metrics for tracking improvement over time
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_metrics (
                checkpoint_id INTEGER PRIMARY KEY,
                date DATE NOT NULL,
                trade_count INTEGER NOT NULL,
                rolling_win_rate REAL,
                cumulative_win_rate REAL,
                rolling_pnl REAL,
                cumulative_pnl REAL,
                lessons_active INTEGER DEFAULT 0,
                prompt_changes INTEGER DEFAULT 0,
                strategy_weights JSON,
                avg_confidence REAL,
                regime_at_checkpoint VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Indexes for learning tables
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_lessons_created ON agent_lessons(created_at)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_proposals_agent ON prompt_proposals(agent_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_proposals_status ON prompt_proposals(status)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ab_tests_agent ON ab_tests(agent_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ab_tests_status ON ab_tests(status)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_injections_active ON lesson_injections(is_active)"
        )

        self.conn.commit()
