#!/usr/bin/env python3
"""
Database Setup and Management
Creates all necessary database schemas and tables
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages all database creation and setup"""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.data_path.mkdir(exist_ok=True)
        
        # Database file paths
        self.databases = {
            'trading_bot': self.data_path / 'trading_bot.db',
            'paper_trades': self.data_path / 'paper_trades.db',
            'market_data': self.data_path / 'market_data.db',
            'scanner_cache': self.data_path / 'scanner_cache.db',
            'volatility_data': self.data_path / 'volatility_data.db',
            'execution_queue': self.data_path / 'execution_queue.db'
        }
        
        logger.info("🗄️ Database Manager initialized")
    
    def create_all_databases(self):
        """Create all database schemas"""
        try:
            self.create_trading_bot_db()
            self.create_paper_trades_db()
            self.create_market_data_db()
            self.create_scanner_cache_db()
            self.create_volatility_data_db()
            self.create_execution_queue_db()
            
            logger.info("✅ All databases created successfully")
            
        except Exception as e:
            logger.error(f"❌ Database creation error: {e}")
            raise
    
    def create_trading_bot_db(self):
        """Create main trading bot database"""
        conn = sqlite3.connect(self.databases['trading_bot'])
        cursor = conn.cursor()
        
        # Bot configuration table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bot_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_key TEXT UNIQUE NOT NULL,
                config_value TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Trading sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                initial_capital REAL NOT NULL,
                final_capital REAL,
                total_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                max_drawdown REAL DEFAULT 0.0,
                win_rate REAL DEFAULT 0.0,
                session_notes TEXT,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Strategy performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                signals_generated INTEGER DEFAULT 0,
                trades_executed INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                win_rate REAL DEFAULT 0.0,
                avg_hold_time REAL DEFAULT 0.0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES trading_sessions(session_id)
            )
        """)
        
        # System logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                log_level TEXT NOT NULL,
                component TEXT NOT NULL,
                message TEXT NOT NULL,
                details TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_time ON trading_sessions(start_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy_perf ON strategy_performance(session_id, strategy_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_time ON system_logs(timestamp)")
        
        # Insert default configuration
        default_configs = [
            ('initial_capital', '100000.0'),
            ('paper_mode', 'true'),
            ('max_positions', '10'),
            ('risk_per_trade', '1.0'),
            ('stop_loss_pct', '0.5'),
            ('take_profit_pct', '1.0')
        ]
        
        for key, value in default_configs:
            cursor.execute("""
                INSERT OR IGNORE INTO bot_config (config_key, config_value)
                VALUES (?, ?)
            """, (key, value))
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Trading bot database created")
    
    def create_paper_trades_db(self):
        """Create paper trading database"""
        conn = sqlite3.connect(self.databases['paper_trades'])
        cursor = conn.cursor()
        
        # Orders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL,
                status TEXT NOT NULL,
                filled_quantity INTEGER DEFAULT 0,
                filled_price REAL,
                placed_time DATETIME NOT NULL,
                filled_time DATETIME,
                slippage REAL DEFAULT 0.0,
                total_cost REAL DEFAULT 0.0,
                spread_pct REAL DEFAULT 0.0
            )
        """)
        
        # Positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                avg_entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                net_investment REAL NOT NULL,
                unrealized_pnl REAL DEFAULT 0.0,
                total_pnl REAL DEFAULT 0.0,
                entry_time DATETIME NOT NULL,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                hold_time_minutes REAL DEFAULT 0.0,
                max_profit REAL DEFAULT 0.0,
                max_loss REAL DEFAULT 0.0,
                is_open BOOLEAN DEFAULT 1
            )
        """)
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                entry_time DATETIME NOT NULL,
                exit_time DATETIME NOT NULL,
                hold_time_minutes REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                commission REAL DEFAULT 0.0,
                entry_order_id TEXT,
                exit_order_id TEXT,
                strategy_name TEXT,
                exit_reason TEXT
            )
        """)
        
        # Portfolio snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_capital REAL NOT NULL,
                available_margin REAL NOT NULL,
                used_margin REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                total_pnl REAL NOT NULL,
                open_positions INTEGER NOT NULL,
                daily_trades INTEGER NOT NULL
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_time ON orders(placed_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(exit_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_time ON portfolio_snapshots(timestamp)")
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Paper trades database created")
    
    def create_market_data_db(self):
        """Create market data database"""
        conn = sqlite3.connect(self.databases['market_data'])
        cursor = conn.cursor()
        
        # Real-time quotes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS realtime_quotes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ltp REAL NOT NULL,
                bid REAL,
                ask REAL,
                bid_qty INTEGER,
                ask_qty INTEGER,
                volume INTEGER,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                prev_close REAL,
                change_pct REAL
            )
        """)
        
        # Historical OHLCV data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                timeframe TEXT NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                close_price REAL NOT NULL,
                volume INTEGER NOT NULL,
                UNIQUE(symbol, timestamp, timeframe)
            )
        """)
        
        # Market events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                symbol TEXT,
                description TEXT NOT NULL,
                impact_level TEXT DEFAULT 'LOW',
                data TEXT
            )
        """)
        
        # Instrument master table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS instruments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                exchange TEXT NOT NULL,
                instrument_type TEXT NOT NULL,
                lot_size INTEGER DEFAULT 1,
                tick_size REAL DEFAULT 0.05,
                is_active BOOLEAN DEFAULT 1,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_quotes_symbol_time ON realtime_quotes(symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time ON ohlcv_data(symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON market_events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_instruments_symbol ON instruments(symbol)")
        
        # Insert sample instruments
        sample_instruments = [
            ('RELIANCE', 'Reliance Industries Ltd', 'NSE', 'EQUITY', 1, 0.05),
            ('TCS', 'Tata Consultancy Services Ltd', 'NSE', 'EQUITY', 1, 0.05),
            ('INFY', 'Infosys Ltd', 'NSE', 'EQUITY', 1, 0.05),
            ('HDFCBANK', 'HDFC Bank Ltd', 'NSE', 'EQUITY', 1, 0.05),
            ('ICICIBANK', 'ICICI Bank Ltd', 'NSE', 'EQUITY', 1, 0.05),
            ('NIFTY', 'Nifty 50 Index', 'NSE', 'INDEX', 50, 0.05),
            ('BANKNIFTY', 'Bank Nifty Index', 'NSE', 'INDEX', 25, 0.05)
        ]
        
        for symbol, name, exchange, inst_type, lot_size, tick_size in sample_instruments:
            cursor.execute("""
                INSERT OR IGNORE INTO instruments 
                (symbol, name, exchange, instrument_type, lot_size, tick_size)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol, name, exchange, inst_type, lot_size, tick_size))
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Market data database created")
    
    def create_scanner_cache_db(self):
        """Create scanner cache database"""
        conn = sqlite3.connect(self.databases['scanner_cache'])
        cursor = conn.cursor()
        
        # Scanned stocks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scanned_stocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                current_price REAL,
                volume INTEGER,
                volatility_score REAL,
                liquidity_score REAL,
                spread_pct REAL,
                atr_pct REAL,
                iv_rank REAL,
                price_change_pct REAL,
                volume_ratio REAL,
                market_cap REAL,
                scalping_score REAL,
                scan_timestamp DATETIME,
                is_active BOOLEAN DEFAULT 1,
                UNIQUE(symbol, scan_timestamp)
            )
        """)
        
        # Scan history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_timestamp DATETIME,
                total_stocks_scanned INTEGER,
                avg_volatility REAL,
                top_10_symbols TEXT,
                scan_duration_seconds REAL
            )
        """)
        
        # Scanner configuration table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scanner_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_key TEXT UNIQUE NOT NULL,
                config_value TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scanned_symbol_time ON scanned_stocks(symbol, scan_timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_history_time ON scan_history(scan_timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scanned_score ON scanned_stocks(scalping_score)")
        
        # Insert default scanner configuration
        default_scanner_configs = [
            ('scan_interval_minutes', '20'),
            ('max_stocks_to_scan', '50'),
            ('min_volatility_score', '30'),
            ('min_liquidity_score', '40'),
            ('top_stocks_count', '10')
        ]
        
        for key, value in default_scanner_configs:
            cursor.execute("""
                INSERT OR IGNORE INTO scanner_config (config_key, config_value)
                VALUES (?, ?)
            """, (key, value))
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Scanner cache database created")
    
    def create_volatility_data_db(self):
        """Create volatility analysis database"""
        conn = sqlite3.connect(self.databases['volatility_data'])
        cursor = conn.cursor()
        
        # Volatility metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS volatility_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                price_volatility REAL,
                intraday_range REAL,
                range_percentage REAL,
                atr_14 REAL,
                atr_percentage REAL,
                bollinger_squeeze BOOLEAN,
                vwap_deviation REAL,
                volume_weighted_volatility REAL,
                standard_deviation REAL,
                variance REAL,
                skewness REAL,
                kurtosis REAL,
                volatility_rank REAL,
                volatility_percentile REAL,
                volatility_forecast REAL,
                trend_direction TEXT,
                momentum_score REAL,
                scalping_signal TEXT,
                volatility_breakout BOOLEAN,
                mean_reversion_signal BOOLEAN,
                UNIQUE(symbol, timestamp)
            )
        """)
        
        # Volatility alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS volatility_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT,
                volatility_value REAL,
                threshold_value REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Historical volatility rankings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS volatility_rankings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ranking_date DATE NOT NULL,
                symbol TEXT NOT NULL,
                rank_position INTEGER NOT NULL,
                volatility_score REAL NOT NULL,
                percentile_rank REAL NOT NULL,
                UNIQUE(ranking_date, symbol)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_volatility_symbol_time ON volatility_metrics(symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_symbol_time ON volatility_alerts(symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rankings_date ON volatility_rankings(ranking_date)")
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Volatility data database created")
    
    def create_execution_queue_db(self):
        """Create execution queue database"""
        conn = sqlite3.connect(self.databases['execution_queue'])
        cursor = conn.cursor()
        
        # Execution signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                price REAL NOT NULL,
                confidence REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                indicators TEXT,
                stop_loss REAL,
                take_profit REAL,
                executed BOOLEAN DEFAULT 0,
                order_id TEXT,
                execution_price REAL,
                execution_time DATETIME
            )
        """)
        
        # Active positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS active_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL,
                unrealized_pnl REAL,
                stop_loss REAL,
                take_profit REAL,
                entry_time DATETIME NOT NULL,
                last_update DATETIME,
                is_open BOOLEAN DEFAULT 1
            )
        """)
        
        # Strategy performance tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                signals_generated INTEGER DEFAULT 0,
                signals_executed INTEGER DEFAULT 0,
                successful_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                win_rate REAL DEFAULT 0.0,
                avg_confidence REAL DEFAULT 0.0,
                last_signal_time DATETIME,
                last_trade_time DATETIME,
                is_active BOOLEAN DEFAULT 1,
                UNIQUE(strategy_name, symbol)
            )
        """)
        
        # Risk management events
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                symbol TEXT,
                strategy_name TEXT,
                description TEXT NOT NULL,
                risk_level TEXT DEFAULT 'MEDIUM',
                action_taken TEXT,
                details TEXT
            )
        """)
        
        # Execution statistics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                total_signals INTEGER DEFAULT 0,
                signals_executed INTEGER DEFAULT 0,
                successful_executions INTEGER DEFAULT 0,
                execution_rate REAL DEFAULT 0.0,
                success_rate REAL DEFAULT 0.0,
                avg_execution_time_ms REAL DEFAULT 0.0,
                total_pnl REAL DEFAULT 0.0,
                UNIQUE(date)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol_time ON execution_signals(symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_strategy ON execution_signals(strategy_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON active_positions(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracking_strategy ON strategy_tracking(strategy_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_events_time ON risk_events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_date ON execution_stats(date)")
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Execution queue database created")
    
    def get_database_info(self) -> dict:
        """Get information about all databases"""
        info = {}
        
        for db_name, db_path in self.databases.items():
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Get table count
                cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
                table_count = cursor.fetchone()[0]
                
                # Get database size
                cursor.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                cursor.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                db_size = page_count * page_size
                
                info[db_name] = {
                    'path': str(db_path),
                    'exists': db_path.exists(),
                    'size_bytes': db_size,
                    'size_mb': round(db_size / (1024 * 1024), 2),
                    'table_count': table_count
                }
                
                conn.close()
                
            except Exception as e:
                info[db_name] = {
                    'path': str(db_path),
                    'exists': db_path.exists(),
                    'error': str(e)
                }
        
        return info
    
    def backup_databases(self, backup_path: Path = None):
        """Create backup of all databases"""
        if backup_path is None:
            backup_path = self.data_path.parent / "backups"
        
        backup_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backup_info = {
            'timestamp': timestamp,
            'backed_up': [],
            'errors': []
        }
        
        for db_name, db_path in self.databases.items():
            try:
                if db_path.exists():
                    backup_file = backup_path / f"{db_name}_{timestamp}.db"
                    
                    # Copy database file
                    import shutil
                    shutil.copy2(db_path, backup_file)
                    
                    backup_info['backed_up'].append({
                        'database': db_name,
                        'backup_file': str(backup_file),
                        'original_size': db_path.stat().st_size,
                        'backup_size': backup_file.stat().st_size
                    })
                    
                    logger.info(f"✅ Backed up {db_name} to {backup_file}")
                
            except Exception as e:
                backup_info['errors'].append({
                    'database': db_name,
                    'error': str(e)
                })
                logger.error(f"❌ Backup failed for {db_name}: {e}")
        
        # Save backup manifest
        manifest_file = backup_path / f"backup_manifest_{timestamp}.json"
        with open(manifest_file, 'w') as f:
            json.dump(backup_info, f, indent=2)
        
        logger.info(f"📦 Database backup completed: {len(backup_info['backed_up'])} databases")
        return backup_info
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data from databases"""
        cleanup_info = {
            'timestamp': datetime.now().isoformat(),
            'days_to_keep': days_to_keep,
            'cleaned': [],
            'errors': []
        }
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        cleanup_queries = [
            # Clean old market data
            {
                'database': 'market_data',
                'query': "DELETE FROM realtime_quotes WHERE timestamp < ?",
                'table': 'realtime_quotes'
            },
            # Clean old volatility data
            {
                'database': 'volatility_data',
                'query': "DELETE FROM volatility_metrics WHERE timestamp < ?",
                'table': 'volatility_metrics'
            },
            # Clean old alerts
            {
                'database': 'volatility_data',
                'query': "DELETE FROM volatility_alerts WHERE timestamp < ?",
                'table': 'volatility_alerts'
            },
            # Clean old execution signals
            {
                'database': 'execution_queue',
                'query': "DELETE FROM execution_signals WHERE timestamp < ? AND executed = 1",
                'table': 'execution_signals'
            },
            # Clean old risk events
            {
                'database': 'execution_queue',
                'query': "DELETE FROM risk_events WHERE timestamp < ?",
                'table': 'risk_events'
            }
        ]
        
        for cleanup in cleanup_queries:
            try:
                db_path = self.databases[cleanup['database']]
                if not db_path.exists():
                    continue
                
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Count records before cleanup
                cursor.execute(f"SELECT COUNT(*) FROM {cleanup['table']} WHERE timestamp < ?", (cutoff_date,))
                records_to_delete = cursor.fetchone()[0]
                
                if records_to_delete > 0:
                    # Perform cleanup
                    cursor.execute(cleanup['query'], (cutoff_date,))
                    deleted_count = cursor.rowcount
                    
                    cleanup_info['cleaned'].append({
                        'database': cleanup['database'],
                        'table': cleanup['table'],
                        'records_deleted': deleted_count
                    })
                    
                    logger.info(f"🧹 Cleaned {deleted_count} old records from {cleanup['database']}.{cleanup['table']}")
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                cleanup_info['errors'].append({
                    'database': cleanup['database'],
                    'table': cleanup['table'],
                    'error': str(e)
                })
                logger.error(f"❌ Cleanup failed for {cleanup['database']}.{cleanup['table']}: {e}")
        
        logger.info(f"🧹 Database cleanup completed")
        return cleanup_info
    
    def optimize_databases(self):
        """Optimize all databases (vacuum, reindex)"""
        optimization_info = {
            'timestamp': datetime.now().isoformat(),
            'optimized': [],
            'errors': []
        }
        
        for db_name, db_path in self.databases.items():
            try:
                if not db_path.exists():
                    continue
                
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Get database size before optimization
                cursor.execute("PRAGMA page_count")
                pages_before = cursor.fetchone()[0]
                cursor.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                size_before = pages_before * page_size
                
                # Vacuum database
                cursor.execute("VACUUM")
                
                # Reindex database
                cursor.execute("REINDEX")
                
                # Get database size after optimization
                cursor.execute("PRAGMA page_count")
                pages_after = cursor.fetchone()[0]
                size_after = pages_after * page_size
                
                optimization_info['optimized'].append({
                    'database': db_name,
                    'size_before_mb': round(size_before / (1024 * 1024), 2),
                    'size_after_mb': round(size_after / (1024 * 1024), 2),
                    'space_saved_mb': round((size_before - size_after) / (1024 * 1024), 2),
                    'pages_before': pages_before,
                    'pages_after': pages_after
                })
                
                conn.close()
                
                logger.info(f"⚡ Optimized {db_name}: {round((size_before - size_after) / (1024 * 1024), 2)} MB saved")
                
            except Exception as e:
                optimization_info['errors'].append({
                    'database': db_name,
                    'error': str(e)
                })
                logger.error(f"❌ Optimization failed for {db_name}: {e}")
        
        logger.info(f"⚡ Database optimization completed")
        return optimization_info


# Usage example and testing
if __name__ == "__main__":
    from pathlib import Path
    import tempfile
    
    # Test database manager
    print("🗄️ Testing Database Manager...")
    
    # Use temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        data_path = Path(temp_dir) / "data"
        db_manager = DatabaseManager(data_path)
        
        # Create all databases
        db_manager.create_all_databases()
        
        # Get database info
        info = db_manager.get_database_info()
        print(f"📊 Database Info:")
        for db_name, db_info in info.items():
            print(f"  {db_name}: {db_info.get('table_count', 0)} tables, {db_info.get('size_mb', 0)} MB")
        
        # Test backup
        backup_info = db_manager.backup_databases()
        print(f"📦 Backup completed: {len(backup_info['backed_up'])} databases")
        
        # Test optimization
        opt_info = db_manager.optimize_databases()
        print(f"⚡ Optimization completed: {len(opt_info['optimized'])} databases")
        
        print("✅ Database manager test completed")
