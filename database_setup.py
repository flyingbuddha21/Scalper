#!/usr/bin/env python3
"""
Production-Grade Database Setup for Trading System
PostgreSQL + SQLite hybrid architecture with real-time caching
"""

import asyncio
import asyncpg
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import os
from pathlib import Path

# Import system components
from config_manager import ConfigManager
from utils import Logger, ErrorHandler, DataValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradeRecord:
    """Trade record data structure"""
    trade_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    price: float
    timestamp: datetime
    status: str
    strategy: str
    pnl: Optional[float] = None

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    ltp: float

class TradingDatabase:
    """
    Production-grade database manager with PostgreSQL primary and SQLite cache
    Integrates with the trading bot's config and utility systems
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        self.db_config = self.config['database']
        
        # Database connections
        self.pg_pool = None
        self.sqlite_conn = None
        self.sqlite_path = self.db_config.get('sqlite_path', 'data/realtime_cache.db')
        
        # Initialize logger and error handler
        self.logger = Logger(__name__)
        self.error_handler = ErrorHandler()
        self.data_validator = DataValidator()
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
        
        self.logger.info("Trading Database initialized")
    
    async def initialize(self):
        """Initialize database connections"""
        try:
            # PostgreSQL connection
            self.pg_pool = await asyncpg.create_pool(
                host=self.db_config['postgresql']['host'],
                port=self.db_config['postgresql']['port'],
                database=self.db_config['postgresql']['database'],
                user=self.db_config['postgresql']['user'],
                password=self.db_config['postgresql']['password'],
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            # SQLite connection
            self.sqlite_conn = sqlite3.connect(
                self.sqlite_path,
                check_same_thread=False,
                timeout=30.0
            )
            self.sqlite_conn.execute("PRAGMA journal_mode=WAL")
            self.sqlite_conn.execute("PRAGMA synchronous=NORMAL")
            self.sqlite_conn.execute("PRAGMA cache_size=10000")
            self.sqlite_conn.execute("PRAGMA temp_store=memory")
            
            self.logger.info("Database connections established successfully")
            return True
            
        except Exception as e:
            error_msg = f"Database initialization failed: {str(e)}"
            self.logger.error(error_msg)
            self.error_handler.handle_error(e, "database_initialization")
            raise
    
    async def create_tables(self):
        """Create all required database tables"""
        try:
            await self._create_postgresql_tables()
            await self._create_sqlite_tables()
            await self._create_indexes()
            self.logger.info("Database tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Table creation failed: {str(e)}")
            raise
    
    async def _create_postgresql_tables(self):
        """Create PostgreSQL tables for persistent storage"""
        async with self.pg_pool.acquire() as conn:
            
            # Stocks master table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS stocks (
                    symbol VARCHAR(20) PRIMARY KEY,
                    company_name VARCHAR(255),
                    sector VARCHAR(100),
                    market_cap BIGINT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Market data historical
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data_history (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) REFERENCES stocks(symbol),
                    timestamp TIMESTAMP NOT NULL,
                    open_price DECIMAL(12,4),
                    high_price DECIMAL(12,4),
                    low_price DECIMAL(12,4),
                    close_price DECIMAL(12,4),
                    volume BIGINT,
                    ltp DECIMAL(12,4),
                    timeframe VARCHAR(10) DEFAULT '1min',
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(symbol, timestamp, timeframe)
                )
            """)
            
            # Trading strategies
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trading_strategies (
                    strategy_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(100) UNIQUE NOT NULL,
                    description TEXT,
                    parameters JSONB,
                    is_active BOOLEAN DEFAULT true,
                    risk_level INTEGER DEFAULT 3,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Portfolio holdings
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) REFERENCES stocks(symbol),
                    quantity INTEGER NOT NULL,
                    avg_price DECIMAL(12,4) NOT NULL,
                    current_price DECIMAL(12,4),
                    unrealized_pnl DECIMAL(15,4),
                    last_updated TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Trade orders
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_orders (
                    order_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    symbol VARCHAR(20) REFERENCES stocks(symbol),
                    side VARCHAR(10) NOT NULL CHECK (side IN ('BUY', 'SELL')),
                    order_type VARCHAR(20) DEFAULT 'MARKET',
                    quantity INTEGER NOT NULL,
                    price DECIMAL(12,4),
                    executed_price DECIMAL(12,4),
                    executed_quantity INTEGER DEFAULT 0,
                    status VARCHAR(20) DEFAULT 'PENDING',
                    strategy_id UUID REFERENCES trading_strategies(strategy_id),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    executed_at TIMESTAMP,
                    
                    CHECK (quantity > 0),
                    CHECK (price IS NULL OR price > 0),
                    CHECK (status IN ('PENDING', 'PARTIAL', 'FILLED', 'CANCELLED', 'REJECTED'))
                )
            """)
            
            # Trade executions
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_executions (
                    execution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    order_id UUID REFERENCES trade_orders(order_id),
                    symbol VARCHAR(20),
                    side VARCHAR(10),
                    quantity INTEGER NOT NULL,
                    price DECIMAL(12,4) NOT NULL,
                    total_value DECIMAL(15,4) NOT NULL,
                    commission DECIMAL(10,4) DEFAULT 0,
                    taxes DECIMAL(10,4) DEFAULT 0,
                    net_amount DECIMAL(15,4),
                    execution_time TIMESTAMP DEFAULT NOW(),
                    
                    CHECK (quantity > 0),
                    CHECK (price > 0),
                    CHECK (total_value > 0)
                )
            """)
            
            # Portfolio performance tracking
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_performance (
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL,
                    total_value DECIMAL(15,4) NOT NULL,
                    invested_amount DECIMAL(15,4) NOT NULL,
                    realized_pnl DECIMAL(15,4) DEFAULT 0,
                    unrealized_pnl DECIMAL(15,4) DEFAULT 0,
                    day_pnl DECIMAL(15,4) DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT NOW(),
                    
                    UNIQUE(date)
                )
            """)
            
            # Strategy performance
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id SERIAL PRIMARY KEY,
                    strategy_id UUID REFERENCES trading_strategies(strategy_id),
                    date DATE NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    profitable_trades INTEGER DEFAULT 0,
                    total_pnl DECIMAL(15,4) DEFAULT 0,
                    max_drawdown DECIMAL(10,4) DEFAULT 0,
                    sharpe_ratio DECIMAL(8,4),
                    win_rate DECIMAL(5,2),
                    avg_trade_pnl DECIMAL(12,4),
                    created_at TIMESTAMP DEFAULT NOW(),
                    
                    UNIQUE(strategy_id, date)
                )
            """)
            
            # Risk metrics
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20),
                    timestamp TIMESTAMP DEFAULT NOW(),
                    volatility DECIMAL(8,4),
                    beta DECIMAL(6,4),
                    var_95 DECIMAL(12,4),
                    var_99 DECIMAL(12,4),
                    max_position_size INTEGER,
                    current_exposure DECIMAL(15,4),
                    risk_score INTEGER CHECK (risk_score BETWEEN 1 AND 10)
                )
            """)
            
            # System logs
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id SERIAL PRIMARY KEY,
                    level VARCHAR(20) NOT NULL,
                    message TEXT NOT NULL,
                    module VARCHAR(100),
                    function_name VARCHAR(100),
                    timestamp TIMESTAMP DEFAULT NOW(),
                    details JSONB
                )
            """)
            
            # Market sessions
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS market_sessions (
                    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_date DATE NOT NULL,
                    market_open TIMESTAMP,
                    market_close TIMESTAMP,
                    pre_market_start TIMESTAMP,
                    after_market_end TIMESTAMP,
                    is_trading_day BOOLEAN DEFAULT true,
                    notes TEXT
                )
            """)
    
    async def _create_sqlite_tables(self):
        """Create SQLite tables for real-time caching"""
        cursor = self.sqlite_conn.cursor()
        
        try:
            # Real-time market data cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS realtime_quotes (
                    symbol TEXT PRIMARY KEY,
                    ltp REAL NOT NULL,
                    volume INTEGER,
                    bid_price REAL,
                    ask_price REAL,
                    bid_qty INTEGER,
                    ask_qty INTEGER,
                    day_high REAL,
                    day_low REAL,
                    day_open REAL,
                    prev_close REAL,
                    change_percent REAL,
                    last_updated INTEGER NOT NULL
                )
            """)
            
            # Real-time portfolio cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_cache (
                    symbol TEXT PRIMARY KEY,
                    quantity INTEGER NOT NULL,
                    avg_price REAL NOT NULL,
                    current_price REAL,
                    unrealized_pnl REAL,
                    day_pnl REAL,
                    last_updated INTEGER NOT NULL
                )
            """)
            
            # Active orders cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS active_orders (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL,
                    status TEXT NOT NULL,
                    strategy TEXT,
                    created_at INTEGER NOT NULL,
                    last_updated INTEGER NOT NULL
                )
            """)
            
            # Strategy signals cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    strength REAL,
                    price REAL NOT NULL,
                    timestamp INTEGER NOT NULL,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            # Market scanner results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scanner_results (
                    symbol TEXT PRIMARY KEY,
                    scanner_name TEXT NOT NULL,
                    score REAL NOT NULL,
                    signals TEXT,
                    last_price REAL,
                    volume INTEGER,
                    timestamp INTEGER NOT NULL
                )
            """)
            
            # Performance metrics cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_cache (
                    metric_name TEXT PRIMARY KEY,
                    value REAL NOT NULL,
                    last_updated INTEGER NOT NULL
                )
            """)
            
            self.sqlite_conn.commit()
            
        except Exception as e:
            self.sqlite_conn.rollback()
            raise e
    
    async def _create_indexes(self):
        """Create database indexes for performance"""
        async with self.pg_pool.acquire() as conn:
            # PostgreSQL indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data_history(symbol, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_trade_orders_symbol_status ON trade_orders(symbol, status)",
                "CREATE INDEX IF NOT EXISTS idx_trade_orders_created ON trade_orders(created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_trade_executions_time ON trade_executions(execution_time DESC)",
                "CREATE INDEX IF NOT EXISTS idx_portfolio_symbol ON portfolio(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_strategy_performance_date ON strategy_performance(date DESC)",
                "CREATE INDEX IF NOT EXISTS idx_risk_metrics_symbol_time ON risk_metrics(symbol, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level)"
            ]
            
            for index_sql in indexes:
                await conn.execute(index_sql)
        
        # SQLite indexes
        cursor = self.sqlite_conn.cursor()
        sqlite_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_realtime_quotes_updated ON realtime_quotes(last_updated DESC)",
            "CREATE INDEX IF NOT EXISTS idx_strategy_signals_symbol_time ON strategy_signals(symbol, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_scanner_results_score ON scanner_results(score DESC)",
            "CREATE INDEX IF NOT EXISTS idx_active_orders_symbol ON active_orders(symbol)"
        ]
        
        for index_sql in sqlite_indexes:
            cursor.execute(index_sql)
        
        self.sqlite_conn.commit()
    
    # MARKET DATA OPERATIONS
    async def store_market_data(self, data: MarketData):
        """Store market data in both PostgreSQL and SQLite"""
        try:
            # Store in PostgreSQL for historical data
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO market_data_history 
                    (symbol, timestamp, open_price, high_price, low_price, close_price, volume, ltp)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (symbol, timestamp, timeframe) DO UPDATE SET
                    open_price = EXCLUDED.open_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    close_price = EXCLUDED.close_price,
                    volume = EXCLUDED.volume,
                    ltp = EXCLUDED.ltp
                """, data.symbol, data.timestamp, data.open_price, data.high_price,
                    data.low_price, data.close_price, data.volume, data.ltp)
            
            # Update SQLite cache for real-time access
            await self.update_realtime_quote(data)
            
        except Exception as e:
            self.logger.error(f"Failed to store market data: {e}")
            raise
    
    async def update_realtime_quote(self, data: MarketData):
        """Update real-time quote in SQLite cache"""
        try:
            cursor = self.sqlite_conn.cursor()
            
            # Calculate change percentage
            prev_close = data.close_price  # Simplified - should get actual previous close
            change_percent = ((data.ltp - prev_close) / prev_close * 100) if prev_close > 0 else 0
            
            cursor.execute("""
                INSERT OR REPLACE INTO realtime_quotes 
                (symbol, ltp, volume, day_high, day_low, day_open, prev_close, change_percent, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (data.symbol, data.ltp, data.volume, data.high_price, data.low_price,
                  data.open_price, prev_close, change_percent, int(data.timestamp.timestamp())))
            
            self.sqlite_conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to update realtime quote: {e}")
            raise
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price from SQLite cache"""
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("SELECT ltp FROM realtime_quotes WHERE symbol = ?", (symbol,))
            result = cursor.fetchone()
            return result[0] if result else None
            
        except Exception as e:
            self.logger.error(f"Failed to get latest price: {e}")
            return None
    
    async def get_historical_data(self, symbol: str, start_time: datetime, 
                                end_time: datetime, timeframe: str = '1min') -> List[Dict]:
        """Get historical market data"""
        try:
            async with self.pg_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT timestamp, open_price, high_price, low_price, close_price, volume, ltp
                    FROM market_data_history 
                    WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3 AND timeframe = $4
                    ORDER BY timestamp ASC
                """, symbol, start_time, end_time, timeframe)
                
                return [
                    {
                        'timestamp': row['timestamp'],
                        'open': float(row['open_price']),
                        'high': float(row['high_price']),
                        'low': float(row['low_price']),
                        'close': float(row['close_price']),
                        'volume': row['volume'],
                        'ltp': float(row['ltp'])
                    }
                    for row in rows
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to get historical data: {e}")
            return []
    
    # TRADING OPERATIONS
    async def store_trade_order(self, order: Dict) -> str:
        """Store trade order in database"""
        try:
            async with self.pg_pool.acquire() as conn:
                order_id = await conn.fetchval("""
                    INSERT INTO trade_orders 
                    (symbol, side, order_type, quantity, price, status, strategy_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING order_id
                """, order['symbol'], order['side'], order.get('order_type', 'MARKET'),
                    order['quantity'], order.get('price'), order.get('status', 'PENDING'),
                    order.get('strategy_id'))
            
            # Cache in SQLite for real-time access
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT INTO active_orders 
                (order_id, symbol, side, quantity, price, status, strategy, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (str(order_id), order['symbol'], order['side'], order['quantity'],
                  order.get('price'), order.get('status', 'PENDING'), 
                  order.get('strategy'), int(datetime.now().timestamp()),
                  int(datetime.now().timestamp())))
            
            self.sqlite_conn.commit()
            return str(order_id)
            
        except Exception as e:
            self.logger.error(f"Failed to store trade order: {e}")
            raise
    
    async def update_order_status(self, order_id: str, status: str, 
                                executed_price: float = None, executed_quantity: int = None):
        """Update order status"""
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE trade_orders 
                    SET status = $1, executed_price = $2, executed_quantity = $3, 
                        updated_at = NOW(), executed_at = CASE WHEN $1 = 'FILLED' THEN NOW() ELSE executed_at END
                    WHERE order_id = $4
                """, status, executed_price, executed_quantity, order_id)
            
            # Update SQLite cache
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                UPDATE active_orders 
                SET status = ?, last_updated = ?
                WHERE order_id = ?
            """, (status, int(datetime.now().timestamp()), order_id))
            
            # Remove from active orders if filled or cancelled
            if status in ['FILLED', 'CANCELLED', 'REJECTED']:
                cursor.execute("DELETE FROM active_orders WHERE order_id = ?", (order_id,))
            
            self.sqlite_conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to update order status: {e}")
            raise
    
    async def get_active_orders(self, symbol: str = None) -> List[Dict]:
        """Get active orders from cache"""
        try:
            cursor = self.sqlite_conn.cursor()
            
            if symbol:
                cursor.execute("""
                    SELECT * FROM active_orders WHERE symbol = ? ORDER BY created_at DESC
                """, (symbol,))
            else:
                cursor.execute("SELECT * FROM active_orders ORDER BY created_at DESC")
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
            
        except Exception as e:
            self.logger.error(f"Failed to get active orders: {e}")
            return []
    
    # PORTFOLIO OPERATIONS
    async def update_portfolio_position(self, symbol: str, quantity: int, avg_price: float):
        """Update portfolio position"""
        try:
            async with self.pg_pool.acquire() as conn:
                # Check if position exists
                existing = await conn.fetchrow("""
                    SELECT quantity, avg_price FROM portfolio WHERE symbol = $1
                """, symbol)
                
                if existing:
                    # Calculate new average price
                    old_qty = existing['quantity']
                    old_avg = float(existing['avg_price'])
                    
                    new_qty = old_qty + quantity
                    if new_qty != 0:
                        new_avg = ((old_qty * old_avg) + (quantity * avg_price)) / new_qty
                    else:
                        new_avg = 0
                    
                    await conn.execute("""
                        UPDATE portfolio 
                        SET quantity = $1, avg_price = $2, last_updated = NOW()
                        WHERE symbol = $3
                    """, new_qty, new_avg, symbol)
                else:
                    # Create new position
                    await conn.execute("""
                        INSERT INTO portfolio (symbol, quantity, avg_price)
                        VALUES ($1, $2, $3)
                    """, symbol, quantity, avg_price)
            
            # Update cache
            await self._update_portfolio_cache(symbol)
            
        except Exception as e:
            self.logger.error(f"Failed to update portfolio position: {e}")
            raise
    
    async def _update_portfolio_cache(self, symbol: str):
        """Update portfolio cache in SQLite"""
        try:
            async with self.pg_pool.acquire() as conn:
                position = await conn.fetchrow("""
                    SELECT * FROM portfolio WHERE symbol = $1
                """, symbol)
                
                if position:
                    current_price = await self.get_latest_price(symbol)
                    if current_price:
                        unrealized_pnl = (current_price - float(position['avg_price'])) * position['quantity']
                        
                        cursor = self.sqlite_conn.cursor()
                        cursor.execute("""
                            INSERT OR REPLACE INTO portfolio_cache 
                            (symbol, quantity, avg_price, current_price, unrealized_pnl, last_updated)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (symbol, position['quantity'], float(position['avg_price']),
                              current_price, unrealized_pnl, int(datetime.now().timestamp())))
                        
                        self.sqlite_conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to update portfolio cache: {e}")
    
    async def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary from cache"""
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_positions,
                    SUM(quantity * avg_price) as invested_amount,
                    SUM(quantity * current_price) as current_value,
                    SUM(unrealized_pnl) as total_unrealized_pnl
                FROM portfolio_cache
                WHERE quantity != 0
            """)
            
            result = cursor.fetchone()
            if result:
                invested_amount = result[1] or 0
                current_value = result[2] or 0
                return {
                    'total_positions': result[0],
                    'invested_amount': invested_amount,
                    'current_value': current_value,
                    'total_unrealized_pnl': result[3] or 0,
                    'total_return_percent': ((current_value - invested_amount) / invested_amount * 100) if invested_amount > 0 else 0
                }
            
            return {
                'total_positions': 0,
                'invested_amount': 0,
                'current_value': 0,
                'total_unrealized_pnl': 0,
                'total_return_percent': 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get portfolio summary: {e}")
            return {}
    
    # PERFORMANCE TRACKING
    async def log_performance_metric(self, metric_name: str, value: float):
        """Log performance metric to cache"""
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO performance_cache (metric_name, value, last_updated)
                VALUES (?, ?, ?)
            """, (metric_name, value, int(datetime.now().timestamp())))
            
            self.sqlite_conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to log performance metric: {e}")
    
    async def get_performance_metrics(self) -> Dict:
        """Get all performance metrics from cache"""
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("SELECT metric_name, value FROM performance_cache")
            
            return {row[0]: row[1] for row in cursor.fetchall()}
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {}
    
    # UTILITY METHODS
    async def cleanup_old_data(self, days: int = 30):
        """Clean up old data to maintain performance"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            async with self.pg_pool.acquire() as conn:
                # Clean old market data (keep daily/weekly aggregated data)
                await conn.execute("""
                    DELETE FROM market_data_history 
                    WHERE timestamp < $1 AND timeframe = '1min'
                """, cutoff_date)
                
                # Clean old system logs
                await conn.execute("""
                    DELETE FROM system_logs WHERE timestamp < $1
                """, cutoff_date)
            
            # Clean SQLite cache
            cursor = self.sqlite_conn.cursor()
            cutoff_timestamp = int(cutoff_date.timestamp())
            
            cursor.execute("""
                DELETE FROM strategy_signals WHERE timestamp < ? AND is_active = 0
            """, (cutoff_timestamp,))
            
            self.sqlite_conn.commit()
            
            self.logger.info(f"Cleaned up data older than {days} days")
            
        except Exception as e:
            self.logger.error(f"Data cleanup failed: {e}")
    
    async def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            stats = {}
            
            # PostgreSQL stats
            async with self.pg_pool.acquire() as conn:
                pg_stats = await conn.fetch("""
                    SELECT 
                        schemaname,
                        tablename,
                        n_tup_ins as inserts,
                        n_tup_upd as updates,
                        n_tup_del as deletes,
                        n_live_tup as live_rows
                    FROM pg_stat_user_tables
                    ORDER BY tablename
                """)
                
                stats['postgresql'] = [
                    {
                        'table': row['tablename'],
                        'inserts': row['inserts'],
                        'updates': row['updates'],
                        'deletes': row['deletes'],
                        'live_rows': row['live_rows']
                    }
                    for row in pg_stats
                ]
            
            # SQLite stats
            cursor = self.sqlite_conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            sqlite_stats = []
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                sqlite_stats.append({
                    'table': table_name,
                    'row_count': count
                })
            
            stats['sqlite'] = sqlite_stats
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {}
    
    async def close(self):
        """Close database connections"""
        try:
            if self.pg_pool:
                await self.pg_pool.close()
            
            if self.sqlite_conn:
                self.sqlite_conn.close()
            
            self.logger.info("Database connections closed")
            
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")

    # STRATEGY OPERATIONS
    async def store_strategy_signal(self, symbol: str, strategy: str, signal: str, 
                                  strength: float, price: float):
        """Store strategy signal in cache"""
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT INTO strategy_signals 
                (symbol, strategy, signal, strength, price, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol, strategy, signal, strength, price, int(datetime.now().timestamp())))
            
            self.sqlite_conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to store strategy signal: {e}")

    async def get_active_signals(self, symbol: str = None, strategy: str = None) -> List[Dict]:
        """Get active strategy signals"""
        try:
            cursor = self.sqlite_conn.cursor()
            
            query = "SELECT * FROM strategy_signals WHERE is_active = 1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if strategy:
                query += " AND strategy = ?"
                params.append(strategy)
            
            query += " ORDER BY timestamp DESC LIMIT 100"
            
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
            
        except Exception as e:
            self.logger.error(f"Failed to get active signals: {e}")
            return []

    async def update_strategy_performance(self, strategy_id: str, date: datetime,
                                        total_trades: int, profitable_trades: int,
                                        total_pnl: float, max_drawdown: float = None):
        """Update strategy performance metrics"""
        try:
            async with self.pg_pool.acquire() as conn:
                win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
                avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0
                
                await conn.execute("""
                    INSERT INTO strategy_performance 
                    (strategy_id, date, total_trades, profitable_trades, total_pnl, 
                     max_drawdown, win_rate, avg_trade_pnl)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (strategy_id, date) DO UPDATE SET
                    total_trades = EXCLUDED.total_trades,
                    profitable_trades = EXCLUDED.profitable_trades,
                    total_pnl = EXCLUDED.total_pnl,
                    max_drawdown = EXCLUDED.max_drawdown,
                    win_rate = EXCLUDED.win_rate,
                    avg_trade_pnl = EXCLUDED.avg_trade_pnl
                """, strategy_id, date.date(), total_trades, profitable_trades,
                    total_pnl, max_drawdown, win_rate, avg_trade_pnl)
                
        except Exception as e:
            self.logger.error(f"Failed to update strategy performance: {e}")

    # RISK MANAGEMENT OPERATIONS
    async def store_risk_metrics(self, symbol: str, volatility: float, beta: float = None,
                               var_95: float = None, var_99: float = None,
                               max_position_size: int = None, current_exposure: float = None,
                               risk_score: int = None):
        """Store risk metrics for a symbol"""
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO risk_metrics 
                    (symbol, volatility, beta, var_95, var_99, max_position_size, 
                     current_exposure, risk_score)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, symbol, volatility, beta, var_95, var_99, max_position_size,
                    current_exposure, risk_score)
                
        except Exception as e:
            self.logger.error(f"Failed to store risk metrics: {e}")

    async def get_risk_metrics(self, symbol: str) -> Optional[Dict]:
        """Get latest risk metrics for a symbol"""
        try:
            async with self.pg_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT * FROM risk_metrics 
                    WHERE symbol = $1 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, symbol)
                
                if result:
                    return {
                        'symbol': result['symbol'],
                        'volatility': float(result['volatility']) if result['volatility'] else None,
                        'beta': float(result['beta']) if result['beta'] else None,
                        'var_95': float(result['var_95']) if result['var_95'] else None,
                        'var_99': float(result['var_99']) if result['var_99'] else None,
                        'max_position_size': result['max_position_size'],
                        'current_exposure': float(result['current_exposure']) if result['current_exposure'] else None,
                        'risk_score': result['risk_score'],
                        'timestamp': result['timestamp']
                    }
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get risk metrics: {e}")
            return None

    # SCANNER OPERATIONS
    async def store_scanner_result(self, symbol: str, scanner_name: str, score: float,
                                 signals: List[str], last_price: float, volume: int):
        """Store market scanner results"""
        try:
            cursor = self.sqlite_conn.cursor()
            signals_json = json.dumps(signals)
            
            cursor.execute("""
                INSERT OR REPLACE INTO scanner_results 
                (symbol, scanner_name, score, signals, last_price, volume, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, scanner_name, score, signals_json, last_price, volume,
                  int(datetime.now().timestamp())))
            
            self.sqlite_conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to store scanner result: {e}")

    async def get_scanner_results(self, scanner_name: str = None, min_score: float = None,
                                limit: int = 50) -> List[Dict]:
        """Get market scanner results"""
        try:
            cursor = self.sqlite_conn.cursor()
            
            query = "SELECT * FROM scanner_results WHERE 1=1"
            params = []
            
            if scanner_name:
                query += " AND scanner_name = ?"
                params.append(scanner_name)
                
            if min_score:
                query += " AND score >= ?"
                params.append(min_score)
            
            query += " ORDER BY score DESC, timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            
            results = []
            for row in cursor.fetchall():
                result = dict(zip(columns, row))
                result['signals'] = json.loads(result['signals'])
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to get scanner results: {e}")
            return []

    # LOGGING OPERATIONS
    async def log_system_event(self, level: str, message: str, module: str = None,
                             function_name: str = None, details: Dict = None):
        """Log system event to database"""
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO system_logs (level, message, module, function_name, details)
                    VALUES ($1, $2, $3, $4, $5)
                """, level, message, module, function_name, json.dumps(details) if details else None)
                
        except Exception as e:
            # Don't log errors in logging to avoid recursion
            print(f"Failed to log system event: {e}")

    async def get_system_logs(self, level: str = None, module: str = None,
                            hours: int = 24, limit: int = 1000) -> List[Dict]:
        """Get system logs"""
        try:
            async with self.pg_pool.acquire() as conn:
                query = """
                    SELECT * FROM system_logs 
                    WHERE timestamp > NOW() - INTERVAL '%s hours'
                """ % hours
                
                params = []
                param_count = 0
                
                if level:
                    param_count += 1
                    query += f" AND level = ${param_count}"
                    params.append(level)
                
                if module:
                    param_count += 1
                    query += f" AND module = ${param_count}"
                    params.append(module)
                
                query += f" ORDER BY timestamp DESC LIMIT ${param_count + 1}"
                params.append(limit)
                
                rows = await conn.fetch(query, *params)
                
                return [
                    {
                        'id': row['id'],
                        'level': row['level'],
                        'message': row['message'],
                        'module': row['module'],
                        'function_name': row['function_name'],
                        'timestamp': row['timestamp'],
                        'details': row['details']
                    }
                    for row in rows
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to get system logs: {e}")
            return []

    # STOCK MASTER DATA
    async def add_stock(self, symbol: str, company_name: str, sector: str = None,
                       market_cap: int = None):
        """Add stock to master list"""
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO stocks (symbol, company_name, sector, market_cap)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (symbol) DO UPDATE SET
                    company_name = EXCLUDED.company_name,
                    sector = EXCLUDED.sector,
                    market_cap = EXCLUDED.market_cap,
                    updated_at = NOW()
                """, symbol, company_name, sector, market_cap)
                
        except Exception as e:
            self.logger.error(f"Failed to add stock: {e}")

    async def get_stocks(self, sector: str = None) -> List[Dict]:
        """Get stocks from master list"""
        try:
            async with self.pg_pool.acquire() as conn:
                if sector:
                    rows = await conn.fetch("""
                        SELECT * FROM stocks WHERE sector = $1 ORDER BY symbol
                    """, sector)
                else:
                    rows = await conn.fetch("SELECT * FROM stocks ORDER BY symbol")
                
                return [
                    {
                        'symbol': row['symbol'],
                        'company_name': row['company_name'],
                        'sector': row['sector'],
                        'market_cap': row['market_cap'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at']
                    }
                    for row in rows
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to get stocks: {e}")
            return []

    # MARKET SESSION MANAGEMENT
    async def create_market_session(self, session_date: datetime, market_open: datetime,
                                  market_close: datetime, is_trading_day: bool = True):
        """Create market session record"""
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO market_sessions 
                    (session_date, market_open, market_close, is_trading_day)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT DO NOTHING
                """, session_date.date(), market_open, market_close, is_trading_day)
                
        except Exception as e:
            self.logger.error(f"Failed to create market session: {e}")

    async def get_current_market_session(self) -> Optional[Dict]:
        """Get current market session"""
        try:
            today = datetime.now().date()
            
            async with self.pg_pool.acquire() as conn:
                session = await conn.fetchrow("""
                    SELECT * FROM market_sessions 
                    WHERE session_date = $1
                """, today)
                
                if session:
                    return {
                        'session_id': str(session['session_id']),
                        'session_date': session['session_date'],
                        'market_open': session['market_open'],
                        'market_close': session['market_close'],
                        'is_trading_day': session['is_trading_day']
                    }
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get current market session: {e}")
            return None

    # PORTFOLIO ANALYTICS
    async def calculate_daily_pnl(self, date: datetime = None) -> Dict:
        """Calculate daily P&L"""
        try:
            if not date:
                date = datetime.now().date()
            
            async with self.pg_pool.acquire() as conn:
                # Get all trades for the day
                trades = await conn.fetch("""
                    SELECT te.*, to.side
                    FROM trade_executions te
                    JOIN trade_orders to ON te.order_id = to.order_id
                    WHERE DATE(te.execution_time) = $1
                """, date)
                
                total_realized_pnl = 0
                total_trades = len(trades)
                buy_value = 0
                sell_value = 0
                
                for trade in trades:
                    net_amount = float(trade['net_amount'])
                    if trade['side'] == 'BUY':
                        buy_value += net_amount
                    else:  # SELL
                        sell_value += net_amount
                        total_realized_pnl += net_amount  # Simplified P&L calculation
                
                # Get unrealized P&L from current positions
                unrealized_pnl = 0
                positions = await conn.fetch("SELECT * FROM portfolio WHERE quantity != 0")
                
                for position in positions:
                    current_price = await self.get_latest_price(position['symbol'])
                    if current_price:
                        unrealized_pnl += (current_price - float(position['avg_price'])) * position['quantity']
                
                return {
                    'date': date,
                    'realized_pnl': total_realized_pnl,
                    'unrealized_pnl': unrealized_pnl,
                    'total_pnl': total_realized_pnl + unrealized_pnl,
                    'total_trades': total_trades,
                    'buy_value': buy_value,
                    'sell_value': sell_value
                }
                
        except Exception as e:
            self.logger.error(f"Failed to calculate daily P&L: {e}")
            return {}

    async def get_trade_history(self, symbol: str = None, start_date: datetime = None,
                              end_date: datetime = None, limit: int = 100) -> List[Dict]:
        """Get trade execution history"""
        try:
            async with self.pg_pool.acquire() as conn:
                query = """
                    SELECT te.*, to.side, to.strategy_id, ts.name as strategy_name
                    FROM trade_executions te
                    JOIN trade_orders to ON te.order_id = to.order_id
                    LEFT JOIN trading_strategies ts ON to.strategy_id = ts.strategy_id
                    WHERE 1=1
                """
                params = []
                param_count = 0
                
                if symbol:
                    param_count += 1
                    query += f" AND te.symbol = ${param_count}"
                    params.append(symbol)
                
                if start_date:
                    param_count += 1
                    query += f" AND te.execution_time >= ${param_count}"
                    params.append(start_date)
                
                if end_date:
                    param_count += 1
                    query += f" AND te.execution_time <= ${param_count}"
                    params.append(end_date)
                
                query += f" ORDER BY te.execution_time DESC LIMIT ${param_count + 1}"
                params.append(limit)
                
                trades = await conn.fetch(query, *params)
                
                return [
                    {
                        'execution_id': str(trade['execution_id']),
                        'order_id': str(trade['order_id']),
                        'symbol': trade['symbol'],
                        'side': trade['side'],
                        'quantity': trade['quantity'],
                        'price': float(trade['price']),
                        'total_value': float(trade['total_value']),
                        'commission': float(trade['commission']),
                        'taxes': float(trade['taxes']),
                        'net_amount': float(trade['net_amount']),
                        'execution_time': trade['execution_time'],
                        'strategy_name': trade['strategy_name']
                    }
                    for trade in trades
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to get trade history: {e}")
            return []

    # BACKUP AND MAINTENANCE
    async def backup_data(self, backup_path: str):
        """Create database backup"""
        try:
            import subprocess
            from pathlib import Path
            
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # PostgreSQL backup
            pg_backup_file = backup_dir / f"postgresql_backup_{timestamp}.sql"
            pg_config = self.db_config['postgresql']
            
            subprocess.run([
                'pg_dump',
                '-h', pg_config['host'],
                '-p', str(pg_config['port']),
                '-U', pg_config['user'],
                '-d', pg_config['database'],
                '-f', str(pg_backup_file),
                '--no-password'
            ], check=True, env={'PGPASSWORD': pg_config['password']})
            
            # SQLite backup
            sqlite_backup_file = backup_dir / f"sqlite_backup_{timestamp}.db"
            import shutil
            shutil.copy2(self.sqlite_path, sqlite_backup_file)
            
            self.logger.info(f"Database backup completed: {backup_dir}")
            
            return {
                'postgresql_backup': str(pg_backup_file),
                'sqlite_backup': str(sqlite_backup_file),
                'timestamp': timestamp
            }
            
        except Exception as e:
            self.logger.error(f"Database backup failed: {e}")
            raise

    async def verify_data_integrity(self) -> Dict:
        """Verify database data integrity"""
        try:
            issues = []
            
            async with self.pg_pool.acquire() as conn:
                # Check for orphaned records
                orphaned_executions = await conn.fetchval("""
                    SELECT COUNT(*) FROM trade_executions te
                    LEFT JOIN trade_orders to ON te.order_id = to.order_id
                    WHERE to.order_id IS NULL
                """)
                
                if orphaned_executions > 0:
                    issues.append(f"Found {orphaned_executions} orphaned trade executions")
                
                # Check for inconsistent portfolio data
                portfolio_inconsistencies = await conn.fetch("""
                    SELECT symbol, quantity, avg_price 
                    FROM portfolio 
                    WHERE quantity < 0 OR avg_price <= 0
                """)
                
                if portfolio_inconsistencies:
                    issues.extend([f"Invalid portfolio data for {row['symbol']}" for row in portfolio_inconsistencies])
                
                # Check for missing market data
                missing_data_stocks = await conn.fetch("""
                    SELECT s.symbol 
                    FROM stocks s
                    LEFT JOIN market_data_history md ON s.symbol = md.symbol
                    WHERE md.symbol IS NULL
                """)
                
                if missing_data_stocks:
                    issues.extend([f"No market data for {row['symbol']}" for row in missing_data_stocks])
            
            # Check SQLite integrity
            cursor = self.sqlite_conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            sqlite_integrity = cursor.fetchone()[0]
            
            if sqlite_integrity != "ok":
                issues.append(f"SQLite integrity check failed: {sqlite_integrity}")
            
            return {
                'status': 'PASSED' if not issues else 'FAILED',
                'issues': issues,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Data integrity check failed: {e}")
            return {
                'status': 'ERROR',
                'issues': [f"Integrity check error: {str(e)}"],
                'timestamp': datetime.now()
            }

# Example usage and testing
async def main():
    """Example usage of the TradingDatabase"""
    from config_manager import ConfigManager
    
    # Initialize with config
    config_manager = ConfigManager("config/config.yaml")
    trading_db = TradingDatabase(config_manager)
    
    try:
        # Initialize database
        await trading_db.initialize()
        await trading_db.create_tables()
        
        # Add some test stocks
        await trading_db.add_stock("RELIANCE", "Reliance Industries", "Energy", 1500000)
        await trading_db.add_stock("TCS", "Tata Consultancy Services", "IT", 1200000)
        await trading_db.add_stock("INFY", "Infosys", "IT", 800000)
        
        # Test market data storage
        market_data = MarketData(
            symbol="RELIANCE",
            timestamp=datetime.now(),
            open_price=2500.0,
            high_price=2520.0,
            low_price=2495.0,
            close_price=2515.0,
            volume=1000000,
            ltp=2515.0
        )
        
        await trading_db.store_market_data(market_data)
        
        # Test order placement
        order = {
            'symbol': 'RELIANCE',
            'side': 'BUY',
            'quantity': 100,
            'price': 2515.0,
            'status': 'PENDING',
            'strategy': 'momentum_strategy'
        }
        
        order_id = await trading_db.store_trade_order(order)
        print(f"Order placed with ID: {order_id}")
        
        # Test portfolio operations
        await trading_db.update_portfolio_position("RELIANCE", 100, 2515.0)
        portfolio_summary = await trading_db.get_portfolio_summary()
        print(f"Portfolio Summary: {portfolio_summary}")
        
        # Test performance metrics
        await trading_db.log_performance_metric("total_pnl", 1500.0)
        await trading_db.log_performance_metric("win_rate", 65.5)
        
        metrics = await trading_db.get_performance_metrics()
        print(f"Performance Metrics: {metrics}")
        
        # Test data integrity
        integrity_check = await trading_db.verify_data_integrity()
        print(f"Data Integrity: {integrity_check['status']}")
        
        # Test database stats
        stats = await trading_db.get_database_stats()
        print(f"Database Stats: {json.dumps(stats, indent=2, default=str)}")
        
        print("Database setup and testing completed successfully!")
        
    except Exception as e:
        print(f"Error during database testing: {e}")
        raise
    finally:
        await trading_db.close()

if __name__ == "__main__":
    asyncio.run(main())
