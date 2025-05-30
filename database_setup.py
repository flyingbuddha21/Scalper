#!/usr/bin/env python3
"""
Production Database Management System
Handles all database operations for the trading bot
PostgreSQL for production, SQLite for development
"""

import asyncio
import asyncpg
import sqlite3
import logging
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import os
import psycopg2
from contextlib import asynccontextmanager
import aiofiles

@dataclass
class TradeRecord:
    id: Optional[int] = None
    timestamp: datetime = None
    symbol: str = ""
    action: str = ""  # BUY, SELL
    strategy: str = ""
    quantity: int = 0
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.0
    mode: str = ""  # live, paper
    status: str = "PENDING"  # PENDING, FILLED, CANCELLED, FAILED
    realized_pnl: float = 0.0
    commission: float = 0.0
    order_id: Optional[str] = None
    exit_timestamp: Optional[datetime] = None
    exit_reason: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class PerformanceMetrics:
    id: Optional[int] = None
    timestamp: datetime = None
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    portfolio_value: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class DatabaseManager:
    def __init__(self, config):
        """Initialize database manager with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Database configuration
        self.db_type = config.get('database_type', 'sqlite')  # 'postgresql' or 'sqlite'
        self.db_config = config.get('database_config', {})
        
        # Connection pools
        self.pg_pool = None
        self.sqlite_path = self.db_config.get('sqlite_path', 'trading_bot.db')
        
        # Connection status
        self.is_connected = False
        
        # Table schemas
        self.schemas = self._get_table_schemas()
        
        self.logger.info(f"Database Manager initialized - Type: {self.db_type}")
    
    async def initialize(self):
        """Initialize database connection and create tables"""
        try:
            if self.db_type == 'postgresql':
                await self._initialize_postgresql()
            else:
                await self._initialize_sqlite()
            
            # Create tables
            await self._create_tables()
            
            # Run migrations if needed
            await self._run_migrations()
            
            self.is_connected = True
            self.logger.info("✅ Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def _initialize_postgresql(self):
        """Initialize PostgreSQL connection pool"""
        try:
            pg_config = self.db_config.get('postgresql', {})
            
            self.pg_pool = await asyncpg.create_pool(
                host=pg_config.get('host', 'localhost'),
                port=pg_config.get('port', 5432),
                user=pg_config.get('user', 'trading_bot'),
                password=pg_config.get('password', ''),
                database=pg_config.get('database', 'trading_bot'),
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            
            self.logger.info("PostgreSQL connection pool created")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise
    
    async def _initialize_sqlite(self):
        """Initialize SQLite database"""
        try:
            # Ensure directory exists
            db_dir = os.path.dirname(self.sqlite_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
            
            self.logger.info(f"SQLite database path: {self.sqlite_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SQLite: {e}")
            raise
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection (async context manager)"""
        if self.db_type == 'postgresql':
            async with self.pg_pool.acquire() as conn:
                yield conn
        else:
            # For SQLite, we'll use a synchronous connection
            conn = sqlite3.connect(self.sqlite_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
    
    def _get_table_schemas(self) -> Dict[str, str]:
        """Get table schemas for different database types"""
        if self.db_type == 'postgresql':
            return {
                'trades': '''
                    CREATE TABLE IF NOT EXISTS trades (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        symbol VARCHAR(20) NOT NULL,
                        action VARCHAR(10) NOT NULL,
                        strategy VARCHAR(50) NOT NULL,
                        quantity INTEGER NOT NULL,
                        entry_price DECIMAL(15,4) NOT NULL,
                        exit_price DECIMAL(15,4),
                        stop_loss DECIMAL(15,4),
                        take_profit DECIMAL(15,4),
                        confidence DECIMAL(5,4),
                        mode VARCHAR(10) NOT NULL,
                        status VARCHAR(20) DEFAULT 'PENDING',
                        realized_pnl DECIMAL(15,4) DEFAULT 0,
                        commission DECIMAL(10,4) DEFAULT 0,
                        order_id VARCHAR(100),
                        exit_timestamp TIMESTAMP WITH TIME ZONE,
                        exit_reason VARCHAR(100),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                ''',
                'performance_metrics': '''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        losing_trades INTEGER DEFAULT 0,
                        total_pnl DECIMAL(15,4) DEFAULT 0,
                        daily_pnl DECIMAL(15,4) DEFAULT 0,
                        win_rate DECIMAL(5,4) DEFAULT 0,
                        avg_win DECIMAL(15,4) DEFAULT 0,
                        avg_loss DECIMAL(15,4) DEFAULT 0,
                        max_drawdown DECIMAL(5,4) DEFAULT 0,
                        sharpe_ratio DECIMAL(10,6) DEFAULT 0,
                        profit_factor DECIMAL(10,4) DEFAULT 0,
                        portfolio_value DECIMAL(15,4) DEFAULT 0,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                ''',
                'market_data': '''
                    CREATE TABLE IF NOT EXISTS market_data (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        symbol VARCHAR(20) NOT NULL,
                        price DECIMAL(15,4) NOT NULL,
                        volume BIGINT DEFAULT 0,
                        open_price DECIMAL(15,4),
                        high_price DECIMAL(15,4),
                        low_price DECIMAL(15,4),
                        close_price DECIMAL(15,4),
                        data_source VARCHAR(50),
                        quality_score DECIMAL(3,2),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                ''',
                'scan_results': '''
                    CREATE TABLE IF NOT EXISTS scan_results (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        symbol VARCHAR(20) NOT NULL,
                        opportunity_type VARCHAR(50) NOT NULL,
                        probability DECIMAL(5,4) NOT NULL,
                        entry_price DECIMAL(15,4) NOT NULL,
                        target_price DECIMAL(15,4) NOT NULL,
                        stop_loss DECIMAL(15,4) NOT NULL,
                        volume_surge DECIMAL(10,4),
                        momentum_score DECIMAL(10,6),
                        volatility_rating VARCHAR(20),
                        strategy_signals TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                ''',
                'system_logs': '''
                    CREATE TABLE IF NOT EXISTS system_logs (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        level VARCHAR(20) NOT NULL,
                        component VARCHAR(50) NOT NULL,
                        message TEXT NOT NULL,
                        metadata JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                ''',
                'risk_alerts': '''
                    CREATE TABLE IF NOT EXISTS risk_alerts (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        symbol VARCHAR(20) NOT NULL,
                        alert_type VARCHAR(50) NOT NULL,
                        severity VARCHAR(20) NOT NULL,
                        message TEXT NOT NULL,
                        current_value DECIMAL(15,6),
                        threshold DECIMAL(15,6),
                        acknowledged BOOLEAN DEFAULT FALSE,
                        acknowledged_by VARCHAR(100),
                        acknowledged_at TIMESTAMP WITH TIME ZONE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                '''
            }
        else:
            # SQLite schemas
            return {
                'trades': '''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL,
                        stop_loss REAL,
                        take_profit REAL,
                        confidence REAL,
                        mode TEXT NOT NULL,
                        status TEXT DEFAULT 'PENDING',
                        realized_pnl REAL DEFAULT 0,
                        commission REAL DEFAULT 0,
                        order_id TEXT,
                        exit_timestamp DATETIME,
                        exit_reason TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''',
                'performance_metrics': '''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        losing_trades INTEGER DEFAULT 0,
                        total_pnl REAL DEFAULT 0,
                        daily_pnl REAL DEFAULT 0,
                        win_rate REAL DEFAULT 0,
                        avg_win REAL DEFAULT 0,
                        avg_loss REAL DEFAULT 0,
                        max_drawdown REAL DEFAULT 0,
                        sharpe_ratio REAL DEFAULT 0,
                        profit_factor REAL DEFAULT 0,
                        portfolio_value REAL DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''',
                'market_data': '''
                    CREATE TABLE IF NOT EXISTS market_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT NOT NULL,
                        price REAL NOT NULL,
                        volume INTEGER DEFAULT 0,
                        open_price REAL,
                        high_price REAL,
                        low_price REAL,
                        close_price REAL,
                        data_source TEXT,
                        quality_score REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''',
                'scan_results': '''
                    CREATE TABLE IF NOT EXISTS scan_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT NOT NULL,
                        opportunity_type TEXT NOT NULL,
                        probability REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        target_price REAL NOT NULL,
                        stop_loss REAL NOT NULL,
                        volume_surge REAL,
                        momentum_score REAL,
                        volatility_rating TEXT,
                        strategy_signals TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''',
                'system_logs': '''
                    CREATE TABLE IF NOT EXISTS system_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        level TEXT NOT NULL,
                        component TEXT NOT NULL,
                        message TEXT NOT NULL,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''',
                'risk_alerts': '''
                    CREATE TABLE IF NOT EXISTS risk_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT NOT NULL,
                        alert_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        message TEXT NOT NULL,
                        current_value REAL,
                        threshold REAL,
                        acknowledged INTEGER DEFAULT 0,
                        acknowledged_by TEXT,
                        acknowledged_at DATETIME,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                '''
            }
    
    async def _create_tables(self):
        """Create database tables"""
        try:
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    for table_name, schema in self.schemas.items():
                        await conn.execute(schema)
                        self.logger.debug(f"Created/verified table: {table_name}")
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    for table_name, schema in self.schemas.items():
                        conn.execute(schema)
                        self.logger.debug(f"Created/verified table: {table_name}")
                    conn.commit()
            
            # Create indexes
            await self._create_indexes()
            
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise
    
    async def _create_indexes(self):
        """Create database indexes for performance"""
        try:
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)",
                "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_scan_results_symbol ON scan_results(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_scan_results_timestamp ON scan_results(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_system_logs_component ON system_logs(component)",
                "CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_risk_alerts_symbol ON risk_alerts(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_risk_alerts_severity ON risk_alerts(severity)"
            ]
            
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    for index_sql in indexes:
                        try:
                            await conn.execute(index_sql)
                        except Exception as e:
                            self.logger.debug(f"Index creation note: {e}")
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    for index_sql in indexes:
                        try:
                            conn.execute(index_sql)
                        except Exception as e:
                            self.logger.debug(f"Index creation note: {e}")
                    conn.commit()
            
            self.logger.info("Database indexes created/verified")
            
        except Exception as e:
            self.logger.error(f"Error creating indexes: {e}")
    
    async def _run_migrations(self):
        """Run database migrations"""
        try:
            # Check if migrations table exists
            migration_table = '''
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''
            
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    await conn.execute(migration_table.replace('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', 'TIMESTAMP WITH TIME ZONE DEFAULT NOW()'))
                    
                    # Get applied migrations
                    applied = await conn.fetch("SELECT version FROM schema_migrations ORDER BY version")
                    applied_versions = set([row['version'] for row in applied])
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.execute(migration_table)
                    
                    # Get applied migrations
                    cursor = conn.execute("SELECT version FROM schema_migrations ORDER BY version")
                    applied_versions = set([row[0] for row in cursor.fetchall()])
                    conn.commit()
            
            # Define migrations
            migrations = {
                1: "ALTER TABLE trades ADD COLUMN IF NOT EXISTS metadata TEXT",
                2: "ALTER TABLE performance_metrics ADD COLUMN IF NOT EXISTS trading_session_id TEXT",
                # Add more migrations as needed
            }
            
            # Apply pending migrations
            for version, sql in migrations.items():
                if version not in applied_versions:
                    try:
                        await self._apply_migration(version, sql)
                        self.logger.info(f"Applied migration version {version}")
                    except Exception as e:
                        self.logger.warning(f"Migration {version} failed (may be expected): {e}")
            
        except Exception as e:
            self.logger.error(f"Error running migrations: {e}")
    
    async def _apply_migration(self, version: int, sql: str):
        """Apply a single migration"""
        if self.db_type == 'postgresql':
            async with self.get_connection() as conn:
                async with conn.transaction():
                    await conn.execute(sql)
                    await conn.execute(
                        "INSERT INTO schema_migrations (version) VALUES ($1)",
                        version
                    )
        else:
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute(sql)
                conn.execute("INSERT INTO schema_migrations (version) VALUES (?)", (version,))
                conn.commit()
    
    # Trade Management Methods
    async def log_trade(self, trade_data: Dict) -> int:
        """Log a new trade"""
        try:
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    query = '''
                        INSERT INTO trades (
                            timestamp, symbol, action, strategy, quantity, entry_price,
                            stop_loss, take_profit, confidence, mode, order_id
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        RETURNING id
                    '''
                    result = await conn.fetchrow(
                        query,
                        trade_data.get('timestamp', datetime.now()),
                        trade_data['symbol'],
                        trade_data['action'],
                        trade_data['strategy'],
                        trade_data['quantity'],
                        trade_data['price'],
                        trade_data.get('stop_loss'),
                        trade_data.get('take_profit'),
                        trade_data.get('confidence', 0),
                        trade_data.get('mode', 'paper'),
                        trade_data.get('order_id')
                    )
                    return result['id']
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    cursor = conn.execute('''
                        INSERT INTO trades (
                            timestamp, symbol, action, strategy, quantity, entry_price,
                            stop_loss, take_profit, confidence, mode, order_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        trade_data.get('timestamp', datetime.now()),
                        trade_data['symbol'],
                        trade_data['action'],
                        trade_data['strategy'],
                        trade_data['quantity'],
                        trade_data['price'],
                        trade_data.get('stop_loss'),
                        trade_data.get('take_profit'),
                        trade_data.get('confidence', 0),
                        trade_data.get('mode', 'paper'),
                        trade_data.get('order_id')
                    ))
                    conn.commit()
                    return cursor.lastrowid
                    
        except Exception as e:
            self.logger.error(f"Error logging trade: {e}")
            raise
    
    async def update_trade_exit(self, trade_id: int, exit_price: float, exit_reason: str = None):
        """Update trade with exit information"""
        try:
            realized_pnl = 0  # Calculate based on entry/exit prices
            
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    # Get trade details first
                    trade = await conn.fetchrow("SELECT * FROM trades WHERE id = $1", trade_id)
                    if trade:
                        if trade['action'] == 'BUY':
                            realized_pnl = (exit_price - trade['entry_price']) * trade['quantity']
                        else:  # SELL
                            realized_pnl = (trade['entry_price'] - exit_price) * trade['quantity']
                        
                        await conn.execute('''
                            UPDATE trades 
                            SET exit_price = $1, exit_timestamp = $2, exit_reason = $3, 
                                realized_pnl = $4, status = 'FILLED', updated_at = NOW()
                            WHERE id = $5
                        ''', exit_price, datetime.now(), exit_reason, realized_pnl, trade_id)
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    # Get trade details first
                    cursor = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
                    trade = cursor.fetchone()
                    if trade:
                        if trade['action'] == 'BUY':
                            realized_pnl = (exit_price - trade['entry_price']) * trade['quantity']
                        else:  # SELL
                            realized_pnl = (trade['entry_price'] - exit_price) * trade['quantity']
                        
                        conn.execute('''
                            UPDATE trades 
                            SET exit_price = ?, exit_timestamp = ?, exit_reason = ?, 
                                realized_pnl = ?, status = 'FILLED', updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        ''', (exit_price, datetime.now(), exit_reason, realized_pnl, trade_id))
                        conn.commit()
                        
        except Exception as e:
            self.logger.error(f"Error updating trade exit: {e}")
            raise
    
    async def get_trades_by_date(self, date: datetime.date) -> List[Dict]:
        """Get trades for a specific date"""
        try:
            start_date = datetime.combine(date, datetime.min.time())
            end_date = datetime.combine(date, datetime.max.time())
            
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    rows = await conn.fetch(
                        "SELECT * FROM trades WHERE timestamp >= $1 AND timestamp <= $2 ORDER BY timestamp",
                        start_date, end_date
                    )
                    return [dict(row) for row in rows]
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        "SELECT * FROM trades WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
                        (start_date, end_date)
                    )
                    return [dict(row) for row in cursor.fetchall()]
                    
        except Exception as e:
            self.logger.error(f"Error getting trades by date: {e}")
            return []
    
    async def get_all_trades(self, limit: int = 1000) -> List[Dict]:
        """Get all trades with optional limit"""
        try:
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    rows = await conn.fetch(
                        "SELECT * FROM trades ORDER BY timestamp DESC LIMIT $1",
                        limit
                    )
                    return [dict(row) for row in rows]
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
                        (limit,)
                    )
                    return [dict(row) for row in cursor.fetchall()]
                    
        except Exception as e:
            self.logger.error(f"Error getting all trades: {e}")
            return []
    
    async def get_open_positions(self) -> List[Dict]:
        """Get currently open positions"""
        try:
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    rows = await conn.fetch(
                        "SELECT * FROM trades WHERE status = 'FILLED' AND exit_price IS NULL ORDER BY timestamp DESC"
                    )
                    return [dict(row) for row in rows]
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        "SELECT * FROM trades WHERE status = 'FILLED' AND exit_price IS NULL ORDER BY timestamp DESC"
                    )
                    return [dict(row) for row in cursor.fetchall()]
                    
        except Exception as e:
            self.logger.error(f"Error getting open positions: {e}")
            return []
    
    # Performance Metrics Methods
    async def save_performance_metrics(self, metrics: Dict):
        """Save performance metrics"""
        try:
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    await conn.execute('''
                        INSERT INTO performance_metrics (
                            total_trades, winning_trades, losing_trades, total_pnl, daily_pnl,
                            win_rate, avg_win, avg_loss, max_drawdown, sharpe_ratio, 
                            profit_factor, portfolio_value
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ''', 
                        metrics.get('total_trades', 0),
                        metrics.get('winning_trades', 0),
                        metrics.get('losing_trades', 0),
                        metrics.get('total_pnl', 0),
                        metrics.get('daily_pnl', 0),
                        metrics.get('win_rate', 0),
                        metrics.get('avg_win', 0),
                        metrics.get('avg_loss', 0),
                        metrics.get('max_drawdown', 0),
                        metrics.get('sharpe_ratio', 0),
                        metrics.get('profit_factor', 0),
                        metrics.get('portfolio_value', 0)
                    )
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.execute('''
                        INSERT INTO performance_metrics (
                            total_trades, winning_trades, losing_trades, total_pnl, daily_pnl,
                            win_rate, avg_win, avg_loss, max_drawdown, sharpe_ratio, 
                            profit_factor, portfolio_value
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        metrics.get('total_trades', 0),
                        metrics.get('winning_trades', 0),
                        metrics.get('losing_trades', 0),
                        metrics.get('total_pnl', 0),
                        metrics.get('daily_pnl', 0),
                        metrics.get('win_rate', 0),
                        metrics.get('avg_win', 0),
                        metrics.get('avg_loss', 0),
                        metrics.get('max_drawdown', 0),
                        metrics.get('sharpe_ratio', 0),
                        metrics.get('profit_factor', 0),
                        metrics.get('portfolio_value', 0)
                    ))
                    conn.commit()
                    
        except Exception as e:
            self.logger.error(f"Error saving performance metrics: {e}")
    
    async def get_latest_performance_metrics(self) -> Optional[Dict]:
        """Get latest performance metrics"""
        try:
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    row = await conn.fetchrow(
                        "SELECT * FROM performance_metrics ORDER BY timestamp DESC LIMIT 1"
                    )
                    return dict(row) if row else None
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        "SELECT * FROM performance_metrics ORDER BY timestamp DESC LIMIT 1"
                    )
                    row = cursor.fetchone()
                    return dict(row) if row else None
                    
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return None
    
    # Market Data Methods
    async def store_market_data(self, symbol: str, data: Dict):
        """Store market data tick"""
        try:
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    await conn.execute('''
                        INSERT INTO market_data (
                            symbol, price, volume, open_price, high_price, low_price, 
                            close_price, data_source, quality_score
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ''',
                        symbol,
                        data.get('price', 0),
                        data.get('volume', 0),
                        data.get('open', data.get('open_price')),
                        data.get('high', data.get('high_price')),
                        data.get('low', data.get('low_price')),
                        data.get('close', data.get('close_price')),
                        data.get('data_source', 'live_feed'),
                        data.get('quality_score', 1.0)
                    )
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.execute('''
                        INSERT INTO market_data (
                            symbol, price, volume, open_price, high_price, low_price, 
                            close_price, data_source, quality_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        data.get('price', 0),
                        data.get('volume', 0),
                        data.get('open', data.get('open_price')),
                        data.get('high', data.get('high_price')),
                        data.get('low', data.get('low_price')),
                        data.get('close', data.get('close_price')),
                        data.get('data_source', 'live_feed'),
                        data.get('quality_score', 1.0)
                    ))
                    conn.commit()
                    
        except Exception as e:
            self.logger.error(f"Error storing market data for {symbol}: {e}")
    
    async def get_market_data_history(self, symbol: str, hours_back: int = 24) -> List[Dict]:
        """Get market data history for a symbol"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    rows = await conn.fetch(
                        "SELECT * FROM market_data WHERE symbol = $1 AND timestamp >= $2 ORDER BY timestamp",
                        symbol, cutoff_time
                    )
                    return [dict(row) for row in rows]
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        "SELECT * FROM market_data WHERE symbol = ? AND timestamp >= ? ORDER BY timestamp",
                        (symbol, cutoff_time)
                    )
                    return [dict(row) for row in cursor.fetchall()]
                    
        except Exception as e:
            self.logger.error(f"Error getting market data history for {symbol}: {e}")
            return []
    
    # Scan Results Methods
    async def store_scan_result(self, scan_result):
        """Store scan result"""
        try:
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    await conn.execute('''
                        INSERT INTO scan_results (
                            symbol, opportunity_type, probability, entry_price, target_price,
                            stop_loss, volume_surge, momentum_score, volatility_rating, strategy_signals
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ''',
                        scan_result.symbol,
                        scan_result.opportunity_type,
                        scan_result.probability,
                        scan_result.entry_price,
                        scan_result.target_price,
                        scan_result.stop_loss,
                        scan_result.volume_surge,
                        scan_result.momentum_score,
                        scan_result.volatility_rating,
                        '|'.join(scan_result.strategy_signals)
                    )
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.execute('''
                        INSERT INTO scan_results (
                            symbol, opportunity_type, probability, entry_price, target_price,
                            stop_loss, volume_surge, momentum_score, volatility_rating, strategy_signals
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        scan_result.symbol,
                        scan_result.opportunity_type,
                        scan_result.probability,
                        scan_result.entry_price,
                        scan_result.target_price,
                        scan_result.stop_loss,
                        scan_result.volume_surge,
                        scan_result.momentum_score,
                        scan_result.volatility_rating,
                        '|'.join(scan_result.strategy_signals)
                    ))
                    conn.commit()
                    
        except Exception as e:
            self.logger.error(f"Error storing scan result: {e}")
    
    async def get_scan_results_history(self, hours_back: int = 24) -> List[Dict]:
        """Get scan results history"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    rows = await conn.fetch(
                        "SELECT * FROM scan_results WHERE timestamp >= $1 ORDER BY timestamp DESC",
                        cutoff_time
                    )
                    return [dict(row) for row in rows]
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        "SELECT * FROM scan_results WHERE timestamp >= ? ORDER BY timestamp DESC",
                        (cutoff_time,)
                    )
                    return [dict(row) for row in cursor.fetchall()]
                    
        except Exception as e:
            self.logger.error(f"Error getting scan results history: {e}")
            return []
    
    # System Logging Methods
    async def log_system_event(self, level: str, component: str, message: str, metadata: Dict = None):
        """Log system event"""
        try:
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    await conn.execute(
                        "INSERT INTO system_logs (level, component, message, metadata) VALUES ($1, $2, $3, $4)",
                        level, component, message, json.dumps(metadata) if metadata else None
                    )
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.execute(
                        "INSERT INTO system_logs (level, component, message, metadata) VALUES (?, ?, ?, ?)",
                        (level, component, message, json.dumps(metadata) if metadata else None)
                    )
                    conn.commit()
                    
        except Exception as e:
            self.logger.error(f"Error logging system event: {e}")
    
    async def get_system_logs(self, component: str = None, level: str = None, hours_back: int = 24) -> List[Dict]:
        """Get system logs with optional filters"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            base_query = "SELECT * FROM system_logs WHERE timestamp >= "
            params = [cutoff_time]
            
            if component:
                base_query += " AND component = "
                params.append(component)
            
            if level:
                base_query += " AND level = "
                params.append(level)
            
            base_query += " ORDER BY timestamp DESC"
            
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    # Build parameterized query
                    param_placeholders = []
                    for i, _ in enumerate(params, 1):
                        param_placeholders.append(f'${i}')
                    
                    query = base_query
                    for i, placeholder in enumerate(param_placeholders):
                        if i == 0:
                            query = query.replace("timestamp >= ", f"timestamp >= {placeholder}")
                        elif component and i == 1:
                            query = query.replace(" AND component = ", f" AND component = {placeholder}")
                        elif level and ((component and i == 2) or (not component and i == 1)):
                            query = query.replace(" AND level = ", f" AND level = {placeholder}")
                    
                    rows = await conn.fetch(query, *params)
                    return [dict(row) for row in rows]
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.row_factory = sqlite3.Row
                    
                    # Build parameterized query for SQLite
                    query = base_query
                    for _ in params:
                        query = query.replace(" = ", " = ? ", 1)
                        query = query.replace(">= ", ">= ? ", 1)
                    
                    cursor = conn.execute(query, params)
                    return [dict(row) for row in cursor.fetchall()]
                    
        except Exception as e:
            self.logger.error(f"Error getting system logs: {e}")
            return []
    
    # Risk Alert Methods
    async def store_risk_alert(self, alert):
        """Store risk alert"""
        try:
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    await conn.execute('''
                        INSERT INTO risk_alerts (
                            symbol, alert_type, severity, message, current_value, threshold
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                    ''',
                        alert.symbol,
                        alert.alert_type,
                        alert.severity,
                        alert.message,
                        alert.current_value,
                        alert.threshold
                    )
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.execute('''
                        INSERT INTO risk_alerts (
                            symbol, alert_type, severity, message, current_value, threshold
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        alert.symbol,
                        alert.alert_type,
                        alert.severity,
                        alert.message,
                        alert.current_value,
                        alert.threshold
                    ))
                    conn.commit()
                    
        except Exception as e:
            self.logger.error(f"Error storing risk alert: {e}")
    
    async def get_unacknowledged_alerts(self) -> List[Dict]:
        """Get unacknowledged risk alerts"""
        try:
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    rows = await conn.fetch(
                        "SELECT * FROM risk_alerts WHERE acknowledged = FALSE ORDER BY timestamp DESC"
                    )
                    return [dict(row) for row in rows]
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        "SELECT * FROM risk_alerts WHERE acknowledged = 0 ORDER BY timestamp DESC"
                    )
                    return [dict(row) for row in cursor.fetchall()]
                    
        except Exception as e:
            self.logger.error(f"Error getting unacknowledged alerts: {e}")
            return []
    
    async def acknowledge_alert(self, alert_id: int, acknowledged_by: str):
        """Acknowledge a risk alert"""
        try:
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    await conn.execute(
                        "UPDATE risk_alerts SET acknowledged = TRUE, acknowledged_by = $1, acknowledged_at = NOW() WHERE id = $2",
                        acknowledged_by, alert_id
                    )
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.execute(
                        "UPDATE risk_alerts SET acknowledged = 1, acknowledged_by = ?, acknowledged_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (acknowledged_by, alert_id)
                    )
                    conn.commit()
                    
        except Exception as e:
            self.logger.error(f"Error acknowledging alert {alert_id}: {e}")
    
    # Analytics and Reporting Methods
    async def get_trading_summary(self, days_back: int = 30) -> Dict:
        """Get comprehensive trading summary"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    # Get trade statistics
                    trade_stats = await conn.fetchrow('''
                        SELECT 
                            COUNT(*) as total_trades,
                            COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) as winning_trades,
                            COUNT(CASE WHEN realized_pnl < 0 THEN 1 END) as losing_trades,
                            SUM(realized_pnl) as total_pnl,
                            AVG(CASE WHEN realized_pnl > 0 THEN realized_pnl END) as avg_win,
                            AVG(CASE WHEN realized_pnl < 0 THEN realized_pnl END) as avg_loss,
                            MAX(realized_pnl) as best_trade,
                            MIN(realized_pnl) as worst_trade
                        FROM trades 
                        WHERE timestamp >= $1 AND status = 'FILLED' AND exit_price IS NOT NULL
                    ''', cutoff_date)
                    
                    # Get daily P&L
                    daily_pnl = await conn.fetch('''
                        SELECT 
                            DATE(timestamp) as trade_date,
                            SUM(realized_pnl) as daily_pnl
                        FROM trades 
                        WHERE timestamp >= $1 AND status = 'FILLED' AND exit_price IS NOT NULL
                        GROUP BY DATE(timestamp)
                        ORDER BY trade_date
                    ''', cutoff_date)
                    
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.row_factory = sqlite3.Row
                    
                    # Get trade statistics
                    cursor = conn.execute('''
                        SELECT 
                            COUNT(*) as total_trades,
                            COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) as winning_trades,
                            COUNT(CASE WHEN realized_pnl < 0 THEN 1 END) as losing_trades,
                            SUM(realized_pnl) as total_pnl,
                            AVG(CASE WHEN realized_pnl > 0 THEN realized_pnl END) as avg_win,
                            AVG(CASE WHEN realized_pnl < 0 THEN realized_pnl END) as avg_loss,
                            MAX(realized_pnl) as best_trade,
                            MIN(realized_pnl) as worst_trade
                        FROM trades 
                        WHERE timestamp >= ? AND status = 'FILLED' AND exit_price IS NOT NULL
                    ''', (cutoff_date,))
                    trade_stats = cursor.fetchone()
                    
                    # Get daily P&L
                    cursor = conn.execute('''
                        SELECT 
                            DATE(timestamp) as trade_date,
                            SUM(realized_pnl) as daily_pnl
                        FROM trades 
                        WHERE timestamp >= ? AND status = 'FILLED' AND exit_price IS NOT NULL
                        GROUP BY DATE(timestamp)
                        ORDER BY trade_date
                    ''', (cutoff_date,))
                    daily_pnl = cursor.fetchall()
            
            # Calculate additional metrics
            win_rate = (trade_stats['winning_trades'] / trade_stats['total_trades']) if trade_stats['total_trades'] > 0 else 0
            profit_factor = abs(trade_stats['avg_win'] / trade_stats['avg_loss']) if trade_stats['avg_loss'] and trade_stats['avg_loss'] < 0 else 0
            
            return {
                'period_days': days_back,
                'total_trades': trade_stats['total_trades'] or 0,
                'winning_trades': trade_stats['winning_trades'] or 0,
                'losing_trades': trade_stats['losing_trades'] or 0,
                'win_rate': win_rate,
                'total_pnl': float(trade_stats['total_pnl'] or 0),
                'avg_win': float(trade_stats['avg_win'] or 0),
                'avg_loss': float(trade_stats['avg_loss'] or 0),
                'best_trade': float(trade_stats['best_trade'] or 0),
                'worst_trade': float(trade_stats['worst_trade'] or 0),
                'profit_factor': profit_factor,
                'daily_pnl': [{'date': row['trade_date'], 'pnl': float(row['daily_pnl'])} for row in daily_pnl]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting trading summary: {e}")
            return {}
    
    async def get_symbol_performance(self, symbol: str) -> Dict:
        """Get performance metrics for a specific symbol"""
        try:
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    stats = await conn.fetchrow('''
                        SELECT 
                            COUNT(*) as total_trades,
                            COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) as winning_trades,
                            SUM(realized_pnl) as total_pnl,
                            AVG(realized_pnl) as avg_pnl,
                            MAX(realized_pnl) as best_trade,
                            MIN(realized_pnl) as worst_trade,
                            AVG(confidence) as avg_confidence
                        FROM trades 
                        WHERE symbol = $1 AND status = 'FILLED' AND exit_price IS NOT NULL
                    ''', symbol)
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute('''
                        SELECT 
                            COUNT(*) as total_trades,
                            COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) as winning_trades,
                            SUM(realized_pnl) as total_pnl,
                            AVG(realized_pnl) as avg_pnl,
                            MAX(realized_pnl) as best_trade,
                            MIN(realized_pnl) as worst_trade,
                            AVG(confidence) as avg_confidence
                        FROM trades 
                        WHERE symbol = ? AND status = 'FILLED' AND exit_price IS NOT NULL
                    ''', (symbol,))
                    stats = cursor.fetchone()
            
            if stats and stats['total_trades'] > 0:
                return {
                    'symbol': symbol,
                    'total_trades': stats['total_trades'],
                    'winning_trades': stats['winning_trades'],
                    'win_rate': stats['winning_trades'] / stats['total_trades'],
                    'total_pnl': float(stats['total_pnl'] or 0),
                    'avg_pnl': float(stats['avg_pnl'] or 0),
                    'best_trade': float(stats['best_trade'] or 0),
                    'worst_trade': float(stats['worst_trade'] or 0),
                    'avg_confidence': float(stats['avg_confidence'] or 0)
                }
            else:
                return {'symbol': symbol, 'message': 'No trading history found'}
                
        except Exception as e:
            self.logger.error(f"Error getting symbol performance for {symbol}: {e}")
            return {}
    
    async def get_strategy_performance(self) -> List[Dict]:
        """Get performance metrics by strategy"""
        try:
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    rows = await conn.fetch('''
                        SELECT 
                            strategy,
                            COUNT(*) as total_trades,
                            COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) as winning_trades,
                            SUM(realized_pnl) as total_pnl,
                            AVG(realized_pnl) as avg_pnl,
                            AVG(confidence) as avg_confidence
                        FROM trades 
                        WHERE status = 'FILLED' AND exit_price IS NOT NULL
                        GROUP BY strategy
                        ORDER BY total_pnl DESC
                    ''')
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute('''
                        SELECT 
                            strategy,
                            COUNT(*) as total_trades,
                            COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) as winning_trades,
                            SUM(realized_pnl) as total_pnl,
                            AVG(realized_pnl) as avg_pnl,
                            AVG(confidence) as avg_confidence
                        FROM trades 
                        WHERE status = 'FILLED' AND exit_price IS NOT NULL
                        GROUP BY strategy
                        ORDER BY total_pnl DESC
                    ''')
                    rows = cursor.fetchall()
            
            result = []
            for row in rows:
                result.append({
                    'strategy': row['strategy'],
                    'total_trades': row['total_trades'],
                    'winning_trades': row['winning_trades'],
                    'win_rate': row['winning_trades'] / row['total_trades'] if row['total_trades'] > 0 else 0,
                    'total_pnl': float(row['total_pnl'] or 0),
                    'avg_pnl': float(row['avg_pnl'] or 0),
                    'avg_confidence': float(row['avg_confidence'] or 0)
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting strategy performance: {e}")
            return []
    
    # Data Export Methods
    async def export_trades_to_csv(self, filepath: str, days_back: int = 30):
        """Export trades to CSV file"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            trades = await self.get_trades_by_date_range(cutoff_date, datetime.now())
            
            if trades:
                df = pd.DataFrame(trades)
                df.to_csv(filepath, index=False)
                self.logger.info(f"Exported {len(trades)} trades to {filepath}")
            else:
                self.logger.warning("No trades to export")
                
        except Exception as e:
            self.logger.error(f"Error exporting trades to CSV: {e}")
    
    async def get_trades_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get trades within date range"""
        try:
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    rows = await conn.fetch(
                        "SELECT * FROM trades WHERE timestamp >= $1 AND timestamp <= $2 ORDER BY timestamp",
                        start_date, end_date
                    )
                    return [dict(row) for row in rows]
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        "SELECT * FROM trades WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
                        (start_date, end_date)
                    )
                    return [dict(row) for row in cursor.fetchall()]
                    
        except Exception as e:
            self.logger.error(f"Error getting trades by date range: {e}")
            return []
    
    # Database Maintenance Methods
    async def optimize_database(self):
        """Optimize database performance"""
        try:
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    await conn.execute("VACUUM ANALYZE")
                    self.logger.info("PostgreSQL database optimized")
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.execute("VACUUM")
                    conn.execute("ANALYZE")
                    conn.commit()
                    self.logger.info("SQLite database optimized")
                    
        except Exception as e:
            self.logger.error(f"Error optimizing database: {e}")
    
    async def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data beyond retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            tables_to_clean = ['market_data', 'system_logs', 'scan_results']
            
            for table in tables_to_clean:
                if self.db_type == 'postgresql':
                    async with self.get_connection() as conn:
                        result = await conn.execute(
                            f"DELETE FROM {table} WHERE timestamp < $1",
                            cutoff_date
                        )
                        deleted_count = result.split()[-1] if isinstance(result, str) else 0
                else:
                    with sqlite3.connect(self.sqlite_path) as conn:
                        cursor = conn.execute(
                            f"DELETE FROM {table} WHERE timestamp < ?",
                            (cutoff_date,)
                        )
                        deleted_count = cursor.rowcount
                        conn.commit()
                
                self.logger.info(f"Cleaned {deleted_count} old records from {table}")
            
            # Optimize after cleanup
            await self.optimize_database()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    async def backup_database(self, backup_path: str):
        """Create database backup"""
        try:
            if self.db_type == 'postgresql':
                # For PostgreSQL, we'd typically use pg_dump
                self.logger.warning("PostgreSQL backup requires pg_dump - implement externally")
            else:
                # For SQLite, copy the file
                import shutil
                shutil.copy2(self.sqlite_path, backup_path)
                self.logger.info(f"SQLite database backed up to {backup_path}")
                
        except Exception as e:
            self.logger.error(f"Error backing up database: {e}")
    
    async def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            stats = {
                'database_type': self.db_type,
                'connection_status': self.is_connected,
                'timestamp': datetime.now().isoformat()
            }
            
            # Get table row counts
            tables = ['trades', 'performance_metrics', 'market_data', 'scan_results', 'system_logs', 'risk_alerts']
            
            if self.db_type == 'postgresql':
                async with self.get_connection() as conn:
                    for table in tables:
                        try:
                            result = await conn.fetchrow(f"SELECT COUNT(*) as count FROM {table}")
                            stats[f'{table}_count'] = result['count']
                        except Exception as e:
                            stats[f'{table}_count'] = f'Error: {e}'
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    for table in tables:
                        try:
                            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                            stats[f'{table}_count'] = cursor.fetchone()[0]
                        except Exception as e:
                            stats[f'{table}_count'] = f'Error: {e}'
            
            # Get database size
            if self.db_type == 'sqlite':
                try:
                    stats['database_size_mb'] = os.path.getsize(self.sqlite_path) / (1024 * 1024)
                except:
                    stats['database_size_mb'] = 'Unknown'
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {'error': str(e)}
    
    # Cleanup and shutdown
    async def close(self):
        """Close database connections"""
        try:
            if self.pg_pool:
                await self.pg_pool.close()
                self.logger.info("PostgreSQL connection pool closed")
            
            self.is_connected = False
            self.logger.info("Database manager closed")
            
        except Exception as e:
            self.logger.error(f"Error closing database: {e}")

# Integration helper for bot_core.py
async def initialize_database_manager(config):
    """Initialize database manager"""
    db_manager = DatabaseManager(config)
    await db_manager.initialize()
    return db_manager

if __name__ == "__main__":
    # Test database setup
    async def test_database():
        config = {
            'database_type': 'sqlite',
            'database_config': {
                'sqlite_path': 'test_trading_bot.db'
            }
        }
        
        db_manager = DatabaseManager(config)
        await db_manager.initialize()
        
        # Test trade logging
        trade_data = {
            'symbol': 'RELIANCE',
            'action': 'BUY',
            'strategy': 'TEST_STRATEGY',
            'quantity': 100,
            'price': 2500.0,
            'confidence': 0.8,
            'mode': 'paper'
        }
        
        trade_id = await db_manager.log_trade(trade_data)
        print(f"Trade logged with ID: {trade_id}")
        
        # Test performance metrics
        metrics = {
            'total_trades': 1,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'portfolio_value': 100000.0
        }
        
        await db_manager.save_performance_metrics(metrics)
        print("Performance metrics saved")
        
        # Get stats
        stats = await db_manager.get_database_stats()
        print("Database Stats:", stats)
        
        await db_manager.close()
    
    # Run test
    asyncio.run(test_database())
