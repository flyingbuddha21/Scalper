#!/usr/bin/env python3
"""
Production-Grade Database Setup for Trading System with User Risk Management
PostgreSQL + SQLite hybrid architecture with real-time caching and user-specific features
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
import hashlib
import secrets

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
    user_id: str
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

@dataclass
class UserRiskConfig:
    """User risk configuration data structure"""
    user_id: str
    capital: float
    risk_per_trade_percent: float
    daily_loss_limit_percent: float
    max_concurrent_trades: int
    risk_reward_ratio: float
    max_position_size_percent: float
    stop_loss_percent: float
    take_profit_percent: float
    trading_start_time: str
    trading_end_time: str
    auto_square_off: bool
    paper_trading_mode: bool
    last_updated: datetime

class DatabaseManager:
    """
    Enhanced database manager with user risk management and multi-user support
    PostgreSQL primary + SQLite cache with user-specific features
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        self.db_config = self.config['database']
        
        # Database connections
        self.pg_pool = None
        self.sqlite_conn = None
        self.sqlite_path = self.db_config.get('sqlite_path', 'data/realtime_cache.db')
        
        # User management
        self.current_user_id = None
        
        # Initialize logger and error handler
        self.logger = Logger(__name__)
        self.error_handler = ErrorHandler()
        self.data_validator = DataValidator()
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
        
        self.logger.info("Enhanced Trading Database with User Management initialized")
    
    async def initialize(self):
        """Initialize database connections and create tables"""
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
            
            # Create all tables
            await self.create_tables()
            
            self.logger.info("Database connections established and tables created successfully")
            return True
            
        except Exception as e:
            error_msg = f"Database initialization failed: {str(e)}"
            self.logger.error(error_msg)
            self.error_handler.handle_error(e, "database_initialization")
            raise
    
    async def create_tables(self):
        """Create all required database tables including user management"""
        try:
            await self._create_user_management_tables()
            await self._create_trading_tables()
            await self._create_sqlite_tables()
            await self._create_indexes()
            self.logger.info("All database tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Table creation failed: {str(e)}")
            raise
    
    async def _create_user_management_tables(self):
        """Create user management and risk configuration tables"""
        async with self.pg_pool.acquire() as conn:
            
            # Users table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id VARCHAR(50) PRIMARY KEY,
                    username VARCHAR(100) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    is_active BOOLEAN DEFAULT TRUE,
                    subscription_type VARCHAR(20) DEFAULT 'basic',
                    last_login TIMESTAMP,
                    login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP
                )
            """)
            
            # User trading configuration (enhanced)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_trading_config (
                    user_id VARCHAR(50) PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
                    capital DECIMAL(15,2) NOT NULL DEFAULT 100000.00,
                    risk_per_trade_percent DECIMAL(5,2) NOT NULL DEFAULT 2.00,
                    daily_loss_limit_percent DECIMAL(5,2) NOT NULL DEFAULT 5.00,
                    max_concurrent_trades INTEGER NOT NULL DEFAULT 5,
                    risk_reward_ratio DECIMAL(5,2) NOT NULL DEFAULT 2.00,
                    max_position_size_percent DECIMAL(5,2) NOT NULL DEFAULT 20.00,
                    stop_loss_percent DECIMAL(5,2) NOT NULL DEFAULT 3.00,
                    take_profit_percent DECIMAL(5,2) NOT NULL DEFAULT 6.00,
                    trading_start_time TIME NOT NULL DEFAULT '09:15:00',
                    trading_end_time TIME NOT NULL DEFAULT '15:30:00',
                    auto_square_off BOOLEAN DEFAULT TRUE,
                    paper_trading_mode BOOLEAN DEFAULT TRUE,
                    goodwill_api_key VARCHAR(255),
                    goodwill_secret_key VARCHAR(255),
                    goodwill_user_id VARCHAR(100),
                    webhook_url VARCHAR(500),
                    telegram_chat_id VARCHAR(50),
                    email_notifications BOOLEAN DEFAULT TRUE,
                    sms_notifications BOOLEAN DEFAULT FALSE,
                    max_daily_trades INTEGER DEFAULT 50,
                    trailing_stop_loss BOOLEAN DEFAULT FALSE,
                    bracket_order_enabled BOOLEAN DEFAULT TRUE,
                    last_updated TIMESTAMP DEFAULT NOW(),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # User trading sessions
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_trading_sessions (
                    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(50) REFERENCES users(user_id) ON DELETE CASCADE,
                    session_start TIMESTAMP NOT NULL,
                    session_end TIMESTAMP,
                    trading_mode VARCHAR(10) NOT NULL, -- 'live' or 'paper'
                    initial_capital DECIMAL(15,2) NOT NULL,
                    final_capital DECIMAL(15,2),
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    max_drawdown DECIMAL(15,2) DEFAULT 0,
                    total_pnl DECIMAL(15,2) DEFAULT 0,
                    commission_paid DECIMAL(10,2) DEFAULT 0,
                    taxes_paid DECIMAL(10,2) DEFAULT 0,
                    risk_limits_breached INTEGER DEFAULT 0,
                    auto_stopped BOOLEAN DEFAULT FALSE,
                    stop_reason VARCHAR(255),
                    strategies_used TEXT[], -- Array of strategy names
                    symbols_traded TEXT[], -- Array of symbols
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Daily trading reports
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_trading_reports (
                    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(50) REFERENCES users(user_id) ON DELETE CASCADE,
                    report_date DATE NOT NULL,
                    daily_pnl DECIMAL(15,2) DEFAULT 0,
                    portfolio_value DECIMAL(15,2) DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    win_rate DECIMAL(5,2) DEFAULT 0,
                    avg_win DECIMAL(12,2) DEFAULT 0,
                    avg_loss DECIMAL(12,2) DEFAULT 0,
                    max_concurrent_positions INTEGER DEFAULT 0,
                    risk_limit_breaches INTEGER DEFAULT 0,
                    auto_square_off_used BOOLEAN DEFAULT FALSE,
                    trading_mode VARCHAR(10),
                    commission_paid DECIMAL(10,2) DEFAULT 0,
                    taxes_paid DECIMAL(10,2) DEFAULT 0,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(user_id, report_date)
                )
            """)
            
            # User risk alerts
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_risk_alerts (
                    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(50) REFERENCES users(user_id) ON DELETE CASCADE,
                    alert_type VARCHAR(50) NOT NULL, -- 'DAILY_LOSS', 'POSITION_SIZE', 'VOLATILITY', etc.
                    alert_level VARCHAR(20) NOT NULL, -- 'INFO', 'WARNING', 'CRITICAL'
                    message TEXT NOT NULL,
                    symbol VARCHAR(20),
                    current_value DECIMAL(15,2),
                    threshold_value DECIMAL(15,2),
                    is_acknowledged BOOLEAN DEFAULT FALSE,
                    acknowledged_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # User API credentials (encrypted)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_api_credentials (
                    credential_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(50) REFERENCES users(user_id) ON DELETE CASCADE,
                    broker_name VARCHAR(50) NOT NULL, -- 'goodwill', 'zerodha', etc.
                    api_key_encrypted TEXT NOT NULL,
                    secret_key_encrypted TEXT NOT NULL,
                    user_id_encrypted TEXT,
                    encryption_key_hash VARCHAR(64) NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    last_used TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(user_id, broker_name)
                )
            """)
    
    async def _create_trading_tables(self):
        """Create enhanced trading tables with user context"""
        async with self.pg_pool.acquire() as conn:
            
            # Stocks master table (unchanged)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS stocks (
                    symbol VARCHAR(20) PRIMARY KEY,
                    company_name VARCHAR(255),
                    sector VARCHAR(100),
                    market_cap BIGINT,
                    exchange VARCHAR(10) DEFAULT 'NSE',
                    isin VARCHAR(20),
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Enhanced trades table with user context
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(50) REFERENCES users(user_id) ON DELETE CASCADE,
                    session_id UUID REFERENCES user_trading_sessions(session_id),
                    symbol VARCHAR(20) REFERENCES stocks(symbol),
                    action VARCHAR(10) NOT NULL, -- BUY, SELL
                    strategy VARCHAR(50) NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price DECIMAL(12,4) NOT NULL,
                    exit_price DECIMAL(12,4),
                    stop_loss DECIMAL(12,4),
                    take_profit DECIMAL(12,4),
                    order_type VARCHAR(20) DEFAULT 'MARKET', -- MARKET, LIMIT, SL, SL-M
                    status VARCHAR(20) DEFAULT 'PENDING', -- PENDING, FILLED, PARTIAL, CANCELLED
                    entry_time TIMESTAMP DEFAULT NOW(),
                    exit_time TIMESTAMP,
                    holding_period INTERVAL,
                    realized_pnl DECIMAL(15,4) DEFAULT 0,
                    unrealized_pnl DECIMAL(15,4) DEFAULT 0,
                    commission DECIMAL(10,4) DEFAULT 0,
                    taxes DECIMAL(10,4) DEFAULT 0,
                    net_pnl DECIMAL(15,4) DEFAULT 0,
                    risk_reward_ratio DECIMAL(6,2),
                    trade_confidence DECIMAL(4,2), -- Strategy confidence 0-100
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    
                    CHECK (quantity > 0),
                    CHECK (entry_price > 0),
                    CHECK (action IN ('BUY', 'SELL')),
                    CHECK (status IN ('PENDING', 'FILLED', 'PARTIAL', 'CANCELLED', 'REJECTED'))
                )
            """)
            
            # User portfolio holdings
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_portfolio (
                    portfolio_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(50) REFERENCES users(user_id) ON DELETE CASCADE,
                    symbol VARCHAR(20) REFERENCES stocks(symbol),
                    quantity INTEGER NOT NULL,
                    avg_price DECIMAL(12,4) NOT NULL,
                    invested_amount DECIMAL(15,4) NOT NULL,
                    current_price DECIMAL(12,4),
                    current_value DECIMAL(15,4),
                    unrealized_pnl DECIMAL(15,4) DEFAULT 0,
                    unrealized_pnl_percent DECIMAL(8,4) DEFAULT 0,
                    day_pnl DECIMAL(15,4) DEFAULT 0,
                    day_pnl_percent DECIMAL(8,4) DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT NOW(),
                    created_at TIMESTAMP DEFAULT NOW(),
                    
                    UNIQUE(user_id, symbol),
                    CHECK (quantity != 0),
                    CHECK (avg_price > 0)
                )
            """)
            
            # User trade orders (enhanced)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_trade_orders (
                    order_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(50) REFERENCES users(user_id) ON DELETE CASCADE,
                    session_id UUID REFERENCES user_trading_sessions(session_id),
                    symbol VARCHAR(20) REFERENCES stocks(symbol),
                    side VARCHAR(10) NOT NULL CHECK (side IN ('BUY', 'SELL')),
                    order_type VARCHAR(20) DEFAULT 'MARKET',
                    quantity INTEGER NOT NULL,
                    price DECIMAL(12,4),
                    trigger_price DECIMAL(12,4),
                    executed_price DECIMAL(12,4),
                    executed_quantity INTEGER DEFAULT 0,
                    remaining_quantity INTEGER,
                    status VARCHAR(20) DEFAULT 'PENDING',
                    strategy VARCHAR(50),
                    parent_order_id UUID, -- For bracket orders
                    stop_loss DECIMAL(12,4),
                    take_profit DECIMAL(12,4),
                    trailing_stop_loss BOOLEAN DEFAULT FALSE,
                    validity VARCHAR(10) DEFAULT 'DAY', -- DAY, IOC, GTD
                    exchange VARCHAR(10) DEFAULT 'NSE',
                    broker_order_id VARCHAR(50), -- External broker order ID
                    rejection_reason TEXT,
                    commission DECIMAL(10,4) DEFAULT 0,
                    taxes DECIMAL(10,4) DEFAULT 0,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    executed_at TIMESTAMP,
                    cancelled_at TIMESTAMP,
                    
                    CHECK (quantity > 0),
                    CHECK (price IS NULL OR price > 0),
                    CHECK (status IN ('PENDING', 'PARTIAL', 'FILLED', 'CANCELLED', 'REJECTED'))
                )
            """)
            
            # Market data historical (unchanged but with indexes)
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
                    exchange VARCHAR(10) DEFAULT 'NSE',
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(symbol, timestamp, timeframe, exchange)
                )
            """)
            
            # User-specific strategy performance
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_strategy_performance (
                    performance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(50) REFERENCES users(user_id) ON DELETE CASCADE,
                    strategy_name VARCHAR(100) NOT NULL,
                    date DATE NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    profitable_trades INTEGER DEFAULT 0,
                    total_pnl DECIMAL(15,4) DEFAULT 0,
                    max_drawdown DECIMAL(10,4) DEFAULT 0,
                    sharpe_ratio DECIMAL(8,4),
                    win_rate DECIMAL(5,2),
                    avg_trade_pnl DECIMAL(12,4),
                    avg_holding_period INTERVAL,
                    max_concurrent_trades INTEGER DEFAULT 0,
                    risk_adjusted_return DECIMAL(8,4),
                    created_at TIMESTAMP DEFAULT NOW(),
                    
                    UNIQUE(user_id, strategy_name, date)
                )
            """)
            
            # User risk metrics
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_risk_metrics (
                    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(50) REFERENCES users(user_id) ON DELETE CASCADE,
                    date DATE NOT NULL,
                    portfolio_value DECIMAL(15,2) NOT NULL,
                    daily_var_95 DECIMAL(12,4), -- Value at Risk 95%
                    daily_var_99 DECIMAL(12,4), -- Value at Risk 99%
                    max_drawdown DECIMAL(10,4),
                    portfolio_beta DECIMAL(6,4),
                    sharpe_ratio DECIMAL(8,4),
                    sortino_ratio DECIMAL(8,4),
                    calmar_ratio DECIMAL(8,4),
                    total_exposure DECIMAL(15,2),
                    leverage_ratio DECIMAL(6,2),
                    concentration_risk DECIMAL(5,2), -- % in largest position
                    sector_concentration JSONB, -- Sector-wise breakdown
                    volatility_7d DECIMAL(8,4),
                    volatility_30d DECIMAL(8,4),
                    risk_score INTEGER CHECK (risk_score BETWEEN 1 AND 10),
                    created_at TIMESTAMP DEFAULT NOW(),
                    
                    UNIQUE(user_id, date)
                )
            """)
            
            # User notifications
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_notifications (
                    notification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(50) REFERENCES users(user_id) ON DELETE CASCADE,
                    type VARCHAR(50) NOT NULL, -- 'TRADE', 'RISK', 'SYSTEM', 'ALERT'
                    title VARCHAR(255) NOT NULL,
                    message TEXT NOT NULL,
                    priority VARCHAR(20) DEFAULT 'MEDIUM', -- LOW, MEDIUM, HIGH, CRITICAL
                    is_read BOOLEAN DEFAULT FALSE,
                    is_sent BOOLEAN DEFAULT FALSE,
                    delivery_method VARCHAR(20), -- 'EMAIL', 'SMS', 'WEBHOOK', 'TELEGRAM'
                    related_trade_id UUID REFERENCES trades(trade_id),
                    related_symbol VARCHAR(20),
                    scheduled_for TIMESTAMP,
                    sent_at TIMESTAMP,
                    read_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
    
    async def _create_sqlite_tables(self):
        """Create SQLite tables for real-time caching with user context"""
        cursor = self.sqlite_conn.cursor()
        
        try:
            # Real-time market data cache (unchanged)
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
            
            # User-specific portfolio cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_portfolio_cache (
                    cache_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    avg_price REAL NOT NULL,
                    current_price REAL,
                    unrealized_pnl REAL,
                    day_pnl REAL,
                    last_updated INTEGER NOT NULL,
                    UNIQUE(user_id, symbol)
                )
            """)
            
            # User active orders cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_active_orders (
                    order_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
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
            
            # User strategy signals cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_strategy_signals (
                    signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    strength REAL,
                    price REAL NOT NULL,
                    timestamp INTEGER NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    confidence REAL DEFAULT 0.5
                )
            """)
            
            # User performance metrics cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_performance_cache (
                    cache_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    last_updated INTEGER NOT NULL,
                    UNIQUE(user_id, metric_name)
                )
            """)
            
            # User risk alerts cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_risk_alerts_cache (
                    alert_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    alert_level TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    is_acknowledged BOOLEAN DEFAULT 0
                )
            """)
            
            # User session cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_session_cache (
                    user_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    trading_mode TEXT NOT NULL,
                    daily_pnl REAL DEFAULT 0,
                    daily_trades INTEGER DEFAULT 0,
                    risk_status TEXT DEFAULT 'ACTIVE',
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
            # User management indexes
            user_indexes = [
                "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
                "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)",
                "CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active)",
                "CREATE INDEX IF NOT EXISTS idx_user_config_user_id ON user_trading_config(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_trading_sessions_user_id ON user_trading_sessions(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_trading_sessions_start ON user_trading_sessions(session_start DESC)",
                "CREATE INDEX IF NOT EXISTS idx_daily_reports_user_date ON daily_trading_reports(user_id, report_date DESC)",
                "CREATE INDEX IF NOT EXISTS idx_risk_alerts_user_type ON user_risk_alerts(user_id, alert_type)",
                "CREATE INDEX IF NOT EXISTS idx_risk_alerts_level ON user_risk_alerts(alert_level, created_at DESC)"
            ]
            
            # Trading indexes
            trading_indexes = [
                "CREATE INDEX IF NOT EXISTS idx_trades_user_id ON trades(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_trades_symbol_user ON trades(symbol, user_id)",
                "CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time DESC)",
                "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)",
                "CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy)",
                "CREATE INDEX IF NOT EXISTS idx_user_portfolio_user_id ON user_portfolio(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_user_portfolio_symbol ON user_portfolio(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_user_orders_user_id ON user_trade_orders(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_user_orders_symbol ON user_trade_orders(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_user_orders_status ON user_trade_orders(status)",
                "CREATE INDEX IF NOT EXISTS idx_user_orders_created ON user_trade_orders(created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_strategy_perf_user_date ON user_strategy_performance(user_id, date DESC)",
                "CREATE INDEX IF NOT EXISTS idx_risk_metrics_user_date ON user_risk_metrics(user_id, date DESC)",
                "CREATE INDEX IF NOT EXISTS idx_notifications_user_type ON user_notifications(user_id, type)",
                "CREATE INDEX IF NOT EXISTS idx_notifications_unread ON user_notifications(user_id, is_read) WHERE is_read = FALSE"
            ]
            
            # Market data indexes (unchanged)
            market_indexes = [
                "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data_history(symbol, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_market_data_timeframe ON market_data_history(timeframe, timestamp DESC)"
            ]
            
            all_indexes = user_indexes + trading_indexes + market_indexes
            
            for index_sql in all_indexes:
                await conn.execute(index_sql)
        
        # SQLite indexes
        cursor = self.sqlite_conn.cursor()
        sqlite_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_realtime_quotes_updated ON realtime_quotes(last_updated DESC)",
            "CREATE INDEX IF NOT EXISTS idx_user_portfolio_cache_user ON user_portfolio_cache(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_active_orders_user ON user_active_orders(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_signals_user_symbol ON user_strategy_signals(user_id, symbol)",
            "CREATE INDEX IF NOT EXISTS idx_user_signals_timestamp ON user_strategy_signals(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_user_performance_user ON user_performance_cache(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_alerts_user ON user_risk_alerts_cache(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_session_user ON user_session_cache(user_id)"
        ]
        
        for index_sql in sqlite_indexes:
            cursor.execute(index_sql)
        
        self.sqlite_conn.commit()
    
    # USER MANAGEMENT METHODS
    
    async def create_user(self, username: str, email: str, password: str, 
                         subscription_type: str = 'basic') -> str:
        """Create a new user account"""
        try:
            # Generate user ID
            user_id = f"user_{secrets.token_hex(8)}"
            
            # Hash password
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO users (user_id, username, email, password_hash, subscription_type)
                    VALUES ($1, $2, $3, $4, $5)
                """, user_id, username, email, password_hash, subscription_type)
                
                # Create default trading configuration
                await conn.execute("""
                    INSERT INTO user_trading_config (user_id) VALUES ($1)
                """, user_id)
            
            self.logger.info(f"User created successfully: {user_id}")
            return user_id
            
        except Exception as e:
            self.logger.error(f"Failed to create user: {e}")
            raise
    
    async def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return user_id if successful"""
        try:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            async with self.pg_pool.acquire() as conn:
                user = await conn.fetchrow("""
                    SELECT user_id, is_active, locked_until, login_attempts
                    FROM users 
                    WHERE (username = $1 OR email = $1) AND password_hash = $2
                """, username, password_hash)
                
                if user:
                    # Check if account is locked
                    if user['locked_until'] and datetime.now() < user['locked_until']:
                        raise Exception("Account is temporarily locked")
                    
                    if not user['is_active']:
                        raise Exception("Account is deactivated")
                    
                    # Reset login attempts on successful login
                    await conn.execute("""
                        UPDATE users 
                        SET login_attempts = 0, last_login = NOW(), locked_until = NULL
                        WHERE user_id = $1
                    """, user['user_id'])
                    
                    self.current_user_id = user['user_id']
                    return user['user_id']
                else:
                    # Increment login attempts for username/email
                    await conn.execute("""
                        UPDATE users 
                        SET login_attempts = login_attempts + 1,
                            locked_until = CASE 
                                WHEN login_attempts >= 4 THEN NOW() + INTERVAL '30 minutes'
                                ELSE locked_until 
                            END
                        WHERE username = $1 OR email = $1
                    """, username)
                    
                    return None
                    
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            return None
    
    async def get_user_config(self, user_id: str) -> Optional[Dict]:
        """Get user trading configuration"""
        try:
            async with self.pg_pool.acquire() as conn:
                config = await conn.fetchrow("""
                    SELECT * FROM user_trading_config WHERE user_id = $1
                """, user_id)
                
                if config:
                    return {
                        'user_id': config['user_id'],
                        'capital': float(config['capital']),
                        'risk_per_trade_percent': float(config['risk_per_trade_percent']),
                        'daily_loss_limit_percent': float(config['daily_loss_limit_percent']),
                        'max_concurrent_trades': config['max_concurrent_trades'],
                        'risk_reward_ratio': float(config['risk_reward_ratio']),
                        'max_position_size_percent': float(config['max_position_size_percent']),
                        'stop_loss_percent': float(config['stop_loss_percent']),
                        'take_profit_percent': float(config['take_profit_percent']),
                        'trading_start_time': config['trading_start_time'].strftime('%H:%M'),
                        'trading_end_time': config['trading_end_time'].strftime('%H:%M'),
                        'auto_square_off': config['auto_square_off'],
                        'paper_trading_mode': config['paper_trading_mode'],
                        'max_daily_trades': config['max_daily_trades'],
                        'trailing_stop_loss': config['trailing_stop_loss'],
                        'bracket_order_enabled': config['bracket_order_enabled'],
                        'email_notifications': config['email_notifications'],
                        'sms_notifications': config['sms_notifications'],
                        'last_updated': config['last_updated']
                    }
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get user config: {e}")
            return None
    
    async def update_user_config(self, user_id: str, config_data: Dict) -> bool:
        """Update user trading configuration"""
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE user_trading_config SET
                        capital = $2,
                        risk_per_trade_percent = $3,
                        daily_loss_limit_percent = $4,
                        max_concurrent_trades = $5,
                        risk_reward_ratio = $6,
                        max_position_size_percent = $7,
                        stop_loss_percent = $8,
                        take_profit_percent = $9,
                        trading_start_time = $10,
                        trading_end_time = $11,
                        auto_square_off = $12,
                        paper_trading_mode = $13,
                        max_daily_trades = $14,
                        trailing_stop_loss = $15,
                        bracket_order_enabled = $16,
                        email_notifications = $17,
                        sms_notifications = $18,
                        last_updated = NOW()
                    WHERE user_id = $1
                """, 
                user_id,
                config_data.get('capital'),
                config_data.get('risk_per_trade_percent'),
                config_data.get('daily_loss_limit_percent'),
                config_data.get('max_concurrent_trades'),
                config_data.get('risk_reward_ratio'),
                config_data.get('max_position_size_percent'),
                config_data.get('stop_loss_percent'),
                config_data.get('take_profit_percent'),
                datetime.strptime(config_data.get('trading_start_time'), '%H:%M').time(),
                datetime.strptime(config_data.get('trading_end_time'), '%H:%M').time(),
                config_data.get('auto_square_off'),
                config_data.get('paper_trading_mode'),
                config_data.get('max_daily_trades', 50),
                config_data.get('trailing_stop_loss', False),
                config_data.get('bracket_order_enabled', True),
                config_data.get('email_notifications', True),
                config_data.get('sms_notifications', False))
            
            self.logger.info(f"User config updated for {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update user config: {e}")
            return False
    
    async def start_trading_session(self, user_id: str, trading_mode: str, 
                                  initial_capital: float) -> str:
        """Start a new trading session for user"""
        try:
            async with self.pg_pool.acquire() as conn:
                session_id = await conn.fetchval("""
                    INSERT INTO user_trading_sessions 
                    (user_id, session_start, trading_mode, initial_capital)
                    VALUES ($1, NOW(), $2, $3)
                    RETURNING session_id
                """, user_id, trading_mode, initial_capital)
                
                # Update session cache
                cursor = self.sqlite_conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO user_session_cache 
                    (user_id, session_id, trading_mode, last_updated)
                    VALUES (?, ?, ?, ?)
                """, (user_id, str(session_id), trading_mode, int(datetime.now().timestamp())))
                
                self.sqlite_conn.commit()
                
                self.logger.info(f"Trading session started for user {user_id}: {session_id}")
                return str(session_id)
                
        except Exception as e:
            self.logger.error(f"Failed to start trading session: {e}")
            raise
    
    async def end_trading_session(self, user_id: str, session_id: str, 
                                final_capital: float, session_stats: Dict) -> bool:
        """End trading session and save statistics"""
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE user_trading_sessions SET
                        session_end = NOW(),
                        final_capital = $3,
                        total_trades = $4,
                        winning_trades = $5,
                        losing_trades = $6,
                        total_pnl = $7,
                        max_drawdown = $8,
                        commission_paid = $9,
                        taxes_paid = $10,
                        risk_limits_breached = $11,
                        auto_stopped = $12,
                        stop_reason = $13,
                        strategies_used = $14,
                        symbols_traded = $15
                    WHERE user_id = $1 AND session_id = $2
                """, 
                user_id, session_id, final_capital,
                session_stats.get('total_trades', 0),
                session_stats.get('winning_trades', 0),
                session_stats.get('losing_trades', 0),
                session_stats.get('total_pnl', 0),
                session_stats.get('max_drawdown', 0),
                session_stats.get('commission_paid', 0),
                session_stats.get('taxes_paid', 0),
                session_stats.get('risk_limits_breached', 0),
                session_stats.get('auto_stopped', False),
                session_stats.get('stop_reason'),
                session_stats.get('strategies_used', []),
                session_stats.get('symbols_traded', []))
            
            # Remove from session cache
            cursor = self.sqlite_conn.cursor()
            cursor.execute("DELETE FROM user_session_cache WHERE user_id = ?", (user_id,))
            self.sqlite_conn.commit()
            
            self.logger.info(f"Trading session ended for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to end trading session: {e}")
            return False
    
    # USER TRADING OPERATIONS
    
    async def log_user_trade(self, user_id: str, trade_data: Dict) -> str:
        """Log a trade for specific user"""
        try:
            async with self.pg_pool.acquire() as conn:
                trade_id = await conn.fetchval("""
                    INSERT INTO trades (
                        user_id, session_id, symbol, action, strategy, quantity,
                        entry_price, stop_loss, take_profit, order_type, status,
                        trade_confidence, notes
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    RETURNING trade_id
                """, 
                user_id,
                trade_data.get('session_id'),
                trade_data['symbol'],
                trade_data['action'],
                trade_data['strategy'],
                trade_data['quantity'],
                trade_data['entry_price'],
                trade_data.get('stop_loss'),
                trade_data.get('take_profit'),
                trade_data.get('order_type', 'MARKET'),
                trade_data.get('status', 'PENDING'),
                trade_data.get('confidence', 0.5),
                trade_data.get('notes'))
            
            self.logger.info(f"Trade logged for user {user_id}: {trade_id}")
            return str(trade_id)
            
        except Exception as e:
            self.logger.error(f"Failed to log user trade: {e}")
            raise
    
    async def update_trade_exit(self, trade_id: str, exit_price: float, 
                              exit_time: datetime, realized_pnl: float) -> bool:
        """Update trade with exit information"""
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE trades SET
                        exit_price = $2,
                        exit_time = $3,
                        realized_pnl = $4,
                        status = 'FILLED',
                        holding_period = $3 - entry_time,
                        updated_at = NOW()
                    WHERE trade_id = $1
                """, trade_id, exit_price, exit_time, realized_pnl)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update trade exit: {e}")
            return False
    
    async def get_user_trades(self, user_id: str, start_date: datetime = None, 
                            end_date: datetime = None, limit: int = 100) -> List[Dict]:
        """Get trades for specific user"""
        try:
            async with self.pg_pool.acquire() as conn:
                query = """
                    SELECT * FROM trades WHERE user_id = $1
                """
                params = [user_id]
                param_count = 1
                
                if start_date:
                    param_count += 1
                    query += f" AND entry_time >= ${param_count}"
                    params.append(start_date)
                
                if end_date:
                    param_count += 1
                    query += f" AND entry_time <= ${param_count}"
                    params.append(end_date)
                
                query += f" ORDER BY entry_time DESC LIMIT ${param_count + 1}"
                params.append(limit)
                
                trades = await conn.fetch(query, *params)
                
                return [
                    {
                        'trade_id': str(trade['trade_id']),
                        'symbol': trade['symbol'],
                        'action': trade['action'],
                        'strategy': trade['strategy'],
                        'quantity': trade['quantity'],
                        'entry_price': float(trade['entry_price']),
                        'exit_price': float(trade['exit_price']) if trade['exit_price'] else None,
                        'stop_loss': float(trade['stop_loss']) if trade['stop_loss'] else None,
                        'take_profit': float(trade['take_profit']) if trade['take_profit'] else None,
                        'status': trade['status'],
                        'entry_time': trade['entry_time'],
                        'exit_time': trade['exit_time'],
                        'holding_period': str(trade['holding_period']) if trade['holding_period'] else None,
                        'realized_pnl': float(trade['realized_pnl']) if trade['realized_pnl'] else 0,
                        'commission': float(trade['commission']) if trade['commission'] else 0,
                        'net_pnl': float(trade['net_pnl']) if trade['net_pnl'] else 0,
                        'trade_confidence': float(trade['trade_confidence']) if trade['trade_confidence'] else 0,
                        'notes': trade['notes']
                    }
                    for trade in trades
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to get user trades: {e}")
            return []
    
    async def get_user_portfolio(self, user_id: str) -> List[Dict]:
        """Get user's current portfolio"""
        try:
            async with self.pg_pool.acquire() as conn:
                positions = await conn.fetch("""
                    SELECT up.*, s.company_name 
                    FROM user_portfolio up
                    JOIN stocks s ON up.symbol = s.symbol
                    WHERE up.user_id = $1 AND up.quantity != 0
                    ORDER BY up.current_value DESC
                """, user_id)
                
                return [
                    {
                        'symbol': pos['symbol'],
                        'company_name': pos['company_name'],
                        'quantity': pos['quantity'],
                        'avg_price': float(pos['avg_price']),
                        'invested_amount': float(pos['invested_amount']),
                        'current_price': float(pos['current_price']) if pos['current_price'] else 0,
                        'current_value': float(pos['current_value']) if pos['current_value'] else 0,
                        'unrealized_pnl': float(pos['unrealized_pnl']),
                        'unrealized_pnl_percent': float(pos['unrealized_pnl_percent']),
                        'day_pnl': float(pos['day_pnl']),
                        'day_pnl_percent': float(pos['day_pnl_percent']),
                        'last_updated': pos['last_updated']
                    }
                    for pos in positions
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to get user portfolio: {e}")
            return []
    
    async def update_user_portfolio(self, user_id: str, symbol: str, 
                                  quantity_change: int, price: float) -> bool:
        """Update user portfolio position"""
        try:
            async with self.pg_pool.acquire() as conn:
                # Get existing position
                existing = await conn.fetchrow("""
                    SELECT quantity, avg_price, invested_amount 
                    FROM user_portfolio 
                    WHERE user_id = $1 AND symbol = $2
                """, user_id, symbol)
                
                if existing:
                    old_qty = existing['quantity']
                    old_avg = float(existing['avg_price'])
                    old_invested = float(existing['invested_amount'])
                    
                    new_qty = old_qty + quantity_change
                    trade_value = quantity_change * price
                    
                    if new_qty == 0:
                        # Position closed
                        await conn.execute("""
                            DELETE FROM user_portfolio 
                            WHERE user_id = $1 AND symbol = $2
                        """, user_id, symbol)
                    else:
                        # Calculate new average price
                        if quantity_change > 0:  # Buy
                            new_invested = old_invested + trade_value
                            new_avg = new_invested / new_qty
                        else:  # Sell
                            new_invested = old_invested
                            new_avg = old_avg  # Keep same average for sells
                        
                        await conn.execute("""
                            UPDATE user_portfolio SET
                                quantity = $3,
                                avg_price = $4,
                                invested_amount = $5,
                                last_updated = NOW()
                            WHERE user_id = $1 AND symbol = $2
                        """, user_id, symbol, new_qty, new_avg, new_invested)
                else:
                    # New position
                    if quantity_change > 0:
                        invested_amount = quantity_change * price
                        await conn.execute("""
                            INSERT INTO user_portfolio 
                            (user_id, symbol, quantity, avg_price, invested_amount)
                            VALUES ($1, $2, $3, $4, $5)
                        """, user_id, symbol, quantity_change, price, invested_amount)
                
                # Update cache
                await self._update_user_portfolio_cache(user_id, symbol)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update user portfolio: {e}")
            return False
    
    async def _update_user_portfolio_cache(self, user_id: str, symbol: str):
        """Update user portfolio cache"""
        try:
            async with self.pg_pool.acquire() as conn:
                position = await conn.fetchrow("""
                    SELECT * FROM user_portfolio 
                    WHERE user_id = $1 AND symbol = $2
                """, user_id, symbol)
                
                cursor = self.sqlite_conn.cursor()
                
                if position:
                    current_price = await self.get_latest_price(symbol)
                    if current_price:
                        unrealized_pnl = (current_price - float(position['avg_price'])) * position['quantity']
                        
                        cursor.execute("""
                            INSERT OR REPLACE INTO user_portfolio_cache 
                            (cache_id, user_id, symbol, quantity, avg_price, current_price, 
                             unrealized_pnl, last_updated)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (f"{user_id}_{symbol}", user_id, symbol, position['quantity'],
                              float(position['avg_price']), current_price, unrealized_pnl,
                              int(datetime.now().timestamp())))
                else:
                    # Remove from cache if position deleted
                    cursor.execute("""
                        DELETE FROM user_portfolio_cache 
                        WHERE user_id = ? AND symbol = ?
                    """, (user_id, symbol))
                
                self.sqlite_conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to update portfolio cache: {e}")
    
    # USER PERFORMANCE AND ANALYTICS
    
    async def calculate_user_daily_pnl(self, user_id: str, date: datetime = None) -> Dict:
        """Calculate daily P&L for specific user"""
        try:
            if not date:
                date = datetime.now().date()
            
            async with self.pg_pool.acquire() as conn:
                # Get realized P&L from trades
                realized_pnl = await conn.fetchval("""
                    SELECT COALESCE(SUM(realized_pnl - commission - taxes), 0)
                    FROM trades 
                    WHERE user_id = $1 AND DATE(entry_time) = $2 AND status = 'FILLED'
                """, user_id, date) or 0
                
                # Get unrealized P&L from current positions
                unrealized_pnl = await conn.fetchval("""
                    SELECT COALESCE(SUM(unrealized_pnl), 0)
                    FROM user_portfolio 
                    WHERE user_id = $1
                """, user_id) or 0
                
                # Get trade statistics
                trade_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_trades,
                        COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) as winning_trades,
                        COUNT(CASE WHEN realized_pnl <= 0 THEN 1 END) as losing_trades,
                        COALESCE(SUM(commission + taxes), 0) as total_charges
                    FROM trades 
                    WHERE user_id = $1 AND DATE(entry_time) = $2 AND status = 'FILLED'
                """, user_id, date)
                
                return {
                    'user_id': user_id,
                    'date': date,
                    'realized_pnl': float(realized_pnl),
                    'unrealized_pnl': float(unrealized_pnl),
                    'total_pnl': float(realized_pnl) + float(unrealized_pnl),
                    'total_trades': trade_stats['total_trades'],
                    'winning_trades': trade_stats['winning_trades'],
                    'losing_trades': trade_stats['losing_trades'],
                    'win_rate': (trade_stats['winning_trades'] / trade_stats['total_trades'] * 100) if trade_stats['total_trades'] > 0 else 0,
                    'total_charges': float(trade_stats['total_charges'])
                }
                
        except Exception as e:
            self.logger.error(f"Failed to calculate user daily P&L: {e}")
            return {}
    
    async def save_daily_report(self, user_id: str, report_data: Dict) -> bool:
        """Save daily trading report for user"""
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO daily_trading_reports (
                        user_id, report_date, daily_pnl, portfolio_value, total_trades,
                        winning_trades, losing_trades, win_rate, avg_win, avg_loss,
                        max_concurrent_positions, risk_limit_breaches, auto_square_off_used,
                        trading_mode, commission_paid, taxes_paid
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    ON CONFLICT (user_id, report_date) DO UPDATE SET
                        daily_pnl = EXCLUDED.daily_pnl,
                        portfolio_value = EXCLUDED.portfolio_value,
                        total_trades = EXCLUDED.total_trades,
                        winning_trades = EXCLUDED.winning_trades,
                        losing_trades = EXCLUDED.losing_trades,
                        win_rate = EXCLUDED.win_rate,
                        avg_win = EXCLUDED.avg_win,
                        avg_loss = EXCLUDED.avg_loss,
                        max_concurrent_positions = EXCLUDED.max_concurrent_positions,
                        risk_limit_breaches = EXCLUDED.risk_limit_breaches,
                        auto_square_off_used = EXCLUDED.auto_square_off_used,
                        trading_mode = EXCLUDED.trading_mode,
                        commission_paid = EXCLUDED.commission_paid,
                        taxes_paid = EXCLUDED.taxes_paid,
                        updated_at = NOW()
                """, 
                user_id, report_data['date'], report_data['daily_pnl'],
                report_data['portfolio_value'], report_data['total_trades'],
                report_data['winning_trades'], report_data['losing_trades'],
                report_data['win_rate'], report_data['avg_win'], report_data['avg_loss'],
                report_data['max_concurrent_positions'], report_data['risk_limit_breaches'],
                report_data['auto_square_off_used'], report_data['trading_mode'],
                report_data['commission_paid'], report_data['taxes_paid'])
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save daily report: {e}")
            return False
    
    async def get_user_daily_report(self, user_id: str, date: str = None) -> Optional[Dict]:
        """Get daily report for user"""
        try:
            report_date = date or datetime.now().date().isoformat()
            
            async with self.pg_pool.acquire() as conn:
                report = await conn.fetchrow("""
                    SELECT * FROM daily_trading_reports 
                    WHERE user_id = $1 AND report_date = $2
                """, user_id, report_date)
                
                if report:
                    return {
                        'user_id': report['user_id'],
                        'report_date': report['report_date'].isoformat(),
                        'daily_pnl': float(report['daily_pnl']),
                        'portfolio_value': float(report['portfolio_value']),
                        'total_trades': report['total_trades'],
                        'winning_trades': report['winning_trades'],
                        'losing_trades': report['losing_trades'],
                        'win_rate': float(report['win_rate']),
                        'avg_win': float(report['avg_win']),
                        'avg_loss': float(report['avg_loss']),
                        'max_concurrent_positions': report['max_concurrent_positions'],
                        'risk_limit_breaches': report['risk_limit_breaches'],
                        'auto_square_off_used': report['auto_square_off_used'],
                        'trading_mode': report['trading_mode'],
                        'commission_paid': float(report['commission_paid']),
                        'taxes_paid': float(report['taxes_paid']),
                        'created_at': report['created_at'],
                        'updated_at': report['updated_at']
                    }
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get daily report: {e}")
            return None
    
    # RISK MANAGEMENT METHODS
    
    async def create_risk_alert(self, user_id: str, alert_type: str, alert_level: str,
                              message: str, symbol: str = None, current_value: float = None,
                              threshold_value: float = None) -> str:
        """Create risk alert for user"""
        try:
            async with self.pg_pool.acquire() as conn:
                alert_id = await conn.fetchval("""
                    INSERT INTO user_risk_alerts 
                    (user_id, alert_type, alert_level, message, symbol, current_value, threshold_value)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING alert_id
                """, user_id, alert_type, alert_level, message, symbol, current_value, threshold_value)
                
                # Cache critical alerts
                if alert_level in ['CRITICAL', 'WARNING']:
                    cursor = self.sqlite_conn.cursor()
                    cursor.execute("""
                        INSERT INTO user_risk_alerts_cache 
                        (alert_id, user_id, alert_type, message, alert_level, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (str(alert_id), user_id, alert_type, message, alert_level,
                          int(datetime.now().timestamp())))
                    self.sqlite_conn.commit()
                
                return str(alert_id)
                
        except Exception as e:
            self.logger.error(f"Failed to create risk alert: {e}")
            raise
    
    async def get_user_risk_alerts(self, user_id: str, unacknowledged_only: bool = True) -> List[Dict]:
        """Get risk alerts for user"""
        try:
            async with self.pg_pool.acquire() as conn:
                query = """
                    SELECT * FROM user_risk_alerts 
                    WHERE user_id = $1
                """
                params = [user_id]
                
                if unacknowledged_only:
                    query += " AND is_acknowledged = FALSE"
                
                query += " ORDER BY created_at DESC LIMIT 50"
                
                alerts = await conn.fetch(query, *params)
                
                return [
                    {
                        'alert_id': str(alert['alert_id']),
                        'alert_type': alert['alert_type'],
                        'alert_level': alert['alert_level'],
                        'message': alert['message'],
                        'symbol': alert['symbol'],
                        'current_value': float(alert['current_value']) if alert['current_value'] else None,
                        'threshold_value': float(alert['threshold_value']) if alert['threshold_value'] else None,
                        'is_acknowledged': alert['is_acknowledged'],
                        'created_at': alert['created_at']
                    }
                    for alert in alerts
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to get risk alerts: {e}")
            return []
    
    async def acknowledge_risk_alert(self, alert_id: str) -> bool:
        """Acknowledge a risk alert"""
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE user_risk_alerts 
                    SET is_acknowledged = TRUE, acknowledged_at = NOW()
                    WHERE alert_id = $1
                """, alert_id)
                
                # Remove from cache
                cursor = self.sqlite_conn.cursor()
                cursor.execute("""
                    UPDATE user_risk_alerts_cache 
                    SET is_acknowledged = 1 
                    WHERE alert_id = ?
                """, (alert_id,))
                self.sqlite_conn.commit()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to acknowledge alert: {e}")
            return False
    
    # NOTIFICATION METHODS
    
    async def create_notification(self, user_id: str, notification_type: str, title: str,
                                message: str, priority: str = 'MEDIUM', 
                                delivery_method: str = 'EMAIL') -> str:
        """Create notification for user"""
        try:
            async with self.pg_pool.acquire() as conn:
                notification_id = await conn.fetchval("""
                    INSERT INTO user_notifications 
                    (user_id, type, title, message, priority, delivery_method)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING notification_id
                """, user_id, notification_type, title, message, priority, delivery_method)
                
                return str(notification_id)
                
        except Exception as e:
            self.logger.error(f"Failed to create notification: {e}")
            raise
    
    async def get_user_notifications(self, user_id: str, unread_only: bool = False, 
                                   limit: int = 50) -> List[Dict]:
        """Get notifications for user"""
        try:
            async with self.pg_pool.acquire() as conn:
                query = """
                    SELECT * FROM user_notifications 
                    WHERE user_id = $1
                """
                params = [user_id]
                
                if unread_only:
                    query += " AND is_read = FALSE"
                
                query += f" ORDER BY created_at DESC LIMIT ${len(params) + 1}"
                params.append(limit)
                
                notifications = await conn.fetch(query, *params)
                
                return [
                    {
                        'notification_id': str(notif['notification_id']),
                        'type': notif['type'],
                        'title': notif['title'],
                        'message': notif['message'],
                        'priority': notif['priority'],
                        'is_read': notif['is_read'],
                        'delivery_method': notif['delivery_method'],
                        'created_at': notif['created_at'],
                        'read_at': notif['read_at']
                    }
                    for notif in notifications
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to get notifications: {e}")
            return []
    
    async def mark_notification_read(self, notification_id: str) -> bool:
        """Mark notification as read"""
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE user_notifications 
                    SET is_read = TRUE, read_at = NOW()
                    WHERE notification_id = $1
                """, notification_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to mark notification as read: {e}")
            return False
    
    # STRATEGY PERFORMANCE METHODS
    
    async def update_user_strategy_performance(self, user_id: str, strategy_name: str,
                                             date: datetime, performance_data: Dict) -> bool:
        """Update strategy performance for user"""
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO user_strategy_performance (
                        user_id, strategy_name, date, total_trades, profitable_trades,
                        total_pnl, max_drawdown, sharpe_ratio, win_rate, avg_trade_pnl,
                        avg_holding_period, max_concurrent_trades, risk_adjusted_return
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (user_id, strategy_name, date) DO UPDATE SET
                        total_trades = EXCLUDED.total_trades,
                        profitable_trades = EXCLUDED.profitable_trades,
                        total_pnl = EXCLUDED.total_pnl,
                        max_drawdown = EXCLUDED.max_drawdown,
                        sharpe_ratio = EXCLUDED.sharpe_ratio,
                        win_rate = EXCLUDED.win_rate,
                        avg_trade_pnl = EXCLUDED.avg_trade_pnl,
                        avg_holding_period = EXCLUDED.avg_holding_period,
                        max_concurrent_trades = EXCLUDED.max_concurrent_trades,
                        risk_adjusted_return = EXCLUDED.risk_adjusted_return
                """, 
                user_id, strategy_name, date.date(),
                performance_data['total_trades'],
                performance_data['profitable_trades'],
                performance_data['total_pnl'],
                performance_data.get('max_drawdown'),
                performance_data.get('sharpe_ratio'),
                performance_data['win_rate'],
                performance_data['avg_trade_pnl'],
                performance_data.get('avg_holding_period'),
                performance_data.get('max_concurrent_trades'),
                performance_data.get('risk_adjusted_return'))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update strategy performance: {e}")
            return False
    
    async def get_user_strategy_performance(self, user_id: str, strategy_name: str = None,
                                          days: int = 30) -> List[Dict]:
        """Get strategy performance for user"""
        try:
            start_date = datetime.now().date() - timedelta(days=days)
            
            async with self.pg_pool.acquire() as conn:
                if strategy_name:
                    query = """
                        SELECT * FROM user_strategy_performance 
                        WHERE user_id = $1 AND strategy_name = $2 AND date >= $3
                        ORDER BY date DESC
                    """
                    params = [user_id, strategy_name, start_date]
                else:
                    query = """
                        SELECT * FROM user_strategy_performance 
                        WHERE user_id = $1 AND date >= $2
                        ORDER BY date DESC, strategy_name
                    """
                    params = [user_id, start_date]
                
                performance = await conn.fetch(query, *params)
                
                return [
                    {
                        'strategy_name': perf['strategy_name'],
                        'date': perf['date'].isoformat(),
                        'total_trades': perf['total_trades'],
                        'profitable_trades': perf['profitable_trades'],
                        'total_pnl': float(perf['total_pnl']),
                        'max_drawdown': float(perf['max_drawdown']) if perf['max_drawdown'] else 0,
                        'sharpe_ratio': float(perf['sharpe_ratio']) if perf['sharpe_ratio'] else 0,
                        'win_rate': float(perf['win_rate']),
                        'avg_trade_pnl': float(perf['avg_trade_pnl']),
                        'avg_holding_period': str(perf['avg_holding_period']) if perf['avg_holding_period'] else None,
                        'max_concurrent_trades': perf['max_concurrent_trades'],
                        'risk_adjusted_return': float(perf['risk_adjusted_return']) if perf['risk_adjusted_return'] else 0
                    }
                    for perf in performance
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to get strategy performance: {e}")
            return []
    
    # USER CACHE METHODS
    
    async def update_user_performance_cache(self, user_id: str, metrics: Dict):
        """Update user performance metrics in cache"""
        try:
            cursor = self.sqlite_conn.cursor()
            
            for metric_name, value in metrics.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO user_performance_cache 
                    (cache_id, user_id, metric_name, value, last_updated)
                    VALUES (?, ?, ?, ?, ?)
                """, (f"{user_id}_{metric_name}", user_id, metric_name, value,
                      int(datetime.now().timestamp())))
            
            self.sqlite_conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to update performance cache: {e}")
    
    async def get_user_performance_cache(self, user_id: str) -> Dict:
        """Get user performance metrics from cache"""
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                SELECT metric_name, value FROM user_performance_cache 
                WHERE user_id = ?
            """, (user_id,))
            
            return {row[0]: row[1] for row in cursor.fetchall()}
            
        except Exception as e:
            self.logger.error(f"Failed to get performance cache: {e}")
            return {}
    
    async def update_user_session_cache(self, user_id: str, session_data: Dict):
        """Update user session cache"""
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO user_session_cache 
                (user_id, session_id, trading_mode, daily_pnl, daily_trades, 
                 risk_status, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, session_data.get('session_id'),
                  session_data.get('trading_mode'),
                  session_data.get('daily_pnl', 0),
                  session_data.get('daily_trades', 0),
                  session_data.get('risk_status', 'ACTIVE'),
                  int(datetime.now().timestamp())))
            
            self.sqlite_conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to update session cache: {e}")
    
    # API CREDENTIALS MANAGEMENT
    
    def _encrypt_api_key(self, api_key: str, encryption_key: str) -> str:
        """Encrypt API key (simplified - use proper encryption in production)"""
        import base64
        combined = f"{api_key}:{encryption_key}"
        return base64.b64encode(combined.encode()).decode()
    
    def _decrypt_api_key(self, encrypted_key: str, encryption_key: str) -> str:
        """Decrypt API key (simplified - use proper encryption in production)"""
        import base64
        try:
            decoded = base64.b64decode(encrypted_key.encode()).decode()
            api_key, _ = decoded.split(':', 1)
            return api_key
        except:
            return None
    
    async def store_user_api_credentials(self, user_id: str, broker_name: str,
                                       api_key: str, secret_key: str, 
                                       broker_user_id: str = None) -> bool:
        """Store encrypted API credentials for user"""
        try:
            # Generate encryption key
            encryption_key = secrets.token_hex(32)
            encryption_key_hash = hashlib.sha256(encryption_key.encode()).hexdigest()
            
            # Encrypt credentials
            encrypted_api_key = self._encrypt_api_key(api_key, encryption_key)
            encrypted_secret_key = self._encrypt_api_key(secret_key, encryption_key)
            encrypted_user_id = self._encrypt_api_key(broker_user_id, encryption_key) if broker_user_id else None
            
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO user_api_credentials 
                    (user_id, broker_name, api_key_encrypted, secret_key_encrypted,
                     user_id_encrypted, encryption_key_hash)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (user_id, broker_name) DO UPDATE SET
                        api_key_encrypted = EXCLUDED.api_key_encrypted,
                        secret_key_encrypted = EXCLUDED.secret_key_encrypted,
                        user_id_encrypted = EXCLUDED.user_id_encrypted,
                        encryption_key_hash = EXCLUDED.encryption_key_hash,
                        updated_at = NOW()
                """, user_id, broker_name, encrypted_api_key, encrypted_secret_key,
                    encrypted_user_id, encryption_key_hash)
            
            # Store encryption key securely (in production, use key management service)
            # For now, we'll update the user config table
            await conn.execute("""
                UPDATE user_trading_config SET
                    goodwill_api_key = $2,
                    goodwill_secret_key = $3,
                    goodwill_user_id = $4
                WHERE user_id = $1
            """, user_id, encrypted_api_key[:50], encrypted_secret_key[:50], 
                encrypted_user_id[:50] if encrypted_user_id else None)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store API credentials: {e}")
            return False
    
    async def get_user_api_credentials(self, user_id: str, broker_name: str) -> Optional[Dict]:
        """Get decrypted API credentials for user"""
        try:
            async with self.pg_pool.acquire() as conn:
                creds = await conn.fetchrow("""
                    SELECT * FROM user_api_credentials 
                    WHERE user_id = $1 AND broker_name = $2 AND is_active = TRUE
                """, user_id, broker_name)
                
                if creds:
                    # In production, retrieve encryption key from secure storage
                    # For now, we'll use a placeholder
                    encryption_key = "placeholder_key"  # This should be retrieved securely
                    
                    return {
                        'api_key': self._decrypt_api_key(creds['api_key_encrypted'], encryption_key),
                        'secret_key': self._decrypt_api_key(creds['secret_key_encrypted'], encryption_key),
                        'user_id': self._decrypt_api_key(creds['user_id_encrypted'], encryption_key) if creds['user_id_encrypted'] else None,
                        'last_used': creds['last_used']
                    }
                
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get API credentials: {e}")
            return None
    
    # ENHANCED UTILITY METHODS
    
    async def get_user_dashboard_data(self, user_id: str) -> Dict:
        """Get comprehensive dashboard data for user"""
        try:
            # Get today's P&L
            daily_pnl = await self.calculate_user_daily_pnl(user_id)
            
            # Get portfolio summary
            portfolio = await self.get_user_portfolio(user_id)
            portfolio_value = sum(pos['current_value'] for pos in portfolio)
            
            # Get recent trades
            recent_trades = await self.get_user_trades(user_id, limit=10)
            
            # Get active alerts
            alerts = await self.get_user_risk_alerts(user_id, unacknowledged_only=True)
            
            # Get performance cache
            performance_cache = await self.get_user_performance_cache(user_id)
            
            # Get user config
            user_config = await self.get_user_config(user_id)
            
            return {
                'user_id': user_id,
                'daily_pnl': daily_pnl,
                'portfolio_value': portfolio_value,
                'portfolio_positions': len(portfolio),
                'recent_trades': recent_trades,
                'unacknowledged_alerts': len(alerts),
                'alerts': alerts[:5],  # Last 5 alerts
                'performance_metrics': performance_cache,
                'user_config': user_config,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get dashboard data: {e}")
            return {}
    
    async def cleanup_user_data(self, user_id: str, days_to_keep: int = 90):
        """Clean up old user data"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            async with self.pg_pool.acquire() as conn:
                # Clean old notifications
                await conn.execute("""
                    DELETE FROM user_notifications 
                    WHERE user_id = $1 AND created_at < $2 AND is_read = TRUE
                """, user_id, cutoff_date)
                
                # Clean old risk alerts
                await conn.execute("""
                    DELETE FROM user_risk_alerts 
                    WHERE user_id = $1 AND created_at < $2 AND is_acknowledged = TRUE
                """, user_id, cutoff_date)
                
                # Clean old strategy performance data
                strategy_cutoff = datetime.now() - timedelta(days=365)  # Keep 1 year
                await conn.execute("""
                    DELETE FROM user_strategy_performance 
                    WHERE user_id = $1 AND date < $2
                """, user_id, strategy_cutoff.date())
            
            self.logger.info(f"Cleaned up old data for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup user data: {e}")
    
    async def export_user_data(self, user_id: str, start_date: datetime, 
                             end_date: datetime) -> Dict:
        """Export user data for given date range"""
        try:
            # Get trades
            trades = await self.get_user_trades(user_id, start_date, end_date, limit=10000)
            
            # Get daily reports
            async with self.pg_pool.acquire() as conn:
                reports = await conn.fetch("""
                    SELECT * FROM daily_trading_reports 
                    WHERE user_id = $1 AND report_date BETWEEN $2 AND $3
                    ORDER BY report_date
                """, user_id, start_date.date(), end_date.date())
                
                daily_reports = [
                    {
                        'date': report['report_date'].isoformat(),
                        'daily_pnl': float(report['daily_pnl']),
                        'portfolio_value': float(report['portfolio_value']),
                        'total_trades': report['total_trades'],
                        'win_rate': float(report['win_rate'])
                    }
                    for report in reports
                ]
            
            # Get strategy performance
            strategy_performance = await self.get_user_strategy_performance(
                user_id, days=(end_date - start_date).days
            )
            
            return {
                'user_id': user_id,
                'export_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                },
                'trades': trades,
                'daily_reports': daily_reports,
                'strategy_performance': strategy_performance,
                'export_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to export user data: {e}")
            return {}
    
    async def get_user_statistics(self, user_id: str) -> Dict:
        """Get comprehensive user statistics"""
        try:
            async with self.pg_pool.acquire() as conn:
                # Basic stats
                basic_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_trades,
                        COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) as winning_trades,
                        COALESCE(SUM(realized_pnl), 0) as total_realized_pnl,
                        COALESCE(AVG(realized_pnl), 0) as avg_trade_pnl,
                        COALESCE(MAX(realized_pnl), 0) as best_trade,
                        COALESCE(MIN(realized_pnl), 0) as worst_trade,
                        COUNT(DISTINCT symbol) as symbols_traded,
                        COUNT(DISTINCT strategy) as strategies_used
                    FROM trades 
                    WHERE user_id = $1 AND status = 'FILLED'
                """, user_id)
                
                # Monthly performance
                monthly_stats = await conn.fetch("""
                    SELECT 
                        DATE_TRUNC('month', entry_time) as month,
                        COUNT(*) as trades,
                        SUM(realized_pnl) as monthly_pnl
                    FROM trades 
                    WHERE user_id = $1 AND status = 'FILLED'
                      AND entry_time >= NOW() - INTERVAL '12 months'
                    GROUP BY DATE_TRUNC('month', entry_time)
                    ORDER BY month DESC
                """, user_id)
                
                # Strategy breakdown
                strategy_stats = await conn.fetch("""
                    SELECT 
                        strategy,
                        COUNT(*) as trades,
                        COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) as winning_trades,
                        SUM(realized_pnl) as total_pnl
                    FROM trades 
                    WHERE user_id = $1 AND status = 'FILLED'
                    GROUP BY strategy
                    ORDER BY total_pnl DESC
                """, user_id)
                
                win_rate = (basic_stats['winning_trades'] / basic_stats['total_trades'] * 100) if basic_stats['total_trades'] > 0 else 0
                
                return {
                    'user_id': user_id,
                    'total_trades': basic_stats['total_trades'],
                    'winning_trades': basic_stats['winning_trades'],
                    'losing_trades': basic_stats['total_trades'] - basic_stats['winning_trades'],
                    'win_rate': round(win_rate, 2),
                    'total_realized_pnl': float(basic_stats['total_realized_pnl']),
                    'avg_trade_pnl': float(basic_stats['avg_trade_pnl']),
                    'best_trade': float(basic_stats['best_trade']),
                    'worst_trade': float(basic_stats['worst_trade']),
                    'symbols_traded': basic_stats['symbols_traded'],
                    'strategies_used': basic_stats['strategies_used'],
                    'monthly_performance': [
                        {
                            'month': stat['month'].strftime('%Y-%m'),
                            'trades': stat['trades'],
                            'pnl': float(stat['monthly_pnl'])
                        }
                        for stat in monthly_stats
                    ],
                    'strategy_breakdown': [
                        {
                            'strategy': stat['strategy'],
                            'trades': stat['trades'],
                            'winning_trades': stat['winning_trades'],
                            'win_rate': round(stat['winning_trades'] / stat['trades'] * 100, 2) if stat['trades'] > 0 else 0,
                            'total_pnl': float(stat['total_pnl'])
                        }
                        for stat in strategy_stats
                    ]
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get user statistics: {e}")
            return {}
    
    # Keep all existing methods from the original DatabaseManager class
    # (market data operations, trading operations, etc.)
    
    async def store_market_data(self, data: MarketData):
        """Store market data in both PostgreSQL and SQLite (unchanged)"""
        try:
            # Store in PostgreSQL for historical data
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO market_data_history 
                    (symbol, timestamp, open_price, high_price, low_price, close_price, volume, ltp)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (symbol, timestamp, timeframe, exchange) DO UPDATE SET
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
        """Update real-time quote in SQLite cache (unchanged)"""
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
        """Get latest price from SQLite cache (unchanged)"""
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("SELECT ltp FROM realtime_quotes WHERE symbol = ?", (symbol,))
            result = cursor.fetchone()
            return result[0] if result else None
            
        except Exception as e:
            self.logger.error(f"Failed to get latest price: {e}")
            return None
    
    async def close(self):
        """Close database connections (unchanged)"""
        try:
            if self.pg_pool:
                await self.pg_pool.close()
            
            if self.sqlite_conn:
                self.sqlite_conn.close()
            
            self.logger.info("Database connections closed")
            
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")
    
    # Add a connection context manager for easy database access
    async def get_connection(self):
        """Get database connection context manager"""
        return self.pg_pool.acquire()

# Example usage and testing for user management
async def test_user_management():
    """Test user management functionality"""
    from config_manager import ConfigManager
    
    # Initialize with config
    config_manager = ConfigManager("config/config.yaml")
    trading_db = DatabaseManager(config_manager)
    
    try:
        # Initialize database
        await trading_db.initialize()
        
        # Create test user
        user_id = await trading_db.create_user(
            username="test_trader",
            email="test@example.com",
            password="secure_password",
            subscription_type="premium"
        )
        print(f"Created user: {user_id}")
        
        # Authenticate user
        auth_user_id = await trading_db.authenticate_user("test_trader", "secure_password")
        print(f"Authenticated user: {auth_user_id}")
        
        # Update user config
        config_data = {
            'capital': 200000.0,
            'risk_per_trade_percent': 1.5,
            'daily_loss_limit_percent': 3.0,
            'max_concurrent_trades': 3,
            'risk_reward_ratio': 2.5,
            'max_position_size_percent': 15.0,
            'stop_loss_percent': 2.5,
            'take_profit_percent': 5.0,
            'trading_start_time': '09:30',
            'trading_end_time': '15:15',
            'auto_square_off': True,
            'paper_trading_mode': False,  # Live trading
            'email_notifications': True
        }
        
        await trading_db.update_user_config(user_id, config_data)
        print("User config updated")
        
        # Start trading session
        session_id = await trading_db.start_trading_session(user_id, "live", 200000.0)
        print(f"Trading session started: {session_id}")
        
        # Log a test trade
        trade_data = {
            'session_id': session_id,
            'symbol': 'RELIANCE',
            'action': 'BUY',
            'strategy': 'momentum_breakout',
            'quantity': 50,
            'entry_price': 2515.0,
            'stop_loss': 2440.0,
            'take_profit': 2640.0,
            'confidence': 0.85,
            'notes': 'Test trade for user system'
        }
        
        trade_id = await trading_db.log_user_trade(user_id, trade_data)
        print(f"Trade logged: {trade_id}")
        
        # Update portfolio
        await trading_db.update_user_portfolio(user_id, 'RELIANCE', 50, 2515.0)
        print("Portfolio updated")
        
        # Create risk alert
        alert_id = await trading_db.create_risk_alert(
            user_id=user_id,
            alert_type='POSITION_SIZE',
            alert_level='WARNING',
            message='Position size approaching limit',
            symbol='RELIANCE',
            current_value=125750.0,
            threshold_value=120000.0
        )
        print(f"Risk alert created: {alert_id}")
        
        # Get dashboard data
        dashboard = await trading_db.get_user_dashboard_data(user_id)
        print(f"Dashboard data: {json.dumps(dashboard, indent=2, default=str)}")
        
        # Get user statistics
        stats = await trading_db.get_user_statistics(user_id)
        print(f"User statistics: {json.dumps(stats, indent=2, default=str)}")
        
        print("User management testing completed successfully!")
        
    except Exception as e:
        print(f"Error during user management testing: {e}")
        raise
    finally:
        await trading_db.close()

if __name__ == "__main__":
    # Test the enhanced database system
    asyncio.run(test_user_management())
