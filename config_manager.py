#!/usr/bin/env python3
"""
Advanced Configuration Manager for User-Based Trading Bot
Handles all configuration settings, user risk parameters, and environment variables
Integrated with database system and user risk management from webapp
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, time, timedelta
import pytz
import asyncpg
import sqlite3
from threading import Lock

# Import system components for integration
try:
    from utils import Logger, ErrorHandler, DataValidator
except ImportError:
    # Fallback if utils not available
    Logger = logging.getLogger
    ErrorHandler = Exception
    DataValidator = object

@dataclass
class DatabaseConfig:
    """Database configuration for PostgreSQL + SQLite hybrid"""
    postgresql: Dict[str, Any] = None
    sqlite_path: str = "data/realtime_cache.db"
    connection_pool_size: int = 20
    enable_logging: bool = True
    backup_enabled: bool = True
    backup_schedule: str = "daily"
    
    def __post_init__(self):
        if self.postgresql is None:
            self.postgresql = {
                "host": os.getenv("DB_HOST", "localhost"),
                "port": int(os.getenv("DB_PORT", 5432)),
                "database": os.getenv("DB_NAME", "trading_bot"),
                "user": os.getenv("DB_USER", "trading_user"),
                "password": os.getenv("DB_PASSWORD", "change_this_password")
            }

@dataclass
class TradingConfig:
    """Base trading configuration - user-specific settings loaded from database"""
    trading_mode: str = "paper"  # paper, live
    auto_schedule: bool = True
    market_hours_ist: Dict[str, str] = None
    enable_premarket: bool = False
    enable_afterhours: bool = False
    default_user_id: str = "default_user"
    max_users_concurrent: int = 10
    session_timeout_minutes: int = 480  # 8 hours
    default_timeframe: str = "1min"
    supported_timeframes: List[str] = None
    
    def __post_init__(self):
        if self.market_hours_ist is None:
            self.market_hours_ist = {
                "market_open": "09:15",
                "market_close": "15:30",
                "pre_market_start": "09:00",
                "after_market_end": "16:00"
            }
        if self.supported_timeframes is None:
            self.supported_timeframes = ["1min", "5min", "15min", "1h", "1d"]

@dataclass
class DefaultRiskConfig:
    """Default risk management configuration for new users"""
    capital: float = 100000.0
    risk_per_trade_percent: float = 2.0
    daily_loss_limit_percent: float = 5.0
    max_concurrent_trades: int = 5
    risk_reward_ratio: float = 2.0
    max_position_size_percent: float = 20.0
    stop_loss_percent: float = 3.0
    take_profit_percent: float = 6.0
    trading_start_time: str = "09:15"
    trading_end_time: str = "15:30"
    auto_square_off: bool = True
    paper_trading_mode: bool = True
    max_daily_trades: int = 50
    trailing_stop_loss: bool = False
    bracket_order_enabled: bool = True

@dataclass
class APIConfig:
    """API configuration for brokers and data feeds"""
    goodwill: Dict[str, Any] = None
    rate_limit_requests: int = 300
    rate_limit_window: int = 60
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        if self.goodwill is None:
            self.goodwill = {
                "base_url": os.getenv("GOODWILL_BASE_URL", "https://api.goodwill.com"),
                "api_key": os.getenv("GOODWILL_API_KEY", ""),
                "secret_key": os.getenv("GOODWILL_SECRET_KEY", ""),
                "user_id": os.getenv("GOODWILL_USER_ID", "")
            }

@dataclass
class FlaskConfig:
    """Flask webapp configuration"""
    secret_key: str = ""
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 5000
    max_content_length: int = 16 * 1024 * 1024  # 16MB
    session_timeout: int = 3600  # 1 hour
    csrf_enabled: bool = True
    
    def __post_init__(self):
        if not self.secret_key:
            self.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(32).hex())

@dataclass
class NotificationConfig:
    """Notification system configuration"""
    email_enabled: bool = True
    sms_enabled: bool = False
    webhook_enabled: bool = False
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    webhook_url: str = ""
    
    def __post_init__(self):
        self.email_username = os.getenv("EMAIL_USERNAME", "")
        self.email_password = os.getenv("EMAIL_PASSWORD", "")
        self.webhook_url = os.getenv("WEBHOOK_URL", "")

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    file_path: str = "logs/trading_bot.log"
    max_file_size_mb: int = 100
    backup_count: int = 7
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    enable_console: bool = True
    enable_file: bool = True

class ConfigManager:
    """
    Centralized configuration management with user risk integration
    Connects with database to load user-specific risk parameters
    """
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        self.config_path = self.config_dir / config_file
        
        # Thread safety
        self._lock = Lock()
        
        # Database connection
        self.db_pool = None
        self.sqlite_conn = None
        
        # Initialize configuration objects
        self.database = DatabaseConfig()
        self.trading = TradingConfig()
        self.default_risk = DefaultRiskConfig()
        self.api = APIConfig()
        self.flask = FlaskConfig()
        self.notifications = NotificationConfig()
        self.logging_config = LoggingConfig()
        
        # User risk configurations cache
        self._user_configs = {}
        self._config_cache_timeout = 300  # 5 minutes
        self._last_cache_update = {}
        
        # Load configuration
        self.load_config()
        self.load_environment_variables()
        self.setup_logging()
        
        # Initialize logger after setup
        self.logger = Logger(__name__) if hasattr(Logger, '__call__') else logging.getLogger(__name__)
    
    async def initialize_database_connection(self):
        """Initialize database connection for loading user configs"""
        try:
            # PostgreSQL connection pool
            self.db_pool = await asyncpg.create_pool(
                host=self.database.postgresql['host'],
                port=self.database.postgresql['port'],
                database=self.database.postgresql['database'],
                user=self.database.postgresql['user'],
                password=self.database.postgresql['password'],
                min_size=2,
                max_size=5
            )
            
            # SQLite connection for cache
            os.makedirs(os.path.dirname(self.database.sqlite_path), exist_ok=True)
            self.sqlite_conn = sqlite3.connect(
                self.database.sqlite_path,
                check_same_thread=False
            )
            
            self.logger.info("Database connections initialized for config manager")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database connections: {e}")
            return False
    
    async def load_user_risk_config(self, user_id: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Load user-specific risk configuration from database"""
        try:
            # Check cache first
            if not force_refresh and user_id in self._user_configs:
                last_update = self._last_cache_update.get(user_id, 0)
                if (datetime.now().timestamp() - last_update) < self._config_cache_timeout:
                    return self._user_configs[user_id]
            
            if not self.db_pool:
                await self.initialize_database_connection()
            
            async with self.db_pool.acquire() as conn:
                config_row = await conn.fetchrow("""
                    SELECT * FROM user_trading_config WHERE user_id = $1
                """, user_id)
                
                if config_row:
                    user_config = {
                        'user_id': config_row['user_id'],
                        'capital': float(config_row['capital']),
                        'risk_per_trade_percent': float(config_row['risk_per_trade_percent']),
                        'daily_loss_limit_percent': float(config_row['daily_loss_limit_percent']),
                        'max_concurrent_trades': config_row['max_concurrent_trades'],
                        'risk_reward_ratio': float(config_row['risk_reward_ratio']),
                        'max_position_size_percent': float(config_row['max_position_size_percent']),
                        'stop_loss_percent': float(config_row['stop_loss_percent']),
                        'take_profit_percent': float(config_row['take_profit_percent']),
                        'trading_start_time': config_row['trading_start_time'].strftime('%H:%M'),
                        'trading_end_time': config_row['trading_end_time'].strftime('%H:%M'),
                        'auto_square_off': config_row['auto_square_off'],
                        'paper_trading_mode': config_row['paper_trading_mode'],
                        'max_daily_trades': config_row.get('max_daily_trades', 50),
                        'trailing_stop_loss': config_row.get('trailing_stop_loss', False),
                        'bracket_order_enabled': config_row.get('bracket_order_enabled', True),
                        'email_notifications': config_row.get('email_notifications', True),
                        'sms_notifications': config_row.get('sms_notifications', False),
                        'last_updated': config_row['last_updated']
                    }
                    
                    # Cache the configuration
                    with self._lock:
                        self._user_configs[user_id] = user_config
                        self._last_cache_update[user_id] = datetime.now().timestamp()
                    
                    return user_config
                else:
                    # Return default configuration for new users
                    return self.get_default_user_config()
                    
        except Exception as e:
            self.logger.error(f"Error loading user risk config for {user_id}: {e}")
            return self.get_default_user_config()
    
    def get_default_user_config(self) -> Dict[str, Any]:
        """Get default user configuration"""
        return {
            'capital': self.default_risk.capital,
            'risk_per_trade_percent': self.default_risk.risk_per_trade_percent,
            'daily_loss_limit_percent': self.default_risk.daily_loss_limit_percent,
            'max_concurrent_trades': self.default_risk.max_concurrent_trades,
            'risk_reward_ratio': self.default_risk.risk_reward_ratio,
            'max_position_size_percent': self.default_risk.max_position_size_percent,
            'stop_loss_percent': self.default_risk.stop_loss_percent,
            'take_profit_percent': self.default_risk.take_profit_percent,
            'trading_start_time': self.default_risk.trading_start_time,
            'trading_end_time': self.default_risk.trading_end_time,
            'auto_square_off': self.default_risk.auto_square_off,
            'paper_trading_mode': self.default_risk.paper_trading_mode,
            'max_daily_trades': self.default_risk.max_daily_trades,
            'trailing_stop_loss': self.default_risk.trailing_stop_loss,
            'bracket_order_enabled': self.default_risk.bracket_order_enabled
        }
    
    async def update_user_risk_config(self, user_id: str, config_data: Dict[str, Any]) -> bool:
        """Update user risk configuration in database"""
        try:
            if not self.db_pool:
                await self.initialize_database_connection()
            
            async with self.db_pool.acquire() as conn:
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
                config_data['capital'],
                config_data['risk_per_trade_percent'],
                config_data['daily_loss_limit_percent'],
                config_data['max_concurrent_trades'],
                config_data['risk_reward_ratio'],
                config_data['max_position_size_percent'],
                config_data['stop_loss_percent'],
                config_data['take_profit_percent'],
                datetime.strptime(config_data['trading_start_time'], '%H:%M').time(),
                datetime.strptime(config_data['trading_end_time'], '%H:%M').time(),
                config_data['auto_square_off'],
                config_data['paper_trading_mode'],
                config_data.get('max_daily_trades', 50),
                config_data.get('trailing_stop_loss', False),
                config_data.get('bracket_order_enabled', True),
                config_data.get('email_notifications', True),
                config_data.get('sms_notifications', False))
            
            # Update cache
            with self._lock:
                if user_id in self._user_configs:
                    self._user_configs[user_id].update(config_data)
                    self._last_cache_update[user_id] = datetime.now().timestamp()
            
            self.logger.info(f"Updated user risk config for {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating user risk config: {e}")
            return False
    
    def load_config(self):
        """Load configuration from JSON file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Update configuration objects
                if 'database' in config_data:
                    self.database = DatabaseConfig(**config_data['database'])
                if 'trading' in config_data:
                    self.trading = TradingConfig(**config_data['trading'])
                if 'default_risk' in config_data:
                    self.default_risk = DefaultRiskConfig(**config_data['default_risk'])
                if 'api' in config_data:
                    self.api = APIConfig(**config_data['api'])
                if 'flask' in config_data:
                    self.flask = FlaskConfig(**config_data['flask'])
                if 'notifications' in config_data:
                    self.notifications = NotificationConfig(**config_data['notifications'])
                if 'logging' in config_data:
                    self.logging_config = LoggingConfig(**config_data['logging'])
                
                logging.info(f"Configuration loaded from {self.config_path}")
            else:
                # Create default configuration file
                self.save_config()
                logging.info(f"Created default configuration file: {self.config_path}")
                
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            logging.info("Using default configuration")
    
    def load_environment_variables(self):
        """Load sensitive configuration from environment variables"""
        try:
            # Database credentials
            if os.getenv('DATABASE_URL'):
                # Parse DATABASE_URL if provided
                db_url = os.getenv('DATABASE_URL')
                # Update postgresql config from URL
                
            # Flask configuration
            if os.getenv('FLASK_SECRET_KEY'):
                self.flask.secret_key = os.getenv('FLASK_SECRET_KEY')
            if os.getenv('FLASK_DEBUG'):
                self.flask.debug = os.getenv('FLASK_DEBUG').lower() == 'true'
            if os.getenv('FLASK_PORT'):
                self.flask.port = int(os.getenv('FLASK_PORT'))
            
            # Trading mode
            if os.getenv('TRADING_MODE'):
                self.trading.trading_mode = os.getenv('TRADING_MODE')
            if os.getenv('AUTO_SCHEDULE'):
                self.trading.auto_schedule = os.getenv('AUTO_SCHEDULE').lower() == 'true'
            
            # Notification settings
            if os.getenv('EMAIL_USERNAME'):
                self.notifications.email_username = os.getenv('EMAIL_USERNAME')
            if os.getenv('EMAIL_PASSWORD'):
                self.notifications.email_password = os.getenv('EMAIL_PASSWORD')
            
            # Logging
            if os.getenv('LOG_LEVEL'):
                self.logging_config.level = os.getenv('LOG_LEVEL')
            
            logging.info("Environment variables loaded")
            
        except Exception as e:
            logging.error(f"Error loading environment variables: {e}")
    
    def save_config(self):
        """Save current configuration to JSON file (excluding sensitive data)"""
        try:
            config_data = {
                'database': {
                    'postgresql': {
                        'host': self.database.postgresql['host'],
                        'port': self.database.postgresql['port'],
                        'database': self.database.postgresql['database']
                        # Exclude user/password - use env vars
                    },
                    'sqlite_path': self.database.sqlite_path,
                    'connection_pool_size': self.database.connection_pool_size,
                    'enable_logging': self.database.enable_logging,
                    'backup_enabled': self.database.backup_enabled,
                    'backup_schedule': self.database.backup_schedule
                },
                'trading': asdict(self.trading),
                'default_risk': asdict(self.default_risk),
                'api': {
                    'goodwill': {
                        'base_url': self.api.goodwill['base_url']
                        # Exclude API keys - use env vars
                    },
                    'rate_limit_requests': self.api.rate_limit_requests,
                    'rate_limit_window': self.api.rate_limit_window,
                    'timeout': self.api.timeout,
                    'max_retries': self.api.max_retries,
                    'retry_delay': self.api.retry_delay
                },
                'flask': {
                    'debug': self.flask.debug,
                    'host': self.flask.host,
                    'port': self.flask.port,
                    'max_content_length': self.flask.max_content_length,
                    'session_timeout': self.flask.session_timeout,
                    'csrf_enabled': self.flask.csrf_enabled
                    # Exclude secret_key - use env vars
                },
                'notifications': {
                    'email_enabled': self.notifications.email_enabled,
                    'sms_enabled': self.notifications.sms_enabled,
                    'webhook_enabled': self.notifications.webhook_enabled,
                    'smtp_server': self.notifications.smtp_server,
                    'smtp_port': self.notifications.smtp_port
                    # Exclude credentials - use env vars
                },
                'logging': asdict(self.logging_config)
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=4, default=str)
            
            logging.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        try:
            # Create logs directory
            logs_dir = Path(self.logging_config.file_path).parent
            logs_dir.mkdir(exist_ok=True, parents=True)
            
            # Configure logging
            log_level = getattr(logging, self.logging_config.level.upper(), logging.INFO)
            
            # Clear existing handlers
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            
            root_logger.setLevel(log_level)
            
            # Create formatters
            formatter = logging.Formatter(self.logging_config.format_string)
            
            # Console handler
            if self.logging_config.enable_console:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(log_level)
                console_handler.setFormatter(formatter)
                root_logger.addHandler(console_handler)
            
            # File handler with rotation
            if self.logging_config.enable_file:
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(
                    self.logging_config.file_path,
                    maxBytes=self.logging_config.max_file_size_mb * 1024 * 1024,
                    backupCount=self.logging_config.backup_count
                )
                file_handler.setLevel(log_level)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
            
            logging.info("Logging configuration completed")
            
        except Exception as e:
            print(f"Error setting up logging: {e}")
    
    def get_config(self, section: str = None) -> Any:
        """Get configuration section or entire config"""
        if section:
            return getattr(self, section, None)
        else:
            return {
                'database': self.database,
                'trading': self.trading,
                'default_risk': self.default_risk,
                'api': self.api,
                'flask': self.flask,
                'notifications': self.notifications,
                'logging': self.logging_config
            }
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return getattr(self, key, default)
    
    def is_trading_hours(self, user_config: Dict[str, Any] = None) -> bool:
        """Check if current time is within trading hours"""
        try:
            # Get IST timezone
            ist = pytz.timezone('Asia/Kolkata')
            current_time = datetime.now(ist).time()
            current_day = datetime.now(ist).weekday()
            
            # Skip weekends (Saturday=5, Sunday=6)
            if current_day >= 5:
                return False
            
            # Use user-specific trading hours if available
            if user_config:
                start_time = datetime.strptime(user_config['trading_start_time'], '%H:%M').time()
                end_time = datetime.strptime(user_config['trading_end_time'], '%H:%M').time()
            else:
                # Use default market hours
                start_time = datetime.strptime(self.trading.market_hours_ist['market_open'], '%H:%M').time()
                end_time = datetime.strptime(self.trading.market_hours_ist['market_close'], '%H:%M').time()
            
            return start_time <= current_time <= end_time
            
        except Exception as e:
            self.logger.error(f"Error checking trading hours: {e}")
            return False
    
    def validate_user_config(self, user_config: Dict[str, Any]) -> Dict[str, bool]:
        """Validate user risk configuration"""
        validation_results = {}
        
        try:
            # Risk parameters validation
            validation_results['capital_valid'] = user_config['capital'] > 0
            validation_results['risk_per_trade_valid'] = 0 < user_config['risk_per_trade_percent'] <= 10
            validation_results['daily_loss_limit_valid'] = 0 < user_config['daily_loss_limit_percent'] <= 20
            validation_results['max_trades_valid'] = 0 < user_config['max_concurrent_trades'] <= 50
            validation_results['risk_reward_valid'] = user_config['risk_reward_ratio'] > 0
            validation_results['position_size_valid'] = 0 < user_config['max_position_size_percent'] <= 100
            validation_results['stop_loss_valid'] = 0 < user_config['stop_loss_percent'] <= 50
            validation_results['take_profit_valid'] = 0 < user_config['take_profit_percent'] <= 100
            
            # Trading hours validation
            try:
                start_time = datetime.strptime(user_config['trading_start_time'], '%H:%M')
                end_time = datetime.strptime(user_config['trading_end_time'], '%H:%M')
                validation_results['trading_hours_valid'] = start_time < end_time
            except:
                validation_results['trading_hours_valid'] = False
            
            # Overall validation
            validation_results['overall_valid'] = all(validation_results.values())
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating user config: {e}")
            return {'overall_valid': False}
    
    def calculate_derived_risk_params(self, user_config: Dict[str, Any]) -> Dict[str, float]:
        """Calculate derived risk parameters from user config"""
        try:
            capital = user_config['capital']
            
            return {
                'max_position_value': capital * (user_config['max_position_size_percent'] / 100),
                'daily_loss_limit_amount': capital * (user_config['daily_loss_limit_percent'] / 100),
                'risk_per_trade_amount': capital * (user_config['risk_per_trade_percent'] / 100),
                'max_total_exposure': capital * 0.8,  # 80% max exposure
                'emergency_stop_loss': capital * 0.1  # 10% emergency stop
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating derived risk parameters: {e}")
            return {}
    
    async def close(self):
        """Close database connections"""
        try:
            if self.db_pool:
                await self.db_pool.close()
            if self.sqlite_conn:
                self.sqlite_conn.close()
            self.logger.info("Config manager database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing config manager connections: {e}")
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"""
Trading Bot Configuration:
- Mode: {self.trading.trading_mode.upper()}
- Auto Schedule: {self.trading.auto_schedule}
- Default Capital: ₹{self.default_risk.capital:,.2f}
- Default Risk/Trade: {self.default_risk.risk_per_trade_percent}%
- Max Users: {self.trading.max_users_concurrent}
- Database: PostgreSQL + SQLite
- Flask Port: {self.flask.port}
"""

# Global configuration instance
config_manager = ConfigManager()

def get_config() -> ConfigManager:
    """Get the global configuration instance"""
    return config_manager

# Convenience functions for common operations
async def get_user_config(user_id: str) -> Dict[str, Any]:
    """Get user configuration"""
    return await config_manager.load_user_risk_config(user_id)

async def update_user_config(user_id: str, config_data: Dict[str, Any]) -> bool:
    """Update user configuration"""
    return await config_manager.update_user_risk_config(user_id, config_data)

def is_trading_time(user_config: Dict[str, Any] = None) -> bool:
    """Check if it's trading time"""
    return config_manager.is_trading_hours(user_config)

def validate_config(user_config: Dict[str, Any]) -> Dict[str, bool]:
    """Validate user configuration"""
    return config_manager.validate_user_config(user_config)
