"""
Advanced Configuration Manager for Momentum Trading Bot
Handles all configuration settings, risk parameters, and environment variables
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_risk_per_trade: float = 2.0  # Maximum % of portfolio to risk per trade
    max_portfolio_risk: float = 10.0  # Maximum total portfolio risk %
    stop_loss_percentage: float = 5.0  # Default stop loss %
    trailing_stop_percentage: float = 3.0  # Trailing stop %
    max_positions: int = 10  # Maximum concurrent positions
    position_sizing_method: str = "fixed_risk"  # fixed_risk, kelly, equal_weight
    volatility_lookback: int = 20  # Days for volatility calculation
    risk_free_rate: float = 0.02  # Risk-free rate for calculations

@dataclass
class TradingConfig:
    """Trading strategy configuration"""
    strategy_name: str = "momentum_breakout"
    timeframe: str = "1min"  # 1min, 5min, 15min, 1h, 1d
    momentum_period: int = 20
    volume_threshold: float = 1.5  # Volume multiplier
    price_change_threshold: float = 2.0  # Price change % threshold
    min_price: float = 5.0  # Minimum stock price
    max_price: float = 500.0  # Maximum stock price
    market_cap_min: int = 100000000  # Minimum market cap
    enable_premarket: bool = True
    enable_afterhours: bool = False
    paper_trading: bool = True  # Start with paper trading

@dataclass
class APIConfig:
    """API configuration settings"""
    goodwill_api_key: str = ""
    goodwill_secret: str = ""
    goodwill_base_url: str = "https://api.goodwillapi.com"
    rate_limit_requests: int = 300  # Requests per minute
    rate_limit_window: int = 60  # Time window in seconds
    timeout: int = 30  # Request timeout in seconds
    max_retries: int = 3
    retry_delay: float = 1.0

@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_type: str = "sqlite"  # sqlite, postgresql, mysql
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "momentum_trading.db"
    db_user: str = ""
    db_password: str = ""
    connection_pool_size: int = 10
    enable_logging: bool = True

@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    enable_email_alerts: bool = False
    email_smtp_server: str = ""
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    alert_recipients: list = None
    enable_discord_webhook: bool = False
    discord_webhook_url: str = ""
    enable_slack_webhook: bool = False
    slack_webhook_url: str = ""
    log_level: str = "INFO"
    max_log_files: int = 7
    log_file_size_mb: int = 100

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        self.config_path = self.config_dir / config_file
        
        # Initialize configuration objects
        self.risk = RiskConfig()
        self.trading = TradingConfig()
        self.api = APIConfig()
        self.database = DatabaseConfig()
        self.monitoring = MonitoringConfig()
        
        # Load configuration
        self.load_config()
        self.load_environment_variables()
        
        # Set up logging
        self.setup_logging()
    
    def load_config(self):
        """Load configuration from JSON file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Update configuration objects
                if 'risk' in config_data:
                    self.risk = RiskConfig(**config_data['risk'])
                if 'trading' in config_data:
                    self.trading = TradingConfig(**config_data['trading'])
                if 'api' in config_data:
                    self.api = APIConfig(**config_data['api'])
                if 'database' in config_data:
                    self.database = DatabaseConfig(**config_data['database'])
                if 'monitoring' in config_data:
                    monitoring_data = config_data['monitoring']
                    if 'alert_recipients' not in monitoring_data:
                        monitoring_data['alert_recipients'] = []
                    self.monitoring = MonitoringConfig(**monitoring_data)
                
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
            # API credentials
            if os.getenv('GOODWILL_API_KEY'):
                self.api.goodwill_api_key = os.getenv('GOODWILL_API_KEY')
            if os.getenv('GOODWILL_SECRET'):
                self.api.goodwill_secret = os.getenv('GOODWILL_SECRET')
            
            # Database credentials
            if os.getenv('DB_USER'):
                self.database.db_user = os.getenv('DB_USER')
            if os.getenv('DB_PASSWORD'):
                self.database.db_password = os.getenv('DB_PASSWORD')
            if os.getenv('DB_HOST'):
                self.database.db_host = os.getenv('DB_HOST')
            if os.getenv('DB_NAME'):
                self.database.db_name = os.getenv('DB_NAME')
            
            # Email credentials
            if os.getenv('EMAIL_USERNAME'):
                self.monitoring.email_username = os.getenv('EMAIL_USERNAME')
            if os.getenv('EMAIL_PASSWORD'):
                self.monitoring.email_password = os.getenv('EMAIL_PASSWORD')
            
            # Webhook URLs
            if os.getenv('DISCORD_WEBHOOK_URL'):
                self.monitoring.discord_webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
            if os.getenv('SLACK_WEBHOOK_URL'):
                self.monitoring.slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL')
            
            # Trading mode
            if os.getenv('PAPER_TRADING'):
                self.trading.paper_trading = os.getenv('PAPER_TRADING').lower() == 'true'
            
            logging.info("Environment variables loaded")
            
        except Exception as e:
            logging.error(f"Error loading environment variables: {e}")
    
    def save_config(self):
        """Save current configuration to JSON file"""
        try:
            config_data = {
                'risk': asdict(self.risk),
                'trading': asdict(self.trading),
                'api': {k: v for k, v in asdict(self.api).items() 
                       if k not in ['goodwill_api_key', 'goodwill_secret']},  # Don't save secrets
                'database': {k: v for k, v in asdict(self.database).items() 
                           if k not in ['db_user', 'db_password']},  # Don't save credentials
                'monitoring': {k: v for k, v in asdict(self.monitoring).items() 
                             if k not in ['email_username', 'email_password']}  # Don't save credentials
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
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            # Configure logging
            log_level = getattr(logging, self.monitoring.log_level.upper(), logging.INFO)
            
            # Create formatters
            detailed_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            simple_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            
            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(log_level)
            
            # Clear existing handlers
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(simple_formatter)
            root_logger.addHandler(console_handler)
            
            # File handler with rotation
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                logs_dir / "momentum_bot.log",
                maxBytes=self.monitoring.log_file_size_mb * 1024 * 1024,
                backupCount=self.monitoring.max_log_files
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)
            
            # Error file handler
            error_handler = RotatingFileHandler(
                logs_dir / "momentum_bot_errors.log",
                maxBytes=self.monitoring.log_file_size_mb * 1024 * 1024,
                backupCount=self.monitoring.max_log_files
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(error_handler)
            
            logging.info("Logging configuration completed")
            
        except Exception as e:
            print(f"Error setting up logging: {e}")
    
    def update_config(self, section: str, key: str, value: Any):
        """Update a specific configuration value"""
        try:
            config_obj = getattr(self, section, None)
            if config_obj and hasattr(config_obj, key):
                setattr(config_obj, key, value)
                self.save_config()
                logging.info(f"Updated {section}.{key} = {value}")
                return True
            else:
                logging.error(f"Invalid configuration path: {section}.{key}")
                return False
                
        except Exception as e:
            logging.error(f"Error updating configuration: {e}")
            return False
    
    def get_config(self, section: str, key: str = None) -> Any:
        """Get configuration value"""
        try:
            config_obj = getattr(self, section, None)
            if config_obj:
                if key:
                    return getattr(config_obj, key, None)
                else:
                    return config_obj
            return None
            
        except Exception as e:
            logging.error(f"Error getting configuration: {e}")
            return None
    
    def validate_config(self) -> Dict[str, bool]:
        """Validate configuration settings"""
        validation_results = {}
        
        try:
            # Validate risk configuration
            validation_results['risk_valid'] = (
                0 < self.risk.max_risk_per_trade <= 10 and
                0 < self.risk.max_portfolio_risk <= 50 and
                0 < self.risk.stop_loss_percentage <= 20 and
                0 < self.risk.trailing_stop_percentage <= 10 and
                self.risk.max_positions > 0
            )
            
            # Validate trading configuration
            validation_results['trading_valid'] = (
                self.trading.timeframe in ['1min', '5min', '15min', '1h', '1d'] and
                self.trading.momentum_period > 0 and
                self.trading.volume_threshold > 0 and
                self.trading.min_price > 0 and
                self.trading.max_price > self.trading.min_price
            )
            
            # Validate API configuration
            validation_results['api_valid'] = (
                len(self.api.goodwill_api_key) > 0 and
                len(self.api.goodwill_secret) > 0 and
                self.api.rate_limit_requests > 0 and
                self.api.timeout > 0
            )
            
            # Validate database configuration
            validation_results['database_valid'] = (
                self.database.db_type in ['sqlite', 'postgresql', 'mysql'] and
                len(self.database.db_name) > 0
            )
            
            # Overall validation
            validation_results['overall_valid'] = all(validation_results.values())
            
            return validation_results
            
        except Exception as e:
            logging.error(f"Error validating configuration: {e}")
            return {'overall_valid': False}
    
    def get_database_url(self) -> str:
        """Get database connection URL"""
        try:
            if self.database.db_type == 'sqlite':
                return f"sqlite:///{self.database.db_name}"
            elif self.database.db_type == 'postgresql':
                return f"postgresql://{self.database.db_user}:{self.database.db_password}@{self.database.db_host}:{self.database.db_port}/{self.database.db_name}"
            elif self.database.db_type == 'mysql':
                return f"mysql://{self.database.db_user}:{self.database.db_password}@{self.database.db_host}:{self.database.db_port}/{self.database.db_name}"
            else:
                return f"sqlite:///{self.database.db_name}"
                
        except Exception as e:
            logging.error(f"Error getting database URL: {e}")
            return f"sqlite:///{self.database.db_name}"
    
    def is_trading_hours(self) -> bool:
        """Check if current time is within Indian market trading hours"""
        try:
            from datetime import datetime, time
            import pytz
            
            # Get current time in IST (Indian Standard Time)
            ist = pytz.timezone('Asia/Kolkata')
            current_time = datetime.now(ist).time()
            
            # NSE Regular trading hours: 9:15 AM - 3:30 PM IST
            market_open = time(9, 15)
            market_close = time(15, 30)
            
            # Pre-market: 9:00 AM - 9:15 AM IST
            premarket_open = time(9, 0)
            
            # After-hours: 3:30 PM - 4:00 PM IST (closing auction)
            afterhours_close = time(16, 0)
            
            # Check regular hours
            if market_open <= current_time <= market_close:
                return True
            
            # Check pre-market if enabled
            if self.trading.enable_premarket and premarket_open <= current_time < market_open:
                return True
            
            # Check after-hours if enabled
            if self.trading.enable_afterhours and market_close < current_time <= afterhours_close:
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Error checking trading hours: {e}")
            return False
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"""
Configuration Summary:
- Risk: Max {self.risk.max_risk_per_trade}% per trade, {self.risk.max_portfolio_risk}% portfolio
- Trading: {self.trading.strategy_name} on {self.trading.timeframe} timeframe
- Mode: {'Paper Trading' if self.trading.paper_trading else 'Live Trading'}
- Max Positions: {self.risk.max_positions}
- API Rate Limit: {self.api.rate_limit_requests} requests per minute
"""

# Global configuration instance
config = ConfigManager()

def get_config() -> ConfigManager:
    """Get the global configuration instance"""
    return config
