#!/usr/bin/env python3
"""
Common Utilities
Shared utility functions used across the trading bot system
"""

import os
import json
import logging
import hashlib
import hmac
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

logger = logging.getLogger(__name__)

class TimeUtils:
    """Time-related utility functions"""
    
    @staticmethod
    def get_market_session() -> str:
        """Get current market session"""
        now = datetime.now().time()
        
        # Indian market timings
        pre_market_start = datetime.strptime('09:00', '%H:%M').time()
        market_open = datetime.strptime('09:15', '%H:%M').time()
        market_close = datetime.strptime('15:30', '%H:%M').time()
        post_market_end = datetime.strptime('16:00', '%H:%M').time()
        
        if now < pre_market_start:
            return 'PRE_MARKET_CLOSED'
        elif pre_market_start <= now < market_open:
            return 'PRE_MARKET'
        elif market_open <= now <= market_close:
            return 'MARKET_OPEN'
        elif market_close < now <= post_market_end:
            return 'POST_MARKET'
        else:
            return 'MARKET_CLOSED'
    
    @staticmethod
    def is_market_open() -> bool:
        """Check if market is currently open"""
        return TimeUtils.get_market_session() == 'MARKET_OPEN'
    
    @staticmethod
    def get_market_open_time() -> datetime:
        """Get today's market open time"""
        today = datetime.now().date()
        return datetime.combine(today, datetime.strptime('09:15', '%H:%M').time())
    
    @staticmethod
    def get_market_close_time() -> datetime:
        """Get today's market close time"""
        today = datetime.now().date()
        return datetime.combine(today, datetime.strptime('15:30', '%H:%M').time())
    
    @staticmethod
    def get_time_to_market_close() -> timedelta:
        """Get time remaining until market close"""
        close_time = TimeUtils.get_market_close_time()
        now = datetime.now()
        
        if now > close_time:
            # Market already closed, return time to next day's close
            tomorrow_close = close_time + timedelta(days=1)
            return tomorrow_close - now
        else:
            return close_time - now
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    @staticmethod
    def get_trading_days_between(start_date: datetime, end_date: datetime) -> int:
        """Get number of trading days between two dates (excluding weekends)"""
        days = 0
        current = start_date.date()
        end = end_date.date()
        
        while current <= end:
            # Skip weekends (Saturday=5, Sunday=6)
            if current.weekday() < 5:
                days += 1
            current += timedelta(days=1)
        
        return days


class MathUtils:
    """Mathematical utility functions"""
    
    @staticmethod
    def round_to_tick_size(price: float, tick_size: float = 0.05) -> float:
        """Round price to nearest tick size"""
        return round(price / tick_size) * tick_size
    
    @staticmethod
    def calculate_percentage_change(old_value: float, new_value: float) -> float:
        """Calculate percentage change"""
        if old_value == 0:
            return 0.0
        return ((new_value - old_value) / old_value) * 100
    
    @staticmethod
    def calculate_compound_return(returns: List[float]) -> float:
        """Calculate compound return from list of returns"""
        if not returns:
            return 0.0
        
        compound = 1.0
        for ret in returns:
            compound *= (1 + ret / 100)
        
        return (compound - 1) * 100
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = [r - risk_free_rate for r in returns]
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return
    
    @staticmethod
    def calculate_max_drawdown(cumulative_returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not cumulative_returns:
            return 0.0
        
        peak = cumulative_returns[0]
        max_dd = 0.0
        
        for value in cumulative_returns:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak if peak != 0 else 0
            max_dd = max(max_dd, drawdown)
        
        return max_dd * 100
    
    @staticmethod
    def calculate_position_size(account_balance: float, risk_per_trade: float, 
                              entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk management"""
        if entry_price <= 0 or stop_loss <= 0 or entry_price == stop_loss:
            return 0
        
        risk_amount = account_balance * (risk_per_trade / 100)
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        position_size = int(risk_amount / price_risk)
        return max(1, position_size)


class StringUtils:
    """String utility functions"""
    
    @staticmethod
    def generate_unique_id() -> str:
        """Generate unique ID"""
        return str(uuid.uuid4())
    
    @staticmethod
    def generate_order_id(prefix: str = "ORD") -> str:
        """Generate unique order ID"""
        timestamp = int(time.time() * 1000)  # milliseconds
        return f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}"
    
    @staticmethod
    def sanitize_symbol(symbol: str) -> str:
        """Sanitize trading symbol"""
        return symbol.upper().strip()
    
    @staticmethod
    def format_currency(amount: float, currency: str = "INR") -> str:
        """Format currency amount"""
        if currency == "INR":
            return f"₹{amount:,.2f}"
        else:
            return f"{amount:,.2f} {currency}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 2) -> str:
        """Format percentage value"""
        return f"{value:+.{decimals}f}%"
    
    @staticmethod
    def truncate_string(text: str, max_length: int = 50) -> str:
        """Truncate string to max length"""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."


class SecurityUtils:
    """Security utility functions"""
    
    @staticmethod
    def generate_api_signature(data: Dict, secret_key: str) -> str:
        """Generate HMAC signature for API requests"""
        # Sort data by keys
        sorted_data = dict(sorted(data.items()))
        
        # Create query string
        query_string = "&".join([f"{k}={v}" for k, v in sorted_data.items()])
        
        # Generate HMAC SHA256 signature
        signature = hmac.new(
            secret_key.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    @staticmethod
    def hash_password(password: str, salt: str = None) -> Tuple[str, str]:
        """Hash password with salt"""
        if salt is None:
            salt = os.urandom(32).hex()
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            100000  # iterations
        ).hex()
        
        return password_hash, salt
    
    @staticmethod
    def verify_password(password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash"""
        computed_hash, _ = SecurityUtils.hash_password(password, salt)
        return computed_hash == password_hash
    
    @staticmethod
    def sanitize_input(input_string: str) -> str:
        """Sanitize user input"""
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '|', '`']
        sanitized = input_string
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized.strip()


class FileUtils:
    """File utility functions"""
    
    @staticmethod
    def ensure_directory(directory_path: Union[str, Path]) -> bool:
        """Ensure directory exists"""
        try:
            Path(directory_path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"❌ Create directory error: {e}")
            return False
    
    @staticmethod
    def read_json_file(file_path: Union[str, Path]) -> Optional[Dict]:
        """Read JSON file safely"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Read JSON file error: {e}")
            return None
    
    @staticmethod
    def write_json_file(file_path: Union[str, Path], data: Dict) -> bool:
        """Write JSON file safely"""
        try:
            # Ensure directory exists
            FileUtils.ensure_directory(Path(file_path).parent)
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"❌ Write JSON file error: {e}")
            return False
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> Optional[int]:
        """Get file size in bytes"""
        try:
            return Path(file_path).stat().st_size
        except Exception:
            return None
    
    @staticmethod
    def backup_file(file_path: Union[str, Path], backup_dir: Union[str, Path] = None) -> bool:
        """Create backup of file"""
        try:
            import shutil
            
            source_path = Path(file_path)
            if not source_path.exists():
                return False
            
            if backup_dir is None:
                backup_dir = source_path.parent / "backups"
            
            FileUtils.ensure_directory(backup_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{source_path.stem}_{timestamp}{source_path.suffix}"
            backup_path = Path(backup_dir) / backup_name
            
            shutil.copy2(source_path, backup_path)
            return True
            
        except Exception as e:
            logger.error(f"❌ Backup file error: {e}")
            return False


class DataUtils:
    """Data processing utility functions"""
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame by removing NaN and invalid values"""
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    @staticmethod
    def calculate_technical_indicators(prices: List[float], period: int = 14) -> Dict:
        """Calculate basic technical indicators"""
        if len(prices) < period:
            return {}
        
        prices_array = np.array(prices)
        
        # Simple Moving Average
        sma = np.mean(prices_array[-period:])
        
        # Exponential Moving Average
        multiplier = 2 / (period + 1)
        ema = prices_array[0]
        for price in prices_array[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        # RSI calculation
        deltas = np.diff(prices_array)
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = np.abs(losses)
        
        avg_gain = np.mean(gains[-period:]) if len(gains) >= period else 0
        avg_loss = np.mean(losses[-period:]) if len(losses) >= period else 0
        
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs)) if rs != 0 else 50
        
        # Bollinger Bands
        bb_period = min(20, len(prices_array))
        bb_sma = np.mean(prices_array[-bb_period:])
        bb_std = np.std(prices_array[-bb_period:])
        bb_upper = bb_sma + (2 * bb_std)
        bb_lower = bb_sma - (2 * bb_std)
        
        return {
            'sma': round(sma, 2),
            'ema': round(ema, 2),
            'rsi': round(rsi, 2),
            'bb_upper': round(bb_upper, 2),
            'bb_middle': round(bb_sma, 2),
            'bb_lower': round(bb_lower, 2),
            'current_price': round(prices_array[-1], 2)
        }
    
    @staticmethod
    def resample_ohlc_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLC data to different timeframe"""
        try:
            if 'timestamp' not in df.columns:
                return df
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Resample based on timeframe
            resampled = df.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            # Remove NaN rows
            resampled = resampled.dropna()
            
            return resampled.reset_index()
            
        except Exception as e:
            logger.error(f"❌ Resample OHLC error: {e}")
            return df
    
    @staticmethod
    def calculate_correlation(series1: List[float], series2: List[float]) -> float:
        """Calculate correlation between two series"""
        try:
            if len(series1) != len(series2) or len(series1) < 2:
                return 0.0
            
            correlation = np.corrcoef(series1, series2)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0


class ValidationUtils:
    """Data validation utility functions"""
    
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """Validate trading symbol format"""
        if not symbol or not isinstance(symbol, str):
            return False
        
        # Basic validation: alphanumeric, length 3-20
        symbol = symbol.strip().upper()
        return (
            symbol.isalnum() and 
            3 <= len(symbol) <= 20 and
            not symbol.isdigit()  # Not all numbers
        )
    
    @staticmethod
    def validate_price(price: float) -> bool:
        """Validate price value"""
        return isinstance(price, (int, float)) and price > 0 and price < 1000000
    
    @staticmethod
    def validate_quantity(quantity: int) -> bool:
        """Validate quantity value"""
        return isinstance(quantity, int) and quantity > 0 and quantity <= 100000
    
    @staticmethod
    def validate_percentage(value: float, min_val: float = -100, max_val: float = 100) -> bool:
        """Validate percentage value"""
        return isinstance(value, (int, float)) and min_val <= value <= max_val
    
    @staticmethod
    def validate_order_type(order_type: str) -> bool:
        """Validate order type"""
        valid_types = ['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT']
        return order_type.upper() in valid_types
    
    @staticmethod
    def validate_order_side(side: str) -> bool:
        """Validate order side"""
        valid_sides = ['BUY', 'SELL']
        return side.upper() in valid_sides
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email)) if email else False
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Validate phone number (Indian format)"""
        import re
        # Indian phone number pattern
        pattern = r'^[6-9]\d{9}$'
        return bool(re.match(pattern, phone.replace(' ', '').replace('-', ''))) if phone else False


class LoggingUtils:
    """Logging utility functions"""
    
    @staticmethod
    def setup_logger(name: str, log_file: str = None, level: str = 'INFO') -> logging.Logger:
        """Setup logger with file and console handlers"""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            FileUtils.ensure_directory(Path(log_file).parent)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def log_trade_execution(logger: logging.Logger, symbol: str, side: str, 
                          quantity: int, price: float, order_id: str):
        """Log trade execution with standardized format"""
        logger.info(
            f"🔄 TRADE EXECUTED | Symbol: {symbol} | Side: {side} | "
            f"Qty: {quantity} | Price: ₹{price:.2f} | Order: {order_id}"
        )
    
    @staticmethod
    def log_signal_generation(logger: logging.Logger, symbol: str, strategy: str, 
                            signal_type: str, confidence: float):
        """Log signal generation with standardized format"""
        logger.info(
            f"📊 SIGNAL GENERATED | Symbol: {symbol} | Strategy: {strategy} | "
            f"Signal: {signal_type} | Confidence: {confidence:.1f}%"
        )
    
    @staticmethod
    def log_error_with_context(logger: logging.Logger, error: Exception, 
                             context: Dict = None):
        """Log error with additional context"""
        error_msg = f"❌ ERROR: {str(error)}"
        if context:
            error_msg += f" | Context: {json.dumps(context, default=str)}"
        
        logger.error(error_msg, exc_info=True)


class PerformanceUtils:
    """Performance monitoring utility functions"""
    
    @staticmethod
    def time_function(func):
        """Decorator to time function execution"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            logger.debug(f"⏱️ {func.__name__} executed in {end_time - start_time:.4f}s")
            return result
        return wrapper
    
    @staticmethod
    def measure_memory_usage():
        """Measure current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': round(memory_info.rss / (1024 * 1024), 2),
                'vms_mb': round(memory_info.vms / (1024 * 1024), 2),
                'percent': round(process.memory_percent(), 2)
            }
        except Exception:
            return {}
    
    @staticmethod
    def profile_code_block(description: str = "Code block"):
        """Context manager to profile code execution"""
        class ProfileContext:
            def __init__(self, desc):
                self.description = desc
                self.start_time = None
                self.start_memory = None
            
            def __enter__(self):
                self.start_time = time.time()
                self.start_memory = PerformanceUtils.measure_memory_usage()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.time()
                end_memory = PerformanceUtils.measure_memory_usage()
                
                duration = end_time - self.start_time
                memory_change = (
                    end_memory.get('rss_mb', 0) - self.start_memory.get('rss_mb', 0)
                ) if self.start_memory and end_memory else 0
                
                logger.debug(
                    f"📈 PROFILE | {self.description} | "
                    f"Time: {duration:.4f}s | Memory: {memory_change:+.2f}MB"
                )
        
        return ProfileContext(description)


class ConfigurationUtils:
    """Configuration utility functions"""
    
    @staticmethod
    def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
        """Merge two configuration dictionaries"""
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigurationUtils.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    @staticmethod
    def validate_config_structure(config: Dict, required_keys: List[str]) -> Tuple[bool, List[str]]:
        """Validate configuration structure"""
        missing_keys = []
        
        def check_nested_key(cfg: Dict, key_path: str):
            keys = key_path.split('.')
            current = cfg
            
            for key in keys:
                if not isinstance(current, dict) or key not in current:
                    return False
                current = current[key]
            
            return True
        
        for key in required_keys:
            if not check_nested_key(config, key):
                missing_keys.append(key)
        
        return len(missing_keys) == 0, missing_keys
    
    @staticmethod
    def get_environment_variable(var_name: str, default_value: Any = None) -> Any:
        """Get environment variable with default value"""
        value = os.getenv(var_name, default_value)
        
        # Try to convert to appropriate type
        if isinstance(default_value, bool):
            return value.lower() in ('true', '1', 'yes', 'on') if isinstance(value, str) else bool(value)
        elif isinstance(default_value, int):
            try:
                return int(value)
            except (ValueError, TypeError):
                return default_value
        elif isinstance(default_value, float):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default_value
        
        return value


class NetworkUtils:
    """Network utility functions"""
    
    @staticmethod
    def is_port_available(port: int, host: str = 'localhost') -> bool:
        """Check if port is available"""
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                return result != 0  # Port is available if connection fails
        except Exception:
            return False
    
    @staticmethod
    def get_local_ip() -> str:
        """Get local IP address"""
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                return sock.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    @staticmethod
    def test_internet_connection(timeout: int = 5) -> bool:
        """Test internet connectivity"""
        try:
            import urllib.request
            urllib.request.urlopen('http://google.com', timeout=timeout)
            return True
        except Exception:
            return False


# Global utility functions
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value for zero division"""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to integer"""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def chunks(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def retry_on_exception(max_attempts: int = 3, delay: float = 1.0):
    """Decorator for retrying function on exception"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"⚠️ Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error(f"❌ All {max_attempts} attempts failed for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator


# Usage examples and testing
if __name__ == "__main__":
    # Test utility functions
    print("🔧 Testing Utility Functions...")
    
    # Test time utils
    market_session = TimeUtils.get_market_session()
    print(f"📅 Current market session: {market_session}")
    
    is_open = TimeUtils.is_market_open()
    print(f"🏦 Market is open: {is_open}")
    
    # Test math utils
    returns = [1.5, -0.8, 2.1, -1.2, 0.9]
    sharpe = MathUtils.calculate_sharpe_ratio(returns)
    print(f"📊 Sharpe ratio: {sharpe:.3f}")
    
    # Test string utils
    order_id = StringUtils.generate_order_id("TEST")
    print(f"🆔 Generated order ID: {order_id}")
    
    formatted_amount = StringUtils.format_currency(12345.67)
    print(f"💰 Formatted amount: {formatted_amount}")
    
    # Test validation utils
    valid_symbol = ValidationUtils.validate_symbol("RELIANCE")
    print(f"✅ RELIANCE is valid symbol: {valid_symbol}")
    
    valid_price = ValidationUtils.validate_price(2500.75)
    print(f"✅ 2500.75 is valid price: {valid_price}")
    
    # Test data utils
    prices = [100, 102, 98, 105, 103, 99, 107, 106, 104, 108, 110, 109, 112, 115, 113]
    indicators = DataUtils.calculate_technical_indicators(prices)
    print(f"📈 Technical indicators: RSI={indicators.get('rsi', 0):.1f}")
    
    # Test performance measurement
    memory_usage = PerformanceUtils.measure_memory_usage()
    print(f"💾 Memory usage: {memory_usage.get('rss_mb', 0)} MB")
    
    # Test network utils
    local_ip = NetworkUtils.get_local_ip()
    print(f"🌐 Local IP: {local_ip}")
    
    port_available = NetworkUtils.is_port_available(8080)
    print(f"🔌 Port 8080 available: {port_available}")
    
    print("✅ Utility functions test completed")
