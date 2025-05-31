#!/usr/bin/env python3
"""
MARKET-READY Data Manager - Central Data Processing Hub
Handles dual-pipeline architecture with real-time filtering and validation
Compatible with live NSE market data and dual-database storage
"""

import asyncio
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
from collections import defaultdict, deque
import math

# Import system components with fallback handling
try:
    from config_manager import ConfigManager, get_config
    from utils import Logger, ErrorHandler, DataValidator
    from security_manager import SecurityManager
    from database_setup import DatabaseManager, get_database_manager
    from websocket_manager import MarketDataMessage, MessageType, DataFeed
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    
    # Fallback implementations for immediate deployment
    class Logger:
        def __init__(self, name): 
            self.logger = logging.getLogger(name)
        def info(self, msg): 
            self.logger.info(msg)
        def error(self, msg): 
            self.logger.error(msg)
        def warning(self, msg): 
            self.logger.warning(msg)
        def debug(self, msg): 
            self.logger.debug(msg)

    class ErrorHandler:
        def handle_error(self, e, context): 
            logging.error(f"{context}: {e}")

    class DataValidator:
        def validate_price(self, price): 
            return isinstance(price, (int, float)) and price > 0
        def validate_volume(self, volume):
            return isinstance(volume, int) and volume >= 0
        def validate_symbol(self, symbol):
            return isinstance(symbol, str) and len(symbol) > 0

    class ConfigManager:
        def get_config(self):
            return {
                'data_processing': {
                    'min_price': 1.0,
                    'max_price': 100000.0,
                    'min_volume': 1000,
                    'max_symbols_sqlite': 50,
                    'price_change_threshold': 0.15,
                    'volume_spike_threshold': 2.0,
                    'processing_batch_size': 100
                },
                'database': {
                    'postgresql_enabled': True,
                    'sqlite_enabled': True,
                    'batch_insert_size': 500
                }
            }
    
    def get_config():
        return ConfigManager()
    
    # Mock database manager for standalone operation
    class DatabaseManager:
        def __init__(self):
            pass
        
        async def store_market_data_postgresql(self, data):
            pass
        
        async def store_filtered_data_sqlite(self, data):
            pass
        
        async def get_symbol_metadata(self, symbol):
            return {'sector': 'Unknown', 'market_cap': 'Unknown'}
    
    def get_database_manager():
        return DatabaseManager()

    # Mock market data message for standalone operation
    @dataclass
    class MarketDataMessage:
        symbol: str
        message_type: str
        data: Dict[str, Any]
        timestamp: datetime
        source: str

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataQuality(Enum):
    """Market data quality levels"""
    EXCELLENT = "EXCELLENT"      # Perfect data, ready for trading
    GOOD = "GOOD"               # Minor issues, usable for trading
    ACCEPTABLE = "ACCEPTABLE"   # Some issues, use with caution
    POOR = "POOR"              # Significant issues, filter out
    INVALID = "INVALID"        # Corrupt data, reject completely

class SymbolCategory(Enum):
    """Symbol categorization for filtering"""
    LARGE_CAP = "LARGE_CAP"           # > 20,000 crores
    MID_CAP = "MID_CAP"               # 5,000 - 20,000 crores
    SMALL_CAP = "SMALL_CAP"           # 500 - 5,000 crores
    MICRO_CAP = "MICRO_CAP"           # < 500 crores
    HIGH_VOLUME = "HIGH_VOLUME"       # High liquidity
    BREAKOUT_CANDIDATE = "BREAKOUT"   # Technical breakout pattern
    MOMENTUM = "MOMENTUM"             # Strong momentum
    SCALPING_SUITABLE = "SCALPING"    # Perfect for scalping

@dataclass
class MarketDataPoint:
    """Standardized market data structure"""
    symbol: str
    exchange: str
    timestamp: datetime
    last_price: float
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    value: float
    bid_price: float
    ask_price: float
    bid_qty: int
    ask_qty: int
    change: float
    change_percent: float
    average_price: float
    total_buy_qty: int = 0
    total_sell_qty: int = 0
    quality: MarketDataQuality = MarketDataQuality.GOOD
    category: Set[SymbolCategory] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timestamp': self.timestamp.isoformat(),
            'last_price': self.last_price,
            'open_price': self.open_price,
            'high_price': self.high_price,
            'low_price': self.low_price,
            'close_price': self.close_price,
            'volume': self.volume,
            'value': self.value,
            'bid_price': self.bid_price,
            'ask_price': self.ask_price,
            'bid_qty': self.bid_qty,
            'ask_qty': self.ask_qty,
            'change': self.change,
            'change_percent': self.change_percent,
            'average_price': self.average_price,
            'total_buy_qty': self.total_buy_qty,
            'total_sell_qty': self.total_sell_qty,
            'quality': self.quality.value,
            'categories': [cat.value for cat in self.category]
        }

@dataclass
class FilteringCriteria:
    """Criteria for filtering symbols for real-time trading"""
    min_price: float = 10.0
    max_price: float = 5000.0
    min_volume: int = 100000
    min_value: float = 1000000.0  # 10 lakh rupees
    max_spread_percent: float = 0.5
    min_change_percent: float = 0.5
    max_change_percent: float = 15.0
    required_categories: Set[SymbolCategory] = field(default_factory=lambda: {
        SymbolCategory.HIGH_VOLUME,
        SymbolCategory.SCALPING_SUITABLE
    })

class DataQualityAnalyzer:
    """Analyzes and validates market data quality"""
    
    def __init__(self):
        self.logger = Logger(__name__)
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self.last_update: Dict[str, datetime] = {}
    
    def analyze_data_quality(self, data_point: MarketDataPoint) -> MarketDataQuality:
        """Comprehensive data quality analysis"""
        try:
            issues = []
            
            # Price validation
            if not self._validate_price_data(data_point):
                issues.append("price_invalid")
            
            # Volume validation
            if not self._validate_volume_data(data_point):
                issues.append("volume_invalid")
            
            # Timestamp validation
            if not self._validate_timestamp(data_point):
                issues.append("timestamp_stale")
            
            # Cross-validation with history
            if not self._validate_against_history(data_point):
                issues.append("historical_anomaly")
            
            # Market structure validation
            if not self._validate_market_structure(data_point):
                issues.append("market_structure_invalid")
            
            # Determine quality based on issues
            if not issues:
                return MarketDataQuality.EXCELLENT
            elif len(issues) == 1 and "timestamp_stale" in issues:
                return MarketDataQuality.GOOD
            elif len(issues) <= 2:
                return MarketDataQuality.ACCEPTABLE
            elif len(issues) <= 3:
                return MarketDataQuality.POOR
            else:
                return MarketDataQuality.INVALID
                
        except Exception as e:
            self.logger.error(f"Data quality analysis error for {data_point.symbol}: {e}")
            return MarketDataQuality.POOR
    
    def _validate_price_data(self, data_point: MarketDataPoint) -> bool:
        """Validate price-related data"""
        try:
            # Basic price validation
            if data_point.last_price <= 0:
                return False
            
            # OHLC validation
            if data_point.open_price > 0:
                if not (data_point.low_price <= data_point.open_price <= data_point.high_price):
                    return False
                if not (data_point.low_price <= data_point.close_price <= data_point.high_price):
                    return False
                if not (data_point.low_price <= data_point.last_price <= data_point.high_price):
                    return False
            
            # Bid-Ask validation
            if data_point.bid_price > 0 and data_point.ask_price > 0:
                if data_point.bid_price >= data_point.ask_price:
                    return False
                
                # Spread validation (should be reasonable)
                spread_percent = ((data_point.ask_price - data_point.bid_price) / data_point.last_price) * 100
                if spread_percent > 5.0:  # 5% spread is suspicious
                    return False
            
            # Circuit breaker validation (NSE has 5%, 10%, 20% limits)
            if data_point.close_price > 0 and abs(data_point.change_percent) > 25:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Price validation error: {e}")
            return False
    
    def _validate_volume_data(self, data_point: MarketDataPoint) -> bool:
        """Validate volume-related data"""
        try:
            # Basic volume validation
            if data_point.volume < 0:
                return False
            
            # Value validation
            expected_value = data_point.volume * data_point.average_price
            if data_point.value > 0 and abs(data_point.value - expected_value) / expected_value > 0.1:
                return False
            
            # Bid-Ask quantity validation
            if data_point.bid_qty < 0 or data_point.ask_qty < 0:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Volume validation error: {e}")
            return False
    
    def _validate_timestamp(self, data_point: MarketDataPoint) -> bool:
        """Validate data timestamp"""
        try:
            current_time = datetime.now()
            
            # Data should not be from future
            if data_point.timestamp > current_time + timedelta(minutes=1):
                return False
            
            # Data should not be too stale (more than 5 minutes old)
            if current_time - data_point.timestamp > timedelta(minutes=5):
                return False
            
            # Check for reasonable update frequency
            last_update = self.last_update.get(data_point.symbol)
            if last_update:
                time_diff = (data_point.timestamp - last_update).total_seconds()
                if time_diff < 0.1:  # Updates too frequent (less than 100ms)
                    return False
            
            self.last_update[data_point.symbol] = data_point.timestamp
            return True
            
        except Exception as e:
            self.logger.error(f"Timestamp validation error: {e}")
            return False
    
    def _validate_against_history(self, data_point: MarketDataPoint) -> bool:
        """Validate against historical data"""
        try:
            symbol = data_point.symbol
            
            # Add to history
            self.price_history[symbol].append(data_point.last_price)
            self.volume_history[symbol].append(data_point.volume)
            
            # Need at least 3 data points for validation
            if len(self.price_history[symbol]) < 3:
                return True
            
            # Price change validation
            prev_prices = list(self.price_history[symbol])[-3:-1]
            current_price = data_point.last_price
            
            # Check for unrealistic price jumps
            for prev_price in prev_prices:
                if prev_price > 0:
                    change_percent = abs((current_price - prev_price) / prev_price) * 100
                    if change_percent > 20:  # 20% jump is suspicious in short term
                        return False
            
            # Volume validation against history
            prev_volumes = list(self.volume_history[symbol])[-5:]
            if len(prev_volumes) >= 3:
                avg_volume = statistics.mean(prev_volumes[:-1])
                if avg_volume > 0:
                    volume_ratio = data_point.volume / avg_volume
                    if volume_ratio > 50:  # 50x volume spike is suspicious
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Historical validation error: {e}")
            return True  # Don't reject due to validation errors
    
    def _validate_market_structure(self, data_point: MarketDataPoint) -> bool:
        """Validate market microstructure"""
        try:
            # Validate that last price is reasonable compared to bid/ask
            if data_point.bid_price > 0 and data_point.ask_price > 0:
                if not (data_point.bid_price <= data_point.last_price <= data_point.ask_price):
                    # Allow small deviation due to timing
                    mid_price = (data_point.bid_price + data_point.ask_price) / 2
                    deviation = abs(data_point.last_price - mid_price) / mid_price * 100
                    if deviation > 2:  # 2% deviation is too much
                        return False
            
            # Validate change calculation
            if data_point.close_price > 0:
                expected_change = data_point.last_price - data_point.close_price
                if abs(data_point.change - expected_change) > 0.01:
                    return False
                
                expected_change_percent = (expected_change / data_point.close_price) * 100
                if abs(data_point.change_percent - expected_change_percent) > 0.01:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Market structure validation error: {e}")
            return True

class SymbolCategorizer:
    """Categorizes symbols based on market data characteristics"""
    
    def __init__(self):
        self.logger = Logger(__name__)
        self.volume_percentiles: Dict[str, float] = {}
        self.price_percentiles: Dict[str, float] = {}
        self.last_analysis = datetime.now()
    
    def categorize_symbol(self, data_point: MarketDataPoint, 
                         market_stats: Dict[str, Any] = None) -> Set[SymbolCategory]:
        """Categorize symbol based on current market data"""
        try:
            categories = set()
            
            # Price-based categorization (need market cap data for accurate classification)
            categories.update(self._categorize_by_price(data_point))
            
            # Volume-based categorization
            categories.update(self._categorize_by_volume(data_point, market_stats))
            
            # Technical pattern categorization
            categories.update(self._categorize_by_technical_patterns(data_point))
            
            # Liquidity categorization
            categories.update(self._categorize_by_liquidity(data_point))
            
            # Scalping suitability
            if self._is_scalping_suitable(data_point):
                categories.add(SymbolCategory.SCALPING_SUITABLE)
            
            return categories
            
        except Exception as e:
            self.logger.error(f"Symbol categorization error for {data_point.symbol}: {e}")
            return {SymbolCategory.SMALL_CAP}  # Default category
    
    def _categorize_by_price(self, data_point: MarketDataPoint) -> Set[SymbolCategory]:
        """Categorize based on price range (proxy for market cap)"""
        categories = set()
        
        # Using price as proxy for market cap (not perfect but practical)
        if data_point.last_price >= 1000:
            categories.add(SymbolCategory.LARGE_CAP)
        elif data_point.last_price >= 200:
            categories.add(SymbolCategory.MID_CAP)
        elif data_point.last_price >= 50:
            categories.add(SymbolCategory.SMALL_CAP)
        else:
            categories.add(SymbolCategory.MICRO_CAP)
        
        return categories
    
    def _categorize_by_volume(self, data_point: MarketDataPoint, 
                            market_stats: Dict[str, Any] = None) -> Set[SymbolCategory]:
        """Categorize based on volume characteristics"""
        categories = set()
        
        # High volume threshold (can be adjusted based on market stats)
        if data_point.volume >= 500000:  # 5 lakh shares
            categories.add(SymbolCategory.HIGH_VOLUME)
        
        # Value-based volume assessment
        if data_point.value >= 50000000:  # 5 crore rupees
            categories.add(SymbolCategory.HIGH_VOLUME)
        
        return categories
    
    def _categorize_by_technical_patterns(self, data_point: MarketDataPoint) -> Set[SymbolCategory]:
        """Categorize based on technical patterns"""
        categories = set()
        
        # Momentum detection
        if abs(data_point.change_percent) >= 2.0:
            categories.add(SymbolCategory.MOMENTUM)
        
        # Breakout detection (simplified)
        if data_point.high_price > 0 and data_point.low_price > 0:
            day_range_percent = ((data_point.high_price - data_point.low_price) / data_point.low_price) * 100
            if day_range_percent >= 3.0:
                categories.add(SymbolCategory.BREAKOUT_CANDIDATE)
        
        return categories
    
    def _categorize_by_liquidity(self, data_point: MarketDataPoint) -> Set[SymbolCategory]:
        """Categorize based on liquidity metrics"""
        categories = set()
        
        # Bid-Ask spread analysis
        if data_point.bid_price > 0 and data_point.ask_price > 0:
            spread_percent = ((data_point.ask_price - data_point.bid_price) / data_point.last_price) * 100
            
            if spread_percent <= 0.1:  # Very tight spread
                categories.add(SymbolCategory.HIGH_VOLUME)
            
            # Depth analysis
            total_depth = data_point.bid_qty + data_point.ask_qty
            if total_depth >= 10000:  # Good depth
                categories.add(SymbolCategory.HIGH_VOLUME)
        
        return categories
    
    def _is_scalping_suitable(self, data_point: MarketDataPoint) -> bool:
        """Determine if symbol is suitable for scalping"""
        try:
            # Criteria for scalping suitability
            criteria_met = 0
            total_criteria = 6
            
            # 1. Sufficient volume
            if data_point.volume >= 100000:
                criteria_met += 1
            
            # 2. Sufficient value
            if data_point.value >= 5000000:  # 50 lakh rupees
                criteria_met += 1
            
            # 3. Reasonable price range
            if 20 <= data_point.last_price <= 2000:
                criteria_met += 1
            
            # 4. Tight spread
            if data_point.bid_price > 0 and data_point.ask_price > 0:
                spread_percent = ((data_point.ask_price - data_point.bid_price) / data_point.last_price) * 100
                if spread_percent <= 0.2:
                    criteria_met += 1
            
            # 5. Some volatility (but not too much)
            if 0.5 <= abs(data_point.change_percent) <= 8.0:
                criteria_met += 1
            
            # 6. Good depth
            total_depth = data_point.bid_qty + data_point.ask_qty
            if total_depth >= 5000:
                criteria_met += 1
            
            # Need at least 4 out of 6 criteria
            return criteria_met >= 4
            
        except Exception as e:
            self.logger.error(f"Scalping suitability check error: {e}")
            return False

class DataManager:
    """Central data processing hub with dual-pipeline architecture"""
    
    def __init__(self, config_manager=None, database_manager=None, security_manager=None):
        # Initialize configuration
        self.config_manager = config_manager or get_config()
        self.database_manager = database_manager or get_database_manager()
        self.security_manager = security_manager
        
        try:
            self.config = self.config_manager.get_config()
            self.data_config = self.config.get('data_processing', {})
            self.db_config = self.config.get('database', {})
        except:
            # Fallback configuration
            self.data_config = {
                'min_price': 1.0,
                'max_price': 100000.0,
                'min_volume': 1000,
                'max_symbols_sqlite': 50,
                'price_change_threshold': 0.15,
                'volume_spike_threshold': 2.0,
                'processing_batch_size': 100
            }
            self.db_config = {
                'postgresql_enabled': True,
                'sqlite_enabled': True,
                'batch_insert_size': 500
            }
        
        # Initialize utilities
        self.logger = Logger(__name__)
        self.error_handler = ErrorHandler()
        self.data_validator = DataValidator()
        self.quality_analyzer = DataQualityAnalyzer()
        self.symbol_categorizer = SymbolCategorizer()
        
        # Data processing state
        self.processed_symbols: Set[str] = set()
        self.filtered_symbols: Dict[str, MarketDataPoint] = {}
        self.symbol_metadata: Dict[str, Dict] = {}
        self.processing_stats = {
            'total_processed': 0,
            'total_filtered': 0,
            'quality_distribution': defaultdict(int),
            'categories_distribution': defaultdict(int),
            'last_update': datetime.now()
        }
        
        # Pipeline control
        self.postgresql_pipeline_active = self.db_config.get('postgresql_enabled', True)
        self.sqlite_pipeline_active = self.db_config.get('sqlite_enabled', True)
        
        # Filtering criteria
        self.filtering_criteria = FilteringCriteria(
            min_price=self.data_config.get('min_price', 10.0),
            max_price=self.data_config.get('max_price', 5000.0),
            min_volume=self.data_config.get('min_volume', 100000),
            min_value=self.data_config.get('min_value', 1000000.0),
            max_spread_percent=self.data_config.get('max_spread_percent', 0.5),
            min_change_percent=self.data_config.get('min_change_percent', 0.5),
            max_change_percent=self.data_config.get('max_change_percent', 15.0)
        )
        
        # Batch processing
        self.postgresql_batch: List[MarketDataPoint] = []
        self.sqlite_batch: List[MarketDataPoint] = []
        self.batch_size = self.db_config.get('batch_insert_size', 500)
        
        # Performance tracking
        self.performance_metrics = {
            'processing_rate': 0.0,
            'filtering_rate': 0.0,
            'postgresql_write_rate': 0.0,
            'sqlite_write_rate': 0.0,
            'last_performance_update': time.time()
        }
        
        self.logger.info("Data Manager initialized with dual-pipeline architecture")
    
    async def initialize(self) -> bool:
        """Initialize data manager and start processing pipelines"""
        try:
            # Load symbol metadata
            await self._load_symbol_metadata()
            
            # Start background tasks
            asyncio.create_task(self._postgresql_pipeline_worker())
            asyncio.create_task(self._sqlite_pipeline_worker())
            asyncio.create_task(self._performance_monitor())
            asyncio.create_task(self._cleanup_worker())
            
            self.logger.info("Data Manager pipelines started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Data Manager initialization failed: {e}")
            self.error_handler.handle_error(e, "data_manager_init")
            return False
    
    async def process_market_data(self, market_message: MarketDataMessage) -> bool:
        """Main entry point for processing market data"""
        try:
            # Convert to standardized format
            data_point = await self._convert_to_data_point(market_message)
            if not data_point:
                return False
            
            # Analyze data quality
            data_point.quality = self.quality_analyzer.analyze_data_quality(data_point)
            
            # Skip invalid data
            if data_point.quality == MarketDataQuality.INVALID:
                self.logger.warning(f"Invalid data rejected for {data_point.symbol}")
                return False
            
            # Categorize symbol
            data_point.category = self.symbol_categorizer.categorize_symbol(data_point)
            
            # Update processing stats
            self._update_processing_stats(data_point)
            
            # Pipeline 1: All symbols → PostgreSQL (background)
            if self.postgresql_pipeline_active:
                await self._queue_for_postgresql(data_point)
            
            # Pipeline 2: Filter and queue for SQLite (real-time)
            if self.sqlite_pipeline_active:
                if await self._should_filter_for_sqlite(data_point):
                    await self._queue_for_sqlite(data_point)
                    self.filtered_symbols[data_point.symbol] = data_point
            
            self.processed_symbols.add(data_point.symbol)
            return True
            
        except Exception as e:
            self.logger.error(f"Market data processing error: {e}")
            self.error_handler.handle_error(e, "market_data_processing")
            return False
    
    async def _convert_to_data_point(self, market_message: MarketDataMessage) -> Optional[MarketDataPoint]:
        """Convert WebSocket message to standardized data point"""
        try:
            data = market_message.data
            
            # Extract symbol information
            symbol = market_message.symbol
            exchange = data.get('exchange', 'NSE')
            
            # Handle different message formats (Goodwill WebSocket format)
            if 'last_price' in data:
                # Standard format
                last_price = float(data.get('last_price', 0))
                volume = int(data.get('volume', 0))
                value = float(data.get('value', last_price * volume))
            elif 'lp' in data:
                # Goodwill format
                last_price = float(data.get('lp', 0))
                volume = int(data.get('v', 0))
                value = float(data.get('value', last_price * volume))
            else:
                self.logger.warning(f"Unknown data format for {symbol}")
                return None
            
            # Basic validation
            if not self.data_validator.validate_price(last_price):
                return None
            if not self.data_validator.validate_volume(volume):
                return None
            
            # Create data point
            data_point = MarketDataPoint(
                symbol=symbol,
                exchange=exchange,
                timestamp=market_message.timestamp,
                last_price=last_price,
                open_price=float(data.get('open_price', data.get('o', 0))),
                high_price=float(data.get('high_price', data.get('h', 0))),
                low_price=float(data.get('low_price', data.get('l', 0))),
                close_price=float(data.get('close_price', data.get('c', 0))),
                volume=volume,
                value=value,
                bid_price=float(data.get('bid_price', data.get('bp1', 0))),
                ask_price=float(data.get('ask_price', data.get('sp1', 0))),
                bid_qty=int(data.get('bid_qty', data.get('bq1', 0))),
                ask_qty=int(data.get('ask_qty', data.get('sq1', 0))),
                change=float(data.get('change', 0)),
                change_percent=float(data.get('change_percent', data.get('pc', 0))),
                average_price=float(data.get('average_price', data.get('ap', last_price))),
                total_buy_qty=int(data.get('total_buy_qty', data.get('tbq', 0))),
                total_sell_qty=int(data.get('total_sell_qty', data.get('tsq', 0)))
            )
            
            return data_point
            
        except Exception as e:
            self.logger.error(f"Data conversion error for {market_message.symbol}: {e}")
            return None
    
    async def _should_filter_for_sqlite(self, data_point: MarketDataPoint) -> bool:
        """Determine if symbol should be filtered for SQLite storage"""
        try:
            criteria = self.filtering_criteria
            
            # Quality check
            if data_point.quality in [MarketDataQuality.POOR, MarketDataQuality.INVALID]:
                return False
            
            # Price range check
            if not (criteria.min_price <= data_point.last_price <= criteria.max_price):
                return False
            
            # Volume check
            if data_point.volume < criteria.min_volume:
                return False
            
            # Value check
            if data_point.value < criteria.min_value:
                return False
            
            # Spread check (if bid/ask available)
            if data_point.bid_price > 0 and data_point.ask_price > 0:
                spread_percent = ((data_point.ask_price - data_point.bid_price) / data_point.last_price) * 100
                if spread_percent > criteria.max_spread_percent:
                    return False
            
            # Change percent check
            change_abs = abs(data_point.change_percent)
            if not (criteria.min_change_percent <= change_abs <= criteria.max_change_percent):
                return False
            
            # Category check
            if not any(cat in data_point.category for cat in criteria.required_categories):
                return False
            
            # Scalping suitability
            if SymbolCategory.SCALPING_SUITABLE not in data_point.category:
                return False
            
            # Check if we're within SQLite limit
            max_symbols = self.data_config.get('max_symbols_sqlite', 50)
            if len(self.filtered_symbols) >= max_symbols:
                # Only accept if it's better than the worst current symbol
                if not await self._is_better_than_worst(data_point):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Filtering check error for {data_point.symbol}: {e}")
            return False
    
    async def _is_better_than_worst(self, new_data_point: MarketDataPoint) -> bool:
        """Check if new symbol is better than the worst in current filtered set"""
        try:
            if not self.filtered_symbols:
                return True
            
            # Calculate score for new data point
            new_score = self._calculate_symbol_score(new_data_point)
            
            # Find worst current symbol
            worst_symbol = None
            worst_score = float('inf')
            
            for symbol, data_point in self.filtered_symbols.items():
                score = self._calculate_symbol_score(data_point)
                if score < worst_score:
                    worst_score = score
                    worst_symbol = symbol
            
            # Replace if new symbol is better
            if new_score > worst_score:
                if worst_symbol:
                    del self.filtered_symbols[worst_symbol]
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Symbol comparison error: {e}")
            return False
    
    def _calculate_symbol_score(self, data_point: MarketDataPoint) -> float:
        """Calculate a score for symbol ranking (higher is better)"""
        try:
            score = 0.0
            
            # Volume score (normalized)
            volume_score = min(data_point.volume / 1000000, 10.0)  # Max 10 points
            score += volume_score
            
            # Value score (normalized)
            value_score = min(data_point.value / 10000000, 10.0)  # Max 10 points
            score += value_score
            
            # Volatility score (moderate volatility is good for scalping)
            volatility = abs(data_point.change_percent)
            if 1.0 <= volatility <= 5.0:
                volatility_score = 10.0
            elif 0.5 <= volatility < 1.0 or 5.0 < volatility <= 8.0:
                volatility_score = 7.0
            elif volatility < 0.5 or volatility > 10.0:
                volatility_score = 2.0
            else:
                volatility_score = 5.0
            score += volatility_score
            
            # Spread score (tighter spread is better)
            if data_point.bid_price > 0 and data_point.ask_price > 0:
                spread_percent = ((data_point.ask_price - data_point.bid_price) / data_point.last_price) * 100
                if spread_percent <= 0.05:
                    spread_score = 10.0
                elif spread_percent <= 0.1:
                    spread_score = 8.0
                elif spread_percent <= 0.2:
                    spread_score = 6.0
                else:
                    spread_score = 3.0
                score += spread_score
            
            # Quality score
            quality_scores = {
                MarketDataQuality.EXCELLENT: 10.0,
                MarketDataQuality.GOOD: 8.0,
                MarketDataQuality.ACCEPTABLE: 5.0,
                MarketDataQuality.POOR: 2.0,
                MarketDataQuality.INVALID: 0.0
            }
            score += quality_scores.get(data_point.quality, 0.0)
            
            # Category bonus
            category_bonuses = {
                SymbolCategory.LARGE_CAP: 3.0,
                SymbolCategory.MID_CAP: 2.0,
                SymbolCategory.HIGH_VOLUME: 5.0,
                SymbolCategory.SCALPING_SUITABLE: 8.0,
                SymbolCategory.MOMENTUM: 4.0,
                SymbolCategory.BREAKOUT_CANDIDATE: 3.0
            }
            
            for category in data_point.category:
                score += category_bonuses.get(category, 0.0)
            
            return score
            
        except Exception as e:
            self.logger.error(f"Score calculation error: {e}")
            return 0.0
    
    async def _queue_for_postgresql(self, data_point: MarketDataPoint):
        """Queue data point for PostgreSQL storage"""
        try:
            self.postgresql_batch.append(data_point)
            
            # Process batch when full
            if len(self.postgresql_batch) >= self.batch_size:
                await self._flush_postgresql_batch()
                
        except Exception as e:
            self.logger.error(f"PostgreSQL queuing error: {e}")
    
    async def _queue_for_sqlite(self, data_point: MarketDataPoint):
        """Queue data point for SQLite storage"""
        try:
            self.sqlite_batch.append(data_point)
            
            # Process batch when full (smaller batches for real-time)
            sqlite_batch_size = min(self.batch_size // 10, 50)
            if len(self.sqlite_batch) >= sqlite_batch_size:
                await self._flush_sqlite_batch()
                
        except Exception as e:
            self.logger.error(f"SQLite queuing error: {e}")
    
    async def _postgresql_pipeline_worker(self):
        """Background worker for PostgreSQL pipeline"""
        try:
            while True:
                await asyncio.sleep(5)  # Process every 5 seconds
                
                if self.postgresql_batch:
                    await self._flush_postgresql_batch()
                
        except asyncio.CancelledError:
            self.logger.info("PostgreSQL pipeline worker cancelled")
        except Exception as e:
            self.logger.error(f"PostgreSQL pipeline worker error: {e}")
    
    async def _sqlite_pipeline_worker(self):
        """Real-time worker for SQLite pipeline"""
        try:
            while True:
                await asyncio.sleep(1)  # Process every second
                
                if self.sqlite_batch:
                    await self._flush_sqlite_batch()
                
        except asyncio.CancelledError:
            self.logger.info("SQLite pipeline worker cancelled")
        except Exception as e:
            self.logger.error(f"SQLite pipeline worker error: {e}")
    
    async def _flush_postgresql_batch(self):
        """Flush batch to PostgreSQL"""
        try:
            if not self.postgresql_batch:
                return
            
            batch_data = [dp.to_dict() for dp in self.postgresql_batch]
            
            # Store in PostgreSQL
            await self.database_manager.store_market_data_postgresql(batch_data)
            
            # Update performance metrics
            self.performance_metrics['postgresql_write_rate'] = len(batch_data) / 5.0  # per second
            
            self.logger.info(f"Stored {len(batch_data)} symbols in PostgreSQL")
            self.postgresql_batch.clear()
            
        except Exception as e:
            self.logger.error(f"PostgreSQL batch flush error: {e}")
            self.error_handler.handle_error(e, "postgresql_batch_flush")
    
    async def _flush_sqlite_batch(self):
        """Flush batch to SQLite"""
        try:
            if not self.sqlite_batch:
                return
            
            batch_data = [dp.to_dict() for dp in self.sqlite_batch]
            
            # Store in SQLite
            await self.database_manager.store_filtered_data_sqlite(batch_data)
            
            # Update performance metrics
            self.performance_metrics['sqlite_write_rate'] = len(batch_data) / 1.0  # per second
            
            self.logger.info(f"Stored {len(batch_data)} filtered symbols in SQLite")
            self.sqlite_batch.clear()
            
        except Exception as e:
            self.logger.error(f"SQLite batch flush error: {e}")
            self.error_handler.handle_error(e, "sqlite_batch_flush")
    
    async def _load_symbol_metadata(self):
        """Load symbol metadata for enhanced categorization"""
        try:
            # This would typically load from a symbol master file or database
            # For now, we'll implement basic metadata
            known_large_caps = [
                'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HDFC', 'ICICIBANK',
                'KOTAKBANK', 'BHARTIARTL', 'ITC', 'SBIN', 'LT', 'ASIANPAINT'
            ]
            
            for symbol in known_large_caps:
                self.symbol_metadata[symbol] = {
                    'sector': 'Unknown',
                    'market_cap': 'Large',
                    'category': SymbolCategory.LARGE_CAP
                }
            
            self.logger.info(f"Loaded metadata for {len(self.symbol_metadata)} symbols")
            
        except Exception as e:
            self.logger.error(f"Metadata loading error: {e}")
    
    def _update_processing_stats(self, data_point: MarketDataPoint):
        """Update processing statistics"""
        try:
            self.processing_stats['total_processed'] += 1
            self.processing_stats['quality_distribution'][data_point.quality.value] += 1
            
            for category in data_point.category:
                self.processing_stats['categories_distribution'][category.value] += 1
            
            if data_point.symbol in self.filtered_symbols:
                self.processing_stats['total_filtered'] += 1
            
            self.processing_stats['last_update'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Stats update error: {e}")
    
    async def _performance_monitor(self):
        """Monitor and log performance metrics"""
        try:
            while True:
                await asyncio.sleep(60)  # Update every minute
                
                current_time = time.time()
                time_diff = current_time - self.performance_metrics['last_performance_update']
                
                if time_diff > 0:
                    # Calculate processing rates
                    self.performance_metrics['processing_rate'] = self.processing_stats['total_processed'] / time_diff
                    self.performance_metrics['filtering_rate'] = self.processing_stats['total_filtered'] / time_diff
                    
                    # Log performance metrics
                    self.logger.info(f"Performance: Processing={self.performance_metrics['processing_rate']:.1f}/s, "
                                   f"Filtering={self.performance_metrics['filtering_rate']:.1f}/s, "
                                   f"PostgreSQL={self.performance_metrics['postgresql_write_rate']:.1f}/s, "
                                   f"SQLite={self.performance_metrics['sqlite_write_rate']:.1f}/s")
                    
                    # Reset counters
                    self.processing_stats['total_processed'] = 0
                    self.processing_stats['total_filtered'] = 0
                    self.performance_metrics['last_performance_update'] = current_time
                
        except asyncio.CancelledError:
            self.logger.info("Performance monitor cancelled")
        except Exception as e:
            self.logger.error(f"Performance monitor error: {e}")
    
    async def _cleanup_worker(self):
        """Periodic cleanup of old data and optimization"""
        try:
            while True:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up old filtered symbols (keep only recent)
                current_time = datetime.now()
                symbols_to_remove = []
                
                for symbol, data_point in self.filtered_symbols.items():
                    if current_time - data_point.timestamp > timedelta(minutes=30):
                        symbols_to_remove.append(symbol)
                
                for symbol in symbols_to_remove:
                    del self.filtered_symbols[symbol]
                
                if symbols_to_remove:
                    self.logger.info(f"Cleaned up {len(symbols_to_remove)} stale filtered symbols")
                
                # Optimize data structures
                self.quality_analyzer.price_history.clear()
                self.quality_analyzer.volume_history.clear()
                
        except asyncio.CancelledError:
            self.logger.info("Cleanup worker cancelled")
        except Exception as e:
            self.logger.error(f"Cleanup worker error: {e}")
    
    async def get_filtered_symbols(self) -> Dict[str, Dict[str, Any]]:
        """Get current filtered symbols for SQLite"""
        try:
            result = {}
            for symbol, data_point in self.filtered_symbols.items():
                result[symbol] = data_point.to_dict()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Get filtered symbols error: {e}")
            return {}
    
    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        try:
            stats = self.processing_stats.copy()
            stats.update(self.performance_metrics)
            stats['filtered_symbols_count'] = len(self.filtered_symbols)
            stats['processed_symbols_count'] = len(self.processed_symbols)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Get statistics error: {e}")
            return {}
    
    async def update_filtering_criteria(self, new_criteria: Dict[str, Any]) -> bool:
        """Update filtering criteria from user configuration"""
        try:
            # Update filtering criteria
            if 'min_price' in new_criteria:
                self.filtering_criteria.min_price = float(new_criteria['min_price'])
            if 'max_price' in new_criteria:
                self.filtering_criteria.max_price = float(new_criteria['max_price'])
            if 'min_volume' in new_criteria:
                self.filtering_criteria.min_volume = int(new_criteria['min_volume'])
            if 'min_value' in new_criteria:
                self.filtering_criteria.min_value = float(new_criteria['min_value'])
            if 'max_spread_percent' in new_criteria:
                self.filtering_criteria.max_spread_percent = float(new_criteria['max_spread_percent'])
            if 'min_change_percent' in new_criteria:
                self.filtering_criteria.min_change_percent = float(new_criteria['min_change_percent'])
            if 'max_change_percent' in new_criteria:
                self.filtering_criteria.max_change_percent = float(new_criteria['max_change_percent'])
            
            self.logger.info("Filtering criteria updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Criteria update error: {e}")
            return False
    
    async def force_flush_batches(self):
        """Force flush all pending batches"""
        try:
            if self.postgresql_batch:
                await self._flush_postgresql_batch()
            
            if self.sqlite_batch:
                await self._flush_sqlite_batch()
            
            self.logger.info("All batches flushed successfully")
            
        except Exception as e:
            self.logger.error(f"Force flush error: {e}")
    
    async def get_symbol_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed analysis for a specific symbol"""
        try:
            if symbol in self.filtered_symbols:
                data_point = self.filtered_symbols[symbol]
                
                analysis = {
                    'symbol': symbol,
                    'data': data_point.to_dict(),
                    'score': self._calculate_symbol_score(data_point),
                    'quality': data_point.quality.value,
                    'categories': [cat.value for cat in data_point.category],
                    'scalping_suitable': SymbolCategory.SCALPING_SUITABLE in data_point.category,
                    'analysis_time': datetime.now().isoformat()
                }
                
                return analysis
            
            return None
            
        except Exception as e:
            self.logger.error(f"Symbol analysis error for {symbol}: {e}")
            return None
    
    async def export_filtered_data_csv(self, filepath: str) -> bool:
        """Export current filtered data to CSV"""
        try:
            import csv
            
            if not self.filtered_symbols:
                self.logger.warning("No filtered data to export")
                return False
            
            # Prepare data for CSV
            fieldnames = [
                'symbol', 'exchange', 'timestamp', 'last_price', 'volume', 'value',
                'change_percent', 'quality', 'categories', 'score'
            ]
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for symbol, data_point in self.filtered_symbols.items():
                    row = {
                        'symbol': data_point.symbol,
                        'exchange': data_point.exchange,
                        'timestamp': data_point.timestamp.isoformat(),
                        'last_price': data_point.last_price,
                        'volume': data_point.volume,
                        'value': data_point.value,
                        'change_percent': data_point.change_percent,
                        'quality': data_point.quality.value,
                        'categories': '|'.join([cat.value for cat in data_point.category]),
                        'score': self._calculate_symbol_score(data_point)
                    }
                    writer.writerow(row)
            
            self.logger.info(f"Exported {len(self.filtered_symbols)} symbols to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"CSV export error: {e}")
            return False
    
    async def stop(self):
        """Stop data manager and flush remaining data"""
        try:
            self.logger.info("Stopping Data Manager...")
            
            # Flush any remaining batches
            await self.force_flush_batches()
            
            # Clear data structures
            self.postgresql_batch.clear()
            self.sqlite_batch.clear()
            self.filtered_symbols.clear()
            self.processed_symbols.clear()
            
            self.logger.info("Data Manager stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping Data Manager: {e}")

# Singleton instance
_data_manager = None

def get_data_manager() -> DataManager:
    """Get singleton data manager instance"""
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager()
    return _data_manager

# Utility functions for integration
async def start_data_processing_system(config_manager=None, database_manager=None, 
                                     security_manager=None) -> DataManager:
    """Start the complete data processing system"""
    try:
        # Initialize data manager
        data_manager = DataManager(config_manager, database_manager, security_manager)
        
        # Initialize the system
        success = await data_manager.initialize()
        if not success:
            raise Exception("Data manager initialization failed")
        
        return data_manager
        
    except Exception as e:
        logger.error(f"Failed to start data processing system: {e}")
        raise

async def test_data_processing():
    """Test data processing functionality"""
    try:
        print("🧪 Testing Data Processing System...")
        
        # Start data manager
        data_manager = await start_data_processing_system()
        
        # Create test market data
        test_data = MarketDataMessage(
            symbol="RELIANCE",
            message_type="touchline",
            data={
                'exchange': 'NSE',
                'last_price': 2450.75,
                'open_price': 2440.00,
                'high_price': 2460.00,
                'low_price': 2435.00,
                'close_price': 2445.00,
                'volume': 1500000,
                'value': 3675000000,
                'bid_price': 2450.50,
                'ask_price': 2450.80,
                'bid_qty': 1000,
                'ask_qty': 1200,
                'change': 5.75,
                'change_percent': 0.235,
                'average_price': 2450.00
            },
            timestamp=datetime.now(),
            source="test"
        )
        
        # Process test data
        success = await data_manager.process_market_data(test_data)
        if success:
            print("✅ Data processing successful")
            
            # Get processing statistics
            stats = await data_manager.get_processing_statistics()
            print(f"📊 Processing Stats: {stats}")
            
            # Get filtered symbols
            filtered = await data_manager.get_filtered_symbols()
            print(f"🔍 Filtered Symbols: {len(filtered)}")
            
        else:
            print("❌ Data processing failed")
        
        print("✅ Test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

# Example usage and main function
async def main():
    """Example usage of data manager"""
    try:
        print("🚀 Starting Market-Ready Data Processing System...")
        
        # Start data processing system
        data_manager = await start_data_processing_system()
        
        print("✅ Data processing system started successfully")
        print(f"📊 Initial Stats: {await data_manager.get_processing_statistics()}")
        
        # Run test
        await test_data_processing()
        
        print("🔄 Data processing system running... (Press Ctrl+C to stop)")
        
        try:
            # Wait indefinitely
            await asyncio.Event().wait()
            
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            await data_manager.stop()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    # Run data processing system
    asyncio.run(main())
