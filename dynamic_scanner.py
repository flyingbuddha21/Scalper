#!/usr/bin/env python3
"""
MARKET-READY Dynamic Scanner - Real-Time Symbol Filtering
Scans 2000+ NSE symbols and filters to 20-50 best trading opportunities
Uses real technical analysis and pattern detection for scalping
"""

import asyncio
import json
import logging
import time
import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np

# Import system components with fallback handling
try:
    from config_manager import ConfigManager, get_config
    from utils import Logger, ErrorHandler, DataValidator
    from security_manager import SecurityManager
    from database_setup import DatabaseManager, get_database_manager
    from data_manager import DataManager, get_data_manager, MarketDataPoint, SymbolCategory
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
        def validate_symbol(self, symbol):
            return isinstance(symbol, str) and len(symbol) > 0

    class ConfigManager:
        def get_config(self):
            return {
                'scanner': {
                    'scan_interval': 60,
                    'update_interval': 300,
                    'max_filtered_symbols': 50,
                    'min_volume_spike': 2.0,
                    'min_price_movement': 1.0,
                    'technical_indicators': ['RSI', 'MACD', 'BOLLINGER'],
                    'pattern_detection': True
                }
            }
    
    def get_config():
        return ConfigManager()
    
    # Mock classes for standalone operation
    class DatabaseManager:
        async def store_scanner_results(self, data): pass
        async def get_historical_data(self, symbol, periods): return []
        async def store_technical_indicators(self, data): pass
    
    def get_database_manager():
        return DatabaseManager()
    
    class DataManager:
        async def get_filtered_symbols(self): return {}
        async def get_processing_statistics(self): return {}
    
    def get_data_manager():
        return DataManager()
    
    # Mock data structures
    class SymbolCategory:
        HIGH_VOLUME = "HIGH_VOLUME"
        SCALPING_SUITABLE = "SCALPING_SUITABLE"
        MOMENTUM = "MOMENTUM"
        BREAKOUT_CANDIDATE = "BREAKOUT"
    
    @dataclass
    class MarketDataPoint:
        symbol: str
        last_price: float
        volume: int
        change_percent: float
        high_price: float = 0
        low_price: float = 0
        open_price: float = 0
        close_price: float = 0
        category: Set = field(default_factory=set)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalPattern(Enum):
    """Technical analysis patterns for scalping"""
    HAMMER = "HAMMER"                    # Reversal pattern
    DOJI = "DOJI"                       # Indecision pattern
    ENGULFING_BULLISH = "ENGULFING_BULL" # Strong bullish pattern
    ENGULFING_BEARISH = "ENGULFING_BEAR" # Strong bearish pattern
    BREAKOUT_UPWARD = "BREAKOUT_UP"     # Price breakout above resistance
    BREAKOUT_DOWNWARD = "BREAKOUT_DOWN" # Price breakdown below support
    SQUEEZE = "SQUEEZE"                 # Low volatility before explosion
    MOMENTUM_BULL = "MOMENTUM_BULL"     # Strong upward momentum
    MOMENTUM_BEAR = "MOMENTUM_BEAR"     # Strong downward momentum
    REVERSAL_BULL = "REVERSAL_BULL"     # Bullish reversal signal
    REVERSAL_BEAR = "REVERSAL_BEAR"     # Bearish reversal signal
    CONSOLIDATION = "CONSOLIDATION"     # Sideways movement

class ScanningCriteria(Enum):
    """Scanning criteria for different market conditions"""
    VOLUME_SPIKE = "VOLUME_SPIKE"       # Unusual volume activity
    PRICE_BREAKOUT = "PRICE_BREAKOUT"   # Price breaking key levels
    VOLATILITY_EXPANSION = "VOLATILITY_EXPANSION"
    MOMENTUM_SHIFT = "MOMENTUM_SHIFT"   # Change in momentum
    PATTERN_FORMATION = "PATTERN_FORMATION"
    LIQUIDITY_SURGE = "LIQUIDITY_SURGE"
    GAP_TRADING = "GAP_TRADING"         # Opening gaps
    RANGE_BOUND = "RANGE_BOUND"         # Trading in range

@dataclass
class TechnicalIndicators:
    """Technical indicators for a symbol"""
    rsi: float = 0.0                    # Relative Strength Index (0-100)
    macd: float = 0.0                   # MACD line
    macd_signal: float = 0.0            # MACD signal line
    macd_histogram: float = 0.0         # MACD histogram
    bb_upper: float = 0.0               # Bollinger Bands upper
    bb_lower: float = 0.0               # Bollinger Bands lower
    bb_middle: float = 0.0              # Bollinger Bands middle (SMA)
    bb_width: float = 0.0               # Bollinger Bands width
    atr: float = 0.0                    # Average True Range
    volume_sma: float = 0.0             # Volume Simple Moving Average
    price_sma_5: float = 0.0            # 5-period price SMA
    price_sma_10: float = 0.0           # 10-period price SMA
    price_sma_20: float = 0.0           # 20-period price SMA
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'rsi': self.rsi,
            'macd': self.macd,
            'macd_signal': self.macd_signal,
            'macd_histogram': self.macd_histogram,
            'bb_upper': self.bb_upper,
            'bb_lower': self.bb_lower,
            'bb_middle': self.bb_middle,
            'bb_width': self.bb_width,
            'atr': self.atr,
            'volume_sma': self.volume_sma,
            'price_sma_5': self.price_sma_5,
            'price_sma_10': self.price_sma_10,
            'price_sma_20': self.price_sma_20
        }

@dataclass
class ScanResult:
    """Result of symbol scanning"""
    symbol: str
    exchange: str
    timestamp: datetime
    current_price: float
    volume: int
    change_percent: float
    
    # Technical analysis
    technical_indicators: TechnicalIndicators
    detected_patterns: List[TechnicalPattern]
    
    # Scoring
    scalping_score: float               # 0-100 scalping suitability
    momentum_score: float               # 0-100 momentum strength
    volatility_score: float             # 0-100 volatility level
    liquidity_score: float              # 0-100 liquidity rating
    overall_score: float                # 0-100 overall rating
    
    # Criteria matches
    matching_criteria: List[ScanningCriteria]
    
    # Risk metrics
    support_level: float = 0.0
    resistance_level: float = 0.0
    risk_reward_ratio: float = 0.0
    
    # Market data
    bid_ask_spread: float = 0.0
    market_depth: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timestamp': self.timestamp.isoformat(),
            'current_price': self.current_price,
            'volume': self.volume,
            'change_percent': self.change_percent,
            'technical_indicators': self.technical_indicators.to_dict(),
            'detected_patterns': [p.value for p in self.detected_patterns],
            'scalping_score': self.scalping_score,
            'momentum_score': self.momentum_score,
            'volatility_score': self.volatility_score,
            'liquidity_score': self.liquidity_score,
            'overall_score': self.overall_score,
            'matching_criteria': [c.value for c in self.matching_criteria],
            'support_level': self.support_level,
            'resistance_level': self.resistance_level,
            'risk_reward_ratio': self.risk_reward_ratio,
            'bid_ask_spread': self.bid_ask_spread,
            'market_depth': self.market_depth
        }

class TechnicalAnalyzer:
    """Advanced technical analysis for scalping"""
    
    def __init__(self):
        self.logger = Logger(__name__)
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.indicator_cache: Dict[str, TechnicalIndicators] = {}
        
    def calculate_indicators(self, symbol: str, data_point: MarketDataPoint,
                           historical_data: List[Dict] = None) -> TechnicalIndicators:
        """Calculate comprehensive technical indicators"""
        try:
            # Add current data to history
            self.price_history[symbol].append(data_point.last_price)
            self.volume_history[symbol].append(data_point.volume)
            
            # Need at least 20 periods for meaningful calculations
            if len(self.price_history[symbol]) < 20:
                return TechnicalIndicators()
            
            prices = list(self.price_history[symbol])
            volumes = list(self.volume_history[symbol])
            
            indicators = TechnicalIndicators()
            
            # RSI calculation
            indicators.rsi = self._calculate_rsi(prices)
            
            # MACD calculation
            macd_values = self._calculate_macd(prices)
            indicators.macd = macd_values['macd']
            indicators.macd_signal = macd_values['signal']
            indicators.macd_histogram = macd_values['histogram']
            
            # Bollinger Bands
            bb_values = self._calculate_bollinger_bands(prices)
            indicators.bb_upper = bb_values['upper']
            indicators.bb_lower = bb_values['lower']
            indicators.bb_middle = bb_values['middle']
            indicators.bb_width = bb_values['width']
            
            # ATR (Average True Range)
            if data_point.high_price > 0 and data_point.low_price > 0:
                indicators.atr = self._calculate_atr(prices, data_point.high_price, data_point.low_price)
            
            # Volume indicators
            indicators.volume_sma = self._calculate_sma(volumes, 20)
            
            # Price moving averages
            indicators.price_sma_5 = self._calculate_sma(prices, 5)
            indicators.price_sma_10 = self._calculate_sma(prices, 10)
            indicators.price_sma_20 = self._calculate_sma(prices, 20)
            
            # Cache indicators
            self.indicator_cache[symbol] = indicators
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Indicator calculation error for {symbol}: {e}")
            return TechnicalIndicators()
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            gains = [d if d > 0 else 0 for d in deltas]
            losses = [-d if d < 0 else 0 for d in deltas]
            
            if len(gains) < period:
                return 50.0
            
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return round(rsi, 2)
            
        except Exception as e:
            self.logger.error(f"RSI calculation error: {e}")
            return 50.0
    
    def _calculate_macd(self, prices: List[float]) -> Dict[str, float]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            if len(prices) < 26:
                return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
            
            # Calculate EMAs
            ema_12 = self._calculate_ema(prices, 12)
            ema_26 = self._calculate_ema(prices, 26)
            
            macd_line = ema_12 - ema_26
            
            # For signal line, we'd need MACD history, simplified here
            signal_line = macd_line * 0.8  # Simplified approximation
            histogram = macd_line - signal_line
            
            return {
                'macd': round(macd_line, 4),
                'signal': round(signal_line, 4),
                'histogram': round(histogram, 4)
            }
            
        except Exception as e:
            self.logger.error(f"MACD calculation error: {e}")
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20, 
                                 std_dev: float = 2.0) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        try:
            if len(prices) < period:
                current_price = prices[-1] if prices else 0
                return {
                    'upper': current_price * 1.02,
                    'lower': current_price * 0.98,
                    'middle': current_price,
                    'width': current_price * 0.04
                }
            
            sma = self._calculate_sma(prices, period)
            recent_prices = prices[-period:]
            
            variance = sum((price - sma) ** 2 for price in recent_prices) / period
            std = math.sqrt(variance)
            
            upper_band = sma + (std_dev * std)
            lower_band = sma - (std_dev * std)
            width = upper_band - lower_band
            
            return {
                'upper': round(upper_band, 2),
                'lower': round(lower_band, 2),
                'middle': round(sma, 2),
                'width': round(width, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Bollinger Bands calculation error: {e}")
            return {'upper': 0.0, 'lower': 0.0, 'middle': 0.0, 'width': 0.0}
    
    def _calculate_atr(self, prices: List[float], high: float, low: float, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            if len(prices) < 2:
                return high - low if high > low else 0.0
            
            # Simplified ATR calculation
            true_range = max(
                high - low,
                abs(high - prices[-2]),
                abs(low - prices[-2])
            )
            
            return round(true_range, 2)
            
        except Exception as e:
            self.logger.error(f"ATR calculation error: {e}")
            return 0.0
    
    def _calculate_sma(self, values: List[float], period: int) -> float:
        """Calculate Simple Moving Average"""
        try:
            if len(values) < period:
                return sum(values) / len(values) if values else 0.0
            
            return sum(values[-period:]) / period
            
        except Exception as e:
            self.logger.error(f"SMA calculation error: {e}")
            return 0.0
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        try:
            if len(prices) < period:
                return self._calculate_sma(prices, len(prices))
            
            multiplier = 2 / (period + 1)
            ema = prices[0]
            
            for price in prices[1:]:
                ema = (price * multiplier) + (ema * (1 - multiplier))
            
            return ema
            
        except Exception as e:
            self.logger.error(f"EMA calculation error: {e}")
            return 0.0
    
    def detect_patterns(self, symbol: str, data_point: MarketDataPoint,
                       indicators: TechnicalIndicators) -> List[TechnicalPattern]:
        """Detect technical patterns for scalping"""
        try:
            patterns = []
            
            if len(self.price_history[symbol]) < 5:
                return patterns
            
            prices = list(self.price_history[symbol])
            current_price = data_point.last_price
            
            # Pattern detection logic
            patterns.extend(self._detect_candlestick_patterns(data_point, prices))
            patterns.extend(self._detect_breakout_patterns(current_price, indicators))
            patterns.extend(self._detect_momentum_patterns(data_point, indicators))
            patterns.extend(self._detect_reversal_patterns(indicators))
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Pattern detection error for {symbol}: {e}")
            return []
    
    def _detect_candlestick_patterns(self, data_point: MarketDataPoint, 
                                   prices: List[float]) -> List[TechnicalPattern]:
        """Detect candlestick patterns"""
        patterns = []
        
        try:
            if data_point.high_price == 0 or data_point.low_price == 0:
                return patterns
            
            open_price = data_point.open_price if data_point.open_price > 0 else prices[-2] if len(prices) >= 2 else data_point.last_price
            close_price = data_point.last_price
            high_price = data_point.high_price
            low_price = data_point.low_price
            
            # Body and shadow calculations
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            
            if total_range == 0:
                return patterns
            
            # Doji pattern (small body)
            if body_size <= (total_range * 0.1):
                patterns.append(TechnicalPattern.DOJI)
            
            # Hammer pattern (long lower shadow, small upper shadow)
            if (lower_shadow >= body_size * 2 and 
                upper_shadow <= body_size * 0.5 and
                body_size > 0):
                patterns.append(TechnicalPattern.HAMMER)
            
            # Engulfing patterns (need previous candle)
            if len(prices) >= 2:
                prev_open = prices[-2]
                prev_close = prices[-1] if len(prices) > 1 else open_price
                
                # Bullish engulfing
                if (close_price > open_price and prev_close < prev_open and
                    close_price > prev_open and open_price < prev_close):
                    patterns.append(TechnicalPattern.ENGULFING_BULLISH)
                
                # Bearish engulfing
                if (close_price < open_price and prev_close > prev_open and
                    close_price < prev_open and open_price > prev_close):
                    patterns.append(TechnicalPattern.ENGULFING_BEARISH)
            
        except Exception as e:
            self.logger.error(f"Candlestick pattern detection error: {e}")
        
        return patterns
    
    def _detect_breakout_patterns(self, current_price: float, 
                                indicators: TechnicalIndicators) -> List[TechnicalPattern]:
        """Detect breakout patterns"""
        patterns = []
        
        try:
            # Bollinger Band breakouts
            if indicators.bb_upper > 0 and indicators.bb_lower > 0:
                # Upward breakout
                if current_price > indicators.bb_upper:
                    patterns.append(TechnicalPattern.BREAKOUT_UPWARD)
                
                # Downward breakout
                elif current_price < indicators.bb_lower:
                    patterns.append(TechnicalPattern.BREAKOUT_DOWNWARD)
                
                # Squeeze pattern (narrow bands)
                elif (indicators.bb_width > 0 and 
                      indicators.bb_width < (current_price * 0.02)):  # Less than 2% width
                    patterns.append(TechnicalPattern.SQUEEZE)
            
            # Moving average breakouts
            if indicators.price_sma_5 > 0 and indicators.price_sma_20 > 0:
                # Golden cross (5 SMA above 20 SMA)
                if (indicators.price_sma_5 > indicators.price_sma_20 and
                    current_price > indicators.price_sma_5):
                    patterns.append(TechnicalPattern.BREAKOUT_UPWARD)
                
                # Death cross (5 SMA below 20 SMA)
                elif (indicators.price_sma_5 < indicators.price_sma_20 and
                      current_price < indicators.price_sma_5):
                    patterns.append(TechnicalPattern.BREAKOUT_DOWNWARD)
        
        except Exception as e:
            self.logger.error(f"Breakout pattern detection error: {e}")
        
        return patterns
    
    def _detect_momentum_patterns(self, data_point: MarketDataPoint,
                                indicators: TechnicalIndicators) -> List[TechnicalPattern]:
        """Detect momentum patterns"""
        patterns = []
        
        try:
            # RSI momentum
            if indicators.rsi > 70:
                patterns.append(TechnicalPattern.MOMENTUM_BULL)
            elif indicators.rsi < 30:
                patterns.append(TechnicalPattern.MOMENTUM_BEAR)
            
            # MACD momentum
            if indicators.macd > indicators.macd_signal and indicators.macd_histogram > 0:
                patterns.append(TechnicalPattern.MOMENTUM_BULL)
            elif indicators.macd < indicators.macd_signal and indicators.macd_histogram < 0:
                patterns.append(TechnicalPattern.MOMENTUM_BEAR)
            
            # Price momentum
            if abs(data_point.change_percent) >= 2.0:
                if data_point.change_percent > 0:
                    patterns.append(TechnicalPattern.MOMENTUM_BULL)
                else:
                    patterns.append(TechnicalPattern.MOMENTUM_BEAR)
        
        except Exception as e:
            self.logger.error(f"Momentum pattern detection error: {e}")
        
        return patterns
    
    def _detect_reversal_patterns(self, indicators: TechnicalIndicators) -> List[TechnicalPattern]:
        """Detect reversal patterns"""
        patterns = []
        
        try:
            # RSI reversal signals
            if indicators.rsi >= 80:  # Extremely overbought
                patterns.append(TechnicalPattern.REVERSAL_BEAR)
            elif indicators.rsi <= 20:  # Extremely oversold
                patterns.append(TechnicalPattern.REVERSAL_BULL)
            
            # MACD divergence (simplified)
            if indicators.macd > 0 and indicators.macd_histogram < 0:
                patterns.append(TechnicalPattern.REVERSAL_BEAR)
            elif indicators.macd < 0 and indicators.macd_histogram > 0:
                patterns.append(TechnicalPattern.REVERSAL_BULL)
        
        except Exception as e:
            self.logger.error(f"Reversal pattern detection error: {e}")
        
        return patterns

class ScalpingScorer:
    """Advanced scoring system for scalping opportunities"""
    
    def __init__(self):
        self.logger = Logger(__name__)
    
    def calculate_scalping_score(self, data_point: MarketDataPoint, 
                               indicators: TechnicalIndicators,
                               patterns: List[TechnicalPattern]) -> float:
        """Calculate scalping suitability score (0-100)"""
        try:
            score = 0.0
            
            # Volume score (30% weight)
            volume_score = self._score_volume(data_point)
            score += volume_score * 0.3
            
            # Volatility score (25% weight)
            volatility_score = self._score_volatility(data_point, indicators)
            score += volatility_score * 0.25
            
            # Liquidity score (20% weight)
            liquidity_score = self._score_liquidity(data_point)
            score += liquidity_score * 0.2
            
            # Pattern score (15% weight)
            pattern_score = self._score_patterns(patterns)
            score += pattern_score * 0.15
            
            # Technical score (10% weight)
            technical_score = self._score_technical_indicators(indicators)
            score += technical_score * 0.1
            
            return min(100.0, max(0.0, score))
            
        except Exception as e:
            self.logger.error(f"Scalping score calculation error: {e}")
            return 0.0
    
    def _score_volume(self, data_point: MarketDataPoint) -> float:
        """Score based on volume characteristics"""
        try:
            # Volume thresholds for scoring
            if data_point.volume >= 1000000:      # 10 lakh+
                return 100.0
            elif data_point.volume >= 500000:     # 5 lakh+
                return 80.0
            elif data_point.volume >= 200000:     # 2 lakh+
                return 60.0
            elif data_point.volume >= 100000:     # 1 lakh+
                return 40.0
            elif data_point.volume >= 50000:      # 50k+
                return 20.0
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Volume scoring error: {e}")
            return 0.0
    
    def _score_volatility(self, data_point: MarketDataPoint, 
                         indicators: TechnicalIndicators) -> float:
        """Score based on volatility (optimal for scalping)"""
        try:
            change_abs = abs(data_point.change_percent)
            
            # Optimal volatility for scalping: 1-5%
            if 1.0 <= change_abs <= 3.0:
                return 100.0
            elif 0.5 <= change_abs < 1.0 or 3.0 < change_abs <= 5.0:
                return 80.0
            elif change_abs < 0.5:
                return 30.0  # Too low volatility
            elif 5.0 < change_abs <= 8.0:
                return 60.0  # High but manageable
            else:
                return 20.0  # Too high volatility
                
        except Exception as e:
            self.logger.error(f"Volatility scoring error: {e}")
            return 0.0
    
    def _score_liquidity(self, data_point: MarketDataPoint) -> float:
        """Score based on liquidity metrics"""
        try:
            score = 0.0
            
            # Value-based liquidity
            if hasattr(data_point, 'value'):
                value = getattr(data_point, 'value', data_point.volume * data_point.last_price)
                if value >= 50000000:      # 5 crore+
                    score += 50.0
                elif value >= 20000000:    # 2 crore+
                    score += 40.0
                elif value >= 10000000:    # 1 crore+
                    score += 30.0
                elif value >= 5000000:     # 50 lakh+
                    score += 20.0
                else:
                    score += 10.0
            
            # Bid-ask spread (if available)
            if (hasattr(data_point, 'bid_price') and hasattr(data_point, 'ask_price') and
                data_point.bid_price > 0 and data_point.ask_price > 0):
                spread_percent = ((data_point.ask_price - data_point.bid_price) / data_point.last_price) * 100
                
                if spread_percent <= 0.05:
                    score += 50.0
                elif spread_percent <= 0.1:
                    score += 40.0
                elif spread_percent <= 0.2:
                    score += 30.0
                elif spread_percent <= 0.5:
                    score += 20.0
                else:
                    score += 10.0
            else:
                score += 25.0  # Default if spread not available
            
            return min(100.0, score)
            
        except Exception as e:
            self.logger.error(f"Liquidity scoring error: {e}")
            return 0.0
    
    def _score_patterns(self, patterns: List[TechnicalPattern]) -> float:
        """Score based on detected patterns"""
        try:
            if not patterns:
                return 20.0  # Neutral score for no patterns
            
            # Pattern scoring weights
            pattern_scores = {
                TechnicalPattern.HAMMER: 90.0,
                TechnicalPattern.DOJI: 70.0,
                TechnicalPattern.ENGULFING_BULLISH: 95.0,
                TechnicalPattern.ENGULFING_BEARISH: 95.0,
                TechnicalPattern.BREAKOUT_UPWARD: 85.0,
                TechnicalPattern.BREAKOUT_DOWNWARD: 85.0,
                TechnicalPattern.SQUEEZE: 80.0,
                TechnicalPattern.MOMENTUM_BULL: 75.0,
                TechnicalPattern.MOMENTUM_BEAR: 75.0,
                TechnicalPattern.REVERSAL_BULL: 88.0,
                TechnicalPattern.REVERSAL_BEAR: 88.0,
                TechnicalPattern.CONSOLIDATION: 60.0
            }
            
            # Calculate weighted average of detected patterns
            total_score = 0.0
            for pattern in patterns:
                total_score += pattern_scores.get(pattern, 50.0)
            
            # Average score, but bonus for multiple good patterns
            base_score = total_score / len(patterns)
            
            # Bonus for multiple patterns (up to 20% bonus)
            pattern_bonus = min(20.0, len(patterns) * 5.0)
            
            return min(100.0, base_score + pattern_bonus)
            
        except Exception as e:
            self.logger.error(f"Pattern scoring error: {e}")
            return 20.0
    
    def _score_technical_indicators(self, indicators: TechnicalIndicators) -> float:
        """Score based on technical indicator alignment"""
        try:
            score = 0.0
            indicators_count = 0
            
            # RSI scoring (optimal zones for scalping)
            if 30 <= indicators.rsi <= 70:  # Neutral zone - good for scalping
                score += 80.0
            elif 20 <= indicators.rsi < 30 or 70 < indicators.rsi <= 80:  # Near extremes
                score += 60.0
            else:  # Extreme zones
                score += 40.0
            indicators_count += 1
            
            # MACD scoring
            if abs(indicators.macd_histogram) > 0:  # Active momentum
                score += 70.0
            else:
                score += 50.0
            indicators_count += 1
            
            # Bollinger Band position scoring
            if indicators.bb_width > 0:
                # Price near middle band is good for scalping
                if indicators.bb_middle > 0:
                    score += 60.0
                else:
                    score += 40.0
                indicators_count += 1
            
            # ATR scoring (volatility measure)
            if indicators.atr > 0:
                score += 65.0
                indicators_count += 1
            
            return score / indicators_count if indicators_count > 0 else 50.0
            
        except Exception as e:
            self.logger.error(f"Technical indicator scoring error: {e}")
            return 50.0
    
    def calculate_momentum_score(self, data_point: MarketDataPoint,
                               indicators: TechnicalIndicators) -> float:
        """Calculate momentum strength score"""
        try:
            score = 0.0
            
            # Price momentum
            change_abs = abs(data_point.change_percent)
            if change_abs >= 3.0:
                score += 40.0
            elif change_abs >= 1.5:
                score += 30.0
            elif change_abs >= 0.5:
                score += 20.0
            else:
                score += 10.0
            
            # RSI momentum
            if indicators.rsi > 50:
                score += (indicators.rsi - 50) * 0.6  # 0-30 points
            else:
                score += (50 - indicators.rsi) * 0.6  # 0-30 points
            
            # MACD momentum
            if indicators.macd_histogram > 0:
                score += 30.0
            elif indicators.macd_histogram < 0:
                score += 30.0
            else:
                score += 15.0
            
            return min(100.0, score)
            
        except Exception as e:
            self.logger.error(f"Momentum score calculation error: {e}")
            return 0.0
    
    def calculate_volatility_score(self, data_point: MarketDataPoint,
                                 indicators: TechnicalIndicators) -> float:
        """Calculate volatility score for scalping"""
        try:
            score = 0.0
            
            # Price volatility
            change_abs = abs(data_point.change_percent)
            if 1.0 <= change_abs <= 4.0:  # Optimal for scalping
                score += 50.0
            elif 0.5 <= change_abs < 1.0 or 4.0 < change_abs <= 6.0:
                score += 35.0
            elif change_abs < 0.5:
                score += 15.0  # Too low
            else:
                score += 20.0  # Too high
            
            # ATR volatility
            if indicators.atr > 0 and data_point.last_price > 0:
                atr_percent = (indicators.atr / data_point.last_price) * 100
                if 0.5 <= atr_percent <= 3.0:
                    score += 30.0
                elif atr_percent < 0.5:
                    score += 15.0
                else:
                    score += 20.0
            
            # Bollinger Band width
            if indicators.bb_width > 0 and data_point.last_price > 0:
                bb_width_percent = (indicators.bb_width / data_point.last_price) * 100
                if 2.0 <= bb_width_percent <= 8.0:
                    score += 20.0
                else:
                    score += 10.0
            
            return min(100.0, score)
            
        except Exception as e:
            self.logger.error(f"Volatility score calculation error: {e}")
            return 0.0

class DynamicScanner:
    """Main dynamic scanner for real-time symbol filtering"""
    
    def __init__(self, config_manager=None, database_manager=None, 
                 data_manager=None, security_manager=None):
        # Initialize configuration
        self.config_manager = config_manager or get_config()
        self.database_manager = database_manager or get_database_manager()
        self.data_manager = data_manager or get_data_manager()
        self.security_manager = security_manager
        
        try:
            self.config = self.config_manager.get_config()
            self.scanner_config = self.config.get('scanner', {})
        except:
            # Fallback configuration
            self.scanner_config = {
                'scan_interval': 60,
                'update_interval': 300,
                'max_filtered_symbols': 50,
                'min_volume_spike': 2.0,
                'min_price_movement': 1.0,
                'technical_indicators': ['RSI', 'MACD', 'BOLLINGER'],
                'pattern_detection': True
            }
        
        # Initialize components
        self.logger = Logger(__name__)
        self.error_handler = ErrorHandler()
        self.data_validator = DataValidator()
        self.technical_analyzer = TechnicalAnalyzer()
        self.scalping_scorer = ScalpingScorer()
        
        # Scanner state
        self.scan_results: Dict[str, ScanResult] = {}
        self.last_scan_time = datetime.now()
        self.scan_count = 0
        self.processing_stats = {
            'total_scanned': 0,
            'total_filtered': 0,
            'scan_duration': 0.0,
            'last_scan': None
        }
        
        # Configuration
        self.scan_interval = self.scanner_config.get('scan_interval', 60)  # seconds
        self.update_interval = self.scanner_config.get('update_interval', 300)  # seconds
        self.max_symbols = self.scanner_config.get('max_filtered_symbols', 50)
        
        # Scanning criteria
        self.active_criteria = [
            ScanningCriteria.VOLUME_SPIKE,
            ScanningCriteria.PRICE_BREAKOUT,
            ScanningCriteria.VOLATILITY_EXPANSION,
            ScanningCriteria.PATTERN_FORMATION,
            ScanningCriteria.LIQUIDITY_SURGE
        ]
        
        # Performance tracking
        self.performance_metrics = {
            'scans_per_minute': 0.0,
            'symbols_per_second': 0.0,
            'success_rate': 0.0,
            'last_performance_update': time.time()
        }
        
        self.logger.info("Dynamic Scanner initialized for real-time filtering")
    
    async def initialize(self) -> bool:
        """Initialize scanner and start scanning tasks"""
        try:
            # Start scanning tasks
            asyncio.create_task(self._scanning_worker())
            asyncio.create_task(self._update_worker())
            asyncio.create_task(self._performance_monitor())
            asyncio.create_task(self._cleanup_worker())
            
            self.logger.info("Dynamic Scanner started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Scanner initialization failed: {e}")
            self.error_handler.handle_error(e, "scanner_init")
            return False
    
    async def scan_symbols(self) -> Dict[str, ScanResult]:
        """Main scanning function"""
        try:
            start_time = time.time()
            
            # Get current market data from data manager
            market_data = await self.data_manager.get_filtered_symbols()
            
            if not market_data:
                self.logger.warning("No market data available for scanning")
                return {}
            
            self.logger.info(f"Scanning {len(market_data)} symbols...")
            
            # Process each symbol
            scan_results = {}
            processed_count = 0
            
            for symbol, data_dict in market_data.items():
                try:
                    # Convert to MarketDataPoint
                    data_point = self._dict_to_market_data_point(data_dict)
                    if not data_point:
                        continue
                    
                    # Perform comprehensive analysis
                    scan_result = await self._analyze_symbol(symbol, data_point)
                    
                    if scan_result and scan_result.overall_score >= 60.0:  # Minimum score threshold
                        scan_results[symbol] = scan_result
                    
                    processed_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error scanning symbol {symbol}: {e}")
                    continue
            
            # Filter to top symbols
            filtered_results = self._filter_top_symbols(scan_results)
            
            # Update scanner state
            self.scan_results = filtered_results
            self.last_scan_time = datetime.now()
            self.scan_count += 1
            
            # Update statistics
            scan_duration = time.time() - start_time
            self.processing_stats.update({
                'total_scanned': processed_count,
                'total_filtered': len(filtered_results),
                'scan_duration': scan_duration,
                'last_scan': datetime.now().isoformat()
            })
            
            # Store results in database
            await self._store_scan_results(filtered_results)
            
            self.logger.info(f"Scan completed: {processed_count} scanned, {len(filtered_results)} filtered in {scan_duration:.2f}s")
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Symbol scanning error: {e}")
            self.error_handler.handle_error(e, "symbol_scanning")
            return {}
    
    def _dict_to_market_data_point(self, data_dict: Dict) -> Optional[MarketDataPoint]:
        """Convert dictionary to MarketDataPoint"""
        try:
            return MarketDataPoint(
                symbol=data_dict.get('symbol', ''),
                last_price=float(data_dict.get('last_price', 0)),
                volume=int(data_dict.get('volume', 0)),
                change_percent=float(data_dict.get('change_percent', 0)),
                high_price=float(data_dict.get('high_price', 0)),
                low_price=float(data_dict.get('low_price', 0)),
                open_price=float(data_dict.get('open_price', 0)),
                close_price=float(data_dict.get('close_price', 0)),
                category=set()  # Will be populated by analysis
            )
        except Exception as e:
            self.logger.error(f"Data conversion error: {e}")
            return None
    
    async def _analyze_symbol(self, symbol: str, data_point: MarketDataPoint) -> Optional[ScanResult]:
        """Comprehensive symbol analysis"""
        try:
            # Calculate technical indicators
            indicators = self.technical_analyzer.calculate_indicators(symbol, data_point)
            
            # Detect patterns
            patterns = self.technical_analyzer.detect_patterns(symbol, data_point, indicators)
            
            # Calculate scores
            scalping_score = self.scalping_scorer.calculate_scalping_score(data_point, indicators, patterns)
            momentum_score = self.scalping_scorer.calculate_momentum_score(data_point, indicators)
            volatility_score = self.scalping_scorer.calculate_volatility_score(data_point, indicators)
            liquidity_score = self.scalping_scorer._score_liquidity(data_point)
            
            # Calculate overall score (weighted average)
            overall_score = (
                scalping_score * 0.4 +
                momentum_score * 0.25 +
                volatility_score * 0.2 +
                liquidity_score * 0.15
            )
            
            # Determine matching criteria
            matching_criteria = self._check_scanning_criteria(data_point, indicators, patterns)
            
            # Calculate support/resistance levels
            support_level, resistance_level = self._calculate_support_resistance(data_point, indicators)
            
            # Calculate risk/reward ratio
            risk_reward_ratio = self._calculate_risk_reward(data_point, support_level, resistance_level)
            
            # Create scan result
            scan_result = ScanResult(
                symbol=symbol,
                exchange=getattr(data_point, 'exchange', 'NSE'),
                timestamp=datetime.now(),
                current_price=data_point.last_price,
                volume=data_point.volume,
                change_percent=data_point.change_percent,
                technical_indicators=indicators,
                detected_patterns=patterns,
                scalping_score=scalping_score,
                momentum_score=momentum_score,
                volatility_score=volatility_score,
                liquidity_score=liquidity_score,
                overall_score=overall_score,
                matching_criteria=matching_criteria,
                support_level=support_level,
                resistance_level=resistance_level,
                risk_reward_ratio=risk_reward_ratio,
                bid_ask_spread=self._calculate_spread(data_point),
                market_depth=self._calculate_market_depth(data_point)
            )
            
            return scan_result
            
        except Exception as e:
            self.logger.error(f"Symbol analysis error for {symbol}: {e}")
            return None
    
    def _check_scanning_criteria(self, data_point: MarketDataPoint, 
                               indicators: TechnicalIndicators,
                               patterns: List[TechnicalPattern]) -> List[ScanningCriteria]:
        """Check which scanning criteria the symbol matches"""
        matching_criteria = []
        
        try:
            # Volume spike check
            if indicators.volume_sma > 0:
                volume_ratio = data_point.volume / indicators.volume_sma
                if volume_ratio >= 2.0:
                    matching_criteria.append(ScanningCriteria.VOLUME_SPIKE)
            
            # Price breakout check
            if (TechnicalPattern.BREAKOUT_UPWARD in patterns or 
                TechnicalPattern.BREAKOUT_DOWNWARD in patterns):
                matching_criteria.append(ScanningCriteria.PRICE_BREAKOUT)
            
            # Volatility expansion check
            if abs(data_point.change_percent) >= 1.5:
                matching_criteria.append(ScanningCriteria.VOLATILITY_EXPANSION)
            
            # Momentum shift check
            if (TechnicalPattern.MOMENTUM_BULL in patterns or 
                TechnicalPattern.MOMENTUM_BEAR in patterns):
                matching_criteria.append(ScanningCriteria.MOMENTUM_SHIFT)
            
            # Pattern formation check
            if patterns:
                matching_criteria.append(ScanningCriteria.PATTERN_FORMATION)
            
            # Liquidity surge check
            if hasattr(data_point, 'value'):
                value = getattr(data_point, 'value', data_point.volume * data_point.last_price)
                if value >= 10000000:  # 1 crore+
                    matching_criteria.append(ScanningCriteria.LIQUIDITY_SURGE)
            
            # Gap trading check
            if (data_point.open_price > 0 and data_point.close_price > 0 and
                abs((data_point.open_price - data_point.close_price) / data_point.close_price) > 0.02):
                matching_criteria.append(ScanningCriteria.GAP_TRADING)
            
            # Range bound check
            if (indicators.bb_width > 0 and data_point.last_price > 0 and
                (indicators.bb_width / data_point.last_price) < 0.02):
                matching_criteria.append(ScanningCriteria.RANGE_BOUND)
        
        except Exception as e:
            self.logger.error(f"Criteria checking error: {e}")
        
        return matching_criteria
    
    def _calculate_support_resistance(self, data_point: MarketDataPoint,
                                    indicators: TechnicalIndicators) -> Tuple[float, float]:
        """Calculate support and resistance levels"""
        try:
            current_price = data_point.last_price
            
            # Use Bollinger Bands as support/resistance
            if indicators.bb_upper > 0 and indicators.bb_lower > 0:
                resistance = indicators.bb_upper
                support = indicators.bb_lower
            else:
                # Fallback to price-based levels
                resistance = current_price * 1.02  # 2% above
                support = current_price * 0.98     # 2% below
            
            return support, resistance
            
        except Exception as e:
            self.logger.error(f"Support/resistance calculation error: {e}")
            return data_point.last_price * 0.98, data_point.last_price * 1.02
    
    def _calculate_risk_reward(self, data_point: MarketDataPoint,
                             support: float, resistance: float) -> float:
        """Calculate risk/reward ratio"""
        try:
            current_price = data_point.last_price
            
            if support > 0 and resistance > current_price:
                potential_profit = resistance - current_price
                potential_loss = current_price - support
                
                if potential_loss > 0:
                    return potential_profit / potential_loss
            
            return 1.0  # Default 1:1 ratio
            
        except Exception as e:
            self.logger.error(f"Risk/reward calculation error: {e}")
            return 1.0
    
    def _calculate_spread(self, data_point: MarketDataPoint) -> float:
        """Calculate bid-ask spread percentage"""
        try:
            if (hasattr(data_point, 'bid_price') and hasattr(data_point, 'ask_price') and
                data_point.bid_price > 0 and data_point.ask_price > 0):
                spread = ((data_point.ask_price - data_point.bid_price) / data_point.last_price) * 100
                return round(spread, 4)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Spread calculation error: {e}")
            return 0.0
    
    def _calculate_market_depth(self, data_point: MarketDataPoint) -> int:
        """Calculate market depth"""
        try:
            if (hasattr(data_point, 'bid_qty') and hasattr(data_point, 'ask_qty')):
                return getattr(data_point, 'bid_qty', 0) + getattr(data_point, 'ask_qty', 0)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Market depth calculation error: {e}")
            return 0
    
    def _filter_top_symbols(self, scan_results: Dict[str, ScanResult]) -> Dict[str, ScanResult]:
        """Filter to top symbols based on overall score"""
        try:
            if len(scan_results) <= self.max_symbols:
                return scan_results
            
            # Sort by overall score
            sorted_results = sorted(
                scan_results.items(),
                key=lambda x: x[1].overall_score,
                reverse=True
            )
            
            # Take top symbols
            top_results = dict(sorted_results[:self.max_symbols])
            
            self.logger.info(f"Filtered {len(scan_results)} symbols to top {len(top_results)}")
            
            return top_results
            
        except Exception as e:
            self.logger.error(f"Symbol filtering error: {e}")
            return scan_results
    
    async def _store_scan_results(self, results: Dict[str, ScanResult]):
        """Store scan results in database"""
        try:
            if not results:
                return
            
            # Prepare data for storage
            storage_data = {
                'timestamp': datetime.now().isoformat(),
                'scan_count': self.scan_count,
                'total_symbols': len(results),
                'results': {symbol: result.to_dict() for symbol, result in results.items()}
            }
            
            # Store in database
            await self.database_manager.store_scanner_results(storage_data)
            
            self.logger.debug(f"Stored scan results for {len(results)} symbols")
            
        except Exception as e:
            self.logger.error(f"Scan results storage error: {e}")
    
    async def _scanning_worker(self):
        """Background worker for regular scanning"""
        try:
            while True:
                start_time = time.time()
                
                # Perform scan
                await self.scan_symbols()
                
                # Calculate sleep time
                scan_duration = time.time() - start_time
                sleep_time = max(0, self.scan_interval - scan_duration)
                
                await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            self.logger.info("Scanning worker cancelled")
        except Exception as e:
            self.logger.error(f"Scanning worker error: {e}")
    
    async def _update_worker(self):
        """Background worker for updating filtered results"""
        try:
            while True:
                await asyncio.sleep(self.update_interval)
                
                # Update stored results every update_interval
                if self.scan_results:
                    await self._store_scan_results(self.scan_results)
                
        except asyncio.CancelledError:
            self.logger.info("Update worker cancelled")
        except Exception as e:
            self.logger.error(f"Update worker error: {e}")
    
    async def _performance_monitor(self):
        """Monitor scanner performance"""
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = time.time()
                time_diff = current_time - self.performance_metrics['last_performance_update']
                
                if time_diff > 0:
                    # Calculate performance metrics
                    self.performance_metrics['scans_per_minute'] = 60.0 / self.scan_interval
                    
                    if self.processing_stats['scan_duration'] > 0:
                        symbols_scanned = self.processing_stats['total_scanned']
                        self.performance_metrics['symbols_per_second'] = symbols_scanned / self.processing_stats['scan_duration']
                    
                    if self.processing_stats['total_scanned'] > 0:
                        self.performance_metrics['success_rate'] = (
                            self.processing_stats['total_filtered'] / self.processing_stats['total_scanned']
                        ) * 100
                    
                    # Log performance
                    self.logger.info(f"Scanner Performance: "
                                   f"Scans/min={self.performance_metrics['scans_per_minute']:.1f}, "
                                   f"Symbols/sec={self.performance_metrics['symbols_per_second']:.1f}, "
                                   f"Success rate={self.performance_metrics['success_rate']:.1f}%")
                    
                    self.performance_metrics['last_performance_update'] = current_time
                
        except asyncio.CancelledError:
            self.logger.info("Performance monitor cancelled")
        except Exception as e:
            self.logger.error(f"Performance monitor error: {e}")
    
    async def _cleanup_worker(self):
        """Periodic cleanup of old data"""
        try:
            while True:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up old cached data
                current_time = datetime.now()
                
                # Clear old technical analysis cache
                self.technical_analyzer.indicator_cache.clear()
                
                # Clear old price/volume history for inactive symbols
                active_symbols = set(self.scan_results.keys())
                
                symbols_to_remove = []
                for symbol in self.technical_analyzer.price_history.keys():
                    if symbol not in active_symbols:
                        symbols_to_remove.append(symbol)
                
                for symbol in symbols_to_remove:
                    del self.technical_analyzer.price_history[symbol]
                    del self.technical_analyzer.volume_history[symbol]
                
                if symbols_to_remove:
                    self.logger.info(f"Cleaned up cached data for {len(symbols_to_remove)} inactive symbols")
                
        except asyncio.CancelledError:
            self.logger.info("Cleanup worker cancelled")
        except Exception as e:
            self.logger.error(f"Cleanup worker error: {e}")
    
    async def get_scan_results(self) -> Dict[str, Dict[str, Any]]:
        """Get current scan results"""
        try:
            result = {}
            for symbol, scan_result in self.scan_results.items():
                result[symbol] = scan_result.to_dict()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Get scan results error: {e}")
            return {}
    
    async def get_top_symbols(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get top N symbols by overall score"""
        try:
            sorted_results = sorted(
                self.scan_results.items(),
                key=lambda x: x[1].overall_score,
                reverse=True
            )
            
            top_symbols = []
            for symbol, result in sorted_results[:count]:
                symbol_data = result.to_dict()
                symbol_data['rank'] = len(top_symbols) + 1
                top_symbols.append(symbol_data)
            
            return top_symbols
            
        except Exception as e:
            self.logger.error(f"Get top symbols error: {e}")
            return []
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get scanner statistics"""
        try:
            stats = self.processing_stats.copy()
            stats.update(self.performance_metrics)
            stats['scan_count'] = self.scan_count
            stats['last_scan_time'] = self.last_scan_time.isoformat()
            stats['active_symbols'] = len(self.scan_results)
            stats['scan_interval'] = self.scan_interval
            stats['max_symbols'] = self.max_symbols
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Get statistics error: {e}")
            return {}
    
    async def update_configuration(self, new_config: Dict[str, Any]) -> bool:
        """Update scanner configuration"""
        try:
            if 'scan_interval' in new_config:
                self.scan_interval = int(new_config['scan_interval'])
            
            if 'max_filtered_symbols' in new_config:
                self.max_symbols = int(new_config['max_filtered_symbols'])
            
            if 'active_criteria' in new_config:
                criteria_mapping = {
                    'VOLUME_SPIKE': ScanningCriteria.VOLUME_SPIKE,
                    'PRICE_BREAKOUT': ScanningCriteria.PRICE_BREAKOUT,
                    'VOLATILITY_EXPANSION': ScanningCriteria.VOLATILITY_EXPANSION,
                    'MOMENTUM_SHIFT': ScanningCriteria.MOMENTUM_SHIFT,
                    'PATTERN_FORMATION': ScanningCriteria.PATTERN_FORMATION,
                    'LIQUIDITY_SURGE': ScanningCriteria.LIQUIDITY_SURGE
                }
                
                self.active_criteria = [
                    criteria_mapping[criteria] 
                    for criteria in new_config['active_criteria']
                    if criteria in criteria_mapping
                ]
            
            self.logger.info("Scanner configuration updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration update error: {e}")
            return False
    
    async def export_results_csv(self, filepath: str) -> bool:
        """Export scan results to CSV"""
        try:
            import csv
            
            if not self.scan_results:
                self.logger.warning("No scan results to export")
                return False
            
            # Prepare CSV data
            fieldnames = [
                'symbol', 'exchange', 'current_price', 'volume', 'change_percent',
                'scalping_score', 'momentum_score', 'volatility_score', 'liquidity_score',
                'overall_score', 'patterns', 'criteria', 'support_level', 'resistance_level',
                'risk_reward_ratio', 'rsi', 'macd', 'bb_position'
            ]
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for symbol, result in self.scan_results.items():
                    # Calculate Bollinger Band position
                    bb_position = "MIDDLE"
                    if result.technical_indicators.bb_upper > 0 and result.technical_indicators.bb_lower > 0:
                        if result.current_price > result.technical_indicators.bb_upper:
                            bb_position = "ABOVE_UPPER"
                        elif result.current_price < result.technical_indicators.bb_lower:
                            bb_position = "BELOW_LOWER"
                        elif result.current_price > result.technical_indicators.bb_middle:
                            bb_position = "UPPER_HALF"
                        else:
                            bb_position = "LOWER_HALF"
                    
                    row = {
                        'symbol': result.symbol,
                        'exchange': result.exchange,
                        'current_price': result.current_price,
                        'volume': result.volume,
                        'change_percent': result.change_percent,
                        'scalping_score': result.scalping_score,
                        'momentum_score': result.momentum_score,
                        'volatility_score': result.volatility_score,
                        'liquidity_score': result.liquidity_score,
                        'overall_score': result.overall_score,
                        'patterns': '|'.join([p.value for p in result.detected_patterns]),
                        'criteria': '|'.join([c.value for c in result.matching_criteria]),
                        'support_level': result.support_level,
                        'resistance_level': result.resistance_level,
                        'risk_reward_ratio': result.risk_reward_ratio,
                        'rsi': result.technical_indicators.rsi,
                        'macd': result.technical_indicators.macd,
                        'bb_position': bb_position
                    }
                    writer.writerow(row)
            
            self.logger.info(f"Exported {len(self.scan_results)} scan results to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"CSV export error: {e}")
            return False
    
    async def force_scan(self) -> Dict[str, ScanResult]:
        """Force an immediate scan regardless of interval"""
        try:
            self.logger.info("Forcing immediate scan...")
            return await self.scan_symbols()
            
        except Exception as e:
            self.logger.error(f"Force scan error: {e}")
            return {}
    
    async def get_symbol_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed analysis for a specific symbol"""
        try:
            if symbol in self.scan_results:
                result = self.scan_results[symbol]
                details = result.to_dict()
                
                # Add additional analysis
                details['analysis'] = {
                    'scalping_suitability': self._get_scalping_suitability(result),
                    'risk_level': self._get_risk_level(result),
                    'momentum_direction': self._get_momentum_direction(result),
                    'volatility_category': self._get_volatility_category(result),
                    'trading_recommendation': self._get_trading_recommendation(result)
                }
                
                return details
            
            return None
            
        except Exception as e:
            self.logger.error(f"Symbol details error for {symbol}: {e}")
            return None
    
    def _get_scalping_suitability(self, result: ScanResult) -> str:
        """Determine scalping suitability level"""
        if result.scalping_score >= 90:
            return "EXCELLENT"
        elif result.scalping_score >= 75:
            return "VERY_GOOD"
        elif result.scalping_score >= 60:
            return "GOOD"
        elif result.scalping_score >= 45:
            return "FAIR"
        else:
            return "POOR"
    
    def _get_risk_level(self, result: ScanResult) -> str:
        """Determine risk level"""
        if result.volatility_score >= 80:
            return "HIGH"
        elif result.volatility_score >= 60:
            return "MEDIUM_HIGH"
        elif result.volatility_score >= 40:
            return "MEDIUM"
        elif result.volatility_score >= 20:
            return "LOW_MEDIUM"
        else:
            return "LOW"
    
    def _get_momentum_direction(self, result: ScanResult) -> str:
        """Determine momentum direction"""
        if TechnicalPattern.MOMENTUM_BULL in result.detected_patterns:
            return "BULLISH"
        elif TechnicalPattern.MOMENTUM_BEAR in result.detected_patterns:
            return "BEARISH"
        elif result.change_percent > 0.5:
            return "BULLISH"
        elif result.change_percent < -0.5:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _get_volatility_category(self, result: ScanResult) -> str:
        """Categorize volatility level"""
        change_abs = abs(result.change_percent)
        
        if change_abs >= 5.0:
            return "VERY_HIGH"
        elif change_abs >= 3.0:
            return "HIGH"
        elif change_abs >= 1.5:
            return "MEDIUM"
        elif change_abs >= 0.5:
            return "LOW_MEDIUM"
        else:
            return "LOW"
    
    def _get_trading_recommendation(self, result: ScanResult) -> str:
        """Generate trading recommendation"""
        if result.overall_score >= 85:
            return "STRONG_BUY"
        elif result.overall_score >= 70:
            return "BUY"
        elif result.overall_score >= 55:
            return "WEAK_BUY"
        elif result.overall_score >= 45:
            return "HOLD"
        elif result.overall_score >= 30:
            return "WEAK_SELL"
        else:
            return "AVOID"
    
    async def stop(self):
        """Stop scanner and cleanup"""
        try:
            self.logger.info("Stopping Dynamic Scanner...")
            
            # Store final results
            if self.scan_results:
                await self._store_scan_results(self.scan_results)
            
            # Clear data structures
            self.scan_results.clear()
            self.technical_analyzer.price_history.clear()
            self.technical_analyzer.volume_history.clear()
            self.technical_analyzer.indicator_cache.clear()
            
            self.logger.info("Dynamic Scanner stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping Dynamic Scanner: {e}")

# Singleton instance
_dynamic_scanner = None

def get_dynamic_scanner() -> DynamicScanner:
    """Get singleton dynamic scanner instance"""
    global _dynamic_scanner
    if _dynamic_scanner is None:
        _dynamic_scanner = DynamicScanner()
    return _dynamic_scanner

# Utility functions for integration
async def start_scanning_system(config_manager=None, database_manager=None,
                              data_manager=None, security_manager=None) -> DynamicScanner:
    """Start the complete scanning system"""
    try:
        # Initialize scanner
        scanner = DynamicScanner(config_manager, database_manager, data_manager, security_manager)
        
        # Initialize the system
        success = await scanner.initialize()
        if not success:
            raise Exception("Scanner initialization failed")
        
        return scanner
        
    except Exception as e:
        logger.error(f"Failed to start scanning system: {e}")
        raise

async def test_dynamic_scanning():
    """Test dynamic scanning functionality"""
    try:
        print("🧪 Testing Dynamic Scanning System...")
        
        # Start scanner
        scanner = await start_scanning_system()
        
        # Wait for initial scan
        await asyncio.sleep(5)
        
        # Force a scan
        results = await scanner.force_scan()
        
        if results:
            print(f"✅ Scan successful: {len(results)} symbols found")
            
            # Get top symbols
            top_symbols = await scanner.get_top_symbols(5)
            print(f"🔝 Top 5 symbols:")
            for i, symbol_data in enumerate(top_symbols, 1):
                print(f"   {i}. {symbol_data['symbol']}: {symbol_data['overall_score']:.1f}")
            
            # Get statistics
            stats = await scanner.get_statistics()
            print(f"📊 Scanner Stats: {stats}")
            
            # Test symbol details
            if top_symbols:
                symbol = top_symbols[0]['symbol']
                details = await scanner.get_symbol_details(symbol)
                if details:
                    print(f"📋 Details for {symbol}:")
                    print(f"   Scalping Score: {details['scalping_score']:.1f}")
                    print(f"   Patterns: {details['detected_patterns']}")
                    print(f"   Recommendation: {details['analysis']['trading_recommendation']}")
        
        else:
            print("❌ No scan results found")
        
        print("✅ Test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

# Example usage and main function
async def main():
    """Example usage of dynamic scanner"""
    try:
        print("🚀 Starting Market-Ready Dynamic Scanner...")
        
        # Start scanning system
        scanner = await start_scanning_system()
        
        print("✅ Dynamic Scanner started successfully")
        print(f"📊 Initial Stats: {await scanner.get_statistics()}")
        
        # Run test
        await test_dynamic_scanning()
        
        print("🔄 Scanner running... (Press Ctrl+C to stop)")
        
        try:
            # Monitor scanning every 30 seconds
            while True:
                await asyncio.sleep(30)
                
                stats = await scanner.get_statistics()
                results = await scanner.get_scan_results()
                
                print(f"📊 Scanner Status: {len(results)} symbols filtered, "
                      f"Last scan: {stats.get('last_scan', 'Never')}")
                
                if results:
                    top_3 = await scanner.get_top_symbols(3)
                    print(f"🔝 Top 3: {[f\"{s['symbol']}({s['overall_score']:.0f})\" for s in top_3]}")
            
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            await scanner.stop()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    # Run dynamic scanner
    asyncio.run(main())
