#!/usr/bin/env python3
"""
MARKET-READY Volatility Analyzer - Advanced Volatility Analysis Engine
Deep analysis of 20-50 filtered symbols to select top 10 for trading
Runs every 15 minutes with sophisticated volatility modeling and risk assessment
"""

import asyncio
import json
import logging
import time
import math
import statistics
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import warnings

# Import system components with fallback handling
try:
    from config_manager import ConfigManager, get_config
    from utils import Logger, ErrorHandler, DataValidator
    from security_manager import SecurityManager
    from database_setup import DatabaseManager, get_database_manager
    from dynamic_scanner import DynamicScanner, get_dynamic_scanner, ScanResult
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
                'volatility_analyzer': {
                    'analysis_interval': 900,  # 15 minutes
                    'lookback_periods': 50,
                    'max_selected_symbols': 10,
                    'min_volatility_threshold': 0.5,
                    'max_volatility_threshold': 8.0,
                    'correlation_threshold': 0.7,
                    'var_confidence': 0.95
                }
            }
    
    def get_config():
        return ConfigManager()
    
    # Mock classes for standalone operation
    class DatabaseManager:
        async def store_volatility_analysis(self, data): pass
        async def get_historical_volatility(self, symbol, periods): return []
        async def store_top_symbols(self, data): pass
    
    def get_database_manager():
        return DatabaseManager()
    
    class DynamicScanner:
        async def get_scan_results(self): return {}
        async def get_top_symbols(self, count): return []
    
    def get_dynamic_scanner():
        return DynamicScanner()
    
    # Mock data structures
    @dataclass
    class ScanResult:
        symbol: str = ""
        current_price: float = 0.0
        volume: int = 0
        change_percent: float = 0.0
        scalping_score: float = 0.0
        overall_score: float = 0.0

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress numpy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_UP = "TRENDING_UP"         # Strong upward trend
    TRENDING_DOWN = "TRENDING_DOWN"     # Strong downward trend
    RANGING = "RANGING"                 # Sideways movement
    BREAKOUT_UP = "BREAKOUT_UP"        # Upward breakout from range
    BREAKOUT_DOWN = "BREAKOUT_DOWN"    # Downward breakout from range
    REVERSAL_UP = "REVERSAL_UP"        # Bullish reversal
    REVERSAL_DOWN = "REVERSAL_DOWN"    # Bearish reversal
    HIGH_VOLATILITY = "HIGH_VOLATILITY" # Chaotic, high volatility
    LOW_VOLATILITY = "LOW_VOLATILITY"   # Very low volatility
    SQUEEZE = "SQUEEZE"                 # Volatility compression

class VolatilityType(Enum):
    """Types of volatility characteristics"""
    REALIZED = "REALIZED"               # Historical realized volatility
    IMPLIED = "IMPLIED"                 # Market implied volatility
    INTRADAY = "INTRADAY"              # Within-day volatility
    OVERNIGHT = "OVERNIGHT"             # Overnight volatility
    CLUSTERED = "CLUSTERED"            # Volatility clustering
    MEAN_REVERTING = "MEAN_REVERTING"  # Mean reverting volatility

class RiskLevel(Enum):
    """Risk level classifications"""
    VERY_LOW = "VERY_LOW"      # < 1% daily volatility
    LOW = "LOW"                # 1-2% daily volatility
    MEDIUM = "MEDIUM"          # 2-4% daily volatility
    HIGH = "HIGH"              # 4-6% daily volatility
    VERY_HIGH = "VERY_HIGH"    # > 6% daily volatility
    EXTREME = "EXTREME"        # > 10% daily volatility

@dataclass
class VolatilityMetrics:
    """Comprehensive volatility metrics for a symbol"""
    symbol: str
    
    # Basic volatility measures
    realized_volatility: float = 0.0      # Annualized realized volatility
    intraday_volatility: float = 0.0      # Average intraday volatility
    rolling_volatility: float = 0.0       # 20-period rolling volatility
    
    # Advanced volatility measures
    garch_volatility: float = 0.0         # GARCH model forecast
    volatility_persistence: float = 0.0    # How long volatility lasts
    volatility_clustering: float = 0.0     # Tendency for volatility clusters
    
    # Risk measures
    value_at_risk_1d: float = 0.0         # 1-day VaR at 95% confidence
    expected_shortfall: float = 0.0        # Expected loss beyond VaR
    maximum_drawdown: float = 0.0          # Maximum historical drawdown
    
    # Market regime
    current_regime: MarketRegime = MarketRegime.RANGING
    regime_probability: float = 0.0        # Confidence in regime classification
    regime_duration: int = 0               # How long in current regime
    
    # Momentum and trend
    momentum_strength: float = 0.0         # Strength of current momentum
    momentum_sustainability: float = 0.0   # Likelihood momentum continues
    trend_direction: float = 0.0           # -1 (down) to +1 (up)
    trend_strength: float = 0.0            # Strength of trend
    
    # Liquidity timing
    optimal_entry_time: str = ""           # Best time to enter
    optimal_exit_time: str = ""            # Best time to exit
    liquidity_score: float = 0.0           # Current liquidity rating
    
    # Forecasts (next 15-30 minutes)
    volatility_forecast: float = 0.0       # Expected volatility
    price_range_forecast: Tuple[float, float] = (0.0, 0.0)  # Expected price range
    breakout_probability: float = 0.0      # Probability of breakout
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'realized_volatility': self.realized_volatility,
            'intraday_volatility': self.intraday_volatility,
            'rolling_volatility': self.rolling_volatility,
            'garch_volatility': self.garch_volatility,
            'volatility_persistence': self.volatility_persistence,
            'volatility_clustering': self.volatility_clustering,
            'value_at_risk_1d': self.value_at_risk_1d,
            'expected_shortfall': self.expected_shortfall,
            'maximum_drawdown': self.maximum_drawdown,
            'current_regime': self.current_regime.value,
            'regime_probability': self.regime_probability,
            'regime_duration': self.regime_duration,
            'momentum_strength': self.momentum_strength,
            'momentum_sustainability': self.momentum_sustainability,
            'trend_direction': self.trend_direction,
            'trend_strength': self.trend_strength,
            'optimal_entry_time': self.optimal_entry_time,
            'optimal_exit_time': self.optimal_exit_time,
            'liquidity_score': self.liquidity_score,
            'volatility_forecast': self.volatility_forecast,
            'price_range_forecast': list(self.price_range_forecast),
            'breakout_probability': self.breakout_probability
        }

@dataclass
class TradingOpportunity:
    """Complete trading opportunity analysis"""
    symbol: str
    rank: int                              # 1-10 ranking
    
    # Symbol data
    current_price: float
    volume: int
    change_percent: float
    
    # Volatility analysis
    volatility_metrics: VolatilityMetrics
    
    # Risk assessment
    risk_level: RiskLevel
    position_size_recommendation: float     # Recommended position size
    stop_loss_level: float                 # Recommended stop loss
    take_profit_level: float               # Recommended take profit
    
    # Entry/exit strategy
    entry_price_range: Tuple[float, float] # Optimal entry range
    exit_price_range: Tuple[float, float]  # Optimal exit range
    max_holding_time: int                  # Max minutes to hold
    
    # Scores and confidence
    overall_score: float                   # Combined score (0-100)
    confidence_level: float                # Confidence in analysis (0-100)
    scalping_suitability: float            # Suitability for scalping (0-100)
    
    # Analysis metadata
    analysis_timestamp: datetime
    next_analysis_time: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'rank': self.rank,
            'current_price': self.current_price,
            'volume': self.volume,
            'change_percent': self.change_percent,
            'volatility_metrics': self.volatility_metrics.to_dict(),
            'risk_level': self.risk_level.value,
            'position_size_recommendation': self.position_size_recommendation,
            'stop_loss_level': self.stop_loss_level,
            'take_profit_level': self.take_profit_level,
            'entry_price_range': list(self.entry_price_range),
            'exit_price_range': list(self.exit_price_range),
            'max_holding_time': self.max_holding_time,
            'overall_score': self.overall_score,
            'confidence_level': self.confidence_level,
            'scalping_suitability': self.scalping_suitability,
            'analysis_timestamp': self.analysis_timestamp.isoformat(),
            'next_analysis_time': self.next_analysis_time.isoformat()
        }

class VolatilityCalculator:
    """Advanced volatility calculation engine"""
    
    def __init__(self):
        self.logger = Logger(__name__)
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.return_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.volatility_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
    def calculate_volatility_metrics(self, symbol: str, scan_result: ScanResult,
                                   historical_data: List[Dict] = None) -> VolatilityMetrics:
        """Calculate comprehensive volatility metrics"""
        try:
            # Update price history
            self.price_history[symbol].append(scan_result.current_price)
            
            # Calculate returns
            if len(self.price_history[symbol]) >= 2:
                prev_price = list(self.price_history[symbol])[-2]
                current_return = (scan_result.current_price - prev_price) / prev_price
                self.return_history[symbol].append(current_return)
            
            # Need sufficient data for meaningful calculations
            if len(self.return_history[symbol]) < 20:
                return self._default_volatility_metrics(symbol)
            
            returns = list(self.return_history[symbol])
            prices = list(self.price_history[symbol])
            
            metrics = VolatilityMetrics(symbol=symbol)
            
            # Basic volatility measures
            metrics.realized_volatility = self._calculate_realized_volatility(returns)
            metrics.intraday_volatility = self._calculate_intraday_volatility(scan_result)
            metrics.rolling_volatility = self._calculate_rolling_volatility(returns)
            
            # Advanced volatility measures
            metrics.garch_volatility = self._calculate_garch_volatility(returns)
            metrics.volatility_persistence = self._calculate_volatility_persistence(returns)
            metrics.volatility_clustering = self._calculate_volatility_clustering(returns)
            
            # Risk measures
            metrics.value_at_risk_1d = self._calculate_var(returns)
            metrics.expected_shortfall = self._calculate_expected_shortfall(returns)
            metrics.maximum_drawdown = self._calculate_max_drawdown(prices)
            
            # Market regime analysis
            regime_analysis = self._analyze_market_regime(prices, returns)
            metrics.current_regime = regime_analysis['regime']
            metrics.regime_probability = regime_analysis['probability']
            metrics.regime_duration = regime_analysis['duration']
            
            # Momentum analysis
            momentum_analysis = self._analyze_momentum(prices, returns)
            metrics.momentum_strength = momentum_analysis['strength']
            metrics.momentum_sustainability = momentum_analysis['sustainability']
            metrics.trend_direction = momentum_analysis['direction']
            metrics.trend_strength = momentum_analysis['trend_strength']
            
            # Liquidity timing
            timing_analysis = self._analyze_liquidity_timing(scan_result)
            metrics.optimal_entry_time = timing_analysis['entry_time']
            metrics.optimal_exit_time = timing_analysis['exit_time']
            metrics.liquidity_score = timing_analysis['liquidity_score']
            
            # Forecasts
            forecast = self._generate_volatility_forecast(returns, prices)
            metrics.volatility_forecast = forecast['volatility']
            metrics.price_range_forecast = forecast['price_range']
            metrics.breakout_probability = forecast['breakout_probability']
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Volatility calculation error for {symbol}: {e}")
            return self._default_volatility_metrics(symbol)
    
    def _default_volatility_metrics(self, symbol: str) -> VolatilityMetrics:
        """Return default metrics when insufficient data"""
        return VolatilityMetrics(
            symbol=symbol,
            realized_volatility=15.0,  # Default 15% annualized
            current_regime=MarketRegime.RANGING,
            regime_probability=50.0
        )
    
    def _calculate_realized_volatility(self, returns: List[float]) -> float:
        """Calculate annualized realized volatility"""
        try:
            if len(returns) < 5:
                return 15.0
            
            # Calculate standard deviation of returns
            std_dev = statistics.stdev(returns)
            
            # Annualize (assuming 252 trading days, 6.5 hours per day, 60 minutes per hour)
            # For minute data: sqrt(252 * 390) where 390 = 6.5 * 60
            annualization_factor = math.sqrt(252 * 390)
            annualized_vol = std_dev * annualization_factor * 100
            
            return min(100.0, max(1.0, annualized_vol))
            
        except Exception as e:
            self.logger.error(f"Realized volatility calculation error: {e}")
            return 15.0
    
    def _calculate_intraday_volatility(self, scan_result: ScanResult) -> float:
        """Calculate current intraday volatility"""
        try:
            # Use change_percent as proxy for intraday volatility
            return abs(scan_result.change_percent)
            
        except Exception as e:
            self.logger.error(f"Intraday volatility calculation error: {e}")
            return 2.0
    
    def _calculate_rolling_volatility(self, returns: List[float], window: int = 20) -> float:
        """Calculate rolling volatility"""
        try:
            if len(returns) < window:
                return self._calculate_realized_volatility(returns)
            
            recent_returns = returns[-window:]
            return self._calculate_realized_volatility(recent_returns)
            
        except Exception as e:
            self.logger.error(f"Rolling volatility calculation error: {e}")
            return 15.0
    
    def _calculate_garch_volatility(self, returns: List[float]) -> float:
        """Simplified GARCH volatility forecast"""
        try:
            if len(returns) < 30:
                return self._calculate_realized_volatility(returns)
            
            # Simplified GARCH(1,1) approximation
            # σ²(t+1) = ω + α*ε²(t) + β*σ²(t)
            # Using typical parameters: ω=0.000001, α=0.1, β=0.85
            
            recent_returns = returns[-30:]
            long_term_var = statistics.variance(recent_returns)
            recent_squared_return = recent_returns[-1] ** 2
            
            # Simplified GARCH forecast
            omega = 0.000001
            alpha = 0.1
            beta = 0.85
            
            forecast_variance = omega + alpha * recent_squared_return + beta * long_term_var
            forecast_volatility = math.sqrt(forecast_variance) * math.sqrt(252 * 390) * 100
            
            return min(100.0, max(1.0, forecast_volatility))
            
        except Exception as e:
            self.logger.error(f"GARCH volatility calculation error: {e}")
            return self._calculate_realized_volatility(returns)
    
    def _calculate_volatility_persistence(self, returns: List[float]) -> float:
        """Calculate volatility persistence (how long volatility shocks last)"""
        try:
            if len(returns) < 20:
                return 50.0
            
            # Calculate squared returns (volatility proxy)
            squared_returns = [r**2 for r in returns]
            
            # Calculate autocorrelation of squared returns
            if len(squared_returns) >= 10:
                # Simple lag-1 autocorrelation
                mean_sq = statistics.mean(squared_returns)
                numerator = sum((squared_returns[i] - mean_sq) * (squared_returns[i-1] - mean_sq) 
                              for i in range(1, len(squared_returns)))
                denominator = sum((sq - mean_sq)**2 for sq in squared_returns)
                
                if denominator > 0:
                    autocorr = numerator / denominator
                    persistence = max(0.0, min(100.0, autocorr * 100))
                    return persistence
            
            return 50.0
            
        except Exception as e:
            self.logger.error(f"Volatility persistence calculation error: {e}")
            return 50.0
    
    def _calculate_volatility_clustering(self, returns: List[float]) -> float:
        """Calculate volatility clustering tendency"""
        try:
            if len(returns) < 15:
                return 50.0
            
            # Calculate absolute returns (volatility proxy)
            abs_returns = [abs(r) for r in returns]
            
            # Count volatility clusters (consecutive periods of high volatility)
            threshold = statistics.mean(abs_returns) + statistics.stdev(abs_returns)
            
            clusters = 0
            in_cluster = False
            
            for abs_ret in abs_returns:
                if abs_ret > threshold:
                    if not in_cluster:
                        clusters += 1
                        in_cluster = True
                else:
                    in_cluster = False
            
            # Normalize clustering score
            max_possible_clusters = len(abs_returns) // 3
            clustering_score = (clusters / max_possible_clusters) * 100 if max_possible_clusters > 0 else 50.0
            
            return min(100.0, max(0.0, clustering_score))
            
        except Exception as e:
            self.logger.error(f"Volatility clustering calculation error: {e}")
            return 50.0
    
    def _calculate_var(self, returns: List[float], confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        try:
            if len(returns) < 10:
                return 2.0
            
            # Sort returns and find percentile
            sorted_returns = sorted(returns)
            percentile_index = int((1 - confidence) * len(sorted_returns))
            
            if percentile_index < len(sorted_returns):
                var = abs(sorted_returns[percentile_index]) * 100
                return min(20.0, max(0.1, var))
            
            return 2.0
            
        except Exception as e:
            self.logger.error(f"VaR calculation error: {e}")
            return 2.0
    
    def _calculate_expected_shortfall(self, returns: List[float], confidence: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        try:
            if len(returns) < 10:
                return 3.0
            
            # Sort returns and calculate average of worst (1-confidence) returns
            sorted_returns = sorted(returns)
            cutoff_index = int((1 - confidence) * len(sorted_returns))
            
            if cutoff_index > 0:
                worst_returns = sorted_returns[:cutoff_index]
                expected_shortfall = abs(statistics.mean(worst_returns)) * 100
                return min(30.0, max(0.1, expected_shortfall))
            
            return 3.0
            
        except Exception as e:
            self.logger.error(f"Expected shortfall calculation error: {e}")
            return 3.0
    
    def _calculate_max_drawdown(self, prices: List[float]) -> float:
        """Calculate maximum drawdown"""
        try:
            if len(prices) < 5:
                return 5.0
            
            peak = prices[0]
            max_drawdown = 0.0
            
            for price in prices[1:]:
                if price > peak:
                    peak = price
                else:
                    drawdown = (peak - price) / peak
                    max_drawdown = max(max_drawdown, drawdown)
            
            return min(50.0, max(0.0, max_drawdown * 100))
            
        except Exception as e:
            self.logger.error(f"Max drawdown calculation error: {e}")
            return 5.0
    
    def _analyze_market_regime(self, prices: List[float], returns: List[float]) -> Dict[str, Any]:
        """Analyze current market regime"""
        try:
            if len(prices) < 10 or len(returns) < 10:
                return {
                    'regime': MarketRegime.RANGING,
                    'probability': 50.0,
                    'duration': 5
                }
            
            # Calculate trend indicators
            short_ma = statistics.mean(prices[-5:])
            long_ma = statistics.mean(prices[-10:])
            current_price = prices[-1]
            
            # Calculate volatility level
            recent_vol = statistics.stdev(returns[-10:]) * 100
            avg_vol = statistics.stdev(returns) * 100
            
            # Determine regime
            regime = MarketRegime.RANGING
            probability = 60.0
            
            if short_ma > long_ma * 1.01:  # 1% threshold
                if recent_vol > avg_vol * 1.5:
                    regime = MarketRegime.BREAKOUT_UP
                    probability = 75.0
                else:
                    regime = MarketRegime.TRENDING_UP
                    probability = 70.0
            elif short_ma < long_ma * 0.99:  # 1% threshold
                if recent_vol > avg_vol * 1.5:
                    regime = MarketRegime.BREAKOUT_DOWN
                    probability = 75.0
                else:
                    regime = MarketRegime.TRENDING_DOWN
                    probability = 70.0
            elif recent_vol > avg_vol * 2.0:
                regime = MarketRegime.HIGH_VOLATILITY
                probability = 80.0
            elif recent_vol < avg_vol * 0.5:
                regime = MarketRegime.LOW_VOLATILITY
                probability = 80.0
            
            # Estimate duration (simplified)
            duration = min(30, max(5, len(prices) // 2))
            
            return {
                'regime': regime,
                'probability': probability,
                'duration': duration
            }
            
        except Exception as e:
            self.logger.error(f"Market regime analysis error: {e}")
            return {
                'regime': MarketRegime.RANGING,
                'probability': 50.0,
                'duration': 5
            }
    
    def _analyze_momentum(self, prices: List[float], returns: List[float]) -> Dict[str, Any]:
        """Analyze momentum characteristics"""
        try:
            if len(prices) < 10 or len(returns) < 10:
                return {
                    'strength': 50.0,
                    'sustainability': 50.0,
                    'direction': 0.0,
                    'trend_strength': 50.0
                }
            
            # Momentum strength (based on recent returns)
            recent_returns = returns[-5:]
            momentum_strength = abs(sum(recent_returns)) * 1000  # Scale up
            momentum_strength = min(100.0, max(0.0, momentum_strength))
            
            # Momentum direction
            avg_return = statistics.mean(recent_returns)
            direction = max(-1.0, min(1.0, avg_return * 1000))
            
            # Momentum sustainability (consistency of direction)
            positive_returns = sum(1 for r in recent_returns if r > 0)
            negative_returns = sum(1 for r in recent_returns if r < 0)
            
            if len(recent_returns) > 0:
                consistency = max(positive_returns, negative_returns) / len(recent_returns)
                sustainability = consistency * 100
            else:
                sustainability = 50.0
            
            # Trend strength (price vs moving averages)
            if len(prices) >= 10:
                short_ma = statistics.mean(prices[-5:])
                long_ma = statistics.mean(prices[-10:])
                current_price = prices[-1]
                
                if long_ma > 0:
                    trend_strength = abs((short_ma - long_ma) / long_ma) * 1000
                    trend_strength = min(100.0, max(0.0, trend_strength))
                else:
                    trend_strength = 50.0
            else:
                trend_strength = 50.0
            
            return {
                'strength': momentum_strength,
                'sustainability': sustainability,
                'direction': direction,
                'trend_strength': trend_strength
            }
            
        except Exception as e:
            self.logger.error(f"Momentum analysis error: {e}")
            return {
                'strength': 50.0,
                'sustainability': 50.0,
                'direction': 0.0,
                'trend_strength': 50.0
            }
    
    def _analyze_liquidity_timing(self, scan_result: ScanResult) -> Dict[str, Any]:
        """Analyze optimal timing for liquidity"""
        try:
            current_hour = datetime.now().hour
            current_minute = datetime.now().minute
            
            # Indian market hours: 9:15 AM to 3:30 PM
            # Best liquidity typically: 9:15-10:30, 11:00-12:00, 2:00-3:30
            
            optimal_times = {
                'entry_time': 'IMMEDIATE',
                'exit_time': 'WITHIN_30_MIN',
                'liquidity_score': 70.0
            }
            
            # Morning session (high liquidity)
            if 9 <= current_hour <= 10:
                optimal_times['liquidity_score'] = 90.0
                optimal_times['entry_time'] = 'IMMEDIATE'
                optimal_times['exit_time'] = 'WITHIN_15_MIN'
            
            # Mid-morning (good liquidity)
            elif 11 <= current_hour <= 12:
                optimal_times['liquidity_score'] = 85.0
                optimal_times['entry_time'] = 'IMMEDIATE'
                optimal_times['exit_time'] = 'WITHIN_20_MIN'
            
            # Afternoon session (good liquidity)
            elif 14 <= current_hour <= 15:
                optimal_times['liquidity_score'] = 80.0
                optimal_times['entry_time'] = 'IMMEDIATE'
                optimal_times['exit_time'] = 'WITHIN_25_MIN'
            
            # Low liquidity periods
            else:
                optimal_times['liquidity_score'] = 60.0
                optimal_times['entry_time'] = 'WAIT_FOR_VOLUME'
                optimal_times['exit_time'] = 'WITHIN_45_MIN'
            
            # Adjust based on volume
            if scan_result.volume > 500000:  # High volume
                optimal_times['liquidity_score'] *= 1.1
            elif scan_result.volume < 100000:  # Low volume
                optimal_times['liquidity_score'] *= 0.8
            
            optimal_times['liquidity_score'] = min(100.0, optimal_times['liquidity_score'])
            
            return optimal_times
            
        except Exception as e:
            self.logger.error(f"Liquidity timing analysis error: {e}")
            return {
                'entry_time': 'IMMEDIATE',
                'exit_time': 'WITHIN_30_MIN',
                'liquidity_score': 70.0
            }
    
    def _generate_volatility_forecast(self, returns: List[float], 
                                    prices: List[float]) -> Dict[str, Any]:
        """Generate volatility and price forecasts"""
        try:
            if len(returns) < 10 or len(prices) < 10:
                current_price = prices[-1] if prices else 100.0
                return {
                    'volatility': 15.0,
                    'price_range': (current_price * 0.98, current_price * 1.02),
                    'breakout_probability': 30.0
                }
            
            current_price = prices[-1]
            recent_volatility = statistics.stdev(returns[-10:]) * 100
            
            # Forecast next period volatility (GARCH-like)
            forecast_volatility = recent_volatility * 0.9 + statistics.stdev(returns) * 0.1
            
            # Calculate expected price range (2 standard deviations)
            expected_range = forecast_volatility / 100 * current_price / math.sqrt(252 * 390 / 15)  # 15-minute adjustment
            lower_bound = current_price - expected_range
            upper_bound = current_price + expected_range
            
            # Breakout probability (based on volatility compression)
            avg_volatility = statistics.mean([statistics.stdev(returns[i:i+5]) for i in range(0, len(returns)-5, 5)])
            current_volatility = statistics.stdev(returns[-5:])
            
            if avg_volatility > 0:
                compression_ratio = current_volatility / avg_volatility
                breakout_probability = max(0.0, min(100.0, (1 - compression_ratio) * 100))
            else:
                breakout_probability = 30.0
            
            return {
                'volatility': min(100.0, max(1.0, forecast_volatility)),
                'price_range': (max(0.01, lower_bound), upper_bound),
                'breakout_probability': breakout_probability
            }
            
        except Exception as e:
            self.logger.error(f"Volatility forecast error: {e}")
            current_price = prices[-1] if prices else 100.0
            return {
                'volatility': 15.0,
                'price_range': (current_price * 0.98, current_price * 1.02),
                'breakout_probability': 30.0
            }

class RiskAssessment:
    """Advanced risk assessment for trading opportunities"""
    
    def __init__(self):
        self.logger = Logger(__name__)
    
    def assess_risk_level(self, volatility_metrics: VolatilityMetrics, 
                         scan_result: ScanResult) -> RiskLevel:
        """Determine risk level based on volatility metrics"""
        try:
            # Primary risk indicator: realized volatility
            vol = volatility_metrics.realized_volatility
            
            if vol < 10.0:
                return RiskLevel.VERY_LOW
            elif vol < 20.0:
                return RiskLevel.LOW
            elif vol < 35.0:
                return RiskLevel.MEDIUM
            elif vol < 50.0:
                return RiskLevel.HIGH
            elif vol < 80.0:
                return RiskLevel.VERY_HIGH
            else:
                return RiskLevel.EXTREME
                
        except Exception as e:
            self.logger.error(f"Risk level assessment error: {e}")
            return RiskLevel.MEDIUM
    
    def calculate_position_size(self, volatility_metrics: VolatilityMetrics,
                              scan_result: ScanResult, account_balance: float = 100000.0) -> float:
        """Calculate recommended position size based on risk"""
        try:
            # Risk-based position sizing using VaR
            var_1d = volatility_metrics.value_at_risk_1d
            
            # Risk per trade: 1% of account for medium risk, adjust based on VaR
            base_risk_percent = 1.0
            
            # Adjust based on VaR
            if var_1d <= 1.0:
                risk_multiplier = 1.5  # Low risk, can take larger position
            elif var_1d <= 2.0:
                risk_multiplier = 1.0  # Normal risk
            elif var_1d <= 4.0:
                risk_multiplier = 0.7  # Higher risk, smaller position
            else:
                risk_multiplier = 0.5  # Very high risk, much smaller position
            
            # Calculate position size
            risk_amount = account_balance * (base_risk_percent / 100) * risk_multiplier
            stop_loss_percent = var_1d * 1.5  # Stop loss at 1.5x VaR
            
            if stop_loss_percent > 0:
                max_shares = risk_amount / (scan_result.current_price * stop_loss_percent / 100)
                
                # Cap at reasonable limits
                max_position_value = account_balance * 0.20  # Max 20% of account per position
                max_shares_by_value = max_position_value / scan_result.current_price
                
                recommended_shares = min(max_shares, max_shares_by_value)
                return max(1.0, recommended_shares)
            
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Position size calculation error: {e}")
            return 1.0
    
    def calculate_stop_loss(self, volatility_metrics: VolatilityMetrics,
                          scan_result: ScanResult) -> float:
        """Calculate recommended stop loss level"""
        try:
            current_price = scan_result.current_price
            
            # Use VaR-based stop loss
            var_percent = volatility_metrics.value_at_risk_1d
            
            # Stop loss at 1.5x VaR, minimum 0.5%, maximum 5%
            stop_loss_percent = max(0.5, min(5.0, var_percent * 1.5))
            
            # Direction-based stop loss
            if scan_result.change_percent > 0:  # Bullish, stop below
                stop_loss = current_price * (1 - stop_loss_percent / 100)
            else:  # Bearish, stop above
                stop_loss = current_price * (1 + stop_loss_percent / 100)
            
            return max(0.01, stop_loss)
            
        except Exception as e:
            self.logger.error(f"Stop loss calculation error: {e}")
            return scan_result.current_price * 0.98
    
    def calculate_take_profit(self, volatility_metrics: VolatilityMetrics,
                            scan_result: ScanResult) -> float:
        """Calculate recommended take profit level"""
        try:
            current_price = scan_result.current_price
            
            # Use volatility forecast for take profit
            forecast_range = volatility_metrics.price_range_forecast
            
            if forecast_range and len(forecast_range) == 2:
                lower, upper = forecast_range
                
                # Take profit at 70% of expected range
                if scan_result.change_percent > 0:  # Bullish
                    take_profit = current_price + (upper - current_price) * 0.7
                else:  # Bearish
                    take_profit = current_price - (current_price - lower) * 0.7
                
                return max(0.01, take_profit)
            
            # Fallback: 2:1 risk-reward ratio
            var_percent = volatility_metrics.value_at_risk_1d
            profit_percent = var_percent * 2.0
            
            if scan_result.change_percent > 0:
                return current_price * (1 + profit_percent / 100)
            else:
                return current_price * (1 - profit_percent / 100)
                
        except Exception as e:
            self.logger.error(f"Take profit calculation error: {e}")
            return scan_result.current_price * 1.02

class OpportunityScorer:
    """Advanced scoring system for trading opportunities"""
    
    def __init__(self):
        self.logger = Logger(__name__)
    
    def calculate_overall_score(self, volatility_metrics: VolatilityMetrics,
                              scan_result: ScanResult, risk_level: RiskLevel) -> float:
        """Calculate comprehensive opportunity score"""
        try:
            score = 0.0
            
            # Volatility score (25% weight)
            volatility_score = self._score_volatility_suitability(volatility_metrics)
            score += volatility_score * 0.25
            
            # Momentum score (20% weight)
            momentum_score = self._score_momentum_quality(volatility_metrics)
            score += momentum_score * 0.20
            
            # Risk-reward score (20% weight)
            risk_reward_score = self._score_risk_reward(volatility_metrics, risk_level)
            score += risk_reward_score * 0.20
            
            # Liquidity score (15% weight)
            liquidity_score = volatility_metrics.liquidity_score
            score += liquidity_score * 0.15
            
            # Market regime score (10% weight)
            regime_score = self._score_market_regime(volatility_metrics)
            score += regime_score * 0.10
            
            # Scanner score integration (10% weight)
            scanner_score = getattr(scan_result, 'scalping_score', 70.0)
            score += scanner_score * 0.10
            
            return min(100.0, max(0.0, score))
            
        except Exception as e:
            self.logger.error(f"Overall score calculation error: {e}")
            return 50.0
    
    def _score_volatility_suitability(self, metrics: VolatilityMetrics) -> float:
        """Score volatility suitability for scalping"""
        try:
            vol = metrics.realized_volatility
            
            # Optimal volatility for scalping: 15-35%
            if 15.0 <= vol <= 35.0:
                return 100.0
            elif 10.0 <= vol < 15.0 or 35.0 < vol <= 45.0:
                return 80.0
            elif 5.0 <= vol < 10.0 or 45.0 < vol <= 60.0:
                return 60.0
            elif vol < 5.0 or vol > 80.0:
                return 20.0
            else:
                return 40.0
                
        except Exception as e:
            self.logger.error(f"Volatility suitability scoring error: {e}")
            return 50.0
    
    def _score_momentum_quality(self, metrics: VolatilityMetrics) -> float:
        """Score momentum quality"""
        try:
            # Combine momentum strength and sustainability
            strength = metrics.momentum_strength
            sustainability = metrics.momentum_sustainability
            
            # Weighted combination
            momentum_score = strength * 0.6 + sustainability * 0.4
            
            return min(100.0, max(0.0, momentum_score))
            
        except Exception as e:
            self.logger.error(f"Momentum quality scoring error: {e}")
            return 50.0
    
    def _score_risk_reward(self, metrics: VolatilityMetrics, risk_level: RiskLevel) -> float:
        """Score risk-reward profile"""
        try:
            # Risk level scoring
            risk_scores = {
                RiskLevel.VERY_LOW: 60.0,    # Too low risk, limited profits
                RiskLevel.LOW: 80.0,         # Good risk level
                RiskLevel.MEDIUM: 100.0,     # Optimal risk level
                RiskLevel.HIGH: 70.0,        # Higher risk, but manageable
                RiskLevel.VERY_HIGH: 40.0,   # Too risky
                RiskLevel.EXTREME: 10.0      # Extremely risky
            }
            
            risk_score = risk_scores.get(risk_level, 50.0)
            
            # VaR-based adjustment
            var_score = 100.0 - min(50.0, metrics.value_at_risk_1d * 10)
            
            # Combined score
            return (risk_score + var_score) / 2
            
        except Exception as e:
            self.logger.error(f"Risk-reward scoring error: {e}")
            return 50.0
    
    def _score_market_regime(self, metrics: VolatilityMetrics) -> float:
        """Score based on market regime favorability"""
        try:
            regime = metrics.current_regime
            probability = metrics.regime_probability
            
            # Regime favorability for scalping
            regime_scores = {
                MarketRegime.TRENDING_UP: 85.0,
                MarketRegime.TRENDING_DOWN: 85.0,
                MarketRegime.RANGING: 70.0,
                MarketRegime.BREAKOUT_UP: 95.0,
                MarketRegime.BREAKOUT_DOWN: 95.0,
                MarketRegime.REVERSAL_UP: 80.0,
                MarketRegime.REVERSAL_DOWN: 80.0,
                MarketRegime.HIGH_VOLATILITY: 60.0,
                MarketRegime.LOW_VOLATILITY: 40.0,
                MarketRegime.SQUEEZE: 90.0
            }
            
            base_score = regime_scores.get(regime, 50.0)
            
            # Adjust by confidence
            confidence_factor = probability / 100.0
            adjusted_score = base_score * confidence_factor + 50.0 * (1 - confidence_factor)
            
            return min(100.0, max(0.0, adjusted_score))
            
        except Exception as e:
            self.logger.error(f"Market regime scoring error: {e}")
            return 50.0
    
    def calculate_confidence_level(self, volatility_metrics: VolatilityMetrics,
                                 scan_result: ScanResult) -> float:
        """Calculate confidence in the analysis"""
        try:
            confidence = 70.0  # Base confidence
            
            # Regime confidence
            regime_confidence = volatility_metrics.regime_probability
            confidence += (regime_confidence - 50.0) * 0.3
            
            # Data quality factor (based on volatility persistence)
            persistence = volatility_metrics.volatility_persistence
            if persistence > 70.0:
                confidence += 10.0
            elif persistence < 30.0:
                confidence -= 10.0
            
            # Volume factor
            volume = getattr(scan_result, 'volume', 100000)
            if volume > 500000:
                confidence += 5.0
            elif volume < 50000:
                confidence -= 10.0
            
            return min(100.0, max(20.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Confidence calculation error: {e}")
            return 70.0

class VolatilityAnalyzer:
    """Main volatility analyzer for top symbol selection"""
    
    def __init__(self, config_manager=None, database_manager=None,
                 dynamic_scanner=None, security_manager=None):
        # Initialize configuration
        self.config_manager = config_manager or get_config()
        self.database_manager = database_manager or get_database_manager()
        self.dynamic_scanner = dynamic_scanner or get_dynamic_scanner()
        self.security_manager = security_manager
        
        try:
            self.config = self.config_manager.get_config()
            self.analyzer_config = self.config.get('volatility_analyzer', {})
        except:
            # Fallback configuration
            self.analyzer_config = {
                'analysis_interval': 900,  # 15 minutes
                'lookback_periods': 50,
                'max_selected_symbols': 10,
                'min_volatility_threshold': 0.5,
                'max_volatility_threshold': 8.0,
                'correlation_threshold': 0.7,
                'var_confidence': 0.95
            }
        
        # Initialize components
        self.logger = Logger(__name__)
        self.error_handler = ErrorHandler()
        self.data_validator = DataValidator()
        self.volatility_calculator = VolatilityCalculator()
        self.risk_assessment = RiskAssessment()
        self.opportunity_scorer = OpportunityScorer()
        
        # Analyzer state
        self.top_opportunities: Dict[str, TradingOpportunity] = {}
        self.last_analysis_time = datetime.now()
        self.analysis_count = 0
        
        # Configuration
        self.analysis_interval = self.analyzer_config.get('analysis_interval', 900)  # 15 minutes
        self.max_symbols = self.analyzer_config.get('max_selected_symbols', 10)
        self.correlation_threshold = self.analyzer_config.get('correlation_threshold', 0.7)
        
        # Performance tracking
        self.analysis_stats = {
            'total_analyzed': 0,
            'total_selected': 0,
            'analysis_duration': 0.0,
            'last_analysis': None,
            'success_rate': 0.0
        }
        
        self.logger.info("Volatility Analyzer initialized for top symbol selection")
    
    async def initialize(self) -> bool:
        """Initialize analyzer and start analysis tasks"""
        try:
            # Start analysis tasks
            asyncio.create_task(self._analysis_worker())
            asyncio.create_task(self._performance_monitor())
            asyncio.create_task(self._cleanup_worker())
            
            self.logger.info("Volatility Analyzer started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Analyzer initialization failed: {e}")
            self.error_handler.handle_error(e, "volatility_analyzer_init")
            return False
    
    async def analyze_symbols(self) -> Dict[str, TradingOpportunity]:
        """Main analysis function - select top 10 symbols every 15 minutes"""
        try:
            start_time = time.time()
            
            # Get scan results from dynamic scanner
            scan_results = await self.dynamic_scanner.get_scan_results()
            
            if not scan_results:
                self.logger.warning("No scan results available for analysis")
                return {}
            
            self.logger.info(f"Analyzing {len(scan_results)} symbols for volatility...")
            
            # Analyze each symbol
            analyzed_opportunities = {}
            
            for symbol, scan_data in scan_results.items():
                try:
                    # Convert scan data to ScanResult object
                    scan_result = self._dict_to_scan_result(scan_data)
                    if not scan_result:
                        continue
                    
                    # Calculate volatility metrics
                    volatility_metrics = self.volatility_calculator.calculate_volatility_metrics(
                        symbol, scan_result
                    )
                    
                    # Assess risk
                    risk_level = self.risk_assessment.assess_risk_level(volatility_metrics, scan_result)
                    
                    # Calculate scores
                    overall_score = self.opportunity_scorer.calculate_overall_score(
                        volatility_metrics, scan_result, risk_level
                    )
                    confidence_level = self.opportunity_scorer.calculate_confidence_level(
                        volatility_metrics, scan_result
                    )
                    
                    # Skip if score too low
                    if overall_score < 60.0:
                        continue
                    
                    # Calculate position management
                    position_size = self.risk_assessment.calculate_position_size(
                        volatility_metrics, scan_result
                    )
                    stop_loss = self.risk_assessment.calculate_stop_loss(
                        volatility_metrics, scan_result
                    )
                    take_profit = self.risk_assessment.calculate_take_profit(
                        volatility_metrics, scan_result
                    )
                    
                    # Create trading opportunity
                    opportunity = TradingOpportunity(
                        symbol=symbol,
                        rank=0,  # Will be assigned later
                        current_price=scan_result.current_price,
                        volume=scan_result.volume,
                        change_percent=scan_result.change_percent,
                        volatility_metrics=volatility_metrics,
                        risk_level=risk_level,
                        position_size_recommendation=position_size,
                        stop_loss_level=stop_loss,
                        take_profit_level=take_profit,
                        entry_price_range=self._calculate_entry_range(scan_result, volatility_metrics),
                        exit_price_range=self._calculate_exit_range(scan_result, volatility_metrics),
                        max_holding_time=self._calculate_max_holding_time(volatility_metrics),
                        overall_score=overall_score,
                        confidence_level=confidence_level,
                        scalping_suitability=getattr(scan_result, 'scalping_score', 70.0),
                        analysis_timestamp=datetime.now(),
                        next_analysis_time=datetime.now() + timedelta(seconds=self.analysis_interval)
                    )
                    
                    analyzed_opportunities[symbol] = opportunity
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing symbol {symbol}: {e}")
                    continue
            
            # Filter for correlation and select top symbols
            top_opportunities = await self._select_top_symbols(analyzed_opportunities)
            
            # Assign ranks
            ranked_opportunities = self._assign_ranks(top_opportunities)
            
            # Update analyzer state
            self.top_opportunities = ranked_opportunities
            self.last_analysis_time = datetime.now()
            self.analysis_count += 1
            
            # Update statistics
            analysis_duration = time.time() - start_time
            self.analysis_stats.update({
                'total_analyzed': len(analyzed_opportunities),
                'total_selected': len(ranked_opportunities),
                'analysis_duration': analysis_duration,
                'last_analysis': datetime.now().isoformat(),
                'success_rate': (len(ranked_opportunities) / len(scan_results)) * 100 if scan_results else 0
            })
            
            # Store results
            await self._store_analysis_results(ranked_opportunities)
            
            self.logger.info(f"Analysis completed: {len(analyzed_opportunities)} analyzed, "
                           f"{len(ranked_opportunities)} selected in {analysis_duration:.2f}s")
            
            return ranked_opportunities
            
        except Exception as e:
            self.logger.error(f"Symbol analysis error: {e}")
            self.error_handler.handle_error(e, "symbol_analysis")
            return {}
    
    def _dict_to_scan_result(self, scan_data: Dict) -> Optional[ScanResult]:
        """Convert dictionary to ScanResult object"""
        try:
            return ScanResult(
                symbol=scan_data.get('symbol', ''),
                current_price=float(scan_data.get('current_price', 0)),
                volume=int(scan_data.get('volume', 0)),
                change_percent=float(scan_data.get('change_percent', 0)),
                scalping_score=float(scan_data.get('scalping_score', 70.0)),
                overall_score=float(scan_data.get('overall_score', 70.0))
            )
        except Exception as e:
            self.logger.error(f"Scan result conversion error: {e}")
            return None
    
    def _calculate_entry_range(self, scan_result: ScanResult, 
                             volatility_metrics: VolatilityMetrics) -> Tuple[float, float]:
        """Calculate optimal entry price range"""
        try:
            current_price = scan_result.current_price
            
            # Use volatility forecast for entry range
            forecast_range = volatility_metrics.price_range_forecast
            
            if forecast_range and len(forecast_range) == 2:
                lower_forecast, upper_forecast = forecast_range
                
                # Entry range is narrower than forecast range
                if scan_result.change_percent > 0:  # Bullish bias
                    entry_lower = current_price - (current_price - lower_forecast) * 0.3
                    entry_upper = current_price + (upper_forecast - current_price) * 0.2
                else:  # Bearish bias
                    entry_lower = current_price - (current_price - lower_forecast) * 0.2
                    entry_upper = current_price + (upper_forecast - current_price) * 0.3
                
                return (max(0.01, entry_lower), entry_upper)
            
            # Fallback: +/- 0.5% of current price
            return (current_price * 0.995, current_price * 1.005)
            
        except Exception as e:
            self.logger.error(f"Entry range calculation error: {e}")
            return (scan_result.current_price * 0.995, scan_result.current_price * 1.005)
    
    def _calculate_exit_range(self, scan_result: ScanResult,
                            volatility_metrics: VolatilityMetrics) -> Tuple[float, float]:
        """Calculate optimal exit price range"""
        try:
            current_price = scan_result.current_price
            var_percent = volatility_metrics.value_at_risk_1d
            
            # Exit range based on expected profit/loss
            profit_target = var_percent * 2.0  # 2:1 risk-reward
            
            if scan_result.change_percent > 0:  # Bullish
                exit_lower = current_price * (1 + profit_target * 0.5 / 100)
                exit_upper = current_price * (1 + profit_target / 100)
            else:  # Bearish
                exit_lower = current_price * (1 - profit_target / 100)
                exit_upper = current_price * (1 - profit_target * 0.5 / 100)
            
            return (max(0.01, exit_lower), exit_upper)
            
        except Exception as e:
            self.logger.error(f"Exit range calculation error: {e}")
            return (scan_result.current_price * 1.01, scan_result.current_price * 1.02)
    
    def _calculate_max_holding_time(self, volatility_metrics: VolatilityMetrics) -> int:
        """Calculate maximum recommended holding time in minutes"""
        try:
            # Base holding time on volatility and regime
            base_time = 30  # 30 minutes default
            
            # Adjust based on volatility
            vol = volatility_metrics.realized_volatility
            if vol > 50.0:
                base_time = 15  # High volatility, shorter holds
            elif vol > 30.0:
                base_time = 25
            elif vol < 15.0:
                base_time = 45  # Low volatility, longer holds
            
            # Adjust based on regime
            regime = volatility_metrics.current_regime
            if regime in [MarketRegime.BREAKOUT_UP, MarketRegime.BREAKOUT_DOWN]:
                base_time = 20  # Quick breakout trades
            elif regime == MarketRegime.SQUEEZE:
                base_time = 15  # Very quick squeeze plays
            elif regime == MarketRegime.RANGING:
                base_time = 40  # Longer range trades
            
            return max(5, min(60, base_time))
            
        except Exception as e:
            self.logger.error(f"Max holding time calculation error: {e}")
            return 30
    
    async def _select_top_symbols(self, opportunities: Dict[str, TradingOpportunity]) -> Dict[str, TradingOpportunity]:
        """Select top symbols considering correlation and diversification"""
        try:
            if len(opportunities) <= self.max_symbols:
                return opportunities
            
            # Sort by overall score
            sorted_opportunities = sorted(
                opportunities.items(),
                key=lambda x: x[1].overall_score,
                reverse=True
            )
            
            # Select top symbols with correlation filtering
            selected = {}
            correlation_matrix = {}
            
            for symbol, opportunity in sorted_opportunities:
                if len(selected) >= self.max_symbols:
                    break
                
                # Check correlation with already selected symbols
                is_correlated = False
                
                for selected_symbol in selected.keys():
                    correlation = await self._calculate_correlation(symbol, selected_symbol)
                    
                    if correlation > self.correlation_threshold:
                        is_correlated = True
                        self.logger.debug(f"Skipping {symbol} due to high correlation ({correlation:.2f}) with {selected_symbol}")
                        break
                
                if not is_correlated:
                    selected[symbol] = opportunity
                    self.logger.debug(f"Selected {symbol} with score {opportunity.overall_score:.1f}")
            
            self.logger.info(f"Selected {len(selected)} symbols from {len(opportunities)} candidates")
            return selected
            
        except Exception as e:
            self.logger.error(f"Top symbol selection error: {e}")
            # Fallback: just take top N by score
            sorted_opportunities = sorted(
                opportunities.items(),
                key=lambda x: x[1].overall_score,
                reverse=True
            )
            return dict(sorted_opportunities[:self.max_symbols])
    
    async def _calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two symbols"""
        try:
            # Get price history for both symbols
            history1 = list(self.volatility_calculator.return_history.get(symbol1, []))
            history2 = list(self.volatility_calculator.return_history.get(symbol2, []))
            
            if len(history1) < 10 or len(history2) < 10:
                return 0.0  # Assume no correlation if insufficient data
            
            # Align histories (take common length)
            min_length = min(len(history1), len(history2))
            aligned_history1 = history1[-min_length:]
            aligned_history2 = history2[-min_length:]
            
            # Calculate correlation coefficient
            if min_length >= 3:
                try:
                    import numpy as np
                    correlation = np.corrcoef(aligned_history1, aligned_history2)[0, 1]
                    return abs(correlation) if not np.isnan(correlation) else 0.0
                except:
                    # Fallback manual calculation
                    mean1 = statistics.mean(aligned_history1)
                    mean2 = statistics.mean(aligned_history2)
                    
                    numerator = sum((aligned_history1[i] - mean1) * (aligned_history2[i] - mean2) 
                                  for i in range(min_length))
                    
                    sum_sq1 = sum((x - mean1) ** 2 for x in aligned_history1)
                    sum_sq2 = sum((x - mean2) ** 2 for x in aligned_history2)
                    
                    denominator = math.sqrt(sum_sq1 * sum_sq2)
                    
                    if denominator > 0:
                        correlation = numerator / denominator
                        return abs(correlation)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Correlation calculation error between {symbol1} and {symbol2}: {e}")
            return 0.0
    
    def _assign_ranks(self, opportunities: Dict[str, TradingOpportunity]) -> Dict[str, TradingOpportunity]:
        """Assign ranks to selected opportunities"""
        try:
            # Sort by overall score
            sorted_items = sorted(
                opportunities.items(),
                key=lambda x: x[1].overall_score,
                reverse=True
            )
            
            # Assign ranks
            ranked_opportunities = {}
            for rank, (symbol, opportunity) in enumerate(sorted_items, 1):
                opportunity.rank = rank
                ranked_opportunities[symbol] = opportunity
            
            return ranked_opportunities
            
        except Exception as e:
            self.logger.error(f"Rank assignment error: {e}")
            return opportunities
    
    async def _store_analysis_results(self, opportunities: Dict[str, TradingOpportunity]):
        """Store analysis results in database"""
        try:
            if not opportunities:
                return
            
            # Prepare data for storage
            storage_data = {
                'timestamp': datetime.now().isoformat(),
                'analysis_count': self.analysis_count,
                'total_opportunities': len(opportunities),
                'opportunities': {symbol: opp.to_dict() for symbol, opp in opportunities.items()},
                'analysis_stats': self.analysis_stats
            }
            
            # Store in database
            await self.database_manager.store_volatility_analysis(storage_data)
            await self.database_manager.store_top_symbols(storage_data)
            
            self.logger.debug(f"Stored analysis results for {len(opportunities)} opportunities")
            
        except Exception as e:
            self.logger.error(f"Analysis results storage error: {e}")
    
    async def _analysis_worker(self):
        """Background worker for regular analysis every 15 minutes"""
        try:
            while True:
                start_time = time.time()
                
                # Perform analysis
                await self.analyze_symbols()
                
                # Calculate sleep time (15 minutes interval)
                analysis_duration = time.time() - start_time
                sleep_time = max(0, self.analysis_interval - analysis_duration)
                
                self.logger.info(f"Next analysis in {sleep_time/60:.1f} minutes")
                await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            self.logger.info("Analysis worker cancelled")
        except Exception as e:
            self.logger.error(f"Analysis worker error: {e}")
    
    async def _performance_monitor(self):
        """Monitor analyzer performance"""
        try:
            while True:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Log performance metrics
                stats = self.analysis_stats
                self.logger.info(f"Analyzer Performance: "
                               f"Success rate={stats['success_rate']:.1f}%, "
                               f"Last analysis: {stats['last_analysis']}, "
                               f"Selected: {stats['total_selected']}/{stats['total_analyzed']}")
                
        except asyncio.CancelledError:
            self.logger.info("Performance monitor cancelled")
        except Exception as e:
            self.logger.error(f"Performance monitor error: {e}")
    
    async def _cleanup_worker(self):
        """Periodic cleanup of old data"""
        try:
            while True:
                await asyncio.sleep(1800)  # Run every 30 minutes
                
                # Clean up old volatility calculation cache
                current_time = datetime.now()
                
                # Clear old data from volatility calculator
                for symbol in list(self.volatility_calculator.price_history.keys()):
                    # Keep only active symbols
                    if symbol not in self.top_opportunities:
                        del self.volatility_calculator.price_history[symbol]
                        if symbol in self.volatility_calculator.return_history:
                            del self.volatility_calculator.return_history[symbol]
                        if symbol in self.volatility_calculator.volatility_history:
                            del self.volatility_calculator.volatility_history[symbol]
                
                self.logger.debug("Cleaned up old volatility calculation data")
                
        except asyncio.CancelledError:
            self.logger.info("Cleanup worker cancelled")
        except Exception as e:
            self.logger.error(f"Cleanup worker error: {e}")
    
    async def get_top_opportunities(self) -> Dict[str, Dict[str, Any]]:
        """Get current top trading opportunities"""
        try:
            result = {}
            for symbol, opportunity in self.top_opportunities.items():
                result[symbol] = opportunity.to_dict()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Get top opportunities error: {e}")
            return {}
    
    async def get_opportunity_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed analysis for a specific opportunity"""
        try:
            if symbol in self.top_opportunities:
                opportunity = self.top_opportunities[symbol]
                details = opportunity.to_dict()
                
                # Add additional analysis
                details['detailed_analysis'] = {
                    'volatility_category': self._categorize_volatility(opportunity.volatility_metrics),
                    'risk_assessment': self._assess_detailed_risk(opportunity),
                    'trading_strategy': self._recommend_trading_strategy(opportunity),
                    'market_timing': self._analyze_market_timing(opportunity),
                    'exit_strategy': self._recommend_exit_strategy(opportunity)
                }
                
                return details
            
            return None
            
        except Exception as e:
            self.logger.error(f"Opportunity details error for {symbol}: {e}")
            return None
    
    def _categorize_volatility(self, metrics: VolatilityMetrics) -> str:
        """Categorize volatility characteristics"""
        vol = metrics.realized_volatility
        
        if vol < 10:
            return "VERY_LOW_VOLATILITY"
        elif vol < 20:
            return "LOW_VOLATILITY"
        elif vol < 35:
            return "MODERATE_VOLATILITY"
        elif vol < 50:
            return "HIGH_VOLATILITY"
        else:
            return "EXTREME_VOLATILITY"
    
    def _assess_detailed_risk(self, opportunity: TradingOpportunity) -> Dict[str, Any]:
        """Provide detailed risk assessment"""
        metrics = opportunity.volatility_metrics
        
        return {
            'risk_level': opportunity.risk_level.value,
            'max_loss_estimate': f"{metrics.value_at_risk_1d:.2f}%",
            'expected_shortfall': f"{metrics.expected_shortfall:.2f}%",
            'volatility_persistence': f"{metrics.volatility_persistence:.1f}%",
            'regime_stability': f"{metrics.regime_probability:.1f}%",
            'correlation_risk': 'LOW' if len(self.top_opportunities) <= 5 else 'MEDIUM'
        }
    
    def _recommend_trading_strategy(self, opportunity: TradingOpportunity) -> Dict[str, Any]:
        """Recommend specific trading strategy"""
        metrics = opportunity.volatility_metrics
        regime = metrics.current_regime
        
        strategies = {
            MarketRegime.TRENDING_UP: "MOMENTUM_LONG",
            MarketRegime.TRENDING_DOWN: "MOMENTUM_SHORT", 
            MarketRegime.RANGING: "MEAN_REVERSION",
            MarketRegime.BREAKOUT_UP: "BREAKOUT_LONG",
            MarketRegime.BREAKOUT_DOWN: "BREAKOUT_SHORT",
            MarketRegime.SQUEEZE: "VOLATILITY_EXPANSION",
            MarketRegime.REVERSAL_UP: "REVERSAL_LONG",
            MarketRegime.REVERSAL_DOWN: "REVERSAL_SHORT"
        }
        
        recommended_strategy = strategies.get(regime, "NEUTRAL_SCALPING")
        
        return {
            'primary_strategy': recommended_strategy,
            'entry_timing': 'IMMEDIATE' if metrics.liquidity_score > 80 else 'WAIT_FOR_SETUP',
            'position_sizing': 'AGGRESSIVE' if opportunity.confidence_level > 85 else 'CONSERVATIVE',
            'holding_period': f"{opportunity.max_holding_time} minutes"
        }
    
    def _analyze_market_timing(self, opportunity: TradingOpportunity) -> Dict[str, Any]:
        """Analyze optimal market timing"""
        metrics = opportunity.volatility_metrics
        
        return {
            'optimal_entry_time': metrics.optimal_entry_time,
            'optimal_exit_time': metrics.optimal_exit_time,
            'liquidity_rating': f"{metrics.liquidity_score:.1f}/100",
            'volatility_forecast': f"{metrics.volatility_forecast:.1f}%",
            'breakout_probability': f"{metrics.breakout_probability:.1f}%"
        }
    
    def _recommend_exit_strategy(self, opportunity: TradingOpportunity) -> Dict[str, Any]:
        """Recommend exit strategy"""
        return {
            'stop_loss': f"₹{opportunity.stop_loss_level:.2f}",
            'take_profit': f"₹{opportunity.take_profit_level:.2f}",
            'exit_range': f"₹{opportunity.exit_price_range[0]:.2f} - ₹{opportunity.exit_price_range[1]:.2f}",
            'max_holding_time': f"{opportunity.max_holding_time} minutes",
            'trail_stop': 'RECOMMENDED' if opportunity.volatility_metrics.momentum_strength > 70 else 'NOT_RECOMMENDED'
        }
    
    async def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        try:
            stats = self.analysis_stats.copy()
            stats.update({
                'analysis_count': self.analysis_count,
                'last_analysis_time': self.last_analysis_time.isoformat(),
                'active_opportunities': len(self.top_opportunities),
                'analysis_interval_minutes': self.analysis_interval / 60,
                'max_symbols': self.max_symbols,
                'correlation_threshold': self.correlation_threshold,
                'next_analysis_in': (
                    self.last_analysis_time + timedelta(seconds=self.analysis_interval) - datetime.now()
                ).total_seconds() / 60
            })
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Get statistics error: {e}")
            return {}
    
    async def force_analysis(self) -> Dict[str, TradingOpportunity]:
        """Force immediate analysis regardless of interval"""
        try:
            self.logger.info("Forcing immediate volatility analysis...")
            return await self.analyze_symbols()
            
        except Exception as e:
            self.logger.error(f"Force analysis error: {e}")
            return {}
    
    async def update_configuration(self, new_config: Dict[str, Any]) -> bool:
        """Update analyzer configuration"""
        try:
            if 'analysis_interval' in new_config:
                self.analysis_interval = int(new_config['analysis_interval'])
            
            if 'max_selected_symbols' in new_config:
                self.max_symbols = int(new_config['max_selected_symbols'])
            
            if 'correlation_threshold' in new_config:
                self.correlation_threshold = float(new_config['correlation_threshold'])
            
            self.logger.info("Analyzer configuration updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration update error: {e}")
            return False
    
    async def export_opportunities_csv(self, filepath: str) -> bool:
        """Export current opportunities to CSV"""
        try:
            import csv
            
            if not self.top_opportunities:
                self.logger.warning("No opportunities to export")
                return False
            
            # Prepare CSV data
            fieldnames = [
                'rank', 'symbol', 'current_price', 'volume', 'change_percent',
                'overall_score', 'confidence_level', 'risk_level', 'volatility',
                'regime', 'momentum_strength', 'position_size', 'stop_loss',
                'take_profit', 'max_holding_time', 'entry_range', 'exit_range'
            ]
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for symbol, opportunity in self.top_opportunities.items():
                    row = {
                        'rank': opportunity.rank,
                        'symbol': opportunity.symbol,
                        'current_price': opportunity.current_price,
                        'volume': opportunity.volume,
                        'change_percent': opportunity.change_percent,
                        'overall_score': opportunity.overall_score,
                        'confidence_level': opportunity.confidence_level,
                        'risk_level': opportunity.risk_level.value,
                        'volatility': opportunity.volatility_metrics.realized_volatility,
                        'regime': opportunity.volatility_metrics.current_regime.value,
                        'momentum_strength': opportunity.volatility_metrics.momentum_strength,
                        'position_size': opportunity.position_size_recommendation,
                        'stop_loss': opportunity.stop_loss_level,
                        'take_profit': opportunity.take_profit_level,
                        'max_holding_time': opportunity.max_holding_time,
                        'entry_range': f"{opportunity.entry_price_range[0]:.2f}-{opportunity.entry_price_range[1]:.2f}",
                        'exit_range': f"{opportunity.exit_price_range[0]:.2f}-{opportunity.exit_price_range[1]:.2f}"
                    }
                    writer.writerow(row)
            
            self.logger.info(f"Exported {len(self.top_opportunities)} opportunities to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"CSV export error: {e}")
            return False
    
    async def stop(self):
        """Stop analyzer and cleanup"""
        try:
            self.logger.info("Stopping Volatility Analyzer...")
            
            # Store final results
            if self.top_opportunities:
                await self._store_analysis_results(self.top_opportunities)
            
            # Clear data structures
            self.top_opportunities.clear()
            self.volatility_calculator.price_history.clear()
            self.volatility_calculator.return_history.clear()
            self.volatility_calculator.volatility_history.clear()
            
            self.logger.info("Volatility Analyzer stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping Volatility Analyzer: {e}")

# Singleton instance
_volatility_analyzer = None

def get_volatility_analyzer() -> VolatilityAnalyzer:
    """Get singleton volatility analyzer instance"""
    global _volatility_analyzer
    if _volatility_analyzer is None:
        _volatility_analyzer = VolatilityAnalyzer()
    return _volatility_analyzer

# Utility functions for integration
async def start_volatility_analysis_system(config_manager=None, database_manager=None,
                                         dynamic_scanner=None, security_manager=None) -> VolatilityAnalyzer:
    """Start the complete volatility analysis system"""
    try:
        # Initialize analyzer
        analyzer = VolatilityAnalyzer(config_manager, database_manager, dynamic_scanner, security_manager)
        
        # Initialize the system
        success = await analyzer.initialize()
        if not success:
            raise Exception("Volatility analyzer initialization failed")
        
        return analyzer
        
    except Exception as e:
        logger.error(f"Failed to start volatility analysis system: {e}")
        raise

async def test_volatility_analysis():
    """Test volatility analysis functionality"""
    try:
        print("🧪 Testing Volatility Analysis System...")
        
        # Start analyzer
        analyzer = await start_volatility_analysis_system()
        
        # Wait for initial analysis
        await asyncio.sleep(10)
        
        # Force an analysis
        opportunities = await analyzer.force_analysis()
        
        if opportunities:
            print(f"✅ Analysis successful: {len(opportunities)} opportunities found")
            
            # Get top opportunities
            top_opps = await analyzer.get_top_opportunities()
            print(f"🔝 Top opportunities:")
            for i, (symbol, opp_data) in enumerate(list(top_opps.items())[:5], 1):
                print(f"   {i}. {symbol}: Score={opp_data['overall_score']:.1f}, "
                      f"Risk={opp_data['risk_level']}, Vol={opp_data['volatility_metrics']['realized_volatility']:.1f}%")
            
            # Get statistics
            stats = await analyzer.get_analysis_statistics()
            print(f"📊 Analysis Stats: {stats}")
            
            # Test detailed analysis
            if opportunities:
                symbol = list(opportunities.keys())[0]
                details = await analyzer.get_opportunity_details(symbol)
                if details:
                    print(f"📋 Details for {symbol}:")
                    print(f"   Strategy: {details['detailed_analysis']['trading_strategy']['primary_strategy']}")
                    print(f"   Risk: {details['detailed_analysis']['risk_assessment']['risk_level']}")
                    print(f"   Regime: {details['volatility_metrics']['current_regime']}")
        
        else:
            print("❌ No analysis results found")
        
        print("✅ Test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

# Example usage and main function
async def main():
    """Example usage of volatility analyzer"""
    try:
        print("🚀 Starting Market-Ready Volatility Analysis System...")
        
        # Start analysis system
        analyzer = await start_volatility_analysis_system()
        
        print("✅ Volatility Analyzer started successfully")
        print(f"📊 Initial Stats: {await analyzer.get_analysis_statistics()}")
        
        # Run test
        await test_volatility_analysis()
        
        print("🔄 Analyzer running every 15 minutes... (Press Ctrl+C to stop)")
        
        try:
            # Monitor analysis every 5 minutes
            while True:
                await asyncio.sleep(300)  # 5 minutes
                
                stats = await analyzer.get_analysis_statistics()
                opportunities = await analyzer.get_top_opportunities()
                
                print(f"📊 Analysis Status: {len(opportunities)} opportunities, "
                      f"Next analysis in: {stats.get('next_analysis_in', 0):.1f} minutes")
                
                if opportunities:
                    top_3 = list(opportunities.items())[:3]
                    print(f"🔝 Top 3: {[(s, f\"{d['overall_score']:.0f}\") for s, d in top_3]}")
            
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            await analyzer.stop()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    # Run volatility analyzer
    asyncio.run(main())
