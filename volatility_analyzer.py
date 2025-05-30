#!/usr/bin/env python3
"""
Market-Ready Volatility and Risk Analyzer
Direct integration with DataManager and WebSocket feeds
No mocking - purely adaptive to real market data
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

@dataclass
class VolatilityMetrics:
    symbol: str
    current_volatility: float
    historical_volatility: float
    volatility_percentile: float
    beta: float
    var_95: float
    var_99: float
    sharpe_ratio: float
    max_drawdown: float
    volatility_trend: str
    risk_rating: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class RiskAlert:
    symbol: str
    alert_type: str
    severity: str
    message: str
    current_value: float
    threshold: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class VolatilityAnalyzer:
    def __init__(self, config):
        """Initialize volatility analyzer with real market data integration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Integration with bot components
        self.data_manager = None  # Will be set by bot_core
        self.websocket_manager = None  # Will be set by bot_core
        
        # Real-time data storage - no pre-filled data
        self.price_data = defaultdict(lambda: deque(maxlen=500))
        self.returns_data = defaultdict(lambda: deque(maxlen=500))
        self.volume_data = defaultdict(lambda: deque(maxlen=500))
        self.tick_data = defaultdict(lambda: deque(maxlen=1000))  # High-frequency data
        
        # Market benchmark - auto-detected from live feed
        self.market_returns = deque(maxlen=250)
        self.market_symbol = None
        self.market_candidates = ['NIFTY', 'NIFTY50', 'SENSEX', 'BANKNIFTY', '^NSEI']
        self.market_proxy_symbols = set()  # Top liquid symbols for proxy
        
        # Volatility calculations cache
        self.volatility_cache = {}
        self.last_update = {}
        
        # Adaptive risk thresholds - adjust based on market regime
        self.base_thresholds = {
            'low': 0.015,
            'medium': 0.03, 
            'high': 0.05,
            'extreme': 0.08
        }
        self.volatility_thresholds = self.base_thresholds.copy()
        
        # Alert system
        self.risk_alerts = deque(maxlen=100)
        self.alert_cooldown = defaultdict(lambda: datetime.min)
        
        # Market regime detection
        self.market_volatility_history = deque(maxlen=100)
        self.current_market_regime = 'NORMAL'
        
        # Data quality tracking
        self.data_quality_scores = defaultdict(float)
        self.last_data_update = defaultdict(datetime)
        
        self.logger.info("Volatility Analyzer initialized - ready for live market data")
    
    def set_data_manager(self, data_manager):
        """Set data manager reference from bot_core"""
        self.data_manager = data_manager
        self.logger.info("Data manager connected to volatility analyzer")
    
    def set_websocket_manager(self, websocket_manager):
        """Set websocket manager reference from bot_core"""
        self.websocket_manager = websocket_manager
        self.logger.info("WebSocket manager connected to volatility analyzer")
    
    async def start_real_time_processing(self):
        """Start processing real-time market data"""
        try:
            if not self.data_manager:
                raise Exception("Data manager not connected")
            
            # Subscribe to real-time data updates
            await self._subscribe_to_data_feeds()
            
            # Start background processing tasks
            asyncio.create_task(self._continuous_processing_loop())
            asyncio.create_task(self._market_regime_monitor())
            asyncio.create_task(self._adaptive_threshold_adjuster())
            
            self.logger.info("✅ Real-time volatility processing started")
            
        except Exception as e:
            self.logger.error(f"Failed to start real-time processing: {e}")
            raise
    
    async def _subscribe_to_data_feeds(self):
        """Subscribe to real-time data feeds"""
        try:
            # Get live symbols from data manager
            if hasattr(self.data_manager, 'get_active_symbols'):
                active_symbols = await self.data_manager.get_active_symbols()
                self.logger.info(f"Monitoring {len(active_symbols)} symbols for volatility")
            
            # Auto-detect market benchmark from available symbols
            await self._detect_market_benchmark()
            
        except Exception as e:
            self.logger.error(f"Error subscribing to data feeds: {e}")
    
    async def _detect_market_benchmark(self):
        """Auto-detect market benchmark from live data"""
        try:
            if not self.data_manager:
                return
            
            # Check for standard market indices in live feed
            available_symbols = await self.data_manager.get_available_symbols()
            
            for candidate in self.market_candidates:
                if candidate in available_symbols:
                    self.market_symbol = candidate
                    self.logger.info(f"Market benchmark detected: {self.market_symbol}")
                    return
            
            # If no index found, create market proxy from most liquid stocks
            if available_symbols and len(available_symbols) >= 10:
                # Select top 10 most active symbols as market proxy
                volume_data = {}
                for symbol in list(available_symbols)[:50]:  # Check first 50 symbols
                    try:
                        recent_data = await self.data_manager.get_recent_data(symbol, periods=5)
                        if recent_data and 'volume' in recent_data:
                            avg_volume = np.mean([d.get('volume', 0) for d in recent_data])
                            volume_data[symbol] = avg_volume
                    except:
                        continue
                
                # Select top 10 by volume for market proxy
                if volume_data:
                    sorted_symbols = sorted(volume_data.items(), key=lambda x: x[1], reverse=True)
                    self.market_proxy_symbols = set([s[0] for s in sorted_symbols[:10]])
                    self.market_symbol = 'MARKET_PROXY'
                    self.logger.info(f"Market proxy created from top 10 liquid symbols")
            
        except Exception as e:
            self.logger.error(f"Error detecting market benchmark: {e}")
    
    async def update(self, market_data: Dict[str, Dict]):
        """Update with real-time market data from data manager"""
        try:
            current_time = datetime.now()
            
            for symbol, data in market_data.items():
                await self._process_live_data(symbol, data, current_time)
            
            # Update market benchmark
            await self._update_market_benchmark(market_data)
            
            # Update market volatility tracking
            await self._update_market_volatility_tracking()
            
        except Exception as e:
            self.logger.error(f"Error updating with market data: {e}")
    
    async def _process_live_data(self, symbol: str, data: Dict, timestamp: datetime):
        """Process live market data for a symbol"""
        try:
            # Extract price from whatever format the feed provides
            price = self._extract_price_from_feed(data)
            volume = self._extract_volume_from_feed(data)
            
            if price <= 0:
                return
            
            # Store tick data
            tick_info = {
                'price': price,
                'volume': volume,
                'timestamp': timestamp
            }
            self.tick_data[symbol].append(tick_info)
            
            # Update price series
            self.price_data[symbol].append(price)
            self.volume_data[symbol].append(volume)
            
            # Calculate returns if we have previous price
            if len(self.price_data[symbol]) > 1:
                prev_price = list(self.price_data[symbol])[-2]
                if prev_price > 0:
                    return_pct = (price - prev_price) / prev_price
                    self.returns_data[symbol].append(return_pct)
            
            # Update data quality score
            self._update_data_quality(symbol, data)
            
            # Calculate volatility if we have enough data
            if len(self.returns_data[symbol]) >= 20:
                await self._calculate_real_time_volatility(symbol)
            
            # Check for risk alerts
            await self._check_real_time_alerts(symbol)
            
        except Exception as e:
            self.logger.error(f"Error processing live data for {symbol}: {e}")
    
    def _extract_price_from_feed(self, data: Dict) -> float:
        """Extract price from any market data format"""
        # Try all possible price field names
        price_fields = [
            'price', 'last_price', 'ltp', 'close', 'last', 'current_price',
            'last_traded_price', 'market_price', 'trade_price', 'bid', 'ask'
        ]
        
        for field in price_fields:
            if field in data:
                try:
                    value = data[field]
                    if value is not None and value > 0:
                        return float(value)
                except (ValueError, TypeError):
                    continue
        
        # Try nested structures
        if 'quote' in data and isinstance(data['quote'], dict):
            return self._extract_price_from_feed(data['quote'])
        
        if 'tick' in data and isinstance(data['tick'], dict):
            return self._extract_price_from_feed(data['tick'])
        
        # Try OHLC data
        ohlc_fields = ['close', 'high', 'low', 'open']
        for field in ohlc_fields:
            if field in data:
                try:
                    value = data[field]
                    if value is not None and value > 0:
                        return float(value)
                except (ValueError, TypeError):
                    continue
        
        return 0.0
    
    def _extract_volume_from_feed(self, data: Dict) -> int:
        """Extract volume from any market data format"""
        volume_fields = [
            'volume', 'vol', 'quantity', 'qty', 'total_volume',
            'day_volume', 'traded_volume', 'market_volume'
        ]
        
        for field in volume_fields:
            if field in data:
                try:
                    value = data[field]
                    if value is not None:
                        return int(float(value))
                except (ValueError, TypeError):
                    continue
        
        # Check nested structures
        if 'quote' in data and isinstance(data['quote'], dict):
            return self._extract_volume_from_feed(data['quote'])
        
        return 0
    
    def _update_data_quality(self, symbol: str, data: Dict):
        """Update data quality score based on completeness"""
        try:
            score = 0.0
            total_fields = 0
            
            # Check for essential fields
            essential_fields = ['price', 'volume', 'timestamp']
            for field in essential_fields:
                total_fields += 1
                if self._has_valid_field(data, field):
                    score += 1.0
            
            # Check for additional fields
            additional_fields = ['open', 'high', 'low', 'close', 'bid', 'ask']
            for field in additional_fields:
                total_fields += 0.5
                if self._has_valid_field(data, field):
                    score += 0.5
            
            # Calculate quality score (0-1)
            quality_score = score / total_fields if total_fields > 0 else 0
            self.data_quality_scores[symbol] = quality_score
            self.last_data_update[symbol] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating data quality for {symbol}: {e}")
    
    def _has_valid_field(self, data: Dict, field: str) -> bool:
        """Check if data has valid field"""
        if field in data and data[field] is not None:
            try:
                float(data[field])
                return True
            except:
                pass
        return False
    
    async def _calculate_real_time_volatility(self, symbol: str):
        """Calculate volatility metrics from live data"""
        try:
            returns = list(self.returns_data[symbol])
            prices = list(self.price_data[symbol])
            
            if len(returns) < 20:
                return
            
            # Current volatility (rolling 20-period)
            recent_returns = returns[-20:]
            current_vol = np.std(recent_returns) * np.sqrt(252)  # Annualized
            
            # Historical volatility (all available data)
            historical_vol = np.std(returns) * np.sqrt(252)
            
            # Volatility percentile
            vol_percentile = self._calculate_volatility_percentile(returns, current_vol)
            
            # Beta calculation
            beta = await self._calculate_live_beta(symbol, returns)
            
            # Value at Risk
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            
            # Maximum drawdown
            max_drawdown = self._calculate_max_drawdown(prices)
            
            # Volatility trend
            vol_trend = self._determine_volatility_trend(returns)
            
            # Risk rating (adaptive to market regime)
            risk_rating = self._determine_adaptive_risk_rating(current_vol)
            
            # Create metrics
            metrics = VolatilityMetrics(
                symbol=symbol,
                current_volatility=current_vol,
                historical_volatility=historical_vol,
                volatility_percentile=vol_percentile,
                beta=beta,
                var_95=var_95,
                var_99=var_99,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                volatility_trend=vol_trend,
                risk_rating=risk_rating
            )
            
            # Cache metrics
            self.volatility_cache[symbol] = metrics
            self.last_update[symbol] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility for {symbol}: {e}")
    
    def _calculate_volatility_percentile(self, returns: List[float], current_vol: float) -> float:
        """Calculate volatility percentile from historical data"""
        try:
            if len(returns) < 40:
                return 50.0
            
            vol_series = []
            for i in range(20, len(returns), 5):  # Calculate every 5 periods
                period_returns = returns[max(0, i-20):i]
                if len(period_returns) >= 10:
                    period_vol = np.std(period_returns) * np.sqrt(252)
                    vol_series.append(period_vol)
            
            if len(vol_series) < 5:
                return 50.0
            
            # Calculate percentile
            sorted_vols = sorted(vol_series)
            position = 0
            for vol in sorted_vols:
                if current_vol > vol:
                    position += 1
                else:
                    break
            
            percentile = (position / len(sorted_vols)) * 100
            return min(99.0, max(1.0, percentile))
            
        except Exception:
            return 50.0
    
    async def _calculate_live_beta(self, symbol: str, returns: List[float]) -> float:
        """Calculate beta using live market data"""
        try:
            if len(self.market_returns) < 20 or len(returns) < 20:
                return 1.0
            
            # Align data lengths
            min_length = min(len(returns), len(self.market_returns))
            symbol_returns = returns[-min_length:]
            market_returns = list(self.market_returns)[-min_length:]
            
            # Calculate beta
            if len(symbol_returns) >= 10 and len(market_returns) >= 10:
                covariance = np.cov(symbol_returns, market_returns)[0][1]
                market_variance = np.var(market_returns)
                
                if market_variance > 0:
                    beta = covariance / market_variance
                    return max(0, min(5.0, beta))  # Cap between 0 and 5
            
            return 1.0
            
        except Exception:
            return 1.0
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) < 20:
                return 0.0
            
            # Risk-free rate (configurable, default 6% annually)
            risk_free_rate = self.config.get('risk_free_rate', 0.06) / 252
            
            excess_returns = [r - risk_free_rate for r in returns]
            mean_excess = np.mean(excess_returns)
            std_excess = np.std(excess_returns)
            
            if std_excess > 0:
                return (mean_excess / std_excess) * np.sqrt(252)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_max_drawdown(self, prices: List[float]) -> float:
        """Calculate maximum drawdown"""
        try:
            if len(prices) < 2:
                return 0.0
            
            peak = prices[0]
            max_dd = 0.0
            
            for price in prices[1:]:
                if price > peak:
                    peak = price
                
                drawdown = (peak - price) / peak if peak > 0 else 0
                max_dd = max(max_dd, drawdown)
            
            return max_dd
            
        except Exception:
            return 0.0
    
    def _determine_volatility_trend(self, returns: List[float]) -> str:
        """Determine volatility trend"""
        try:
            if len(returns) < 40:
                return 'STABLE'
            
            # Compare recent vs previous volatility
            recent_vol = np.std(returns[-20:])
            previous_vol = np.std(returns[-40:-20])
            
            if previous_vol > 0:
                change = (recent_vol - previous_vol) / previous_vol
                
                if change > 0.2:
                    return 'INCREASING'
                elif change < -0.2:
                    return 'DECREASING'
            
            return 'STABLE'
            
        except Exception:
            return 'STABLE'
    
    def _determine_adaptive_risk_rating(self, volatility: float) -> str:
        """Determine risk rating adaptive to market regime"""
        try:
            # Use adaptive thresholds based on market regime
            thresholds = self.volatility_thresholds
            
            if volatility <= thresholds['low']:
                return 'LOW'
            elif volatility <= thresholds['medium']:
                return 'MEDIUM'
            elif volatility <= thresholds['high']:
                return 'HIGH'
            else:
                return 'EXTREME'
                
        except Exception:
            return 'MEDIUM'
    
    async def _update_market_benchmark(self, market_data: Dict[str, Dict]):
        """Update market benchmark from live data"""
        try:
            if self.market_symbol == 'MARKET_PROXY':
                # Calculate market proxy return
                await self._calculate_market_proxy_return(market_data)
            elif self.market_symbol and self.market_symbol in market_data:
                # Direct market index data
                price = self._extract_price_from_feed(market_data[self.market_symbol])
                if price > 0:
                    if hasattr(self, '_last_market_price') and self._last_market_price > 0:
                        market_return = (price - self._last_market_price) / self._last_market_price
                        self.market_returns.append(market_return)
                    
                    self._last_market_price = price
            
        except Exception as e:
            self.logger.error(f"Error updating market benchmark: {e}")
    
    async def _calculate_market_proxy_return(self, market_data: Dict[str, Dict]):
        """Calculate market proxy return from liquid symbols"""
        try:
            if not self.market_proxy_symbols:
                return
            
            returns = []
            for symbol in self.market_proxy_symbols:
                if symbol in market_data and symbol in self.returns_data:
                    if len(self.returns_data[symbol]) > 0:
                        returns.append(self.returns_data[symbol][-1])
            
            if len(returns) >= 3:  # Need at least 3 symbols for proxy
                market_return = np.mean(returns)
                self.market_returns.append(market_return)
            
        except Exception as e:
            self.logger.error(f"Error calculating market proxy: {e}")
    
    async def _continuous_processing_loop(self):
        """Continuous background processing"""
        while True:
            try:
                # Clean old data
                await self._cleanup_old_data()
                
                # Update data quality metrics
                await self._update_overall_data_quality()
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                self.logger.error(f"Error in continuous processing: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_data(self):
        """Clean up old data to manage memory"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            for symbol in list(self.tick_data.keys()):
                # Keep only recent tick data
                recent_ticks = [
                    tick for tick in self.tick_data[symbol] 
                    if tick.get('timestamp', datetime.min) > cutoff_time
                ]
                self.tick_data[symbol] = deque(recent_ticks, maxlen=1000)
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    # Public API methods
    async def get_symbol_volatility(self, symbol: str) -> Optional[float]:
        """Get current volatility for symbol"""
        if symbol in self.volatility_cache:
            return self.volatility_cache[symbol].current_volatility
        return None
    
    async def get_symbol_metrics(self, symbol: str) -> Optional[VolatilityMetrics]:
        """Get complete metrics for symbol"""
        return self.volatility_cache.get(symbol)
    
    async def get_all_metrics(self) -> Dict[str, VolatilityMetrics]:
        """Get all volatility metrics"""
        return self.volatility_cache.copy()
    
    async def is_market_ready(self) -> bool:
        """Check if analyzer is ready with live market data"""
        return (
            len(self.volatility_cache) > 0 and
            self.data_manager is not None and
            len(self.price_data) > 0
        )
    
    async def get_data_quality_report(self) -> Dict:
        """Get data quality report"""
        try:
            total_symbols = len(self.data_quality_scores)
            if total_symbols == 0:
                return {'status': 'NO_DATA', 'message': 'No market data received yet'}
            
            avg_quality = np.mean(list(self.data_quality_scores.values()))
            recent_updates = sum(
                1 for timestamp in self.last_data_update.values()
                if timestamp > datetime.now() - timedelta(minutes=5)
            )
            
            return {
                'status': 'LIVE' if avg_quality > 0.7 else 'DEGRADED',
                'total_symbols': total_symbols,
                'average_quality_score': avg_quality,
                'symbols_updated_recently': recent_updates,
                'market_benchmark': self.market_symbol,
                'market_regime': self.current_market_regime,
                'last_update': max(self.last_data_update.values()).isoformat() if self.last_data_update else None
            }
            
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}

# Integration helper for bot_core.py
async def initialize_volatility_analyzer(config, data_manager, websocket_manager):
    """Initialize volatility analyzer with proper connections"""
    analyzer = VolatilityAnalyzer(config)
    analyzer.set_data_manager(data_manager)
    analyzer.set_websocket_manager(websocket_manager)
    await analyzer.start_real_time_processing()
    return analyzer
