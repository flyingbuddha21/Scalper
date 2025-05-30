#!/usr/bin/env python3
"""
Market-Ready Dynamic Scanner
Real-time market scanning integrated with DataManager and WebSocket feeds
No mocking - purely adaptive to live market data
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import talib

@dataclass
class ScanResult:
    symbol: str
    opportunity_type: str
    probability: float
    entry_price: float
    target_price: float
    stop_loss: float
    volume_surge: float
    momentum_score: float
    volatility_rating: str
    strategy_signals: List[str]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class DynamicScanner:
    def __init__(self, config):
        """Initialize market scanner with live data integration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Integration with bot components
        self.data_manager = None
        self.websocket_manager = None
        self.volatility_analyzer = None
        
        # Real-time data storage
        self.price_data = defaultdict(lambda: deque(maxlen=100))
        self.volume_data = defaultdict(lambda: deque(maxlen=100))
        self.ohlc_data = defaultdict(lambda: deque(maxlen=50))
        self.tick_timestamps = defaultdict(lambda: deque(maxlen=100))
        
        # Market filters - adaptive to live conditions
        self.min_volume = config.get('min_volume', 50000)
        self.min_price = config.get('min_price', 10.0)
        self.max_price = config.get('max_price', 10000.0)
        self.min_market_cap = config.get('min_market_cap', 100000000)  # 100M
        
        # Technical indicators cache
        self.indicators_cache = defaultdict(dict)
        self.last_scan_time = {}
        
        # Scan results and tracking
        self.scan_results = deque(maxlen=1000)
        self.watchlist = set()
        self.blacklist = set()
        self.is_running = False
        
        # Adaptive scanning parameters
        self.volume_surge_threshold = 2.0
        self.momentum_threshold = 0.02
        self.breakout_threshold = 0.015
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        # Market regime awareness
        self.market_volatility = 0.02
        self.market_trend = 'NEUTRAL'
        
        # Performance tracking
        self.scan_stats = {
            'total_scans': 0,
            'opportunities_found': 0,
            'symbols_processed': 0,
            'last_scan_duration': 0
        }
        
        self.logger.info("Dynamic Scanner initialized for live market data")
    
    def set_data_manager(self, data_manager):
        """Set data manager reference"""
        self.data_manager = data_manager
        self.logger.info("Data manager connected to scanner")
    
    def set_websocket_manager(self, websocket_manager):
        """Set websocket manager reference"""
        self.websocket_manager = websocket_manager
        self.logger.info("WebSocket manager connected to scanner")
    
    def set_volatility_analyzer(self, volatility_analyzer):
        """Set volatility analyzer reference"""
        self.volatility_analyzer = volatility_analyzer
        self.logger.info("Volatility analyzer connected to scanner")
    
    async def start(self):
        """Start the dynamic scanner with live data"""
        try:
            if not self.data_manager:
                raise Exception("Data manager not connected")
            
            self.is_running = True
            
            # Initialize symbol universe from live data
            await self._initialize_symbol_universe()
            
            # Start background scanning tasks
            asyncio.create_task(self._continuous_scanning_loop())
            asyncio.create_task(self._market_regime_monitor())
            asyncio.create_task(self._watchlist_manager())
            
            self.logger.info("✅ Dynamic Scanner started with live data")
            
        except Exception as e:
            self.logger.error(f"Error starting scanner: {e}")
            raise
    
    async def stop(self):
        """Stop the scanner"""
        self.is_running = False
        self.logger.info("Dynamic Scanner stopped")
    
    async def _initialize_symbol_universe(self):
        """Initialize scanning universe from live market data"""
        try:
            if hasattr(self.data_manager, 'get_tradeable_symbols'):
                symbols = await self.data_manager.get_tradeable_symbols()
                self.logger.info(f"Initialized with {len(symbols)} tradeable symbols")
            
            # Apply initial filters
            await self._apply_initial_filters()
            
        except Exception as e:
            self.logger.error(f"Error initializing symbol universe: {e}")
    
    async def _apply_initial_filters(self):
        """Apply initial filters to symbol universe"""
        try:
            if not self.data_manager:
                return
            
            # Get market data for filtering
            all_symbols = await self.data_manager.get_available_symbols()
            
            filtered_symbols = []
            for symbol in all_symbols:
                try:
                    # Get recent data for filtering
                    recent_data = await self.data_manager.get_recent_data(symbol, periods=5)
                    if not recent_data:
                        continue
                    
                    # Price filter
                    latest_price = self._extract_price_from_data(recent_data[-1])
                    if not (self.min_price <= latest_price <= self.max_price):
                        continue
                    
                    # Volume filter
                    avg_volume = np.mean([self._extract_volume_from_data(d) for d in recent_data])
                    if avg_volume < self.min_volume:
                        continue
                    
                    filtered_symbols.append(symbol)
                    
                except Exception:
                    continue
            
            self.logger.info(f"Filtered to {len(filtered_symbols)} symbols for scanning")
            
        except Exception as e:
            self.logger.error(f"Error applying initial filters: {e}")
    
    async def update_market_data(self, market_data: Dict[str, Dict]):
        """Update scanner with real-time market data"""
        try:
            for symbol, data in market_data.items():
                await self._process_symbol_data(symbol, data)
            
            # Update market regime
            await self._update_market_regime(market_data)
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
    
    async def _process_symbol_data(self, symbol: str, data: Dict):
        """Process live symbol data"""
        try:
            price = self._extract_price_from_data(data)
            volume = self._extract_volume_from_data(data)
            timestamp = self._extract_timestamp_from_data(data)
            
            if price <= 0:
                return
            
            # Store data
            self.price_data[symbol].append(price)
            self.volume_data[symbol].append(volume)
            self.tick_timestamps[symbol].append(timestamp)
            
            # Extract OHLC if available
            ohlc = self._extract_ohlc_from_data(data)
            if ohlc:
                self.ohlc_data[symbol].append(ohlc)
            
            # Update technical indicators
            await self._update_indicators(symbol)
            
        except Exception as e:
            self.logger.error(f"Error processing data for {symbol}: {e}")
    
    def _extract_price_from_data(self, data: Dict) -> float:
        """Extract price from live data feed"""
        price_fields = [
            'price', 'last_price', 'ltp', 'close', 'last', 'current_price',
            'last_traded_price', 'market_price', 'trade_price'
        ]
        
        for field in price_fields:
            if field in data and data[field] is not None:
                try:
                    return float(data[field])
                except (ValueError, TypeError):
                    continue
        
        # Try nested structures
        if 'quote' in data:
            return self._extract_price_from_data(data['quote'])
        
        return 0.0
    
    def _extract_volume_from_data(self, data: Dict) -> int:
        """Extract volume from live data feed"""
        volume_fields = ['volume', 'vol', 'quantity', 'total_volume', 'day_volume']
        
        for field in volume_fields:
            if field in data and data[field] is not None:
                try:
                    return int(float(data[field]))
                except (ValueError, TypeError):
                    continue
        
        return 0
    
    def _extract_timestamp_from_data(self, data: Dict) -> datetime:
        """Extract timestamp from data"""
        timestamp_fields = ['timestamp', 'time', 'datetime', 'tick_time']
        
        for field in timestamp_fields:
            if field in data and data[field] is not None:
                try:
                    if isinstance(data[field], datetime):
                        return data[field]
                    elif isinstance(data[field], (int, float)):
                        return datetime.fromtimestamp(data[field])
                    else:
                        return datetime.fromisoformat(str(data[field]))
                except:
                    continue
        
        return datetime.now()
    
    def _extract_ohlc_from_data(self, data: Dict) -> Optional[Dict]:
        """Extract OHLC data if available"""
        try:
            ohlc = {}
            ohlc_fields = ['open', 'high', 'low', 'close']
            
            for field in ohlc_fields:
                if field in data and data[field] is not None:
                    ohlc[field] = float(data[field])
            
            if len(ohlc) == 4:
                return ohlc
            
            return None
            
        except:
            return None
    
    async def _update_indicators(self, symbol: str):
        """Update technical indicators for symbol"""
        try:
            prices = list(self.price_data[symbol])
            volumes = list(self.volume_data[symbol])
            
            if len(prices) < 14:  # Minimum for RSI
                return
            
            indicators = {}
            prices_np = np.array(prices, dtype=float)
            
            # Moving averages
            if len(prices) >= 20:
                indicators['sma_20'] = np.mean(prices[-20:])
                indicators['sma_5'] = np.mean(prices[-5:])
                indicators['sma_10'] = np.mean(prices[-10:])
            
            if len(prices) >= 50:
                indicators['sma_50'] = np.mean(prices[-50:])
            
            # RSI
            try:
                rsi = talib.RSI(prices_np, timeperiod=14)
                indicators['rsi'] = rsi[-1] if not np.isnan(rsi[-1]) else 50
            except:
                # Fallback RSI calculation
                indicators['rsi'] = self._calculate_simple_rsi(prices)
            
            # MACD
            try:
                macd, signal, histogram = talib.MACD(prices_np)
                indicators['macd'] = macd[-1] if not np.isnan(macd[-1]) else 0
                indicators['macd_signal'] = signal[-1] if not np.isnan(signal[-1]) else 0
                indicators['macd_histogram'] = histogram[-1] if not np.isnan(histogram[-1]) else 0
            except:
                indicators['macd'] = 0
                indicators['macd_signal'] = 0
                indicators['macd_histogram'] = 0
            
            # Volume indicators
            if len(volumes) >= 20:
                indicators['avg_volume_20'] = np.mean(volumes[-20:])
                indicators['volume_ratio'] = volumes[-1] / indicators['avg_volume_20'] if indicators['avg_volume_20'] > 0 else 1
            
            # Price momentum
            if len(prices) >= 5:
                indicators['momentum_5'] = (prices[-1] - prices[-5]) / prices[-5]
            
            if len(prices) >= 10:
                indicators['momentum_10'] = (prices[-1] - prices[-10]) / prices[-10]
            
            # Volatility
            if len(prices) >= 20:
                returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, min(21, len(prices)))]
                indicators['volatility'] = np.std(returns)
            
            # Support and resistance levels
            if len(prices) >= 20:
                indicators['support'] = min(prices[-20:])
                indicators['resistance'] = max(prices[-20:])
            
            # Store indicators
            self.indicators_cache[symbol] = indicators
            
        except Exception as e:
            self.logger.error(f"Error updating indicators for {symbol}: {e}")
    
    def _calculate_simple_rsi(self, prices: List[float], period: int = 14) -> float:
        """Simple RSI calculation fallback"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            gains = []
            losses = []
            
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            if len(gains) < period:
                return 50.0
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return max(0, min(100, rsi))
            
        except:
            return 50.0
    
    async def _continuous_scanning_loop(self):
        """Continuous market scanning loop"""
        while self.is_running:
            try:
                start_time = datetime.now()
                
                # Perform market scan
                opportunities = await self.scan_market()
                
                # Update statistics
                scan_duration = (datetime.now() - start_time).total_seconds()
                self.scan_stats['total_scans'] += 1
                self.scan_stats['opportunities_found'] += len(opportunities)
                self.scan_stats['last_scan_duration'] = scan_duration
                
                # Log results
                if opportunities:
                    self.logger.info(f"Scan completed: {len(opportunities)} opportunities found in {scan_duration:.2f}s")
                
                # Adaptive scan frequency based on market activity
                scan_interval = self._calculate_adaptive_scan_interval()
                await asyncio.sleep(scan_interval)
                
            except Exception as e:
                self.logger.error(f"Error in scanning loop: {e}")
                await asyncio.sleep(30)
    
    def _calculate_adaptive_scan_interval(self) -> int:
        """Calculate adaptive scan interval based on market conditions"""
        try:
            base_interval = 30  # Base 30 seconds
            
            # Adjust based on market volatility
            if self.market_volatility > 0.03:  # High volatility
                return max(10, base_interval // 2)
            elif self.market_volatility < 0.015:  # Low volatility
                return min(60, base_interval * 2)
            
            return base_interval
            
        except:
            return 30
    
    async def scan_market(self) -> List[ScanResult]:
        """Main market scanning function"""
        try:
            if not self.is_running:
                return []
            
            opportunities = []
            symbols_scanned = 0
            
            # Get active symbols from data manager
            active_symbols = list(self.price_data.keys())
            
            for symbol in active_symbols:
                try:
                    # Skip if insufficient data
                    if len(self.price_data[symbol]) < 20:
                        continue
                    
                    # Skip blacklisted symbols
                    if symbol in self.blacklist:
                        continue
                    
                    # Basic filters
                    if not await self._meets_scanning_criteria(symbol):
                        continue
                    
                    symbols_scanned += 1
                    
                    # Scan for different opportunity types
                    breakout_opp = await self._scan_breakout(symbol)
                    if breakout_opp:
                        opportunities.append(breakout_opp)
                    
                    momentum_opp = await self._scan_momentum(symbol)
                    if momentum_opp:
                        opportunities.append(momentum_opp)
                    
                    volume_opp = await self._scan_volume_surge(symbol)
                    if volume_opp:
                        opportunities.append(volume_opp)
                    
                    reversal_opp = await self._scan_reversal(symbol)
                    if reversal_opp:
                        opportunities.append(reversal_opp)
                    
                    gap_opp = await self._scan_gap_opportunities(symbol)
                    if gap_opp:
                        opportunities.append(gap_opp)
                    
                except Exception as e:
                    self.logger.debug(f"Error scanning {symbol}: {e}")
                    continue
            
            # Update statistics
            self.scan_stats['symbols_processed'] = symbols_scanned
            
            # Sort by probability and remove duplicates
            opportunities = self._deduplicate_opportunities(opportunities)
            opportunities.sort(key=lambda x: x.probability, reverse=True)
            
            # Store results
            self.scan_results.extend(opportunities)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error in market scan: {e}")
            return []
    
    async def _meets_scanning_criteria(self, symbol: str) -> bool:
        """Check if symbol meets basic scanning criteria"""
        try:
            current_price = list(self.price_data[symbol])[-1]
            
            # Price range check
            if not (self.min_price <= current_price <= self.max_price):
                return False
            
            # Volume check
            if symbol in self.indicators_cache:
                avg_volume = self.indicators_cache[symbol].get('avg_volume_20', 0)
                if avg_volume < self.min_volume:
                    return False
            
            # Get additional market data from volatility analyzer
            if self.volatility_analyzer:
                should_avoid, reason = await self.volatility_analyzer.should_avoid_symbol(symbol)
                if should_avoid:
                    return False
            
            return True
            
        except Exception:
            return False
    
    async def _scan_breakout(self, symbol: str) -> Optional[ScanResult]:
        """Scan for breakout opportunities"""
        try:
            if symbol not in self.indicators_cache:
                return None
            
            indicators = self.indicators_cache[symbol]
            prices = list(self.price_data[symbol])
            current_price = prices[-1]
            
            resistance = indicators.get('resistance', 0)
            if resistance == 0:
                return None
            
            # Check for breakout above resistance
            breakout_level = resistance * (1 + self.breakout_threshold)
            
            if current_price > breakout_level:
                volume_ratio = indicators.get('volume_ratio', 1)
                
                # Volume confirmation required
                if volume_ratio > 1.5:
                    # Additional confirmations
                    rsi = indicators.get('rsi', 50)
                    momentum = indicators.get('momentum_5', 0)
                    
                    # Calculate probability
                    vol_score = min((volume_ratio - 1.5) * 0.2, 0.3)
                    momentum_score = min(momentum * 10, 0.2)
                    rsi_score = 0.1 if 40 < rsi < 80 else 0
                    
                    probability = min(0.95, 0.5 + vol_score + momentum_score + rsi_score)
                    
                    return ScanResult(
                        symbol=symbol,
                        opportunity_type='BREAKOUT',
                        probability=probability,
                        entry_price=current_price,
                        target_price=current_price * 1.03,
                        stop_loss=resistance * 0.98,
                        volume_surge=volume_ratio,
                        momentum_score=momentum,
                        volatility_rating=self._get_volatility_rating(symbol),
                        strategy_signals=['BREAKOUT', 'VOLUME_SURGE', 'RESISTANCE_BREAK']
                    )
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error scanning breakout for {symbol}: {e}")
            return None
    
    async def _scan_momentum(self, symbol: str) -> Optional[ScanResult]:
        """Scan for momentum opportunities"""
        try:
            if symbol not in self.indicators_cache:
                return None
            
            indicators = self.indicators_cache[symbol]
            current_price = list(self.price_data[symbol])[-1]
            
            momentum_5 = indicators.get('momentum_5', 0)
            momentum_10 = indicators.get('momentum_10', 0)
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            
            # Strong momentum criteria
            if (momentum_5 > self.momentum_threshold and 
                momentum_10 > 0 and 
                30 < rsi < 75 and 
                macd > macd_signal):
                
                volume_ratio = indicators.get('volume_ratio', 1)
                
                # Calculate probability
                momentum_strength = min(momentum_5 * 10, 0.4)
                rsi_score = 0.2 if 40 < rsi < 70 else 0.1
                macd_score = 0.1 if macd > macd_signal else 0
                volume_score = min((volume_ratio - 1) * 0.1, 0.2)
                
                probability = min(0.9, 0.3 + momentum_strength + rsi_score + macd_score + volume_score)
                
                # Dynamic target based on momentum strength
                target_multiplier = 1 + min(momentum_5 * 2, 0.05)
                
                return ScanResult(
                    symbol=symbol,
                    opportunity_type='MOMENTUM',
                    probability=probability,
                    entry_price=current_price,
                    target_price=current_price * target_multiplier,
                    stop_loss=current_price * 0.98,
                    volume_surge=volume_ratio,
                    momentum_score=momentum_5,
                    volatility_rating=self._get_volatility_rating(symbol),
                    strategy_signals=['MOMENTUM', 'RSI_FAVORABLE', 'MACD_BULLISH']
                )
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error scanning momentum for {symbol}: {e}")
            return None
    
    async def _scan_volume_surge(self, symbol: str) -> Optional[ScanResult]:
        """Scan for volume surge opportunities"""
        try:
            if symbol not in self.indicators_cache:
                return None
            
            indicators = self.indicators_cache[symbol]
            volume_ratio = indicators.get('volume_ratio', 1)
            
            if volume_ratio > self.volume_surge_threshold:
                current_price = list(self.price_data[symbol])[-1]
                momentum = indicators.get('momentum_5', 0)
                rsi = indicators.get('rsi', 50)
                
                # Positive price movement with volume
                if momentum > 0:
                    # Calculate probability
                    volume_score = min((volume_ratio - 2) * 0.15, 0.4)
                    momentum_score = min(momentum * 8, 0.3)
                    rsi_score = 0.1 if 30 < rsi < 80 else 0
                    
                    probability = min(0.85, 0.35 + volume_score + momentum_score + rsi_score)
                    
                    return ScanResult(
                        symbol=symbol,
                        opportunity_type='VOLUME_SURGE',
                        probability=probability,
                        entry_price=current_price,
                        target_price=current_price * 1.025,
                        stop_loss=current_price * 0.985,
                        volume_surge=volume_ratio,
                        momentum_score=momentum,
                        volatility_rating=self._get_volatility_rating(symbol),
                        strategy_signals=['VOLUME_SURGE', 'PRICE_MOMENTUM']
                    )
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error scanning volume surge for {symbol}: {e}")
            return None
    
    async def _scan_reversal(self, symbol: str) -> Optional[ScanResult]:
        """Scan for reversal opportunities"""
        try:
            if symbol not in self.indicators_cache:
                return None
            
            indicators = self.indicators_cache[symbol]
            rsi = indicators.get('rsi', 50)
            support = indicators.get('support', 0)
            current_price = list(self.price_data[symbol])[-1]
            
            # Oversold reversal
            if rsi < self.rsi_oversold and support > 0:
                support_distance = (current_price - support) / support
                
                # Near support level
                if support_distance < 0.02:  # Within 2% of support
                    volume_ratio = indicators.get('volume_ratio', 1)
                    
                    # Calculate probability
                    rsi_score = min((30 - rsi) * 0.02, 0.4)
                    support_score = 0.2 if support_distance < 0.01 else 0.1
                    volume_score = min((volume_ratio - 1) * 0.1, 0.2)
                    
                    probability = min(0.8, 0.3 + rsi_score + support_score + volume_score)
                    
                    return ScanResult(
                        symbol=symbol,
                        opportunity_type='REVERSAL',
                        probability=probability,
                        entry_price=current_price,
                        target_price=current_price * 1.04,
                        stop_loss=support * 0.98,
                        volume_surge=volume_ratio,
                        momentum_score=indicators.get('momentum_5', 0),
                        volatility_rating=self._get_volatility_rating(symbol),
                        strategy_signals=['OVERSOLD_RSI', 'SUPPORT_BOUNCE']
                    )
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error scanning reversal for {symbol}: {e}")
            return None
    
    async def _scan_gap_opportunities(self, symbol: str) -> Optional[ScanResult]:
        """Scan for gap up/down opportunities"""
        try:
            if len(self.price_data[symbol]) < 2:
                return None
            
            prices = list(self.price_data[symbol])
            current_price = prices[-1]
            previous_price = prices[-2]
            
            gap_percent = (current_price - previous_price) / previous_price
            
            # Significant gap (> 2%)
            if abs(gap_percent) > 0.02:
                indicators = self.indicators_cache.get(symbol, {})
                volume_ratio = indicators.get('volume_ratio', 1)
                rsi = indicators.get('rsi', 50)
                
                # Gap up with volume
                if gap_percent > 0 and volume_ratio > 1.5:
                    probability = min(0.75, 0.4 + min(gap_percent * 5, 0.2) + min((volume_ratio - 1.5) * 0.1, 0.15))
                    
                    return ScanResult(
                        symbol=symbol,
                        opportunity_type='GAP_UP',
                        probability=probability,
                        entry_price=current_price,
                        target_price=current_price * (1 + gap_percent * 0.5),
                        stop_loss=previous_price,
                        volume_surge=volume_ratio,
                        momentum_score=gap_percent,
                        volatility_rating=self._get_volatility_rating(symbol),
                        strategy_signals=['GAP_UP', 'VOLUME_CONFIRMATION']
                    )
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error scanning gap for {symbol}: {e}")
            return None
    
    def _get_volatility_rating(self, symbol: str) -> str:
        """Get volatility rating for symbol"""
        try:
            if self.volatility_analyzer:
                metrics = asyncio.create_task(self.volatility_analyzer.get_symbol_metrics(symbol))
                if hasattr(metrics, 'result') and metrics.result():
                    return metrics.result().risk_rating
            
            # Fallback to local calculation
            indicators = self.indicators_cache.get(symbol, {})
            volatility = indicators.get('volatility', 0.02)
            
            if volatility < 0.015:
                return 'LOW'
            elif volatility < 0.03:
                return 'MEDIUM'
            elif volatility < 0.05:
                return 'HIGH'
            else:
                return 'EXTREME'
                
        except:
            return 'MEDIUM'
    
    def _deduplicate_opportunities(self, opportunities: List[ScanResult]) -> List[ScanResult]:
        """Remove duplicate opportunities for same symbol"""
        try:
            symbol_best = {}
            
            for opp in opportunities:
                key = opp.symbol
                if key not in symbol_best or opp.probability > symbol_best[key].probability:
                    symbol_best[key] = opp
            
            return list(symbol_best.values())
            
        except Exception:
            return opportunities
    
    async def _update_market_regime(self, market_data: Dict[str, Dict]):
        """Update market regime based on overall market data"""
        try:
            if len(market_data) < 10:
                return
            
            # Calculate market-wide metrics
            price_changes = []
            volume_ratios = []
            
            for symbol, data in market_data.items():
                if symbol in self.indicators_cache:
                    indicators = self.indicators_cache[symbol]
                    momentum = indicators.get('momentum_5', 0)
                    vol_ratio = indicators.get('volume_ratio', 1)
                    
                  price_changes.append(momentum)
            volume_ratios.append(vol_ratio)
            
            if len(price_changes) >= 10:
                avg_momentum = np.mean(price_changes)
                avg_volume = np.mean(volume_ratios)
                volatility = np.std(price_changes)
                
                # Update market metrics
                self.market_volatility = volatility
                
                # Determine market trend
                if avg_momentum > 0.01:
                    self.market_trend = 'BULLISH'
                elif avg_momentum < -0.01:
                    self.market_trend = 'BEARISH'
                else:
                    self.market_trend = 'NEUTRAL'
                
                # Adjust scanning parameters based on regime
                await self._adjust_scanning_parameters(volatility, avg_volume)
                
        except Exception as e:
            self.logger.error(f"Error updating market regime: {e}")
    
    async def _adjust_scanning_parameters(self, market_volatility: float, avg_volume: float):
        """Adjust scanning parameters based on market conditions"""
        try:
            # Adjust thresholds based on market volatility
            if market_volatility > 0.03:  # High volatility market
                self.volume_surge_threshold = 1.5  # Lower threshold
                self.momentum_threshold = 0.015   # Lower momentum required
                self.breakout_threshold = 0.01    # Easier breakouts
            elif market_volatility < 0.01:  # Low volatility market
                self.volume_surge_threshold = 3.0  # Higher threshold
                self.momentum_threshold = 0.025   # Higher momentum required
                self.breakout_threshold = 0.02    # Stronger breakouts needed
            else:  # Normal market
                self.volume_surge_threshold = 2.0
                self.momentum_threshold = 0.02
                self.breakout_threshold = 0.015
            
            # Adjust RSI levels based on market trend
            if self.market_trend == 'BULLISH':
                self.rsi_oversold = 35  # Higher oversold level
                self.rsi_overbought = 75
            elif self.market_trend == 'BEARISH':
                self.rsi_oversold = 25  # Lower oversold level
                self.rsi_overbought = 65
            else:
                self.rsi_oversold = 30
                self.rsi_overbought = 70
                
        except Exception as e:
            self.logger.error(f"Error adjusting scanning parameters: {e}")
    
    async def _market_regime_monitor(self):
        """Monitor and log market regime changes"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Log market regime info
                self.logger.info(f"Market Regime - Trend: {self.market_trend}, "
                               f"Volatility: {self.market_volatility:.3f}, "
                               f"Volume Threshold: {self.volume_surge_threshold}")
                
            except Exception as e:
                self.logger.error(f"Error in market regime monitor: {e}")
                await asyncio.sleep(300)
    
    async def _watchlist_manager(self):
        """Manage dynamic watchlist based on scan results"""
        while self.is_running:
            try:
                await asyncio.sleep(600)  # Update every 10 minutes
                
                # Add high-probability symbols to watchlist
                recent_opportunities = [
                    opp for opp in self.scan_results 
                    if opp.timestamp > datetime.now() - timedelta(minutes=30)
                    and opp.probability > 0.7
                ]
                
                for opp in recent_opportunities:
                    self.watchlist.add(opp.symbol)
                
                # Remove symbols that haven't shown opportunities lately
                stale_symbols = []
                for symbol in self.watchlist:
                    recent_opps = [
                        opp for opp in self.scan_results
                        if opp.symbol == symbol 
                        and opp.timestamp > datetime.now() - timedelta(hours=2)
                    ]
                    
                    if not recent_opps:
                        stale_symbols.append(symbol)
                
                for symbol in stale_symbols:
                    self.watchlist.discard(symbol)
                
                self.logger.debug(f"Watchlist updated: {len(self.watchlist)} symbols")
                
            except Exception as e:
                self.logger.error(f"Error in watchlist manager: {e}")
                await asyncio.sleep(600)
    
    # Public API Methods
    async def get_scan_results(self, 
                             opportunity_type: Optional[str] = None, 
                             min_probability: float = 0.0,
                             max_age_minutes: int = 60) -> List[ScanResult]:
        """Get recent scan results with filters"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
            results = [
                r for r in self.scan_results 
                if r.timestamp > cutoff_time
            ]
            
            # Filter by opportunity type
            if opportunity_type:
                results = [r for r in results if r.opportunity_type == opportunity_type]
            
            # Filter by minimum probability
            if min_probability > 0:
                results = [r for r in results if r.probability >= min_probability]
            
            # Sort by probability
            results.sort(key=lambda x: x.probability, reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting scan results: {e}")
            return []
    
    async def get_top_opportunities(self, limit: int = 10) -> List[ScanResult]:
        """Get top opportunities by probability"""
        try:
            recent_results = await self.get_scan_results(max_age_minutes=30)
            return recent_results[:limit]
        except Exception as e:
            self.logger.error(f"Error getting top opportunities: {e}")
            return []
    
    async def get_opportunities_by_type(self) -> Dict[str, List[ScanResult]]:
        """Get opportunities grouped by type"""
        try:
            recent_results = await self.get_scan_results(max_age_minutes=60)
            
            by_type = defaultdict(list)
            for result in recent_results:
                by_type[result.opportunity_type].append(result)
            
            # Sort each type by probability
            for opp_type in by_type:
                by_type[opp_type].sort(key=lambda x: x.probability, reverse=True)
            
            return dict(by_type)
            
        except Exception as e:
            self.logger.error(f"Error getting opportunities by type: {e}")
            return {}
    
    async def get_watchlist_symbols(self, min_probability: float = 0.6) -> List[str]:
        """Get current watchlist symbols"""
        try:
            # Get symbols from recent high-probability scans
            high_prob_symbols = set()
            recent_results = await self.get_scan_results(
                min_probability=min_probability, 
                max_age_minutes=30
            )
            
            for result in recent_results:
                high_prob_symbols.add(result.symbol)
            
            # Combine with managed watchlist
            all_symbols = high_prob_symbols.union(self.watchlist)
            
            return sorted(list(all_symbols))
            
        except Exception as e:
            self.logger.error(f"Error getting watchlist symbols: {e}")
            return []
    
    async def add_to_watchlist(self, symbol: str):
        """Add symbol to watchlist"""
        try:
            self.watchlist.add(symbol)
            self.logger.info(f"Added {symbol} to watchlist")
        except Exception as e:
            self.logger.error(f"Error adding {symbol} to watchlist: {e}")
    
    async def remove_from_watchlist(self, symbol: str):
        """Remove symbol from watchlist"""
        try:
            self.watchlist.discard(symbol)
            self.logger.info(f"Removed {symbol} from watchlist")
        except Exception as e:
            self.logger.error(f"Error removing {symbol} from watchlist: {e}")
    
    async def add_to_blacklist(self, symbol: str, reason: str = "Manual"):
        """Add symbol to blacklist"""
        try:
            self.blacklist.add(symbol)
            self.logger.info(f"Added {symbol} to blacklist - Reason: {reason}")
        except Exception as e:
            self.logger.error(f"Error adding {symbol} to blacklist: {e}")
    
    async def remove_from_blacklist(self, symbol: str):
        """Remove symbol from blacklist"""
        try:
            self.blacklist.discard(symbol)
            self.logger.info(f"Removed {symbol} from blacklist")
        except Exception as e:
            self.logger.error(f"Error removing {symbol} from blacklist: {e}")
    
    async def get_scanner_stats(self) -> Dict:
        """Get comprehensive scanner statistics"""
        try:
            total_symbols = len(self.price_data)
            recent_opportunities = len([
                r for r in self.scan_results 
                if r.timestamp > datetime.now() - timedelta(hours=1)
            ])
            
            # Opportunities by type
            opp_by_type = defaultdict(int)
            for result in self.scan_results:
                if result.timestamp > datetime.now() - timedelta(hours=24):
                    opp_by_type[result.opportunity_type] += 1
            
            # Average probability
            recent_results = [
                r for r in self.scan_results 
                if r.timestamp > datetime.now() - timedelta(hours=6)
            ]
            avg_probability = np.mean([r.probability for r in recent_results]) if recent_results else 0
            
            return {
                'timestamp': datetime.now().isoformat(),
                'scanner_status': 'RUNNING' if self.is_running else 'STOPPED',
                'market_regime': {
                    'trend': self.market_trend,
                    'volatility': self.market_volatility,
                    'volume_threshold': self.volume_surge_threshold
                },
                'scanning_stats': self.scan_stats.copy(),
                'symbols': {
                    'total_monitored': total_symbols,
                    'watchlist_size': len(self.watchlist),
                    'blacklist_size': len(self.blacklist)
                },
                'opportunities': {
                    'total_24h': len([r for r in self.scan_results if r.timestamp > datetime.now() - timedelta(hours=24)]),
                    'recent_1h': recent_opportunities,
                    'by_type_24h': dict(opp_by_type),
                    'average_probability_6h': avg_probability
                },
                'thresholds': {
                    'volume_surge': self.volume_surge_threshold,
                    'momentum': self.momentum_threshold,
                    'breakout': self.breakout_threshold,
                    'rsi_oversold': self.rsi_oversold,
                    'rsi_overbought': self.rsi_overbought
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting scanner stats: {e}")
            return {'error': str(e)}
    
    async def get_symbol_analysis(self, symbol: str) -> Dict:
        """Get detailed analysis for specific symbol"""
        try:
            if symbol not in self.indicators_cache:
                return {'error': f'No data available for {symbol}'}
            
            indicators = self.indicators_cache[symbol]
            prices = list(self.price_data[symbol])
            volumes = list(self.volume_data[symbol])
            
            # Recent scan results for this symbol
            symbol_opportunities = [
                r for r in self.scan_results 
                if r.symbol == symbol and r.timestamp > datetime.now() - timedelta(hours=24)
            ]
            
            # Price analysis
            price_change_1 = (prices[-1] - prices[-2]) / prices[-2] if len(prices) >= 2 else 0
            price_change_5 = indicators.get('momentum_5', 0)
            
            # Volume analysis
            current_volume = volumes[-1] if volumes else 0
            avg_volume = indicators.get('avg_volume_20', 0)
            volume_ratio = indicators.get('volume_ratio', 1)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': prices[-1] if prices else 0,
                'price_analysis': {
                    'change_1_tick': price_change_1,
                    'change_5_ticks': price_change_5,
                    'support_level': indicators.get('support', 0),
                    'resistance_level': indicators.get('resistance', 0)
                },
                'volume_analysis': {
                    'current_volume': current_volume,
                    'average_volume_20': avg_volume,
                    'volume_ratio': volume_ratio,
                    'volume_rating': 'HIGH' if volume_ratio > 2 else 'NORMAL' if volume_ratio > 0.5 else 'LOW'
                },
                'technical_indicators': {
                    'rsi': indicators.get('rsi', 50),
                    'sma_20': indicators.get('sma_20', 0),
                    'sma_50': indicators.get('sma_50', 0),
                    'macd': indicators.get('macd', 0),
                    'macd_signal': indicators.get('macd_signal', 0),
                    'volatility': indicators.get('volatility', 0)
                },
                'scan_results': {
                    'total_opportunities_24h': len(symbol_opportunities),
                    'latest_opportunity': symbol_opportunities[0].__dict__ if symbol_opportunities else None,
                    'opportunity_types': list(set([r.opportunity_type for r in symbol_opportunities]))
                },
                'status': {
                    'in_watchlist': symbol in self.watchlist,
                    'in_blacklist': symbol in self.blacklist,
                    'volatility_rating': self._get_volatility_rating(symbol)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting symbol analysis for {symbol}: {e}")
            return {'error': str(e)}
    
    async def update_scanning_parameters(self, params: Dict):
        """Update scanning parameters"""
        try:
            updated_params = []
            
            if 'min_volume' in params:
                self.min_volume = params['min_volume']
                updated_params.append('min_volume')
            
            if 'volume_surge_threshold' in params:
                self.volume_surge_threshold = params['volume_surge_threshold']
                updated_params.append('volume_surge_threshold')
            
            if 'momentum_threshold' in params:
                self.momentum_threshold = params['momentum_threshold']
                updated_params.append('momentum_threshold')
            
            if 'breakout_threshold' in params:
                self.breakout_threshold = params['breakout_threshold']
                updated_params.append('breakout_threshold')
            
            if 'rsi_oversold' in params:
                self.rsi_oversold = params['rsi_oversold']
                updated_params.append('rsi_oversold')
            
            if 'rsi_overbought' in params:
                self.rsi_overbought = params['rsi_overbought']
                updated_params.append('rsi_overbought')
            
            self.logger.info(f"Scanner parameters updated: {updated_params}")
            
        except Exception as e:
            self.logger.error(f"Error updating scanning parameters: {e}")
            raise
    
    async def export_opportunities_csv(self, filepath: str, hours_back: int = 24):
        """Export recent opportunities to CSV"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            opportunities = [
                r for r in self.scan_results 
                if r.timestamp > cutoff_time
            ]
            
            if not opportunities:
                self.logger.warning("No opportunities to export")
                return
            
            # Convert to DataFrame
            data = []
            for opp in opportunities:
                data.append({
                    'timestamp': opp.timestamp.isoformat(),
                    'symbol': opp.symbol,
                    'opportunity_type': opp.opportunity_type,
                    'probability': opp.probability,
                    'entry_price': opp.entry_price,
                    'target_price': opp.target_price,
                    'stop_loss': opp.stop_loss,
                    'volume_surge': opp.volume_surge,
                    'momentum_score': opp.momentum_score,
                    'volatility_rating': opp.volatility_rating,
                    'strategy_signals': '|'.join(opp.strategy_signals)
                })
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            
            self.logger.info(f"Exported {len(opportunities)} opportunities to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting opportunities: {e}")
    
    async def get_performance_metrics(self) -> Dict:
        """Get scanner performance metrics"""
        try:
            # Calculate success rate (opportunities that would have been profitable)
            profitable_opportunities = 0
            total_opportunities = 0
            
            # Look at opportunities from 2+ hours ago to allow time for price movement
            cutoff_time = datetime.now() - timedelta(hours=2)
            old_opportunities = [
                r for r in self.scan_results 
                if r.timestamp < cutoff_time and r.timestamp > datetime.now() - timedelta(hours=24)
            ]
            
            for opp in old_opportunities:
                total_opportunities += 1
                
                # Check if target was hit (simplified check)
                if symbol in self.price_data:
                    current_price = list(self.price_data[opp.symbol])[-1] if self.price_data[opp.symbol] else 0
                    
                    if current_price > 0:
                        price_change = (current_price - opp.entry_price) / opp.entry_price
                        target_change = (opp.target_price - opp.entry_price) / opp.entry_price
                        
                        if price_change >= target_change * 0.5:  # At least 50% of target achieved
                            profitable_opportunities += 1
            
            success_rate = (profitable_opportunities / total_opportunities) if total_opportunities > 0 else 0
            
            return {
                'timestamp': datetime.now().isoformat(),
                'scanning_performance': {
                    'total_scans_24h': len([r for r in self.scan_results if r.timestamp > datetime.now() - timedelta(hours=24)]),
                    'opportunities_per_scan': len(self.scan_results) / max(self.scan_stats['total_scans'], 1),
                    'average_scan_duration': self.scan_stats['last_scan_duration'],
                    'symbols_per_scan': self.scan_stats['symbols_processed']
                },
                'opportunity_performance': {
                    'total_opportunities_analyzed': total_opportunities,
                    'profitable_opportunities': profitable_opportunities,
                    'success_rate': success_rate,
                    'average_probability': np.mean([r.probability for r in old_opportunities]) if old_opportunities else 0
                },
                'efficiency_metrics': {
                    'opportunities_per_symbol': len(self.scan_results) / max(len(self.price_data), 1),
                    'high_probability_ratio': len([r for r in self.scan_results if r.probability > 0.7]) / max(len(self.scan_results), 1),
                    'watchlist_hit_rate': len([r for r in self.scan_results if r.symbol in self.watchlist]) / max(len(self.scan_results), 1)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {'error': str(e)}

# Integration helper for bot_core.py
async def initialize_dynamic_scanner(config, data_manager, websocket_manager, volatility_analyzer):
    """Initialize dynamic scanner with proper connections"""
    scanner = DynamicScanner(config)
    scanner.set_data_manager(data_manager)
    scanner.set_websocket_manager(websocket_manager)
    scanner.set_volatility_analyzer(volatility_analyzer)
    await scanner.start()
    return scanner

if __name__ == "__main__":
    # Test scanner
    async def test_scanner():
        config = {
            'min_volume': 50000,
            'min_price': 10.0,
            'max_price': 5000.0
        }
        
        scanner = DynamicScanner(config)
        
        # Test with sample data
        sample_data = {
            'RELIANCE': {
                'price': 2500.0,
                'volume': 1500000,
                'timestamp': datetime.now()
            },
            'TCS': {
                'price': 3200.0,
                'volume': 800000,
                'timestamp': datetime.now()
            }
        }
        
        await scanner.update_market_data(sample_data)
        
        # Get stats
        stats = await scanner.get_scanner_stats()
        print("Scanner Stats:", stats)
    
    # Run test
    asyncio.run(test_scanner())
