#!/usr/bin/env python3
"""
Execution Manager for Top 10 Active Stocks
Manages real-time execution of scalping strategies on selected stocks
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import json
from collections import deque
import queue

logger = logging.getLogger(__name__)

class ExecutionStatus(Enum):
    IDLE = "idle"
    MONITORING = "monitoring"
    SIGNAL_DETECTED = "signal_detected"
    ORDER_PLACED = "order_placed"
    POSITION_OPEN = "position_open"
    EXITING = "exiting"
    COMPLETED = "completed"
    ERROR = "error"

class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

@dataclass
class ExecutionStock:
    symbol: str
    asset_type: str
    scalping_score: float
    volatility_score: float
    liquidity_score: float
    
    # Real-time data
    current_price: float = 0.0
    bid_price: float = 0.0
    ask_price: float = 0.0
    volume: int = 0
    last_update: datetime = field(default_factory=datetime.now)
    
    # Trading state
    status: ExecutionStatus = ExecutionStatus.IDLE
    position_size: int = 0
    entry_price: float = 0.0
    current_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Strategy parameters
    stop_loss_pct: float = 0.5
    take_profit_pct: float = 1.0
    max_position_size: int = 100
    min_tick_size: float = 0.05
    
    # Performance tracking
    trades_today: int = 0
    profit_trades: int = 0
    loss_trades: int = 0
    total_pnl: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'asset_type': self.asset_type,
            'scalping_score': round(self.scalping_score, 2),
            'volatility_score': round(self.volatility_score, 2),
            'liquidity_score': round(self.liquidity_score, 2),
            'current_price': round(self.current_price, 2),
            'bid_price': round(self.bid_price, 2),
            'ask_price': round(self.ask_price, 2),
            'volume': self.volume,
            'last_update': self.last_update.isoformat(),
            'status': self.status.value,
            'position_size': self.position_size,
            'entry_price': round(self.entry_price, 2),
            'current_pnl': round(self.current_pnl, 2),
            'unrealized_pnl': round(self.unrealized_pnl, 2),
            'trades_today': self.trades_today,
            'profit_trades': self.profit_trades,
            'loss_trades': self.loss_trades,
            'total_pnl': round(self.total_pnl, 2)
        }

@dataclass
class TradingSignal:
    symbol: str
    signal_type: SignalType
    price: float
    confidence: float
    timestamp: datetime
    indicators: Dict[str, Any]
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'price': round(self.price, 2),
            'confidence': round(self.confidence, 2),
            'timestamp': self.timestamp.isoformat(),
            'indicators': self.indicators,
            'stop_loss': round(self.stop_loss, 2) if self.stop_loss else None,
            'take_profit': round(self.take_profit, 2) if self.take_profit else None
        }

# 10 HIGHLY SUCCESSFUL SCALPING STRATEGIES FOR GOODWILL L1 DATA

class Strategy1_BidAskScalping:
    """Strategy 1: Bid-Ask Spread Scalping (Most Profitable)"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bid_ask_history = deque(maxlen=50)
        self.trades_today = 0
        self.win_rate = 0.0
        self.avg_profit = 0.0
        
    def analyze_signal(self, market_data: Dict) -> Optional[TradingSignal]:
        """Analyze L1 bid-ask data for scalping opportunities"""
        try:
            bid = market_data.get('bid', 0)
            ask = market_data.get('ask', 0)
            ltp = market_data.get('ltp', 0)
            volume = market_data.get('volume', 0)
            
            if bid <= 0 or ask <= 0:
                return None
            
            spread = ask - bid
            spread_pct = (spread / ltp) * 100
            mid_price = (bid + ask) / 2
            
            self.bid_ask_history.append({
                'bid': bid, 'ask': ask, 'ltp': ltp, 'spread': spread,
                'timestamp': datetime.now()
            })
            
            if len(self.bid_ask_history) < 10:
                return None
            
            # Strategy: Enter when price hits bid/ask with tight spread
            if spread_pct < 0.05:  # Tight spread
                if ltp <= bid + (spread * 0.3):  # Price near bid
                    return TradingSignal(
                        symbol=self.symbol,
                        signal_type=SignalType.BUY,
                        price=bid + 0.05,  # Slightly above bid
                        confidence=85.0,
                        timestamp=datetime.now(),
                        indicators={'spread_pct': spread_pct, 'position': 'near_bid'},
                        take_profit=ask - 0.05,
                        stop_loss=bid - 0.10
                    )
                elif ltp >= ask - (spread * 0.3):  # Price near ask
                    return TradingSignal(
                        symbol=self.symbol,
                        signal_type=SignalType.SELL,
                        price=ask - 0.05,  # Slightly below ask
                        confidence=85.0,
                        timestamp=datetime.now(),
                        indicators={'spread_pct': spread_pct, 'position': 'near_ask'},
                        take_profit=bid + 0.05,
                        stop_loss=ask + 0.10
                    )
            
            return None
            
        except Exception as e:
            logger.debug(f"Strategy1 error: {e}")
            return None

class Strategy2_VolumeSpike:
    """Strategy 2: Volume Spike Momentum (High Win Rate)"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.volume_history = deque(maxlen=30)
        self.price_history = deque(maxlen=30)
        
    def analyze_signal(self, market_data: Dict) -> Optional[TradingSignal]:
        """Volume spike with price momentum"""
        try:
            volume = market_data.get('volume', 0)
            ltp = market_data.get('ltp', 0)
            
            self.volume_history.append(volume)
            self.price_history.append(ltp)
            
            if len(self.volume_history) < 20:
                return None
            
            avg_volume = np.mean(list(self.volume_history)[:-5])
            current_volume = volume
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            if volume_ratio > 2.5:  # 2.5x average volume
                prices = list(self.price_history)
                price_change = (prices[-1] - prices[-5]) / prices[-5] * 100
                
                if price_change > 0.3:  # Price rising with volume
                    return TradingSignal(
                        symbol=self.symbol,
                        signal_type=SignalType.BUY,
                        price=ltp,
                        confidence=80.0,
                        timestamp=datetime.now(),
                        indicators={'volume_ratio': volume_ratio, 'price_change': price_change},
                        take_profit=ltp * 1.005,  # 0.5% target
                        stop_loss=ltp * 0.997    # 0.3% stop
                    )
                elif price_change < -0.3:  # Price falling with volume
                    return TradingSignal(
                        symbol=self.symbol,
                        signal_type=SignalType.SELL,
                        price=ltp,
                        confidence=80.0,
                        timestamp=datetime.now(),
                        indicators={'volume_ratio': volume_ratio, 'price_change': price_change},
                        take_profit=ltp * 0.995,  # 0.5% target
                        stop_loss=ltp * 1.003    # 0.3% stop
                    )
            
            return None
            
        except Exception as e:
            logger.debug(f"Strategy2 error: {e}")
            return None

class Strategy3_TickMomentum:
    """Strategy 3: Tick-by-Tick Momentum (Ultra Fast)"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.tick_data = deque(maxlen=20)
        
    def analyze_signal(self, market_data: Dict) -> Optional[TradingSignal]:
        """Analyze tick momentum"""
        try:
            ltp = market_data.get('ltp', 0)
            timestamp = datetime.now()
            
            self.tick_data.append({'price': ltp, 'time': timestamp})
            
            if len(self.tick_data) < 10:
                return None
            
            # Calculate tick momentum
            recent_ticks = list(self.tick_data)[-10:]
            up_ticks = sum(1 for i in range(1, len(recent_ticks)) 
                          if recent_ticks[i]['price'] > recent_ticks[i-1]['price'])
            down_ticks = sum(1 for i in range(1, len(recent_ticks)) 
                           if recent_ticks[i]['price'] < recent_ticks[i-1]['price'])
            
            momentum_score = (up_ticks - down_ticks) / 9.0 * 100
            
            if momentum_score >= 70:  # Strong upward momentum
                return TradingSignal(
                    symbol=self.symbol,
                    signal_type=SignalType.BUY,
                    price=ltp,
                    confidence=75.0,
                    timestamp=timestamp,
                    indicators={'momentum_score': momentum_score, 'up_ticks': up_ticks},
                    take_profit=ltp * 1.003,
                    stop_loss=ltp * 0.998
                )
            elif momentum_score <= -70:  # Strong downward momentum
                return TradingSignal(
                    symbol=self.symbol,
                    signal_type=SignalType.SELL,
                    price=ltp,
                    confidence=75.0,
                    timestamp=timestamp,
                    indicators={'momentum_score': momentum_score, 'down_ticks': down_ticks},
                    take_profit=ltp * 0.997,
                    stop_loss=ltp * 1.002
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"Strategy3 error: {e}")
            return None

class Strategy4_OrderBookImbalance:
    """Strategy 4: Order Book Imbalance (L1 Depth)"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.depth_history = deque(maxlen=15)
        
    def analyze_signal(self, market_data: Dict) -> Optional[TradingSignal]:
        """Analyze order book imbalance"""
        try:
            bid = market_data.get('bid', 0)
            ask = market_data.get('ask', 0)
            bid_qty = market_data.get('bid_qty', 0)
            ask_qty = market_data.get('ask_qty', 0)
            ltp = market_data.get('ltp', 0)
            
            if bid_qty <= 0 or ask_qty <= 0:
                return None
            
            total_qty = bid_qty + ask_qty
            bid_ratio = bid_qty / total_qty
            imbalance = (bid_qty - ask_qty) / total_qty
            
            self.depth_history.append({
                'bid_ratio': bid_ratio,
                'imbalance': imbalance,
                'timestamp': datetime.now()
            })
            
            if len(self.depth_history) < 5:
                return None
            
            avg_imbalance = np.mean([d['imbalance'] for d in self.depth_history])
            
            if imbalance > 0.6 and avg_imbalance > 0.4:  # Strong bid side
                return TradingSignal(
                    symbol=self.symbol,
                    signal_type=SignalType.BUY,
                    price=ltp,
                    confidence=78.0,
                    timestamp=datetime.now(),
                    indicators={'imbalance': imbalance, 'avg_imbalance': avg_imbalance},
                    take_profit=ltp * 1.004,
                    stop_loss=ltp * 0.997
                )
            elif imbalance < -0.6 and avg_imbalance < -0.4:  # Strong ask side
                return TradingSignal(
                    symbol=self.symbol,
                    signal_type=SignalType.SELL,
                    price=ltp,
                    confidence=78.0,
                    timestamp=datetime.now(),
                    indicators={'imbalance': imbalance, 'avg_imbalance': avg_imbalance},
                    take_profit=ltp * 0.996,
                    stop_loss=ltp * 1.003
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"Strategy4 error: {e}")
            return None

class Strategy5_MicroTrend:
    """Strategy 5: Micro Trend Following (5-tick EMA)"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.prices = deque(maxlen=25)
        self.ema_5 = None
        self.ema_12 = None
        
    def analyze_signal(self, market_data: Dict) -> Optional[TradingSignal]:
        """5-tick and 12-tick EMA crossover"""
        try:
            ltp = market_data.get('ltp', 0)
            self.prices.append(ltp)
            
            if len(self.prices) < 15:
                return None
            
            # Calculate EMAs
            prices_list = list(self.prices)
            self.ema_5 = self._calculate_ema(prices_list, 5)
            self.ema_12 = self._calculate_ema(prices_list, 12)
            
            if self.ema_5 is None or self.ema_12 is None:
                return None
            
            # Previous EMAs
            prev_ema_5 = self._calculate_ema(prices_list[:-1], 5)
            prev_ema_12 = self._calculate_ema(prices_list[:-1], 12)
            
            if prev_ema_5 is None or prev_ema_12 is None:
                return None
            
            # Crossover detection
            bullish_cross = (self.ema_5 > self.ema_12 and prev_ema_5 <= prev_ema_12)
            bearish_cross = (self.ema_5 < self.ema_12 and prev_ema_5 >= prev_ema_12)
            
            if bullish_cross and ltp > self.ema_5:
                return TradingSignal(
                    symbol=self.symbol,
                    signal_type=SignalType.BUY,
                    price=ltp,
                    confidence=82.0,
                    timestamp=datetime.now(),
                    indicators={'ema_5': self.ema_5, 'ema_12': self.ema_12, 'cross': 'bullish'},
                    take_profit=ltp * 1.0035,
                    stop_loss=ltp * 0.9985
                )
            elif bearish_cross and ltp < self.ema_5:
                return TradingSignal(
                    symbol=self.symbol,
                    signal_type=SignalType.SELL,
                    price=ltp,
                    confidence=82.0,
                    timestamp=datetime.now(),
                    indicators={'ema_5': self.ema_5, 'ema_12': self.ema_12, 'cross': 'bearish'},
                    take_profit=ltp * 0.9965,
                    stop_loss=ltp * 1.0015
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"Strategy5 error: {e}")
            return None
    
    def _calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate EMA"""
        if len(prices) < period:
            return None
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema

class Strategy6_SpreadCompression:
    """Strategy 6: Spread Compression Breakout"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.spread_history = deque(maxlen=20)
        
    def analyze_signal(self, market_data: Dict) -> Optional[TradingSignal]:
        """Tight spread followed by breakout"""
        try:
            bid = market_data.get('bid', 0)
            ask = market_data.get('ask', 0)
            ltp = market_data.get('ltp', 0)
            
            if bid <= 0 or ask <= 0:
                return None
            
            spread = ask - bid
            spread_pct = (spread / ltp) * 100
            
            self.spread_history.append({
                'spread': spread,
                'spread_pct': spread_pct,
                'ltp': ltp,
                'timestamp': datetime.now()
            })
            
            if len(self.spread_history) < 10:
                return None
            
            recent_spreads = [s['spread_pct'] for s in list(self.spread_history)[-10:]]
            avg_spread = np.mean(recent_spreads[:-2])
            current_spread = spread_pct
            
            # Compression followed by expansion
            if avg_spread < 0.08 and current_spread > avg_spread * 1.5:  # Spread expanding
                price_move = (ltp - self.spread_history[-5]['ltp']) / self.spread_history[-5]['ltp'] * 100
                
                if price_move > 0.2:  # Upward breakout
                    return TradingSignal(
                        symbol=self.symbol,
                        signal_type=SignalType.BUY,
                        price=ltp,
                        confidence=79.0,
                        timestamp=datetime.now(),
                        indicators={'spread_expansion': current_spread/avg_spread, 'price_move': price_move},
                        take_profit=ltp * 1.006,
                        stop_loss=ltp * 0.996
                    )
                elif price_move < -0.2:  # Downward breakout
                    return TradingSignal(
                        symbol=self.symbol,
                        signal_type=SignalType.SELL,
                        price=ltp,
                        confidence=79.0,
                        timestamp=datetime.now(),
                        indicators={'spread_expansion': current_spread/avg_spread, 'price_move': price_move},
                        take_profit=ltp * 0.994,
                        stop_loss=ltp * 1.004
                    )
            
            return None
            
        except Exception as e:
            logger.debug(f"Strategy6 error: {e}")
            return None

class Strategy7_PriceAction:
    """Strategy 7: Price Action Patterns (Support/Resistance)"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.price_levels = deque(maxlen=50)
        self.support_resistance = {'support': [], 'resistance': []}
        
    def analyze_signal(self, market_data: Dict) -> Optional[TradingSignal]:
        """Support/Resistance breakout scalping"""
        try:
            ltp = market_data.get('ltp', 0)
            volume = market_data.get('volume', 0)
            
            self.price_levels.append({'price': ltp, 'volume': volume, 'time': datetime.now()})
            
            if len(self.price_levels) < 30:
                return None
            
            # Find recent highs and lows
            prices = [p['price'] for p in self.price_levels]
            self._update_support_resistance(prices)
            
            if not self.support_resistance['support'] or not self.support_resistance['resistance']:
                return None
            
            nearest_support = max([s for s in self.support_resistance['support'] if s < ltp], default=0)
            nearest_resistance = min([r for r in self.support_resistance['resistance'] if r > ltp], default=float('inf'))
            
            # Breakout signals
            if nearest_resistance != float('inf') and ltp >= nearest_resistance * 1.0005:  # Resistance breakout
                return TradingSignal(
                    symbol=self.symbol,
                    signal_type=SignalType.BUY,
                    price=ltp,
                    confidence=76.0,
                    timestamp=datetime.now(),
                    indicators={'breakout': 'resistance', 'level': nearest_resistance},
                    take_profit=ltp * 1.008,
                    stop_loss=nearest_resistance * 0.999
                )
            elif nearest_support > 0 and ltp <= nearest_support * 0.9995:  # Support breakdown
                return TradingSignal(
                    symbol=self.symbol,
                    signal_type=SignalType.SELL,
                    price=ltp,
                    confidence=76.0,
                    timestamp=datetime.now(),
                    indicators={'breakout': 'support', 'level': nearest_support},
                    take_profit=ltp * 0.992,
                    stop_loss=nearest_support * 1.001
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"Strategy7 error: {e}")
            return None
    
    def _update_support_resistance(self, prices: List[float]):
        """Update support and resistance levels"""
        try:
            # Simple pivot point calculation
            highs = []
            lows = []
            
            for i in range(2, len(prices) - 2):
                if prices[i] > prices[i-1] and prices[i] > prices[i+1] and prices[i] > prices[i-2] and prices[i] > prices[i+2]:
                    highs.append(prices[i])
                elif prices[i] < prices[i-1] and prices[i] < prices[i+1] and prices[i] < prices[i-2] and prices[i] < prices[i+2]:
                    lows.append(prices[i])
            
            self.support_resistance['resistance'] = sorted(highs)[-3:] if highs else []
            self.support_resistance['support'] = sorted(lows)[-3:] if lows else []
            
        except Exception as e:
            logger.debug(f"Support/Resistance calculation error: {e}")

class Strategy8_VolumeProfile:
    """Strategy 8: Volume Profile & VWAP"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.vwap_data = deque(maxlen=100)
        
    def analyze_signal(self, market_data: Dict) -> Optional[TradingSignal]:
        """VWAP deviation signals"""
        try:
            ltp = market_data.get('ltp', 0)
            volume = market_data.get('volume', 0)
            
            self.vwap_data.append({'price': ltp, 'volume': volume})
            
            if len(self.vwap_data) < 20:
                return None
            
            # Calculate VWAP
            total_volume = sum(d['volume'] for d in self.vwap_data)
            if total_volume == 0:
                return None
            
            vwap = sum(d['price'] * d['volume'] for d in self.vwap_data) / total_volume
            deviation = (ltp - vwap) / vwap * 100
            
            # Standard deviation
            price_deviations = [(d['price'] - vwap) ** 2 for d in self.vwap_data]
            std_dev = np.sqrt(np.mean(price_deviations))
            std_dev_pct = (std_dev / vwap) * 100
            
            # Mean reversion signals
            if deviation < -2 * std_dev_pct and deviation < -0.5:  # Oversold vs VWAP
                return TradingSignal(
                    symbol=self.symbol,
                    signal_type=SignalType.BUY,
                    price=ltp,
                    confidence=77.0,
                    timestamp=datetime.now(),
                    indicators={'vwap_deviation': deviation, 'std_dev': std_dev_pct},
                    take_profit=vwap,
                    stop_loss=ltp * 0.996
                )
            elif deviation > 2 * std_dev_pct and deviation > 0.5:  # Overbought vs VWAP
                return TradingSignal(
                    symbol=self.symbol,
                    signal_type=SignalType.SELL,
                    price=ltp,
                    confidence=77.0,
                    timestamp=datetime.now(),
                    indicators={'vwap_deviation': deviation, 'std_dev': std_dev_pct},
                    take_profit=vwap,
                    stop_loss=ltp * 1.004
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"Strategy8 error: {e}")
            return None

class Strategy9_TimeBasedMomentum:
    """Strategy 9: Time-Based Momentum (Opening/Closing Hour)"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.momentum_data = deque(maxlen=30)
        
    def analyze_signal(self, market_data: Dict) -> Optional[TradingSignal]:
        """Time-based momentum patterns"""
        try:
            ltp = market_data.get('ltp', 0)
            volume = market_data.get('volume', 0)
            current_time = datetime.now().time()
            
            # Focus on high-volume periods
            is_opening_hour = datetime.strptime('09:15', '%H:%M').time() <= current_time <= datetime.strptime('10:30', '%H:%M').time()
            is_closing_hour = datetime.strptime('14:30', '%H:%M').time() <= current_time <= datetime.strptime('15:30', '%H:%M').time()
            
            if not (is_opening_hour or is_closing_hour):
                return None
            
            self.momentum_data.append({
                'price': ltp,
                'volume': volume,
                'time': current_time
            })
            
            if len(self.momentum_data) < 10:
                return None
            
            # Calculate price momentum
            recent_prices = [d['price'] for d in list(self.momentum_data)[-10:]]
            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100
            
            # Volume confirmation
            recent_volumes = [d['volume'] for d in list(self.momentum_data)[-5:]]
            avg_volume = np.mean(recent_volumes)
            
            if momentum > 0.4 and volume > avg_volume * 1.2:  # Strong upward momentum
                return TradingSignal(
                    symbol=self.symbol,
                    signal_type=SignalType.BUY,
                    price=ltp,
                    confidence=81.0,
                    timestamp=datetime.now(),
                    indicators={'momentum': momentum, 'period': 'opening' if is_opening_hour else 'closing'},
                    take_profit=ltp * 1.007,
                    stop_loss=ltp * 0.995
                )
            elif momentum < -0.4 and volume > avg_volume * 1.2:  # Strong downward momentum
                return TradingSignal(
                    symbol=self.symbol,
                    signal_type=SignalType.SELL,
                    price=ltp,
                    confidence=81.0,
                    timestamp=datetime.now(),
                    indicators={'momentum': momentum, 'period': 'opening' if is_opening_hour else 'closing'},
                    take_profit=ltp * 0.993,
                    stop_loss=ltp * 1.005
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"Strategy9 error: {e}")
            return None

class Strategy10_MultiTimeframe:
    """Strategy 10: Multi-Timeframe Confluence"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.tick_data = deque(maxlen=100)
        self.minute_data = deque(maxlen=10)
        self.last_minute_close = None
        
    def analyze_signal(self, market_data: Dict) -> Optional[TradingSignal]:
        """Multi-timeframe analysis"""
        try:
            ltp = market_data.get('ltp', 0)
            volume = market_data.get('volume', 0)
            timestamp = datetime.now()
            
            self.tick_data.append({'price': ltp, 'volume': volume, 'time': timestamp})
            
            # Create minute bars
            current_minute = timestamp.replace(second=0, microsecond=0)
            
            if not self.minute_data or self.minute_data[-1]['time'] < current_minute:
                if self.tick_data:
                    minute_prices = [t['price'] for t in self.tick_data if t['time'] >= current_minute - timedelta(minutes=1)]
                    if minute_prices:
                        minute_bar = {
                            'open': minute_prices[0],
                            'high': max(minute_prices),
                            'low': min(minute_prices),
                            'close': minute_prices[-1],
                            'volume': sum(t['volume'] for t in self.tick_data if t['time'] >= current_minute - timedelta(minutes=1)),
                            'time': current_minute
                        }
                        self.minute_data.append(minute_bar)
            
            if len(self.minute_data) < 5 or len(self.tick_data) < 20:
                return None
            
            # Minute trend
            minute_prices = [bar['close'] for bar in list(self.minute_data)[-5:]]
            minute_trend = 'up' if minute_prices[-1] > minute_prices[0] else 'down'
            
            # Tick momentum
            tick_prices = [t['price'] for t in list(self.tick_data)[-10:]]
            tick_momentum = (tick_prices[-1] - tick_prices[0]) / tick_prices[0] * 100
            
            # Confluence signal
            if minute_trend == 'up' and tick_momentum > 0.2:  # Both timeframes bullish
                return TradingSignal(
                    symbol=self.symbol,
                    signal_type=SignalType.BUY,
                    price=ltp,
                    confidence=88.0,
                    timestamp=timestamp,
                    indicators={'minute_trend': minute_trend, 'tick_momentum': tick_momentum},
                    take_profit=ltp * 1.008,
                    stop_loss=ltp * 0.994
                )
            elif minute_trend == 'down' and tick_momentum < -0.2:  # Both timeframes bearish
                return TradingSignal(
                    symbol=self.symbol,
                    signal_type=SignalType.SELL,
                    price=ltp,
                    confidence=88.0,
                    timestamp=timestamp,
                    indicators={'minute_trend': minute_trend, 'tick_momentum': tick_momentum},
                    take_profit=ltp * 0.992,
                    stop_loss=ltp * 1.006
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"Strategy10 error: {e}")
            return None


# REAL-TIME EXECUTION MANAGER WITH STRATEGY APPLICATION
class ExecutionManager:
    """Manages real-time execution of 10 scalping strategies on top stocks"""
    
    def __init__(self, api_handler, paper_engine, db_path: str = "data/execution_queue.db"):
        self.api = api_handler
        self.paper_engine = paper_engine
        self.db_path = db_path
        
        # Active stocks for execution
        self.active_stocks: Dict[str, ExecutionStock] = {}
        self.max_active_stocks = 10
        
        # Strategy instances for each stock
        self.strategies: Dict[str, List] = {}
        
        # Real-time data processing
        self.market_data_queue = queue.Queue(maxsize=1000)
        self.signal_queue = queue.Queue(maxsize=500)
        self.execution_queue = queue.Queue(maxsize=200)
        
        # Threading control
        self.running = False
        self.threads = {}
        
        # Performance tracking
        self.execution_stats = {
            'signals_generated': 0,
            'orders_placed': 0,
            'successful_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0
        }
        
        # Risk management
        self.risk_manager = RiskManager()
        
        # Initialize database
        self._init_database()
        
        logger.info("⚡ Execution Manager initialized")
    
    def _init_database(self):
        """Initialize execution database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    price REAL NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    indicators TEXT,
                    stop_loss REAL,
                    take_profit REAL,
                    executed BOOLEAN DEFAULT 0,
                    order_id TEXT,
                    execution_price REAL,
                    execution_time DATETIME
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS active_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL,
                    unrealized_pnl REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    entry_time DATETIME NOT NULL,
                    last_update DATETIME,
                    is_open BOOLEAN DEFAULT 1
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Execution database initialized")
            
        except Exception as e:
            logger.error(f"❌ Execution database init error: {e}")
    
    def set_active_stocks(self, stocks_data: List[Dict]):
        """Set the top 10 stocks for active execution"""
        try:
            # Clear existing stocks
            self.active_stocks.clear()
            self.strategies.clear()
            
            # Add new stocks (top 10)
            for i, stock_data in enumerate(stocks_data[:self.max_active_stocks]):
                symbol = stock_data['symbol']
                
                # Create ExecutionStock
                execution_stock = ExecutionStock(
                    symbol=symbol,
                    asset_type=stock_data.get('asset_type', 'equity'),
                    scalping_score=stock_data.get('scalping_score', 0),
                    volatility_score=stock_data.get('volatility_score', 0),
                    liquidity_score=stock_data.get('liquidity_score', 0),
                    current_price=stock_data.get('current_price', 0)
                )
                
                self.active_stocks[symbol] = execution_stock
                
                # Initialize all 10 strategies for this stock
                self.strategies[symbol] = [
                    Strategy1_BidAskScalping(symbol),
                    Strategy2_VolumeSpike(symbol),
                    Strategy3_TickMomentum(symbol),
                    Strategy4_OrderBookImbalance(symbol),
                    Strategy5_MicroTrend(symbol),
                    Strategy6_SpreadCompression(symbol),
                    Strategy7_PriceAction(symbol),
                    Strategy8_VolumeProfile(symbol),
                    Strategy9_TimeBasedMomentum(symbol),
                    Strategy10_MultiTimeframe(symbol)
                ]
                
                logger.info(f"📈 Added {symbol} for active execution (rank #{i+1})")
            
            logger.info(f"✅ Set {len(self.active_stocks)} stocks for active execution")
            
        except Exception as e:
            logger.error(f"❌ Set active stocks error: {e}")
    
    def start_execution(self):
        """Start real-time execution system"""
        try:
            self.running = True
            
            # Start data processing thread
            self.threads['data_processor'] = threading.Thread(
                target=self._process_market_data,
                daemon=True,
                name="DataProcessor"
            )
            self.threads['data_processor'].start()
            
            # Start signal generation thread
            self.threads['signal_generator'] = threading.Thread(
                target=self._generate_signals,
                daemon=True,
                name="SignalGenerator"
            )
            self.threads['signal_generator'].start()
            
            # Start order execution thread
            self.threads['order_executor'] = threading.Thread(
                target=self._execute_orders,
                daemon=True,
                name="OrderExecutor"
            )
            self.threads['order_executor'].start()
            
            # Start position monitoring thread
            self.threads['position_monitor'] = threading.Thread(
                target=self._monitor_positions,
                daemon=True,
                name="PositionMonitor"
            )
            self.threads['position_monitor'].start()
            
            logger.info("🚀 Execution Manager started - all threads running")
            
        except Exception as e:
            logger.error(f"❌ Start execution error: {e}")
    
    def stop_execution(self):
        """Stop execution system"""
        try:
            self.running = False
            
            # Wait for threads to finish
            for thread_name, thread in self.threads.items():
                if thread.is_alive():
                    thread.join(timeout=5)
                    logger.info(f"⏹️ {thread_name} stopped")
            
            logger.info("⏹️ Execution Manager stopped")
            
        except Exception as e:
            logger.error(f"❌ Stop execution error: {e}")
    
    def feed_market_data(self, symbol: str, market_data: Dict):
        """Feed real-time market data to processing queue"""
        try:
            if symbol in self.active_stocks:
                data_packet = {
                    'symbol': symbol,
                    'data': market_data,
                    'timestamp': datetime.now()
                }
                
                # Add to processing queue (non-blocking)
                if not self.market_data_queue.full():
                    self.market_data_queue.put(data_packet)
                else:
                    logger.warning(f"⚠️ Market data queue full for {symbol}")
                    
        except Exception as e:
            logger.debug(f"Feed market data error: {e}")
    
    def _process_market_data(self):
        """Process incoming market data in real-time"""
        while self.running:
            try:
                # Get market data with timeout
                data_packet = self.market_data_queue.get(timeout=1)
                
                symbol = data_packet['symbol']
                market_data = data_packet['data']
                timestamp = data_packet['timestamp']
                
                if symbol not in self.active_stocks:
                    continue
                
                # Update stock data
                execution_stock = self.active_stocks[symbol]
                execution_stock.current_price = market_data.get('ltp', 0)
                execution_stock.bid_price = market_data.get('bid', 0)
                execution_stock.ask_price = market_data.get('ask', 0)
                execution_stock.volume = market_data.get('volume', 0)
                execution_stock.last_update = timestamp
                
                # Calculate unrealized P&L if position exists
                if execution_stock.position_size != 0:
                    price_diff = execution_stock.current_price - execution_stock.entry_price
                    if execution_stock.position_size > 0:  # Long position
                        execution_stock.unrealized_pnl = price_diff * execution_stock.position_size
                    else:  # Short position
                        execution_stock.unrealized_pnl = -price_diff * abs(execution_stock.position_size)
                
                # Apply all strategies to this market data
                self._apply_strategies_to_data(symbol, market_data)
                
                # Mark task as done
                self.market_data_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Process market data error: {e}")
    
    def _apply_strategies_to_data(self, symbol: str, market_data: Dict):
        """Apply all 10 strategies to incoming market data"""
        try:
            if symbol not in self.strategies:
                return
            
            strategy_instances = self.strategies[symbol]
            
            # Apply each strategy
            for i, strategy in enumerate(strategy_instances):
                try:
                    signal = strategy.analyze_signal(market_data)
                    
                    if signal:
                        # Add strategy name to signal
                        signal_data = signal.to_dict()
                        signal_data['strategy_name'] = strategy.__class__.__name__
                        signal_data['strategy_id'] = i + 1
                        
                        # Add to signal queue
                        if not self.signal_queue.full():
                            self.signal_queue.put(signal_data)
                            self.execution_stats['signals_generated'] += 1
                            
                            logger.info(f"📊 Signal: {symbol} {signal.signal_type.value} from {strategy.__class__.__name__} (conf: {signal.confidence}%)")
                        
                except Exception as e:
                    logger.debug(f"Strategy {i+1} error for {symbol}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Apply strategies error: {e}")
    
    def _generate_signals(self):
        """Process and validate generated signals"""
        while self.running:
            try:
                # Get signal with timeout
                signal_data = self.signal_queue.get(timeout=1)
                
                symbol = signal_data['symbol']
                
                # Validate signal
                if self._validate_signal(signal_data):
                    # Save signal to database
                    self._save_signal_to_db(signal_data)
                    
                    # Check risk management
                    if self.risk_manager.check_signal_risk(signal_data, self.active_stocks[symbol]):
                        # Add to execution queue
                        if not self.execution_queue.full():
                            self.execution_queue.put(signal_data)
                            logger.info(f"✅ Signal validated: {symbol} {signal_data['signal_type']}")
                        else:
                            logger.warning(f"⚠️ Execution queue full for {symbol}")
                    else:
                        logger.warning(f"⚠️ Signal failed risk check: {symbol}")
                
                self.signal_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Generate signals error: {e}")
    
    def _validate_signal(self, signal_data: Dict) -> bool:
        """Validate signal quality"""
        try:
            # Basic validation
            if signal_data['confidence'] < 70:  # Minimum confidence
                return False
            
            symbol = signal_data['symbol']
            if symbol not in self.active_stocks:
                return False
            
            execution_stock = self.active_stocks[symbol]
            
            # Check if stock is already in position (avoid over-trading)
            if execution_stock.status in [ExecutionStatus.ORDER_PLACED, ExecutionStatus.POSITION_OPEN]:
                return False
            
            # Check price validity
            current_price = execution_stock.current_price
            signal_price = signal_data['price']
            
            if abs(signal_price - current_price) / current_price > 0.01:  # 1% price deviation
                return False
            
            # Check minimum time between signals (avoid spam)
            time_since_last = (datetime.now() - execution_stock.last_update).seconds
            if time_since_last < 5:  # Minimum 5 seconds
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Signal validation error: {e}")
            return False
    
    def _execute_orders(self):
        """Execute validated trading signals"""
        while self.running:
            try:
                # Get execution signal with timeout
                signal_data = self.execution_queue.get(timeout=1)
                
                symbol = signal_data['symbol']
                execution_stock = self.active_stocks[symbol]
                
                # Calculate position size
                position_size = self._calculate_position_size(execution_stock, signal_data)
                
                if position_size == 0:
                    continue
                
                # Place order
                order_result = self._place_scalping_order(signal_data, position_size)
                
                if order_result and order_result.get('success'):
                    # Update execution stock status
                    execution_stock.status = ExecutionStatus.ORDER_PLACED
                    
                    # Update statistics
                    self.execution_stats['orders_placed'] += 1
                    
                    logger.info(f"📝 Order placed: {symbol} {signal_data['signal_type']} qty: {position_size}")
                else:
                    logger.error(f"❌ Order failed: {symbol} {signal_data['signal_type']}")
                
                self.execution_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Execute orders error: {e}")
    
    def _calculate_position_size(self, execution_stock: ExecutionStock, signal_data: Dict) -> int:
        """Calculate appropriate position size"""
        try:
            # Base position size from stock's max size
            base_size = execution_stock.max_position_size
            
            # Adjust based on confidence
            confidence = signal_data['confidence']
            confidence_multiplier = confidence / 100
            
            # Adjust based on volatility (lower volatility = larger size)
            volatility_multiplier = 1.0
            if execution_stock.volatility_score > 80:
                volatility_multiplier = 0.5
            elif execution_stock.volatility_score > 60:
                volatility_multiplier = 0.7
            
            # Calculate final size
            position_size = int(base_size * confidence_multiplier * volatility_multiplier)
            
            # Minimum and maximum limits
            position_size = max(1, min(position_size, execution_stock.max_position_size))
            
            return position_size
            
        except Exception as e:
            logger.debug(f"Position size calculation error: {e}")
            return 0
    
    def _place_scalping_order(self, signal_data: Dict, position_size: int) -> Optional[Dict]:
        """Place actual trading order"""
        try:
            symbol = signal_data['symbol']
            signal_type = signal_data['signal_type']
            price = signal_data['price']
            
            # Determine order side
            if signal_type in ['buy', 'exit_short']:
                side = 'BUY'
            else:
                side = 'SELL'
            
            # Use paper trading engine
            if self.paper_engine:
                from paper_trading_engine import OrderType
                
                order_id = self.paper_engine.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=position_size,
                    order_type=OrderType.MARKET  # Market orders for scalping
                )
                
                if order_id:
                    return {
                        'success': True,
                        'order_id': order_id,
                        'mode': 'paper'
                    }
            
            # For live trading (when implemented)
            # order_response = self.api.place_order(...)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Place scalping order error: {e}")
            return None
    
    def _monitor_positions(self):
        """Monitor open positions for exit signals"""
        while self.running:
            try:
                time.sleep(1)  # Check every second
                
                for symbol, execution_stock in self.active_stocks.items():
                    if execution_stock.status == ExecutionStatus.POSITION_OPEN:
                        self._check_position_exit(execution_stock)
                        
            except Exception as e:
                logger.error(f"❌ Monitor positions error: {e}")
    
    def _check_position_exit(self, execution_stock: ExecutionStock):
        """Check if position should be exited"""
        try:
            current_price = execution_stock.current_price
            entry_price = execution_stock.entry_price
            
            if entry_price == 0:
                return
            
            # Calculate profit/loss percentage
            if execution_stock.position_size > 0:  # Long position
                pnl_pct = (current_price - entry_price) / entry_price * 100
            else:  # Short position
                pnl_pct = (entry_price - current_price) / entry_price * 100
            
            # Exit conditions
            should_exit = False
            exit_reason = ""
            
            # Stop loss
            if pnl_pct <= -execution_stock.stop_loss_pct:
                should_exit = True
                exit_reason = "stop_loss"
            
            # Take profit
            elif pnl_pct >= execution_stock.take_profit_pct:
                should_exit = True
                exit_reason = "take_profit"
            
            # Time-based exit (scalping - quick exits)
            elif (datetime.now() - execution_stock.last_update).seconds > 300:  # 5 minutes max
                should_exit = True
                exit_reason = "time_exit"
            
            if should_exit:
                self._exit_position(execution_stock, exit_reason)
                
        except Exception as e:
            logger.debug(f"Check position exit error: {e}")
    
    def _exit_position(self, execution_stock: ExecutionStock, reason: str):
        """Exit open position"""
        try:
            symbol = execution_stock.symbol
            position_size = execution_stock.position_size
            
            if position_size == 0:
                return
            
            # Determine exit side
            exit_side = 'SELL' if position_size > 0 else 'BUY'
            
            # Place exit order
            if self.paper_engine:
                from paper_trading_engine import OrderType
                
                order_id = self.paper_engine.place_order(
                    symbol=symbol,
                    side=exit_side,
                    quantity=abs(position_size),
                    order_type=OrderType.MARKET
                )
                
                if order_id:
                    execution_stock.status = ExecutionStatus.EXITING
                    logger.info(f"🔒 Exiting position: {symbol} reason: {reason}")
                    
                    # Update statistics
                    if execution_stock.unrealized_pnl > 0:
                        self.execution_stats['successful_trades'] += 1
                    
                    self.execution_stats['total_pnl'] += execution_stock.unrealized_pnl
                    
                    # Reset position
                    execution_stock.position_size = 0
                    execution_stock.entry_price = 0
                    execution_stock.unrealized_pnl = 0
                    execution_stock.status = ExecutionStatus.COMPLETED
                    
        except Exception as e:
            logger.error(f"❌ Exit position error: {e}")
    
    def _save_signal_to_db(self, signal_data: Dict):
        """Save signal to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO execution_signals 
                (symbol, strategy_name, signal_type, price, confidence, timestamp, 
                 indicators, stop_loss, take_profit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_data['symbol'],
                signal_data.get('strategy_name', ''),
                signal_data['signal_type'],
                signal_data['price'],
                signal_data['confidence'],
                signal_data['timestamp'],
                json.dumps(signal_data.get('indicators', {})),
                signal_data.get('stop_loss'),
                signal_data.get('take_profit')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.debug(f"Save signal to DB error: {e}")
    
    def get_execution_summary(self) -> Dict:
        """Get execution summary"""
        try:
            # Calculate win rate
            total_trades = self.execution_stats['successful_trades'] + max(0, self.execution_stats['orders_placed'] - self.execution_stats['successful_trades'])
            win_rate = (self.execution_stats['successful_trades'] / total_trades * 100) if total_trades > 0 else 0
            
            # Get active positions
            active_positions = []
            for symbol, stock in self.active_stocks.items():
                if stock.position_size != 0:
                    active_positions.append({
                        'symbol': symbol,
                        'side': 'LONG' if stock.position_size > 0 else 'SHORT',
                        'size': abs(stock.position_size),
                        'entry_price': stock.entry_price,
                        'current_price': stock.current_price,
                        'unrealized_pnl': stock.unrealized_pnl,
                        'status': stock.status.value
                    })
            
            return {
                'execution_stats': {
                    'signals_generated': self.execution_stats['signals_generated'],
                    'orders_placed': self.execution_stats['orders_placed'],
                    'successful_trades': self.execution_stats['successful_trades'],
                    'total_pnl': round(self.execution_stats['total_pnl'], 2),
                    'win_rate': round(win_rate, 1)
                },
                'active_stocks': [stock.to_dict() for stock in self.active_stocks.values()],
                'active_positions': active_positions,
                'system_status': {
                    'running': self.running,
                    'active_stocks_count': len(self.active_stocks),
                    'queue_sizes': {
                        'market_data': self.market_data_queue.qsize(),
                        'signals': self.signal_queue.qsize(),
                        'execution': self.execution_queue.qsize()
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Execution summary error: {e}")
            return {'error': str(e)}


# RISK MANAGEMENT
class RiskManager:
    """Risk management for scalping execution"""
    
    def __init__(self):
        self.max_daily_trades_per_stock = 50
        self.max_daily_loss_per_stock = 1000.0
        self.max_concurrent_positions = 5
        self.daily_trade_counts = {}
        self.daily_pnl = {}
        
    def check_signal_risk(self, signal_data: Dict, execution_stock: ExecutionStock) -> bool:
        """Check if signal passes risk management"""
        try:
            symbol = signal_data['symbol']
            today = datetime.now().date()
            
            # Initialize daily counters
            if symbol not in self.daily_trade_counts:
                self.daily_trade_counts[symbol] = {}
            if today not in self.daily_trade_counts[symbol]:
                self.daily_trade_counts[symbol][today] = 0
            
            if symbol not in self.daily_pnl:
                self.daily_pnl[symbol] = {}
            if today not in self.daily_pnl[symbol]:
                self.daily_pnl[symbol][today] = 0.0
            
            # Check daily trade limit
            if self.daily_trade_counts[symbol][today] >= self.max_daily_trades_per_stock:
                return False
            
            # Check daily loss limit
            if self.daily_pnl[symbol][today] <= -self.max_daily_loss_per_stock:
                return False
            
            # Check confidence threshold based on recent performance
            if execution_stock.trades_today > 5:
                win_rate = execution_stock.profit_trades / execution_stock.trades_today
                if win_rate < 0.4 and signal_data['confidence'] < 85:  # Higher threshold if losing
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Risk check error: {e}")
            return False


# INTEGRATION EXAMPLE
if __name__ == "__main__":
    # Example usage showing real-time data flow
    
    # Mock API and paper engine
    class MockAPI:
        pass
    
    class MockPaperEngine:
        def place_order(self, symbol, side, quantity, order_type):
            return f"ORDER_{symbol}_{int(time.time())}"
    
    # Initialize execution manager
    execution_manager = ExecutionManager(MockAPI(), MockPaperEngine())
    
    # Set top 10 stocks from scanner
    top_stocks = [
        {'symbol': 'RELIANCE', 'asset_type': 'equity', 'scalping_score': 85, 'volatility_score': 75, 'liquidity_score': 90, 'current_price': 2500},
        {'symbol': 'TCS', 'asset_type': 'equity', 'scalping_score': 82, 'volatility_score': 70, 'liquidity_score': 88, 'current_price': 3200},
        # ... more stocks
    ]
    
    execution_manager.set_active_stocks(top_stocks)
    execution_manager.start_execution()
    
    # Simulate real-time market data feed
    import random
    
    for i in range(100):
        for symbol in ['RELIANCE', 'TCS']:
            # Simulate L1 market data from Goodwill API
            market_data = {
                'ltp': 2500 + random.uniform(-10, 10),
                'bid': 2499.95,
                'ask': 2500.05,
                'bid_qty': random.randint(100, 1000),
                'ask_qty': random.randint(100, 1000),
                'volume': random.randint(1000, 10000)
            }
            
            # Feed to execution manager
            execution_manager.feed_market_data(symbol, market_data)
        
        time.sleep(0.1)  # 100ms intervals
    
    # Get summary
    summary = execution_manager.get_execution_summary()
    print(f"📊 Execution Summary: {json.dumps(summary, indent=2)}")
    
    execution_manager.stop_execution()
