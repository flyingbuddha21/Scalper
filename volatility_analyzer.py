#!/usr/bin/env python3
"""
Complete Real-time Volatility Analyzer
Analyzes volatility patterns and provides scalping insights
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import sqlite3
import json
from collections import deque
import statistics

logger = logging.getLogger(__name__)

class VolatilityType(Enum):
    INTRADAY = "intraday"
    HISTORICAL = "historical"
    IMPLIED = "implied"
    REALIZED = "realized"

@dataclass
class VolatilityMetrics:
    symbol: str
    timestamp: datetime
    
    # Price-based volatility
    price_volatility: float
    intraday_range: float
    range_percentage: float
    
    # Time-based metrics
    atr_14: float
    atr_percentage: float
    bollinger_squeeze: bool
    
    # Volume-weighted metrics
    vwap_deviation: float
    volume_weighted_volatility: float
    
    # Statistical measures
    standard_deviation: float
    variance: float
    skewness: float
    kurtosis: float
    
    # Volatility rankings
    volatility_rank: float
    volatility_percentile: float
    
    # Predictive measures
    volatility_forecast: float
    trend_direction: str
    momentum_score: float
    
    # Trading signals
    scalping_signal: str
    volatility_breakout: bool
    mean_reversion_signal: bool
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'price_volatility': round(self.price_volatility, 3),
            'intraday_range': round(self.intraday_range, 2),
            'range_percentage': round(self.range_percentage, 2),
            'atr_14': round(self.atr_14, 2),
            'atr_percentage': round(self.atr_percentage, 3),
            'bollinger_squeeze': self.bollinger_squeeze,
            'vwap_deviation': round(self.vwap_deviation, 3),
            'volume_weighted_volatility': round(self.volume_weighted_volatility, 3),
            'standard_deviation': round(self.standard_deviation, 3),
            'variance': round(self.variance, 4),
            'skewness': round(self.skewness, 3),
            'kurtosis': round(self.kurtosis, 3),
            'volatility_rank': round(self.volatility_rank, 1),
            'volatility_percentile': round(self.volatility_percentile, 1),
            'volatility_forecast': round(self.volatility_forecast, 3),
            'trend_direction': self.trend_direction,
            'momentum_score': round(self.momentum_score, 2),
            'scalping_signal': self.scalping_signal,
            'volatility_breakout': self.volatility_breakout,
            'mean_reversion_signal': self.mean_reversion_signal
        }

class RealTimeVolatilityAnalyzer:
    def __init__(self, api_handler, db_path: str = "data/volatility_data.db"):
        self.api = api_handler
        self.db_path = db_path
        self.running = False
        
        # Real-time data storage
        self.price_buffers = {}  # symbol -> deque of prices
        self.volume_buffers = {}  # symbol -> deque of volumes
        self.volatility_cache = {}  # symbol -> VolatilityMetrics
        
        # Configuration
        self.buffer_size = 500  # Keep last 500 ticks
        self.analysis_interval = 30  # Analyze every 30 seconds
        
        # Historical data cache
        self.historical_cache = {}
        self.cache_expiry = {}
        
        # Callbacks
        self.update_callbacks = []
        self.alert_callbacks = []
        
        # Initialize database
        self._init_database()
        
        logger.info("📊 Volatility Analyzer initialized")
    
    def _init_database(self):
        """Initialize volatility database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS volatility_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    price_volatility REAL,
                    intraday_range REAL,
                    range_percentage REAL,
                    atr_14 REAL,
                    atr_percentage REAL,
                    bollinger_squeeze BOOLEAN,
                    vwap_deviation REAL,
                    volume_weighted_volatility REAL,
                    standard_deviation REAL,
                    variance REAL,
                    skewness REAL,
                    kurtosis REAL,
                    volatility_rank REAL,
                    volatility_percentile REAL,
                    volatility_forecast REAL,
                    trend_direction TEXT,
                    momentum_score REAL,
                    scalping_signal TEXT,
                    volatility_breakout BOOLEAN,
                    mean_reversion_signal BOOLEAN,
                    UNIQUE(symbol, timestamp)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS volatility_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    message TEXT,
                    volatility_value REAL,
                    threshold_value REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_volatility_symbol_time ON volatility_metrics(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_symbol_time ON volatility_alerts(symbol, timestamp)")
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Volatility database initialized")
            
        except Exception as e:
            logger.error(f"❌ Volatility database init error: {e}")
    
    def add_symbols(self, symbols: List[str]):
        """Add symbols for volatility analysis"""
        for symbol in symbols:
            if symbol not in self.price_buffers:
                self.price_buffers[symbol] = deque(maxlen=self.buffer_size)
                self.volume_buffers[symbol] = deque(maxlen=self.buffer_size)
                logger.info(f"📈 Added {symbol} for volatility analysis")
    
    def remove_symbol(self, symbol: str):
        """Remove symbol from analysis"""
        if symbol in self.price_buffers:
            del self.price_buffers[symbol]
            del self.volume_buffers[symbol]
            if symbol in self.volatility_cache:
                del self.volatility_cache[symbol]
            logger.info(f"📉 Removed {symbol} from volatility analysis")
    
    def start_analysis(self):
        """Start real-time volatility analysis"""
        try:
            self.running = True
            
            # Start analysis thread
            analysis_thread = threading.Thread(
                target=self._analysis_loop,
                daemon=True,
                name="VolatilityAnalysis"
            )
            analysis_thread.start()
            
            logger.info("🚀 Volatility analysis started")
            
        except Exception as e:
            logger.error(f"❌ Start analysis error: {e}")
    
    def stop_analysis(self):
        """Stop volatility analysis"""
        self.running = False
        logger.info("⏹️ Volatility analysis stopped")
    
    def update_data(self, symbol: str, price: float, volume: int, timestamp: datetime = None):
        """Update real-time data for symbol"""
        try:
            if symbol not in self.price_buffers:
                self.add_symbols([symbol])
            
            if timestamp is None:
                timestamp = datetime.now()
            
            self.price_buffers[symbol].append((price, timestamp))
            self.volume_buffers[symbol].append((volume, timestamp))
            
        except Exception as e:
            logger.debug(f"Update data error for {symbol}: {e}")
    
    def _analysis_loop(self):
        """Main analysis loop"""
        while self.running:
            try:
                for symbol in list(self.price_buffers.keys()):
                    if len(self.price_buffers[symbol]) >= 20:
                        metrics = self._calculate_volatility_metrics(symbol)
                        if metrics:
                            self.volatility_cache[symbol] = metrics
                            self._save_metrics_to_db(metrics)
                            self._check_alerts(metrics)
                
                # Notify callbacks
                self._notify_callbacks()
                
                time.sleep(self.analysis_interval)
                
            except Exception as e:
                logger.error(f"❌ Analysis loop error: {e}")
                time.sleep(60)
    
    def _calculate_volatility_metrics(self, symbol: str) -> Optional[VolatilityMetrics]:
        """Calculate comprehensive volatility metrics"""
        try:
            if symbol not in self.price_buffers or len(self.price_buffers[symbol]) < 20:
                return None
            
            price_data = list(self.price_buffers[symbol])
            volume_data = list(self.volume_buffers[symbol])
            
            prices = [p[0] for p in price_data]
            volumes = [v[0] for v in volume_data]
            timestamps = [p[1] for p in price_data]
            
            current_price = prices[-1]
            
            # Price-based volatility
            returns = np.diff(prices) / prices[:-1]
            price_volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized
            
            # Intraday range
            high_price = max(prices[-20:])  # Last 20 ticks
            low_price = min(prices[-20:])
            intraday_range = high_price - low_price
            range_percentage = (intraday_range / current_price) * 100
            
            # ATR calculation
            atr_14 = self._calculate_atr(prices, 14)
            atr_percentage = (atr_14 / current_price) * 100
            
            # Bollinger Bands squeeze
            bollinger_squeeze = self._check_bollinger_squeeze(prices)
            
            # VWAP and deviation
            vwap = self._calculate_vwap(prices, volumes)
            vwap_deviation = ((current_price - vwap) / vwap) * 100 if vwap > 0 else 0
            
            # Volume-weighted volatility
            volume_weighted_volatility = self._calculate_volume_weighted_volatility(prices, volumes)
            
            # Statistical measures
            std_dev = np.std(prices[-50:]) if len(prices) >= 50 else np.std(prices)
            variance = np.var(prices[-50:]) if len(prices) >= 50 else np.var(prices)
            
            # Skewness and Kurtosis
            recent_returns = returns[-30:] if len(returns) >= 30 else returns
            skewness = self._calculate_skewness(recent_returns)
            kurtosis = self._calculate_kurtosis(recent_returns)
            
            # Volatility ranking
            volatility_rank, volatility_percentile = self._calculate_volatility_rank(symbol, price_volatility)
            
            # Volatility forecast
            volatility_forecast = self._forecast_volatility(prices, volumes)
            
            # Trend analysis
            trend_direction, momentum_score = self._analyze_trend(prices)
            
            # Trading signals
            scalping_signal = self._generate_scalping_signal(
                price_volatility, atr_percentage, vwap_deviation, trend_direction
            )
            
            volatility_breakout = self._detect_volatility_breakout(
                price_volatility, atr_percentage, bollinger_squeeze
            )
            
            mean_reversion_signal = self._detect_mean_reversion(
                vwap_deviation, volatility_rank
            )
            
            return VolatilityMetrics(
                symbol=symbol,
                timestamp=datetime.now(),
                price_volatility=price_volatility,
                intraday_range=intraday_range,
                range_percentage=range_percentage,
                atr_14=atr_14,
                atr_percentage=atr_percentage,
                bollinger_squeeze=bollinger_squeeze,
                vwap_deviation=vwap_deviation,
                volume_weighted_volatility=volume_weighted_volatility,
                standard_deviation=std_dev,
                variance=variance,
                skewness=skewness,
                kurtosis=kurtosis,
                volatility_rank=volatility_rank,
                volatility_percentile=volatility_percentile,
                volatility_forecast=volatility_forecast,
                trend_direction=trend_direction,
                momentum_score=momentum_score,
                scalping_signal=scalping_signal,
                volatility_breakout=volatility_breakout,
                mean_reversion_signal=mean_reversion_signal
            )
            
        except Exception as e:
            logger.debug(f"Volatility metrics calculation error for {symbol}: {e}")
            return None
    
    def _calculate_atr(self, prices: List[float], period: int) -> float:
        """Calculate Average True Range"""
        try:
            if len(prices) < period + 1:
                return 0.0
            
            true_ranges = []
            for i in range(1, len(prices)):
                high_low = abs(prices[i] - prices[i-1])  # Simplified for tick data
                true_ranges.append(high_low)
            
            if len(true_ranges) >= period:
                return np.mean(true_ranges[-period:])
            else:
                return np.mean(true_ranges)
                
        except Exception:
            return 0.0
    
    def _check_bollinger_squeeze(self, prices: List[float]) -> bool:
        """Check for Bollinger Bands squeeze"""
        try:
            if len(prices) < 20:
                return False
            
            # Simple squeeze detection
            recent_prices = prices[-20:]
            std_dev = np.std(recent_prices)
            mean_price = np.mean(recent_prices)
            
            # Squeeze when standard deviation is low relative to price
            squeeze_threshold = mean_price * 0.01  # 1% of price
            return std_dev < squeeze_threshold
            
        except Exception:
            return False
    
    def _calculate_vwap(self, prices: List[float], volumes: List[int]) -> float:
        """Calculate Volume Weighted Average Price"""
        try:
            if len(prices) != len(volumes) or len(prices) == 0:
                return 0.0
            
            total_volume = sum(volumes)
            if total_volume == 0:
                return np.mean(prices)
            
            vwap = sum(p * v for p, v in zip(prices, volumes)) / total_volume
            return vwap
            
        except Exception:
            return 0.0
    
    def _calculate_volume_weighted_volatility(self, prices: List[float], volumes: List[int]) -> float:
        """Calculate volume-weighted volatility"""
        try:
            if len(prices) < 2:
                return 0.0
            
            returns = np.diff(prices) / prices[:-1]
            volume_weights = volumes[1:] if len(volumes) > len(returns) else volumes[:len(returns)]
            
            if len(returns) != len(volume_weights):
                return np.std(returns) * 100
            
            total_volume = sum(volume_weights)
            if total_volume == 0:
                return np.std(returns) * 100
            
            # Weight returns by volume
            weighted_returns = [r * v / total_volume for r, v in zip(returns, volume_weights)]
            return np.std(weighted_returns) * 100
            
        except Exception:
            return 0.0
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness"""
        try:
            if len(data) < 3:
                return 0.0
            
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            if std_val == 0:
                return 0.0
            
            skew = np.mean([((x - mean_val) / std_val) ** 3 for x in data])
            return skew
            
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis"""
        try:
            if len(data) < 4:
                return 0.0
            
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            if std_val == 0:
                return 0.0
            
            kurt = np.mean([((x - mean_val) / std_val) ** 4 for x in data]) - 3
            return kurt
            
        except Exception:
            return 0.0
    
    def _calculate_volatility_rank(self, symbol: str, current_vol: float) -> Tuple[float, float]:
        """Calculate volatility rank and percentile"""
        try:
            # Get historical volatility data from database
            historical_vols = self._get_historical_volatility(symbol, days=252)  # 1 year
            
            if not historical_vols:
                return 50.0, 50.0  # Default middle rank
            
            # Calculate rank
            rank = sum(1 for vol in historical_vols if current_vol > vol) / len(historical_vols) * 100
            
            # Calculate percentile
            percentile = np.percentile(historical_vols, 50)  # Median percentile
            
            return rank, percentile
            
        except Exception:
            return 50.0, 50.0
    
    def _get_historical_volatility(self, symbol: str, days: int) -> List[float]:
        """Get historical volatility data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            cursor.execute("""
                SELECT price_volatility FROM volatility_metrics 
                WHERE symbol = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (symbol, cutoff_date))
            
            results = [row[0] for row in cursor.fetchall() if row[0] is not None]
            conn.close()
            
            return results
            
        except Exception as e:
            logger.debug(f"Historical volatility error: {e}")
            return []
    
    def _forecast_volatility(self, prices: List[float], volumes: List[int]) -> float:
        """Simple volatility forecast"""
        try:
            if len(prices) < 10:
                return 0.0
            
            # Simple exponential smoothing for volatility forecast
            recent_volatility = np.std(prices[-10:]) / np.mean(prices[-10:]) * 100
            medium_volatility = np.std(prices[-20:]) / np.mean(prices[-20:]) * 100 if len(prices) >= 20 else recent_volatility
            
            # Weighted forecast
            forecast = (recent_volatility * 0.7) + (medium_volatility * 0.3)
            
            return forecast
            
        except Exception:
            return 0.0
    
    def _analyze_trend(self, prices: List[float]) -> Tuple[str, float]:
        """Analyze price trend and momentum"""
        try:
            if len(prices) < 10:
                return "neutral", 0.0
            
            # Simple trend analysis
            recent_prices = prices[-10:]
            older_prices = prices[-20:-10] if len(prices) >= 20 else prices[:len(prices)//2]
            
            recent_avg = np.mean(recent_prices)
            older_avg = np.mean(older_prices)
            
            if recent_avg > older_avg * 1.002:  # 0.2% threshold
                trend = "up"
                momentum = ((recent_avg - older_avg) / older_avg) * 100
            elif recent_avg < older_avg * 0.998:
                trend = "down"
                momentum = ((recent_avg - older_avg) / older_avg) * 100
            else:
                trend = "neutral"
                momentum = 0.0
            
            # Cap momentum score
            momentum = max(-10, min(momentum, 10))
            
            return trend, momentum
            
        except Exception:
            return "neutral", 0.0
    
    def _generate_scalping_signal(self, price_vol: float, atr_pct: float, 
                                vwap_dev: float, trend: str) -> str:
        """Generate scalping signal based on volatility"""
        try:
            # High volatility + trend alignment = scalping opportunity
            if price_vol > 20 and atr_pct > 1.0:  # High volatility
                if trend == "up" and vwap_dev < -0.3:  # Uptrend, price below VWAP
                    return "BUY"
                elif trend == "down" and vwap_dev > 0.3:  # Downtrend, price above VWAP
                    return "SELL"
            
            # Low volatility = avoid trading
            if price_vol < 10 or atr_pct < 0.5:
                return "AVOID"
            
            return "NEUTRAL"
            
        except Exception:
            return "NEUTRAL"
    
    def _detect_volatility_breakout(self, price_vol: float, atr_pct: float, squeeze: bool) -> bool:
        """Detect volatility breakout"""
        try:
            # Breakout after squeeze with high volatility
            if squeeze and price_vol > 25 and atr_pct > 1.5:
                return True
            
            # Sudden volatility spike
            if price_vol > 30 and atr_pct > 2.0:
                return True
            
            return False
            
        except Exception:
            return False
    
    def _detect_mean_reversion(self, vwap_dev: float, vol_rank: float) -> bool:
        """Detect mean reversion opportunity"""
        try:
            # High VWAP deviation in low volatility environment
            if abs(vwap_dev) > 1.0 and vol_rank < 30:
                return True
            
            # Extreme deviation
            if abs(vwap_dev) > 2.0:
                return True
            
            return False
            
        except Exception:
            return False
    
    def _save_metrics_to_db(self, metrics: VolatilityMetrics):
        """Save metrics to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO volatility_metrics 
                (symbol, timestamp, price_volatility, intraday_range, range_percentage,
                 atr_14, atr_percentage, bollinger_squeeze, vwap_deviation,
                 volume_weighted_volatility, standard_deviation, variance,
                 skewness, kurtosis, volatility_rank, volatility_percentile,
                 volatility_forecast, trend_direction, momentum_score,
                 scalping_signal, volatility_breakout, mean_reversion_signal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.symbol, metrics.timestamp, metrics.price_volatility,
                metrics.intraday_range, metrics.range_percentage,
                metrics.atr_14, metrics.atr_percentage, metrics.bollinger_squeeze,
                metrics.vwap_deviation, metrics.volume_weighted_volatility,
                metrics.standard_deviation, metrics.variance,
                metrics.skewness, metrics.kurtosis, metrics.volatility_rank,
                metrics.volatility_percentile, metrics.volatility_forecast,
                metrics.trend_direction, metrics.momentum_score,
                metrics.scalping_signal, metrics.volatility_breakout,
                metrics.mean_reversion_signal
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.debug(f"Save metrics error: {e}")
    
    def _check_alerts(self, metrics: VolatilityMetrics):
        """Check for volatility alerts"""
        try:
            alerts = []
            
            # High volatility alert
            if metrics.price_volatility > 50:
                alerts.append({
                    'type': 'HIGH_VOLATILITY',
                    'message': f'{metrics.symbol} volatility spike: {metrics.price_volatility:.1f}%',
                    'value': metrics.price_volatility
                })
            
            # Volatility breakout alert
            if metrics.volatility_breakout:
                alerts.append({
                    'type': 'VOLATILITY_BREAKOUT',
                    'message': f'{metrics.symbol} volatility breakout detected',
                    'value': metrics.price_volatility
                })
            
            # Mean reversion alert
            if metrics.mean_reversion_signal:
                alerts.append({
                    'type': 'MEAN_REVERSION',
                    'message': f'{metrics.symbol} mean reversion opportunity',
                    'value': metrics.vwap_deviation
                })
            
            # Save alerts to database
            for alert in alerts:
                self._save_alert(metrics.symbol, alert)
                
            # Notify alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(metrics.symbol, alerts)
                except Exception as e:
                    logger.debug(f"Alert callback error: {e}")
                    
        except Exception as e:
            logger.debug(f"Check alerts error: {e}")
    
    def _save_alert(self, symbol: str, alert: Dict):
        """Save alert to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO volatility_alerts 
                (symbol, alert_type, message, volatility_value)
                VALUES (?, ?, ?, ?)
            """, (
                symbol,
                alert['type'],
                alert['message'],
                alert['value']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.debug(f"Save alert error: {e}")
    
    def _notify_callbacks(self):
        """Notify update callbacks"""
        for callback in self.update_callbacks:
            try:
                callback(dict(self.volatility_cache))
            except Exception as e:
                logger.debug(f"Update callback error: {e}")
    
    def add_update_callback(self, callback):
        """Add update callback"""
        self.update_callbacks.append(callback)
    
    def add_alert_callback(self, callback):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def get_volatility_summary(self) -> Dict:
        """Get volatility summary for all symbols"""
        try:
            summary = {
                'total_symbols': len(self.volatility_cache),
                'high_volatility_count': 0,
                'breakout_signals': 0,
                'mean_reversion_signals': 0,
                'scalping_opportunities': 0,
                'symbol_metrics': {}
            }
            
            for symbol, metrics in self.volatility_cache.items():
                if metrics.price_volatility > 30:
                    summary['high_volatility_count'] += 1
                
                if metrics.volatility_breakout:
                    summary['breakout_signals'] += 1
                
                if metrics.mean_reversion_signal:
                    summary['mean_reversion_signals'] += 1
                
                if metrics.scalping_signal in ['BUY', 'SELL']:
                    summary['scalping_opportunities'] += 1
                
                summary['symbol_metrics'][symbol] = metrics.to_dict()
            
            return summary
            
        except Exception as e:
            logger.error(f"Volatility summary error: {e}")
            return {'error': str(e)}
    
    def get_alerts_history(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            cursor.execute("""
                SELECT symbol, alert_type, message, volatility_value, timestamp
                FROM volatility_alerts 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, (cutoff_time,))
            
            alerts = []
            for row in cursor.fetchall():
                alerts.append({
                    'symbol': row[0],
                    'alert_type': row[1],
                    'message': row[2],
                    'volatility_value': row[3],
                    'timestamp': row[4]
                })
            
            conn.close()
            return alerts
            
        except Exception as e:
            logger.error(f"Alerts history error: {e}")
            return []
    
    def cleanup(self):
        """Cleanup analyzer resources"""
        try:
            self.stop_analysis()
            
            # Clear data
            self.price_buffers.clear()
            self.volume_buffers.clear()
            self.volatility_cache.clear()
            self.update_callbacks.clear()
            self.alert_callbacks.clear()
            
            logger.info("🧹 Volatility analyzer cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Volatility analyzer cleanup error: {e}")


# Usage example
if __name__ == "__main__":
    # Test volatility analyzer
    import random
    
    analyzer = RealTimeVolatilityAnalyzer(None)
    
    # Add test symbols
    analyzer.add_symbols(['RELIANCE', 'TCS', 'INFY'])
    
    # Start analysis
    analyzer.start_analysis()
    
    # Add callback
    def volatility_update(metrics_dict):
        print(f"📊 Volatility update: {len(metrics_dict)} symbols")
    
    analyzer.add_update_callback(volatility_update)
    
    # Simulate market data
    # Simulate market data
    base_prices = {'RELIANCE': 2500, 'TCS': 3200, 'INFY': 1450}
    
    for i in range(100):
        for symbol in ['RELIANCE', 'TCS', 'INFY']:
            base_price = base_prices[symbol]
            
            # Simulate price with volatility
            price = base_price + random.uniform(-base_price * 0.02, base_price * 0.02)
            volume = random.randint(1000, 10000)
            
            # Update analyzer
            analyzer.update_data(symbol, price, volume)
        
        time.sleep(0.1)  # 100ms intervals
    
    # Wait for analysis
    time.sleep(35)  # Wait for analysis cycle
    
    # Get summary
    summary = analyzer.get_volatility_summary()
    print(f"📈 Volatility Summary: {json.dumps(summary, indent=2)}")
    
    # Get alerts
    alerts = analyzer.get_alerts_history(1)  # Last hour
    print(f"🚨 Recent Alerts: {len(alerts)}")
    
    analyzer.cleanup()
    print("✅ Volatility analyzer test completed")
