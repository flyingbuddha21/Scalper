#!/usr/bin/env python3
"""
Market Data Manager
Handles market data collection, caching, and distribution
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import sqlite3
import json
from collections import deque
import queue

logger = logging.getLogger(__name__)

class MarketDataManager:
    """Manages market data collection and distribution"""
    
    def __init__(self, api_handler, db_path: str = "data/market_data.db"):
        self.api = api_handler
        self.db_path = db_path
        self.running = False
        
        # Data storage
        self.realtime_data = {}  # symbol -> latest data
        self.historical_cache = {}  # symbol -> historical data
        self.subscribers = []  # List of callback functions
        
        # Data queues
        self.data_queue = queue.Queue(maxsize=1000)
        self.update_queue = queue.Queue(maxsize=500)
        
        # Configuration
        self.update_interval = 1.0  # 1 second
        self.cache_duration = timedelta(hours=1)  # Cache for 1 hour
        
        # Statistics
        self.stats = {
            'updates_processed': 0,
            'api_calls_made': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }
        
        logger.info("📊 Market Data Manager initialized")
    
    def start_data_collection(self, symbols: List[str]):
        """Start collecting market data for symbols"""
        try:
            self.running = True
            self.symbols = symbols
            
            # Start data collection thread
            data_thread = threading.Thread(
                target=self._data_collection_loop,
                daemon=True,
                name="DataCollection"
            )
            data_thread.start()
            
            # Start data processing thread
            process_thread = threading.Thread(
                target=self._data_processing_loop,
                daemon=True,
                name="DataProcessing"
            )
            process_thread.start()
            
            logger.info(f"🚀 Market data collection started for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"❌ Start data collection error: {e}")
    
    def stop_data_collection(self):
        """Stop market data collection"""
        self.running = False
        logger.info("⏹️ Market data collection stopped")
    
    def _data_collection_loop(self):
        """Main data collection loop"""
        while self.running:
            try:
                for symbol in self.symbols:
                    try:
                        # Get market data from API
                        market_data = self._fetch_market_data(symbol)
                        
                        if market_data:
                            # Add to processing queue
                            if not self.data_queue.full():
                                self.data_queue.put({
                                    'symbol': symbol,
                                    'data': market_data,
                                    'timestamp': datetime.now()
                                })
                            
                            self.stats['api_calls_made'] += 1
                        
                    except Exception as e:
                        logger.debug(f"Data collection error for {symbol}: {e}")
                        self.stats['errors'] += 1
                        continue
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"❌ Data collection loop error: {e}")
                time.sleep(5)
    
    def _data_processing_loop(self):
        """Process collected market data"""
        while self.running:
            try:
                # Get data from queue
                data_packet = self.data_queue.get(timeout=1)
                
                symbol = data_packet['symbol']
                market_data = data_packet['data']
                timestamp = data_packet['timestamp']
                
                # Update realtime data
                self.realtime_data[symbol] = {
                    **market_data,
                    'last_updated': timestamp
                }
                
                # Save to database
                self._save_to_database(symbol, market_data, timestamp)
                
                # Notify subscribers
                self._notify_subscribers(symbol, market_data)
                
                self.stats['updates_processed'] += 1
                self.data_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Data processing error: {e}")
    
    def _fetch_market_data(self, symbol: str) -> Optional[Dict]:
        """Fetch market data for symbol"""
        try:
            if self.api and hasattr(self.api, 'get_quotes'):
                # Real API call
                response = self.api.get_quotes([symbol])
                
                if response and response.get('status') == 'success':
                    data = response.get('data', {})
                    if symbol in data:
                        return self._normalize_market_data(data[symbol])
                
            # Fallback to mock data
            return self._generate_mock_data(symbol)
            
        except Exception as e:
            logger.debug(f"Fetch market data error for {symbol}: {e}")
            return self._generate_mock_data(symbol)
    
    def _normalize_market_data(self, raw_data: Dict) -> Dict:
        """Normalize market data from API"""
        try:
            return {
                'ltp': float(raw_data.get('ltp', 0)),
                'bid': float(raw_data.get('bid', 0)),
                'ask': float(raw_data.get('ask', 0)),
                'bid_qty': int(raw_data.get('bid_qty', 0)),
                'ask_qty': int(raw_data.get('ask_qty', 0)),
                'volume': int(raw_data.get('volume', 0)),
                'open': float(raw_data.get('open', 0)),
                'high': float(raw_data.get('high', 0)),
                'low': float(raw_data.get('low', 0)),
                'prev_close': float(raw_data.get('prev_close', 0)),
                'change_pct': float(raw_data.get('change_pct', 0))
            }
            
        except Exception as e:
            logger.debug(f"Normalize data error: {e}")
            return {}
    
    def _generate_mock_data(self, symbol: str) -> Dict:
        """Generate mock market data for testing"""
        import random
        
        # Base prices for common symbols
        base_prices = {
            'RELIANCE': 2500, 'TCS': 3200, 'INFY': 1450,
            'HDFCBANK': 1650, 'ICICIBANK': 950, 'NIFTY': 19500,
            'BANKNIFTY': 44000
        }
        
        base_price = base_prices.get(symbol, 1000)
        
        # Generate realistic mock data
        ltp = base_price + random.uniform(-base_price * 0.02, base_price * 0.02)
        spread = random.uniform(0.05, 0.50)
        
        return {
            'ltp': round(ltp, 2),
            'bid': round(ltp - spread/2, 2),
            'ask': round(ltp + spread/2, 2),
            'bid_qty': random.randint(100, 2000),
            'ask_qty': random.randint(100, 2000),
            'volume': random.randint(1000, 50000),
            'open': round(ltp + random.uniform(-10, 10), 2),
            'high': round(ltp + random.uniform(0, 15), 2),
            'low': round(ltp - random.uniform(0, 15), 2),
            'prev_close': round(ltp + random.uniform(-5, 5), 2),
            'change_pct': round(random.uniform(-3, 3), 2)
        }
    
    def _save_to_database(self, symbol: str, data: Dict, timestamp: datetime):
        """Save market data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO realtime_quotes 
                (symbol, timestamp, ltp, bid, ask, bid_qty, ask_qty, volume,
                 open_price, high_price, low_price, prev_close, change_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, timestamp, data.get('ltp', 0), data.get('bid', 0),
                data.get('ask', 0), data.get('bid_qty', 0), data.get('ask_qty', 0),
                data.get('volume', 0), data.get('open', 0), data.get('high', 0),
                data.get('low', 0), data.get('prev_close', 0), data.get('change_pct', 0)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.debug(f"Save to database error: {e}")
    
    def _notify_subscribers(self, symbol: str, data: Dict):
        """Notify all subscribers of data update"""
        for callback in self.subscribers:
            try:
                callback(symbol, data)
            except Exception as e:
                logger.debug(f"Subscriber notification error: {e}")
    
    def subscribe(self, callback_function):
        """Subscribe to market data updates"""
        self.subscribers.append(callback_function)
        logger.info("📡 New subscriber added to market data feed")
    
    def unsubscribe(self, callback_function):
        """Unsubscribe from market data updates"""
        if callback_function in self.subscribers:
            self.subscribers.remove(callback_function)
            logger.info("📡 Subscriber removed from market data feed")
    
    def get_latest_data(self, symbol: str) -> Optional[Dict]:
        """Get latest market data for symbol"""
        return self.realtime_data.get(symbol)
    
    def get_historical_data(self, symbol: str, hours: int = 24) -> List[Dict]:
        """Get historical data for symbol"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{hours}h"
            if cache_key in self.historical_cache:
                cached_data, cache_time = self.historical_cache[cache_key]
                if datetime.now() - cache_time < self.cache_duration:
                    self.stats['cache_hits'] += 1
                    return cached_data
            
            # Fetch from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            cursor.execute("""
                SELECT timestamp, ltp, bid, ask, volume, open_price, high_price, low_price
                FROM realtime_quotes 
                WHERE symbol = ? AND timestamp > ?
                ORDER BY timestamp ASC
            """, (symbol, cutoff_time))
            
            rows = cursor.fetchall()
            conn.close()
            
            historical_data = []
            for row in rows:
                historical_data.append({
                    'timestamp': row[0],
                    'ltp': row[1],
                    'bid': row[2],
                    'ask': row[3],
                    'volume': row[4],
                    'open': row[5],
                    'high': row[6],
                    'low': row[7]
                })
            
            # Cache the result
            self.historical_cache[cache_key] = (historical_data, datetime.now())
            self.stats['cache_misses'] += 1
            
            return historical_data
            
        except Exception as e:
            logger.error(f"❌ Get historical data error: {e}")
            return []
    
    def get_ohlc_data(self, symbol: str, timeframe: str = '1min', periods: int = 100) -> pd.DataFrame:
        """Get OHLC data as DataFrame"""
        try:
            historical_data = self.get_historical_data(symbol, hours=24)
            
            if not historical_data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Resample to timeframe
            ohlc = df['ltp'].resample(timeframe).ohlc()
            volume = df['volume'].resample(timeframe).sum()
            
            # Combine OHLC and volume
            ohlc_data = pd.concat([ohlc, volume], axis=1)
            ohlc_data.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Remove NaN rows and get last N periods
            ohlc_data = ohlc_data.dropna().tail(periods)
            
            return ohlc_data.reset_index()
            
        except Exception as e:
            logger.error(f"❌ Get OHLC data error: {e}")
            return pd.DataFrame()
    
    def get_market_summary(self) -> Dict:
        """Get market data summary"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'symbols_tracked': len(self.symbols) if hasattr(self, 'symbols') else 0,
                'realtime_data_count': len(self.realtime_data),
                'subscribers_count': len(self.subscribers),
                'running': self.running,
                'statistics': self.stats.copy(),
                'queue_sizes': {
                    'data_queue': self.data_queue.qsize(),
                    'update_queue': self.update_queue.qsize()
                }
            }
            
            # Add symbol-wise latest data
            summary['latest_prices'] = {}
            for symbol, data in self.realtime_data.items():
                summary['latest_prices'][symbol] = {
                    'ltp': data.get('ltp', 0),
                    'change_pct': data.get('change_pct', 0),
                    'volume': data.get('volume', 0),
                    'last_updated': data.get('last_updated', datetime.now()).isoformat()
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Market summary error: {e}")
            return {'error': str(e)}
    
    def cleanup_old_data(self, days: int = 7):
        """Clean up old market data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            cursor.execute("""
                DELETE FROM realtime_quotes 
                WHERE timestamp < ?
            """, (cutoff_date,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"🧹 Cleaned up {deleted_count} old market data records")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Cleanup old data error: {e}")
            return 0
    
    def get_statistics(self) -> Dict:
        """Get data manager statistics"""
        return {
            'statistics': self.stats.copy(),
            'uptime_seconds': time.time() - getattr(self, '_start_time', time.time()),
            'symbols_count': len(self.symbols) if hasattr(self, 'symbols') else 0,
            'cache_size': len(self.historical_cache),
            'realtime_data_size': len(self.realtime_data),
            'subscribers_count': len(self.subscribers)
        }


# Usage example and testing
if __name__ == "__main__":
    import time
    
    # Test market data manager
    print("📊 Testing Market Data Manager...")
    
    # Mock API for testing
    class MockAPI:
        def get_quotes(self, symbols):
            return {
                'status': 'success',
                'data': {
                    symbols[0]: {
                        'ltp': 2500.75,
                        'bid': 2500.50,
                        'ask': 2501.00,
                        'volume': 125000
                    }
                }
            }
    
    # Initialize manager
    mock_api = MockAPI()
    data_manager = MarketDataManager(mock_api)
    
    # Add subscriber
    def data_callback(symbol, data):
        print(f"📈 Data update: {symbol} @ ₹{data.get('ltp', 0)}")
    
    data_manager.subscribe(data_callback)
    
    # Start data collection
    symbols = ['RELIANCE', 'TCS', 'INFY']
    data_manager.start_data_collection(symbols)
    
    # Let it run for a bit
    time.sleep(5)
    
    # Get summary
    summary = data_manager.get_market_summary()
    print(f"📊 Summary: {summary['symbols_tracked']} symbols, {summary['statistics']['updates_processed']} updates")
    
    # Get latest data
    latest = data_manager.get_latest_data('RELIANCE')
    if latest:
        print(f"💰 RELIANCE latest: ₹{latest.get('ltp', 0)}")
    
    # Stop manager
    data_manager.stop_data_collection()
    print("✅ Market data manager test completed")
