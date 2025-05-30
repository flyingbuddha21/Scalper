"""
Advanced Data Manager for Indian Market Trading Bot
Handles real-time market data from Goodwill API with OHLC, Tick, and L1 feeds
Optimized for high-frequency scalping strategies
"""

import logging
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import time
import json

from config_manager import get_config
from utils import format_currency, format_percentage

@dataclass
class TickData:
    """Individual tick data structure"""
    symbol: str
    timestamp: datetime
    last_price: float
    volume: int
    bid_price: float
    ask_price: float
    bid_qty: int
    ask_qty: int
    total_buy_qty: int
    total_sell_qty: int
    change: float
    change_percent: float

@dataclass
class OHLCCandle:
    """OHLC candle data structure"""
    symbol: str
    timestamp: datetime
    timeframe: str  # 1min, 5min, 15min
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    trades: int

@dataclass
class L1OrderBook:
    """Level 1 order book data"""
    symbol: str
    timestamp: datetime
    bid_price: float
    bid_qty: int
    ask_price: float
    ask_qty: int
    spread: float
    spread_percent: float
    imbalance: float  # (bid_qty - ask_qty) / (bid_qty + ask_qty)

class MarketDataManager:
    """Comprehensive market data management for Indian equities"""
    
    def __init__(self):
        self.config = get_config()
        self.session: Optional[aiohttp.ClientSession] = """
Advanced Data Manager for Indian Market Trading Bot
Uses Goodwill WebSocket for real-time market data feeds
Handles OHLC, Tick, and L1 data from Goodwill WebSocket
"""

import logging
import pandas as pd
import numpy as np
import asyncio
import aiohttp
import websockets
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import time

from config_manager import get_config
from utils import format_currency, format_percentage
from goodwill_api_handler import get_goodwill_handler

@dataclass
class TickData:
    """Individual tick data structure"""
    symbol: str
    timestamp: datetime
    last_price: float
    volume: int
    bid_price: float
    ask_price: float
    bid_qty: int
    ask_qty: int
    total_buy_qty: int
    total_sell_qty: int
    change: float
    change_percent: float

@dataclass
class OHLCCandle:
    """OHLC candle data structure"""
    symbol: str
    timestamp: datetime
    timeframe: str  # 1min, 5min, 15min
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    trades: int

@dataclass
class L1OrderBook:
    """Level 1 order book data"""
    symbol: str
    timestamp: datetime
    bid_price: float
    bid_qty: int
    ask_price: float
    ask_qty: int
    spread: float
    spread_percent: float
    imbalance: float  # (bid_qty - ask_qty) / (bid_qty + ask_qty)

class GoodwillWebSocketManager:
    """Real-time market data manager using Goodwill WebSocket feed"""
    
    def __init__(self):
        self.config = get_config()
        self.goodwill_handler = get_goodwill_handler()
        self.session: Optional[aiohttp.ClientSession] = None
        
        # WebSocket connection
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.websocket_url = "wss://giga.gwcindia.in/NorenWSTP/"
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        
        # Data storage
        self.tick_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.ohlc_buffers: Dict[str, Dict[str, deque]] = defaultdict(lambda: {
            '1min': deque(maxlen=500),
            '5min': deque(maxlen=200),
            '15min': deque(maxlen=100)
        })
        self.l1_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Real-time subscribers
        self.tick_subscribers: List[Callable] = []
        self.ohlc_subscribers: List[Callable] = []
        self.l1_subscribers: List[Callable] = []
        
        # Market status
        self.market_status = "CLOSED"
        self.subscribed_symbols: set = set()
        
        # Data quality tracking
        self.data_quality_metrics = {
            'tick_count': 0,
            'ohlc_count': 0,
            'l1_count': 0,
            'last_update': None,
            'connection_status': 'DISCONNECTED',
            'error_count': 0,
            'websocket_reconnects': 0
        }
        
        logging.info("Goodwill WebSocket Data Manager initialized")
    
    async def initialize(self):
        """Initialize WebSocket connection and data feeds"""
        try:
            # Create aiohttp session
            timeout = aiohttp.ClientTimeout(total=self.config.api.timeout)
            connector = aiohttp.TCPConnector(limit=50, limit_per_host=30)
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    'User-Agent': 'MomentumBot/1.0'
                }
            )
            
            # Initialize Goodwill API handler
            if not self.goodwill_handler.is_authenticated:
                await self.goodwill_handler.initialize()
            
            # Connect to WebSocket
            if self.goodwill_handler.is_authenticated:
                await self._connect_websocket()
            else:
                logging.warning("Goodwill API not authenticated - WebSocket connection will be established after authentication")
            
            # Start background tasks
            asyncio.create_task(self._websocket_monitor())
            asyncio.create_task(self._data_quality_monitor())
            
            logging.info("Goodwill WebSocket Data Manager initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize Goodwill WebSocket Data Manager: {e}")
            raise
    
    async def _connect_websocket(self):
        """Connect to Goodwill WebSocket feed"""
        try:
            if not self.goodwill_handler.is_authenticated:
                logging.error("Cannot connect to WebSocket: not authenticated")
                return False
            
            logging.info(f"Connecting to Goodwill WebSocket: {self.websocket_url}")
            
            # Connect to WebSocket
            self.websocket = await websockets.connect(
                self.websocket_url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            
            # Send connection request
            connection_request = {
                "t": "c",
                "uid": self.goodwill_handler.login_session.client_id,
                "actid": self.goodwill_handler.login_session.client_id,
                "source": "API",
                "susertoken": self.goodwill_handler.login_session.user_session_id
            }
            
            await self.websocket.send(json.dumps(connection_request))
            
            # Wait for connection acknowledgment
            response = await self.websocket.recv()
            response_data = json.loads(response)
            
            if response_data.get('t') == 'ck' and response_data.get('s') == 'OK':
                self.is_connected = True
                self.data_quality_metrics['connection_status'] = 'CONNECTED'
                self.reconnect_attempts = 0
                
                logging.info("✅ Connected to Goodwill WebSocket successfully")
                
                # Start message processing
                asyncio.create_task(self._process_websocket_messages())
                
                return True
            else:
                logging.error(f"WebSocket connection failed: {response_data}")
                return False
                
        except Exception as e:
            logging.error(f"Error connecting to WebSocket: {e}")
            self.is_connected = False
            return False
    
    async def _process_websocket_messages(self):
        """Process incoming WebSocket messages"""
        try:
            while self.is_connected and self.websocket:
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=60)
                    await self._handle_websocket_message(message)
                    
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    if self.websocket:
                        await self.websocket.ping()
                    
                except websockets.exceptions.ConnectionClosed:
                    logging.warning("WebSocket connection closed")
                    self.is_connected = False
                    break
                    
                except Exception as e:
                    logging.error(f"Error processing WebSocket message: {e}")
                    self.data_quality_metrics['error_count'] += 1
                    
        except Exception as e:
            logging.error(f"Error in WebSocket message processing: {e}")
            self.is_connected = False
    
    async def _handle_websocket_message(self, message: str):
        """Handle individual WebSocket message"""
        try:
            data = json.loads(message)
            message_type = data.get('t', '')
            
            if message_type == 'tf':  # Touchline feed
                await self._process_touchline_feed(data)
            elif message_type == 'df':  # Depth feed (L1 data)
                await self._process_depth_feed(data)
            elif message_type == 'tk':  # Touchline acknowledgment
                logging.info(f"Subscribed to touchline for {data.get('ts', 'unknown')}")
            elif message_type == 'dk':  # Depth acknowledgment
                logging.info(f"Subscribed to depth for {data.get('ts', 'unknown')}")
            elif message_type == 'uk':  # Unsubscribe acknowledgment
                logging.info(f"Unsubscribed from touchline")
            elif message_type == 'udk':  # Unsubscribe depth acknowledgment
                logging.info(f"Unsubscribed from depth")
            else:
                logging.debug(f"Unknown message type: {message_type}")
                
        except Exception as e:
            logging.error(f"Error handling WebSocket message: {e}")
    
    async def _process_touchline_feed(self, data: Dict):
        """Process touchline feed data (real-time prices)"""
        try:
            symbol = self._convert_token_to_symbol(data.get('e', ''), data.get('tk', ''))
            if not symbol:
                return
            
            # Create tick data
            tick = TickData(
                symbol=symbol,
                timestamp=datetime.now(),
                last_price=float(data.get('lp', 0)),
                volume=int(data.get('v', 0)),
                bid_price=float(data.get('bp1', 0
        
        # Data storage
        self.tick_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.ohlc_buffers: Dict[str, Dict[str, deque]] = defaultdict(lambda: {
            '1min': deque(maxlen=500),
            '5min': deque(maxlen=200),
            '15min': deque(maxlen=100)
        })
        self.l1_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Real-time subscribers
        self.tick_subscribers: List[Callable] = []
        self.ohlc_subscribers: List[Callable] = []
        self.l1_subscribers: List[Callable] = []
        
        # Market status
        self.market_status = "CLOSED"
        self.subscribed_symbols: set = set()
        
        # Data quality tracking
        self.data_quality_metrics = {
            'tick_count': 0,
            'ohlc_count': 0,
            'l1_count': 0,
            'last_update': None,
            'connection_status': 'DISCONNECTED',
            'error_count': 0
        }
        
        # Rate limiting
        self.rate_limiter = {
            'requests_per_minute': 0,
            'last_minute': datetime.now().minute,
            'max_requests': self.config.api.rate_limit_requests
        }
        
        logging.info("Market Data Manager initialized")
    
    async def initialize(self):
        """Initialize connection and data feeds"""
        try:
            # Create aiohttp session
            timeout = aiohttp.ClientTimeout(total=self.config.api.timeout)
            connector = aiohttp.TCPConnector(limit=50, limit_per_host=30)
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    'Authorization': f'Bearer {self.config.api.goodwill_api_key}',
                    'Content-Type': 'application/json',
                    'User-Agent': 'MomentumBot/1.0'
                }
            )
            
            # Check market status
            await self._update_market_status()
            
            # Start background tasks
            asyncio.create_task(self._market_data_loop())
            asyncio.create_task(self._data_quality_monitor())
            
            self.data_quality_metrics['connection_status'] = 'CONNECTED'
            logging.info("Market Data Manager initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize Market Data Manager: {e}")
            raise
    
    async def subscribe_symbol(self, symbol: str) -> bool:
        """Subscribe to real-time data for a symbol"""
        try:
            if symbol in self.subscribed_symbols:
                return True
            
            # Subscribe to Goodwill API feeds
            subscription_payload = {
                'symbols': [symbol],
                'feed_types': ['tick', 'ohlc_1min', 'ohlc_5min', 'level1'],
                'subscription_id': f"momentum_bot_{symbol}_{int(time.time())}"
            }
            
            url = f"{self.config.api.goodwill_base_url}/market-data/subscribe"
            
            async with self.session.post(url, json=subscription_payload) as response:
                if response.status == 200:
                    self.subscribed_symbols.add(symbol)
                    logging.info(f"Successfully subscribed to {symbol}")
                    return True
                else:
                    error_text = await response.text()
                    logging.error(f"Failed to subscribe to {symbol}: {error_text}")
                    return False
                    
        except Exception as e:
            logging.error(f"Error subscribing to {symbol}: {e}")
            return False
    
    async def unsubscribe_symbol(self, symbol: str) -> bool:
        """Unsubscribe from real-time data for a symbol"""
        try:
            if symbol not in self.subscribed_symbols:
                return True
            
            url = f"{self.config.api.goodwill_base_url}/market-data/unsubscribe"
            payload = {'symbols': [symbol]}
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    self.subscribed_symbols.discard(symbol)
                    
                    # Clear buffers
                    if symbol in self.tick_buffers:
                        del self.tick_buffers[symbol]
                    if symbol in self.ohlc_buffers:
                        del self.ohlc_buffers[symbol]
                    if symbol in self.l1_buffers:
                        del self.l1_buffers[symbol]
                    
                    logging.info(f"Successfully unsubscribed from {symbol}")
                    return True
                else:
                    logging.error(f"Failed to unsubscribe from {symbol}")
                    return False
                    
        except Exception as e:
            logging.error(f"Error unsubscribing from {symbol}: {e}")
            return False
    
    async def get_historical_data(self, symbol: str, timeframe: str = '1min', 
                                 days: int = 5) -> pd.DataFrame:
        """Get historical OHLC data for indicators calculation"""
        try:
            if not await self._check_rate_limit():
                await asyncio.sleep(1)
                return pd.DataFrame()
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"{self.config.api.goodwill_base_url}/market-data/historical"
            params = {
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'limit': 1000
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'candles' in data and data['candles']:
                        df = pd.DataFrame(data['candles'])
                        
                        # Ensure proper column types
                        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                        for col in numeric_cols:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        # Convert timestamp
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            df = df.sort_values('timestamp').reset_index(drop=True)
                        
                        logging.info(f"Retrieved {len(df)} historical candles for {symbol}")
                        return df
                    else:
                        logging.warning(f"No historical data available for {symbol}")
                        return pd.DataFrame()
                else:
                    error_text = await response.text()
                    logging.error(f"Failed to get historical data for {symbol}: {error_text}")
                    return pd.DataFrame()
                    
        except Exception as e:
            logging.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_latest_tick(self, symbol: str) -> Optional[TickData]:
        """Get the latest tick data for a symbol"""
        try:
            if symbol not in self.tick_buffers or not self.tick_buffers[symbol]:
                return None
            
            return self.tick_buffers[symbol][-1]
            
        except Exception as e:
            logging.error(f"Error getting latest tick for {symbol}: {e}")
            return None
    
    def get_latest_ohlc(self, symbol: str, timeframe: str = '1min') -> Optional[OHLCCandle]:
        """Get the latest OHLC candle for a symbol"""
        try:
            if (symbol not in self.ohlc_buffers or 
                timeframe not in self.ohlc_buffers[symbol] or
                not self.ohlc_buffers[symbol][timeframe]):
                return None
            
            return self.ohlc_buffers[symbol][timeframe][-1]
            
        except Exception as e:
            logging.error(f"Error getting latest OHLC for {symbol}: {e}")
            return None
    
    def get_latest_l1(self, symbol: str) -> Optional[L1OrderBook]:
        """Get the latest L1 order book data for a symbol"""
        try:
            if symbol not in self.l1_buffers or not self.l1_buffers[symbol]:
                return None
            
            return self.l1_buffers[symbol][-1]
            
        except Exception as e:
            logging.error(f"Error getting latest L1 for {symbol}: {e}")
            return None
    
    def get_ohlc_dataframe(self, symbol: str, timeframe: str = '1min', 
                          periods: int = 200) -> pd.DataFrame:
        """Get OHLC data as pandas DataFrame for strategy analysis"""
        try:
            if (symbol not in self.ohlc_buffers or 
                timeframe not in self.ohlc_buffers[symbol]):
                return pd.DataFrame()
            
            candles = list(self.ohlc_buffers[symbol][timeframe])[-periods:]
            
            if not candles:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for candle in candles:
                data.append({
                    'timestamp': candle.timestamp,
                    'open': candle.open,
                    'high': candle.high,
                    'low': candle.low,
                    'close': candle.close,
                    'volume': candle.volume,
                    'vwap': candle.vwap
                })
            
            df = pd.DataFrame(data)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Error getting OHLC DataFrame for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_market_data_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market data snapshot for strategy analysis"""
        try:
            latest_tick = self.get_latest_tick(symbol)
            latest_ohlc = self.get_latest_ohlc(symbol, '1min')
            latest_l1 = self.get_latest_l1(symbol)
            ohlc_df = self.get_ohlc_dataframe(symbol, '1min', 100)
            
            snapshot = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'tick_data': asdict(latest_tick) if latest_tick else {},
                'ohlc_data': {
                    'latest_candle': asdict(latest_ohlc) if latest_ohlc else {},
                    'candles': ohlc_df.to_dict('records') if not ohlc_df.empty else []
                },
                'l1_data': asdict(latest_l1) if latest_l1 else {},
                'current_price': latest_tick.last_price if latest_tick else 0,
                'current_volume': latest_tick.volume if latest_tick else 0,
                'market_status': self.market_status,
                'data_quality': self._get_symbol_data_quality(symbol)
            }
            
            return snapshot
            
        except Exception as e:
            logging.error(f"Error getting market data snapshot for {symbol}: {e}")
            return {}
    
    async def _market_data_loop(self):
        """Main market data processing loop"""
        while True:
            try:
                if self.market_status != "OPEN":
                    await asyncio.sleep(5)
                    continue
                
                # Fetch real-time updates for all subscribed symbols
                for symbol in list(self.subscribed_symbols):
                    await self._fetch_real_time_data(symbol)
                
                await asyncio.sleep(0.1)  # High frequency updates for scalping
                
            except Exception as e:
                logging.error(f"Error in market data loop: {e}")
                await asyncio.sleep(1)
    
    async def _fetch_real_time_data(self, symbol: str):
        """Fetch real-time data for a specific symbol"""
        try:
            if not await self._check_rate_limit():
                return
            
            url = f"{self.config.api.goodwill_base_url}/market-data/realtime/{symbol}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Process different data types
                    if 'tick' in data:
                        await self._process_tick_data(symbol, data['tick'])
                    
                    if 'ohlc' in data:
                        await self._process_ohlc_data(symbol, data['ohlc'])
                    
                    if 'level1' in data:
                        await self._process_l1_data(symbol, data['level1'])
                    
                    self.data_quality_metrics['last_update'] = datetime.now()
                    
                else:
                    self.data_quality_metrics['error_count'] += 1
                    if response.status == 429:  # Rate limited
                        await asyncio.sleep(1)
                    
        except Exception as e:
            logging.error(f"Error fetching real-time data for {symbol}: {e}")
            self.data_quality_metrics['error_count'] += 1
    
    async def _process_tick_data(self, symbol: str, tick_data: Dict):
        """Process incoming tick data"""
        try:
            tick = TickData(
                symbol=symbol,
                timestamp=datetime.fromisoformat(tick_data.get('timestamp', datetime.now().isoformat())),
                last_price=float(tick_data.get('last_price', 0)),
                volume=int(tick_data.get('volume', 0)),
                bid_price=float(tick_data.get('bid_price', 0)),
                ask_price=float(tick_data.get('ask_price', 0)),
                bid_qty=int(tick_data.get('bid_qty', 0)),
                ask_qty=int(tick_data.get('ask_qty', 0)),
                total_buy_qty=int(tick_data.get('total_buy_qty', 0)),
                total_sell_qty=int(tick_data.get('total_sell_qty', 0)),
                change=float(tick_data.get('change', 0)),
                change_percent=float(tick_data.get('change_percent', 0))
            )
            
            self.tick_buffers[symbol].append(tick)
            self.data_quality_metrics['tick_count'] += 1
            
            # Notify subscribers
            for callback in self.tick_subscribers:
                try:
                    await callback(symbol, tick)
                except Exception as e:
                    logging.error(f"Error in tick subscriber callback: {e}")
                    
        except Exception as e:
            logging.error(f"Error processing tick data for {symbol}: {e}")
    
    async def _process_ohlc_data(self, symbol: str, ohlc_data: Dict):
        """Process incoming OHLC candle data"""
        try:
            for timeframe, candle_data in ohlc_data.items():
                if timeframe not in ['1min', '5min', '15min']:
                    continue
                
                candle = OHLCCandle(
                    symbol=symbol,
                    timestamp=datetime.fromisoformat(candle_data.get('timestamp', datetime.now().isoformat())),
                    timeframe=timeframe,
                    open=float(candle_data.get('open', 0)),
                    high=float(candle_data.get('high', 0)),
                    low=float(candle_data.get('low', 0)),
                    close=float(candle_data.get('close', 0)),
                    volume=int(candle_data.get('volume', 0)),
                    vwap=float(candle_data.get('vwap', 0)),
                    trades=int(candle_data.get('trades', 0))
                )
                
                self.ohlc_buffers[symbol][timeframe].append(candle)
                self.data_quality_metrics['ohlc_count'] += 1
                
                # Notify subscribers
                for callback in self.ohlc_subscribers:
                    try:
                        await callback(symbol, candle)
                    except Exception as e:
                        logging.error(f"Error in OHLC subscriber callback: {e}")
                        
        except Exception as e:
            logging.error(f"Error processing OHLC data for {symbol}: {e}")
    
    async def _process_l1_data(self, symbol: str, l1_data: Dict):
        """Process incoming Level 1 order book data"""
        try:
            bid_price = float(l1_data.get('bid_price', 0))
            ask_price = float(l1_data.get('ask_price', 0))
            bid_qty = int(l1_data.get('bid_qty', 0))
            ask_qty = int(l1_data.get('ask_qty', 0))
            
            spread = ask_price - bid_price if ask_price > 0 and bid_price > 0 else 0
            spread_percent = (spread / ask_price * 100) if ask_price > 0 else 0
            imbalance = ((bid_qty - ask_qty) / (bid_qty + ask_qty)) if (bid_qty + ask_qty) > 0 else 0
            
            l1 = L1OrderBook(
                symbol=symbol,
                timestamp=datetime.fromisoformat(l1_data.get('timestamp', datetime.now().isoformat())),
                bid_price=bid_price,
                bid_qty=bid_qty,
                ask_price=ask_price,
                ask_qty=ask_qty,
                spread=spread,
                spread_percent=spread_percent,
                imbalance=imbalance
            )
            
            self.l1_buffers[symbol].append(l1)
            self.data_quality_metrics['l1_count'] += 1
            
            # Notify subscribers
            for callback in self.l1_subscribers:
                try:
                    await callback(symbol, l1)
                except Exception as e:
                    logging.error(f"Error in L1 subscriber callback: {e}")
                    
        except Exception as e:
            logging.error(f"Error processing L1 data for {symbol}: {e}")
    
    async def _update_market_status(self):
        """Update current market status"""
        try:
            url = f"{self.config.api.goodwill_base_url}/market-data/status"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    status_data = await response.json()
                    self.market_status = status_data.get('market_status', 'UNKNOWN')
                    logging.info(f"Market status: {self.market_status}")
                else:
                    logging.warning("Failed to get market status")
                    
        except Exception as e:
            logging.error(f"Error updating market status: {e}")
    
    async def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        current_minute = datetime.now().minute
        
        if current_minute != self.rate_limiter['last_minute']:
            self.rate_limiter['requests_per_minute'] = 0
            self.rate_limiter['last_minute'] = current_minute
        
        if self.rate_limiter['requests_per_minute'] >= self.rate_limiter['max_requests']:
            return False
        
        self.rate_limiter['requests_per_minute'] += 1
        return True
    
    async def _data_quality_monitor(self):
        """Monitor data quality and connection health"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.now()
                last_update = self.data_quality_metrics.get('last_update')
                
                if last_update:
                    time_since_update = (current_time - last_update).total_seconds()
                    
                    if time_since_update > 30:  # No updates for 30+ seconds
                        logging.warning(f"No market data updates for {time_since_update:.0f} seconds")
                        self.data_quality_metrics['connection_status'] = 'STALE'
                    else:
                        self.data_quality_metrics['connection_status'] = 'CONNECTED'
                
                # Log quality metrics
                metrics = self.data_quality_metrics
                logging.info(f"Data Quality - Ticks: {metrics['tick_count']}, "
                           f"OHLC: {metrics['ohlc_count']}, L1: {metrics['l1_count']}, "
                           f"Errors: {metrics['error_count']}, Status: {metrics['connection_status']}")
                
            except Exception as e:
                logging.error(f"Error in data quality monitor: {e}")
    
    def _get_symbol_data_quality(self, symbol: str) -> Dict[str, Any]:
        """Get data quality metrics for a specific symbol"""
        try:
            tick_count = len(self.tick_buffers.get(symbol, []))
            ohlc_count = sum(len(self.ohlc_buffers.get(symbol, {}).get(tf, [])) 
                           for tf in ['1min', '5min', '15min'])
            l1_count = len(self.l1_buffers.get(symbol, []))
            
            latest_tick = self.get_latest_tick(symbol)
            data_age = 0
            if latest_tick:
                data_age = (datetime.now() - latest_tick.timestamp).total_seconds()
            
            return {
                'tick_buffer_size': tick_count,
                'ohlc_buffer_size': ohlc_count,
                'l1_buffer_size': l1_count,
                'data_age_seconds': data_age,
                'is_subscribed': symbol in self.subscribed_symbols
            }
            
        except Exception as e:
            logging.error(f"Error getting data quality for {symbol}: {e}")
            return {}
    
    def add_tick_subscriber(self, callback: Callable):
        """Add callback for tick data updates"""
        self.tick_subscribers.append(callback)
    
    def add_ohlc_subscriber(self, callback: Callable):
        """Add callback for OHLC data updates"""
        self.ohlc_subscribers.append(callback)
    
    def add_l1_subscriber(self, callback: Callable):
        """Add callback for L1 data updates"""
        self.l1_subscribers.append(callback)
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Unsubscribe from all symbols
            for symbol in list(self.subscribed_symbols):
                await self.unsubscribe_symbol(symbol)
            
            # Close session
            if self.session:
                await self.session.close()
            
            logging.info("Market Data Manager cleaned up successfully")
            
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

# Global instance
data_manager = MarketDataManager()

def get_data_manager() -> MarketDataManager:
    """Get the global data manager instance"""
    return data_manager
