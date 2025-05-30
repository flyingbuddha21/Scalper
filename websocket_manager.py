#!/usr/bin/env python3
"""
WebSocket Manager for Real-time Market Data
Handles multiple WebSocket connections for live data feeds
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import ssl
import aiohttp
from urllib.parse import urlencode
import time
import threading
from collections import defaultdict

# Import system components
from config_manager import ConfigManager
from data_manager import DataManager
from database_setup import TradingDatabase
from security_manager import SecurityManager
from goodwill_api_handler import GoodwillAPIHandler, get_goodwill_handler
from utils import Logger, ErrorHandler, DataValidator
from bot_core import TradingBot

class ConnectionStatus(Enum):
    """WebSocket connection status"""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    ERROR = "ERROR"

class DataFeedType(Enum):
    """Types of data feeds"""
    LIVE_QUOTES = "LIVE_QUOTES"
    ORDER_UPDATES = "ORDER_UPDATES"
    MARKET_DEPTH = "MARKET_DEPTH"
    TRADE_UPDATES = "TRADE_UPDATES"
    NEWS_FEED = "NEWS_FEED"
    TECHNICAL_INDICATORS = "TECHNICAL_INDICATORS"

@dataclass
class WebSocketConfig:
    """WebSocket connection configuration"""
    url: str
    name: str
    feed_type: DataFeedType
    symbols: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    auth_required: bool = False
    reconnect_interval: int = 5
    max_reconnect_attempts: int = 10
    heartbeat_interval: int = 30
    message_timeout: int = 10

@dataclass
class MarketTick:
    """Real-time market tick data"""
    symbol: str
    ltp: float
    volume: int
    bid: float
    ask: float
    bid_qty: int
    ask_qty: int
    timestamp: datetime
    change: float = 0.0
    change_percent: float = 0.0

@dataclass
class OrderUpdate:
    """Order status update"""
    order_id: str
    symbol: str
    status: str
    executed_qty: int
    executed_price: float
    timestamp: datetime
    message: str = ""

class WebSocketConnection:
    """Individual WebSocket connection handler"""
    
    def __init__(self, config: WebSocketConfig, callback: Callable, 
                 security_manager: SecurityManager = None):
        self.config = config
        self.callback = callback
        self.security_manager = security_manager
        
        # Connection state
        self.websocket = None
        self.status = ConnectionStatus.DISCONNECTED
        self.reconnect_attempts = 0
        self.last_message_time = None
        self.is_running = False
        
        # Initialize logger
        self.logger = Logger(f"WebSocket-{config.name}")
        self.error_handler = ErrorHandler()
        
        # Message handlers
        self.message_handlers = {
            DataFeedType.LIVE_QUOTES: self._handle_quote_message,
            DataFeedType.ORDER_UPDATES: self._handle_order_message,
            DataFeedType.MARKET_DEPTH: self._handle_depth_message,
            DataFeedType.TRADE_UPDATES: self._handle_trade_message,
            DataFeedType.NEWS_FEED: self._handle_news_message
        }
    
    async def connect(self):
        """Connect to WebSocket"""
        try:
            self.status = ConnectionStatus.CONNECTING
            self.logger.info(f"Connecting to {self.config.url}")
            
            # Setup SSL context if needed
            ssl_context = None
            if self.config.url.startswith('wss://'):
                ssl_context = ssl.create_default_context()
            
            # Add authentication headers if required
            headers = self.config.headers.copy()
            if self.config.auth_required and self.security_manager:
                auth_token = await self._get_auth_token()
                if auth_token:
                    headers['Authorization'] = f'Bearer {auth_token}'
            
            # Connect to WebSocket
            self.websocket = await websockets.connect(
                self.config.url,
                extra_headers=headers,
                ssl=ssl_context,
                ping_interval=self.config.heartbeat_interval,
                ping_timeout=self.config.message_timeout,
                close_timeout=10
            )
            
            self.status = ConnectionStatus.CONNECTED
            self.reconnect_attempts = 0
            self.last_message_time = datetime.now()
            
            self.logger.info(f"Connected to {self.config.name}")
            
            # Send subscription message
            await self._subscribe_to_feeds()
            
            return True
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.logger.error(f"Connection failed: {e}")
            self.error_handler.handle_error(e, f"websocket_connect_{self.config.name}")
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        try:
            self.is_running = False
            
            if self.websocket and not self.websocket.closed:
                await self.websocket.close()
            
            self.status = ConnectionStatus.DISCONNECTED
            self.websocket = None
            
            self.logger.info(f"Disconnected from {self.config.name}")
            
        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")
    
    async def start_listening(self):
        """Start listening for messages"""
        self.is_running = True
        
        while self.is_running:
            try:
                if self.status != ConnectionStatus.CONNECTED:
                    if not await self._reconnect():
                        await asyncio.sleep(self.config.reconnect_interval)
                        continue
                
                # Listen for messages
                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=self.config.message_timeout
                    )
                    
                    await self._process_message(message)
                    self.last_message_time = datetime.now()
                    
                except asyncio.TimeoutError:
                    # Check for heartbeat timeout
                    if self._is_heartbeat_timeout():
                        self.logger.warning("Heartbeat timeout detected")
                        await self._reconnect()
                    continue
                    
                except websockets.exceptions.ConnectionClosed:
                    self.logger.warning("Connection closed by server")
                    self.status = ConnectionStatus.DISCONNECTED
                    await self._reconnect()
                    continue
                
            except Exception as e:
                self.logger.error(f"Listening error: {e}")
                self.error_handler.handle_error(e, f"websocket_listen_{self.config.name}")
                await asyncio.sleep(1)
    
    async def _reconnect(self):
        """Attempt to reconnect"""
        if self.reconnect_attempts >= self.config.max_reconnect_attempts:
            self.logger.error("Max reconnection attempts reached")
            self.status = ConnectionStatus.ERROR
            return False
        
        self.status = ConnectionStatus.RECONNECTING
        self.reconnect_attempts += 1
        
        self.logger.info(f"Reconnecting... (attempt {self.reconnect_attempts})")
        
        # Close existing connection
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
        
        await asyncio.sleep(self.config.reconnect_interval)
        return await self.connect()
    
    async def _get_auth_token(self) -> Optional[str]:
        """Get authentication token from security manager"""
        try:
            if self.security_manager:
                # This would typically be implemented based on the specific API
                # For now, return a placeholder
                return "auth_token_placeholder"
            return None
        except Exception as e:
            self.logger.error(f"Failed to get auth token: {e}")
            return None
    
    async def _subscribe_to_feeds(self):
        """Subscribe to data feeds"""
        try:
            if self.config.feed_type == DataFeedType.LIVE_QUOTES:
                subscription = {
                    "action": "subscribe",
                    "feeds": ["live_quotes"],
                    "symbols": self.config.symbols
                }
            elif self.config.feed_type == DataFeedType.ORDER_UPDATES:
                subscription = {
                    "action": "subscribe",
                    "feeds": ["order_updates"]
                }
            elif self.config.feed_type == DataFeedType.MARKET_DEPTH:
                subscription = {
                    "action": "subscribe",
                    "feeds": ["market_depth"],
                    "symbols": self.config.symbols,
                    "depth": 5
                }
            else:
                subscription = {
                    "action": "subscribe",
                    "feeds": [self.config.feed_type.value.lower()]
                }
            
            await self.websocket.send(json.dumps(subscription))
            self.logger.info(f"Subscribed to {self.config.feed_type.value}")
            
        except Exception as e:
            self.logger.error(f"Subscription failed: {e}")
    
    async def _process_message(self, message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            # Handle different message types
            handler = self.message_handlers.get(self.config.feed_type)
            if handler:
                processed_data = await handler(data)
                if processed_data:
                    await self.callback(self.config.feed_type, processed_data)
            else:
                # Generic message handling
                await self.callback(self.config.feed_type, data)
                
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON message: {message[:100]}")
        except Exception as e:
            self.logger.error(f"Message processing error: {e}")
            self.error_handler.handle_error(e, f"websocket_message_{self.config.name}")
    
    async def _handle_quote_message(self, data: Dict) -> Optional[MarketTick]:
        """Handle live quote message"""
        try:
            if 'symbol' in data and 'ltp' in data:
                return MarketTick(
                    symbol=data['symbol'],
                    ltp=float(data['ltp']),
                    volume=int(data.get('volume', 0)),
                    bid=float(data.get('bid', 0)),
                    ask=float(data.get('ask', 0)),
                    bid_qty=int(data.get('bid_qty', 0)),
                    ask_qty=int(data.get('ask_qty', 0)),
                    timestamp=datetime.now(),
                    change=float(data.get('change', 0)),
                    change_percent=float(data.get('change_percent', 0))
                )
            return None
        except Exception as e:
            self.logger.error(f"Quote message parsing error: {e}")
            return None
    
    async def _handle_order_message(self, data: Dict) -> Optional[OrderUpdate]:
        """Handle order update message"""
        try:
            if 'order_id' in data and 'status' in data:
                return OrderUpdate(
                    order_id=data['order_id'],
                    symbol=data.get('symbol', ''),
                    status=data['status'],
                    executed_qty=int(data.get('executed_qty', 0)),
                    executed_price=float(data.get('executed_price', 0)),
                    timestamp=datetime.now(),
                    message=data.get('message', '')
                )
            return None
        except Exception as e:
            self.logger.error(f"Order message parsing error: {e}")
            return None
    
    async def _handle_depth_message(self, data: Dict) -> Optional[Dict]:
        """Handle market depth message"""
        try:
            if 'symbol' in data and 'bids' in data and 'asks' in data:
                return {
                    'symbol': data['symbol'],
                    'bids': data['bids'],
                    'asks': data['asks'],
                    'timestamp': datetime.now()
                }
            return None
        except Exception as e:
            self.logger.error(f"Depth message parsing error: {e}")
            return None
    
    async def _handle_trade_message(self, data: Dict) -> Optional[Dict]:
        """Handle trade update message"""
        try:
            return {
                'trade_id': data.get('trade_id'),
                'symbol': data.get('symbol'),
                'price': float(data.get('price', 0)),
                'quantity': int(data.get('quantity', 0)),
                'timestamp': datetime.now(),
                'raw_data': data
            }
        except Exception as e:
            self.logger.error(f"Trade message parsing error: {e}")
            return None
    
    async def _handle_news_message(self, data: Dict) -> Optional[Dict]:
        """Handle news feed message"""
        try:
            return {
                'headline': data.get('headline'),
                'content': data.get('content'),
                'symbols': data.get('symbols', []),
                'timestamp': datetime.now(),
                'source': data.get('source')
            }
        except Exception as e:
            self.logger.error(f"News message parsing error: {e}")
            return None
    
    def _is_heartbeat_timeout(self) -> bool:
        """Check if heartbeat timeout occurred"""
        if not self.last_message_time:
            return False
        
        timeout_threshold = timedelta(seconds=self.config.heartbeat_interval * 2)
        return datetime.now() - self.last_message_time > timeout_threshold

class WebSocketManager:
    """Main WebSocket manager for handling multiple connections including Goodwill feeds"""
    
    def __init__(self, config_manager: ConfigManager, data_manager: DataManager,
                 trading_db: TradingDatabase, security_manager: SecurityManager):
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.trading_db = trading_db
        self.security_manager = security_manager
        
        # Get Goodwill API handler instance
        self.goodwill_handler = get_goodwill_handler()
        
        # Initialize logger and error handler
        self.logger = Logger(__name__)
        self.error_handler = ErrorHandler()
        self.data_validator = DataValidator()
        
        # WebSocket connections
        self.connections: Dict[str, WebSocketConnection] = {}
        self.goodwill_websocket = None
        self.is_running = False
        
        # Data handlers
        self.data_handlers = {
            DataFeedType.LIVE_QUOTES: self._handle_live_quotes,
            DataFeedType.ORDER_UPDATES: self._handle_order_updates,
            DataFeedType.MARKET_DEPTH: self._handle_market_depth,
            DataFeedType.TRADE_UPDATES: self._handle_trade_updates,
            DataFeedType.NEWS_FEED: self._handle_news_feed
        }
        
        # Statistics
        self.stats = {
            'messages_received': defaultdict(int),
            'last_message_time': defaultdict(lambda: None),
            'connection_uptime': defaultdict(lambda: None),
            'goodwill_connection_status': 'DISCONNECTED'
        }
    
    async def initialize(self):
        """Initialize WebSocket manager with Goodwill integration"""
        try:
            config = self.config_manager.get_config()
            websocket_config = config.get('websockets', {})
            
            # Setup Goodwill WebSocket connection if authenticated
            if self.goodwill_handler.is_authenticated:
                await self._setup_goodwill_websocket()
            
            # Create other WebSocket connections from config
            for conn_name, conn_config in websocket_config.items():
                if conn_name == 'goodwill':  # Skip, handled separately
                    continue
                    
                ws_config = WebSocketConfig(
                    url=conn_config['url'],
                    name=conn_name,
                    feed_type=DataFeedType(conn_config['feed_type']),
                    symbols=conn_config.get('symbols', []),
                    headers=conn_config.get('headers', {}),
                    auth_required=conn_config.get('auth_required', False),
                    reconnect_interval=conn_config.get('reconnect_interval', 5),
                    max_reconnect_attempts=conn_config.get('max_reconnect_attempts', 10),
                    heartbeat_interval=conn_config.get('heartbeat_interval', 30)
                )
                
                connection = WebSocketConnection(
                    ws_config, 
                    self._message_callback,
                    self.security_manager if ws_config.auth_required else None
                )
                
                self.connections[conn_name] = connection
            
            self.logger.info(f"WebSocket manager initialized with {len(self.connections)} connections + Goodwill feed")
            
        except Exception as e:
            self.logger.error(f"WebSocket manager initialization failed: {e}")
            self.error_handler.handle_error(e, "websocket_manager_init")
            raise
    
    async def _setup_goodwill_websocket(self):
        """Setup Goodwill WebSocket connection"""
        try:
            # Use the correct Goodwill WebSocket URL from documentation
            goodwill_ws_url = "wss://giga.gwcindia.in/NorenWSTP/"
            
            # Create Goodwill-specific WebSocket config
            goodwill_config = WebSocketConfig(
                url=goodwill_ws_url,
                name="goodwill_feeds",
                feed_type=DataFeedType.LIVE_QUOTES,
                symbols=[],  # Will be populated based on subscriptions
                auth_required=True,
                reconnect_interval=5,
                max_reconnect_attempts=10,
                heartbeat_interval=30
            )
            
            # Create Goodwill WebSocket connection
            self.goodwill_websocket = WebSocketConnection(
                goodwill_config,
                self._goodwill_message_callback,
                self.security_manager
            )
            
            self.connections['goodwill'] = self.goodwill_websocket
            self.stats['goodwill_connection_status'] = 'INITIALIZED'
            
            self.logger.info("Goodwill WebSocket connection setup completed with URL: wss://giga.gwcindia.in/NorenWSTP/")
            
        except Exception as e:
            self.logger.error(f"Goodwill WebSocket setup failed: {e}")
            self.stats['goodwill_connection_status'] = 'ERROR'
    
    async def start_all_connections(self):
        """Start all WebSocket connections"""
        try:
            self.is_running = True
            
            # Start all connections concurrently
            tasks = []
            for name, connection in self.connections.items():
                task = asyncio.create_task(self._start_connection(name, connection))
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket connections: {e}")
            raise
    
    async def _start_connection(self, name: str, connection: WebSocketConnection):
        """Start individual WebSocket connection"""
        try:
            self.stats['connection_uptime'][name] = datetime.now()
            
            if await connection.connect():
                await connection.start_listening()
            else:
                self.logger.error(f"Failed to start connection: {name}")
                
        except Exception as e:
            self.logger.error(f"Connection {name} error: {e}")
            self.error_handler.handle_error(e, f"websocket_connection_{name}")
    
    async def stop_all_connections(self):
        """Stop all WebSocket connections"""
        try:
            self.is_running = False
            
            # Disconnect all connections
            for connection in self.connections.values():
                await connection.disconnect()
            
            self.logger.info("All WebSocket connections stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping connections: {e}")
    
    async def _message_callback(self, feed_type: DataFeedType, data: Any):
        """Handle incoming WebSocket messages"""
        try:
            # Update statistics
            self.stats['messages_received'][feed_type] += 1
            self.stats['last_message_time'][feed_type] = datetime.now()
            
            # Route to appropriate handler
            handler = self.data_handlers.get(feed_type)
            if handler:
                await handler(data)
            else:
                self.logger.warning(f"No handler for feed type: {feed_type}")
                
        except Exception as e:
            self.logger.error(f"Message callback error: {e}")
            self.error_handler.handle_error(e, f"websocket_callback_{feed_type}")
    
    async def _handle_live_quotes(self, tick: MarketTick):
        """Handle live quote data"""
        try:
            # Validate data
            if not self.data_validator.validate_price(tick.ltp):
                self.logger.warning(f"Invalid price for {tick.symbol}: {tick.ltp}")
                return
            
            # Update data manager
            await self.data_manager.update_live_price(tick.symbol, tick.ltp, tick.volume)
            
            # Update database cache
            await self.trading_db.update_realtime_quote(
                self.trading_db.MarketData(
                    symbol=tick.symbol,
                    timestamp=tick.timestamp,
                    open_price=tick.ltp,  # Simplified
                    high_price=tick.ltp,
                    low_price=tick.ltp,
                    close_price=tick.ltp,
                    volume=tick.volume,
                    ltp=tick.ltp
                )
            )
            
            # Log for monitoring
            self.logger.debug(f"Quote update: {tick.symbol} @ {tick.ltp}")
            
        except Exception as e:
            self.logger.error(f"Live quotes handling error: {e}")
    
    async def _handle_order_updates(self, order_update: OrderUpdate):
        """Handle order status updates"""
        try:
            # Update database
            await self.trading_db.update_order_status(
                order_update.order_id,
                order_update.status,
                order_update.executed_price if order_update.executed_price > 0 else None,
                order_update.executed_qty if order_update.executed_qty > 0 else None
            )
            
            # Log order update
            self.logger.info(f"Order update: {order_update.order_id} - {order_update.status}")
            
            # Notify trading bot if order is filled
            if order_update.status == 'FILLED':
                # This would trigger portfolio updates, etc.
                pass
                
        except Exception as e:
            self.logger.error(f"Order updates handling error: {e}")
    
    async def _handle_market_depth(self, depth_data: Dict):
        """Handle market depth data"""
        try:
            symbol = depth_data['symbol']
            
            # Store market depth for analysis
            # This could be used for advanced order execution strategies
            self.logger.debug(f"Market depth update: {symbol}")
            
        except Exception as e:
            self.logger.error(f"Market depth handling error: {e}")
    
    async def _handle_trade_updates(self, trade_data: Dict):
        """Handle trade execution updates"""
        try:
            # Log trade execution
            self.logger.info(f"Trade executed: {trade_data}")
            
            # Update portfolio if it's our trade
            # This would be matched against our orders
            
        except Exception as e:
            self.logger.error(f"Trade updates handling error: {e}")
    
    async def _goodwill_message_callback(self, feed_type: DataFeedType, data: Any):
        """Handle Goodwill-specific WebSocket messages"""
        try:
            # Update statistics
            self.stats['messages_received'][feed_type] += 1
            self.stats['last_message_time'][feed_type] = datetime.now()
            self.stats['goodwill_connection_status'] = 'CONNECTED'
            
            # Parse Goodwill message format
            if isinstance(data, dict):
                message_type = data.get('t', '')
                
                if message_type == 'tf':  # Trade feed
                    await self._handle_goodwill_quote(data)
                elif message_type == 'df':  # Depth feed
                    await self._handle_goodwill_depth(data)
                elif message_type == 'om':  # Order update
                    await self._handle_goodwill_order_update(data)
                else:
                    self.logger.debug(f"Unknown Goodwill message type: {message_type}")
            
        except Exception as e:
            self.logger.error(f"Goodwill message callback error: {e}")
            self.error_handler.handle_error(e, "goodwill_websocket_callback")
    
    async def _handle_goodwill_quote(self, data: Dict):
        """Handle Goodwill quote/trade feed"""
        try:
            # Parse Goodwill quote format
            symbol = data.get('tk', '')  # Token
            ltp = float(data.get('lp', 0))  # Last traded price
            volume = int(data.get('v', 0))  # Volume
            
            # Create MarketTick object
            tick = MarketTick(
                symbol=symbol,
                ltp=ltp,
                volume=volume,
                bid=float(data.get('bp1', 0)),
                ask=float(data.get('sp1', 0)),
                bid_qty=int(data.get('bq1', 0)),
                ask_qty=int(data.get('sq1', 0)),
                timestamp=datetime.now(),
                change=float(data.get('c', 0)),
                change_percent=float(data.get('pc', 0))
            )
            
            # Process through main handler
            await self._handle_live_quotes(tick)
            
        except Exception as e:
            self.logger.error(f"Goodwill quote handling error: {e}")
    
    async def _handle_goodwill_depth(self, data: Dict):
        """Handle Goodwill market depth feed"""
        try:
            symbol = data.get('tk', '')
            
            # Parse bid/ask arrays
            bids = []
            asks = []
            
            for i in range(1, 6):  # Top 5 levels
                bid_price = data.get(f'bp{i}')
                bid_qty = data.get(f'bq{i}')
                ask_price = data.get(f'sp{i}')
                ask_qty = data.get(f'sq{i}')
                
                if bid_price and bid_qty:
                    bids.append([float(bid_price), int(bid_qty)])
                if ask_price and ask_qty:
                    asks.append([float(ask_price), int(ask_qty)])
            
            depth_data = {
                'symbol': symbol,
                'bids': bids,
                'asks': asks,
                'timestamp': datetime.now()
            }
            
            await self._handle_market_depth(depth_data)
            
        except Exception as e:
            self.logger.error(f"Goodwill depth handling error: {e}")
    
    async def _handle_goodwill_order_update(self, data: Dict):
        """Handle Goodwill order update messages"""
        try:
            order_update = OrderUpdate(
                order_id=data.get('nstordno', ''),
                symbol=data.get('tsym', ''),
                status=data.get('st', ''),
                executed_qty=int(data.get('fillqty', 0)),
                executed_price=float(data.get('flprc', 0)),
                timestamp=datetime.now(),
                message=data.get('emsg', '')
            )
            
            await self._handle_order_updates(order_update)
            
        except Exception as e:
            self.logger.error(f"Goodwill order update handling error: {e}")
    
    async def subscribe_to_goodwill_symbols(self, symbols: List[str]):
        """Subscribe to Goodwill symbols for live data"""
        try:
            if not self.goodwill_handler.is_authenticated:
                self.logger.error("Goodwill not authenticated, cannot subscribe to symbols")
                return False
            
            # Get symbol tokens from Goodwill API
            symbol_tokens = []
            for symbol in symbols:
                search_results = await self.goodwill_handler.search_symbols(symbol)
                if search_results:
                    # Get the first matching result and format as per Goodwill docs
                    result = search_results[0]
                    exchange = result.get('exchange', 'NSE')
                    token = result.get('token', '')
                    if token:
                        # Format: "EXCHANGE|TOKEN" as per documentation
                        symbol_tokens.append(f"{exchange}|{token}")
            
            # Send subscription message to Goodwill WebSocket
            if self.goodwill_websocket and symbol_tokens:
                # Subscribe to touchline data as per documentation
                subscription_message = {
                    "t": "t",  # Touchline subscription
                    "k": "#".join(symbol_tokens)  # Join tokens with #
                }
                
                if self.goodwill_websocket.websocket:
                    await self.goodwill_websocket.websocket.send(json.dumps(subscription_message))
                    self.logger.info(f"Subscribed to Goodwill symbols: {symbols} -> {symbol_tokens}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Goodwill symbol subscription error: {e}")
            return False
    
    async def _handle_news_feed(self, news_data: Dict):
        """Handle news feed data"""
        try:
            # Store news for sentiment analysis
            self.logger.info(f"News update: {news_data.get('headline', '')[:50]}...")
            
            # Could trigger strategy adjustments based on news
            
        except Exception as e:
            self.logger.error(f"News feed handling error: {e}")
    
    def get_connection_status(self) -> Dict:
        """Get status of all WebSocket connections"""
        status = {}
        
        for name, connection in self.connections.items():
            status[name] = {
                'status': connection.status.value,
                'reconnect_attempts': connection.reconnect_attempts,
                'last_message_time': self.stats['last_message_time'][connection.config.feed_type],
                'messages_received': self.stats['messages_received'][connection.config.feed_type],
                'uptime': datetime.now() - self.stats['connection_uptime'][name] if self.stats['connection_uptime'][name] else None,
                'feed_type': connection.config.feed_type.value
            }
        
        return status
    
    def get_statistics(self) -> Dict:
        """Get WebSocket statistics"""
        return {
            'total_connections': len(self.connections),
            'active_connections': sum(1 for conn in self.connections.values() 
                                    if conn.status == ConnectionStatus.CONNECTED),
            'total_messages': sum(self.stats['messages_received'].values()),
            'messages_by_feed': dict(self.stats['messages_received']),
            'last_activity': max(self.stats['last_message_time'].values()) if self.stats['last_message_time'] else None
        }
    
    async def add_symbols(self, connection_name: str, symbols: List[str]):
        """Add symbols to WebSocket subscription"""
        try:
            if connection_name in self.connections:
                connection = self.connections[connection_name]
                connection.config.symbols.extend(symbols)
                
                # Re-subscribe if connected
                if connection.status == ConnectionStatus.CONNECTED:
                    await connection._subscribe_to_feeds()
                
                self.logger.info(f"Added symbols to {connection_name}: {symbols}")
            else:
                self.logger.error(f"Connection not found: {connection_name}")
                
        except Exception as e:
            self.logger.error(f"Error adding symbols: {e}")
    
    async def remove_symbols(self, connection_name: str, symbols: List[str]):
        """Remove symbols from WebSocket subscription"""
        try:
            if connection_name in self.connections:
                connection = self.connections[connection_name]
                for symbol in symbols:
                    if symbol in connection.config.symbols:
                        connection.config.symbols.remove(symbol)
                
                # Re-subscribe if connected
                if connection.status == ConnectionStatus.CONNECTED:
                    await connection._subscribe_to_feeds()
                
                self.logger.info(f"Removed symbols from {connection_name}: {symbols}")
            else:
                self.logger.error(f"Connection not found: {connection_name}")
                
        except Exception as e:
            self.logger.error(f"Error removing symbols: {e}")

# Example usage and testing
async def main():
    """Example usage of WebSocketManager"""
    from config_manager import ConfigManager
    from data_manager import DataManager
    from database_setup import TradingDatabase
    from security_manager import SecurityManager
    
    # Initialize components
    config_manager = ConfigManager("config/config.yaml")
    trading_db = TradingDatabase(config_manager)
    await trading_db.initialize()
    
    data_manager = DataManager(config_manager, trading_db)
    await data_manager.initialize()
    
    security_manager = SecurityManager(trading_db, config_manager)
    await security_manager.initialize()
    
    # Initialize WebSocket manager
    ws_manager = WebSocketManager(config_manager, data_manager, trading_db, security_manager)
    await ws_manager.initialize()
    
    try:
        # Start WebSocket connections
        print("Starting WebSocket connections...")
        
        # This would run indefinitely in production
        task = asyncio.create_task(ws_manager.start_all_connections())
        
        # Let it run for a bit to test
        await asyncio.sleep(30)
        
        # Check status
        status = ws_manager.get_connection_status()
        print(f"Connection Status: {json.dumps(status, indent=2, default=str)}")
        
        stats = ws_manager.get_statistics()
        print(f"Statistics: {json.dumps(stats, indent=2, default=str)}")
        
    except KeyboardInterrupt:
        print("Stopping WebSocket connections...")
    finally:
        await ws_manager.stop_all_connections()
        await trading_db.close()

if __name__ == "__main__":
    asyncio.run(main())
