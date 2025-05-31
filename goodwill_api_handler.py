#!/usr/bin/env python3
"""
Enhanced Production-Ready Goodwill API Handler with Login Flow & Auto-Refresh
Complete implementation with web-based authentication, token management, and real-time data
Based on real Goodwill API documentation: https://developer.gwcindia.in/api/
"""

import asyncio
import aiohttp
import json
import hashlib
import hmac
import time
import logging
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import urllib.parse
from pathlib import Path
import webbrowser
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# Import system components
from config_manager import ConfigManager, get_config
from utils import Logger, ErrorHandler, DataValidator
from security_manager import SecurityManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order type enumeration for Goodwill API"""
    MARKET = 1
    LIMIT = 2
    STOP_LOSS = 3
    STOP_LOSS_MARKET = 4

class OrderSide(Enum):
    """Order side enumeration"""
    BUY = 1
    SELL = -1

class ProductType(Enum):
    """Product type enumeration for Goodwill API"""
    CASH = "CNC"        # Cash & Carry
    INTRADAY = "INTRADAY"  # Intraday
    MARGIN = "MARGIN"   # Margin
    COVER = "CO"        # Cover Order
    BRACKET = "BO"      # Bracket Order

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    PARTIAL = "PARTIAL"

class AuthenticationState(Enum):
    """Authentication state enumeration"""
    NOT_AUTHENTICATED = "NOT_AUTHENTICATED"
    AUTHENTICATION_PENDING = "AUTHENTICATION_PENDING"
    AUTHENTICATED = "AUTHENTICATED"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"

@dataclass
class Quote:
    """Market quote data"""
    symbol: str
    exchange: str
    token: str
    last_price: float
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    change: float
    change_percent: float
    bid_price: float
    ask_price: float
    bid_qty: int
    ask_qty: int
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'token': self.token,
            'last_price': self.last_price,
            'open_price': self.open_price,
            'high_price': self.high_price,
            'low_price': self.low_price,
            'close_price': self.close_price,
            'volume': self.volume,
            'change': self.change,
            'change_percent': self.change_percent,
            'bid_price': self.bid_price,
            'ask_price': self.ask_price,
            'bid_qty': self.bid_qty,
            'ask_qty': self.ask_qty,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class Order:
    """Order data structure"""
    order_id: str
    symbol: str
    exchange: str
    side: OrderSide
    order_type: OrderType
    product_type: ProductType
    quantity: int
    price: float
    disclosed_qty: int = 0
    trigger_price: float = 0.0
    validity: str = "DAY"
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    pending_qty: int = 0
    average_price: float = 0.0
    order_time: Optional[datetime] = None
    exchange_order_id: str = ""
    rejection_reason: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'product_type': self.product_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'disclosed_qty': self.disclosed_qty,
            'trigger_price': self.trigger_price,
            'validity': self.validity,
            'status': self.status.value,
            'filled_qty': self.filled_qty,
            'pending_qty': self.pending_qty,
            'average_price': self.average_price,
            'order_time': self.order_time.isoformat() if self.order_time else None,
            'exchange_order_id': self.exchange_order_id,
            'rejection_reason': self.rejection_reason
        }

@dataclass
class Position:
    """Position data structure"""
    symbol: str
    exchange: str
    product_type: ProductType
    quantity: int
    average_price: float
    last_price: float
    unrealized_pnl: float
    realized_pnl: float
    day_pnl: float
    day_pnl_percent: float
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'product_type': self.product_type.value,
            'quantity': self.quantity,
            'average_price': self.average_price,
            'last_price': self.last_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'day_pnl': self.day_pnl,
            'day_pnl_percent': self.day_pnl_percent
        }

@dataclass
class AuthenticationResult:
    """Authentication result"""
    success: bool
    access_token: str = ""
    refresh_token: str = ""
    user_session: str = ""
    user_id: str = ""
    user_name: str = ""
    broker_name: str = ""
    email: str = ""
    mobile: str = ""
    exchanges: List[str] = field(default_factory=list)
    products: List[str] = field(default_factory=list)
    error_message: str = ""
    expires_at: Optional[datetime] = None
    refresh_expires_at: Optional[datetime] = None

class TokenManager:
    """Manages access and refresh tokens with automatic renewal"""
    
    def __init__(self, goodwill_handler):
        self.goodwill_handler = goodwill_handler
        self.access_token = ""
        self.refresh_token = ""
        self.expires_at = None
        self.refresh_expires_at = None
        self.auto_refresh_task = None
        self.refresh_callbacks: List[Callable] = []
        
    def set_tokens(self, access_token: str, refresh_token: str = "", 
                   expires_in: int = 86400, refresh_expires_in: int = 604800):
        """Set access and refresh tokens"""
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.expires_at = datetime.now() + timedelta(seconds=expires_in)
        
        if refresh_token:
            self.refresh_expires_at = datetime.now() + timedelta(seconds=refresh_expires_in)
        
        # Start auto-refresh if we have a refresh token
        if refresh_token and not self.auto_refresh_task:
            self.start_auto_refresh()
    
    def is_token_valid(self) -> bool:
        """Check if access token is still valid"""
        if not self.access_token or not self.expires_at:
            return False
        
        # Consider token expired 5 minutes before actual expiry
        buffer_time = timedelta(minutes=5)
        return datetime.now() < (self.expires_at - buffer_time)
    
    def is_refresh_token_valid(self) -> bool:
        """Check if refresh token is still valid"""
        if not self.refresh_token or not self.refresh_expires_at:
            return False
        
        return datetime.now() < self.refresh_expires_at
    
    async def refresh_access_token(self) -> bool:
        """Refresh the access token using refresh token (Goodwill specific)"""
        try:
            if not self.is_refresh_token_valid():
                logger.warning("Refresh token expired, full re-authentication required")
                return False
            
            # Goodwill API refresh token endpoint
            refresh_payload = {
                'fyers_id': self.goodwill_handler.api_key,
                'refresh_token': self.refresh_token
            }
            
            response = await self.goodwill_handler._make_request(
                'POST',
                '/login/refresh_accesstoken/',
                data=refresh_payload,
                authenticated=False
            )
            
            if response and response.get('s') == 'ok':
                # Update tokens
                data = response.get('data', {})
                new_access_token = data.get('access_token', '')
                new_refresh_token = data.get('refresh_token', self.refresh_token)
                expires_in = 86400  # 24 hours for Goodwill
                
                self.set_tokens(new_access_token, new_refresh_token, expires_in)
                
                # Notify callbacks
                for callback in self.refresh_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(new_access_token)
                        else:
                            callback(new_access_token)
                    except Exception as e:
                        logger.error(f"Token refresh callback error: {e}")
                
                logger.info("Access token refreshed successfully")
                return True
            else:
                logger.error(f"Token refresh failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return False
    
    def start_auto_refresh(self):
        """Start automatic token refresh task"""
        if self.auto_refresh_task:
            self.auto_refresh_task.cancel()
        
        self.auto_refresh_task = asyncio.create_task(self._auto_refresh_loop())
    
    async def _auto_refresh_loop(self):
        """Auto-refresh loop that runs in background"""
        try:
            while True:
                if not self.expires_at:
                    break
                
                # Calculate when to refresh (10 minutes before expiry)
                refresh_time = self.expires_at - timedelta(minutes=10)
                current_time = datetime.now()
                
                if current_time >= refresh_time:
                    # Time to refresh
                    success = await self.refresh_access_token()
                    if not success:
                        logger.error("Auto token refresh failed")
                        break
                else:
                    # Wait until refresh time
                    wait_seconds = (refresh_time - current_time).total_seconds()
                    await asyncio.sleep(min(wait_seconds, 300))  # Max 5 minutes sleep
                
        except asyncio.CancelledError:
            logger.info("Auto refresh task cancelled")
        except Exception as e:
            logger.error(f"Auto refresh loop error: {e}")
    
    def add_refresh_callback(self, callback: Callable):
        """Add callback to be called when token is refreshed"""
        self.refresh_callbacks.append(callback)
    
    def clear_tokens(self):
        """Clear all tokens and stop auto-refresh"""
        self.access_token = ""
        self.refresh_token = ""
        self.expires_at = None
        self.refresh_expires_at = None
        
        if self.auto_refresh_task:
            self.auto_refresh_task.cancel()
            self.auto_refresh_task = None

class GoodwillAPIHandler:
    """Enhanced Production-ready Goodwill API handler with login flow and auto-refresh"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        # Initialize configuration
        self.config_manager = config_manager or get_config()
        self.config = self.config_manager.get_config()
        self.api_config = self.config.get('goodwill_api', {})
        
        # Initialize utilities
        self.logger = Logger(__name__)
        self.error_handler = ErrorHandler()
        self.data_validator = DataValidator()
        
        # API configuration - Using real Goodwill API
        self.base_url = self.api_config.get('base_url', 'https://api.gwcindia.in/v1')
        self.api_key = self.api_config.get('api_key', '')
        self.api_secret = self.api_config.get('api_secret', '')
        self.user_id = self.api_config.get('user_id', '')
        self.redirect_url = self.api_config.get('redirect_url', 'https://ant.aliceblueonline.com')
        
        # Authentication state management
        self.auth_state = AuthenticationState.NOT_AUTHENTICATED
        self.token_manager = TokenManager(self)
        self.user_info = {}
        
        # HTTP session
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=30)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        self.request_count = 0
        self.rate_limit_window = 60  # 1 minute
        self.max_requests_per_minute = 300
        
        # Cache for frequently accessed data
        self.symbol_cache = {}
        self.exchange_cache = {}
        self.last_cache_update = 0
        self.cache_ttl = 300  # 5 minutes
        
        # Order tracking
        self.active_orders = {}
        self.order_history = {}
        
        # Error tracking
        self.error_count = 0
        self.last_error_time = 0
        self.max_errors_before_disconnect = 5
        
        # Connection callbacks
        self.connection_callbacks: List[Callable] = []
        self.disconnection_callbacks: List[Callable] = []
        
        self.logger.info("Enhanced Goodwill API Handler initialized")
    
    @property
    def is_authenticated(self) -> bool:
        """Check if currently authenticated"""
        return (self.auth_state == AuthenticationState.AUTHENTICATED and 
                self.token_manager.is_token_valid())
    
    async def initialize(self) -> bool:
        """Initialize HTTP session and prepare for authentication"""
        try:
            # Create HTTP session
            connector = aiohttp.TCPConnector(
                limit=100,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=60
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=self.timeout,
                headers={
                    'User-Agent': 'GoodwillTradingBot/1.0',
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                }
            )
            
            # Try to load saved tokens
            await self._load_saved_tokens()
            
            # Validate existing authentication
            if self.token_manager.access_token:
                is_valid = await self._validate_existing_session()
                if is_valid:
                    self.auth_state = AuthenticationState.AUTHENTICATED
                    self.logger.info("Existing session is valid")
                    return True
            
            self.logger.info("Goodwill API handler initialized, authentication required")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Goodwill API: {e}")
            self.error_handler.handle_error(e, "goodwill_initialization")
            return False
    
    def start_login_process(self) -> Optional[str]:
        """
        STEP 1: Generate login URL with API key and secret
        User inputs API key and secret via Flask dashboard
        """
        try:
            if not self.api_key or not self.api_secret:
                raise ValueError("API key and secret are required")
            
            # Generate login URL according to Goodwill API documentation
            # This will redirect to Goodwill login page in another tab
            login_url = (
                f"{self.base_url}/login/getlg/"
                f"?v=4&fyers_id={self.api_key}&app_id={self.api_key}"
                f"&redirect_uri={self.redirect_url}&state=sample_state"
            )
            
            self.auth_state = AuthenticationState.AUTHENTICATION_PENDING
            self.logger.info(f"Step 1: Login URL generated")
            
            return login_url
            
        except Exception as e:
            self.logger.error(f"Failed to start login process: {e}")
            self.error_handler.handle_error(e, "start_login_process")
            return None
    
    async def complete_login_with_request_token(self, request_token: str) -> bool:
        """
        STEP 3: Complete login using request token from redirect URL
        User copies the URL with request_token and pastes it in input box
        """
        try:
            if not request_token:
                raise ValueError("Request token is required")
            
            self.logger.info(f"Step 3: Completing login with request token")
            
            # Generate session according to Goodwill API documentation
            session_payload = {
                'fyers_id': self.api_key,
                'app_secret': self.api_secret,
                'request_token': request_token
            }
            
            # Make session generation request
            response = await self._make_request(
                'POST',
                '/login/create_session/',
                data=session_payload,
                authenticated=False
            )
            
            if response and response.get('s') == 'ok':
                # Extract session details
                data = response.get('data', {})
                access_token = data.get('access_token', '')
                refresh_token = data.get('refresh_token', '')
                
                if access_token:
                    # Store tokens - Goodwill tokens typically expire in 24 hours
                    expires_in = 86400  # 24 hours
                    self.token_manager.set_tokens(access_token, refresh_token, expires_in)
                    
                    # Fetch user profile information
                    await self._fetch_user_info()
                    
                    # Save tokens for future use
                    await self._save_tokens()
                    
                    self.auth_state = AuthenticationState.AUTHENTICATED
                    
                    # Notify connection callbacks
                    for callback in self.connection_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback()
                            else:
                                callback()
                        except Exception as e:
                            self.logger.error(f"Connection callback error: {e}")
                    
                    self.logger.info("Goodwill authentication completed successfully")
                    return True
                else:
                    raise ValueError("No access token received")
            else:
                error_msg = response.get('message', 'Session creation failed') if response else 'No response from server'
                self.auth_state = AuthenticationState.AUTHENTICATION_FAILED
                self.logger.error(f"Session creation failed: {error_msg}")
                return False
                
        except Exception as e:
            self.auth_state = AuthenticationState.AUTHENTICATION_FAILED
            self.logger.error(f"Login completion error: {e}")
            self.error_handler.handle_error(e, "complete_login")
            return False
    
    async def logout(self) -> bool:
        """Logout from Goodwill API"""
        try:
            if self.is_authenticated:
                # Revoke tokens if possible
                try:
                    revoke_payload = {
                        'fyers_id': self.api_key,
                        'token': self.token_manager.access_token
                    }
                    
                    await self._make_request(
                        'DELETE',
                        '/login/logout/',
                        data=revoke_payload
                    )
                except Exception as e:
                    self.logger.warning(f"Token revocation failed: {e}")
            
            # Clear tokens and state
            self.token_manager.clear_tokens()
            self.auth_state = AuthenticationState.NOT_AUTHENTICATED
            self.user_info = {}
            
            # Clear saved tokens
            await self._clear_saved_tokens()
            
            # Notify disconnection callbacks
            for callback in self.disconnection_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    self.logger.error(f"Disconnection callback error: {e}")
            
            self.logger.info("Logged out from Goodwill API")
            return True
            
        except Exception as e:
            self.logger.error(f"Logout error: {e}")
            return False
    
    async def place_order(self, order_data: Dict) -> Order:
        """Place a trading order using Goodwill API"""
        try:
            # Ensure authentication
            if not await self._ensure_authenticated():
                raise Exception("Authentication failed")
            
            # Validate order data
            validation_result = self._validate_order_data(order_data)
            if not validation_result['valid']:
                raise ValueError(f"Invalid order data: {validation_result['errors']}")
            
            # Prepare order payload according to Goodwill API format
            order_payload = {
                'symbol': f"{order_data['exchange']}:{order_data['symbol']}",
                'qty': order_data['quantity'],
                'type': self._convert_order_type(order_data.get('order_type', OrderType.MARKET.value)),
                'side': 1 if order_data['side'] == OrderSide.BUY.value else -1,
                'productType': self._convert_product_type(order_data.get('product_type', ProductType.INTRADAY.value)),
                'limitPrice': order_data.get('price', 0),
                'stopPrice': order_data.get('trigger_price', 0),
                'validity': order_data.get('validity', 'DAY'),
                'disclosedQty': order_data.get('disclosed_qty', 0),
                'offlineOrder': False,
                'stopLoss': 0,
                'takeProfit': 0
            }
            
            # Make order placement request
            response = await self._make_request(
                'POST',
                '/orders/',
                data=order_payload
            )
            
            if response and response.get('s') == 'ok':
                # Create order object
                data = response.get('data', {})
                order = Order(
                    order_id=data.get('id', ''),
                    symbol=order_data['symbol'],
                    exchange=order_data['exchange'],
                    side=OrderSide(order_data['side']),
                    order_type=OrderType(order_data.get('order_type', OrderType.MARKET.value)),
                    product_type=ProductType(order_data.get('product_type', ProductType.INTRADAY.value)),
                    quantity=order_data['quantity'],
                    price=order_data.get('price', 0),
                    disclosed_qty=order_data.get('disclosed_qty', 0),
                    trigger_price=order_data.get('trigger_price', 0),
                    validity=order_data.get('validity', 'DAY'),
                    status=OrderStatus.PENDING,
                    order_time=datetime.now()
                )
                
                # Store in active orders
                self.active_orders[order.order_id] = order
                
                self.logger.info(f"Order placed successfully: {order.order_id}")
                return order
                
            else:
                error_msg = response.get('message', 'Order placement failed') if response else 'No response from server'
                self.logger.error(f"Order placement failed: {error_msg}")
                
                # Create failed order
                order = Order(
                    order_id="",
                    symbol=order_data['symbol'],
                    exchange=order_data['exchange'],
                    side=OrderSide(order_data['side']),
                    order_type=OrderType(order_data.get('order_type', OrderType.MARKET.value)),
                    product_type=ProductType(order_data.get('product_type', ProductType.INTRADAY.value)),
                    quantity=order_data['quantity'],
                    price=order_data.get('price', 0),
                    status=OrderStatus.REJECTED,
                    rejection_reason=error_msg,
                    order_time=datetime.now()
                )
                
                return order
                
        except Exception as e:
            self.logger.error(f"Order placement error: {e}")
            self.error_handler.handle_error(e, "order_placement")
            
            # Return failed order
            return Order(
                order_id="",
                symbol=order_data.get('symbol', ''),
                exchange=order_data.get('exchange', ''),
                side=OrderSide(order_data.get('side', OrderSide.BUY.value)),
                order_type=OrderType.MARKET,
                product_type=ProductType.INTRADAY,
                quantity=order_data.get('quantity', 0),
                price=order_data.get('price', 0),
                status=OrderStatus.REJECTED,
                rejection_reason=str(e),
                order_time=datetime.now()
            )
    
    async def get_quote(self, exchange: str, token: str) -> Optional[Quote]:
        """Get market quote for a symbol using Goodwill API"""
        try:
            # Ensure authentication
            if not await self._ensure_authenticated():
                return None
            
            # Format symbol according to Goodwill API
            symbol = f"{exchange}:{token}"
            
            response = await self._make_request(
                'GET',
                f'/data/quotes/?symbols={symbol}',
                authenticated=True
            )
            
            if response and response.get('s') == 'ok':
                data = response.get('d', [])
                if data and len(data) > 0:
                    quote_data = data[0]
                    
                    quote = Quote(
                        symbol=quote_data.get('n', ''),
                        exchange=exchange,
                        token=token,
                        last_price=float(quote_data.get('v', {}).get('lp', 0)),
                        open_price=float(quote_data.get('v', {}).get('o', 0)),
                        high_price=float(quote_data.get('v', {}).get('h', 0)),
                        low_price=float(quote_data.get('v', {}).get('l', 0)),
                        close_price=float(quote_data.get('v', {}).get('prev_close_price', 0)),
                        volume=int(quote_data.get('v', {}).get('volume', 0)),
                        change=float(quote_data.get('v', {}).get('ch', 0)),
                        change_percent=float(quote_data.get('v', {}).get('chp', 0)),
                        bid_price=float(quote_data.get('v', {}).get('bid', 0)),
                        ask_price=float(quote_data.get('v', {}).get('ask', 0)),
                        bid_qty=int(quote_data.get('v', {}).get('bid_size', 0)),
                        ask_qty=int(quote_data.get('v', {}).get('ask_size', 0)),
                        timestamp=datetime.now()
                    )
                    
                    return quote
            
            self.logger.warning(f"Quote not found for {exchange}:{token}")
            return None
                
        except Exception as e:
            self.logger.error(f"Error getting quote: {e}")
            self.error_handler.handle_error(e, "get_quote")
            return None

    async def search_symbols(self, search_term: str, exchange: str = "NSE") -> List[Dict]:
        """Search for symbols using Goodwill API"""
        try:
            if not await self._ensure_authenticated():
                return []
            
            response = await self._make_request(
                'GET',
                f'/data/symbol_master/?symbol={search_term}&exchange={exchange}',
                authenticated=True
            )
            
            if response and response.get('s') == 'ok':
                symbols = []
                for symbol_data in response.get('d', []):
                    symbols.append({
                        'symbol': symbol_data.get('display_name', ''),
                        'exchange': symbol_data.get('exchange', ''),
                        'token': symbol_data.get('token', ''),
                        'instrument_type': symbol_data.get('instrument_type', ''),
                        'lot_size': int(symbol_data.get('lot_size', 1)),
                        'tick_size': float(symbol_data.get('tick_size', 0.05)),
                        'company_name': symbol_data.get('description', ''),
                        'expiry': symbol_data.get('expiry', ''),
                        'strike_price': float(symbol_data.get('strike_price', 0))
                    })
                
                return symbols
            else:
                self.logger.warning(f"No symbols found for search term: {search_term}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error searching symbols: {e}")
            self.error_handler.handle_error(e, "search_symbols")
            return []

    async def get_positions(self) -> List[Position]:
        """Get current positions using Goodwill API"""
        try:
            if not await self._ensure_authenticated():
                return []
            
            response = await self._make_request(
                'GET',
                '/positions/',
                authenticated=True
            )
            
            if response and response.get('s') == 'ok':
                positions = []
                
                for pos_data in response.get('netPositions', []):
                    if int(pos_data.get('netQty', 0)) != 0:
                        position = Position(
                            symbol=pos_data.get('symbol', ''),
                            exchange=pos_data.get('exchange', ''),
                            product_type=ProductType(pos_data.get('productType', ProductType.INTRADAY.value)),
                            quantity=int(pos_data.get('netQty', 0)),
                            average_price=float(pos_data.get('avgPrice', 0)),
                            last_price=float(pos_data.get('ltp', 0)),
                            unrealized_pnl=float(pos_data.get('unrealizedPnl', 0)),
                            realized_pnl=float(pos_data.get('realizedPnl', 0)),
                            day_pnl=float(pos_data.get('pl', 0)),
                            day_pnl_percent=0.0  # Calculate if needed
                        )
                        
                        # Calculate day P&L percentage
                        if position.average_price > 0:
                            position.day_pnl_percent = (position.day_pnl / (position.average_price * abs(position.quantity))) * 100
                        
                        positions.append(position)
                
                return positions
            else:
                self.logger.info("No positions found")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            self.error_handler.handle_error(e, "get_positions")
            return []

    async def get_holdings(self) -> List[Dict]:
        """Get holdings/investments using Goodwill API"""
        try:
            if not await self._ensure_authenticated():
                return []
            
            response = await self._make_request(
                'GET',
                '/holdings/',
                authenticated=True
            )
            
            if response and response.get('s') == 'ok':
                holdings = []
                
                for holding_data in response.get('holdings', []):
                    holdings.append({
                        'symbol': holding_data.get('symbol', ''),
                        'exchange': holding_data.get('exchange', ''),
                        'quantity': int(holding_data.get('quantity', 0)),
                        'average_price': float(holding_data.get('costPrice', 0)),
                        'last_price': float(holding_data.get('ltp', 0)),
                        'current_value': float(holding_data.get('marketVal', 0)),
                        'pnl': float(holding_data.get('pl', 0)),
                        'pnl_percent': float(holding_data.get('plPercent', 0)),
                        'product_type': holding_data.get('productType', ''),
                        'collateral_qty': int(holding_data.get('collateralQty', 0)),
                        'collateral_value': float(holding_data.get('collateralVal', 0))
                    })
                
                return holdings
            else:
                self.logger.info("No holdings found")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting holdings: {e}")
            self.error_handler.handle_error(e, "get_holdings")
            return []

    async def get_order_book(self) -> List[Order]:
        """Get all orders for the day using Goodwill API"""
        try:
            if not await self._ensure_authenticated():
                return []
            
            response = await self._make_request(
                'GET',
                '/orders/',
                authenticated=True
            )
            
            if response and response.get('s') == 'ok':
                orders = []
                
                for order_data in response.get('orderBook', []):
                    order = Order(
                        order_id=order_data.get('id', ''),
                        symbol=order_data.get('symbol', '').split(':')[-1] if ':' in order_data.get('symbol', '') else order_data.get('symbol', ''),
                        exchange=order_data.get('symbol', '').split(':')[0] if ':' in order_data.get('symbol', '') else 'NSE',
                        side=OrderSide.BUY if order_data.get('side') == 1 else OrderSide.SELL,
                        order_type=OrderType(order_data.get('type', 1)),
                        product_type=ProductType(order_data.get('productType', ProductType.INTRADAY.value)),
                        quantity=int(order_data.get('qty', 0)),
                        price=float(order_data.get('limitPrice', 0)),
                        disclosed_qty=int(order_data.get('disclosedQty', 0)),
                        trigger_price=float(order_data.get('stopPrice', 0)),
                        validity=order_data.get('validity', 'DAY'),
                        status=self._parse_order_status(order_data.get('status', '')),
                        filled_qty=int(order_data.get('filledQty', 0)),
                        pending_qty=int(order_data.get('qty', 0)) - int(order_data.get('filledQty', 0)),
                        average_price=float(order_data.get('avgPrice', 0)),
                        exchange_order_id=order_data.get('exchOrdId', ''),
                        rejection_reason=order_data.get('message', '')
                    )
                    
                    # Parse order time
                    if order_data.get('orderDateTime'):
                        try:
                            order.order_time = datetime.fromisoformat(order_data['orderDateTime'])
                        except:
                            order.order_time = datetime.now()
                    
                    orders.append(order)
                
                return orders
            else:
                self.logger.warning("No orders found or invalid response")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting order book: {e}")
            self.error_handler.handle_error(e, "get_order_book")
            return []

    async def get_funds(self) -> Dict:
        """Get account funds/margin information using Goodwill API"""
        try:
            if not await self._ensure_authenticated():
                return {}
            
            response = await self._make_request(
                'GET',
                '/funds/',
                authenticated=True
            )
            
            if response and response.get('s') == 'ok':
                fund_data = response.get('fund_limit', [])
                if fund_data:
                    funds = fund_data[0]  # Usually first element contains the fund details
                    return {
                        'available_balance': float(funds.get('availableBalance', 0)),
                        'payin_amount': float(funds.get('payinAmount', 0)),
                        'payout_amount': float(funds.get('payoutAmount', 0)),
                        'utilized_amount': float(funds.get('utilizedAmount', 0)),
                        'blocked_amount': float(funds.get('blockedAmount', 0)),
                        'withdrawable_balance': float(funds.get('withdrawableBalance', 0)),
                        'collateral': float(funds.get('collateral', 0)),
                        'total_balance': float(funds.get('totAvailableBalance', 0))
                    }
                else:
                    return {}
            else:
                self.logger.warning("Funds information not available")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting funds: {e}")
            self.error_handler.handle_error(e, "get_funds")
            return {}

    async def get_trade_book(self) -> List[Dict]:
        """Get trade book for the day using Goodwill API"""
        try:
            if not await self._ensure_authenticated():
                return []
            
            response = await self._make_request(
                'GET',
                '/tradebook/',
                authenticated=True
            )
            
            if response and response.get('s') == 'ok':
                trades = []
                
                for trade_data in response.get('tradeBook', []):
                    trades.append({
                        'order_id': trade_data.get('orderNumber', ''),
                        'symbol': trade_data.get('symbol', ''),
                        'exchange': trade_data.get('exchange', ''),
                        'side': 'BUY' if trade_data.get('side') == 1 else 'SELL',
                        'quantity': int(trade_data.get('qty', 0)),
                        'price': float(trade_data.get('tradePrice', 0)),
                        'product_type': trade_data.get('productType', ''),
                        'trade_time': trade_data.get('tradeTime', ''),
                        'trade_id': trade_data.get('tradeNumber', ''),
                        'trade_value': float(trade_data.get('tradeValue', 0)),
                        'remarks': trade_data.get('remarks', '')
                    })
                
                return trades
            else:
                self.logger.info("No trades found")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting trade book: {e}")
            self.error_handler.handle_error(e, "get_trade_book")
            return []

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order using Goodwill API"""
        try:
            if not await self._ensure_authenticated():
                return False
            
            response = await self._make_request(
                'DELETE',
                f'/orders/{order_id}',
                authenticated=True
            )
            
            if response and response.get('s') == 'ok':
                self.logger.info(f"Order cancelled successfully: {order_id}")
                
                # Update local order status
                if order_id in self.active_orders:
                    self.active_orders[order_id].status = OrderStatus.CANCELLED
                
                return True
            else:
                error_msg = response.get('message', 'Order cancellation failed') if response else 'No response from server'
                self.logger.error(f"Order cancellation failed: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"Order cancellation error: {e}")
            self.error_handler.handle_error(e, "order_cancellation")
            return False

    async def modify_order(self, order_id: str, modifications: Dict) -> bool:
        """Modify an existing order using Goodwill API"""
        try:
            if not await self._ensure_authenticated():
                return False
            
            # Prepare modification payload
            modify_payload = {}
            
            if modifications.get('quantity'):
                modify_payload['qty'] = modifications['quantity']
            if modifications.get('price'):
                modify_payload['limitPrice'] = modifications['price']
            if modifications.get('trigger_price'):
                modify_payload['stopPrice'] = modifications['trigger_price']
            if modifications.get('disclosed_qty'):
                modify_payload['disclosedQty'] = modifications['disclosed_qty']
            
            # Make modification request
            response = await self._make_request(
                'PUT',
                f'/orders/{order_id}',
                data=modify_payload
            )
            
            if response and response.get('s') == 'ok':
                self.logger.info(f"Order modified successfully: {order_id}")
                
                # Update local order if exists
                if order_id in self.active_orders:
                    order = self.active_orders[order_id]
                    if modifications.get('quantity'):
                        order.quantity = modifications['quantity']
                    if modifications.get('price'):
                        order.price = modifications['price']
                
                return True
            else:
                error_msg = response.get('message', 'Order modification failed') if response else 'No response from server'
                self.logger.error(f"Order modification failed: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"Order modification error: {e}")
            self.error_handler.handle_error(e, "order_modification")
            return False

    async def get_historical_data(self, symbol: str, resolution: str = "1", 
                                 from_date: datetime, to_date: datetime) -> List[Dict]:
        """Get historical market data using Goodwill API"""
        try:
            if not await self._ensure_authenticated():
                return []
            
            # Format dates for API
            from_timestamp = int(from_date.timestamp())
            to_timestamp = int(to_date.timestamp())
            
            params = {
                'symbol': symbol,
                'resolution': resolution,
                'date_format': '1',
                'range_from': from_timestamp,
                'range_to': to_timestamp,
                'cont_flag': '1'
            }
            
            # Build query string
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            
            response = await self._make_request(
                'GET',
                f'/data/history/?{query_string}',
                authenticated=True
            )
            
            if response and response.get('s') == 'ok':
                historical_data = []
                candles = response.get('candles', [])
                
                for candle in candles:
                    if len(candle) >= 6:
                        historical_data.append({
                            'timestamp': datetime.fromtimestamp(candle[0]),
                            'open': float(candle[1]),
                            'high': float(candle[2]),
                            'low': float(candle[3]),
                            'close': float(candle[4]),
                            'volume': int(candle[5])
                        })
                
                return historical_data
            else:
                self.logger.warning(f"No historical data found for {symbol}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            self.error_handler.handle_error(e, "get_historical_data")
            return []

    async def get_authentication_status(self) -> Dict:
        """Get current authentication status"""
        return {
            'is_authenticated': self.is_authenticated,
            'access_token': self.token_manager.access_token[:20] + "..." if self.token_manager.access_token else "",
            'user_id': self.user_info.get('client_id', ''),
            'user_name': self.user_info.get('username', ''),
            'email': self.user_info.get('email', ''),
            'expires_at': self.token_manager.expires_at.isoformat() if self.token_manager.expires_at else None,
            'time_to_expiry': str(self.token_manager.expires_at - datetime.now()) if self.token_manager.expires_at else None,
            'last_error': self.last_error_time,
            'error_count': self.error_count,
            'auth_state': self.auth_state.value
        }

    async def get_api_statistics(self) -> Dict:
        """Get API usage statistics"""
        return {
            'total_requests': self.request_count,
            'requests_per_minute': self.request_count if self.request_count < self.max_requests_per_minute else self.max_requests_per_minute,
            'rate_limit_window': self.rate_limit_window,
            'max_requests_per_minute': self.max_requests_per_minute,
            'last_request_time': self.last_request_time,
            'min_request_interval': self.min_request_interval,
            'cache_entries': len(self.symbol_cache),
            'active_orders': len(self.active_orders),
            'error_count': self.error_count,
            'is_rate_limited': self._is_rate_limited(),
            'base_url': self.base_url,
            'api_key_set': bool(self.api_key),
            'session_active': self.session and not self.session.closed
        }

    # Private helper methods
    
    async def _fetch_user_info(self):
        """Fetch user profile information after authentication"""
        try:
            # Get user profile according to Goodwill API
            response = await self._make_request(
                'GET',
                '/user/profile/',
                authenticated=True
            )
            
            if response and response.get('s') == 'ok':
                data = response.get('data', {})
                self.user_info = {
                    'client_id': data.get('fy_id', ''),
                    'username': data.get('name', ''),
                    'email': data.get('email_id', ''),
                    'mobile': data.get('mobile_number', ''),
                    'pan': data.get('PAN', ''),
                    'demat_id': data.get('demat_account', ''),
                    'firm_name': data.get('firm_name', 'Goodwill'),
                    'exchanges': data.get('exchange', []),
                    'segment': data.get('segment', []),
                    'cm_codes': data.get('cm_codes', {}),
                    'fo_codes': data.get('fo_codes', {}),
                    'cd_codes': data.get('cd_codes', {})
                }
                
                # Store client_id as user_id for consistency
                if self.user_info['client_id']:
                    self.user_id = self.user_info['client_id']
                
                self.logger.info(f"User info fetched: {self.user_info['username']} ({self.user_info['client_id']})")
            else:
                self.logger.warning("Failed to fetch user profile")
                
        except Exception as e:
            self.logger.error(f"Error fetching user info: {e}")

    async def _make_request(self, method: str, endpoint: str, data: Dict = None, 
                          authenticated: bool = True) -> Optional[Dict]:
        """Make HTTP request to Goodwill API"""
        try:
            # Rate limiting
            await self._enforce_rate_limit()
            
            # Session validation for authenticated requests
            if authenticated and not await self._ensure_authenticated():
                raise Exception("Authentication required")
            
            # Prepare URL
            url = f"{self.base_url}{endpoint}"
            
            # Prepare headers
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'GoodwillTradingBot/1.0'
            }
            
            if authenticated and self.token_manager.access_token:
                headers['Authorization'] = f'Bearer {self.token_manager.access_token}'
            
            # Prepare request data
            request_data = data or {}
            
            # Make request
            async with self.session.request(
                method=method,
                url=url,
                json=request_data,
                headers=headers
            ) as response:
                
                # Update request count
                self.request_count += 1
                self.last_request_time = time.time()
                
                # Handle response
                if response.status == 200:
                    response_data = await response.json()
                    
                    # Reset error count on successful request
                    self.error_count = 0
                    
                    return response_data
                
                elif response.status == 401:
                    self.logger.warning("Authentication failed - token may be expired")
                    self.auth_state = AuthenticationState.TOKEN_EXPIRED
                    return None
                
                elif response.status == 429:
                    self.logger.warning("Rate limit exceeded")
                    await asyncio.sleep(1)
                    return None
                
                else:
                    self.logger.error(f"API request failed: {response.status}")
                    response_text = await response.text()
                    self.logger.error(f"Response: {response_text}")
                    return None
                    
        except asyncio.TimeoutError:
            self.logger.error(f"Request timeout for {endpoint}")
            self._handle_error()
            return None
            
        except Exception as e:
            self.logger.error(f"Request error for {endpoint}: {e}")
            self._handle_error()
            return None

    async def _ensure_authenticated(self) -> bool:
        """Ensure we have valid authentication"""
        if not self.is_authenticated:
            self.logger.warning("Authentication required - please login first")
            return False
        
        # Check if token needs refresh
        if not self.token_manager.is_token_valid():
            if self.token_manager.is_refresh_token_valid():
                success = await self.token_manager.refresh_access_token()
                if success:
                    self.auth_state = AuthenticationState.AUTHENTICATED
                    return True
                else:
                    self.auth_state = AuthenticationState.TOKEN_EXPIRED
                    return False
            else:
                self.auth_state = AuthenticationState.TOKEN_EXPIRED
                return False
        
        return True

    async def _validate_existing_session(self) -> bool:
        """Validate existing session"""
        try:
            response = await self._make_request(
                'GET',
                '/user/profile/',
                authenticated=True
            )
            
            return response and response.get('s') == 'ok'
            
        except Exception as e:
            self.logger.error(f"Session validation error: {e}")
            return False

    async def _load_saved_tokens(self):
        """Load saved tokens from secure storage"""
        try:
            # This would load from secure storage like keyring
            # For now, we'll skip this implementation
            pass
        except Exception as e:
            self.logger.error(f"Error loading saved tokens: {e}")

    async def _save_tokens(self):
        """Save tokens to secure storage"""
        try:
            # This would save to secure storage like keyring
            # For now, we'll skip this implementation
            pass
        except Exception as e:
            self.logger.error(f"Error saving tokens: {e}")

    async def _clear_saved_tokens(self):
        """Clear saved tokens from secure storage"""
        try:
            # This would clear from secure storage
            pass
        except Exception as e:
            self.logger.error(f"Error clearing saved tokens: {e}")

    async def _enforce_rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        
        # Check if we need to wait
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last_request
            await asyncio.sleep(wait_time)
        
        # Check requests per minute limit
        if self._is_rate_limited():
            self.logger.warning("Rate limit reached, waiting...")
            await asyncio.sleep(60)  # Wait 1 minute
            self.request_count = 0  # Reset counter

    def _is_rate_limited(self) -> bool:
        """Check if we're hitting rate limits"""
        return self.request_count >= self.max_requests_per_minute

    def _handle_error(self):
        """Handle API errors"""
        self.error_count += 1
        self.last_error_time = time.time()
        
        if self.error_count >= self.max_errors_before_disconnect:
            self.logger.error("Too many errors, marking as disconnected...")
            self.auth_state = AuthenticationState.AUTHENTICATION_FAILED

    def _validate_order_data(self, order_data: Dict) -> Dict:
        """Validate order data before placement"""
        errors = []
        
        # Required fields
        required_fields = ['symbol', 'exchange', 'side', 'quantity']
        for field in required_fields:
            if field not in order_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate side
        if order_data.get('side') not in [OrderSide.BUY.value, OrderSide.SELL.value]:
            errors.append("Invalid order side")
        
        # Validate quantity
        if order_data.get('quantity', 0) <= 0:
            errors.append("Quantity must be positive")
        
        # Validate price for limit orders
        order_type = order_data.get('order_type', OrderType.MARKET.value)
        if order_type in [OrderType.LIMIT.value, OrderType.STOP_LOSS.value]:
            if order_data.get('price', 0) <= 0:
                errors.append("Price must be positive for limit/stop orders")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

    def _convert_order_type(self, order_type) -> int:
        """Convert order type to Goodwill API format"""
        type_mapping = {
            OrderType.MARKET.value: 1,
            OrderType.LIMIT.value: 2,
            OrderType.STOP_LOSS.value: 3,
            OrderType.STOP_LOSS_MARKET.value: 4
        }
        return type_mapping.get(order_type, 1)

    def _convert_product_type(self, product_type) -> str:
        """Convert product type to Goodwill API format"""
        return product_type

    def _parse_order_status(self, status_string: str) -> OrderStatus:
        """Parse order status from API response"""
        status_mapping = {
            'PENDING': OrderStatus.PENDING,
            'OPEN': OrderStatus.OPEN,
            'COMPLETE': OrderStatus.COMPLETE,
            'CANCELLED': OrderStatus.CANCELLED,
            'REJECTED': OrderStatus.REJECTED,
            'PARTIAL': OrderStatus.PARTIAL
        }
        
        return status_mapping.get(status_string.upper(), OrderStatus.PENDING)

    def add_connection_callback(self, callback: Callable):
        """Add callback to be called when connected"""
        self.connection_callbacks.append(callback)

    def add_disconnection_callback(self, callback: Callable):
        """Add callback to be called when disconnected"""
        self.disconnection_callbacks.append(callback)

    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.is_authenticated:
                await self.logout()
            
            if self.token_manager.auto_refresh_task:
                self.token_manager.auto_refresh_task.cancel()
            
            if self.session and not self.session.closed:
                await self.session.close()
            
            self.logger.info("Goodwill API handler cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

# Singleton instance
_goodwill_handler = None

def get_goodwill_handler() -> GoodwillAPIHandler:
    """Get singleton Goodwill API handler instance"""
    global _goodwill_handler
    if _goodwill_handler is None:
        _goodwill_handler = GoodwillAPIHandler()
    return _goodwill_handler

# Example usage and testing
async def main():
    """Example usage of Goodwill API handler with 3-step login flow"""
    try:
        # Initialize handler
        handler = get_goodwill_handler()
        
        # Initialize
        success = await handler.initialize()
        if not success:
            print("❌ Failed to initialize Goodwill API")
            return
        
        print("✅ Goodwill API initialized successfully")
        
        # STEP 1: Generate login URL (user inputs API key/secret via dashboard)
        handler.api_key = "YOUR_API_KEY"  # This would come from Flask dashboard
        handler.api_secret = "YOUR_SECRET"  # This would come from Flask dashboard
        
        login_url = handler.start_login_process()
        if login_url:
            print(f"🔗 STEP 1: Login URL generated:")
            print(f"   {login_url}")
            print(f"   Open this URL in browser and complete login")
        
        # STEP 2: User opens URL in browser, logs into Goodwill, gets redirected with request_token
        
        # STEP 3: User copies the redirect URL with request_token and pastes it
        request_token = input("\n📋 STEP 3: Paste the request_token from redirect URL: ")
        
        if request_token:
            success = await handler.complete_login_with_request_token(request_token)
            if success:
                print("✅ Authentication successful!")
                
                # Get authentication status
                auth_status = await handler.get_authentication_status()
                print(f"🔐 User: {auth_status['user_name']} ({auth_status['user_id']})")
                print(f"📧 Email: {auth_status['email']}")
                
                # Test some API calls
                print("\n🧪 Testing API calls...")
                
                # Get funds
                funds = await handler.get_funds()
                if funds:
                    print(f"💰 Available Balance: ₹{funds.get('available_balance', 0):,.2f}")
                
                # Search symbols
                symbols = await handler.search_symbols("RELIANCE", "NSE")
                print(f"🔍 Found {len(symbols)} symbols for RELIANCE")
                
                # Get positions
                positions = await handler.get_positions()
                print(f"📊 Current positions: {len(positions)}")
                
                # Get orders
                orders = await handler.get_order_book()
                print(f"📋 Orders today: {len(orders)}")
                
            else:
                print("❌ Authentication failed")
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'handler' in locals():
            await handler.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
