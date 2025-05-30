#!/usr/bin/env python3
"""
Goodwill API Handler with 3-Step Authentication Flow
Production-ready live trading integration for Indian markets
Official API: https://api.gwcindia.in/v1/
Documentation: https://developer.gwcindia.in/api/
"""

import asyncio
import logging
import json
import time
import hashlib
import hmac
import urllib.parse
import webbrowser
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import requests
import websocket
import threading
from collections import deque

logger = logging.getLogger(__name__)

class AuthenticationState(Enum):
    NOT_AUTHENTICATED = "NOT_AUTHENTICATED"
    STEP1_API_SUBMITTED = "STEP1_API_SUBMITTED"
    STEP2_LOGIN_PENDING = "STEP2_LOGIN_PENDING"
    STEP3_TOKEN_PENDING = "STEP3_TOKEN_PENDING"
    AUTHENTICATED = "AUTHENTICATED"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"

@dataclass
class LivePosition:
    """Live position with integrated risk management"""
    symbol: str
    quantity: int
    side: str  # 'BUY' or 'SELL'
    entry_price: float
    current_price: float
    entry_time: datetime
    strategy: str
    order_id: str
    
    # Adaptive stop loss
    initial_stop: float
    current_stop: float
    trailing_stop_distance: float = 0.0
    trailing_activated: bool = False
    
    # Risk metrics
    position_value: float = 0.0
    unrealized_pnl: float = 0.0
    risk_amount: float = 0.0
    max_favorable_move: float = 0.0
    max_adverse_move: float = 0.0
    
    # Exit tracking
    bars_held: int = 0
    momentum_declining_count: int = 0
    volume_declining_count: int = 0
    profit_target_hit: bool = False
    
    def update_position(self, current_price: float, high: float, low: float):
        """Update position metrics"""
        self.current_price = current_price
        self.position_value = current_price * self.quantity
        
        if self.side == 'BUY':
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
            self.max_favorable_move = max(self.max_favorable_move, (high - self.entry_price) / self.entry_price)
            self.max_adverse_move = min(self.max_adverse_move, (low - self.entry_price) / self.entry_price)
        else:  # SELL
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
            self.max_favorable_move = max(self.max_favorable_move, (self.entry_price - low) / self.entry_price)
            self.max_adverse_move = min(self.max_adverse_move, (self.entry_price - high) / self.entry_price)
    
    def calculate_profit_pct(self) -> float:
        """Calculate profit percentage"""
        if self.side == 'BUY':
            return (self.current_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.current_price) / self.entry_price * 100

class GoodwillAPIHandler:
    """
    Goodwill API Handler with 3-Step Authentication Flow
    Complete integration for Indian market live trading
    Official API: https://api.gwcindia.in/v1/
    Documentation: https://developer.gwcindia.in/api/
    """
    
    def __init__(self, config):
        self.config = config
        
        # API Configuration for Goodwill (Official GWC India)
        self.base_url = "https://api.gwcindia.in/v1"
        self.login_url = "https://gwcindia.in/auth"
        self.api_docs_url = "https://developer.gwcindia.in/api/"
        
        # Authentication state
        self.auth_state = AuthenticationState.NOT_AUTHENTICATED
        self.api_key = None
        self.secret_key = None
        self.request_token = None
        self.access_token = None
        self.client_id = None
        self.session_expiry = None
        
        # Session management
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'GoodwillScalpingBot/1.0'
        })
        
        # WebSocket for real-time data (Official GWC India)
        self.ws = None
        self.ws_thread = None
        self.is_ws_connected = False
        self.ws_url = "wss://ws.gwcindia.in/quotes"
        
        # Risk management integration
        self.risk_profile = None
        self.live_positions: Dict[str, LivePosition] = {}
        self.order_history = deque(maxlen=1000)
        self.rejected_orders = deque(maxlen=100)
        
        # Account info
        self.account_balance = 0.0
        self.available_margin = 0.0
        self.used_margin = 0.0
        self.total_pnl = 0.0
        
        # Trading limits (Indian market specific)
        self.min_order_value = 500  # Minimum order value in INR
        self.max_single_order_value = 1000000  # 10 Lakh INR
        self.daily_loss_limit = 50000  # 50k INR daily loss limit
        self.daily_pnl = 0.0
        
        # Market timing (Indian market hours)
        self.market_hours = {
            'pre_open_start': '09:00',
            'market_open': '09:15',
            'market_close': '15:30',
            'after_hours_end': '16:00'
        }
        
        # Real-time data storage
        self.live_quotes = {}
        self.tick_data = {}
        
        # API Endpoints (Official Goodwill/GWC India API)
        # Base URL: https://api.gwcindia.in/v1/
        # Documentation: https://developer.gwcindia.in/api/
        self.endpoints = {
            'login': '/auth/login',
            'profile': '/user/profile',
            'orders': '/orders',
            'positions': '/portfolio/positions',
            'holdings': '/portfolio/holdings',
            'margins': '/user/margins',
            'instruments': '/instruments',
            'quotes': '/market/quotes',
            'historical': '/market/historical',
            'place_order': '/orders/place',
            'modify_order': '/orders/modify',
            'cancel_order': '/orders/cancel',
            'orderbook': '/orders/book',
            'tradebook': '/orders/trades'
        }
        
        logger.info("🔗 Goodwill API Handler initialized for 3-step authentication")
        logger.info(f"📡 API Base URL: {self.base_url}")
        logger.info(f"📚 API Documentation: {self.api_docs_url}")
    
    # ==========================================
    # 3-STEP AUTHENTICATION FLOW
    # ==========================================
    
    def start_authentication_flow(self) -> Dict:
        """
        STEP 1: Start 3-step authentication flow
        User provides API key and secret key
        """
        try:
            self.auth_state = AuthenticationState.NOT_AUTHENTICATED
            
            return {
                'step': 1,
                'title': 'Step 1: API Credentials',
                'message': 'Please provide your Goodwill API credentials',
                'required_fields': ['api_key', 'secret_key'],
                'instructions': [
                    '1. Login to your Goodwill account',
                    '2. Go to API section and generate API key',
                    '3. Copy your API key and secret key',
                    '4. Paste them below and click Submit'
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Authentication flow start error: {e}")
            return {'error': str(e)}
    
    def submit_api_credentials(self, api_key: str, secret_key: str) -> Dict:
        """
        STEP 2: Submit API credentials and generate login URL
        """
        try:
            self.api_key = api_key.strip()
            self.secret_key = secret_key.strip()
            
            if not self.api_key or not self.secret_key:
                return {
                    'success': False,
                    'error': 'API key and secret key are required'
                }
            
            # Generate login URL with API key
            login_params = {
                'api_key': self.api_key,
                'redirect_url': 'http://localhost:8000/auth/callback',
                'response_type': 'code'
            }
            
            login_url = f"{self.login_url}?" + urllib.parse.urlencode(login_params)
            
            self.auth_state = AuthenticationState.STEP1_API_SUBMITTED
            
            return {
                'step': 2,
                'success': True,
                'title': 'Step 2: Login to Goodwill',
                'message': 'Click the login URL below to authenticate with Goodwill',
                'login_url': login_url,
                'instructions': [
                    '1. Click the login URL below',
                    '2. It will open Goodwill login page in new tab',
                    '3. Login with your Goodwill credentials',
                    '4. After login, you will be redirected to a page with request token',
                    '5. Copy the request token and paste it in Step 3'
                ],
                'next_step': 'Copy the request token from redirect URL'
            }
            
        except Exception as e:
            logger.error(f"❌ API credentials submission error: {e}")
            self.auth_state = AuthenticationState.AUTHENTICATION_FAILED
            return {'success': False, 'error': str(e)}
    
    def open_login_url(self, login_url: str) -> Dict:
        """Helper to open login URL in browser"""
        try:
            webbrowser.open(login_url)
            self.auth_state = AuthenticationState.STEP2_LOGIN_PENDING
            
            return {
                'success': True,
                'message': 'Login URL opened in browser. Please complete login and copy request token.',
                'status': 'Login page opened in browser'
            }
            
        except Exception as e:
            logger.error(f"❌ Login URL open error: {e}")
            return {'success': False, 'error': str(e)}
    
    def submit_request_token(self, request_token: str) -> Dict:
        """
        STEP 3: Submit request token and complete authentication
        """
        try:
            self.request_token = request_token.strip()
            
            if not self.request_token:
                return {
                    'success': False,
                    'error': 'Request token is required'
                }
            
            self.auth_state = AuthenticationState.STEP3_TOKEN_PENDING
            
            # Exchange request token for access token
            auth_result = self._exchange_token_for_access()
            
            if auth_result.get('success'):
                self.auth_state = AuthenticationState.AUTHENTICATED
                
                return {
                    'step': 3,
                    'success': True,
                    'title': 'Authentication Successful!',
                    'message': 'Successfully authenticated with Goodwill API',
                    'client_id': self.client_id,
                    'access_token': self.access_token[:20] + '...' if self.access_token else None,
                    'session_expiry': self.session_expiry.isoformat() if self.session_expiry else None,
                    'account_info': auth_result.get('account_info', {}),
                    'status': 'Ready for live trading'
                }
            else:
                self.auth_state = AuthenticationState.AUTHENTICATION_FAILED
                return {
                    'success': False,
                    'error': auth_result.get('error', 'Token exchange failed')
                }
                
        except Exception as e:
            logger.error(f"❌ Request token submission error: {e}")
            self.auth_state = AuthenticationState.AUTHENTICATION_FAILED
            return {'success': False, 'error': str(e)}
    
    def _exchange_token_for_access(self) -> Dict:
        """Exchange request token for access token"""
        try:
            # Generate checksum for authentication
            checksum_string = self.api_key + self.request_token + self.secret_key
            checksum = hashlib.sha256(checksum_string.encode()).hexdigest()
            
            # Token exchange request
            token_data = {
                'api_key': self.api_key,
                'request_token': self.request_token,
                'checksum': checksum
            }
            
            response = self.session.post(
                f"{self.base_url}{self.endpoints['login']}/token",
                json=token_data
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('status') == 'success':
                    # Extract authentication details
                    self.access_token = result.get('data', {}).get('access_token')
                    self.client_id = result.get('data', {}).get('user_id')
                    
                    # Set session expiry (typically 24 hours)
                    self.session_expiry = datetime.now() + timedelta(hours=24)
                    
                    # Update session headers
                    self.session.headers.update({
                        'Authorization': f'Bearer {self.access_token}',
                        'X-Client-Id': self.client_id
                    })
                    
                    # Get account information
                    account_info = self._get_account_info()
                    
                    logger.info("✅ Successfully authenticated with Goodwill API")
                    
                    return {
                        'success': True,
                        'access_token': self.access_token,
                        'client_id': self.client_id,
                        'session_expiry': self.session_expiry,
                        'account_info': account_info
                    }
                else:
                    error_msg = result.get('message', 'Token exchange failed')
                    logger.error(f"❌ Token exchange error: {error_msg}")
                    return {'success': False, 'error': error_msg}
            
            else:
                logger.error(f"❌ Token exchange HTTP error: {response.status_code} - {response.text}")
                return {'success': False, 'error': f'HTTP {response.status_code}: {response.text}'}
                
        except Exception as e:
            logger.error(f"❌ Token exchange exception: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_account_info(self) -> Dict:
        """Get account information after authentication"""
        try:
            response = self.session.get(f"{self.base_url}{self.endpoints['profile']}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    user_data = data.get('data', {})
                    
                    # Update account information
                    self.account_balance = float(user_data.get('available_balance', 0))
                    self.available_margin = float(user_data.get('available_margin', 0))
                    self.used_margin = float(user_data.get('used_margin', 0))
                    
                    return user_data
                else:
                    logger.warning(f"Profile API error: {data.get('error_msg', 'Unknown error')}")
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Account info error: {e}")
            return {}
    
    # ==========================================
    # API ENDPOINTS IMPLEMENTATION
    # ==========================================
    
    async def get_profile(self) -> Optional[Dict]:
        """Get user profile information"""
        try:
            if not self.access_token:
                return None
            
            response = self.session.get(f"{self.base_url}{self.endpoints['profile']}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    return data.get('data', {})
                else:
                    logger.error(f"Profile error: {data.get('error_msg', 'Unknown error')}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Get profile error: {e}")
            return None
    
    async def get_margins(self) -> Optional[Dict]:
        """Get margin information"""
        try:
            if not self.access_token:
                return None
            
            response = self.session.get(f"{self.base_url}{self.endpoints['margins']}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    margin_data = data.get('data', {})
                    
                    # Update margin information
                    self.available_margin = float(margin_data.get('available_margin', 0))
                    self.used_margin = float(margin_data.get('used_margin', 0))
                    
                    return margin_data
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Get margins error: {e}")
            return None
    
    async def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            if not self.access_token:
                return []
            
            response = self.session.get(f"{self.base_url}{self.endpoints['positions']}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    positions_data = data.get('data', [])
                    
                    # Update live positions
                    self.live_positions.clear()
                    for pos_data in positions_data:
                        if int(pos_data.get('quantity', 0)) != 0:
                            symbol = pos_data['symbol'].replace('-EQ', '')  # Remove -EQ suffix
                            
                            position = LivePosition(
                                symbol=symbol,
                                quantity=abs(int(pos_data['quantity'])),
                                side='BUY' if int(pos_data['quantity']) > 0 else 'SELL',
                                entry_price=float(pos_data.get('avg_price', 0)),
                                current_price=float(pos_data.get('ltp', pos_data.get('avg_price', 0))),
                                entry_time=datetime.now(),  # Approximate
                                strategy='EXISTING',
                                order_id=pos_data.get('order_id', 'UNKNOWN'),
                                initial_stop=float(pos_data.get('avg_price', 0)) * 0.98,
                                current_stop=float(pos_data.get('avg_price', 0)) * 0.98
                            )
                            
                            self.live_positions[symbol] = position
                    
                    return positions_data
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Get positions error: {e}")
            return []
    
    async def get_holdings(self) -> List[Dict]:
        """Get holdings information"""
        try:
            if not self.access_token:
                return []
            
            response = self.session.get(f"{self.base_url}{self.endpoints['holdings']}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    return data.get('data', [])
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Get holdings error: {e}")
            return []
    
    async def get_quote(self, symbol: str, exchange: str = "NSE") -> Optional[Dict]:
        """Get live quote for symbol"""
        try:
            if not self.access_token:
                return None
            
            # Get symbol token first
            token = await self._get_symbol_token(symbol, exchange)
            if not token:
                logger.warning(f"No token found for {symbol}")
                return None
            
            payload = {
                "exchange": exchange,
                "token": token
            }
            
            response = self.session.post(f"{self.base_url}/getquote", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    quote_data = data.get('data', {})
                    
                    return {
                        'symbol': symbol,
                        'price': float(quote_data.get('ltp', quote_data.get('last_price', 0))),
                        'volume': int(quote_data.get('volume', 0)),
                        'high': float(quote_data.get('high', 0)),
                        'low': float(quote_data.get('low', 0)),
                        'open': float(quote_data.get('open', 0)),
                        'change': float(quote_data.get('change', 0)),
                        'change_per': float(quote_data.get('change_percentage', quote_data.get('change_per', 0))),
                        'bid': float(quote_data.get('bid', 0)),
                        'ask': float(quote_data.get('ask', 0)),
                        'timestamp': datetime.now()
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Get quote error for {symbol}: {e}")
            return None
    
    async def _get_symbol_token(self, symbol: str, exchange: str = "NSE") -> Optional[str]:
        """Get symbol token using fetchsymbol API"""
        try:
            if not self.access_token:
                return None
            
            # Check cache first
            cache_key = f"token_{symbol}_{exchange}"
            
            payload = {"s": symbol}  # 's' for search string as per API docs
            
            response = self.session.post(f"{self.base_url}/fetchsymbol", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    results = data.get('data', [])
                    for result in results:
                        # Match symbol with exchange
                        if (result.get('exchange') == exchange and 
                            result.get('symbol', '').upper() == f"{symbol.upper()}-EQ"):
                            return result.get('token')
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Get symbol token error for {symbol}: {e}")
            return None
    
    async def place_order(self, symbol: str, side: str, quantity: int, price: float = 0,
                         order_type: str = "MKT", product: str = "MIS", exchange: str = "NSE") -> Optional[str]:
        """Place order through Goodwill API"""
        try:
            if not self.access_token:
                logger.error("❌ Not authenticated for placing order")
                return None
            
            # Format symbol for API
            formatted_symbol = f"{symbol.upper()}-EQ"
            
            order_payload = {
                "tsym": formatted_symbol,          # Trading Symbol
                "exchange": exchange,              # Exchange (NSE, BSE, NFO, etc.)
                "trantype": side.upper(),          # B (Buy) or S (Sell)
                "validity": "DAY",                 # DAY, IOC, etc.
                "pricetype": order_type.upper(),   # MKT, L (Limit), SL-L, SL-M
                "qty": str(quantity),
                "discqty": "0",                    # Disclosed quantity
                "price": str(price) if order_type.upper() != "MKT" else "0",
                "trgprc": "0",                     # Trigger price for SL orders
                "product": product.upper(),        # MIS (Intraday), CNC (Delivery), NRML (Normal)
                "amo": "NO"                        # After Market Order
            }
            
            response = self.session.post(f"{self.base_url}/placeorder", json=order_payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    order_data = data.get('data', {})
                    order_id = order_data.get('nstordno')  # Order number
                    
                    if order_id:
                        logger.info(f"✅ Order placed: {side} {quantity} {symbol} @ {order_type} {price if order_type != 'MKT' else 'Market'} | ID: {order_id}")
                        
                        # Add to order history
                        self.order_history.append({
                            'order_id': order_id,
                            'symbol': symbol,
                            'side': side,
                            'quantity': quantity,
                            'order_type': order_type,
                            'price': price,
                            'timestamp': datetime.now(),
                            'status': 'PLACED'
                        })
                        
                        return order_id
                    else:
                        logger.error(f"❌ Order placed but no order ID returned")
                        return None
                else:
                    error_msg = data.get('error_msg', 'Order placement failed')
                    logger.error(f"❌ Order failed: {error_msg}")
                    
                    # Add to rejected orders
                    self.rejected_orders.append({
                        'symbol': symbol,
                        'error': error_msg,
                        'timestamp': datetime.now(),
                        'payload': order_payload
                    })
                    
                    return None
            else:
                logger.error(f"❌ Order HTTP error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Place order error: {e}")
            return None
    
    async def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status"""
        try:
            if not self.access_token:
                return None
            
            payload = {"order_id": order_id}
            response = self.session.post(f"{self.base_url}/orderbook", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    orders = data.get('data', [])
                    for order in orders:
                        if order.get('nstordno') == order_id:
                            return {
                                'order_id': order_id,
                                'status': order.get('status', 'UNKNOWN'),
                                'avg_price': float(order.get('avg_price', 0)),
                                'filled_quantity': int(order.get('filled_qty', 0)),
                                'remaining_quantity': int(order.get('pending_qty', 0)),
                                'order_time': order.get('order_time', ''),
                                'symbol': order.get('tsym', '').replace('-EQ', '')
                            }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Get order status error: {e}")
            return None
    
    async def get_orderbook(self) -> List[Dict]:
        """Get complete orderbook"""
        try:
            if not self.access_token:
                return []
            
            response = self.session.get(f"{self.base_url}{self.endpoints['orders']}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    orders = data.get('data', [])
                    
                    # Process and format orders
                    formatted_orders = []
                    for order in orders:
                        formatted_order = {
                            'order_id': order.get('nstordno', ''),
                            'symbol': order.get('tsym', '').replace('-EQ', ''),
                            'side': 'BUY' if order.get('trantype') == 'B' else 'SELL',
                            'quantity': int(order.get('qty', 0)),
                            'price': float(order.get('price', 0)),
                            'avg_price': float(order.get('avg_price', 0)),
                            'filled_quantity': int(order.get('filled_qty', 0)),
                            'pending_quantity': int(order.get('pending_qty', 0)),
                            'status': order.get('status', 'UNKNOWN'),
                            'order_type': order.get('pricetype', 'MKT'),
                            'product': order.get('product', 'MIS'),
                            'exchange': order.get('exchange', 'NSE'),
                            'order_time': order.get('order_time', ''),
                            'update_time': order.get('update_time', ''),
                            'validity': order.get('validity', 'DAY'),
                            'trigger_price': float(order.get('trgprc', 0))
                        }
                        formatted_orders.append(formatted_order)
                    
                    return formatted_orders
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Get orderbook error: {e}")
            return []
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            if not self.access_token:
                return False
            
            payload = {"order_id": order_id}
            response = self.session.post(f"{self.base_url}/cancelorder", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    logger.info(f"✅ Order cancelled: {order_id}")
                    return True
                else:
                    logger.error(f"❌ Cancel order failed: {data.get('error_msg', 'Unknown error')}")
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Cancel order error: {e}")
            return False
    
    async def modify_order(self, order_id: str, quantity: int = None, price: float = None,
                          order_type: str = None, trigger_price: float = None) -> bool:
        """Modify an existing order"""
        try:
            if not self.access_token:
                return False
            
            # Get existing order details first
            order_details = await self.get_order_status(order_id)
            if not order_details:
                logger.error(f"❌ Order not found for modification: {order_id}")
                return False
            
            payload = {
                "order_id": order_id,
                "qty": str(quantity) if quantity else str(order_details.get('quantity', 0)),
                "price": str(price) if price else str(order_details.get('price', 0)),
                "pricetype": order_type if order_type else order_details.get('order_type', 'MKT'),
                "trgprc": str(trigger_price) if trigger_price else "0"
            }
            
            response = self.session.post(f"{self.base_url}/modifyorder", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    logger.info(f"✅ Order modified: {order_id}")
                    return True
                else:
                    logger.error(f"❌ Modify order failed: {data.get('error_msg', 'Unknown error')}")
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Modify order error: {e}")
            return False
    
    # ==========================================
    # WEBSOCKET REAL-TIME DATA
    # ==========================================
    
    def start_websocket(self):
        """Start WebSocket connection for real-time data"""
        try:
            if self.is_ws_connected:
                logger.warning("WebSocket already connected")
                return
            
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    self._process_tick_data(data)
                except Exception as e:
                    logger.error(f"❌ WebSocket message error: {e}")
            
            def on_error(ws, error):
                logger.error(f"❌ WebSocket error: {error}")
                self.is_ws_connected = False
            
            def on_close(ws, close_status_code, close_msg):
                logger.warning(f"🔌 WebSocket closed: {close_status_code} - {close_msg}")
                self.is_ws_connected = False
            
            def on_open(ws):
                logger.info("✅ WebSocket connected")
                self.is_ws_connected = True
                
                # Subscribe to symbols
                for symbol in self.live_positions.keys():
                    self._subscribe_symbol(ws, symbol)
            
            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                header={"Authorization": f"Bearer {self.access_token}"},
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            # Start WebSocket in separate thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
        except Exception as e:
            logger.error(f"❌ WebSocket start error: {e}")
    
    def _subscribe_symbol(self, ws, symbol: str):
        """Subscribe to symbol for real-time data"""
        try:
            subscribe_msg = {
                "action": "subscribe",
                "symbols": [f"{symbol}-EQ"]
            }
            ws.send(json.dumps(subscribe_msg))
            logger.info(f"📡 Subscribed to {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Subscribe error for {symbol}: {e}")
    
    def _process_tick_data(self, tick_data: Dict):
        """Process incoming tick data"""
        try:
            symbol = tick_data.get('symbol', '').replace('-EQ', '')
            if not symbol:
                return
            
            # Update live quotes
            quote = {
                'symbol': symbol,
                'price': float(tick_data.get('ltp', 0)),
                'volume': int(tick_data.get('volume', 0)),
                'high': float(tick_data.get('high', 0)),
                'low': float(tick_data.get('low', 0)),
                'open': float(tick_data.get('open', 0)),
                'change': float(tick_data.get('change', 0)),
                'change_per': float(tick_data.get('change_per', 0)),
                'timestamp': datetime.now()
            }
            
            self.live_quotes[symbol] = quote
            
            # Update position if exists
            if symbol in self.live_positions:
                position = self.live_positions[symbol]
                position.update_position(
                    current_price=quote['price'],
                    high=quote['high'],
                    low=quote['low']
                )
                
                # Check for stop loss triggers
                self._check_stop_loss_triggers(symbol, position)
            
            # Store tick data for analysis
            if symbol not in self.tick_data:
                self.tick_data[symbol] = deque(maxlen=1000)
            
            self.tick_data[symbol].append(quote)
            
        except Exception as e:
            logger.error(f"❌ Process tick data error: {e}")
    
    def _check_stop_loss_triggers(self, symbol: str, position: LivePosition):
        """Check if stop loss should be triggered"""
        try:
            current_price = position.current_price
            stop_price = position.current_stop
            
            should_exit = False
            exit_reason = ""
            
            if position.side == 'BUY':
                if current_price <= stop_price:
                    should_exit = True
                    exit_reason = "Stop Loss Hit"
            else:  # SELL
                if current_price >= stop_price:
                    should_exit = True
                    exit_reason = "Stop Loss Hit"
            
            if should_exit:
                logger.warning(f"🚨 {exit_reason} for {symbol}: {current_price} vs {stop_price}")
                
                # Place exit order
                asyncio.create_task(self._exit_position(symbol, exit_reason))
                
        except Exception as e:
            logger.error(f"❌ Stop loss check error for {symbol}: {e}")
    
    async def _exit_position(self, symbol: str, reason: str):
        """Exit a position"""
        try:
            if symbol not in self.live_positions:
                return
            
            position = self.live_positions[symbol]
            exit_side = 'SELL' if position.side == 'BUY' else 'BUY'
            
            order_id = await self.place_order(
                symbol=symbol,
                side=exit_side,
                quantity=position.quantity,
                order_type='MKT'
            )
            
            if order_id:
                logger.info(f"✅ Exit order placed for {symbol}: {reason} | Order ID: {order_id}")
                
                # Remove from live positions
                del self.live_positions[symbol]
                
                # Calculate final P&L
                final_pnl = position.unrealized_pnl
                self.daily_pnl += final_pnl
                
                logger.info(f"💰 Final P&L for {symbol}: ₹{final_pnl:.2f} | Daily P&L: ₹{self.daily_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Exit position error for {symbol}: {e}")
    
    # ==========================================
    # HISTORICAL DATA METHODS
    # ==========================================
    
    async def get_historical_data(self, symbol: str, exchange: str = "NSE", 
                                 interval: str = "1minute", from_date: str = None, 
                                 to_date: str = None) -> List[Dict]:
        """Get historical data for symbol"""
        try:
            if not self.access_token:
                return []
            
            # Get symbol token first
            token = await self._get_symbol_token(symbol, exchange)
            if not token:
                logger.warning(f"No token found for {symbol}")
                return []
            
            # Default dates if not provided
            if not to_date:
                to_date = datetime.now().strftime('%Y-%m-%d')
            if not from_date:
                from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            payload = {
                "exchange": exchange,
                "token": token,
                "interval": interval,
                "from": from_date,
                "to": to_date
            }
            
            response = self.session.post(f"{self.base_url}{self.endpoints['historical']}", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    historical_data = data.get('data', [])
                    
                    # Format historical data
                    formatted_data = []
                    for bar in historical_data:
                        formatted_bar = {
                            'datetime': bar.get('time', ''),
                            'open': float(bar.get('open', 0)),
                            'high': float(bar.get('high', 0)),
                            'low': float(bar.get('low', 0)),
                            'close': float(bar.get('close', 0)),
                            'volume': int(bar.get('volume', 0))
                        }
                        formatted_data.append(formatted_bar)
                    
                    return formatted_data
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Get historical data error for {symbol}: {e}")
            return []
    
    # ==========================================
    # INSTRUMENTS AND WATCHLIST
    # ==========================================
    
    async def get_instruments(self, exchange: str = "NSE") -> List[Dict]:
        """Get instrument list for exchange"""
        try:
            if not self.access_token:
                return []
            
            payload = {"exchange": exchange}
            response = self.session.post(f"{self.base_url}{self.endpoints['instruments']}", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    return data.get('data', [])
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Get instruments error: {e}")
            return []
    
    async def search_instruments(self, search_term: str, exchange: str = "NSE") -> List[Dict]:
        """Search for instruments"""
        try:
            if not self.access_token:
                return []
            
            payload = {"s": search_term, "exchange": exchange}
            response = self.session.post(f"{self.base_url}/fetchsymbol", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    return data.get('data', [])
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Search instruments error: {e}")
            return []
    
    # ==========================================
    # UTILITY METHODS
    # ==========================================
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return (self.auth_state == AuthenticationState.AUTHENTICATED and 
                self.access_token is not None and
                self.session_expiry is not None and
                datetime.now() < self.session_expiry)
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            now = datetime.now()
            current_time = now.strftime('%H:%M')
            current_weekday = now.weekday()
            
            # Check if it's a weekday (Monday=0, Sunday=6)
            if current_weekday >= 5:  # Saturday or Sunday
                return False
            
            # Check market hours
            return (self.market_hours['market_open'] <= current_time <= 
                   self.market_hours['market_close'])
            
        except Exception as e:
            logger.error(f"❌ Market open check error: {e}")
            return False
    
    def get_authentication_status(self) -> Dict:
        """Get current authentication status"""
        return {
            'authenticated': self.is_authenticated(),
            'auth_state': self.auth_state.value,
            'access_token_exists': self.access_token is not None,
            'client_id': self.client_id,
            'session_expiry': self.session_expiry.isoformat() if self.session_expiry else None,
            'account_balance': self.account_balance,
            'available_margin': self.available_margin,
            'used_margin': self.used_margin,
            'daily_pnl': self.daily_pnl,
            'live_positions_count': len(self.live_positions),
            'market_open': self.is_market_open(),
            'websocket_connected': self.is_ws_connected,
            'api_base_url': self.base_url,
            'api_docs_url': self.api_docs_url
        }
    
    def get_trading_summary(self) -> Dict:
        """Get trading summary for the day"""
        try:
            total_orders = len(self.order_history)
            rejected_orders = len(self.rejected_orders)
            live_positions_count = len(self.live_positions)
            
            # Calculate total unrealized P&L
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.live_positions.values())
            
            # Calculate win/loss statistics
            executed_orders = [order for order in self.order_history if order.get('status') == 'EXECUTED']
            
            return {
                'account_balance': self.account_balance,
                'available_margin': self.available_margin,
                'used_margin': self.used_margin,
                'daily_pnl': self.daily_pnl,
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_orders': total_orders,
                'executed_orders': len(executed_orders),
                'rejected_orders': rejected_orders,
                'live_positions': live_positions_count,
                'market_status': 'OPEN' if self.is_market_open() else 'CLOSED',
                'authentication_status': self.auth_state.value,
                'websocket_status': 'CONNECTED' if self.is_ws_connected else 'DISCONNECTED',
                'api_info': {
                    'base_url': self.base_url,
                    'docs_url': self.api_docs_url,
                    'websocket_url': self.ws_url
                },
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Trading summary error: {e}")
            return {}
    
    def get_live_positions_summary(self) -> List[Dict]:
        """Get summary of all live positions"""
        try:
            positions_summary = []
            
            for symbol, position in self.live_positions.items():
                position_summary = {
                    'symbol': symbol,
                    'side': position.side,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'profit_pct': position.calculate_profit_pct(),
                    'position_value': position.position_value,
                    'stop_price': position.current_stop,
                    'bars_held': position.bars_held,
                    'entry_time': position.entry_time.isoformat(),
                    'strategy': position.strategy
                }
                positions_summary.append(position_summary)
            
            return positions_summary
            
        except Exception as e:
            logger.error(f"❌ Live positions summary error: {e}")
            return []
    
    def get_recent_orders(self, limit: int = 10) -> List[Dict]:
        """Get recent order history"""
        try:
            recent_orders = list(self.order_history)[-limit:]
            return [
                {
                    'order_id': order.get('order_id', ''),
                    'symbol': order.get('symbol', ''),
                    'side': order.get('side', ''),
                    'quantity': order.get('quantity', 0),
                    'order_type': order.get('order_type', ''),
                    'price': order.get('price', 0),
                    'status': order.get('status', ''),
                    'timestamp': order.get('timestamp', datetime.now()).isoformat()
                }
                for order in recent_orders
            ]
            
        except Exception as e:
            logger.error(f"❌ Recent orders error: {e}")
            return []
    
    def get_rejected_orders(self, limit: int = 5) -> List[Dict]:
        """Get recent rejected orders"""
        try:
            recent_rejected = list(self.rejected_orders)[-limit:]
            return [
                {
                    'symbol': order.get('symbol', ''),
                    'error': order.get('error', ''),
                    'timestamp': order.get('timestamp', datetime.now()).isoformat(),
                    'payload': order.get('payload', {})
                }
                for order in recent_rejected
            ]
            
        except Exception as e:
            logger.error(f"❌ Rejected orders error: {e}")
            return []
    
    def stop_websocket(self):
        """Stop WebSocket connection"""
        try:
            if self.ws:
                self.ws.close()
                self.is_ws_connected = False
                logger.info("🔌 WebSocket connection stopped")
            
        except Exception as e:
            logger.error(f"❌ Stop WebSocket error: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Close WebSocket connection
            self.stop_websocket()
            
            # Wait for WebSocket thread to finish
            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=5)
            
            # Clear session
            if self.session:
                self.session.close()
            
            logger.info("🧹 Goodwill API Handler cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    # ==========================================
    # SESSION MANAGEMENT
    # ==========================================
    
    def refresh_session(self) -> bool:
        """Refresh authentication session"""
        try:
            if not self.is_authenticated():
                logger.warning("⚠️ Session not authenticated, cannot refresh")
                return False
            
            # Check if session is close to expiry (within 1 hour)
            if self.session_expiry and datetime.now() + timedelta(hours=1) >= self.session_expiry:
                logger.info("🔄 Session nearing expiry, refreshing...")
                
                # Re-authenticate using stored credentials
                if self.api_key and self.secret_key and self.request_token:
                    auth_result = self._exchange_token_for_access()
                    if auth_result.get('success'):
                        logger.info("✅ Session refreshed successfully")
                        return True
                    else:
                        logger.error(f"❌ Session refresh failed: {auth_result.get('error')}")
                        return False
                else:
                    logger.error("❌ Missing credentials for session refresh")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Session refresh error: {e}")
            return False
    
    def logout(self) -> bool:
        """Logout and clear session"""
        try:
            if self.access_token:
                # Call logout API
                response = self.session.post(f"{self.base_url}{self.endpoints.get('logout', '/auth/logout')}")
                if response.status_code == 200:
                    logger.info("✅ Logged out successfully")
                else:
                    logger.warning(f"⚠️ Logout API failed: {response.status_code}")
            
            # Clear session data
            self.access_token = None
            self.client_id = None
            self.session_expiry = None
            self.auth_state = AuthenticationState.NOT_AUTHENTICATED
            
            # Clear session headers
            self.session.headers.pop('Authorization', None)
            self.session.headers.pop('X-Client-Id', None)
            
            # Stop WebSocket
            self.stop_websocket()
            
            logger.info("🔐 Session cleared")
            return True
            
        except Exception as e:
            logger.error(f"❌ Logout error: {e}")
            return False


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def create_goodwill_handler(config: Dict) -> GoodwillAPIHandler:
    """Create and return Goodwill API handler instance"""
    return GoodwillAPIHandler(config)

async def test_goodwill_connection(handler: GoodwillAPIHandler) -> Dict:
    """Test Goodwill API connection"""
    try:
        if not handler.is_authenticated():
            return {
                'success': False,
                'error': 'Not authenticated',
                'status': handler.auth_state.value,
                'next_step': 'Complete 3-step authentication process'
            }
        
        # Test profile API
        profile = await handler.get_profile()
        if profile:
            return {
                'success': True,
                'message': 'Connection successful',
                'profile': profile,
                'account_balance': handler.account_balance,
                'available_margin': handler.available_margin,
                'api_info': {
                    'base_url': handler.base_url,
                    'docs_url': handler.api_docs_url
                }
            }
        else:
            return {
                'success': False,
                'error': 'Profile API failed - check API credentials'
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def validate_trading_config(config: Dict) -> Dict:
    """Validate trading configuration"""
    required_fields = ['max_daily_loss', 'max_position_size']
    errors = []
    
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate values
    if 'max_daily_loss' in config and config['max_daily_loss'] <= 0:
        errors.append("max_daily_loss must be positive")
    
    if 'max_position_size' in config and config['max_position_size'] <= 0:
        errors.append("max_position_size must be positive")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

# ==========================================
# USAGE EXAMPLE AND TESTING
# ==========================================

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = {
        'max_daily_loss': 50000,  # 50k INR
        'max_position_size': 100000  # 1 Lakh INR
    }
    
    # Validate config
    validation = validate_trading_config(config)
    if not validation['valid']:
        print("❌ Invalid config:", validation['errors'])
        exit(1)
    
    # Create handler
    handler = create_goodwill_handler(config)
    
    # Start authentication flow
    print("🔐 Starting Goodwill API Authentication Flow")
    print("=" * 50)
    
    auth_step1 = handler.start_authentication_flow()
    print("Step 1:", auth_step1)
    
    print("\n📋 Next Steps:")
    print("1. Get your API credentials from Goodwill")
    print("2. Call: handler.submit_api_credentials(api_key, secret_key)")
    print("3. Open the login URL and get request token")
    print("4. Call: handler.submit_request_token(request_token)")
    print("5. Start trading!")
    
    print(f"\n📚 API Documentation: {handler.api_docs_url}")
    print(f"🌐 API Base URL: {handler.base_url}")
    print("\n✅ Goodwill API Handler ready for authentication!")
