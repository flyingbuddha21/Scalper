"""
Goodwill API Handler for Indian Market Trading Bot
Implements proper request token authentication with auto-refresh functionality
Handles all communication with Goodwill Wealth Management APIs
Base URL: https://api.gwcindia.in/v1/
"""

import logging
import aiohttp
import asyncio
import hashlib
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from urllib.parse import urlencode
import threading

from config_manager import get_config
from utils import format_currency, format_percentage

@dataclass
class LoginSession:
    """Login session data structure"""
    client_id: str
    name: str
    email: str
    access_token: str
    user_session_id: str
    enabled_exchanges: List[str]
    enabled_products: List[str]
    enabled_order_types: List[str]
    exchange_details: Dict[str, Any]
    login_time: datetime
    expires_at: datetime

@dataclass
class OrderResponse:
    """Order response from Goodwill API"""
    order_id: str
    status: str
    message: str
    exchange_order_id: Optional[str] = None
    rejection_reason: Optional[str] = None

class GoodwillAPIEndpoints:
    """
    Goodwill Wealth Management API Endpoints
    Base URL: https://api.gwcindia.in/v1/
    """
    
    BASE_URL = "https://api.gwcindia.in/v1"
    
    # Authentication Endpoints
    LOGIN_URL = f"{BASE_URL}/login"                    # GET - Redirect to login page
    LOGIN_RESPONSE = f"{BASE_URL}/login-response"       # POST - Complete login with signature
    LOGOUT = f"{BASE_URL}/logout"                      # GET - Logout
    
    # Profile & Account Endpoints
    PROFILE = f"{BASE_URL}/profile"                    # GET - User profile
    BALANCE = f"{BASE_URL}/balance"                    # GET - Account balance
    
    # Market Data Endpoints
    GET_QUOTE = f"{BASE_URL}/getquote"                 # POST - Get quote for symbol
    FETCH_SYMBOL = f"{BASE_URL}/fetchsymbol"           # POST - Search symbols
    
    # Trading Endpoints
    PLACE_ORDER = f"{BASE_URL}/placeorder"             # POST - Place regular order
    PLACE_BO_ORDER = f"{BASE_URL}/placeboorder"        # POST - Place bracket order
    PLACE_CO_ORDER = f"{BASE_URL}/placecoorder"        # POST - Place cover order
    MODIFY_ORDER = f"{BASE_URL}/modifyorder"           # POST - Modify order
    MODIFY_BO_ORDER = f"{BASE_URL}/modifyboorder"      # POST - Modify bracket order
    MODIFY_CO_ORDER = f"{BASE_URL}/modifycoorder"      # POST - Modify cover order
    CANCEL_ORDER = f"{BASE_URL}/cancelorder"           # POST - Cancel order
    EXIT_BO_ORDER = f"{BASE_URL}/exitboorder"          # POST - Exit bracket order
    EXIT_CO_ORDER = f"{BASE_URL}/exitcoorder"          # POST - Exit cover order
    
    # Portfolio Endpoints
    POSITIONS = f"{BASE_URL}/positions"                # GET - Current positions
    EXIT_POSITION = f"{BASE_URL}/exitposition"         # POST - Exit position
    HOLDINGS = f"{BASE_URL}/holdings"                  # GET - Holdings
    ORDER_BOOK = f"{BASE_URL}/orderbook"               # GET - Order book
    ORDER_HISTORY = f"{BASE_URL}/orderhistory"         # POST - Order history
    TRADE_BOOK = f"{BASE_URL}/tradebook"               # GET - Trade book
    POSITION_CONVERSION = f"{BASE_URL}/positionconversion"  # POST - Convert position
    
    # WebSocket Endpoints
    WEBSOCKET_URL = "wss://giga.gwcindia.in/NorenWSTP/"  # WebSocket connection

class GoodwillAPIHandler:
    """Comprehensive Goodwill API handler with auto-refresh authentication"""
    
    def __init__(self):
        self.config = get_config()
        self.session: Optional[aiohttp.ClientSession] = None
        self.endpoints = GoodwillAPIEndpoints()
        
        # Authentication state
        self.login_session: Optional[LoginSession] = None
        self.is_authenticated = False
        self.auto_refresh_enabled = True
        self.refresh_thread: Optional[threading.Thread] = None
        self.refresh_lock = threading.Lock()
        
        # Rate limiting
        self.rate_limiter = {
            'requests_per_minute': 0,
            'last_minute': datetime.now().minute,
            'max_requests_per_minute': 300,  # Goodwill allows 300 req/min
            'request_timestamps': []
        }
        
        # API call statistics
        self.api_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'auth_failures': 0,
            'rate_limit_hits': 0,
            'avg_response_time_ms': 0,
            'last_refresh_time': None
        }
        
        logging.info("Goodwill API Handler initialized")
    
    async def initialize(self):
        """Initialize HTTP session and try to load existing session"""
        try:
            # Create aiohttp session with optimized settings
            timeout = aiohttp.ClientTimeout(total=self.config.api.timeout)
            connector = aiohttp.TCPConnector(
                limit=50,
                limit_per_host=30,
                keepalive_timeout=300
            )
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'MomentumBot/1.0'
                }
            )
            
            # Try to load existing session
            if await self.load_session_from_file():
                logging.info("✅ Using existing valid session")
            else:
                logging.info("🔐 No valid session found. Please authenticate using 3-step process:")
                logging.info("1. Call start_login_process() to get login URL")
                logging.info("2. Open URL and login to Goodwill account") 
                logging.info("3. Call complete_login_with_request_token() with request token")
            
            logging.info("Goodwill API Handler session initialized")
            
        except Exception as e:
            logging.error(f"Failed to initialize Goodwill API Handler: {e}")
            raise
    
    def print_login_instructions(self):
        """Print detailed login instructions for users"""
        print("\n" + "=" * 80)
        print("🔐 GOODWILL API 3-STEP LOGIN PROCESS")
        print("=" * 80)
        print("STEP 1: Get Login URL")
        print("  • Call: goodwill_handler.start_login_process()")
        print("  • This will generate a login URL with your API key")
        print("")
        print("STEP 2: Browser Login")
        print("  • Open the generated URL in your browser")
        print("  • Login to your Goodwill account")
        print("  • After login, you'll be redirected to a URL like:")
        print("  • https://your-redirect-url.com?request_token=XXXXXXXXXX")
        print("")
        print("STEP 3: Complete Authentication")
        print("  • Copy the request_token from the redirect URL")
        print("  • Call: await goodwill_handler.complete_login_with_request_token('request_token')")
        print("  • Session will be saved to config/goodwill_session.json")
        print("")
        print("🔄 AUTO-REFRESH")
        print("  • Session will auto-refresh to prevent disconnection")
        print("  • Valid for 8 hours, then requires re-authentication")
        print("=" * 80)
        print("")
    
    def get_authentication_status(self) -> Dict[str, Any]:
        """Get current authentication status"""
        return {
            "is_authenticated": self.is_authenticated,
            "user_name": self.login_session.name if self.login_session else None,
            "client_id": self.login_session.client_id if self.login_session else None,
            "email": self.login_session.email if self.login_session else None,
            "login_time": self.login_session.login_time.isoformat() if self.login_session else None,
            "expires_at": self.login_session.expires_at.isoformat() if self.login_session else None,
            "time_until_expiry_minutes": (
                (self.login_session.expires_at - datetime.now()).total_seconds() / 60
                if self.login_session else 0
            ),
            "enabled_exchanges": self.login_session.enabled_exchanges if self.login_session else [],
            "enabled_products": self.login_session.enabled_products if self.login_session else [],
            "auto_refresh_active": self.auto_refresh_enabled and self.is_authenticated,
            "session_file_exists": self._check_session_file_exists()
        }
    
    def _check_session_file_exists(self) -> bool:
        """Check if session file exists"""
        try:
            import os
            return os.path.exists(os.path.join("config", "goodwill_session.json"))
        except:
            return False
    
    def start_login_process(self) -> str:
        """
        STEP 1: Generate login URL with API key
        Returns the URL that user needs to open in browser
        """
        try:
            api_key = self.config.api.goodwill_api_key
            if not api_key:
                logging.error("API key is required. Please set GOODWILL_API_KEY in config.")
                return ""
            
            # Generate login URL
            login_url = f"{self.endpoints.LOGIN_URL}?api_key={api_key}"
            
            logging.info("=" * 60)
            logging.info("🔐 GOODWILL LOGIN PROCESS - STEP 1")
            logging.info("=" * 60)
            logging.info(f"API Key: {api_key}")
            logging.info(f"API Secret: {'*' * len(self.config.api.goodwill_secret) if self.config.api.goodwill_secret else 'NOT SET'}")
            logging.info("")
            logging.info("📍 STEP 1: Open this URL in your browser:")
            logging.info(f"🌐 {login_url}")
            logging.info("")
            logging.info("📍 STEP 2: Login to your Goodwill account")
            logging.info("📍 STEP 3: After login, you'll be redirected to a URL with request_token")
            logging.info("📍 Copy the request_token and call complete_login_with_request_token()")
            logging.info("=" * 60)
            
            return login_url
            
        except Exception as e:
            logging.error(f"Error generating login URL: {e}")
            return ""
    
    async def complete_login_with_request_token(self, request_token: str) -> bool:
        """
        STEP 3: Complete authentication using request token from redirect URL
        This saves the session data as JSON file for future use
        """
        try:
            if not request_token:
                logging.error("Request token is required for authentication")
                return False
            
            logging.info("🔐 GOODWILL LOGIN PROCESS - STEP 3")
            logging.info(f"Processing request token: {request_token[:20]}...")
            
            # Generate signature for authentication
            signature = self._generate_signature(request_token)
            
            # Prepare login response payload
            login_payload = {
                "api_key": self.config.api.goodwill_api_key,
                "request_token": request_token,
                "signature": signature
            }
            
            logging.info("Sending authentication request to Goodwill API...")
            
            # Make login response API call
            async with self.session.post(self.endpoints.LOGIN_RESPONSE, json=login_payload) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('status') == 'success':
                        # Parse login session data
                        session_data = data['data']
                        
                        self.login_session = LoginSession(
                            client_id=session_data.get('clnt_id', ''),
                            name=session_data.get('name', ''),
                            email=session_data.get('email', ''),
                            access_token=session_data.get('access_token', ''),
                            user_session_id=session_data.get('usersessionid', ''),
                            enabled_exchanges=session_data.get('exarr', []),
                            enabled_products=session_data.get('prarr', []),
                            enabled_order_types=session_data.get('orarr', []),
                            exchange_details=session_data.get('exchDetail', {}),
                            login_time=datetime.now(),
                            expires_at=datetime.now() + timedelta(hours=8)  # 8-hour session
                        )
                        
                        self.is_authenticated = True
                        
                        # Save session to JSON file for future use
                        await self._save_session_to_file()
                        
                        # Start auto-refresh mechanism
                        if self.auto_refresh_enabled:
                            self._start_auto_refresh()
                        
                        logging.info("✅ AUTHENTICATION SUCCESSFUL!")
                        logging.info("=" * 60)
                        logging.info(f"👤 User: {self.login_session.name}")
                        logging.info(f"🆔 Client ID: {self.login_session.client_id}")
                        logging.info(f"📧 Email: {self.login_session.email}")
                        logging.info(f"🏦 Exchanges: {', '.join(self.login_session.enabled_exchanges)}")
                        logging.info(f"📦 Products: {', '.join(self.login_session.enabled_products)}")
                        logging.info(f"🔑 Access Token: {self.login_session.access_token[:20]}...")
                        logging.info(f"⏰ Session Expires: {self.login_session.expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
                        logging.info("💾 Session saved to: goodwill_session.json")
                        logging.info("=" * 60)
                        
                        return True
                    else:
                        error_msg = data.get('error_msg', 'Unknown error')
                        error_type = data.get('error_type', 'Unknown')
                        logging.error(f"❌ Authentication failed: {error_msg} (Type: {error_type})")
                        
                        # Common error solutions
                        if "Invalid Signature" in error_msg:
                            logging.error("💡 Solution: Check if API Secret is correct in config")
                        elif "Invalid API key" in error_msg:
                            logging.error("💡 Solution: Check if API Key is correct and not expired")
                        elif "Invalid Request Token" in error_msg:
                            logging.error("💡 Solution: Make sure request_token is copied correctly from redirect URL")
                        
                        return False
                else:
                    error_text = await response.text()
                    logging.error(f"❌ Authentication HTTP error {response.status}: {error_text}")
                    return False
                    
        except Exception as e:
            logging.error(f"❌ Error during authentication: {e}")
            return False
    
    async def _save_session_to_file(self):
        """Save current session data to JSON file"""
        try:
            if not self.login_session:
                return
            
            session_data = {
                "client_id": self.login_session.client_id,
                "name": self.login_session.name,
                "email": self.login_session.email,
                "access_token": self.login_session.access_token,
                "user_session_id": self.login_session.user_session_id,
                "enabled_exchanges": self.login_session.enabled_exchanges,
                "enabled_products": self.login_session.enabled_products,
                "enabled_order_types": self.login_session.enabled_order_types,
                "exchange_details": self.login_session.exchange_details,
                "login_time": self.login_session.login_time.isoformat(),
                "expires_at": self.login_session.expires_at.isoformat(),
                "api_key": self.config.api.goodwill_api_key,
                "saved_at": datetime.now().isoformat()
            }
            
            import os
            session_file = os.path.join("config", "goodwill_session.json")
            
            # Create config directory if it doesn't exist
            os.makedirs("config", exist_ok=True)
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            logging.info(f"💾 Session saved to: {session_file}")
            
        except Exception as e:
            logging.error(f"Error saving session to file: {e}")
    
    async def load_session_from_file(self) -> bool:
        """Load session data from JSON file if available and valid"""
        try:
            import os
            session_file = os.path.join("config", "goodwill_session.json")
            
            if not os.path.exists(session_file):
                logging.info("No saved session file found")
                return False
            
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Check if session is still valid
            expires_at = datetime.fromisoformat(session_data['expires_at'])
            if datetime.now() >= expires_at:
                logging.warning("Saved session has expired")
                return False
            
            # Check if API key matches
            if session_data.get('api_key') != self.config.api.goodwill_api_key:
                logging.warning("Saved session API key doesn't match current config")
                return False
            
            # Restore session
            self.login_session = LoginSession(
                client_id=session_data['client_id'],
                name=session_data['name'],
                email=session_data['email'],
                access_token=session_data['access_token'],
                user_session_id=session_data['user_session_id'],
                enabled_exchanges=session_data['enabled_exchanges'],
                enabled_products=session_data['enabled_products'],
                enabled_order_types=session_data['enabled_order_types'],
                exchange_details=session_data['exchange_details'],
                login_time=datetime.fromisoformat(session_data['login_time']),
                expires_at=expires_at
            )
            
            # Validate session by making a test API call
            if await self._validate_session():
                self.is_authenticated = True
                
                # Start auto-refresh mechanism
                if self.auto_refresh_enabled:
                    self._start_auto_refresh()
                
                logging.info("✅ LOADED SAVED SESSION!")
                logging.info(f"👤 User: {self.login_session.name} ({self.login_session.client_id})")
                logging.info(f"⏰ Expires: {self.login_session.expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
                return True
            else:
                logging.warning("Saved session is invalid, please re-authenticate")
                return False
                
        except Exception as e:
            logging.error(f"Error loading session from file: {e}")
            return False
    
    def _generate_signature(self, request_token: str) -> str:
        """Generate SHA-256 signature for authentication"""
        try:
            api_key = self.config.api.goodwill_api_key
            api_secret = self.config.api.goodwill_secret
            
            # Create checksum: API Key + request_token + API Secret
            checksum = f"{api_key}{request_token}{api_secret}"
            
            # Generate SHA-256 signature
            signature = hashlib.sha256(checksum.encode()).hexdigest()
            
            logging.info(f"Generated signature for request_token: {request_token[:10]}...")
            return signature
            
        except Exception as e:
            logging.error(f"Error generating signature: {e}")
            return ""
    
    def _start_auto_refresh(self):
        """Start auto-refresh mechanism in background thread"""
        try:
            if self.refresh_thread and self.refresh_thread.is_alive():
                return
            
            self.refresh_thread = threading.Thread(
                target=self._auto_refresh_loop,
                daemon=True,
                name="GoodwillAutoRefresh"
            )
            self.refresh_thread.start()
            
            logging.info("Auto-refresh mechanism started")
            
        except Exception as e:
            logging.error(f"Error starting auto-refresh: {e}")
    
    def _auto_refresh_loop(self):
        """Auto-refresh loop to maintain authentication"""
        while self.auto_refresh_enabled and self.is_authenticated:
            try:
                if not self.login_session:
                    break
                
                # Calculate time until token expires
                time_until_expiry = (self.login_session.expires_at - datetime.now()).total_seconds()
                
                # Refresh 30 minutes before expiry
                if time_until_expiry <= 1800:  # 30 minutes
                    logging.info("Access token nearing expiry, attempting refresh...")
                    
                    with self.refresh_lock:
                        # Check if we need to re-authenticate
                        # Note: Goodwill doesn't have explicit refresh token endpoint
                        # Need to implement session validation
                        if not asyncio.run(self._validate_session()):
                            logging.warning("Session validation failed, re-authentication required")
                            self.is_authenticated = False
                            break
                        else:
                            # Extend session expiry
                            self.login_session.expires_at = datetime.now() + timedelta(hours=8)
                            self.api_stats['last_refresh_time'] = datetime.now()
                            logging.info("Session refreshed successfully")
                
                # Sleep for 5 minutes before next check
                time.sleep(300)
                
            except Exception as e:
                logging.error(f"Error in auto-refresh loop: {e}")
                time.sleep(60)  # Sleep 1 minute on error
    
    async def _validate_session(self) -> bool:
        """Validate current session by calling profile endpoint"""
        try:
            if not self.is_authenticated or not self.login_session:
                return False
            
            response = await self._make_authenticated_request('GET', self.endpoints.PROFILE)
            return response is not None and response.get('status') == 'success'
            
        except Exception as e:
            logging.error(f"Error validating session: {e}")
            return False
    
    async def _make_authenticated_request(self, method: str, url: str, 
                                        data: Dict = None, params: Dict = None) -> Optional[Dict]:
        """Make authenticated API request with rate limiting and error handling"""
        try:
            # Check authentication
            if not self.is_authenticated or not self.login_session:
                logging.error("Not authenticated. Please authenticate first.")
                return None
            
            # Check rate limiting
            if not await self._check_rate_limit():
                logging.warning("Rate limit exceeded, waiting...")
                await asyncio.sleep(1)
                return None
            
            # Prepare headers
            headers = {
                'x-api-key': self.config.api.goodwill_api_key,
                'Authorization': f'Bearer {self.login_session.access_token}'
            }
            
            # Record request start time
            start_time = time.time()
            
            # Make request
            async with self.session.request(
                method=method,
                url=url,
                json=data if method in ['POST', 'PUT'] else None,
                params=params,
                headers=headers
            ) as response:
                
                # Calculate response time
                response_time = (time.time() - start_time) * 1000
                self._update_response_time(response_time)
                
                # Update statistics
                self.api_stats['total_requests'] += 1
                
                if response.status == 200:
                    result = await response.json()
                    self.api_stats['successful_requests'] += 1
                    return result
                
                elif response.status == 401:
                    # Authentication failed
                    logging.error("Authentication failed, token may be expired")
                    self.api_stats['auth_failures'] += 1
                    self.is_authenticated = False
                    return None
                
                elif response.status == 429:
                    # Rate limited
                    logging.warning("Rate limited by API")
                    self.api_stats['rate_limit_hits'] += 1
                    return None
                
                else:
                    # Other error
                    error_text = await response.text()
                    logging.error(f"API request failed {response.status}: {error_text}")
                    self.api_stats['failed_requests'] += 1
                    return None
                    
        except Exception as e:
            logging.error(f"Error making authenticated request to {url}: {e}")
            self.api_stats['failed_requests'] += 1
            return None
    
    async def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        current_minute = datetime.now().minute
        current_time = time.time()
        
        # Reset counter if minute changed
        if current_minute != self.rate_limiter['last_minute']:
            self.rate_limiter['requests_per_minute'] = 0
            self.rate_limiter['last_minute'] = current_minute
            self.rate_limiter['request_timestamps'] = []
        
        # Remove timestamps older than 1 minute
        one_minute_ago = current_time - 60
        self.rate_limiter['request_timestamps'] = [
            ts for ts in self.rate_limiter['request_timestamps'] if ts > one_minute_ago
        ]
        
        # Check if we're within limits
        if len(self.rate_limiter['request_timestamps']) >= self.rate_limiter['max_requests_per_minute']:
            return False
        
        # Add current timestamp
        self.rate_limiter['request_timestamps'].append(current_time)
        self.rate_limiter['requests_per_minute'] += 1
        
        return True
    
    def _update_response_time(self, response_time_ms: float):
        """Update average response time statistics"""
        current_avg = self.api_stats['avg_response_time_ms']
        total_requests = self.api_stats['total_requests']
        
        if total_requests == 0:
            self.api_stats['avg_response_time_ms'] = response_time_ms
        else:
            # Calculate rolling average
            self.api_stats['avg_response_time_ms'] = (
                (current_avg * (total_requests - 1) + response_time_ms) / total_requests
            )
    
    # Market Data Methods
    async def get_quote(self, exchange: str, token: str) -> Optional[Dict[str, Any]]:
        """Get live quote for a symbol"""
        try:
            payload = {
                "exchange": exchange,
                "token": token
            }
            
            response = await self._make_authenticated_request('POST', self.endpoints.GET_QUOTE, payload)
            
            if response and response.get('status') == 'success':
                return response['data']
            
            return None
            
        except Exception as e:
            logging.error(f"Error getting quote for {exchange}:{token}: {e}")
            return None
    
    async def search_symbols(self, search_term: str) -> List[Dict[str, Any]]:
        """Search for trading symbols"""
        try:
            payload = {
                "s": search_term
            }
            
            response = await self._make_authenticated_request('POST', self.endpoints.FETCH_SYMBOL, payload)
            
            if response and response.get('status') == 'success':
                return response['data']
            
            return []
            
        except Exception as e:
            logging.error(f"Error searching symbols for '{search_term}': {e}")
            return []
    
    # Trading Methods
    async def place_order(self, order_data: Dict[str, Any]) -> Optional[OrderResponse]:
        """Place a trading order"""
        try:
            # Validate required fields
            required_fields = ['tsym', 'exchange', 'trantype', 'validity', 'pricetype', 'qty', 'price', 'product']
            for field in required_fields:
                if field not in order_data:
                    logging.error(f"Missing required field '{field}' in order data")
                    return None
            
            response = await self._make_authenticated_request('POST', self.endpoints.PLACE_ORDER, order_data)
            
            if response and response.get('status') == 'success':
                order_id = response['data'].get('nstordno', '')
                return OrderResponse(
                    order_id=order_id,
                    status='submitted',
                    message='Order placed successfully'
                )
            else:
                error_msg = response.get('error_msg', 'Unknown error') if response else 'No response'
                return OrderResponse(
                    order_id='',
                    status='rejected',
                    message=error_msg,
                    rejection_reason=error_msg
                )
                
        except Exception as e:
            logging.error(f"Error placing order: {e}")
            return OrderResponse(
                order_id='',
                status='error',
                message=str(e)
            )
    
    async def place_bracket_order(self, order_data: Dict[str, Any]) -> Optional[OrderResponse]:
        """Place a bracket order (BO)"""
        try:
            required_fields = ['exchange', 'tsym', 'trantype', 'qty', 'validity', 'price', 'squareoff', 'stoploss', 'pricetype']
            for field in required_fields:
                if field not in order_data:
                    logging.error(f"Missing required field '{field}' in bracket order data")
                    return None
            
            response = await self._make_authenticated_request('POST', self.endpoints.PLACE_BO_ORDER, order_data)
            
            if response and response.get('status') == 'success':
                order_id = response['data'].get('nstordno', '')
                return OrderResponse(
                    order_id=order_id,
                    status='submitted',
                    message='Bracket order placed successfully'
                )
            else:
                error_msg = response.get('error_msg', 'Unknown error') if response else 'No response'
                return OrderResponse(
                    order_id='',
                    status='rejected',
                    message=error_msg,
                    rejection_reason=error_msg
                )
                
        except Exception as e:
            logging.error(f"Error placing bracket order: {e}")
            return OrderResponse(
                order_id='',
                status='error',
                message=str(e)
            )
    
    async def modify_order(self, order_data: Dict[str, Any]) -> Optional[OrderResponse]:
        """Modify an existing order"""
        try:
            required_fields = ['exchange', 'tsym', 'nstordno', 'trantype', 'pricetype', 'price', 'qty']
            for field in required_fields:
                if field not in order_data:
                    logging.error(f"Missing required field '{field}' in modify order data")
                    return None
            
            response = await self._make_authenticated_request('POST', self.endpoints.MODIFY_ORDER, order_data)
            
            if response and response.get('status') == 'success':
                return OrderResponse(
                    order_id=order_data['nstordno'],
                    status='modified',
                    message='Order modified successfully'
                )
            else:
                error_msg = response.get('error_msg', 'Unknown error') if response else 'No response'
                return OrderResponse(
                    order_id=order_data['nstordno'],
                    status='modify_failed',
                    message=error_msg,
                    rejection_reason=error_msg
                )
                
        except Exception as e:
            logging.error(f"Error modifying order: {e}")
            return OrderResponse(
                order_id=order_data.get('nstordno', ''),
                status='error',
                message=str(e)
            )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            payload = {
                "nstordno": order_id
            }
            
            response = await self._make_authenticated_request('POST', self.endpoints.CANCEL_ORDER, payload)
            
            if response and response.get('status') == 'success':
                logging.info(f"Order {order_id} cancelled successfully")
                return True
            else:
                error_msg = response.get('error_msg', 'Unknown error') if response else 'No response'
                logging.error(f"Failed to cancel order {order_id}: {error_msg}")
                return False
                
        except Exception as e:
            logging.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    # Portfolio Methods
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        try:
            response = await self._make_authenticated_request('GET', self.endpoints.POSITIONS)
            
            if response and response.get('status') == 'success':
                return response.get('data', [])
            
            return []
            
        except Exception as e:
            logging.error(f"Error getting positions: {e}")
            return []
    
    async def get_holdings(self) -> List[Dict[str, Any]]:
        """Get current holdings"""
        try:
            response = await self._make_authenticated_request('GET', self.endpoints.HOLDINGS)
            
            if response and response.get('status') == 'success':
                return response.get('data', [])
            
            return []
            
        except Exception as e:
            logging.error(f"Error getting holdings: {e}")
            return []
    
    async def get_order_book(self) -> List[Dict[str, Any]]:
        """Get order book"""
        try:
            response = await self._make_authenticated_request('GET', self.endpoints.ORDER_BOOK)
            
            if response and response.get('status') == 'success':
                return response.get('data', [])
            
            return []
            
        except Exception as e:
            logging.error(f"Error getting order book: {e}")
            return []
    
    async def get_trade_book(self) -> List[Dict[str, Any]]:
        """Get trade book"""
        try:
            response = await self._make_authenticated_request('GET', self.endpoints.TRADE_BOOK)
            
            if response and response.get('status') == 'success':
                return response.get('data', [])
            
            return []
            
        except Exception as e:
            logging.error(f"Error getting trade book: {e}")
            return []
    
    async def get_balance(self) -> Optional[Dict[str, Any]]:
        """Get account balance"""
        try:
            response = await self._make_authenticated_request('GET', self.endpoints.BALANCE)
            
            if response and response.get('status') == 'success':
                return response['data']
            
            return None
            
        except Exception as e:
            logging.error(f"Error getting balance: {e}")
            return None
    
    async def exit_position(self, position_data: Dict[str, Any]) -> bool:
        """Exit a position"""
        try:
            required_fields = ['segment', 'product', 'netqty', 'token', 'tsym']
            for field in required_fields:
                if field not in position_data:
                    logging.error(f"Missing required field '{field}' in exit position data")
                    return False
            
            response = await self._make_authenticated_request('POST', self.endpoints.EXIT_POSITION, position_data)
            
            if response and response.get('status') == 'success':
                logging.info(f"Position exited successfully: {position_data['tsym']}")
                return True
            else:
                error_msg = response.get('error_msg', 'Unknown error') if response else 'No response'
                logging.error(f"Failed to exit position {position_data['tsym']}: {error_msg}")
                return False
                
        except Exception as e:
            logging.error(f"Error exiting position: {e}")
            return False
    
    async def get_order_updates(self) -> List[Dict[str, Any]]:
        """Get order status updates (simulated for REST API)"""
        try:
            # Since Goodwill uses WebSocket for real-time updates,
            # this method fetches order book and checks for status changes
            order_book = await self.get_order_book()
            return order_book
            
        except Exception as e:
            logging.error(f"Error getting order updates: {e}")
            return []
    
    async def logout(self) -> bool:
        """Logout and invalidate session"""
        try:
            response = await self._make_authenticated_request('GET', self.endpoints.LOGOUT)
            
            if response and response.get('status') == 'success':
                self.is_authenticated = False
                self.login_session = None
                self.auto_refresh_enabled = False
                
                logging.info("Successfully logged out")
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Error during logout: {e}")
            return False
    
    def get_login_url(self) -> str:
        """Get login URL for manual authentication"""
        return f"{self.endpoints.LOGIN_URL}?api_key={self.config.api.goodwill_api_key}"
    
    def get_api_statistics(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        return {
            **self.api_stats,
            'is_authenticated': self.is_authenticated,
            'session_expires_at': self.login_session.expires_at.isoformat() if self.login_session else None,
            'current_user': self.login_session.name if self.login_session else None,
            'client_id': self.login_session.client_id if self.login_session else None,
            'rate_limiter_status': {
                'requests_this_minute': self.rate_limiter['requests_per_minute'],
                'max_requests_per_minute': self.rate_limiter['max_requests_per_minute']
            }
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Stop auto-refresh
            self.auto_refresh_enabled = False
            
            # Logout if authenticated
            if self.is_authenticated:
                await self.logout()
            
            # Close session
            if self.session:
                await self.session.close()
            
            logging.info("Goodwill API Handler cleaned up successfully")
            
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

# Global instance
goodwill_handler = GoodwillAPIHandler()

def get_goodwill_handler() -> GoodwillAPIHandler:
    """Get the global Goodwill API handler instance"""
    return goodwill_handler
