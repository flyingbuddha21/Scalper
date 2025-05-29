#!/usr/bin/env python3
"""
Goodwill API Handler with Complete Login Flow
Handles authentication, market data, and trading operations
"""

import requests
import json
import hashlib
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
from dataclasses import dataclass
import threading
from urllib.parse import urlencode
import base64
import hmac

logger = logging.getLogger(__name__)

@dataclass
class APICredentials:
    api_key: str
    secret_key: str
    user_id: str
    password: str
    totp_secret: Optional[str] = None
    
@dataclass
class LoginSession:
    access_token: str
    request_token: str
    session_token: str
    user_id: str
    login_time: datetime
    expires_at: datetime
    is_active: bool = True

class GoodwillAPIHandler:
    """Complete Goodwill API handler with authentication and trading"""
    
    def __init__(self, credentials: APICredentials):
        self.credentials = credentials
        self.session: Optional[LoginSession] = None
        self.base_url = "https://api.goodwill.co.in"  # Replace with actual base URL
        
        # API endpoints
        self.endpoints = {
            'login': '/api/v1/auth/login',
            'request_token': '/api/v1/auth/request_token',
            'access_token': '/api/v1/auth/access_token',
            'logout': '/api/v1/auth/logout',
            'profile': '/api/v1/user/profile',
            'quotes': '/api/v1/market/quotes',
            'historical': '/api/v1/market/historical',
            'orderbook': '/api/v1/orders',
            'positions': '/api/v1/positions',
            'holdings': '/api/v1/holdings',
            'place_order': '/api/v1/orders/place',
            'modify_order': '/api/v1/orders/modify',
            'cancel_order': '/api/v1/orders/cancel',
            'margins': '/api/v1/margins',
            'instruments': '/api/v1/instruments'
        }
        
        # Session management
        self.auto_refresh = True
        self.refresh_thread = None
        self.last_heartbeat = None
        
        # Rate limiting
        self.request_count = 0
        self.request_limit = 3000  # Per minute
        self.rate_reset_time = time.time() + 60
        
        logger.info("🔐 Goodwill API Handler initialized")
    
    def generate_checksum(self, data: Dict) -> str:
        """Generate API checksum for security"""
        try:
            # Sort data by keys and create query string
            sorted_data = dict(sorted(data.items()))
            query_string = urlencode(sorted_data)
            
            # Create HMAC SHA256 signature
            signature = hmac.new(
                self.credentials.secret_key.encode(),
                query_string.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return signature
            
        except Exception as e:
            logger.error(f"❌ Checksum generation error: {e}")
            return ""
    
    def get_totp_token(self) -> Optional[str]:
        """Generate TOTP token if TOTP secret is provided"""
        try:
            if not self.credentials.totp_secret:
                return None
            
            import pyotp
            totp = pyotp.TOTP(self.credentials.totp_secret)
            return totp.now()
            
        except ImportError:
            logger.warning("⚠️ pyotp not available for TOTP generation")
            return None
        except Exception as e:
            logger.error(f"❌ TOTP generation error: {e}")
            return None
    
    def login(self) -> bool:
        """Complete login flow with request_token method"""
        try:
            logger.info("🔐 Starting Goodwill API login flow...")
            
            # Step 1: Get request token
            request_token = self._get_request_token()
            if not request_token:
                logger.error("❌ Failed to get request token")
                return False
            
            # Step 2: Authenticate with credentials
            if not self._authenticate_user(request_token):
                logger.error("❌ User authentication failed")
                return False
            
            # Step 3: Get access token
            access_token = self._get_access_token(request_token)
            if not access_token:
                logger.error("❌ Failed to get access token")
                return False
            
            # Step 4: Create session
            self._create_session(request_token, access_token)
            
            # Step 5: Verify session
            if self._verify_session():
                logger.info("✅ Goodwill API login successful")
                
                # Start session refresh thread
                self._start_session_refresh()
                return True
            else:
                logger.error("❌ Session verification failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Login flow error: {e}")
            return False
    
    def _get_request_token(self) -> Optional[str]:
        """Step 1: Get request token from API"""
        try:
            url = f"{self.base_url}{self.endpoints['request_token']}"
            
            # Prepare request data
            timestamp = str(int(time.time()))
            request_data = {
                'api_key': self.credentials.api_key,
                'timestamp': timestamp
            }
            
            # Add checksum
            request_data['checksum'] = self.generate_checksum(request_data)
            
            # Make request
            response = requests.post(url, json=request_data, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'success':
                    request_token = data.get('data', {}).get('request_token')
                    logger.info(f"✅ Request token obtained: {request_token[:10]}...")
                    return request_token
                else:
                    logger.error(f"❌ Request token error: {data.get('message', 'Unknown error')}")
                    return None
            else:
                logger.error(f"❌ Request token HTTP error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Get request token error: {e}")
            return None
    
    def _authenticate_user(self, request_token: str) -> bool:
        """Step 2: Authenticate user with credentials"""
        try:
            url = f"{self.base_url}{self.endpoints['login']}"
            
            # Prepare authentication data
            timestamp = str(int(time.time()))
            auth_data = {
                'request_token': request_token,
                'user_id': self.credentials.user_id,
                'password': hashlib.sha256(self.credentials.password.encode()).hexdigest(),
                'timestamp': timestamp
            }
            
            # Add TOTP if available
            totp_token = self.get_totp_token()
            if totp_token:
                auth_data['totp'] = totp_token
            
            # Add checksum
            auth_data['checksum'] = self.generate_checksum(auth_data)
            
            # Make authentication request
            response = requests.post(url, json=auth_data, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'success':
                    logger.info("✅ User authentication successful")
                    return True
                else:
                    error_msg = data.get('message', 'Authentication failed')
                    logger.error(f"❌ Authentication error: {error_msg}")
                    return False
            else:
                logger.error(f"❌ Authentication HTTP error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ User authentication error: {e}")
            return False
    
    def _get_access_token(self, request_token: str) -> Optional[str]:
        """Step 3: Get access token using request token"""
        try:
            url = f"{self.base_url}{self.endpoints['access_token']}"
            
            # Prepare access token request
            timestamp = str(int(time.time()))
            token_data = {
                'api_key': self.credentials.api_key,
                'request_token': request_token,
                'timestamp': timestamp
            }
            
            # Add checksum
            token_data['checksum'] = self.generate_checksum(token_data)
            
            # Make request
            response = requests.post(url, json=token_data, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'success':
                    access_token = data.get('data', {}).get('access_token')
                    logger.info(f"✅ Access token obtained: {access_token[:10]}...")
                    return access_token
                else:
                    logger.error(f"❌ Access token error: {data.get('message', 'Unknown error')}")
                    return None
            else:
                logger.error(f"❌ Access token HTTP error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Get access token error: {e}")
            return None
    
    def _create_session(self, request_token: str, access_token: str):
        """Step 4: Create authenticated session"""
        try:
            self.session = LoginSession(
                access_token=access_token,
                request_token=request_token,
                session_token=f"{request_token}:{access_token}",
                user_id=self.credentials.user_id,
                login_time=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=8),  # 8 hour session
                is_active=True
            )
            
            logger.info("✅ Session created successfully")
            
        except Exception as e:
            logger.error(f"❌ Session creation error: {e}")
    
    def _verify_session(self) -> bool:
        """Step 5: Verify session is working"""
        try:
            profile = self.get_user_profile()
            if profile and profile.get('status') == 'success':
                logger.info("✅ Session verification successful")
                return True
            else:
                logger.error("❌ Session verification failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Session verification error: {e}")
            return False
    
    def _start_session_refresh(self):
        """Start automatic session refresh"""
        if self.auto_refresh and not self.refresh_thread:
            self.refresh_thread = threading.Thread(
                target=self._session_refresh_loop,
                daemon=True,
                name="SessionRefresh"
            )
            self.refresh_thread.start()
            logger.info("🔄 Session refresh thread started")
    
    def _session_refresh_loop(self):
        """Session refresh loop"""
        while self.auto_refresh and self.session and self.session.is_active:
            try:
                # Check if session needs refresh (30 minutes before expiry)
                if datetime.now() >= (self.session.expires_at - timedelta(minutes=30)):
                    logger.info("🔄 Refreshing session...")
                    
                    if not self.refresh_session():
                        logger.error("❌ Session refresh failed")
                        break
                
                # Send heartbeat every 5 minutes
                if not self.last_heartbeat or (datetime.now() - self.last_heartbeat).seconds > 300:
                    self._send_heartbeat()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"❌ Session refresh loop error: {e}")
                time.sleep(60)
    
    def refresh_session(self) -> bool:
        """Refresh the current session"""
        try:
            if not self.session:
                return self.login()
            
            # Try to refresh with current tokens
            new_access_token = self._get_access_token(self.session.request_token)
            
            if new_access_token:
                self.session.access_token = new_access_token
                self.session.session_token = f"{self.session.request_token}:{new_access_token}"
                self.session.expires_at = datetime.now() + timedelta(hours=8)
                
                logger.info("✅ Session refreshed successfully")
                return True
            else:
                # Full re-login required
                logger.info("🔄 Full re-login required")
                return self.login()
                
        except Exception as e:
            logger.error(f"❌ Session refresh error: {e}")
            return False
    
    def _send_heartbeat(self):
        """Send heartbeat to keep session alive"""
        try:
            # Simple API call to keep session active
            self.get_user_profile()
            self.last_heartbeat = datetime.now()
            
        except Exception as e:
            logger.debug(f"Heartbeat error: {e}")
    
    def _make_authenticated_request(self, method: str, endpoint: str, 
                                  data: Optional[Dict] = None) -> Optional[Dict]:
        """Make authenticated API request"""
        try:
            if not self.session or not self.session.is_active:
                logger.error("❌ No active session for API request")
                return None
            
            # Rate limiting check
            if not self._check_rate_limit():
                logger.warning("⚠️ Rate limit exceeded")
                return None
            
            url = f"{self.base_url}{endpoint}"
            
            # Prepare headers
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.session.access_token}',
                'X-Session-Token': self.session.session_token
            }
            
            # Prepare request data
            if data is None:
                data = {}
            
            data['timestamp'] = str(int(time.time()))
            data['checksum'] = self.generate_checksum(data)
            
            # Make request
            if method.upper() == 'GET':
                response = requests.get(url, params=data, headers=headers, timeout=30)
            else:
                response = requests.post(url, json=data, headers=headers, timeout=30)
            
            self.request_count += 1
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                logger.warning("⚠️ Session expired, attempting refresh...")
                if self.refresh_session():
                    # Retry request with new session
                    return self._make_authenticated_request(method, endpoint, data)
                else:
                    logger.error("❌ Session refresh failed")
                    return None
            else:
                logger.error(f"❌ API request error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Authenticated request error: {e}")
            return None
    
    def _check_rate_limit(self) -> bool:
        """Check API rate limiting"""
        current_time = time.time()
        
        if current_time > self.rate_reset_time:
            self.request_count = 0
            self.rate_reset_time = current_time + 60
        
        return self.request_count < self.request_limit
    
    # Market Data Methods
    def get_quotes(self, symbols: List[str]) -> Optional[Dict]:
        """Get real-time quotes for symbols"""
        try:
            data = {'symbols': ','.join(symbols)}
            return self._make_authenticated_request('POST', self.endpoints['quotes'], data)
            
        except Exception as e:
            logger.error(f"❌ Get quotes error: {e}")
            return None
    
    def get_historical_data(self, symbol: str, interval: str = '1minute', 
                          from_date: str = None, to_date: str = None) -> Optional[Dict]:
        """Get historical data for symbol"""
        try:
            data = {
                'symbol': symbol,
                'interval': interval
            }
            
            if from_date:
                data['from'] = from_date
            if to_date:
                data['to'] = to_date
            
            return self._make_authenticated_request('POST', self.endpoints['historical'], data)
            
        except Exception as e:
            logger.error(f"❌ Get historical data error: {e}")
            return None
    
    def get_instruments(self, exchange: str = None) -> Optional[Dict]:
        """Get instrument list"""
        try:
            data = {}
            if exchange:
                data['exchange'] = exchange
            
            return self._make_authenticated_request('GET', self.endpoints['instruments'], data)
            
        except Exception as e:
            logger.error(f"❌ Get instruments error: {e}")
            return None
    
    # Trading Methods
    def place_order(self, symbol: str, transaction_type: str, quantity: int,
                   order_type: str = 'MARKET', price: float = 0.0,
                   product: str = 'MIS', validity: str = 'DAY') -> Optional[Dict]:
        """Place trading order"""
        try:
            order_data = {
                'symbol': symbol,
                'transaction_type': transaction_type.upper(),
                'quantity': quantity,
                'order_type': order_type.upper(),
                'product': product.upper(),
                'validity': validity.upper()
            }
            
            if order_type.upper() in ['LIMIT', 'SL', 'SL-M']:
                order_data['price'] = price
            
            logger.info(f"📝 Placing order: {symbol} {transaction_type} {quantity}")
            
            return self._make_authenticated_request('POST', self.endpoints['place_order'], order_data)
            
        except Exception as e:
            logger.error(f"❌ Place order error: {e}")
            return None
    
    def modify_order(self, order_id: str, quantity: int = None, 
                    price: float = None, order_type: str = None) -> Optional[Dict]:
        """Modify existing order"""
        try:
            modify_data = {'order_id': order_id}
            
            if quantity:
                modify_data['quantity'] = quantity
            if price:
                modify_data['price'] = price
            if order_type:
                modify_data['order_type'] = order_type.upper()
            
            return self._make_authenticated_request('POST', self.endpoints['modify_order'], modify_data)
            
        except Exception as e:
            logger.error(f"❌ Modify order error: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> Optional[Dict]:
        """Cancel existing order"""
        try:
            cancel_data = {'order_id': order_id}
            return self._make_authenticated_request('POST', self.endpoints['cancel_order'], cancel_data)
            
        except Exception as e:
            logger.error(f"❌ Cancel order error: {e}")
            return None
    
    # Portfolio Methods
    def get_positions(self) -> Optional[Dict]:
        """Get current positions"""
        try:
            return self._make_authenticated_request('GET', self.endpoints['positions'])
            
        except Exception as e:
            logger.error(f"❌ Get positions error: {e}")
            return None
    
    def get_holdings(self) -> Optional[Dict]:
        """Get holdings"""
        try:
            return self._make_authenticated_request('GET', self.endpoints['holdings'])
            
        except Exception as e:
            logger.error(f"❌ Get holdings error: {e}")
            return None
    
    def get_orderbook(self) -> Optional[Dict]:
        """Get order book"""
        try:
            return self._make_authenticated_request('GET', self.endpoints['orderbook'])
            
        except Exception as e:
            logger.error(f"❌ Get orderbook error: {e}")
            return None
    
    def get_margins(self) -> Optional[Dict]:
        """Get margin information"""
        try:
            return self._make_authenticated_request('GET', self.endpoints['margins'])
            
        except Exception as e:
            logger.error(f"❌ Get margins error: {e}")
            return None
    
    def get_user_profile(self) -> Optional[Dict]:
        """Get user profile"""
        try:
            return self._make_authenticated_request('GET', self.endpoints['profile'])
            
        except Exception as e:
            logger.error(f"❌ Get user profile error: {e}")
            return None
    
    # Session Management
    def logout(self) -> bool:
        """Logout and cleanup session"""
        try:
            if self.session:
                # Make logout request
                logout_response = self._make_authenticated_request('POST', self.endpoints['logout'])
                
                # Cleanup session
                self.session.is_active = False
                self.session = None
                
                # Stop refresh thread
                self.auto_refresh = False
                
                logger.info("✅ Logout successful")
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Logout error: {e}")
            return False
    
    def is_logged_in(self) -> bool:
        """Check if user is logged in"""
        return (self.session is not None and 
                self.session.is_active and 
                datetime.now() < self.session.expires_at)
    
    def get_session_info(self) -> Optional[Dict]:
        """Get current session information"""
        if not self.session:
            return None
        
        return {
            'user_id': self.session.user_id,
            'login_time': self.session.login_time.isoformat(),
            'expires_at': self.session.expires_at.isoformat(),
            'is_active': self.session.is_active,
            'session_duration': str(datetime.now() - self.session.login_time),
            'time_remaining': str(self.session.expires_at - datetime.now()),
            'request_count': self.request_count
        }


# Usage Example
if __name__ == "__main__":
    # Example usage
    credentials = APICredentials(
        api_key="your_api_key",
        secret_key="your_secret_key",
        user_id="your_user_id",
        password="your_password",
        totp_secret="your_totp_secret"  # Optional
    )
    
    # Initialize API handler
    api = GoodwillAPIHandler(credentials)
    
    # Login
    if api.login():
        print("✅ Login successful")
        
        # Get user profile
        profile = api.get_user_profile()
        print(f"Profile: {profile}")
        
        # Get quotes
        quotes = api.get_quotes(['RELIANCE', 'TCS', 'INFY'])
        print(f"Quotes: {quotes}")
        
        # Get positions
        positions = api.get_positions()
        print(f"Positions: {positions}")
        
        # Logout
        api.logout()
    else:
        print("❌ Login failed")
