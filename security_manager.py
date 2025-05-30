#!/usr/bin/env python3
"""
Production Security Manager for Trading System
Comprehensive security, authentication, authorization, and audit logging
"""

import asyncio
import hashlib
import secrets
import jwt
import bcrypt
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncpg
import aioredis
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import json
import ipaddress
from functools import wraps
import time
import re

# Import system components
from database_setup import TradingDatabase
from config_manager import ConfigManager
from utils import Logger, ErrorHandler, DataValidator
from bot_core import TradingBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('security.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security clearance levels"""
    GUEST = 1
    USER = 2
    TRADER = 3
    ADMIN = 4
    SUPER_ADMIN = 5

class ActionType(Enum):
    """Types of actions for audit logging"""
    LOGIN = "LOGIN"
    LOGOUT = "LOGOUT"
    LOGIN_FAILED = "LOGIN_FAILED"
    TRADE_PLACED = "TRADE_PLACED"
    TRADE_MODIFIED = "TRADE_MODIFIED"
    TRADE_CANCELLED = "TRADE_CANCELLED"
    PORTFOLIO_VIEWED = "PORTFOLIO_VIEWED"
    SETTINGS_CHANGED = "SETTINGS_CHANGED"
    DATA_EXPORT = "DATA_EXPORT"
    ADMIN_ACTION = "ADMIN_ACTION"
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    PASSWORD_CHANGED = "PASSWORD_CHANGED"
    API_ACCESS = "API_ACCESS"
    DATABASE_ACCESS = "DATABASE_ACCESS"

class ThreatLevel(Enum):
    """Threat detection levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class User:
    """User entity with security attributes"""
    user_id: str
    username: str
    email: str
    password_hash: str
    security_level: SecurityLevel
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] =     def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def generate_api_key(self) -> str:
        """Generate secure API key"""
        return secrets.token_urlsafe(32)
    
    def generate_session_id(self) -> str:
        """Generate secure session ID"""
        return secrets.token_urlsafe(32)

class ThreatDetector:
    """Advanced threat detection system"""
    
    def __init__(self, trading_db: TradingDatabase):
        self.trading_db = trading_db
        self.logger = Logger(__name__)
        
        # Threat detection thresholds
        self.thresholds = {
            'failed_login_attempts': 5,
            'api_calls_per_minute': 100,
            'unusual_trading_volume': 10000000,  # 1 crore
            'after_hours_activity': True,
            'geographic_anomaly': True,
            'rapid_order_placement': 50  # orders per minute
        }
    
    async def detect_login_anomalies(self, user_id: str, ip_address: str, user_agent: str) -> ThreatLevel:
        """Detect login-related security threats"""
        try:
            async with self.trading_db.pg_pool.acquire() as conn:
                # Check failed login attempts in last hour
                failed_attempts = await conn.fetchval("""
                    SELECT COUNT(*) FROM security_events 
                    WHERE user_id = $1 
                    AND action = 'LOGIN_FAILED' 
                    AND timestamp > NOW() - INTERVAL '1 hour'
                """, user_id)
                
                # Check geographic anomaly
                recent_locations = await conn.fetch("""
                    SELECT DISTINCT ip_address FROM security_events 
                    WHERE user_id = $1 
                    AND action = 'LOGIN' 
                    AND success = true 
                    AND timestamp > NOW() - INTERVAL '7 days'
                    LIMIT 10
                """, user_id)
                
                # Check device/user agent anomaly
                recent_agents = await conn.fetch("""
                    SELECT DISTINCT details->>'user_agent' as user_agent FROM security_events 
                    WHERE user_id = $1 
                    AND action = 'LOGIN' 
                    AND success = true 
                    AND timestamp > NOW() - INTERVAL '30 days'
                    LIMIT 5
                """, user_id)
                
                threat_level = ThreatLevel.LOW
                
                if failed_attempts >= self.thresholds['failed_login_attempts']:
                    threat_level = ThreatLevel.HIGH
                
                # Check IP geolocation (simplified)
                known_ips = [row['ip_address'] for row in recent_locations]
                if ip_address not in known_ips and len(known_ips) > 0:
                    threat_level = max(threat_level, ThreatLevel.MEDIUM)
                
                # Check user agent
                known_agents = [row['user_agent'] for row in recent_agents if row['user_agent']]
                if user_agent not in known_agents and len(known_agents) > 0:
                    threat_level = max(threat_level, ThreatLevel.MEDIUM)
                
                return threat_level
                
        except Exception as e:
            self.logger.error(f"Error detecting login anomalies: {e}")
            return ThreatLevel.LOW
    
    async def detect_trading_anomalies(self, user_id: str, order_value: float, order_count: int) -> ThreatLevel:
        """Detect trading-related security threats"""
        try:
            async with self.trading_db.pg_pool.acquire() as conn:
                # Check trading volume in last hour
                recent_volume = await conn.fetchval("""
                    SELECT COALESCE(SUM(CAST(details->>'order_value' AS NUMERIC)), 0) 
                    FROM security_events 
                    WHERE user_id = $1 
                    AND action = 'TRADE_PLACED' 
                    AND timestamp > NOW() - INTERVAL '1 hour'
                """, user_id)
                
                # Check order frequency
                recent_orders = await conn.fetchval("""
                    SELECT COUNT(*) FROM security_events 
                    WHERE user_id = $1 
                    AND action = 'TRADE_PLACED' 
                    AND timestamp > NOW() - INTERVAL '1 minute'
                """, user_id)
                
                threat_level = ThreatLevel.LOW
                
                if recent_volume > self.thresholds['unusual_trading_volume']:
                    threat_level = ThreatLevel.HIGH
                
                if recent_orders > self.thresholds['rapid_order_placement']:
                    threat_level = max(threat_level, ThreatLevel.CRITICAL)
                
                # Check after-hours trading
                current_hour = datetime.now().hour
                if current_hour < 9 or current_hour > 15:  # Outside market hours
                    threat_level = max(threat_level, ThreatLevel.MEDIUM)
                
                return threat_level
                
        except Exception as e:
            self.logger.error(f"Error detecting trading anomalies: {e}")
            return ThreatLevel.LOW

class SecurityManager:
    """Main security management system"""
    
    def __init__(self, trading_db: TradingDatabase, config_manager: ConfigManager):
        self.trading_db = trading_db
        self.config_manager = config_manager
        self.config = config_manager.get_config()['security']
        
        # Initialize components
        self.logger = Logger(__name__)
        self.error_handler = ErrorHandler()
        self.data_validator = DataValidator()
        
        # Security components
        self.password_policy = PasswordPolicy()
        self.encryption_manager = EncryptionManager()
        self.threat_detector = ThreatDetector(trading_db)
        
        # Redis for session management
        self.redis_client = None
        
        # Security settings
        self.jwt_secret = self.config.get('jwt_secret', secrets.token_urlsafe(32))
        self.session_timeout = self.config.get('session_timeout', 3600)
        self.max_failed_attempts = self.config.get('max_failed_attempts', 5)
        self.lockout_duration = self.config.get('lockout_duration', 300)  # 5 minutes
    
    async def initialize(self):
        """Initialize security manager"""
        try:
            # Initialize Redis for session management
            if self.config.get('redis'):
                self.redis_client = await aioredis.from_url(
                    self.config['redis']['url'],
                    encoding="utf-8",
                    decode_responses=True
                )
            
            # Create security tables if they don't exist
            await self._create_security_tables()
            
            self.logger.info("Security manager initialized successfully")
            
        except Exception as e:
            error_msg = f"Security manager initialization failed: {str(e)}"
            self.logger.error(error_msg)
            self.error_handler.handle_error(e, "security_initialization")
            raise
    
    async def _create_security_tables(self):
        """Create security-related database tables"""
        async with self.trading_db.pg_pool.acquire() as conn:
            # Users table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    security_level INTEGER NOT NULL DEFAULT 2,
                    is_active BOOLEAN DEFAULT true,
                    created_at TIMESTAMP DEFAULT NOW(),
                    last_login TIMESTAMP,
                    failed_login_attempts INTEGER DEFAULT 0,
                    account_locked BOOLEAN DEFAULT false,
                    lock_until TIMESTAMP,
                    two_factor_secret VARCHAR(255),
                    two_factor_enabled BOOLEAN DEFAULT false,
                    allowed_ips TEXT[],
                    session_timeout INTEGER DEFAULT 3600
                )
            """)
            
            # Security events table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS security_events (
                    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(user_id),
                    action VARCHAR(50) NOT NULL,
                    resource VARCHAR(255),
                    ip_address INET,
                    user_agent TEXT,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    success BOOLEAN,
                    details JSONB,
                    threat_level VARCHAR(20) DEFAULT 'LOW'
                )
            """)
            
            # Sessions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id VARCHAR(255) PRIMARY KEY,
                    user_id UUID REFERENCES users(user_id),
                    created_at TIMESTAMP DEFAULT NOW(),
                    last_accessed TIMESTAMP DEFAULT NOW(),
                    ip_address INET,
                    user_agent TEXT,
                    expires_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT true
                )
            """)
            
            # API keys table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(user_id),
                    api_key VARCHAR(255) UNIQUE NOT NULL,
                    name VARCHAR(100),
                    permissions TEXT[],
                    created_at TIMESTAMP DEFAULT NOW(),
                    last_used TIMESTAMP,
                    is_active BOOLEAN DEFAULT true,
                    expires_at TIMESTAMP
                )
            """)
            
            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_security_events_user_timestamp ON security_events(user_id, timestamp)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_security_events_action ON security_events(action)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_active ON user_sessions(user_id, is_active)")
    
    async def create_user(self, username: str, email: str, password: str, security_level: SecurityLevel = SecurityLevel.USER) -> str:
        """Create new user with security validation"""
        try:
            # Validate inputs
            if not self.data_validator.validate_email(email):
                raise ValueError("Invalid email format")
            
            if not self.data_validator.validate_username(username):
                raise ValueError("Invalid username format")
            
            # Validate password
            is_valid, errors = self.password_policy.validate_password(password, username)
            if not is_valid:
                raise ValueError(f"Password validation failed: {'; '.join(errors)}")
            
            # Hash password
            password_hash = self.encryption_manager.hash_password(password)
            
            async with self.trading_db.pg_pool.acquire() as conn:
                # Check if user already exists
                existing = await conn.fetchval(
                    "SELECT user_id FROM users WHERE username = $1 OR email = $2",
                    username, email
                )
                
                if existing:
                    raise ValueError("User already exists")
                
                # Create user
                user_id = await conn.fetchval("""
                    INSERT INTO users (username, email, password_hash, security_level)
                    VALUES ($1, $2, $3, $4)
                    RETURNING user_id
                """, username, email, password_hash, security_level.value)
                
                # Log security event
                await self.log_security_event(
                    user_id=str(user_id),
                    action=ActionType.ADMIN_ACTION,
                    resource="user_creation",
                    success=True,
                    details={"username": username, "email": email}
                )
                
                self.logger.info(f"User created successfully: {username}")
                return str(user_id)
                
        except Exception as e:
            self.logger.error(f"User creation failed: {e}")
            self.error_handler.handle_error(e, "user_creation")
            raise
    
    async def authenticate_user(self, username: str, password: str, ip_address: str, user_agent: str) -> Optional[Dict]:
        """Authenticate user with security checks"""
        try:
            async with self.trading_db.pg_pool.acquire() as conn:
                # Get user
                user = await conn.fetchrow("""
                    SELECT user_id, username, email, password_hash, security_level,
                           is_active, failed_login_attempts, account_locked, lock_until
                    FROM users WHERE username = $1
                """, username)
                
                if not user:
                    await self.log_security_event(
                        action=ActionType.LOGIN_FAILED,
                        resource="authentication",
                        ip_address=ip_address,
                        user_agent=user_agent,
                        success=False,
                        details={"reason": "user_not_found", "username": username}
                    )
                    return None
                
                # Check if account is locked
                if user['account_locked'] and user['lock_until'] and datetime.now() < user['lock_until']:
                    await self.log_security_event(
                        user_id=str(user['user_id']),
                        action=ActionType.LOGIN_FAILED,
                        resource="authentication",
                        ip_address=ip_address,
                        user_agent=user_agent,
                        success=False,
                        details={"reason": "account_locked"}
                    )
                    return None
                
                # Check if account is active
                if not user['is_active']:
                    await self.log_security_event(
                        user_id=str(user['user_id']),
                        action=ActionType.LOGIN_FAILED,
                        resource="authentication",
                        ip_address=ip_address,
                        user_agent=user_agent,
                        success=False,
                        details={"reason": "account_inactive"}
                    )
                    return None
                
                # Verify password
                if not self.encryption_manager.verify_password(password, user['password_hash']):
                    # Increment failed attempts
                    failed_attempts = user['failed_login_attempts'] + 1
                    
                    # Lock account if too many failures
                    if failed_attempts >= self.max_failed_attempts:
                        lock_until = datetime.now() + timedelta(seconds=self.lockout_duration)
                        await conn.execute("""
                            UPDATE users SET failed_login_attempts = $1, account_locked = true, lock_until = $2
                            WHERE user_id = $3
                        """, failed_attempts, lock_until, user['user_id'])
                    else:
                        await conn.execute("""
                            UPDATE users SET failed_login_attempts = $1
                            WHERE user_id = $2
                        """, failed_attempts, user['user_id'])
                    
                    await self.log_security_event(
                        user_id=str(user['user_id']),
                        action=ActionType.LOGIN_FAILED,
                        resource
    failed_login_attempts: int = 0
    account_locked: bool = False
    lock_until: Optional[datetime] = None
    two_factor_secret: Optional[str] = None
    two_factor_enabled: bool = False
    api_keys: List[str] = field(default_factory=list)
    allowed_ips: List[str] = field(default_factory=list)
    session_timeout: int = 3600  # seconds

@dataclass
class SecurityEvent:
    """Security event for audit logging"""
    event_id: str
    user_id: Optional[str]
    action: ActionType
    resource: str
    ip_address: str
    user_agent: str
    timestamp: datetime
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
    threat_level: ThreatLevel = ThreatLevel.LOW

@dataclass
class Session:
    """User session information"""
    session_id: str
    user_id: str
    created_at: datetime
    last_accessed: datetime
    ip_address: str
    user_agent: str
    expires_at: datetime
    is_active: bool = True

class PasswordPolicy:
    """Password policy enforcement"""
    
    def __init__(self):
        self.min_length = 12
        self.require_uppercase = True
        self.require_lowercase = True
        self.require_digits = True
        self.require_special = True
        self.special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        self.max_age_days = 90
        self.history_count = 12  # Remember last 12 passwords
    
    def validate_password(self, password: str, username: str = None) -> Tuple[bool, List[str]]:
        """Validate password against policy"""
        errors = []
        
        if len(password) < self.min_length:
            errors.append(f"Password must be at least {self.min_length} characters long")
        
        if self.require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.require_digits and not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")
        
        if self.require_special and not any(c in self.special_chars for c in password):
            errors.append(f"Password must contain at least one special character: {self.special_chars}")
        
        # Check for common patterns
        if username and username.lower() in password.lower():
            errors.append("Password cannot contain username")
        
        # Check for sequential characters
        if self._has_sequential_chars(password):
            errors.append("Password cannot contain sequential characters (abc, 123, etc.)")
        
        return len(errors) == 0, errors
    
    def _has_sequential_chars(self, password: str, min_seq_length: int = 3) -> bool:
        """Check for sequential characters"""
        password_lower = password.lower()
        
        for i in range(len(password_lower) - min_seq_length + 1):
            # Check for ascending sequence
            is_ascending = True
            for j in range(min_seq_length - 1):
                if ord(password_lower[i + j + 1]) != ord(password_lower[i + j]) + 1:
                    is_ascending = False
                    break
            
            if is_ascending:
                return True
            
            # Check for descending sequence
            is_descending = True
            for j in range(min_seq_length - 1):
                if ord(password_lower[i + j + 1]) != ord(password_lower[i + j]) - 1:
                    is_descending = False
                    break
            
            if is_descending:
                return True
        
        return False

class EncryptionManager:
    """Data encryption and key management"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        if master_key:
            self.fernet = Fernet(master_key)
        else:
            # Generate new key if none provided
            self.fernet = Fernet(Fernet.generate_key())
        
        self.password_salt = os.urandom(32)
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.fernet.encrypt(data.encode()).decode()
        await self.log_security_event(
                        user_id=str(user['user_id']),
                        action=ActionType.LOGIN_FAILED,
                        resource="authentication",
                        ip_address=ip_address,
                        user_agent=user_agent,
                        success=False,
                        details={"reason": "invalid_password", "failed_attempts": failed_attempts}
                    )
                    return None
                
                # Detect threats
                threat_level = await self.threat_detector.detect_login_anomalies(
                    str(user['user_id']), ip_address, user_agent
                )
                
                # Reset failed attempts on successful login
                await conn.execute("""
                    UPDATE users SET failed_login_attempts = 0, account_locked = false, 
                                   lock_until = null, last_login = NOW()
                    WHERE user_id = $1
                """, user['user_id'])
                
                # Create session
                session = await self.create_session(
                    str(user['user_id']), ip_address, user_agent
                )
                
                # Log successful login
                await self.log_security_event(
                    user_id=str(user['user_id']),
                    action=ActionType.LOGIN,
                    resource="authentication",
                    ip_address=ip_address,
                    user_agent=user_agent,
                    success=True,
                    details={"session_id": session['session_id']},
                    threat_level=threat_level
                )
                
                self.logger.info(f"User authenticated successfully: {username}")
                
                return {
                    "user_id": str(user['user_id']),
                    "username": user['username'],
                    "email": user['email'],
                    "security_level": user['security_level'],
                    "session": session,
                    "threat_level": threat_level.value
                }
                
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            self.error_handler.handle_error(e, "authentication")
            return None
    
    async def create_session(self, user_id: str, ip_address: str, user_agent: str) -> Dict:
        """Create new user session"""
        try:
            session_id = self.encryption_manager.generate_session_id()
            expires_at = datetime.now() + timedelta(seconds=self.session_timeout)
            
            async with self.trading_db.pg_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO user_sessions (session_id, user_id, ip_address, user_agent, expires_at)
                    VALUES ($1, $2, $3, $4, $5)
                """, session_id, user_id, ip_address, user_agent, expires_at)
            
            # Store in Redis if available
            if self.redis_client:
                session_data = {
                    "user_id": user_id,
                    "ip_address": ip_address,
                    "user_agent": user_agent,
                    "created_at": datetime.now().isoformat()
                }
                await self.redis_client.setex(
                    f"session:{session_id}",
                    self.session_timeout,
                    json.dumps(session_data)
                )
            
            return {
                "session_id": session_id,
                "expires_at": expires_at.isoformat(),
                "timeout": self.session_timeout
            }
            
        except Exception as e:
            self.logger.error(f"Session creation failed: {e}")
            raise
    
    async def validate_session(self, session_id: str, ip_address: str = None) -> Optional[Dict]:
        """Validate user session"""
        try:
            # Check Redis first
            if self.redis_client:
                session_data = await self.redis_client.get(f"session:{session_id}")
                if session_data:
                    session = json.loads(session_data)
                    
                    # Validate IP if provided
                    if ip_address and session.get("ip_address") != ip_address:
                        await self.invalidate_session(session_id)
                        return None
                    
                    # Extend session
                    await self.redis_client.expire(f"session:{session_id}", self.session_timeout)
                    return session
            
            # Check database
            async with self.trading_db.pg_pool.acquire() as conn:
                session = await conn.fetchrow("""
                    SELECT s.*, u.username, u.security_level
                    FROM user_sessions s
                    JOIN users u ON s.user_id = u.user_id
                    WHERE s.session_id = $1 AND s.is_active = true AND s.expires_at > NOW()
                """, session_id)
                
                if not session:
                    return None
                
                # Validate IP if provided
                if ip_address and str(session['ip_address']) != ip_address:
                    await self.invalidate_session(session_id)
                    return None
                
                # Update last accessed
                await conn.execute("""
                    UPDATE user_sessions SET last_accessed = NOW()
                    WHERE session_id = $1
                """, session_id)
                
                return {
                    "user_id": str(session['user_id']),
                    "username": session['username'],
                    "security_level": session['security_level'],
                    "ip_address": str(session['ip_address']),
                    "created_at": session['created_at'].isoformat(),
                    "last_accessed": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Session validation failed: {e}")
            return None
    
    async def invalidate_session(self, session_id: str):
        """Invalidate user session"""
        try:
            # Remove from Redis
            if self.redis_client:
                await self.redis_client.delete(f"session:{session_id}")
            
            # Mark as inactive in database
            async with self.trading_db.pg_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE user_sessions SET is_active = false
                    WHERE session_id = $1
                """, session_id)
            
            self.logger.info(f"Session invalidated: {session_id}")
            
        except Exception as e:
            self.logger.error(f"Session invalidation failed: {e}")
    
    async def generate_jwt_token(self, user_id: str, session_id: str, permissions: List[str] = None) -> str:
        """Generate JWT token for API access"""
        try:
            payload = {
                "user_id": user_id,
                "session_id": session_id,
                "permissions": permissions or [],
                "iat": datetime.utcnow(),
                "exp": datetime.utcnow() + timedelta(seconds=self.session_timeout)
            }
            
            token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
            return token
            
        except Exception as e:
            self.logger.error(f"JWT token generation failed: {e}")
            raise
    
    async def validate_jwt_token(self, token: str) -> Optional[Dict]:
        """Validate JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Validate session is still active
            session = await self.validate_session(payload["session_id"])
            if not session:
                return None
            
            return {
                "user_id": payload["user_id"],
                "session_id": payload["session_id"],
                "permissions": payload.get("permissions", []),
                "session": session
            }
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid JWT token: {e}")
            return None
        except Exception as e:
            self.logger.error(f"JWT token validation failed: {e}")
            return None
    
    async def log_security_event(self, action: ActionType, resource: str, success: bool,
                                ip_address: str = None, user_agent: str = None,
                                user_id: str = None, details: Dict = None,
                                threat_level: ThreatLevel = ThreatLevel.LOW):
        """Log security event for audit trail"""
        try:
            event_id = secrets.token_urlsafe(16)
            
            async with self.trading_db.pg_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO security_events 
                    (event_id, user_id, action, resource, ip_address, user_agent, success, details, threat_level)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, event_id, user_id, action.value, resource, ip_address, user_agent,
                    success, json.dumps(details or {}), threat_level.value)
            
            # Alert on high threat levels
            if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                self.logger.warning(f"High threat detected: {action.value} - {threat_level.value}")
            
        except Exception as e:
            self.logger.error(f"Security event logging failed: {e}")
    
    async def check_permissions(self, user_id: str, required_permission: str, resource: str = None) -> bool:
        """Check if user has required permission"""
        try:
            async with self.trading_db.pg_pool.acquire() as conn:
                user = await conn.fetchrow("""
                    SELECT security_level, is_active FROM users WHERE user_id = $1
                """, user_id)
                
                if not user or not user['is_active']:
                    return False
                
                security_level = SecurityLevel(user['security_level'])
                
                # Define permission mappings
                permission_levels = {
                    "view_portfolio": SecurityLevel.USER,
                    "place_trade": SecurityLevel.TRADER,
                    "modify_settings": SecurityLevel.TRADER,
                    "view_all_users": SecurityLevel.ADMIN,
                    "manage_users": SecurityLevel.ADMIN,
                    "system_admin": SecurityLevel.SUPER_ADMIN
                }
                
                required_level = permission_levels.get(required_permission, SecurityLevel.SUPER_ADMIN)
                
                # Log permission check
                await self.log_security_event(
                    user_id=user_id,
                    action=ActionType.API_ACCESS,
                    resource=f"permission_check:{required_permission}",
                    success=security_level.value >= required_level.value,
                    details={"permission": required_permission, "resource": resource}
                )
                
                return security_level.value >= required_level.value
                
        except Exception as e:
            self.logger.error(f"Permission check failed: {e}")
            return False
    
    async def create_api_key(self, user_id: str, name: str, permissions: List[str] = None, expires_days: int = 365) -> str:
        """Create API key for user"""
        try:
            api_key = self.encryption_manager.generate_api_key()
            expires_at = datetime.now() + timedelta(days=expires_days)
            
            async with self.trading_db.pg_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO api_keys (user_id, api_key, name, permissions, expires_at)
                    VALUES ($1, $2, $3, $4, $5)
                """, user_id, api_key, name, permissions or [], expires_at)
            
            await self.log_security_event(
                user_id=user_id,
                action=ActionType.ADMIN_ACTION,
                resource="api_key_creation",
                success=True,
                details={"name": name, "permissions": permissions}
            )
            
            return api_key
            
        except Exception as e:
            self.logger.error(f"API key creation failed: {e}")
            raise
    
    async def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """Validate API key"""
        try:
            async with self.trading_db.pg_pool.acquire() as conn:
                key_data = await conn.fetchrow("""
                    SELECT ak.*, u.username, u.security_level, u.is_active
                    FROM api_keys ak
                    JOIN users u ON ak.user_id = u.user_id
                    WHERE ak.api_key = $1 AND ak.is_active = true AND ak.expires_at > NOW()
                """, api_key)
                
                if not key_data or not key_data['is_active']:
                    return None
                
                # Update last used
                await conn.execute("""
                    UPDATE api_keys SET last_used = NOW()
                    WHERE api_key = $1
                """, api_key)
                
                return {
                    "user_id": str(key_data['user_id']),
                    "username": key_data['username'],
                    "security_level": key_data['security_level'],
                    "permissions": key_data['permissions'],
                    "key_name": key_data['name']
                }
                
        except Exception as e:
            self.logger.error(f"API key validation failed: {e}")
            return None
    
    def require_auth(self, permission: str = None):
        """Decorator for requiring authentication"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract session/token from request (implementation depends on framework)
                # This is a placeholder - actual implementation would extract from headers
                session_id = kwargs.get('session_id')
                api_key = kwargs.get('api_key')
                
                user_data = None
                
                if session_id:
                    session = await self.validate_session(session_id)
                    if session:
                        user_data = session
                
                elif api_key:
                    key_data = await self.validate_api_key(api_key)
                    if key_data:
                        user_data = key_data
                
                if not user_data:
                    raise PermissionError("Authentication required")
                
                if permission and not await self.check_permissions(user_data['user_id'], permission):
                    raise PermissionError(f"Permission denied: {permission}")
                
                # Add user data to kwargs
                kwargs['current_user'] = user_data
                
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    async def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """Change user password with validation"""
        try:
            async with self.trading_db.pg_pool.acquire() as conn:
                user = await conn.fetchrow("""
                    SELECT username, password_hash FROM users WHERE user_id = $1
                """, user_id)
                
                if not user:
                    return False
                
                # Verify old password
                if not self.encryption_manager.verify_password(old_password, user['password_hash']):
                    await self.log_security_event(
                        user_id=user_id,
                        action=ActionType.PASSWORD_CHANGED,
                        resource="password_change",
                        success=False,
                        details={"reason": "invalid_old_password"}
                    )
                    return False
                
                # Validate new password
                is_valid, errors = self.password_policy.validate_password(new_password, user['username'])
                if not is_valid:
                    await self.log_security_event(
                        user_id=user_id,
                        action=ActionType.PASSWORD_CHANGED,
                        resource="password_change",
                        success=False,
                        details={"reason": "policy_violation", "errors": errors}
                    )
                    raise ValueError(f"Password validation failed: {'; '.join(errors)}")
                
                # Hash new password
                new_password_hash = self.encryption_manager.hash_password(new_password)
                
                # Update password
                await conn.execute("""
                    UPDATE users SET password_hash = $1 WHERE user_id = $2
                """, new_password_hash, user_id)
                
                # Invalidate all existing sessions
                await conn.execute("""
                    UPDATE user_sessions SET is_active = false WHERE user_id = $1
                """, user_id)
                
                # Clear Redis sessions
                if self.redis_client:
                    sessions = await conn.fetch("""
                        SELECT session_id FROM user_sessions WHERE user_id = $1
                    """, user_id)
                    
                    for session in sessions:
                        await self.redis_client.delete(f"session:{session['session_id']}")
                
                await self.log_security_event(
                    user_id=user_id,
                    action=ActionType.PASSWORD_CHANGED,
                    resource="password_change",
                    success=True,
                    details={"sessions_invalidated": len(sessions) if 'sessions' in locals() else 0}
                )
                
                self.logger.info(f"Password changed successfully for user: {user_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Password change failed: {e}")
            self.error_handler.handle_error(e, "password_change")
            return False
    
    async def get_security_audit(self, user_id: str = None, days: int = 7) -> List[Dict]:
        """Get security audit log"""
        try:
            async with self.trading_db.pg_pool.acquire() as conn:
                query = """
                    SELECT se.*, u.username
                    FROM security_events se
                    LEFT JOIN users u ON se.user_id = u.user_id
                    WHERE se.timestamp > NOW() - INTERVAL '%s days'
                """ % days
                
                params = []
                if user_id:
                    query += " AND se.user_id = $1"
                    params.append(user_id)
                
                query += " ORDER BY se.timestamp DESC LIMIT 1000"
                
                events = await conn.fetch(query, *params)
                
                return [
                    {
                        "event_id": event['event_id'],
                        "user_id": str(event['user_id']) if event['user_id'] else None,
                        "username": event['username'],
                        "action": event['action'],
                        "resource": event['resource'],
                        "ip_address": str(event['ip_address']) if event['ip_address'] else None,
                        "timestamp": event['timestamp'].isoformat(),
                        "success": event['success'],
                        "threat_level": event['threat_level'],
                        "details": event['details']
                    }
                    for event in events
                ]
                
        except Exception as e:
            self.logger.error(f"Security audit failed: {e}")
            return []
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions and tokens"""
        try:
            async with self.trading_db.pg_pool.acquire() as conn:
                # Clean expired sessions
                deleted_sessions = await conn.fetch("""
                    DELETE FROM user_sessions 
                    WHERE expires_at < NOW() OR (is_active = false AND last_accessed < NOW() - INTERVAL '24 hours')
                    RETURNING session_id
                """)
                
                # Clean expired API keys
                deleted_keys = await conn.fetchval("""
                    UPDATE api_keys SET is_active = false 
                    WHERE expires_at < NOW() AND is_active = true
                """)
                
                # Clean old security events (keep last 90 days)
                deleted_events = await conn.fetchval("""
                    DELETE FROM security_events 
                    WHERE timestamp < NOW() - INTERVAL '90 days'
                """)
                
                self.logger.info(f"Cleanup completed: {len(deleted_sessions)} sessions, {deleted_keys} API keys, {deleted_events} events")
                
        except Exception as e:
            self.logger.error(f"Session cleanup failed: {e}")

# Example usage and integration
async def main():
    """Example usage with integrated components"""
    from config_manager import ConfigManager
    from database_setup import TradingDatabase
    
    # Initialize components
    config_manager = ConfigManager("config/config.yaml")
    trading_db = TradingDatabase(config_manager)
    await trading_db.initialize()
    
    # Initialize security manager
    security_manager = SecurityManager(trading_db, config_manager)
    await security_manager.initialize()
    
    # Example: Create user
    try:
        user_id = await security_manager.create_user(
            username="trader1",
            email="trader1@example.com",
            password="SecurePassword123!",
            security_level=SecurityLevel.TRADER
        )
        print(f"User created: {user_id}")
    except Exception as e:
        print(f"User creation failed: {e}")
    
    # Example: Authenticate user
    auth_result = await security_manager.authenticate_user(
        username="trader1",
        password="SecurePassword123!",
        ip_address="192.168.1.100",
        user_agent="TradingBot/1.0"
    )
    
    if auth_result:
        print(f"Authentication successful: {auth_result['username']}")
        print(f"Session ID: {auth_result['session']['session_id']}")
        print(f"Threat Level: {auth_result['threat_level']}")
    else:
        print("Authentication failed")
    
    # Example: Generate JWT token
    if auth_result:
        token = await security_manager.generate_jwt_token(
            auth_result['user_id'],
            auth_result['session']['session_id'],
            permissions=["view_portfolio", "place_trade"]
        )
        print(f"JWT Token: {token[:50]}...")
    
    # Example: Security audit
    audit_log = await security_manager.get_security_audit(days=1)
    print(f"Security events in last day: {len(audit_log)}")

if __name__ == "__main__":
    asyncio.run(main())
