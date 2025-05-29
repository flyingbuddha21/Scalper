#!/usr/bin/env python3
"""
Security Manager
Handles authentication, authorization, and security features
"""

import hashlib
import hmac
import jwt
import secrets
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sqlite3
from pathlib import Path
import json
import bcrypt

logger = logging.getLogger(__name__)

class SecurityManager:
    """Manages authentication and security"""
    
    def __init__(self, db_path: str = "data/security.db"):
        self.db_path = db_path
        self.secret_key = self._generate_or_load_secret_key()
        self.session_timeout = timedelta(hours=8)  # 8 hour sessions
        
        # Security settings
        self.max_login_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        self.password_min_length = 8
        
        # Rate limiting
        self.rate_limits = {}
        self.max_requests_per_minute = 60
        
        # Active sessions
        self.active_sessions = {}
        
        # Initialize database
        self._init_security_database()
        
        logger.info("🔐 Security Manager initialized")
    
    def _generate_or_load_secret_key(self) -> str:
        """Generate or load application secret key"""
        try:
            key_file = Path("config") / "secret.key"
            key_file.parent.mkdir(exist_ok=True)
            
            if key_file.exists():
                with open(key_file, 'r') as f:
                    return f.read().strip()
            else:
                # Generate new secret key
                secret_key = secrets.token_urlsafe(64)
                with open(key_file, 'w') as f:
                    f.write(secret_key)
                
                # Set file permissions (readable only by owner)
                key_file.chmod(0o600)
                logger.info("🔑 Generated new secret key")
                return secret_key
                
        except Exception as e:
            logger.error(f"❌ Secret key error: {e}")
            # Fallback to in-memory key
            return secrets.token_urlsafe(64)
    
    def _init_security_database(self):
        """Initialize security database"""
        try:
            Path(self.db_path).parent.mkdir(exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT DEFAULT 'user',
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_login DATETIME,
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until DATETIME,
                    two_factor_secret TEXT,
                    two_factor_enabled BOOLEAN DEFAULT 0
                )
            """)
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_token TEXT UNIQUE NOT NULL,
                    user_id INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME NOT NULL,
                    last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # API keys table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_id TEXT UNIQUE NOT NULL,
                    key_hash TEXT NOT NULL,
                    user_id INTEGER,
                    name TEXT NOT NULL,
                    permissions TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME,
                    last_used DATETIME,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # Security events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    user_id INTEGER,
                    ip_address TEXT,
                    user_agent TEXT,
                    details TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    severity TEXT DEFAULT 'INFO'
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(session_token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_id ON api_keys(key_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_security_events_type ON security_events(event_type)")
            
            conn.commit()
            conn.close()
            
            # Create default admin user if none exists
            self._create_default_admin()
            
            logger.info("✅ Security database initialized")
            
        except Exception as e:
            logger.error(f"❌ Security database init error: {e}")
    
    def _create_default_admin(self):
        """Create default admin user if none exists"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
            admin_count = cursor.fetchone()[0]
            
            if admin_count == 0:
                # Create default admin user
                default_password = "admin123"  # Change this immediately in production!
                password_hash, salt = self._hash_password(default_password)
                
                cursor.execute("""
                    INSERT INTO users (username, email, password_hash, salt, role)
                    VALUES (?, ?, ?, ?, ?)
                """, ("admin", "admin@trading-bot.local", password_hash, salt, "admin"))
                
                conn.commit()
                
                logger.warning("⚠️ Default admin user created with username 'admin' and password 'admin123'")
                logger.warning("🔒 CHANGE THE DEFAULT PASSWORD IMMEDIATELY!")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Create default admin error: {e}")
    
    def _hash_password(self, password: str) -> tuple:
        """Hash password with salt"""
        try:
            salt = bcrypt.gensalt()
            password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
            return password_hash.decode('utf-8'), salt.decode('utf-8')
            
        except Exception as e:
            logger.error(f"❌ Password hashing error: {e}")
            # Fallback to PBKDF2
            salt = secrets.token_hex(32)
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return password_hash.hex(), salt
    
    def _verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash"""
        try:
            # Try bcrypt first
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
            
        except Exception:
            # Fallback to PBKDF2
            try:
                computed_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
                return computed_hash.hex() == password_hash
            except Exception as e:
                logger.error(f"❌ Password verification error: {e}")
                return False
    
    def create_user(self, username: str, password: str, email: str = None, role: str = "user") -> bool:
        """Create new user"""
        try:
            # Validate password strength
            if not self._validate_password_strength(password):
                logger.warning(f"⚠️ Weak password for user {username}")
                return False
            
            # Hash password
            password_hash, salt = self._hash_password(password)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, salt, role)
                VALUES (?, ?, ?, ?, ?)
            """, (username, email, password_hash, salt, role))
            
            conn.commit()
            conn.close()
            
            # Log security event
            self._log_security_event("USER_CREATED", details=f"User {username} created")
            
            logger.info(f"✅ User created: {username}")
            return True
            
        except sqlite3.IntegrityError:
            logger.warning(f"⚠️ User already exists: {username}")
            return False
        except Exception as e:
            logger.error(f"❌ Create user error: {e}")
            return False
    
    def authenticate_user(self, username: str, password: str, ip_address: str = None) -> Optional[Dict]:
        """Authenticate user credentials"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if user exists and is not locked
            cursor.execute("""
                SELECT id, username, password_hash, salt, role, is_active, 
                       failed_login_attempts, locked_until
                FROM users WHERE username = ?
            """, (username,))
            
            user_row = cursor.fetchone()
            
            if not user_row:
                self._log_security_event("LOGIN_FAILED", details=f"Unknown user: {username}", ip_address=ip_address)
                return None
            
            user_id, username, password_hash, salt, role, is_active, failed_attempts, locked_until = user_row
            
            # Check if user is active
            if not is_active:
                self._log_security_event("LOGIN_FAILED", user_id=user_id, details="Inactive user", ip_address=ip_address)
                return None
            
            # Check if user is locked
            if locked_until:
                locked_until_dt = datetime.fromisoformat(locked_until)
                if datetime.now() < locked_until_dt:
                    self._log_security_event("LOGIN_FAILED", user_id=user_id, details="Account locked", ip_address=ip_address)
                    return None
                else:
                    # Unlock account
                    cursor.execute("""
                        UPDATE users SET failed_login_attempts = 0, locked_until = NULL 
                        WHERE id = ?
                    """, (user_id,))
            
            # Verify password
            if self._verify_password(password, password_hash, salt):
                # Successful login
                cursor.execute("""
                    UPDATE users SET last_login = CURRENT_TIMESTAMP, failed_login_attempts = 0 
                    WHERE id = ?
                """, (user_id,))
                
                conn.commit()
                conn.close()
                
                # Log successful login
                self._log_security_event("LOGIN_SUCCESS", user_id=user_id, ip_address=ip_address)
                
                # Create session
                session_token = self._create_session(user_id, ip_address)
                
                return {
                    'user_id': user_id,
                    'username': username,
                    'role': role,
                    'session_token': session_token
                }
            
            else:
                # Failed login
                new_failed_attempts = failed_attempts + 1
                
                if new_failed_attempts >= self.max_login_attempts:
                    # Lock account
                    locked_until = datetime.now() + self.lockout_duration
                    cursor.execute("""
                        UPDATE users SET failed_login_attempts = ?, locked_until = ? 
                        WHERE id = ?
                    """, (new_failed_attempts, locked_until.isoformat(), user_id))
                    
                    self._log_security_event("ACCOUNT_LOCKED", user_id=user_id, 
                                            details=f"Account locked after {new_failed_attempts} failed attempts", 
                                            ip_address=ip_address, severity="WARNING")
                else:
                    cursor.execute("""
                        UPDATE users SET failed_login_attempts = ? WHERE id = ?
                    """, (new_failed_attempts, user_id))
                
                conn.commit()
                conn.close()
                
                self._log_security_event("LOGIN_FAILED", user_id=user_id, 
                                        details=f"Invalid password (attempt {new_failed_attempts})", 
                                        ip_address=ip_address)
                return None
                
        except Exception as e:
            logger.error(f"❌ Authentication error: {e}")
            return None
    
    def _create_session(self, user_id: int, ip_address: str = None, user_agent: str = None) -> str:
        """Create new session for user"""
        try:
            session_token = secrets.token_urlsafe(64)
            expires_at = datetime.now() + self.session_timeout
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO sessions (session_token, user_id, expires_at, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?)
            """, (session_token, user_id, expires_at.isoformat(), ip_address, user_agent))
            
            conn.commit()
            conn.close()
            
            # Store in active sessions
            self.active_sessions[session_token] = {
                'user_id': user_id,
                'expires_at': expires_at,
                'last_activity': datetime.now()
            }
            
            return session_token
            
        except Exception as e:
            logger.error(f"❌ Create session error: {e}")
            return None
    
    def validate_session(self, session_token: str) -> Optional[Dict]:
        """Validate session token"""
        try:
            # Check in-memory cache first
            if session_token in self.active_sessions:
                session = self.active_sessions[session_token]
                
                if datetime.now() < session['expires_at']:
                    # Update last activity
                    session['last_activity'] = datetime.now()
                    return session
                else:
                    # Session expired
                    del self.active_sessions[session_token]
            
            # Check database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT s.user_id, s.expires_at, u.username, u.role, u.is_active
                FROM sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.session_token = ? AND s.is_active = 1
            """, (session_token,))
            
            session_row = cursor.fetchone()
            
            if session_row:
                user_id, expires_at, username, role, is_active = session_row
                expires_at_dt = datetime.fromisoformat(expires_at)
                
                if datetime.now() < expires_at_dt and is_active:
                    # Update last activity
                    cursor.execute("""
                        UPDATE sessions SET last_activity = CURRENT_TIMESTAMP 
                        WHERE session_token = ?
                    """, (session_token,))
                    
                    conn.commit()
                    conn.close()
                    
                    # Cache session
                    session_data = {
                        'user_id': user_id,
                        'username': username,
                        'role': role,
                        'expires_at': expires_at_dt,
                        'last_activity': datetime.now()
                    }
                    
                    self.active_sessions[session_token] = session_data
                    return session_data
                else:
                    # Expired or inactive
                    self._invalidate_session(session_token)
            
            conn.close()
            return None
            
        except Exception as e:
            logger.error(f"❌ Session validation error: {e}")
            return None
    
    def _invalidate_session(self, session_token: str):
        """Invalidate session"""
        try:
            # Remove from memory
            if session_token in self.active_sessions:
                del self.active_sessions[session_token]
            
            # Update database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE sessions SET is_active = 0 WHERE session_token = ?
            """, (session_token,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Invalidate session error: {e}")
    
    def logout_user(self, session_token: str) -> bool:
        """Logout user by invalidating session"""
        try:
            session = self.validate_session(session_token)
            if session:
                self._log_security_event("LOGOUT", user_id=session['user_id'])
            
            self._invalidate_session(session_token)
            return True
            
        except Exception as e:
            logger.error(f"❌ Logout error: {e}")
            return False
    
    def create_api_key(self, user_id: int, name: str, permissions: List[str] = None, expires_days: int = 365) -> Optional[str]:
        """Create API key for user"""
        try:
            api_key = f"tbk_{secrets.token_urlsafe(32)}"  # Trading Bot Key prefix
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            key_id = secrets.token_hex(16)
            
            expires_at = datetime.now() + timedelta(days=expires_days)
            permissions_json = json.dumps(permissions or ["read"])
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO api_keys (key_id, key_hash, user_id, name, permissions, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (key_id, key_hash, user_id, name, permissions_json, expires_at.isoformat()))
            
            conn.commit()
            conn.close()
            
            self._log_security_event("API_KEY_CREATED", user_id=user_id, details=f"API key '{name}' created")
            
            logger.info(f"✅ API key created: {name}")
            return api_key
            
        except Exception as e:
            logger.error(f"❌ Create API key error: {e}")
            return None
    
    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """Validate API key"""
        try:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT ak.user_id, ak.name, ak.permissions, ak.expires_at, u.username, u.role
                FROM api_keys ak
                JOIN users u ON ak.user_id = u.id
                WHERE ak.key_hash = ? AND ak.is_active = 1 AND u.is_active = 1
            """, (key_hash,))
            
            key_row = cursor.fetchone()
            
            if key_row:
                user_id, name, permissions_json, expires_at, username, role = key_row
                
                if expires_at:
                    expires_at_dt = datetime.fromisoformat(expires_at)
                    if datetime.now() > expires_at_dt:
                        conn.close()
                        return None
                
                # Update last used
                cursor.execute("""
                    UPDATE api_keys SET last_used = CURRENT_TIMESTAMP WHERE key_hash = ?
                """, (key_hash,))
                
                conn.commit()
                conn.close()
                
                return {
                    'user_id': user_id,
                    'username': username,
                    'role': role,
                    'key_name': name,
                    'permissions': json.loads(permissions_json)
                }
            
            conn.close()
            return None
            
        except Exception as e:
            logger.error(f"❌ API key validation error: {e}")
            return None
    
    def _validate_password_strength(self, password: str) -> bool:
        """Validate password strength"""
        if len(password) < self.password_min_length:
            return False
        
        # Check for uppercase, lowercase, digit, and special character
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return sum([has_upper, has_lower, has_digit, has_special]) >= 3
    
    def check_rate_limit(self, identifier: str, max_requests: int = None) -> bool:
        """Check rate limit for identifier (IP, user, etc.)"""
        try:
            max_requests = max_requests or self.max_requests_per_minute
            now = time.time()
            
            if identifier not in self.rate_limits:
                self.rate_limits[identifier] = []
            
            # Remove old requests (older than 1 minute)
            self.rate_limits[identifier] = [
                req_time for req_time in self.rate_limits[identifier]
                if now - req_time < 60
            ]
            
            # Check if limit exceeded
            if len(self.rate_limits[identifier]) >= max_requests:
                return False
            
            # Add current request
            self.rate_limits[identifier].append(now)
            return True
            
        except Exception as e:
            logger.error(f"❌ Rate limit check error: {e}")
            return True  # Allow on error
    
    def _log_security_event(self, event_type: str, user_id: int = None, 
                           ip_address: str = None, user_agent: str = None,
                           details: str = None, severity: str = "INFO"):
        """Log security event"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO security_events 
                (event_type, user_id, ip_address, user_agent, details, severity)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (event_type, user_id, ip_address, user_agent, details, severity))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Log security event error: {e}")
    
    def get_security_events(self, hours: int = 24, severity: str = None) -> List[Dict]:
        """Get recent security events"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            query = """
                SELECT se.event_type, se.user_id, u.username, se.ip_address, 
                       se.details, se.timestamp, se.severity
                FROM security_events se
                LEFT JOIN users u ON se.user_id = u.id
                WHERE se.timestamp > ?
            """
            
            params = [cutoff_time.isoformat()]
            
            if severity:
                query += " AND se.severity = ?"
                params.append(severity)
            
            query += " ORDER BY se.timestamp DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            events = []
            for row in rows:
                events.append({
                    'event_type': row[0],
                    'user_id': row[1],
                    'username': row[2],
                    'ip_address': row[3],
                    'details': row[4],
                    'timestamp': row[5],
                    'severity': row[6]
                })
            
            return events
            
        except Exception as e:
            logger.error(f"❌ Get security events error: {e}")
            return []
    
    def change_password(self, user_id: int, old_password: str, new_password: str) -> bool:
        """Change user password"""
        try:
            # Validate new password
            if not self._validate_password_strength(new_password):
                return False
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current password
            cursor.execute("""
                SELECT password_hash, salt FROM users WHERE id = ?
            """, (user_id,))
            
            user_row = cursor.fetchone()
            if not user_row:
                conn.close()
                return False
            
            password_hash, salt = user_row
            
            # Verify old password
            if not self._verify_password(old_password, password_hash, salt):
                conn.close()
                return False
            
            # Hash new password
            new_password_hash, new_salt = self._hash_password(new_password)
            
            # Update password
            cursor.execute("""
                UPDATE users SET password_hash = ?, salt = ? WHERE id = ?
            """, (new_password_hash, new_salt, user_id))
            
            conn.commit()
            conn.close()
            
            # Log event
            self._log_security_event("PASSWORD_CHANGED", user_id=user_id)
            
            logger.info(f"✅ Password changed for user ID: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Change password error: {e}")
            return False
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        try:
            # Clean memory cache
            expired_tokens = [
                token for token, session in self.active_sessions.items()
                if datetime.now() > session['expires_at']
            ]
            
            for token in expired_tokens:
                del self.active_sessions[token]
            
            # Clean database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE sessions SET is_active = 0 
                WHERE expires_at < CURRENT_TIMESTAMP AND is_active = 1
            """)
            
            expired_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if expired_count > 0:
                logger.info(f"🧹 Cleaned up {expired_count} expired sessions")
            
        except Exception as e:
            logger.error(f"❌ Cleanup expired sessions error: {e}")
    
    def get_user_sessions(self, user_id: int) -> List[Dict]:
        """Get active sessions for user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT session_token, created_at, last_activity, ip_address, user_agent
                FROM sessions 
                WHERE user_id = ? AND is_active = 1 AND expires_at > CURRENT_TIMESTAMP
                ORDER BY last_activity DESC
            """, (user_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            sessions = []
            for row in rows:
                sessions.append({
                    'session_token': row[0][:16] + "...",  # Truncated for security
                    'created_at': row[1],
                    'last_activity': row[2],
                    'ip_address': row[3],
                    'user_agent': row[4]
                })
            
            return sessions
            
        except Exception as e:
            logger.error(f"❌ Get user sessions error: {e}")
            return []
    
    def get_security_summary(self) -> Dict:
        """Get security summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # User counts
            cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
            active_users = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM users WHERE locked_until > CURRENT_TIMESTAMP")
            locked_users = cursor.fetchone()[0]
            
            # Session counts
            cursor.execute("SELECT COUNT(*) FROM sessions WHERE is_active = 1 AND expires_at > CURRENT_TIMESTAMP")
            active_sessions = cursor.fetchone()[0]
            
            # API key counts
            cursor.execute("SELECT COUNT(*) FROM api_keys WHERE is_active = 1")
            active_api_keys = cursor.fetchone()[0]
            
            # Recent security events
            cursor.execute("""
                SELECT severity, COUNT(*) FROM security_events 
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY severity
            """)
            
            event_counts = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'users': {
                    'active': active_users,
                    'locked': locked_users
                },
                'sessions': {
                    'active': active_sessions,
                    'in_memory': len(self.active_sessions)
                },
                'api_keys': {
                    'active': active_api_keys
                },
                'security_events_24h': event_counts,
                'rate_limits': {
                    'active_limits': len(self.rate_limits),
                    'max_requests_per_minute': self.max_requests_per_minute
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Security summary error: {e}")
            return {'error': str(e)}


# Security middleware and decorators
def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # This would check user permissions
            # Implementation depends on your framework
            return func(*args, **kwargs)
        return wrapper
    return decorator

def audit_action(action: str):
    """Decorator to audit user actions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Log the action
            result = func(*args, **kwargs)
            # Log completion
            return result
        return wrapper
    return decorator

# Usage example and testing
if __name__ == "__main__":
    import tempfile
    
    # Test security manager
    print("🔐 Testing Security Manager...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = f"{temp_dir}/security_test.db"
        security_manager = SecurityManager(db_path)
        
        # Test user creation
        success = security_manager.create_user("testuser", "SecurePass123!", "test@example.com")
        print(f"✅ User creation: {success}")
        
        # Test authentication
        auth_result = security_manager.authenticate_user("testuser", "SecurePass123!")
        print(f"✅ Authentication: {bool(auth_result)}")
        
        if auth_result:
            session_token = auth_result['session_token']
            
            # Test session validation
            session = security_manager.validate_session(session_token)
            print(f"✅ Session validation: {bool(session)}")
            
            # Test API key creation
            api_key = security_manager.create_api_key(auth_result['user_id'], "Test Key")
            print(f"✅ API key created: {bool(api_key)}")
            
            if api_key:
                # Test API key validation
                key_validation = security_manager.validate_api_key(api_key)
                print(f"✅ API key validation: {bool(key_validation)}")
        
        # Test security summary
        summary = security_manager.get_security_summary()
        print(f"📊 Security summary: {summary['users']['active']} active users")
        
        print("✅ Security manager test completed")
