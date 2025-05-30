#!/usr/bin/env python3
"""
Production-Ready Flask Web Application with Goodwill API Integration
Complete trading system web interface with authentication and real-time features
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, List, Optional, Any

import flask
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
import plotly.graph_objs as go
import plotly.utils
from werkzeug.serving import make_server
import threading

# Import system components
from config_manager import ConfigManager
from database_setup import TradingDatabase
from security_manager import SecurityManager, SecurityLevel, ActionType
from monitoring import TradingSystemMonitor
from websocket_manager import WebSocketManager
from data_manager import DataManager
from goodwill_api_handler import get_goodwill_handler
from bot_core import TradingBot
from utils import Logger, ErrorHandler, DataValidator

class TradingFlaskApp:
    """Production-ready Flask application with Goodwill API integration"""
    
    def __init__(self, config_manager: ConfigManager, trading_db: TradingDatabase,
                 security_manager: SecurityManager, monitor: TradingSystemMonitor,
                 websocket_manager: WebSocketManager, data_manager: DataManager,
                 trading_bot: TradingBot):
        
        # Initialize components
        self.config_manager = config_manager
        self.trading_db = trading_db
        self.security_manager = security_manager
        self.monitor = monitor
        self.websocket_manager = websocket_manager
        self.data_manager = data_manager
        self.trading_bot = trading_bot
        
        # Get Goodwill API handler
        self.goodwill_handler = get_goodwill_handler()
        
        # Initialize Flask app
        self.app = Flask(__name__, 
                        static_folder='static',
                        template_folder='templates')
        
        # Get configuration
        self.config = config_manager.get_config()
        self.web_config = self.config.get('web_interface', {})
        
        # Initialize logger and utilities
        self.logger = Logger(__name__)
        self.error_handler = ErrorHandler()
        self.data_validator = DataValidator()
        
        # Configure Flask app
        self._configure_flask()
        
        # Setup security and middleware
        self._setup_security()
        
        # Register routes
        self._register_routes()
        
        # Server instance
        self.server = None
        self.server_thread = None
        
        self.logger.info("Trading Flask App with Goodwill integration initialized")
    
    def _configure_flask(self):
        """Configure Flask application settings"""
        # Basic configuration
        self.app.config['SECRET_KEY'] = self.web_config.get('secret_key', os.urandom(24))
        self.app.config['DEBUG'] = self.web_config.get('debug', False)
        
        # Session configuration
        self.app.config['SESSION_COOKIE_SECURE'] = True
        self.app.config['SESSION_COOKIE_HTTPONLY'] = True
        self.app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
        self.app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
        
        # CORS configuration
        CORS(self.app, origins=['http://localhost:3000'], supports_credentials=True)
    
    def _setup_security(self):
        """Setup security middleware and rate limiting"""
        try:
            # Rate limiting
            self.limiter = Limiter(
                app=self.app,
                key_func=get_remote_address,
                default_limits=["200 per day", "50 per hour"]
            )
            
            # Security headers
            Talisman(self.app, 
                    force_https=False,
                    content_security_policy={
                        'default-src': "'self'",
                        'script-src': "'self' 'unsafe-inline' https://cdn.plot.ly https://cdnjs.cloudflare.com",
                        'style-src': "'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com",
                        'font-src': "'self' https://fonts.gstatic.com",
                    })
            
            # Error handlers
            self._register_error_handlers()
            
        except Exception as e:
            self.logger.error(f"Security setup failed: {e}")
            raise
    
    def _register_error_handlers(self):
        """Register custom error handlers"""
        
        @self.app.errorhandler(400)
        def bad_request(error):
            return jsonify({'error': 'Bad request'}), 400
        
        @self.app.errorhandler(401)
        def unauthorized(error):
            if request.is_json:
                return jsonify({'error': 'Authentication required'}), 401
            return redirect(url_for('login'))
        
        @self.app.errorhandler(403)
        def forbidden(error):
            return jsonify({'error': 'Forbidden'}), 403
        
        @self.app.errorhandler(404)
        def not_found(error):
            if request.is_json:
                return jsonify({'error': 'Not found'}), 404
            return render_template('error.html', error='Page not found'), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            self.logger.error(f"Internal server error: {error}")
            return jsonify({'error': 'Internal server error'}), 500
    
    def require_auth(self, required_level: SecurityLevel = SecurityLevel.USER):
        """Authentication decorator"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                try:
                    # Check session
                    if 'user_id' not in session or 'session_id' not in session:
                        if request.is_json:
                            return jsonify({'error': 'Authentication required'}), 401
                        return redirect(url_for('login'))
                    
                    # Validate session
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    session_data = loop.run_until_complete(
                        self.security_manager.validate_session(
                            session['session_id'],
                            request.remote_addr
                        )
                    )
                    
                    if not session_data:
                        session.clear()
                        if request.is_json:
                            return jsonify({'error': 'Session expired'}), 401
                        flash('Session expired. Please login again.', 'warning')
                        return redirect(url_for('login'))
                    
                    # Check security level
                    user_level = SecurityLevel(session_data['security_level'])
                    if user_level.value < required_level.value:
                        if request.is_json:
                            return jsonify({'error': 'Insufficient permissions'}), 403
                        flash('Insufficient permissions.', 'error')
                        return redirect(url_for('dashboard'))
                    
                    # Add user data to request context
                    flask.g.current_user = session_data
                    
                    return f(*args, **kwargs)
                    
                except Exception as e:
                    self.logger.error(f"Authentication error: {e}")
                    if request.is_json:
                        return jsonify({'error': 'Authentication error'}), 500
                    return redirect(url_for('login'))
            
            return decorated_function
        return decorator
    
    def _register_routes(self):
        """Register all application routes"""
        
        # Main authentication routes
        @self.app.route('/login', methods=['GET', 'POST'])
        @self.limiter.limit("10 per minute")
        def login():
            if request.method == 'GET':
                return render_template('login.html')
            
            try:
                username = request.form.get('username')
                password = request.form.get('password')
                
                if not username or not password:
                    flash('Username and password are required.', 'error')
                    return render_template('login.html'), 400
                
                # Authenticate user
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                auth_result = loop.run_until_complete(
                    self.security_manager.authenticate_user(
                        username, password, request.remote_addr, request.user_agent.string
                    )
                )
                
                if auth_result:
                    session['user_id'] = auth_result['user_id']
                    session['username'] = auth_result['username']
                    session['session_id'] = auth_result['session']['session_id']
                    session['security_level'] = auth_result['security_level']
                    session.permanent = True
                    
                    flash(f'Welcome back, {username}!', 'success')
                    return redirect(url_for('dashboard'))
                else:
                    flash('Invalid username or password.', 'error')
                    return render_template('login.html'), 401
                    
            except Exception as e:
                self.logger.error(f"Login error: {e}")
                flash('Login failed. Please try again.', 'error')
                return render_template('login.html'), 500
        
        @self.app.route('/logout')
        def logout():
            try:
                if 'session_id' in session:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    loop.run_until_complete(
                        self.security_manager.invalidate_session(session['session_id'])
                    )
                
                session.clear()
                flash('You have been logged out.', 'info')
                
            except Exception as e:
                self.logger.error(f"Logout error: {e}")
            
            return redirect(url_for('login'))
        
        # Dashboard with Goodwill integration
        @self.app.route('/')
        @self.require_auth()
        def dashboard():
            try:
                # Get Goodwill authentication status
                goodwill_status = self.goodwill_handler.get_authentication_status()
                
                # Get system data
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                portfolio_summary = loop.run_until_complete(
                    self.trading_db.get_portfolio_summary()
                )
                
                performance_metrics = loop.run_until_complete(
                    self.trading_db.get_performance_metrics()
                )
                
                system_status = self.monitor.get_system_status()
                websocket_stats = self.websocket_manager.get_statistics()
                
                # Get Goodwill data if authenticated
                goodwill_positions = []
                goodwill_holdings = []
                
                if goodwill_status['is_authenticated']:
                    try:
                        goodwill_positions = loop.run_until_complete(
                            self.goodwill_handler.get_positions()
                        )
                        goodwill_holdings = loop.run_until_complete(
                            self.goodwill_handler.get_holdings()
                        )
                    except Exception as e:
                        self.logger.error(f"Error fetching Goodwill data: {e}")
                
                # Get recent trades
                recent_trades = loop.run_until_complete(
                    self.trading_db.get_trade_history(limit=10)
                )
                
                dashboard_data = {
                    'goodwill_status': goodwill_status,
                    'portfolio': portfolio_summary,
                    'performance': performance_metrics,
                    'system_status': system_status,
                    'websocket_stats': websocket_stats,
                    'goodwill_positions': goodwill_positions,
                    'goodwill_holdings': goodwill_holdings,
                    'recent_trades': recent_trades,
                    'current_user': flask.g.current_user
                }
                
                return render_template('dashboard.html', **dashboard_data)
                
            except Exception as e:
                self.logger.error(f"Dashboard error: {e}")
                flash('Error loading dashboard data.', 'error')
                return render_template('dashboard.html', error=True)
        
        # Goodwill API integration routes
        @self.app.route('/goodwill/connect', methods=['POST'])
        @self.require_auth(SecurityLevel.TRADER)
        def goodwill_connect():
            try:
                action = request.form.get('action')
                
                if action == 'start_login':
                    # Generate Goodwill login URL
                    login_url = self.goodwill_handler.start_login_process()
                    
                    if login_url:
                        return jsonify({
                            'success': True,
                            'login_url': login_url,
                            'message': 'Login URL generated. Open in new tab to authenticate.'
                        })
                    else:
                        return jsonify({
                            'success': False,
                            'error': 'Failed to generate login URL. Check API configuration.'
                        }), 400
                
                elif action == 'complete_login':
                    # Complete login with request token
                    request_token = request.form.get('request_token')
                    
                    if not request_token:
                        return jsonify({
                            'success': False,
                            'error': 'Request token is required'
                        }), 400
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    success = loop.run_until_complete(
                        self.goodwill_handler.complete_login_with_request_token(request_token)
                    )
                    
                    if success:
                        # Start WebSocket connection after successful login
                        websocket_success = loop.run_until_complete(
                            self.websocket_manager.start_goodwill_connection()
                        )
                        
                        # Log successful connection
                        loop.run_until_complete(
                            self.security_manager.log_security_event(
                                user_id=flask.g.current_user['user_id'],
                                action=ActionType.LOGIN,
                                resource="goodwill_api",
                                ip_address=request.remote_addr,
                                success=True,
                                details={'websocket_connected': websocket_success}
                            )
                        )
                        
                        return jsonify({
                            'success': True,
                            'message': 'Goodwill authentication successful!',
                            'websocket_connected': websocket_success,
                            'auth_status': self.goodwill_handler.get_authentication_status()
                        })
                    else:
                        return jsonify({
                            'success': False,
                            'error': 'Authentication failed. Please check the request token.'
                        }), 401
                
                elif action == 'logout':
                    # Logout from Goodwill
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    success = loop.run_until_complete(self.goodwill_handler.logout())
                    
                    return jsonify({
                        'success': success,
                        'message': 'Logged out from Goodwill' if success else 'Logout failed'
                    })
                
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Invalid action'
                    }), 400
                    
            except Exception as e:
                self.logger.error(f"Goodwill connection error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        # API routes for AJAX calls
        @self.app.route('/api/goodwill/status')
        @self.require_auth()
        def api_goodwill_status():
            try:
                status = self.goodwill_handler.get_authentication_status()
                stats = self.goodwill_handler.get_api_statistics()
                
                return jsonify({
                    'success': True,
                    'data': {
                        'authentication': status,
                        'statistics': stats
                    }
                })
                
            except Exception as e:
        @self.app.route('/api/system/status')
        @self.require_auth()
        def api_system_status():
            try:
                system_status = self.monitor.get_system_status()
                websocket_status = self.websocket_manager.get_connection_status()
                goodwill_status = self.goodwill_handler.get_authentication_status()
                
                return jsonify({
                    'success': True,
                    'data': {
                        'system': system_status,
                        'websockets': websocket_status,
                        'goodwill': goodwill_status,
                        'timestamp': datetime.now().isoformat()
                    }
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
    
    async def _get_user_trading_config(self, user_id: str) -> Dict:
        """Get user's trading configuration"""
        try:
            async with self.trading_db.pg_pool.acquire() as conn:
                config = await conn.fetchrow("""
                    SELECT * FROM user_trading_config WHERE user_id = $1
                """, user_id)
                
                if config:
                    return {
                        'capital': float(config['capital']),
                        'risk_per_trade_percent': float(config['risk_per_trade_percent']),
                        'daily_loss_limit_percent': float(config['daily_loss_limit_percent']),
                        'max_concurrent_trades': config['max_concurrent_trades'],
                        'risk_reward_ratio': float(config['risk_reward_ratio']),
                        'max_position_size_percent': float(config['max_position_size_percent']),
                        'stop_loss_percent': float(config['stop_loss_percent']),
                        'take_profit_percent': float(config['take_profit_percent']),
                        'trading_start_time': config['trading_start_time'].strftime('%H:%M') if config['trading_start_time'] else '09:15',
                        'trading_end_time': config['trading_end_time'].strftime('%H:%M') if config['trading_end_time'] else '15:30',
                        'auto_square_off': config['auto_square_off'],
                        'paper_trading_mode': config['paper_trading_mode'],
                        'last_updated': config['last_updated'].isoformat() if config['last_updated'] else None
                    }
                else:
                    # Return default configuration
                    return {
                        'capital': 100000.0,
                        'risk_per_trade_percent': 2.0,
                        'daily_loss_limit_percent': 5.0,
                        'max_concurrent_trades': 3,
                        'risk_reward_ratio': 2.0,
                        'max_position_size_percent': 10.0,
                        'stop_loss_percent': 2.0,
                        'take_profit_percent': 4.0,
                        'trading_start_time': '09:15',
                        'trading_end_time': '15:30',
                        'auto_square_off': True,
                        'paper_trading_mode': True,
                        'last_updated': None
                    }
                    
        except Exception as e:
            self.logger.error(f"Error getting user trading config: {e}")
            raise
    
    async def _save_user_trading_config(self, user_id: str, config: Dict) -> bool:
        """Save user's trading configuration"""
        try:
            async with self.trading_db.pg_pool.acquire() as conn:
                # First ensure the table exists
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_trading_config (
                        user_id UUID PRIMARY KEY,
                        capital DECIMAL(15,2) NOT NULL DEFAULT 100000,
                        risk_per_trade_percent DECIMAL(5,2) NOT NULL DEFAULT 2.0,
                        daily_loss_limit_percent DECIMAL(5,2) NOT NULL DEFAULT 5.0,
                        max_concurrent_trades INTEGER NOT NULL DEFAULT 3,
                        risk_reward_ratio DECIMAL(5,2) NOT NULL DEFAULT 2.0,
                        max_position_size_percent DECIMAL(5,2) NOT NULL DEFAULT 10.0,
                        stop_loss_percent DECIMAL(5,2) NOT NULL DEFAULT 2.0,
                        take_profit_percent DECIMAL(5,2) NOT NULL DEFAULT 4.0,
                        trading_start_time TIME DEFAULT '09:15:00',
                        trading_end_time TIME DEFAULT '15:30:00',
                        auto_square_off BOOLEAN DEFAULT true,
                        paper_trading_mode BOOLEAN DEFAULT true,
                        last_updated TIMESTAMP DEFAULT NOW(),
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Insert or update configuration
                await conn.execute("""
                    INSERT INTO user_trading_config (
                        user_id, capital, risk_per_trade_percent, daily_loss_limit_percent,
                        max_concurrent_trades, risk_reward_ratio, max_position_size_percent,
                        stop_loss_percent, take_profit_percent, trading_start_time,
                        trading_end_time, auto_square_off, paper_trading_mode, last_updated
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, NOW())
                    ON CONFLICT (user_id) DO UPDATE SET
                        capital = EXCLUDED.capital,
                        risk_per_trade_percent = EXCLUDED.risk_per_trade_percent,
                        daily_loss_limit_percent = EXCLUDED.daily_loss_limit_percent,
                        max_concurrent_trades = EXCLUDED.max_concurrent_trades,
                        risk_reward_ratio = EXCLUDED.risk_reward_ratio,
                        max_position_size_percent = EXCLUDED.max_position_size_percent,
                        stop_loss_percent = EXCLUDED.stop_loss_percent,
                        take_profit_percent = EXCLUDED.take_profit_percent,
                        trading_start_time = EXCLUDED.trading_start_time,
                        trading_end_time = EXCLUDED.trading_end_time,
                        auto_square_off = EXCLUDED.auto_square_off,
                        paper_trading_mode = EXCLUDED.paper_trading_mode,
                        last_updated = NOW()
                """, user_id, config['capital'], config['risk_per_trade_percent'],
                    config['daily_loss_limit_percent'], config['max_concurrent_trades'],
                    config['risk_reward_ratio'], config['max_position_size_percent'],
                    config['stop_loss_percent'], config['take_profit_percent'],
                    config['trading_start_time'], config['trading_end_time'],
                    config['auto_square_off'], config['paper_trading_mode'])
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving user trading config: {e}")
            return False
    
    def _validate_trading_config(self, config: Dict) -> Dict:
        """Validate trading configuration"""
        try:
            errors = []
            
            # Validate capital
            if config.get('capital', 0) <= 0:
                errors.append("Capital must be greater than 0")
            
            # Validate risk per trade
            risk_per_trade = config.get('risk_per_trade_percent', 0)
            if risk_per_trade <= 0 or risk_per_trade > 10:
                errors.append("Risk per trade must be between 0.1% and 10%")
            
            # Validate daily loss limit
            daily_loss = config.get('daily_loss_limit_percent', 0)
            if daily_loss <= 0 or daily_loss > 25:
                errors.append("Daily loss limit must be between 0.1% and 25%")
            
            # Validate max concurrent trades
            max_trades = config.get('max_concurrent_trades', 0)
            if max_trades <= 0 or max_trades > 20:
                errors.append("Max concurrent trades must be between 1 and 20")
            
            # Validate risk reward ratio
            rr_ratio = config.get('risk_reward_ratio', 0)
            if rr_ratio <= 0 or rr_ratio > 10:
                errors.append("Risk reward ratio must be between 0.1 and 10")
            
            # Validate position size
            pos_size = config.get('max_position_size_percent', 0)
            if pos_size <= 0 or pos_size > 50:
                errors.append("Max position size must be between 0.1% and 50%")
            
            # Validate stop loss
            stop_loss = config.get('stop_loss_percent', 0)
            if stop_loss <= 0 or stop_loss > 20:
                errors.append("Stop loss must be between 0.1% and 20%")
            
            # Validate take profit
            take_profit = config.get('take_profit_percent', 0)
            if take_profit <= 0 or take_profit > 50:
                errors.append("Take profit must be between 0.1% and 50%")
            
            return {
                'valid': len(errors) == 0,
                'error': '; '.join(errors) if errors else None
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f"Validation error: {str(e)}"
            }
    
    def _calculate_risk_metrics(self, config: Dict, portfolio: Dict, active_orders: List) -> Dict:
        """Calculate current risk metrics"""
        try:
            capital = config.get('capital', 100000)
            current_value = portfolio.get('current_value', 0)
            
            # Calculate daily P&L percentage
            daily_pnl_percent = 0
            if capital > 0:
                daily_pnl = portfolio.get('total_unrealized_pnl', 0)
                daily_pnl_percent = (daily_pnl / capital) * 100
            
            # Calculate position exposure
            total_exposure = sum(order.get('quantity', 0) * order.get('price', 0) for order in active_orders)
            exposure_percent = (total_exposure / capital * 100) if capital > 0 else 0
            
            # Calculate number of active trades
            active_trades = len(active_orders)
            
            # Risk alerts
            alerts = []
            
            if daily_pnl_percent <= -config.get('daily_loss_limit_percent', 5):
                alerts.append({
                    'type': 'error',
                    'message': f'Daily loss limit exceeded: {daily_pnl_percent:.2f}%'
                })
            
            if active_trades >= config.get('max_concurrent_trades', 3):
                alerts.append({
                    'type': 'warning',
                    'message': f'Maximum concurrent trades reached: {active_trades}'
                })
            
            if exposure_percent > 90:
                alerts.append({
                    'type': 'warning',
                    'message': f'High capital exposure: {exposure_percent:.1f}%'
                })
            
            return {
                'capital': capital,
                'current_value': current_value,
                'daily_pnl_percent': daily_pnl_percent,
                'daily_loss_limit': config.get('daily_loss_limit_percent', 5),
                'risk_per_trade': config.get('risk_per_trade_percent', 2),
                'active_trades': active_trades,
                'max_trades': config.get('max_concurrent_trades', 3),
                'exposure_percent': exposure_percent,
                'risk_reward_ratio': config.get('risk_reward_ratio', 2),
                'alerts': alerts,
                'can_trade': len([a for a in alerts if a['type'] == 'error']) == 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {}
        
        @self.app.route('/api/goodwill/search_symbols', methods=['POST'])
        @self.require_auth()
        def api_goodwill_search():
            try:
                if not self.goodwill_handler.is_authenticated:
                    return jsonify({
                        'success': False,
                        'error': 'Goodwill not authenticated'
                    }), 401
                
                data = request.get_json()
                search_term = data.get('search_term', '')
                
                if not search_term:
                    return jsonify({
                        'success': False,
                        'error': 'Search term required'
                    }), 400
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                results = loop.run_until_complete(
                    self.goodwill_handler.search_symbols(search_term)
                )
                
                return jsonify({
                    'success': True,
                    'data': results
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/goodwill/get_quote', methods=['POST'])
        @self.require_auth()
        def api_goodwill_quote():
            try:
                if not self.goodwill_handler.is_authenticated:
                    return jsonify({
                        'success': False,
                        'error': 'Goodwill not authenticated'
                    }), 401
                
                data = request.get_json()
                exchange = data.get('exchange', 'NSE')
                token = data.get('token', '')
                
                if not token:
                    return jsonify({
                        'success': False,
                        'error': 'Token required'
                    }), 400
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                quote = loop.run_until_complete(
                    self.goodwill_handler.get_quote(exchange, token)
                )
                
                return jsonify({
                    'success': True,
                    'data': quote
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/goodwill/place_order', methods=['POST'])
        @self.require_auth(SecurityLevel.TRADER)
        @self.limiter.limit("10 per minute")
        def api_goodwill_place_order():
            try:
                if not self.goodwill_handler.is_authenticated:
                    return jsonify({
                        'success': False,
                        'error': 'Goodwill not authenticated'
                    }), 401
                
                order_data = request.get_json()
                
                # Validate required fields
                required_fields = ['tsym', 'exchange', 'trantype', 'validity', 'pricetype', 'qty', 'price', 'product']
                for field in required_fields:
                    if field not in order_data:
                        return jsonify({
                            'success': False,
                            'error': f'Missing field: {field}'
                        }), 400
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                order_response = loop.run_until_complete(
                    self.goodwill_handler.place_order(order_data)
                )
                
                if order_response and order_response.status == 'submitted':
                    # Log order placement
                    loop.run_until_complete(
                        self.security_manager.log_security_event(
                            user_id=flask.g.current_user['user_id'],
                            action=ActionType.TRADE_PLACED,
                            resource=f"goodwill_order_{order_data['tsym']}",
                            ip_address=request.remote_addr,
                            success=True,
                            details=order_data
                        )
                    )
                    
                    return jsonify({
                        'success': True,
                        'data': {
                            'order_id': order_response.order_id,
                            'status': order_response.status,
                            'message': order_response.message
                        }
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': order_response.message if order_response else 'Order failed'
                    }), 400
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/websocket/subscribe', methods=['POST'])
        @self.require_auth()
        def api_websocket_subscribe():
            try:
                data = request.get_json()
                symbols = data.get('symbols', [])
                
                if not symbols:
                    return jsonify({
                        'success': False,
                        'error': 'Symbols list required'
                    }), 400
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                success = loop.run_until_complete(
                    self.websocket_manager.subscribe_to_goodwill_symbols(symbols)
                )
                
                return jsonify({
                    'success': success,
                    'message': f'Subscribed to {len(symbols)} symbols' if success else 'Subscription failed',
                    'symbols': symbols
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/trading/config', methods=['GET', 'POST'])
        @self.require_auth(SecurityLevel.TRADER)
        def api_trading_config():
            try:
                if request.method == 'GET':
                    # Get current trading configuration
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Get user's trading config from database
                    config = loop.run_until_complete(
                        self._get_user_trading_config(flask.g.current_user['user_id'])
                    )
                    
                    return jsonify({
                        'success': True,
                        'data': config
                    })
                
                elif request.method == 'POST':
                    # Update trading configuration
                    config_data = request.get_json()
                    
                    # Validate configuration
                    validation_result = self._validate_trading_config(config_data)
                    if not validation_result['valid']:
                        return jsonify({
                            'success': False,
                            'error': validation_result['error']
                        }), 400
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Save configuration
                    success = loop.run_until_complete(
                        self._save_user_trading_config(
                            flask.g.current_user['user_id'], 
                            config_data
                        )
                    )
                    
                    if success:
                        # Log configuration change
                        loop.run_until_complete(
                            self.security_manager.log_security_event(
                                user_id=flask.g.current_user['user_id'],
                                action=ActionType.SETTINGS_CHANGED,
                                resource="trading_configuration",
                                ip_address=request.remote_addr,
                                success=True,
                                details=config_data
                            )
                        )
                        
                        return jsonify({
                            'success': True,
                            'message': 'Trading configuration updated successfully'
                        })
                    else:
                        return jsonify({
                            'success': False,
                            'error': 'Failed to save configuration'
                        }), 500
                        
            except Exception as e:
                self.logger.error(f"Trading config error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/trading/risk_metrics')
        @self.require_auth()
        def api_risk_metrics():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Get current risk metrics
                config = loop.run_until_complete(
                    self._get_user_trading_config(flask.g.current_user['user_id'])
                )
                
                # Calculate current risk exposure
                portfolio_summary = loop.run_until_complete(
                    self.trading_db.get_portfolio_summary()
                )
                
                active_orders = loop.run_until_complete(
                    self.trading_db.get_active_orders()
                )
                
                # Calculate risk metrics
                risk_metrics = self._calculate_risk_metrics(config, portfolio_summary, active_orders)
                
                return jsonify({
                    'success': True,
                    'data': risk_metrics
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
    
    def create_templates(self):
        """Create all HTML templates"""
        templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
        os.makedirs(templates_dir, exist_ok=True)
        
        # Base template
        base_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Trading System{% endblock %}</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        .goodwill-status { border-left: 4px solid #28a745; }
        .goodwill-disconnected { border-left: 4px solid #dc3545; }
        .metric-card { transition: transform 0.2s; }
        .metric-card:hover { transform: translateY(-2px); }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('dashboard') }}">
                <i class="fas fa-chart-line"></i> Trading System
            </a>
            {% if session.username %}
            <div class="navbar-nav ms-auto">
                <span class="navbar-text me-3">{{ session.username }}</span>
                <a class="btn btn-outline-light btn-sm" href="{{ url_for('logout') }}">
                    <i class="fas fa-sign-out-alt"></i> Logout
                </a>
            </div>
            {% endif %}
        </div>
    </nav>
    
    <main class="container-fluid mt-4">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ 'danger' if category == 'error' else category }} alert-dismissible fade show">
                        {{ message }}
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        {% block content %}{% endblock %}
    </main>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/js/bootstrap.bundle.min.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>'''
        
        # Login template
        login_template = '''{% extends "base.html" %}
{% block title %}Login - Trading System{% endblock %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-6 col-lg-4">
        <div class="card shadow">
            <div class="card-header bg-primary text-white text-center">
                <h4><i class="fas fa-sign-in-alt"></i> Trading System Login</h4>
            </div>
            <div class="card-body">
                <form method="POST">
                    <div class="mb-3">
                        <label for="username" class="form-label">Username</label>
                        <div class="input-group">
                            <span class="input-group-text"><i class="fas fa-user"></i></span>
                            <input type="text" class="form-control" id="username" name="username" required>
                        </div>
                    </div>
                    <div class="mb-3">
                        <label for="password" class="form-label">Password</label>
                        <div class="input-group">
                            <span class="input-group-text"><i class="fas fa-lock"></i></span>
                            <input type="password" class="form-control" id="password" name="password" required>
                        </div>
                    </div>
                    <button type="submit" class="btn btn-primary w-100">
                        <i class="fas fa-sign-in-alt"></i> Login
                    </button>
                </form>
            </div>
        </div>
    </div>
</div>
{% endblock %}'''
        
        # Dashboard template with Goodwill integration
        dashboard_template = '''{% extends "base.html" %}
{% block title %}Dashboard - Trading System{% endblock %}

{% block content %}
<!-- Goodwill Status Banner -->
<div class="row mb-3">
    <div class="col-12">
        <div class="card {{ 'goodwill-status' if goodwill_status.is_authenticated else 'goodwill-disconnected' }}">
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <h6 class="mb-1">
                            <i class="fas fa-{{ 'check-circle text-success' if goodwill_status.is_authenticated else 'exclamation-triangle text-warning' }}"></i>
                            Goodwill API Status
                        </h6>
                        {% if goodwill_status.is_authenticated %}
                            <p class="mb-0">
                                <strong>Connected:</strong> {{ goodwill_status.user_name }} ({{ goodwill_status.client_id }})
                                <br><small class="text-muted">Expires: {{ goodwill_status.expires_at }}</small>
                            </p>
                        {% else %}
                            <p class="mb-0 text-muted">Not connected to Goodwill API. Connect to enable live trading.</p>
                        {% endif %}
                    </div>
                    <div>
                        {% if goodwill_status.is_authenticated %}
                            <button class="btn btn-sm btn-warning" onclick="disconnectGoodwill()">
                                <i class="fas fa-unlink"></i> Disconnect
                            </button>
                        {% else %}
                            <button class="btn btn-sm btn-success" onclick="connectGoodwill()">
                                <i class="fas fa-link"></i> Connect to Goodwill
                            </button>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Risk Management & Trading Configuration -->
<div class="row mb-4">
    <div class="col-12">
        <div class="card">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5><i class="fas fa-shield-alt"></i> Risk Management & Trading Configuration</h5>
                <button class="btn btn-sm btn-primary" data-bs-toggle="modal" data-bs-target="#configModal">
                    <i class="fas fa-cog"></i> Configure
                </button>
            </div>
            <div class="card-body">
                <div id="riskMetrics" class="row">
                    <!-- Risk metrics will be loaded here -->
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Trading Configuration Modal -->
<div class="modal fade" id="configModal" tabindex="-1">
    <div class="modal-dialog modal-lg">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title"><i class="fas fa-cog"></i> Trading Configuration</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
            </div>
            <div class="modal-body">
                <form id="tradingConfigForm">
                    <div class="row">
                        <div class="col-md-6">
                            <h6 class="text-primary">Capital & Risk Management</h6>
                            
                            <div class="mb-3">
                                <label for="capital" class="form-label">
                                    <i class="fas fa-rupee-sign"></i> Total Capital
                                </label>
                                <div class="input-group">
                                    <span class="input-group-text">₹</span>
                                    <input type="number" class="form-control" id="capital" name="capital" 
                                           min="10000" max="10000000" step="1000" required>
                                </div>
                                <small class="text-muted">Your total trading capital</small>
                            </div>
                            
                            <div class="mb-3">
                                <label for="riskPerTrade" class="form-label">
                                    <i class="fas fa-percentage"></i> Risk Per Trade
                                </label>
                                <div class="input-group">
                                    <input type="number" class="form-control" id="riskPerTrade" name="risk_per_trade_percent" 
                                           min="0.1" max="10" step="0.1" required>
                                    <span class="input-group-text">%</span>
                                </div>
                                <small class="text-muted">Maximum risk per single trade (recommended: 1-2%)</small>
                            </div>
                            
                            <div class="mb-3">
                                <label for="dailyLossLimit" class="form-label">
                                    <i class="fas fa-exclamation-triangle"></i> Daily Loss Limit
                                </label>
                                <div class="input-group">
                                    <input type="number" class="form-control" id="dailyLossLimit" name="daily_loss_limit_percent" 
                                           min="1" max="25" step="0.5" required>
                                    <span class="input-group-text">%</span>
                                </div>
                                <small class="text-muted">Stop trading if daily loss exceeds this percentage</small>
                            </div>
                            
                            <div class="mb-3">
                                <label for="maxConcurrentTrades" class="form-label">
                                    <i class="fas fa-layer-group"></i> Max Concurrent Trades
                                </label>
                                <input type="number" class="form-control" id="maxConcurrentTrades" name="max_concurrent_trades" 
                                       min="1" max="20" required>
                                <small class="text-muted">Maximum number of simultaneous open positions</small>
                            </div>
                            
                            <div class="mb-3">
                                <label for="riskRewardRatio" class="form-label">
                                    <i class="fas fa-balance-scale"></i> Risk-Reward Ratio
                                </label>
                                <div class="input-group">
                                    <span class="input-group-text">1:</span>
                                    <input type="number" class="form-control" id="riskRewardRatio" name="risk_reward_ratio" 
                                           min="0.5" max="10" step="0.1" required>
                                </div>
                                <small class="text-muted">Target profit vs maximum loss ratio (recommended: 2:1 or higher)</small>
                            </div>
                        </div>
                        
                        <div class="col-md-6">
                            <h6 class="text-primary">Position & Execution Settings</h6>
                            
                            <div class="mb-3">
                                <label for="maxPositionSize" class="form-label">
                                    <i class="fas fa-chart-pie"></i> Max Position Size
                                </label>
                                <div class="input-group">
                                    <input type="number" class="form-control" id="maxPositionSize" name="max_position_size_percent" 
                                           min="1" max="50" step="1" required>
                                    <span class="input-group-text">%</span>
                                </div>
                                <small class="text-muted">Maximum capital allocation per position</small>
                            </div>
                            
                            <div class="mb-3">
                                <label for="stopLoss" class="form-label">
                                    <i class="fas fa-stop-circle"></i> Default Stop Loss
                                </label>
                                <div class="input-group">
                                    <input type="number" class="form-control" id="stopLoss" name="stop_loss_percent" 
                                           min="0.5" max="20" step="0.1" required>
                                    <span class="input-group-text">%</span>
                                </div>
                                <small class="text-muted">Default stop loss percentage from entry price</small>
                            </div>
                            
                            <div class="mb-3">
                                <label for="takeProfit" class="form-label">
                                    <i class="fas fa-bullseye"></i> Default Take Profit
                                </label>
                                <div class="input-group">
                                    <input type="number" class="form-control" id="takeProfit" name="take_profit_percent" 
                                           min="0.5" max="50" step="0.1" required>
                                    <span class="input-group-text">%</span>
                                </div>
                                <small class="text-muted">Default take profit percentage from entry price</small>
                            </div>
                            
                            <div class="row">
                                <div class="col-6">
                                    <div class="mb-3">
                                        <label for="tradingStartTime" class="form-label">
                                            <i class="fas fa-clock"></i> Trading Start
                                        </label>
                                        <input type="time" class="form-control" id="tradingStartTime" name="trading_start_time" required>
                                    </div>
                                </div>
                                <div class="col-6">
                                    <div class="mb-3">
                                        <label for="tradingEndTime" class="form-label">
                                            <i class="fas fa-clock"></i> Trading End
                                        </label>
                                        <input type="time" class="form-control" id="tradingEndTime" name="trading_end_time" required>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="mb-3">
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="autoSquareOff" name="auto_square_off">
                                    <label class="form-check-label" for="autoSquareOff">
                                        <i class="fas fa-power-off"></i> Auto Square-off at Market Close
                                    </label>
                                </div>
                                <small class="text-muted">Automatically close all positions before market close</small>
                            </div>
                            
                            <div class="mb-3">
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="paperTradingMode" name="paper_trading_mode">
                                    <label class="form-check-label" for="paperTradingMode">
                                        <i class="fas fa-flask"></i> Paper Trading Mode
                                    </label>
                                </div>
                                <small class="text-muted">Simulate trades without real money (recommended for testing)</small>
                            </div>
                        </div>
                    </div>
                </form>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                <button type="button" class="btn btn-primary" onclick="saveTradingConfig()">
                    <i class="fas fa-save"></i> Save Configuration
                </button>
            </div>
        </div>
    </div>
</div>

<!-- Portfolio Metrics -->
<div class="row mb-4">
    <div class="col-md-3">
        <div class="card metric-card bg-primary text-white">
            <div class="card-body text-center">
                <i class="fas fa-wallet fa-2x mb-2"></i>
                <h5>Portfolio Value</h5>
                <h3>₹{{ "{:,.2f}".format(portfolio.current_value or 0) }}</h3>
                {% if portfolio.total_return_percent %}
                <small>
                    <i class="fas fa-{{ 'arrow-up' if portfolio.total_return_percent >= 0 else 'arrow-down' }}"></i>
                    {{ "{:.2f}".format(portfolio.total_return_percent) }}%
                </small>
                {% endif %}
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card metric-card bg-success text-white">
            <div class="card-body text-center">
                <i class="fas fa-chart-line fa-2x mb-2"></i>
                <h5>Today's P&L</h5>
                <h3>₹{{ "{:,.2f}".format(performance.get('day_pnl', 0)) }}</h3>
                <small>Realized + Unrealized</small>
            </div>
        text-muted">Expires: {{ goodwill_status.expires_at }}</small>
                            </p>
                        {% else %}
                            <p class="mb-0 text-muted">Not connected to Goodwill API. Connect to enable live trading.</p>
                        {% endif %}
                    </div>
                    <div>
                        {% if goodwill_status.is_authenticated %}
                            <button class="btn btn-sm btn-warning" onclick="disconnectGoodwill()">
                                <i class="fas fa-unlink"></i> Disconnect
                            </button>
                        {% else %}
                            <button class="btn btn-sm btn-success" onclick="connectGoodwill()">
                                <i class="fas fa-link"></i> Connect to Goodwill
                            </button>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Portfolio Metrics -->
<div class="row mb-4">
    <div class="col-md-3">
        <div class="card metric-card bg-primary text-white">
            <div class="card-body text-center">
                <i class="fas fa-wallet fa-2x mb-2"></i>
                <h5>Portfolio Value</h5>
                <h3>₹{{ "{:,.2f}".format(portfolio.current_value or 0) }}</h3>
                {% if portfolio.total_return_percent %}
                <small>
                    <i class="fas fa-{{ 'arrow-up' if portfolio.total_return_percent >= 0 else 'arrow-down' }}"></i>
                    {{ "{:.2f}".format(portfolio.total_return_percent) }}%
                </small>
                {% endif %}
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card metric-card bg-success text-white">
            <div class="card-body text-center">
                <i class="fas fa-chart-line fa-2x mb-2"></i>
                <h5>Today's P&L</h5>
                <h3>₹{{ "{:,.2f}".format(performance.get('day_pnl', 0)) }}</h3>
                <small>Realized + Unrealized</small>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card metric-card bg-info text-white">
            <div class="card-body text-center">
                <i class="fas fa-layer-group fa-2x mb-2"></i>
                <h5>Active Positions</h5>
                <h3>{{ portfolio.total_positions or 0 }}</h3>
                <small>Holdings</small>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card metric-card bg-warning text-white">
            <div class="card-body text-center">
                <i class="fas fa-heartbeat fa-2x mb-2"></i>
                <h5>System Status</h5>
                <h3>{{ system_status.status }}</h3>
                <small>{{ system_status.services|length }} services</small>
            </div>
        </div>
    </div>
</div>

<!-- Quick Trading Actions (if Goodwill connected) -->
{% if goodwill_status.is_authenticated %}
<div class="row mb-4">
    <div class="col-12">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-bolt"></i> Quick Trading Actions</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h6>Search & Quote</h6>
                        <div class="input-group mb-3">
                            <input type="text" class="form-control" id="symbolSearch" placeholder="Search symbol (e.g., RELIANCE)">
                            <button class="btn btn-outline-primary" type="button" id="searchSymbol">
                                <i class="fas fa-search"></i> Search
                            </button>
                        </div>
                        <div id="searchResults"></div>
                    </div>
                    <div class="col-md-6">
                        <h6>WebSocket Subscription</h6>
                        <div class="input-group mb-3">
                            <input type="text" class="form-control" id="subscribeSymbols" placeholder="Symbols to subscribe (comma separated)">
                            <button class="btn btn-outline-success" type="button" id="subscribeBtn">
                                <i class="fas fa-wifi"></i> Subscribe
                            </button>
                        </div>
                        <small class="text-muted">Subscribe to real-time price updates</small>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endif %}

<!-- Recent Trades & System Status -->
<div class="row">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-history"></i> Recent Trades</h5>
            </div>
            <div class="card-body">
                {% if recent_trades %}
                    <div class="table-responsive">
                        <table class="table table-striped table-sm">
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Symbol</th>
                                    <th>Side</th>
                                    <th>Quantity</th>
                                    <th>Price</th>
                                    <th>P&L</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for trade in recent_trades[:10] %}
                                <tr>
                                    <td><small>{{ trade.execution_time.strftime('%H:%M:%S') if trade.execution_time else '-' }}</small></td>
                                    <td><strong>{{ trade.symbol }}</strong></td>
                                    <td>
                                        <span class="badge bg-{{ 'success' if trade.side == 'BUY' else 'danger' }}">
                                            {{ trade.side }}
                                        </span>
                                    </td>
                                    <td>{{ trade.quantity }}</td>
                                    <td>₹{{ "{:.2f}".format(trade.price) }}</td>
                                    <td class="{{ 'text-success' if (trade.pnl or 0) >= 0 else 'text-danger' }}">
                                        {% if trade.pnl %}₹{{ "{:.2f}".format(trade.pnl) }}{% else %}-{% endif %}
                                    </td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                {% else %}
                    <div class="text-center py-4">
                        <i class="fas fa-inbox fa-3x text-muted mb-3"></i>
                        <p class="text-muted">No recent trades</p>
                    </div>
                {% endif %}
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-server"></i> System Health</h5>
            </div>
            <div class="card-body">
                {% for service, status in system_status.services.items() %}
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span>{{ service.title() }}</span>
                    <span class="badge bg-{{ 'success' if status.status == 'HEALTHY' else 'warning' if status.status == 'DEGRADED' else 'danger' }}">
                        {{ status.status }}
                    </span>
                </div>
                {% endfor %}
                
                <hr>
                
                <h6 class="mt-3">WebSocket Status</h6>
                <p class="mb-1"><strong>Active:</strong> {{ websocket_stats.active_connections }}/{{ websocket_stats.total_connections }}</p>
                <p class="mb-1"><strong>Messages:</strong> {{ "{:,}".format(websocket_stats.total_messages) }}</p>
                <p class="mb-0"><strong>Last Activity:</strong> 
                    {% if websocket_stats.last_activity %}
                        {{ websocket_stats.last_activity.strftime('%H:%M:%S') }}
                    {% else %}
                        Never
                    {% endif %}
                </p>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
// Global variables
let tradingConfig = {};
let riskMetrics = {};

// Load trading configuration and risk metrics on page load
document.addEventListener('DOMContentLoaded', function() {
    loadTradingConfig();
    loadRiskMetrics();
    
    // Auto-refresh every 30 seconds
    setInterval(function() {
        loadRiskMetrics();
        updateSystemStatus();
    }, 30000);
});

// Load trading configuration
function loadTradingConfig() {
    fetch('/api/trading/config')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                tradingConfig = data.data;
                populateConfigForm();
            }
        })
        .catch(error => console.error('Error loading trading config:', error));
}

// Load risk metrics
function loadRiskMetrics() {
    fetch('/api/trading/risk_metrics')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                riskMetrics = data.data;
                displayRiskMetrics();
            }
        })
        .catch(error => console.error('Error loading risk metrics:', error));
}

// Display risk metrics
function displayRiskMetrics() {
    const container = document.getElementById('riskMetrics');
    if (!container || !riskMetrics) return;
    
    // Display alerts if any
    let alertsHtml = '';
    if (riskMetrics.alerts && riskMetrics.alerts.length > 0) {
        alertsHtml = '<div class="col-12 mb-3">';
        riskMetrics.alerts.forEach(alert => {
            alertsHtml += `
                <div class="alert alert-${alert.type === 'error' ? 'danger' : 'warning'} mb-2">
                    <i class="fas fa-exclamation-triangle"></i> ${alert.message}
                </div>
            `;
        });
        alertsHtml += '</div>';
    }
    
    container.innerHTML = alertsHtml + `
        <div class="col-md-2">
            <div class="text-center">
                <h6 class="text-muted">Capital</h6>
                <h4 class="text-primary">₹${(riskMetrics.capital || 0).toLocaleString()}</h4>
            </div>
        </div>
        <div class="col-md-2">
            <div class="text-center">
                <h6 class="text-muted">Daily P&L</h6>
                <h4 class="${(riskMetrics.daily_pnl_percent || 0) >= 0 ? 'text-success' : 'text-danger'}">
                    ${(riskMetrics.daily_pnl_percent || 0).toFixed(2)}%
                </h4>
                <small class="text-muted">Limit: ${riskMetrics.daily_loss_limit}%</small>
            </div>
        </div>
        <div class="col-md-2">
            <div class="text-center">
                <h6 class="text-muted">Risk/Trade</h6>
                <h4 class="text-info">${riskMetrics.risk_per_trade}%</h4>
            </div>
        </div>
        <div class="col-md-2">
            <div class="text-center">
                <h6 class="text-muted">Active Trades</h6>
                <h4 class="${riskMetrics.active_trades >= riskMetrics.max_trades ? 'text-warning' : 'text-success'}">
                    ${riskMetrics.active_trades}/${riskMetrics.max_trades}
                </h4>
            </div>
        </div>
        <div class="col-md-2">
            <div class="text-center">
                <h6 class="text-muted">R:R Ratio</h6>
                <h4 class="text-primary">1:${riskMetrics.risk_reward_ratio}</h4>
            </div>
        </div>
        <div class="col-md-2">
            <div class="text-center">
                <h6 class="text-muted">Can Trade</h6>
                <h4 class="${riskMetrics.can_trade ? 'text-success' : 'text-danger'}">
                    <i class="fas fa-${riskMetrics.can_trade ? 'check-circle' : 'times-circle'}"></i>
                </h4>
            </div>
        </div>
    `;
}

// Populate configuration form
function populateConfigForm() {
    document.getElementById('capital').value = tradingConfig.capital || 100000;
    document.getElementById('riskPerTrade').value = tradingConfig.risk_per_trade_percent || 2;
    document.getElementById('dailyLossLimit').value = tradingConfig.daily_loss_limit_percent || 5;
    document.getElementById('maxConcurrentTrades').value = tradingConfig.max_concurrent_trades || 3;
    document.getElementById('riskRewardRatio').value = tradingConfig.risk_reward_ratio || 2;
    document.getElementById('maxPositionSize').value = tradingConfig.max_position_size_percent || 10;
    document.getElementById('stopLoss').value = tradingConfig.stop_loss_percent || 2;
    document.getElementById('takeProfit').value = tradingConfig.take_profit_percent || 4;
    document.getElementById('tradingStartTime').value = tradingConfig.trading_start_time || '09:15';
    document.getElementById('tradingEndTime').value = tradingConfig.trading_end_time || '15:30';
    document.getElementById('autoSquareOff').checked = tradingConfig.auto_square_off !== false;
    document.getElementById('paperTradingMode').checked = tradingConfig.paper_trading_mode !== false;
}

// Save trading configuration
function saveTradingConfig() {
    const form = document.getElementById('tradingConfigForm');
    const formData = new FormData(form);
    
    const config = {};
    for (let [key, value] of formData.entries()) {
        if (key.includes('percent') || key === 'capital' || key === 'risk_reward_ratio') {
            config[key] = parseFloat(value);
        } else if (key === 'max_concurrent_trades') {
            config[key] = parseInt(value);
        } else if (key === 'auto_square_off' || key === 'paper_trading_mode') {
            config[key] = true;
        } else {
            config[key] = value;
        }
    }
    
    // Set unchecked checkboxes to false
    if (!formData.has('auto_square_off')) config.auto_square_off = false;
    if (!formData.has('paper_trading_mode')) config.paper_trading_mode = false;
    
    fetch('/api/trading/config', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(config)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Configuration saved successfully!');
            bootstrap.Modal.getInstance(document.getElementById('configModal')).hide();
            loadTradingConfig();
            loadRiskMetrics();
        } else {
            alert('Error: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error saving configuration');
    });
}

// Goodwill connection functions
function connectGoodwill() {
    // First, generate login URL
    fetch('/goodwill/connect', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: 'action=start_login'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Open login URL in new window
            const loginWindow = window.open(data.login_url, 'goodwill_login', 'width=800,height=600');
            
            // Prompt for request token
            setTimeout(() => {
                const requestToken = prompt(
                    'After logging in to Goodwill, copy the request_token from the redirect URL and paste it here:'
                );
                
                if (requestToken) {
                    completeGoodwillLogin(requestToken);
                }
                
                if (loginWindow) {
                    loginWindow.close();
                }
            }, 5000);
        } else {
            alert('Error: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error connecting to Goodwill');
    });
}

function completeGoodwillLogin(requestToken) {
    fetch('/goodwill/connect', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `action=complete_login&request_token=${encodeURIComponent(requestToken)}`
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Goodwill connected successfully! Page will reload.');
            window.location.reload();
        } else {
            alert('Error: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error completing Goodwill login');
    });
}

function disconnectGoodwill() {
    if (confirm('Are you sure you want to disconnect from Goodwill API?')) {
        fetch('/goodwill/connect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: 'action=logout'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('Disconnected from Goodwill successfully!');
                window.location.reload();
            } else {
                alert('Error disconnecting: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error disconnecting from Goodwill');
        });
    }
}

// Symbol search functionality
document.getElementById('searchSymbol')?.addEventListener('click', function() {
    const searchTerm = document.getElementById('symbolSearch').value.trim();
    if (!searchTerm) {
        alert('Please enter a symbol to search');
        return;
    }
    
    fetch('/api/goodwill/search_symbols', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({search_term: searchTerm})
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displaySearchResults(data.data);
        } else {
            alert('Error: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error searching symbols');
    });
});

function displaySearchResults(results) {
    const container = document.getElementById('searchResults');
    if (!container) return;
    
    if (results.length === 0) {
        container.innerHTML = '<p class="text-muted">No results found</p>';
        return;
    }
    
    let html = '<div class="mt-2"><h6>Search Results:</h6>';
    results.slice(0, 5).forEach(result => {
        html += `
            <div class="border rounded p-2 mb-2">
                <strong>${result.trading_symbol || result.symbol}</strong> - ${result.company_name || 'N/A'}
                <br><small class="text-muted">${result.exchange} | Token: ${result.token}</small>
                <button class="btn btn-sm btn-outline-primary ms-2" onclick="getQuote('${result.exchange}', '${result.token}')">
                    Get Quote
                </button>
            </div>
        `;
    });
    html += '</div>';
    
    container.innerHTML = html;
}

function getQuote(exchange, token) {
    fetch('/api/goodwill/get_quote', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({exchange: exchange, token: token})
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const quote = data.data;
            alert(`Quote for ${quote.trading_symbl}:\nLTP: ₹${quote.last_price}\nChange: ${quote.change} (${quote.change_per}%)\nVolume: ${quote.volume}`);
        } else {
            alert('Error: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error getting quote');
    });
}

// WebSocket subscription
document.getElementById('subscribeBtn')?.addEventListener('click', function() {
    const symbolsInput = document.getElementById('subscribeSymbols').value.trim();
    if (!symbolsInput) {
        alert('Please enter symbols to subscribe');
        return;
    }
    
    const symbols = symbolsInput.split(',').map(s => s.trim()).filter(s => s);
    
    fetch('/api/websocket/subscribe', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({symbols: symbols})
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert(`Successfully subscribed to ${symbols.length} symbols for real-time updates!`);
            document.getElementById('subscribeSymbols').value = '';
        } else {
            alert('Error: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error subscribing to symbols');
    });
});

// Update system status
function updateSystemStatus() {
    fetch('/api/system/status')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update status indicators if needed
                console.log('System status updated');
            }
        })
        .catch(error => console.error('Error updating system status:', error));
}
</script>
{% endblock %}'''
        # Error template
        error_template = '''{% extends "base.html" %}
{% block title %}Error - Trading System{% endblock %}

{% block content %}
<div class="text-center py-5">
    <i class="fas fa-exclamation-triangle fa-5x text-warning mb-4"></i>
    <h1 class="display-4">{{ error or "Oops!" }}</h1>
    <p class="lead">Something went wrong. Please try again or contact support if the problem persists.</p>
    <a href="{{ url_for('dashboard') }}" class="btn btn-primary">
        <i class="fas fa-home"></i> Go to Dashboard
    </a>
</div>
{% endblock %}'''
        
        # Write all templates to files
        with open(os.path.join(templates_dir, 'base.html'), 'w') as f:
            f.write(base_template)
        
        with open(os.path.join(templates_dir, 'login.html'), 'w') as f:
            f.write(login_template)
        
        with open(os.path.join(templates_dir, 'dashboard.html'), 'w') as f:
            # Get the dashboard template from the other artifact
            dashboard_content = dashboard_template
            f.write(dashboard_content)
        
        with open(os.path.join(templates_dir, 'error.html'), 'w') as f:
            f.write(error_template)
        
        self.logger.info("All HTML templates created successfully")
    
    def create_static_files(self):
        """Create static CSS and JS files"""
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        css_dir = os.path.join(static_dir, 'css')
        js_dir = os.path.join(static_dir, 'js')
        
        os.makedirs(css_dir, exist_ok=True)
        os.makedirs(js_dir, exist_ok=True)
        
        # Custom CSS
        custom_css = '''
/* Trading System Custom Styles */

/* Goodwill status styling */
.goodwill-status {
    border-left: 4px solid #28a745;
    background: linear-gradient(135deg, #f8fff9 0%, #e8f8ea 100%);
}

.goodwill-disconnected {
    border-left: 4px solid #dc3545;
    background: linear-gradient(135deg, #fff8f8 0%, #f8e8e8 100%);
}

/* Metric cards */
.metric-card {
    transition: all 0.3s ease;
    border: none;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

/* Risk metrics display */
.risk-metric {
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 10px;
    border: 2px solid transparent;
    transition: all 0.3s ease;
}

.risk-metric-good {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border-color: #28a745;
}

.risk-metric-warning {
    background: linear-gradient(135deg, #fff3cd 0%, #fce4b6 100%);
    border-color: #ffc107;
}

.risk-metric-danger {
    background: linear-gradient(135deg, #f8d7da 0%, #f1b0b7 100%);
    border-color: #dc3545;
}

/* Modal styling */
.modal-content {
    border-radius: 15px;
    border: none;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
}

.modal-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 15px 15px 0 0;
    border-bottom: none;
}

/* Form styling */
.form-control {
    border-radius: 8px;
    border: 2px solid #e9ecef;
    transition: all 0.3s ease;
}

.form-control:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
}

/* Button styling */
.btn {
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.3s ease;
}

.btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

/* Card styling */
.card {
    border: none;
    border-radius: 15px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

.card:hover {
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}

.card-header {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-bottom: 2px solid #dee2e6;
    border-radius: 15px 15px 0 0 !important;
    font-weight: 600;
}

/* Table styling */
.table {
    border-radius: 10px;
    overflow: hidden;
}

.table thead th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    font-weight: 600;
}

/* Alert styling */
.alert {
    border-radius: 10px;
    border: none;
    font-weight: 500;
}

/* Loading animation */
.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(255,255,255,.3);
    border-radius: 50%;
    border-top-color: #fff;
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Navbar styling */
.navbar-brand {
    font-weight: 700;
    font-size: 1.5rem;
}

/* Status indicators */
.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}

.status-healthy { background-color: #28a745; }
.status-warning { background-color: #ffc107; }
.status-danger { background-color: #dc3545; }

/* Responsive improvements */
@media (max-width: 768px) {
    .metric-card {
        margin-bottom: 1rem;
    }
    
    .card-body {
        padding: 1rem 0.75rem;
    }
    
    .table-responsive {
        font-size: 0.875rem;
    }
    
    .modal-dialog {
        margin: 0.5rem;
    }
}

/* Print styles */
@media print {
    .btn, .navbar, .modal { display: none !important; }
    .card { box-shadow: none; border: 1px solid #ddd; }
}
'''
        
        # Custom JavaScript
        custom_js = '''
/* Trading System Custom JavaScript */

// Global configuration
const TradingSystem = {
    refreshInterval: 30000, // 30 seconds
    alertTimeout: 5000,     // 5 seconds
    
    // Initialize the system
    init: function() {
        this.setupEventListeners();
        this.startAutoRefresh();
        this.initializeTooltips();
    },
    
    // Setup global event listeners
    setupEventListeners: function() {
        // Global error handler
        window.addEventListener('error', function(e) {
            console.error('Global error:', e.error);
            TradingSystem.showAlert('An unexpected error occurred', 'danger');
        });
        
        // Handle connection status
        window.addEventListener('online', function() {
            TradingSystem.showAlert('Connection restored', 'success');
        });
        
        window.addEventListener('offline', function() {
            TradingSystem.showAlert('Connection lost', 'warning');
        });
    },
    
    // Start auto-refresh functionality
    startAutoRefresh: function() {
        setInterval(function() {
            if (typeof loadRiskMetrics === 'function') {
                loadRiskMetrics();
            }
            if (typeof updateSystemStatus === 'function') {
                updateSystemStatus();
            }
        }, this.refreshInterval);
    },
    
    // Initialize Bootstrap tooltips
    initializeTooltips: function() {
        if (typeof bootstrap !== 'undefined') {
            var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        }
    },
    
    // Show alert notification
    showAlert: function(message, type = 'info', timeout = null) {
        const alertContainer = document.querySelector('main.container-fluid');
        if (!alertContainer) return;
        
        const alertId = 'alert-' + Date.now();
        const alertDiv = document.createElement('div');
        alertDiv.id = alertId;
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        // Insert at the top
        alertContainer.insertBefore(alertDiv, alertContainer.firstChild);
        
        // Auto-dismiss
        const dismissTimeout = timeout || this.alertTimeout;
        setTimeout(() => {
            const alert = document.getElementById(alertId);
            if (alert) {
                if (typeof bootstrap !== 'undefined') {
                    bootstrap.Alert.getOrCreateInstance(alert).close();
                } else {
                    alert.remove();
                }
            }
        }, dismissTimeout);
    },
    
    // Format currency
    formatCurrency: function(amount, currency = 'INR') {
        return new Intl.NumberFormat('en-IN', {
            style: 'currency',
            currency: currency,
            minimumFractionDigits: 2
        }).format(amount);
    },
    
    // Format number with commas
    formatNumber: function(num) {
        return new Intl.NumberFormat('en-IN').format(num);
    },
    
    // Format percentage
    formatPercentage: function(num, decimals = 2) {
        return (num || 0).toFixed(decimals) + '%';
    },
    
    // Validate form data
    validateForm: function(formId, rules) {
        const form = document.getElementById(formId);
        if (!form) return { valid: false, errors: ['Form not found'] };
        
        const formData = new FormData(form);
        const errors = [];
        
        for (const [field, rule] of Object.entries(rules)) {
            const value = formData.get(field);
            
            if (rule.required && (!value || value.trim() === '')) {
                errors.push(`${rule.label || field} is required`);
                continue;
            }
            
            if (value && rule.min && parseFloat(value) < rule.min) {
                errors.push(`${rule.label || field} must be at least ${rule.min}`);
            }
            
            if (value && rule.max && parseFloat(value) > rule.max) {
                errors.push(`${rule.label || field} must be at most ${rule.max}`);
            }
            
            if (value && rule.pattern && !rule.pattern.test(value)) {
                errors.push(`${rule.label || field} format is invalid`);
            }
        }
        
        return { valid: errors.length === 0, errors: errors };
    },
    
    // Make API request with error handling
    apiRequest: function(url, options = {}) {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            }
        };
        
        const finalOptions = { ...defaultOptions, ...options };
        
        return fetch(url, finalOptions)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return response.json();
            })
            .catch(error => {
                console.error('API request failed:', error);
                this.showAlert('Network error: ' + error.message, 'danger');
                throw error;
            });
    },
    
    // Calculate position size based on risk
    calculatePositionSize: function(capital, riskPercent, entryPrice, stopLoss) {
        if (!capital || !riskPercent || !entryPrice || !stopLoss) {
            return 0;
        }
        
        const riskAmount = (capital * riskPercent) / 100;
        const riskPerShare = Math.abs(entryPrice - stopLoss);
        
        if (riskPerShare <= 0) return 0;
        
        return Math.floor(riskAmount / riskPerShare);
    },
    
    // Calculate risk-reward ratio
    calculateRiskReward: function(entryPrice, stopLoss, takeProfit) {
        const risk = Math.abs(entryPrice - stopLoss);
        const reward = Math.abs(takeProfit - entryPrice);
        
        if (risk <= 0) return 0;
        
        return reward / risk;
    },
    
    // Update real-time clock
    updateClock: function() {
        const clockElement = document.getElementById('currentTime');
        if (clockElement) {
            const now = new Date();
            clockElement.textContent = now.toLocaleTimeString('en-IN');
        }
    },
    
    // Check market hours
    isMarketOpen: function() {
        const now = new Date();
        const currentTime = now.getHours() * 100 + now.getMinutes();
        const isWeekday = now.getDay() >= 1 && now.getDay() <= 5;
        
        // Indian market hours: 9:15 AM to 3:30 PM
        return isWeekday && currentTime >= 915 && currentTime <= 1530;
    },
    
    // Export data to CSV
    exportToCSV: function(data, filename) {
        if (!data || data.length === 0) {
            this.showAlert('No data to export', 'warning');
            return;
        }
        
        const headers = Object.keys(data[0]);
        const csvContent = [
            headers.join(','),
            ...data.map(row => headers.map(header => {
                const value = row[header] || '';
                return typeof value === 'string' && value.includes(',') ? `"${value}"` : value;
            }).join(','))
        ].join('\n');
        
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename || 'trading_data.csv';
        a.click();
        window.URL.revokeObjectURL(url);
    }
};

// Risk calculation utilities
const RiskCalculator = {
    // Calculate maximum position size
    maxPositionSize: function(capital, maxRiskPercent) {
        return (capital * maxRiskPercent) / 100;
    },
    
    // Calculate stop loss price
    stopLossPrice: function(entryPrice, stopLossPercent, side = 'BUY') {
        const multiplier = side === 'BUY' ? (1 - stopLossPercent / 100) : (1 + stopLossPercent / 100);
        return entryPrice * multiplier;
    },
    
    // Calculate take profit price
    takeProfitPrice: function(entryPrice, takeProfitPercent, side = 'BUY') {
        const multiplier = side === 'BUY' ? (1 + takeProfitPercent / 100) : (1 - takeProfitPercent / 100);
        return entryPrice * multiplier;
    },
    
    // Calculate portfolio heat (total risk exposure)
    portfolioHeat: function(positions, capital) {
        const totalRisk = positions.reduce((sum, pos) => {
            return sum + (pos.quantity * Math.abs(pos.entryPrice - pos.stopLoss));
        }, 0);
        
        return (totalRisk / capital) * 100;
    }
};

// WebSocket utilities for real-time updates
const WebSocketManager = {
    connection: null,
    reconnectAttempts: 0,
    maxReconnectAttempts: 5,
    
    connect: function(url) {
        try {
            this.connection = new WebSocket(url);
            
            this.connection.onopen = function() {
                console.log('WebSocket connected');
                TradingSystem.showAlert('Real-time data connected', 'success');
            };
            
            this.connection.onmessage = function(event) {
                const data = JSON.parse(event.data);
                WebSocketManager.handleMessage(data);
            };
            
            this.connection.onclose = function() {
                console.log('WebSocket disconnected');
                WebSocketManager.attemptReconnect(url);
            };
            
            this.connection.onerror = function(error) {
                console.error('WebSocket error:', error);
                TradingSystem.showAlert('Real-time data connection error', 'warning');
            };
            
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
        }
    },
    
    attemptReconnect: function(url) {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            setTimeout(() => {
                console.log(`Attempting WebSocket reconnection (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
                this.connect(url);
            }, 5000 * this.reconnectAttempts);
        }
    },
    
    handleMessage: function(data) {
        // Handle different types of real-time data
        switch (data.type) {
            case 'price_update':
                this.updatePrice(data);
                break;
            case 'order_update':
                this.updateOrder(data);
                break;
            case 'portfolio_update':
                this.updatePortfolio(data);
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    },
    
    updatePrice: function(data) {
        // Update price displays in real-time
        const priceElements = document.querySelectorAll(`[data-symbol="${data.symbol}"]`);
        priceElements.forEach(element => {
            element.textContent = TradingSystem.formatCurrency(data.price);
            
            // Add visual feedback for price changes
            element.classList.add(data.change >= 0 ? 'text-success' : 'text-danger');
            setTimeout(() => {
                element.classList.remove('text-success', 'text-danger');
            }, 1000);
        });
    },
    
    updateOrder: function(data) {
        // Update order status in real-time
        TradingSystem.showAlert(`Order ${data.orderId}: ${data.status}`, 'info');
        
        // Refresh orders table if visible
        if (typeof loadActiveOrders === 'function') {
            loadActiveOrders();
        }
    },
    
    updatePortfolio: function(data) {
        // Update portfolio metrics in real-time
        if (typeof loadRiskMetrics === 'function') {
            loadRiskMetrics();
        }
    },
    
    send: function(message) {
        if (this.connection && this.connection.readyState === WebSocket.OPEN) {
            this.connection.send(JSON.stringify(message));
        }
    }
};

// Initialize system when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    TradingSystem.init();
    
    // Start clock update
    setInterval(TradingSystem.updateClock, 1000);
    
    // Update market status indicator
    const marketStatus = document.getElementById('marketStatus');
    if (marketStatus) {
        marketStatus.textContent = TradingSystem.isMarketOpen() ? 'OPEN' : 'CLOSED';
        marketStatus.className = `badge ${TradingSystem.isMarketOpen() ? 'bg-success' : 'bg-danger'}`;
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl+Shift+C - Open trading configuration
    if (e.ctrlKey && e.shiftKey && e.key === 'C') {
        const configModal = document.getElementById('configModal');
        if (configModal) {
            new bootstrap.Modal(configModal).show();
        }
    }
    
    // Ctrl+Shift+R - Refresh all data
    if (e.ctrlKey && e.shiftKey && e.key === 'R') {
        e.preventDefault();
        if (typeof loadTradingConfig === 'function') loadTradingConfig();
        if (typeof loadRiskMetrics === 'function') loadRiskMetrics();
        TradingSystem.showAlert('Data refreshed', 'success');
    }
});

// Export utilities for global use
window.TradingSystem = TradingSystem;
window.RiskCalculator = RiskCalculator;
window.WebSocketManager = WebSocketManager;
'''

        # Write static files
        with open(os.path.join(css_dir, 'trading.css'), 'w') as f:
            f.write(custom_css)
        
        with open(os.path.join(js_dir, 'trading.js'), 'w') as f:
            f.write(custom_js)
        
        self.logger.info("Static files created successfully")
    
    def start_server(self, host: str = None, port: int = None, threaded: bool = True):
        """Start the Flask server"""
        try:
            host = host or self.web_config.get('host', '0.0.0.0')
            port = port or self.web_config.get('port', 5000)
            
            # Create templates and static files
            self.create_templates()
            self.create_static_files()
            
            if threaded:
                # Run in separate thread
                self.server_thread = threading.Thread(
                    target=self._run_server,
                    args=(host, port),
                    daemon=True
                )
                self.server_thread.start()
                self.logger.info(f"Flask server started on http://{host}:{port} (threaded)")
            else:
                # Run in main thread
                self._run_server(host, port)
                
        except Exception as e:
            self.logger.error(f"Failed to start Flask server: {e}")
            raise
    
    def _run_server(self, host: str, port: int):
        """Run Flask server"""
        try:
            self.server = make_server(host, port, self.app, threaded=True)
            
            self.logger.info(f"Trading System Web Interface")
            self.logger.info(f"Server running on: http://{host}:{port}")
            self.logger.info(f"Dashboard: http://{host}:{port}/")
            self.logger.info(f"Goodwill Integration: Configured")
            self.logger.info(f"Risk Management: Enabled")
            self.logger.info("Press Ctrl+C to stop")
            
            self.server.serve_forever()
            
        except KeyboardInterrupt:
            self.logger.info("Server stopped by user")
        except Exception as e:
            self.logger.error(f"Server error: {e}")
        finally:
            if self.server:
                self.server.shutdown()
    
    def stop_server(self):
        """Stop the Flask server"""
        try:
            if self.server:
                self.server.shutdown()
                self.logger.info("Flask server stopped")
        except Exception as e:
            self.logger.error(f"Error stopping server: {e}")
    
    def get_app(self):
        """Get Flask app instance for external use"""
        return self.app

# Example usage and main function
async def main():
    """Example usage of the complete Flask webapp"""
    from config_manager import ConfigManager
    from database_setup import TradingDatabase
    from security_manager import SecurityManager
    from monitoring import TradingSystemMonitor
    from websocket_manager import WebSocketManager
    from data_manager import DataManager
    from bot_core import TradingBot
    
    try:
        # Initialize all components
        config_manager = ConfigManager("config/config.yaml")
        
        trading_db = TradingDatabase(config_manager)
        await trading_db.initialize()
        await trading_db.create_tables()
        
        security_manager = SecurityManager(trading_db, config_manager)
        await security_manager.initialize()
        
        monitor = TradingSystemMonitor(trading_db, config_manager, None)
        await monitor.initialize()
        
        data_manager = DataManager(config_manager, trading_db)
        await data_manager.initialize()
        
        websocket_manager = WebSocketManager(config_manager, data_manager, trading_db, security_manager)
        await websocket_manager.initialize()
        
        trading_bot = TradingBot(config_manager, trading_db, data_manager)
        
        # Initialize Goodwill API handler
        goodwill_handler = get_goodwill_handler()
        await goodwill_handler.initialize()
        
        # Create Flask webapp
        webapp = TradingFlaskApp(
            config_manager=config_manager,
            trading_db=trading_db,
            security_manager=security_manager,
            monitor=monitor,
            websocket_manager=websocket_manager,
            data_manager=data_manager,
            trading_bot=trading_bot
        )
        
        # Start the server
        webapp.start_server(host='0.0.0.0', port=5000, threaded=False)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        # Cleanup
        if 'trading_db' in locals():
            await trading_db.close()
        if 'goodwill_handler' in locals():
            await goodwill_handler.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
