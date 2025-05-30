#!/usr/bin/env python3
"""
API Routes for Trading Bot Flask Application
Handles all REST API endpoints for user risk management, bot control, and data retrieval
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from functools import wraps
import json

from flask import Blueprint, request, jsonify, session, g
from flask import current_app as app

# Import system components with corrected imports
from config_manager import ConfigManager, get_config
from database_setup import DatabaseManager
from bot_core import TradingBotCore

# Create API blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Global variables for bot instance and database
bot_instance = None
db_manager = None
config_manager = None

def init_api_components():
    """Initialize API components"""
    global bot_instance, db_manager, config_manager
    
    try:
        config_manager = get_config()  # This returns ConfigManager instance
        db_manager = DatabaseManager(config_manager)
        
        # Initialize database connection
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(db_manager.initialize())
        
        logging.info("API components initialized successfully")
        return True
        
    except Exception as e:
        logging.error(f"Failed to initialize API components: {e}")
        return False

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({
                'success': False,
                'message': 'Authentication required'
            }), 401
        
        g.user_id = session['user_id']
        return f(*args, **kwargs)
    
    return decorated_function

def handle_async(f):
    """Decorator to handle async functions in Flask routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(f(*args, **kwargs))
        except Exception as e:
            logging.error(f"Async route error: {e}")
            return jsonify({
                'success': False,
                'message': f'Server error: {str(e)}'
            }), 500
        finally:
            loop.close()
    
    return decorated_function

def get_or_create_bot_instance(user_id: str):
    """Get or create bot instance for user"""
    global bot_instance
    
    try:
        if not bot_instance or bot_instance.user_id != user_id:
            bot_instance = TradingBotCore(user_id=user_id)
        
        return bot_instance
        
    except Exception as e:
        logging.error(f"Error creating bot instance: {e}")
        return None

# Authentication and User Management
@api_bp.route('/login', methods=['POST'])
@handle_async
async def api_login():
    """API login endpoint"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({
                'success': False,
                'message': 'Username and password required'
            }), 400
        
        # Authenticate user
        user_id = await db_manager.authenticate_user(username, password)
        
        if user_id:
            session['user_id'] = user_id
            session['username'] = username
            
            # Load user config
            user_config = await config_manager.load_user_risk_config(user_id)
            
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'user_id': user_id,
                'user_config': user_config
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Invalid credentials'
            }), 401
            
    except Exception as e:
        logging.error(f"Login error: {e}")
        return jsonify({
            'success': False,
            'message': 'Login failed'
        }), 500

@api_bp.route('/logout', methods=['POST'])
def api_logout():
    """API logout endpoint"""
    session.clear()
    return jsonify({
        'success': True,
        'message': 'Logged out successfully'
    })

@api_bp.route('/register', methods=['POST'])
@handle_async
async def api_register():
    """API registration endpoint"""
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not all([username, email, password]):
            return jsonify({
                'success': False,
                'message': 'All fields are required'
            }), 400
        
        # Create user
        user_id = await db_manager.create_user(username, email, password)
        
        return jsonify({
            'success': True,
            'message': 'Registration successful',
            'user_id': user_id
        })
        
    except Exception as e:
        logging.error(f"Registration error: {e}")
        return jsonify({
            'success': False,
            'message': 'Registration failed'
        }), 500

# Dashboard and Status
@api_bp.route('/dashboard-data')
@require_auth
@handle_async
async def get_dashboard_data():
    """Get comprehensive dashboard data"""
    try:
        user_id = g.user_id
        bot = get_or_create_bot_instance(user_id)
        
        # Get bot status
        bot_status = await bot.get_user_status() if bot else {}
        
        # Get recent trades
        recent_trades = await db_manager.get_user_trades(user_id, limit=10)
        
        # Get portfolio data
        portfolio = await db_manager.get_user_portfolio(user_id)
        portfolio_summary = await db_manager.calculate_user_daily_pnl(user_id)
        
        # Get risk alerts
        alerts = await db_manager.get_user_risk_alerts(user_id, unacknowledged_only=True)
        
        # Get performance history for chart
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        pnl_history = await db_manager.get_performance_history(user_id, start_date, end_date)
        
        return jsonify({
            'success': True,
            'data': {
                'portfolio_value': portfolio_summary.get('total_pnl', 0) + config_manager.default_risk.capital,
                'daily_pnl': portfolio_summary.get('daily_pnl', 0),
                'total_trades': portfolio_summary.get('total_trades', 0),
                'win_rate': portfolio_summary.get('win_rate', 0),
                'bot_status': bot_status,
                'recent_trades': recent_trades,
                'alerts': alerts[:5],  # Latest 5 alerts
                'pnl_data': {
                    'labels': [item['date'] for item in pnl_history],
                    'values': [item['pnl'] for item in pnl_history]
                }
            }
        })
        
    except Exception as e:
        logging.error(f"Dashboard data error: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to load dashboard data'
        }), 500

@api_bp.route('/bot-status')
@require_auth
@handle_async
async def get_bot_status():
    """Get current bot status"""
    try:
        user_id = g.user_id
        bot = get_or_create_bot_instance(user_id)
        
        if bot:
            status = await bot.get_user_status()
            return jsonify({
                'success': True,
                'data': status
            })
        else:
            return jsonify({
                'success': True,
                'data': {
                    'is_running': False,
                    'trading_mode': 'paper',
                    'message': 'Bot not initialized'
                }
            })
            
    except Exception as e:
        logging.error(f"Bot status error: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to get bot status'
        }), 500

# User Configuration Management
@api_bp.route('/user-config')
@require_auth
@handle_async
async def get_user_config():
    """Get user trading configuration"""
    try:
        user_id = g.user_id
        user_config = await config_manager.load_user_risk_config(user_id)
        
        return jsonify({
            'success': True,
            'data': user_config
        })
        
    except Exception as e:
        logging.error(f"Get user config error: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to load user configuration'
        }), 500

@api_bp.route('/update-risk-config', methods=['POST'])
@require_auth
@handle_async
async def update_risk_config():
    """Update user risk configuration"""
    try:
        user_id = g.user_id
        config_data = request.get_json()
        
        # Validate configuration
        validation_result = config_manager.validate_user_config(config_data)
        
        if not validation_result.get('overall_valid', False):
            return jsonify({
                'success': False,
                'message': 'Invalid configuration',
                'validation_errors': validation_result
            }), 400
        
        # Update configuration
        success = await config_manager.update_user_risk_config(user_id, config_data)
        
        if success:
            # Update bot instance if running
            bot = get_or_create_bot_instance(user_id)
            if bot:
                await bot.load_user_risk_config()
                await bot.apply_user_risk_settings()
            
            return jsonify({
                'success': True,
                'message': 'Configuration updated successfully',
                'derived_params': config_manager.calculate_derived_risk_params(config_data)
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to update configuration'
            }), 500
            
    except Exception as e:
        logging.error(f"Update risk config error: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to update configuration'
        }), 500

@api_bp.route('/risk-status')
@require_auth
@handle_async
async def get_risk_status():
    """Get current risk status and metrics"""
    try:
        user_id = g.user_id
        
        # Get user config
        user_config = await config_manager.load_user_risk_config(user_id)
        
        # Get current portfolio status
        daily_pnl = await db_manager.calculate_user_daily_pnl(user_id)
        portfolio = await db_manager.get_user_portfolio(user_id)
        
        # Get bot status
        bot = get_or_create_bot_instance(user_id)
        bot_status = await bot.get_user_status() if bot else {}
        
        # Calculate risk metrics
        risk_calculations = config_manager.calculate_derived_risk_params(user_config) if user_config else {}
        
        # Determine risk status
        risk_status = 'Safe'
        if abs(daily_pnl.get('daily_pnl', 0)) > risk_calculations.get('daily_loss_limit_amount', 0):
            risk_status = 'Critical'
        elif len(portfolio) >= user_config.get('max_concurrent_trades', 5):
            risk_status = 'Warning'
        
        return jsonify({
            'success': True,
            'data': {
                'daily_pnl': daily_pnl.get('daily_pnl', 0),
                'active_trades': len(portfolio),
                'risk_status': risk_status,
                'risk_calculations': risk_calculations,
                'bot_running': bot_status.get('is_running', False)
            }
        })
        
    except Exception as e:
        logging.error(f"Risk status error: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to get risk status'
        }), 500

# Portfolio and Trading
@api_bp.route('/portfolio')
@require_auth
@handle_async
async def get_portfolio():
    """Get user portfolio"""
    try:
        user_id = g.user_id
        
        # Get portfolio holdings
        holdings = await db_manager.get_user_portfolio(user_id)
        
        # Calculate summary
        total_value = sum(holding['current_value'] for holding in holdings)
        invested_amount = sum(holding['invested_amount'] for holding in holdings)
        total_pnl = sum(holding['unrealized_pnl'] for holding in holdings)
        total_return_percent = (total_pnl / invested_amount * 100) if invested_amount > 0 else 0
        
        return jsonify({
            'success': True,
            'data': {
                'holdings': holdings,
                'total_value': total_value,
                'invested_amount': invested_amount,
                'total_pnl': total_pnl,
                'total_return_percent': total_return_percent
            }
        })
        
    except Exception as e:
        logging.error(f"Portfolio error: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to load portfolio'
        }), 500

@api_bp.route('/trades')
@require_auth
@handle_async
async def get_trades():
    """Get user trade history"""
    try:
        user_id = g.user_id
        filter_param = request.args.get('filter', 'today')
        limit = int(request.args.get('limit', 100))
        
        # Calculate date range based on filter
        end_date = datetime.now()
        if filter_param == 'today':
            start_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
        elif filter_param == 'week':
            start_date = end_date - timedelta(days=7)
        elif filter_param == 'month':
            start_date = end_date - timedelta(days=30)
        else:  # all
            start_date = None
        
        # Get trades
        trades = await db_manager.get_user_trades(user_id, start_date, end_date, limit)
        
        # Calculate summary
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.get('realized_pnl', 0) > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = sum(t.get('realized_pnl', 0) for t in trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        return jsonify({
            'success': True,
            'data': {
                'trades': trades,
                'summary': {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'avg_pnl': avg_pnl
                }
            }
        })
        
    except Exception as e:
        logging.error(f"Trades error: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to load trades'
        }), 500

@api_bp.route('/performance-history')
@require_auth
@handle_async
async def get_performance_history():
    """Get performance history for charts"""
    try:
        user_id = g.user_id
        days = int(request.args.get('days', 30))
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get daily reports
        daily_reports = []
        current_date = start_date
        
        while current_date <= end_date:
            daily_pnl = await db_manager.calculate_user_daily_pnl(user_id, current_date.date())
            daily_reports.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'pnl': daily_pnl.get('daily_pnl', 0)
            })
            current_date += timedelta(days=1)
        
        return jsonify({
            'success': True,
            'data': {
                'labels': [item['date'] for item in daily_reports],
                'values': [item['pnl'] for item in daily_reports]
            }
        })
        
    except Exception as e:
        logging.error(f"Performance history error: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to load performance history'
        }), 500

# Bot Control
@api_bp.route('/bot-control', methods=['POST'])
@require_auth
@handle_async
async def bot_control():
    """Control bot operations (start/stop/restart)"""
    try:
        user_id = g.user_id
        data = request.get_json()
        action = data.get('action')
        
        if action not in ['start', 'stop', 'restart']:
            return jsonify({
                'success': False,
                'message': 'Invalid action'
            }), 400
        
        bot = get_or_create_bot_instance(user_id)
        if not bot:
            return jsonify({
                'success': False,
                'message': 'Failed to create bot instance'
            }), 500
        
        if action == 'start':
            if not bot.is_running:
                await bot.start()
                message = 'Bot started successfully'
            else:
                message = 'Bot is already running'
                
        elif action == 'stop':
            if bot.is_running:
                await bot.stop()
                message = 'Bot stopped successfully'
            else:
                message = 'Bot is already stopped'
                
        elif action == 'restart':
            if bot.is_running:
                await bot.stop()
            await bot.start()
            message = 'Bot restarted successfully'
        
        # Get updated status
        status = await bot.get_user_status()
        
        return jsonify({
            'success': True,
            'message': message,
            'bot_status': status
        })
        
    except Exception as e:
        logging.error(f"Bot control error: {e}")
        return jsonify({
            'success': False,
            'message': f'Bot control failed: {str(e)}'
        }), 500

@api_bp.route('/manual-trade', methods=['POST'])
@require_auth
@handle_async
async def manual_trade():
    """Execute manual trade"""
    try:
        user_id = g.user_id
        data = request.get_json()
        
        symbol = data.get('symbol')
        action = data.get('action')  # BUY or SELL
        quantity = data.get('quantity')
        price = data.get('price')  # Optional
        
        if not all([symbol, action, quantity]):
            return jsonify({
                'success': False,
                'message': 'Symbol, action, and quantity are required'
            }), 400
        
        bot = get_or_create_bot_instance(user_id)
        if not bot:
            return jsonify({
                'success': False,
                'message': 'Bot not available'
            }), 500
        
        # Execute manual trade
        result = await bot.manual_trade(symbol, action, int(quantity), float(price) if price else None)
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Manual trade error: {e}")
        return jsonify({
            'success': False,
            'message': f'Manual trade failed: {str(e)}'
        }), 500

@api_bp.route('/close-position', methods=['POST'])
@require_auth
@handle_async
async def close_position():
    """Close a specific position"""
    try:
        user_id = g.user_id
        data = request.get_json()
        symbol = data.get('symbol')
        
        if not symbol:
            return jsonify({
                'success': False,
                'message': 'Symbol is required'
            }), 400
        
        bot = get_or_create_bot_instance(user_id)
        if not bot:
            return jsonify({
                'success': False,
                'message': 'Bot not available'
            }), 500
        
        # Close position
        result = await bot.close_position(symbol)
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Close position error: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to close position: {str(e)}'
        }), 500

# Strategies
@api_bp.route('/strategies')
@require_auth
@handle_async
async def get_strategies():
    """Get available strategies and their status"""
    try:
        user_id = g.user_id
        bot = get_or_create_bot_instance(user_id)
        
        if bot:
            strategies = await bot.get_active_strategies()
        else:
            strategies = {}
        
        return jsonify({
            'success': True,
            'data': strategies
        })
        
    except Exception as e:
        logging.error(f"Strategies error: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to load strategies'
        }), 500

@api_bp.route('/toggle-strategy', methods=['POST'])
@require_auth
@handle_async
async def toggle_strategy():
    """Enable/disable a trading strategy"""
    try:
        user_id = g.user_id
        data = request.get_json()
        strategy_name = data.get('strategy')
        enabled = data.get('enabled', True)
        
        if not strategy_name:
            return jsonify({
                'success': False,
                'message': 'Strategy name is required'
            }), 400
        
        bot = get_or_create_bot_instance(user_id)
        if not bot:
            return jsonify({
                'success': False,
                'message': 'Bot not available'
            }), 500
        
        # Toggle strategy
        await bot.toggle_strategy(strategy_name, enabled)
        
        return jsonify({
            'success': True,
            'message': f'Strategy {strategy_name} {"enabled" if enabled else "disabled"}'
        })
        
    except Exception as e:
        logging.error(f"Toggle strategy error: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to toggle strategy: {str(e)}'
        }), 500

# Alerts and Notifications
@api_bp.route('/alerts')
@require_auth
@handle_async
async def get_alerts():
    """Get user risk alerts"""
    try:
        user_id = g.user_id
        unacknowledged_only = request.args.get('unacknowledged_only', 'true').lower() == 'true'
        
        alerts = await db_manager.get_user_risk_alerts(user_id, unacknowledged_only)
        
        return jsonify({
            'success': True,
            'data': alerts
        })
        
    except Exception as e:
        logging.error(f"Alerts error: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to load alerts'
        }), 500

@api_bp.route('/acknowledge-alert/<alert_id>', methods=['POST'])
@require_auth
@handle_async
async def acknowledge_alert(alert_id):
    """Acknowledge a risk alert"""
    try:
        success = await db_manager.acknowledge_risk_alert(alert_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Alert acknowledged'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to acknowledge alert'
            }), 500
            
    except Exception as e:
        logging.error(f"Acknowledge alert error: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to acknowledge alert'
        }), 500

# System Status and Health
@api_bp.route('/system-status')
@require_auth
@handle_async
async def get_system_status():
    """Get comprehensive system status"""
    try:
        user_id = g.user_id
        bot = get_or_create_bot_instance(user_id)
        
        # Get bot health
        bot_health = await bot.get_system_health() if bot else {}
        
        # Check database connection
        try:
            await db_manager.get_database_stats()
            database_connected = True
        except:
            database_connected = False
        
        # Check API configuration
        user_config = await config_manager.load_user_risk_config(user_id)
        api_configured = bool(user_config and user_config.get('goodwill_api_key'))
        
        return jsonify({
            'success': True,
            'data': {
                'bot_running': bot.is_running if bot else False,
                'database_connected': database_connected,
                'api_configured': api_configured,
                'bot_health': bot_health,
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logging.error(f"System status error: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to get system status'
        }), 500

@api_bp.route('/health')
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

# Settings
@api_bp.route('/save-settings/<category>', methods=['POST'])
@require_auth
@handle_async
async def save_settings(category):
    """Save user settings"""
    try:
        user_id = g.user_id
        data = request.get_json()
        
        if category == 'general':
            # Update general settings
            success = await config_manager.update_user_risk_config(user_id, data)
        elif category == 'notifications':
            # Update notification settings
            success = await db_manager.update_user_config(user_id, data)
        elif category == 'api':
            # Store API credentials securely
            success = await db_manager.store_user_api_credentials(
                user_id, 'goodwill',
                data.get('goodwill_api_key'),
                data.get('goodwill_secret_key'),
                data.get('goodwill_user_id')
            )
        else:
            return jsonify({
                'success': False,
                'message': 'Invalid settings category'
            }), 400
        
        if success:
            return jsonify({
                'success': True,
                'message': f'{category.title()} settings saved successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Failed to save {category} settings'
            }), 500
            
    except Exception as e:
        logging.error(f"Save settings error: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to save {category} settings'
        }), 500

# Reports and Analytics
@api_bp.route('/daily-report')
@require_auth
@handle_async
async def get_daily_report():
    """Get daily trading report"""
    try:
        user_id = g.user_id
        date = request.args.get('date')  # Optional date parameter
        
        report = await db_manager.get_user_daily_report(user_id, date)
        
        return jsonify({
            'success': True,
            'data': report
        })
        
    except Exception as e:
        logging.error(f"Daily report error: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to load daily report'
        }), 500

@api_bp.route('/statistics')
@require_auth
@handle_async
async def get_user_statistics():
    """Get comprehensive user trading statistics"""
    try:
        user_id = g.user_id
        
        stats = await db_manager.get_user_statistics(user_id)
        
        return jsonify({
            'success': True,
            'data': stats
        })
        
    except Exception as e:
        logging.error(f"Statistics error: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to load statistics'
        }), 500

# Error handlers
@api_bp.errorhandler(404)
def api_not_found(error):
    return jsonify({
        'success': False,
        'message': 'API endpoint not found'
    }), 404

@api_bp.errorhandler(500)
def api_internal_error(error):
    return jsonify({
        'success': False,
        'message': 'Internal server error'
    }), 500

# Initialize API when blueprint is registered
@api_bp.before_app_first_request
def initialize_api():
    """Initialize API components when app starts"""
    success = init_api_components()
    if not success:
        logging.error("Failed to initialize API components")

# Export the blueprint
__all__ = ['api_bp', 'init_api_components']
