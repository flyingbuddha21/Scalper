#!/usr/bin/env python3
"""
REST API Routes
Provides REST API endpoints for external integration
"""

from flask import Blueprint, request, jsonify, abort
from functools import wraps
import logging
import time
from datetime import datetime
import json
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Create Blueprint for API routes
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# Global references (will be set by main app)
bot_core = None
scanner = None
execution_manager = None
volatility_analyzer = None
data_manager = None
strategy_manager = None

def set_components(**components):
    """Set component references for API routes"""
    global bot_core, scanner, execution_manager, volatility_analyzer, data_manager, strategy_manager
    bot_core = components.get('bot_core')
    scanner = components.get('scanner')
    execution_manager = components.get('execution_manager')
    volatility_analyzer = components.get('volatility_analyzer')
    data_manager = components.get('data_manager')
    strategy_manager = components.get('strategy_manager')

# Rate limiting decorator
def rate_limit(max_calls: int = 100, window: int = 60):
    """Rate limiting decorator"""
    call_times = []
    
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove old calls outside window
            call_times[:] = [t for t in call_times if now - t < window]
            
            if len(call_times) >= max_calls:
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'message': f'Maximum {max_calls} calls per {window} seconds'
                }), 429
            
            call_times.append(now)
            return f(*args, **kwargs)
        return wrapper
    return decorator

# Authentication decorator (simplified)
def require_auth(f):
    """Simple authentication decorator"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authentication required'}), 401
        
        # In production, validate the token here
        token = auth_header.split(' ')[1]
        if token != 'your_api_token':  # Replace with proper validation
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(*args, **kwargs)
    return wrapper

# Error handler decorator
def handle_errors(f):
    """Error handling decorator"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"API error in {f.__name__}: {e}")
            return jsonify({
                'error': 'Internal server error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    return wrapper

# Health and Status Endpoints
@api_bp.route('/health', methods=['GET'])
@rate_limit(max_calls=60, window=60)
@handle_errors
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'services': {
            'bot_core': bool(bot_core),
            'scanner': bool(scanner),
            'execution_manager': bool(execution_manager),
            'volatility_analyzer': bool(volatility_analyzer)
        }
    })

@api_bp.route('/status', methods=['GET'])
@rate_limit(max_calls=30, window=60)
@handle_errors
def system_status():
    """Get comprehensive system status"""
    status = {
        'timestamp': datetime.now().isoformat(),
        'bot_status': bot_core.get_bot_status() if bot_core else {'running': False},
        'scanner_status': scanner.get_scanner_status() if scanner else {'running': False},
        'execution_status': execution_manager.get_execution_summary() if execution_manager else {'running': False}
    }
    
    if volatility_analyzer:
        status['volatility_status'] = volatility_analyzer.get_volatility_summary()
    
    return jsonify(status)

# Bot Management Endpoints
@api_bp.route('/bot/start', methods=['POST'])
@require_auth
@rate_limit(max_calls=5, window=60)
@handle_errors
def start_bot():
    """Start trading bot"""
    if not bot_core:
        return jsonify({'error': 'Bot core not available'}), 404
    
    if bot_core.running:
        return jsonify({'message': 'Bot is already running'}), 200
    
    bot_core.start_bot()
    return jsonify({
        'message': 'Bot started successfully',
        'timestamp': datetime.now().isoformat()
    })

@api_bp.route('/bot/stop', methods=['POST'])
@require_auth
@rate_limit(max_calls=5, window=60)
@handle_errors
def stop_bot():
    """Stop trading bot"""
    if not bot_core:
        return jsonify({'error': 'Bot core not available'}), 404
    
    bot_core.stop_bot()
    return jsonify({
        'message': 'Bot stopped successfully',
        'timestamp': datetime.now().isoformat()
    })

@api_bp.route('/bot/toggle-mode', methods=['POST'])
@require_auth
@rate_limit(max_calls=10, window=60)
@handle_errors
def toggle_trading_mode():
    """Toggle between paper and live trading"""
    if not bot_core:
        return jsonify({'error': 'Bot core not available'}), 404
    
    paper_mode = bot_core.toggle_paper_mode()
    return jsonify({
        'paper_mode': paper_mode,
        'mode': 'paper' if paper_mode else 'live',
        'timestamp': datetime.now().isoformat()
    })

# Portfolio Endpoints
@api_bp.route('/portfolio', methods=['GET'])
@rate_limit(max_calls=60, window=60)
@handle_errors
def get_portfolio():
    """Get portfolio summary"""
    if not bot_core or not bot_core.paper_engine:
        return jsonify({'error': 'Paper engine not available'}), 404
    
    portfolio = bot_core.paper_engine.get_portfolio_summary()
    return jsonify(portfolio)

@api_bp.route('/portfolio/positions', methods=['GET'])
@rate_limit(max_calls=60, window=60)
@handle_errors
def get_positions():
    """Get current positions"""
    if not bot_core or not bot_core.paper_engine:
        return jsonify({'error': 'Paper engine not available'}), 404
    
    portfolio = bot_core.paper_engine.get_portfolio_summary()
    return jsonify({
        'positions': portfolio.get('positions', []),
        'timestamp': datetime.now().isoformat()
    })

@api_bp.route('/portfolio/trades', methods=['GET'])
@rate_limit(max_calls=60, window=60)
@handle_errors
def get_trade_history():
    """Get trade history"""
    if not bot_core or not bot_core.paper_engine:
        return jsonify({'error': 'Paper engine not available'}), 404
    
    # Get query parameters
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    trades = bot_core.paper_engine.get_order_history()
    
    # Apply pagination
    paginated_trades = trades[offset:offset + limit]
    
    return jsonify({
        'trades': paginated_trades,
        'total': len(trades),
        'limit': limit,
        'offset': offset,
        'timestamp': datetime.now().isoformat()
    })

# Scanner Endpoints
@api_bp.route('/scanner/status', methods=['GET'])
@rate_limit(max_calls=30, window=60)
@handle_errors
def get_scanner_status():
    """Get scanner status"""
    if not scanner:
        return jsonify({'error': 'Scanner not available'}), 404
    
    return jsonify(scanner.get_scanner_status())

@api_bp.route('/scanner/top-stocks', methods=['GET'])
@rate_limit(max_calls=60, window=60)
@handle_errors
def get_top_stocks():
    """Get top stocks from scanner"""
    if not scanner:
        return jsonify({'error': 'Scanner not available'}), 404
    
    count = request.args.get('count', 10, type=int)
    count = min(max(count, 1), 50)  # Limit between 1 and 50
    
    top_stocks = scanner.get_top_stocks(count)
    
    return jsonify({
        'stocks': top_stocks,
        'count': len(top_stocks),
        'timestamp': datetime.now().isoformat()
    })

@api_bp.route('/scanner/force-scan', methods=['POST'])
@require_auth
@rate_limit(max_calls=5, window=300)  # Only 5 force scans per 5 minutes
@handle_errors
def force_scanner_update():
    """Force immediate scanner update"""
    if not scanner:
        return jsonify({'error': 'Scanner not available'}), 404
    
    result = scanner.force_scan()
    return jsonify(result)

@api_bp.route('/scanner/config', methods=['GET', 'PUT'])
@rate_limit(max_calls=20, window=60)
@handle_errors
def scanner_config():
    """Get or update scanner configuration"""
    if not scanner:
        return jsonify({'error': 'Scanner not available'}), 404
    
    if request.method == 'GET':
        return jsonify(scanner.get_scanner_status())
    
    elif request.method == 'PUT':
        if not require_auth:
            return jsonify({'error': 'Authentication required'}), 401
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Update scanner interval if provided
        if 'scan_interval_minutes' in data:
            interval = data['scan_interval_minutes']
            if 5 <= interval <= 120:
                success = scanner.update_scan_interval(interval)
                if success:
                    return jsonify({'message': f'Scan interval updated to {interval} minutes'})
                else:
                    return jsonify({'error': 'Failed to update scan interval'}), 500
            else:
                return jsonify({'error': 'Scan interval must be between 5 and 120 minutes'}), 400
        
        return jsonify({'message': 'Configuration updated'})

# Execution Manager Endpoints
@api_bp.route('/execution/status', methods=['GET'])
@rate_limit(max_calls=30, window=60)
@handle_errors
def get_execution_status():
    """Get execution manager status"""
    if not execution_manager:
        return jsonify({'error': 'Execution manager not available'}), 404
    
    return jsonify(execution_manager.get_execution_summary())

@api_bp.route('/execution/signals', methods=['GET'])
@rate_limit(max_calls=60, window=60)
@handle_errors
def get_recent_signals():
    """Get recent trading signals"""
    if not execution_manager:
        return jsonify({'error': 'Execution manager not available'}), 404
    
    # Get query parameters
    limit = request.args.get('limit', 20, type=int)
    hours = request.args.get('hours', 24, type=int)
    
    # This would need to be implemented in execution_manager
    # For now, return a placeholder response
    return jsonify({
        'signals': [],
        'limit': limit,
        'hours': hours,
        'message': 'Signal history endpoint - to be implemented',
        'timestamp': datetime.now().isoformat()
    })

@api_bp.route('/execution/strategies', methods=['GET'])
@rate_limit(max_calls=30, window=60)
@handle_errors
def get_strategy_status():
    """Get strategy status and performance"""
    if not strategy_manager:
        return jsonify({'error': 'Strategy manager not available'}), 404
    
    return jsonify(strategy_manager.get_strategy_status())

@api_bp.route('/execution/strategies/<strategy_name>/toggle', methods=['POST'])
@require_auth
@rate_limit(max_calls=10, window=60)
@handle_errors
def toggle_strategy(strategy_name):
    """Enable/disable a specific strategy"""
    if not strategy_manager:
        return jsonify({'error': 'Strategy manager not available'}), 404
    
    data = request.get_json() or {}
    enabled = data.get('enabled', True)
    
    if enabled:
        strategy_manager.enable_strategy(strategy_name)
    else:
        strategy_manager.disable_strategy(strategy_name)
    
    return jsonify({
        'strategy': strategy_name,
        'enabled': enabled,
        'message': f'Strategy {strategy_name} {"enabled" if enabled else "disabled"}',
        'timestamp': datetime.now().isoformat()
    })

# Market Data Endpoints
@api_bp.route('/market/quote/<symbol>', methods=['GET'])
@rate_limit(max_calls=100, window=60)
@handle_errors
def get_quote(symbol):
    """Get current quote for symbol"""
    if not data_manager:
        return jsonify({'error': 'Data manager not available'}), 404
    
    symbol = symbol.upper()
    latest_data = data_manager.get_latest_data(symbol)
    
    if not latest_data:
        return jsonify({'error': f'No data available for {symbol}'}), 404
    
    return jsonify({
        'symbol': symbol,
        'data': latest_data,
        'timestamp': datetime.now().isoformat()
    })

@api_bp.route('/market/quotes', methods=['POST'])
@rate_limit(max_calls=50, window=60)
@handle_errors
def get_multiple_quotes():
    """Get quotes for multiple symbols"""
    if not data_manager:
        return jsonify({'error': 'Data manager not available'}), 404
    
    data = request.get_json()
    if not data or 'symbols' not in data:
        return jsonify({'error': 'Symbols list required'}), 400
    
    symbols = [s.upper() for s in data['symbols'][:20]]  # Limit to 20 symbols
    quotes = {}
    
    for symbol in symbols:
        latest_data = data_manager.get_latest_data(symbol)
        if latest_data:
            quotes[symbol] = latest_data
    
    return jsonify({
        'quotes': quotes,
        'requested_symbols': symbols,
        'found_symbols': len(quotes),
        'timestamp': datetime.now().isoformat()
    })

@api_bp.route('/market/historical/<symbol>', methods=['GET'])
@rate_limit(max_calls=30, window=60)
@handle_errors
def get_historical_data(symbol):
    """Get historical data for symbol"""
    if not data_manager:
        return jsonify({'error': 'Data manager not available'}), 404
    
    symbol = symbol.upper()
    hours = request.args.get('hours', 24, type=int)
    hours = min(max(hours, 1), 168)  # Limit between 1 hour and 1 week
    
    historical_data = data_manager.get_historical_data(symbol, hours)
    
    return jsonify({
        'symbol': symbol,
        'data': historical_data,
        'hours': hours,
        'data_points': len(historical_data),
        'timestamp': datetime.now().isoformat()
    })

# Volatility Endpoints
@api_bp.route('/volatility/summary', methods=['GET'])
@rate_limit(max_calls=30, window=60)
@handle_errors
def get_volatility_summary():
    """Get volatility analysis summary"""
    if not volatility_analyzer:
        return jsonify({'error': 'Volatility analyzer not available'}), 404
    
    return jsonify(volatility_analyzer.get_volatility_summary())

@api_bp.route('/volatility/alerts', methods=['GET'])
@rate_limit(max_calls=60, window=60)
@handle_errors
def get_volatility_alerts():
    """Get recent volatility alerts"""
    if not volatility_analyzer:
        return jsonify({'error': 'Volatility analyzer not available'}), 404
    
    hours = request.args.get('hours', 24, type=int)
    resolved = request.args.get('resolved')
    
    # Convert string to boolean
    if resolved is not None:
        resolved = resolved.lower() in ('true', '1', 'yes')
    
    alerts = volatility_analyzer.get_alerts(resolved=resolved, hours=hours)
    
    return jsonify({
        'alerts': alerts,
        'hours': hours,
        'resolved_filter': resolved,
        'count': len(alerts),
        'timestamp': datetime.now().isoformat()
    })

# Trading Orders Endpoints
@api_bp.route('/orders', methods=['GET'])
@rate_limit(max_calls=60, window=60)
@handle_errors
def get_orders():
    """Get order history"""
    if not bot_core or not bot_core.paper_engine:
        return jsonify({'error': 'Paper engine not available'}), 404
    
    # Get query parameters
    limit = request.args.get('limit', 50, type=int)
    status = request.args.get('status', 'all')
    
    orders = bot_core.paper_engine.get_order_history()
    
    # Filter by status if specified
    if status != 'all':
        orders = [order for order in orders if order.get('status', '').lower() == status.lower()]
    
    # Apply limit
    orders = orders[:limit]
    
    return jsonify({
        'orders': orders,
        'limit': limit,
        'status_filter': status,
        'count': len(orders),
        'timestamp': datetime.now().isoformat()
    })

@api_bp.route('/orders', methods=['POST'])
@require_auth
@rate_limit(max_calls=20, window=60)
@handle_errors
def place_order():
    """Place a new trading order"""
    if not bot_core or not bot_core.paper_engine:
        return jsonify({'error': 'Paper engine not available'}), 404
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Order data required'}), 400
    
    # Validate required fields
    required_fields = ['symbol', 'side', 'quantity']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    try:
        from paper_trading_engine import OrderType
        
        # Parse order data
        symbol = data['symbol'].upper()
        side = data['side'].upper()
        quantity = int(data['quantity'])
        order_type = OrderType.MARKET if data.get('order_type', 'MARKET').upper() == 'MARKET' else OrderType.LIMIT
        price = float(data.get('price', 0)) if order_type == OrderType.LIMIT else None
        
        # Validate data
        if side not in ['BUY', 'SELL']:
            return jsonify({'error': 'Side must be BUY or SELL'}), 400
        
        if quantity <= 0:
            return jsonify({'error': 'Quantity must be positive'}), 400
        
        if order_type == OrderType.LIMIT and (not price or price <= 0):
            return jsonify({'error': 'Price required for limit orders'}), 400
        
        # Place order
        order_id = bot_core.paper_engine.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price
        )
        
        if order_id:
            return jsonify({
                'order_id': order_id,
                'message': 'Order placed successfully',
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'order_type': order_type.value,
                'price': price,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Failed to place order'}), 500
            
    except ValueError as e:
        return jsonify({'error': f'Invalid data format: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Order placement failed: {str(e)}'}), 500

@api_bp.route('/orders/<order_id>/cancel', methods=['POST'])
@require_auth
@rate_limit(max_calls=20, window=60)
@handle_errors
def cancel_order(order_id):
    """Cancel an existing order"""
    if not bot_core or not bot_core.paper_engine:
        return jsonify({'error': 'Paper engine not available'}), 404
    
    success = bot_core.paper_engine.cancel_order(order_id)
    
    if success:
        return jsonify({
            'order_id': order_id,
            'message': 'Order cancelled successfully',
            'timestamp': datetime.now().isoformat()
        })
    else:
        return jsonify({
            'error': 'Failed to cancel order',
            'order_id': order_id,
            'message': 'Order may not exist or cannot be cancelled'
        }), 400

# Analytics Endpoints
@api_bp.route('/analytics/performance', methods=['GET'])
@rate_limit(max_calls=30, window=60)
@handle_errors
def get_performance_analytics():
    """Get performance analytics"""
    if not bot_core or not bot_core.paper_engine:
        return jsonify({'error': 'Paper engine not available'}), 404
    
    analytics = bot_core.paper_engine.get_trade_analytics()
    
    return jsonify({
        'analytics': analytics,
        'timestamp': datetime.now().isoformat()
    })

@api_bp.route('/analytics/strategies', methods=['GET'])
@rate_limit(max_calls=30, window=60)
@handle_errors
def get_strategy_analytics():
    """Get strategy performance analytics"""
    if not strategy_manager:
        return jsonify({'error': 'Strategy manager not available'}), 404
    
    report = strategy_manager.get_strategy_performance_report()
    rankings = strategy_manager.get_strategy_rankings()
    
    return jsonify({
        'performance_report': report,
        'strategy_rankings': rankings,
        'timestamp': datetime.now().isoformat()
    })

# Configuration Endpoints
@api_bp.route('/config/risk', methods=['GET', 'PUT'])
@rate_limit(max_calls=20, window=60)
@handle_errors
def risk_configuration():
    """Get or update risk management configuration"""
    # This would integrate with config_manager
    if request.method == 'GET':
        return jsonify({
            'risk_config': {
                'max_positions': 10,
                'max_daily_loss': 5000.0,
                'stop_loss_percentage': 0.5,
                'take_profit_percentage': 1.0
            },
            'timestamp': datetime.now().isoformat()
        })
    
    elif request.method == 'PUT':
        if not require_auth:
            return jsonify({'error': 'Authentication required'}), 401
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Configuration data required'}), 400
        
        # Validate and update risk configuration
        # This would use config_manager to update settings
        
        return jsonify({
            'message': 'Risk configuration updated',
            'updated_fields': list(data.keys()),
            'timestamp': datetime.now().isoformat()
        })

# System Control Endpoints
@api_bp.route('/system/restart', methods=['POST'])
@require_auth
@rate_limit(max_calls=2, window=300)  # Only 2 restarts per 5 minutes
@handle_errors
def restart_system():
    """Restart system components"""
    data = request.get_json() or {}
    component = data.get('component', 'all')
    
    restart_results = {}
    
    if component in ['all', 'scanner'] and scanner:
        try:
            scanner.stop_scanning()
            time.sleep(2)
            scanner.start_scanning()
            restart_results['scanner'] = 'restarted'
        except Exception as e:
            restart_results['scanner'] = f'failed: {str(e)}'
    
    if component in ['all', 'execution'] and execution_manager:
        try:
            execution_manager.stop_execution()
            time.sleep(2)
            execution_manager.start_execution()
            restart_results['execution'] = 'restarted'
        except Exception as e:
            restart_results['execution'] = f'failed: {str(e)}'
    
    return jsonify({
        'message': f'Restart initiated for: {component}',
        'results': restart_results,
        'timestamp': datetime.now().isoformat()
    })

@api_bp.route('/system/logs', methods=['GET'])
@require_auth
@rate_limit(max_calls=10, window=60)
@handle_errors
def get_system_logs():
    """Get recent system logs"""
    lines = request.args.get('lines', 100, type=int)
    lines = min(max(lines, 10), 1000)  # Limit between 10 and 1000 lines
    
    try:
        import os
        log_file = 'logs/production.log'
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:]
            
            return jsonify({
                'logs': [line.strip() for line in recent_lines],
                'lines_requested': lines,
                'lines_returned': len(recent_lines),
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'logs': [],
                'message': 'Log file not found',
                'timestamp': datetime.now().isoformat()
            })
    
    except Exception as e:
        return jsonify({
            'error': 'Failed to read logs',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# WebSocket Info Endpoint
@api_bp.route('/websocket/info', methods=['GET'])
@rate_limit(max_calls=30, window=60)
@handle_errors
def websocket_info():
    """Get WebSocket connection information"""
    return jsonify({
        'websocket_url': 'ws://localhost:9001',
        'connection_types': [
            'live_updates',
            'market_data',
            'trading_signals',
            'portfolio_updates',
            'alerts'
        ],
        'message_types': [
            'welcome',
            'live_update',
            'market_data',
            'trading_signal',
            'portfolio_update',
            'alert',
            'error',
            'pong'
        ],
        'timestamp': datetime.now().isoformat()
    })

# Error handlers
@api_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist',
        'timestamp': datetime.now().isoformat()
    }), 404

@api_bp.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'error': 'Method not allowed',
        'message': 'The requested method is not allowed for this endpoint',
        'timestamp': datetime.now().isoformat()
    }), 405

@api_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred',
        'timestamp': datetime.now().isoformat()
    }), 500

# API Documentation endpoint
@api_bp.route('/docs', methods=['GET'])
@rate_limit(max_calls=20, window=60)
@handle_errors
def api_documentation():
    """Get API documentation"""
    docs = {
        'title': 'Trading Bot REST API',
        'version': '1.0.0',
        'description': 'REST API for the Scalping Trading Bot system',
        'base_url': '/api/v1',
        'authentication': 'Bearer token required for write operations',
        'rate_limits': 'Various limits applied per endpoint',
        'endpoints': {
            'health': {
                'GET /health': 'Health check',
                'GET /status': 'System status'
            },
            'bot': {
                'POST /bot/start': 'Start trading bot',
                'POST /bot/stop': 'Stop trading bot',
                'POST /bot/toggle-mode': 'Toggle paper/live mode'
            },
            'portfolio': {
                'GET /portfolio': 'Get portfolio summary',
                'GET /portfolio/positions': 'Get current positions',
                'GET /portfolio/trades': 'Get trade history'
            },
            'scanner': {
                'GET /scanner/status': 'Get scanner status',
                'GET /scanner/top-stocks': 'Get top stocks',
                'POST /scanner/force-scan': 'Force scan (auth required)'
            },
            'execution': {
                'GET /execution/status': 'Get execution status',
                'GET /execution/signals': 'Get recent signals',
                'GET /execution/strategies': 'Get strategy status'
            },
            'market': {
                'GET /market/quote/<symbol>': 'Get quote for symbol',
                'POST /market/quotes': 'Get quotes for multiple symbols',
                'GET /market/historical/<symbol>': 'Get historical data'
            },
            'orders': {
                'GET /orders': 'Get order history',
                'POST /orders': 'Place new order (auth required)',
                'POST /orders/<id>/cancel': 'Cancel order (auth required)'
            },
            'analytics': {
                'GET /analytics/performance': 'Get performance analytics',
                'GET /analytics/strategies': 'Get strategy analytics'
            }
        },
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(docs)

# Usage example and testing
if __name__ == "__main__":
    from flask import Flask
    
    # Test API routes
    app = Flask(__name__)
    app.register_blueprint(api_bp)
    
    print("🔗 Testing API Routes...")
    
    # Mock components for testing
    class MockBotCore:
        running = True
        paper_mode = True
        
        def get_bot_status(self):
            return {'running': self.running, 'paper_mode': self.paper_mode}
        
        def toggle_paper_mode(self):
            self.paper_mode = not self.paper_mode
            return self.paper_mode
    
    # Set mock components
    set_components(bot_core=MockBotCore())
    
    with app.test_client() as client:
        # Test health endpoint
        response = client.get('/api/v1/health')
        print(f"Health check: {response.status_code}")
        
        # Test status endpoint
        response = client.get('/api/v1/status')
        print(f"Status check: {response.status_code}")
        
        # Test docs endpoint
        response = client.get('/api/v1/docs')
        print(f"Documentation: {response.status_code}")
    
    print("✅ API routes test completed")
