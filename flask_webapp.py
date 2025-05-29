#!/usr/bin/env python3
"""
Flask Web Application for Trading Bot
Provides web interface and API endpoints
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
import logging
import json
import os
from datetime import datetime
import threading
import time

logger = logging.getLogger(__name__)

class TradingWebApp:
    """Flask web application for trading bot interface"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'trading_bot_secret_key_2024'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # External components (will be injected)
        self.bot_core = None
        self.scanner = None
        self.execution_manager = None
        self.volatility_analyzer = None
        
        # Setup routes
        self._setup_routes()
        self._setup_websocket_events()
        
        logger.info("🌐 Flask Web App initialized")
    
    def set_components(self, bot_core=None, scanner=None, execution_manager=None, volatility_analyzer=None):
        """Set external components"""
        self.bot_core = bot_core
        self.scanner = scanner
        self.execution_manager = execution_manager
        self.volatility_analyzer = volatility_analyzer
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard"""
            return render_template('dashboard.html')
        
        @self.app.route('/portfolio')
        def portfolio():
            """Portfolio page"""
            return render_template('portfolio.html')
        
        @self.app.route('/settings')
        def settings():
            """Settings page"""
            return render_template('settings.html')
        
        @self.app.route('/api/status')
        def api_status():
            """Get system status"""
            try:
                status = {
                    'timestamp': datetime.now().isoformat(),
                    'bot_status': self.bot_core.get_bot_status() if self.bot_core else {'running': False},
                    'scanner_status': self.scanner.get_scanner_status() if self.scanner else {'running': False},
                    'execution_status': self.execution_manager.get_execution_summary() if self.execution_manager else {'running': False}
                }
                return jsonify(status)
                
            except Exception as e:
                logger.error(f"❌ API status error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/portfolio')
        def api_portfolio():
            """Get portfolio data"""
            try:
                if self.bot_core and self.bot_core.paper_engine:
                    portfolio = self.bot_core.paper_engine.get_portfolio_summary()
                    return jsonify(portfolio)
                else:
                    return jsonify({'error': 'Paper engine not available'}), 404
                    
            except Exception as e:
                logger.error(f"❌ API portfolio error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/scanner/top-stocks')
        def api_top_stocks():
            """Get top stocks from scanner"""
            try:
                if self.scanner:
                    top_stocks = self.scanner.get_top_stocks(10)
                    return jsonify({'stocks': top_stocks})
                else:
                    return jsonify({'error': 'Scanner not available'}), 404
                    
            except Exception as e:
                logger.error(f"❌ API top stocks error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/scanner/force-scan', methods=['POST'])
        def api_force_scan():
            """Force immediate scan"""
            try:
                if self.scanner:
                    result = self.scanner.force_scan()
                    return jsonify(result)
                else:
                    return jsonify({'error': 'Scanner not available'}), 404
                    
            except Exception as e:
                logger.error(f"❌ API force scan error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/execution/summary')
        def api_execution_summary():
            """Get execution summary"""
            try:
                if self.execution_manager:
                    summary = self.execution_manager.get_execution_summary()
                    return jsonify(summary)
                else:
                    return jsonify({'error': 'Execution manager not available'}), 404
                    
            except Exception as e:
                logger.error(f"❌ API execution summary error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/volatility/summary')
        def api_volatility_summary():
            """Get volatility summary"""
            try:
                if self.volatility_analyzer:
                    summary = self.volatility_analyzer.get_volatility_summary()
                    return jsonify(summary)
                else:
                    return jsonify({'error': 'Volatility analyzer not available'}), 404
                    
            except Exception as e:
                logger.error(f"❌ API volatility summary error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/bot/toggle-mode', methods=['POST'])
        def api_toggle_mode():
            """Toggle paper/live trading mode"""
            try:
                if self.bot_core:
                    paper_mode = self.bot_core.toggle_paper_mode()
                    return jsonify({'paper_mode': paper_mode})
                else:
                    return jsonify({'error': 'Bot core not available'}), 404
                    
            except Exception as e:
                logger.error(f"❌ API toggle mode error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/orders/history')
        def api_order_history():
            """Get order history"""
            try:
                if self.bot_core and self.bot_core.paper_engine:
                    orders = self.bot_core.paper_engine.get_order_history()
                    return jsonify({'orders': orders})
                else:
                    return jsonify({'error': 'Paper engine not available'}), 404
                    
            except Exception as e:
                logger.error(f"❌ API order history error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/trades/analytics')
        def api_trade_analytics():
            """Get trade analytics"""
            try:
                if self.bot_core and self.bot_core.paper_engine:
                    analytics = self.bot_core.paper_engine.get_trade_analytics()
                    return jsonify(analytics)
                else:
                    return jsonify({'error': 'Paper engine not available'}), 404
                    
            except Exception as e:
                logger.error(f"❌ API trade analytics error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _setup_websocket_events(self):
        """Setup WebSocket events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            logger.info("🔌 WebSocket client connected")
            emit('status', {'message': 'Connected to trading bot'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            logger.info("🔌 WebSocket client disconnected")
        
        @self.socketio.on('request_update')
        def handle_request_update():
            """Handle update request"""
            try:
                # Send current status
                status_data = {
                    'bot_status': self.bot_core.get_bot_status() if self.bot_core else {},
                    'portfolio': self.bot_core.get_portfolio_summary() if self.bot_core else {},
                    'top_stocks': self.scanner.get_top_stocks(5) if self.scanner else [],
                    'timestamp': datetime.now().isoformat()
                }
                emit('live_update', status_data)
                
            except Exception as e:
                logger.error(f"❌ WebSocket update error: {e}")
                emit('error', {'message': str(e)})
    
    def start_background_updates(self):
        """Start background WebSocket updates"""
        def update_loop():
            while True:
                try:
                    # Broadcast updates every 5 seconds
                    if self.socketio:
                        status_data = {
                            'bot_status': self.bot_core.get_bot_status() if self.bot_core else {},
                            'portfolio': self.bot_core.get_portfolio_summary() if self.bot_core else {},
                            'execution_summary': self.execution_manager.get_execution_summary() if self.execution_manager else {},
                            'timestamp': datetime.now().isoformat()
                        }
                        self.socketio.emit('live_update', status_data, broadcast=True)
                    
                    time.sleep(5)  # Update every 5 seconds
                    
                except Exception as e:
                    logger.debug(f"Background update error: {e}")
                    time.sleep(5)
        
        # Start background thread
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
        
        logger.info("🔄 Background WebSocket updates started")
    
    def run_development(self, host='localhost', port=8080, debug=True):
        """Run in development mode"""
        try:
            self.start_background_updates()
            logger.info(f"🌐 Starting Flask app in development mode on {host}:{port}")
            self.socketio.run(self.app, host=host, port=port, debug=debug)
            
        except Exception as e:
            logger.error(f"❌ Development server error: {e}")
    
    def run_production(self, host='0.0.0.0', port=8080):
        """Run in production mode"""
        try:
            self.start_background_updates()
            logger.info(f"🌐 Starting Flask app in production mode on {host}:{port}")
            
            # Use gunicorn-compatible setup for production
            self.socketio.run(
                self.app, 
                host=host, 
                port=port, 
                debug=False,
                use_reloader=False,
                log_output=True
            )
            
        except Exception as e:
            logger.error(f"❌ Production server error: {e}")
    
    def get_app(self):
        """Get Flask app instance for external WSGI servers"""
        return self.app


# Create app instance for external imports
def create_app():
    """Application factory"""
    webapp = TradingWebApp()
    return webapp.get_app()


if __name__ == "__main__":
    # For testing
    webapp = TradingWebApp()
    webapp.run_development()
