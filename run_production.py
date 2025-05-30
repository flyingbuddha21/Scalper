#!/usr/bin/env python3
"""
Production System Launcher for Advanced Trading Bot
Generates all necessary files and starts the complete production system
"""

import os
import sys
import asyncio
import logging
import signal
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

# Flask application imports
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask import g, current_app

# Import our system components
try:
    from files_generator import ProductionFilesGenerator
    from config_manager import get_config, ConfigManager
    from database_setup import DatabaseManager
    from api_routes import api_bp, init_api_components
    from bot_core import TradingBotCore
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all required files are present in the project directory")
    sys.exit(1)

class TradingBotProductionSystem:
    """Main production system that orchestrates everything"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.app = None
        self.config_manager = None
        self.db_manager = None
        self.bot_instances = {}  # Store bot instances for multiple users
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Setup logging early
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        print("🚀 Trading Bot Production System Initializing...")
    
    def setup_logging(self):
        """Setup production logging"""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'production.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def generate_production_files(self):
        """Generate all necessary production files"""
        print("\n📁 Generating production files...")
        
        try:
            generator = ProductionFilesGenerator(self.project_root)
            generated_files = generator.generate_all_files()
            
            print(f"✅ Generated {len(generated_files)} production files")
            return True
            
        except Exception as e:
            print(f"❌ Failed to generate production files: {e}")
            return False
    
    def check_prerequisites(self):
        """Check if all prerequisites are met"""
        print("\n🔍 Checking prerequisites...")
        
        issues = []
        
        # Check Python version
        if sys.version_info < (3, 8):
            issues.append("Python 3.8+ required")
        
        # Check required files
        required_files = [
            'config_manager.py',
            'database_setup.py', 
            'bot_core.py',
            'api_routes.py'
        ]
        
        for file in required_files:
            if not (self.project_root / file).exists():
                issues.append(f"Missing required file: {file}")
        
        # Check .env file
        env_file = self.project_root / '.env'
        if not env_file.exists():
            print("⚠️  .env file not found - will use defaults")
        
        # Check config directory
        config_dir = self.project_root / 'config'
        if not config_dir.exists():
            print("📁 Creating config directory...")
            config_dir.mkdir(exist_ok=True)
        
        if issues:
            print("❌ Prerequisites check failed:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        print("✅ Prerequisites check passed")
        return True
    
    def initialize_configuration(self):
        """Initialize configuration management"""
        print("\n⚙️  Initializing configuration...")
        
        try:
            self.config_manager = get_config()
            
            # Initialize database connection for config manager
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.config_manager.initialize_database_connection())
            loop.close()
            
            print("✅ Configuration initialized")
            return True
            
        except Exception as e:
            print(f"❌ Configuration initialization failed: {e}")
            return False
    
    def initialize_database(self):
        """Initialize database system"""
        print("\n🗄️  Initializing database system...")
        
        try:
            self.db_manager = DatabaseManager(self.config_manager)
            
            # Initialize database in async context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.db_manager.initialize())
            loop.close()
            
            print("✅ Database system initialized")
            return True
            
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            print("💡 Make sure PostgreSQL is running and configured correctly")
            return False
    
    def create_flask_app(self):
        """Create and configure Flask application"""
        print("\n🌐 Creating Flask application...")
        
        try:
            self.app = Flask(__name__)
            
            # Configure Flask
            flask_config = self.config_manager.flask
            self.app.config['SECRET_KEY'] = flask_config.secret_key
            self.app.config['DEBUG'] = flask_config.debug
            self.app.config['MAX_CONTENT_LENGTH'] = flask_config.max_content_length
            
            # Register API blueprint
            self.app.register_blueprint(api_bp)
            
            # Setup Flask routes
            self.setup_flask_routes()
            
            # Setup error handlers
            self.setup_error_handlers()
            
            print("✅ Flask application created")
            return True
            
        except Exception as e:
            print(f"❌ Flask application creation failed: {e}")
            return False
    
    def setup_flask_routes(self):
        """Setup Flask web routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard"""
            if 'user_id' not in session:
                return redirect(url_for('login'))
            return render_template('dashboard.html')
        
        @self.app.route('/dashboard')
        def dashboard():
            """Dashboard page"""
            if 'user_id' not in session:
                return redirect(url_for('login'))
            return render_template('dashboard.html')
        
        @self.app.route('/login', methods=['GET', 'POST'])
        def login():
            """Login page"""
            if request.method == 'POST':
                username = request.form['username']
                password = request.form['password']
                
                # Use API endpoint for authentication
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    user_id = loop.run_until_complete(
                        self.db_manager.authenticate_user(username, password)
                    )
                    loop.close()
                    
                    if user_id:
                        session['user_id'] = user_id
                        session['username'] = username
                        flash('Login successful!', 'success')
                        return redirect(url_for('dashboard'))
                    else:
                        flash('Invalid credentials', 'error')
                        
                except Exception as e:
                    self.logger.error(f"Login error: {e}")
                    flash('Login failed', 'error')
            
            return render_template('login.html')
        
        @self.app.route('/register', methods=['GET', 'POST'])
        def register():
            """Registration page"""
            if request.method == 'POST':
                username = request.form['username']
                email = request.form['email']
                password = request.form['password']
                
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    user_id = loop.run_until_complete(
                        self.db_manager.create_user(username, email, password)
                    )
                    loop.close()
                    
                    flash('Registration successful! Please login.', 'success')
                    return redirect(url_for('login'))
                    
                except Exception as e:
                    self.logger.error(f"Registration error: {e}")
                    flash('Registration failed', 'error')
            
            return render_template('register.html')
        
        @self.app.route('/logout')
        def logout():
            """Logout"""
            user_id = session.get('user_id')
            if user_id and user_id in self.bot_instances:
                # Stop user's bot instance
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.bot_instances[user_id].stop())
                    loop.close()
                    del self.bot_instances[user_id]
                except Exception as e:
                    self.logger.error(f"Error stopping bot on logout: {e}")
            
            session.clear()
            flash('Logged out successfully', 'info')
            return redirect(url_for('login'))
        
        @self.app.route('/portfolio')
        def portfolio():
            """Portfolio page"""
            if 'user_id' not in session:
                return redirect(url_for('login'))
            return render_template('portfolio.html')
        
        @self.app.route('/trades')
        def trades():
            """Trades page"""
            if 'user_id' not in session:
                return redirect(url_for('login'))
            return render_template('trades.html')
        
        @self.app.route('/risk-management')
        def risk_management():
            """Risk management page"""
            if 'user_id' not in session:
                return redirect(url_for('login'))
            return render_template('risk_management.html')
        
        @self.app.route('/strategies')
        def strategies():
            """Strategies page"""
            if 'user_id' not in session:
                return redirect(url_for('login'))
            return render_template('strategies.html')
        
        @self.app.route('/settings')
        def settings():
            """Settings page"""
            if 'user_id' not in session:
                return redirect(url_for('login'))
            return render_template('settings.html')
        
        @self.app.route('/profile')
        def profile():
            """User profile page"""
            if 'user_id' not in session:
                return redirect(url_for('login'))
            return render_template('settings.html')  # Reuse settings template
        
        @self.app.route('/health')
        def health():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'system': 'Trading Bot Production',
                'version': '1.0.0'
            })
    
    def setup_error_handlers(self):
        """Setup Flask error handlers"""
        
        @self.app.errorhandler(404)
        def not_found(error):
            if request.path.startswith('/api/'):
                return jsonify({
                    'success': False,
                    'message': 'API endpoint not found'
                }), 404
            return render_template('error.html', error_code=404, 
                                 error_message="Page not found"), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            if request.path.startswith('/api/'):
                return jsonify({
                    'success': False,
                    'message': 'Internal server error'
                }), 500
            return render_template('error.html', error_code=500,
                                 error_message="Internal server error"), 500
        
        @self.app.errorhandler(403)
        def forbidden(error):
            return jsonify({
                'success': False,
                'message': 'Access forbidden'
            }), 403
    
    def get_or_create_bot_instance(self, user_id: str) -> Optional[TradingBotCore]:
        """Get or create bot instance for user"""
        try:
            if user_id not in self.bot_instances:
                self.bot_instances[user_id] = TradingBotCore(user_id=user_id)
                self.logger.info(f"Created bot instance for user: {user_id}")
            
            return self.bot_instances[user_id]
            
        except Exception as e:
            self.logger.error(f"Error creating bot instance for {user_id}: {e}")
            return None
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(sig, frame):
            print(f"\n🛑 Received signal {sig}, initiating graceful shutdown...")
            self.shutdown_event.set()
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start_system(self):
        """Start the complete production system"""
        print("\n🎯 Starting Trading Bot Production System...")
        
        try:
            # Initialize API components
            success = init_api_components()
            if not success:
                print("❌ Failed to initialize API components")
                return False
            
            self.is_running = True
            
            # Get Flask configuration
            flask_config = self.config_manager.flask
            
            # Start Flask application
            print(f"🌐 Starting Flask server on {flask_config.host}:{flask_config.port}")
            print(f"🔗 Access the application at: http://{flask_config.host}:{flask_config.port}")
            
            # Start the Flask app
            self.app.run(
                host=flask_config.host,
                port=flask_config.port,
                debug=flask_config.debug,
                threaded=True,
                use_reloader=False  # Disable reloader in production
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start system: {e}")
            return False
    
    def shutdown(self):
        """Gracefully shutdown the system"""
        if not self.is_running:
            return
        
        print("\n🛑 Shutting down Trading Bot Production System...")
        
        try:
            # Stop all bot instances
            if self.bot_instances:
                print("🤖 Stopping bot instances...")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                for user_id, bot in self.bot_instances.items():
                    try:
                        loop.run_until_complete(bot.stop())
                        print(f"   ✅ Stopped bot for user: {user_id}")
                    except Exception as e:
                        print(f"   ❌ Error stopping bot for {user_id}: {e}")
                
                loop.close()
            
            # Close database connections
            if self.db_manager:
                print("🗄️  Closing database connections...")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.db_manager.close())
                loop.close()
            
            # Close config manager connections
            if self.config_manager:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.config_manager.close())
                loop.close()
            
            self.is_running = False
            print("✅ System shutdown complete")
            
        except Exception as e:
            print(f"❌ Error during shutdown: {e}")
    
    def run(self):
        """Main run method"""
        print("="*60)
        print("🚀 TRADING BOT PRODUCTION SYSTEM")
        print("="*60)
        
        try:
            # Step 1: Check prerequisites
            if not self.check_prerequisites():
                return False
            
            # Step 2: Generate production files
            if not self.generate_production_files():
                return False
            
            # Step 3: Initialize configuration
            if not self.initialize_configuration():
                return False
            
            # Step 4: Initialize database
            if not self.initialize_database():
                return False
            
            # Step 5: Create Flask app
            if not self.create_flask_app():
                return False
            
            # Step 6: Setup signal handlers
            self.setup_signal_handlers()
            
            # Step 7: Start the system
            print("\n🎉 All systems initialized successfully!")
            print("📋 System Summary:")
            print(f"   📁 Project root: {self.project_root}")
            print(f"   🐍 Python version: {sys.version}")
            print(f"   🗄️  Database: PostgreSQL + SQLite")
            print(f"   🌐 Web interface: Flask")
            print(f"   🤖 Bot engine: Multi-user support")
            print(f"   📊 Risk management: User-configurable")
            
            return self.start_system()
            
        except KeyboardInterrupt:
            print("\n👋 Shutdown requested by user")
            self.shutdown()
            return True
            
        except Exception as e:
            print(f"\n💥 System error: {e}")
            self.logger.error(f"System error: {e}")
            self.shutdown()
            return False

def print_usage():
    """Print usage information"""
    print("""
Trading Bot Production System

Usage:
    python run_production.py [OPTIONS]

Options:
    --help, -h          Show this help message
    --project-root DIR  Set project root directory (default: current directory)
    --config FILE       Specify config file (default: config/config.json)
    --generate-only     Only generate files, don't start system
    --check-only        Only check prerequisites
    
Examples:
    python run_production.py
    python run_production.py --project-root /path/to/project
    python run_production.py --generate-only
    python run_production.py --check-only
    
Environment Variables:
    Set these in your .env file:
    - DATABASE_URL: PostgreSQL connection string
    - FLASK_SECRET_KEY: Flask secret key
    - GOODWILL_API_KEY: Goodwill API key
    - TRADING_MODE: 'paper' or 'live'
    
For more information, see README.md
""")

def main():
    """Main entry point"""
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Trading Bot Production System')
    parser.add_argument('--project-root', default='.', 
                       help='Project root directory (default: current directory)')
    parser.add_argument('--config', default='config/config.json',
                       help='Config file path (default: config/config.json)')
    parser.add_argument('--generate-only', action='store_true',
                       help='Only generate files, don\'t start system')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check prerequisites')
    parser.add_argument('--version', action='version', version='Trading Bot 1.0.0')
    
    args = parser.parse_args()
    
    # Create production system
    system = TradingBotProductionSystem(args.project_root)
    
    try:
        if args.check_only:
            # Only check prerequisites
            success = system.check_prerequisites()
            sys.exit(0 if success else 1)
        
        elif args.generate_only:
            # Only generate files
            success = system.generate_production_files()
            sys.exit(0 if success else 1)
        
        else:
            # Run the complete system
            success = system.run()
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
