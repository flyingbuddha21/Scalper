#!/usr/bin/env python3
"""
Production Runner for FlyingBuddha Trading Bot
Works with the actual 18 files from GitHub repository
"""

import os
import sys
import json
import time
import logging
import signal
import atexit
import threading
from datetime import datetime
from pathlib import Path

# Ensure we're in the right directory
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# Set up logging
def setup_logging():
    """Setup production logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'production.log'),
            logging.FileHandler(log_dir / f'production_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

logger = setup_logging()

class ProductionRunner:
    """Production runner for FlyingBuddha Trading Bot"""
    
    def __init__(self):
        """Initialize production runner"""
        self.components = {}
        self.is_running = False
        self.startup_time = datetime.now()
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        logger.info("🚀 FlyingBuddha Trading Bot - Production Runner Starting")
        logger.info(f"📁 Working directory: {os.getcwd()}")
        logger.info(f"🐍 Python version: {sys.version}")
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown")
            self.graceful_shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        atexit.register(self.graceful_shutdown)
    
    def check_files(self) -> bool:
        """Check if all required files exist"""
        try:
            logger.info("📁 Checking required files...")
            
            # Actual files from your GitHub repository
            required_files = [
                'api_routes.py',           # REST API endpoints
                'bot_core.py',             # Trading bot orchestration
                'config_manager.py',       # Configuration management
                'data_manager.py',         # Market data management
                'database_setup.py',       # Database schemas
                'dynamic_scanner.py',      # 20-50 stock scanner
                'execution_manager.py',    # 10 scalping strategies
                'file_generator.py',       # Auto file creation
                'flask_webapp.py',         # Web application
                'goodwill_api_handler.py', # Complete API integration
                'monitoring.py',           # System monitoring
                'paper_trading_engine.py', # Paper trading engine
                'security_manager.py',     # Authentication & security
                'strategy_manager.py',     # Strategy coordination
                'utils.py',                # Common utilities
                'volatility_analyzer.py',  # Real-time volatility analysis
                'websocket_manager.py'     # Real-time WebSocket
            ]
            
            existing_files = []
            missing_files = []
            
            for file in required_files:
                if Path(file).exists():
                    existing_files.append(file)
                    logger.info(f"✅ Found: {file}")
                else:
                    missing_files.append(file)
                    logger.warning(f"❌ Missing: {file}")
            
            if missing_files:
                logger.warning(f"⚠️ Missing files: {missing_files}")
                logger.info("💡 Will try to run with available files")
            
            if len(existing_files) < 10:  # Need at least core files
                logger.error("❌ Too many critical files missing!")
                return False
            
            logger.info(f"✅ Found {len(existing_files)}/{len(required_files)} files")
            return True
            
        except Exception as e:
            logger.error(f"❌ File check error: {e}")
            return False
    
    def create_directories(self):
        """Create required directories"""
        try:
            directories = [
                'logs', 'data', 'config', 'backups', 
                'strategies', 'reports', 'temp', 'static'
            ]
            
            for directory in directories:
                Path(directory).mkdir(exist_ok=True)
                logger.debug(f"📁 Directory ensured: {directory}")
            
            logger.info("✅ Required directories created")
            
        except Exception as e:
            logger.error(f"❌ Directory creation error: {e}")
    
    def initialize_config_manager(self) -> bool:
        """Initialize configuration manager"""
        try:
            logger.info("⚙️ Initializing configuration manager...")
            
            from config_manager import ConfigManager
            self.components['config_manager'] = ConfigManager()
            
            logger.info("✅ Configuration manager initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Config manager initialization error: {e}")
            return False
    
    def initialize_security_manager(self) -> bool:
        """Initialize security manager"""
        try:
            logger.info("🔐 Initializing security manager...")
            
            from security_manager import SecurityManager
            self.components['security_manager'] = SecurityManager()
            
            logger.info("✅ Security manager initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Security manager initialization error: {e}")
            return False
    
    def initialize_database(self) -> bool:
        """Initialize database"""
        try:
            logger.info("💾 Initializing database...")
            
            from database_setup import DatabaseManager
            self.components['database'] = DatabaseManager()
            
            # Setup database if needed
            if hasattr(self.components['database'], 'setup_database'):
                self.components['database'].setup_database()
            
            logger.info("✅ Database initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database initialization error: {e}")
            return False
    
    def initialize_data_manager(self) -> bool:
        """Initialize data manager"""
        try:
            logger.info("📊 Initializing data manager...")
            
            from data_manager import DataManager
            self.components['data_manager'] = DataManager()
            
            logger.info("✅ Data manager initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data manager initialization error: {e}")
            return False
    
    def initialize_goodwill_api(self) -> bool:
        """Initialize Goodwill API handler"""
        try:
            logger.info("🔌 Initializing Goodwill API handler...")
            
            from goodwill_api_handler import GoodwillAPIHandler
            self.components['api_handler'] = GoodwillAPIHandler()
            
            logger.info("✅ Goodwill API handler initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Goodwill API handler initialization error: {e}")
            return False
    
    def initialize_dynamic_scanner(self) -> bool:
        """Initialize dynamic scanner"""
        try:
            logger.info("🔍 Initializing dynamic scanner...")
            
            from dynamic_scanner import DynamicScanner
            self.components['scanner'] = DynamicScanner()
            
            logger.info("✅ Dynamic scanner initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Dynamic scanner initialization error: {e}")
            return False
    
    def initialize_volatility_analyzer(self) -> bool:
        """Initialize volatility analyzer"""
        try:
            logger.info("📈 Initializing volatility analyzer...")
            
            from volatility_analyzer import VolatilityAnalyzer
            self.components['volatility_analyzer'] = VolatilityAnalyzer()
            
            logger.info("✅ Volatility analyzer initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Volatility analyzer initialization error: {e}")
            return False
    
    def initialize_strategy_manager(self) -> bool:
        """Initialize strategy manager"""
        try:
            logger.info("⚙️ Initializing strategy manager...")
            
            from strategy_manager import StrategyManager
            self.components['strategy_manager'] = StrategyManager()
            
            logger.info("✅ Strategy manager initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Strategy manager initialization error: {e}")
            return False
    
    def initialize_execution_manager(self) -> bool:
        """Initialize execution manager"""
        try:
            logger.info("⚡ Initializing execution manager...")
            
            from execution_manager import ExecutionManager
            self.components['execution_manager'] = ExecutionManager()
            
            logger.info("✅ Execution manager initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Execution manager initialization error: {e}")
            return False
    
    def initialize_paper_trading(self) -> bool:
        """Initialize paper trading engine"""
        try:
            logger.info("📊 Initializing paper trading engine...")
            
            from paper_trading_engine import PaperTradingEngine
            
            # Create mock API if no real API available
            if 'api_handler' in self.components:
                api = self.components['api_handler']
            else:
                class MockAPI:
                    def get_quote(self, symbol):
                        import random
                        base_prices = {'RELIANCE': 2500, 'TCS': 3500, 'INFY': 1500}
                        base = base_prices.get(symbol, 1000)
                        return {
                            'ltp': base + random.uniform(-50, 50),
                            'high': base + 100,
                            'low': base - 100,
                            'volume': random.randint(100000, 1000000)
                        }
                api = MockAPI()
            
            self.components['paper_engine'] = PaperTradingEngine(api, 100000)
            self.components['paper_engine'].start_realtime_updates()
            
            logger.info("✅ Paper trading engine initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Paper trading initialization error: {e}")
            return False
    
    def initialize_websocket_manager(self) -> bool:
        """Initialize WebSocket manager"""
        try:
            logger.info("🔌 Initializing WebSocket manager...")
            
            from websocket_manager import WebSocketManager
            self.components['websocket_manager'] = WebSocketManager()
            
            logger.info("✅ WebSocket manager initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ WebSocket manager initialization error: {e}")
            return False
    
    def initialize_monitoring(self) -> bool:
        """Initialize monitoring system"""
        try:
            logger.info("📊 Initializing monitoring system...")
            
            from monitoring import MonitoringSystem
            self.components['monitoring'] = MonitoringSystem()
            
            logger.info("✅ Monitoring system initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Monitoring system initialization error: {e}")
            return False
    
    def initialize_bot_core(self) -> bool:
        """Initialize bot core"""
        try:
            logger.info("🤖 Initializing bot core...")
            
            from bot_core import TradingBot
            self.components['bot_core'] = TradingBot()
            
            # Connect all components to bot core
            for name, component in self.components.items():
                if name != 'bot_core':
                    setattr(self.components['bot_core'], name, component)
            
            logger.info("✅ Bot core initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Bot core initialization error: {e}")
            return False
    
    def initialize_flask_webapp(self) -> bool:
        """Initialize Flask web application"""
        try:
            logger.info("🌐 Initializing Flask web application...")
            
            from flask_webapp import create_app
            
            # Create Flask app with all components
            self.components['webapp'] = create_app(self.components)
            
            logger.info("✅ Flask web application initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Flask webapp initialization error: {e}")
            return False
    
    def start_background_services(self):
        """Start background services"""
        try:
            logger.info("🔄 Starting background services...")
            
            # Start scanner if available
            if 'scanner' in self.components:
                if hasattr(self.components['scanner'], 'start_scanning'):
                    threading.Thread(
                        target=self.components['scanner'].start_scanning, 
                        daemon=True
                    ).start()
            
            # Start volatility analyzer if available
            if 'volatility_analyzer' in self.components:
                if hasattr(self.components['volatility_analyzer'], 'start_analysis'):
                    threading.Thread(
                        target=self.components['volatility_analyzer'].start_analysis, 
                        daemon=True
                    ).start()
            
            # Start monitoring if available
            if 'monitoring' in self.components:
                if hasattr(self.components['monitoring'], 'start_monitoring'):
                    threading.Thread(
                        target=self.components['monitoring'].start_monitoring, 
                        daemon=True
                    ).start()
            
            # Start bot core if available
            if 'bot_core' in self.components:
                if hasattr(self.components['bot_core'], 'start'):
                    self.components['bot_core'].start()
            
            logger.info("✅ Background services started")
            
        except Exception as e:
            logger.error(f"❌ Background services start error: {e}")
    
    def start_health_monitor(self):
        """Start health monitoring"""
        def health_monitor():
            logger.info("🏥 Health monitor started")
            
            while self.is_running:
                try:
                    # Basic health checks every 5 minutes
                    if int(time.time()) % 300 == 0:
                        status = {
                            'uptime_minutes': int((datetime.now() - self.startup_time).total_seconds() / 60),
                            'components_loaded': len(self.components),
                            'components': list(self.components.keys())
                        }
                        logger.info(f"🏥 Health Status: {status}")
                    
                    time.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error(f"❌ Health monitor error: {e}")
                    time.sleep(60)
            
            logger.info("🏥 Health monitor stopped")
        
        health_thread = threading.Thread(target=health_monitor, daemon=True)
        health_thread.start()
    
    def get_external_ip(self) -> str:
        """Get external IP address"""
        try:
            import requests
            response = requests.get('http://ifconfig.me', timeout=5)
            return response.text.strip()
        except:
            return "localhost"
    
    def print_startup_info(self):
        """Print startup information"""
        try:
            external_ip = self.get_external_ip()
            
            startup_info = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🤖 FlyingBuddha Trading Bot - READY                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🌐 Dashboard URL: http://{external_ip}:5000                          ║
║  📊 Health Check: http://{external_ip}:5000/api/health                ║
║  📈 Bot Status:   http://{external_ip}:5000/api/status                ║
║                                                                              ║
║  📁 Working Dir:  {os.getcwd():<50} ║
║  🕐 Started At:   {self.startup_time.strftime('%Y-%m-%d %H:%M:%S'):<50} ║
║  📋 Log Files:    logs/production.log                                        ║
║                                                                              ║
║  🔧 Components Loaded: {len(self.components):<45} ║
║     {', '.join(self.components.keys()):<60} ║
║                                                                              ║
║  🎯 Features Available:                                                      ║
║     📊 Dynamic Scanner: 20-50 stocks, auto-refresh every 20 minutes          ║
║     ⚡ 10 Scalping Strategies: Real-time execution on top 10 stocks          ║
║     💰 Paper Trading: ₹1,00,000 virtual capital with real market data        ║
║     🌐 Web Interface: Full dashboard at http://{external_ip}:5000     ║
║     🔄 Real-time Updates: WebSocket streaming                                ║
║     📈 Volatility Analysis: Live market analysis                             ║
║     🛡️ Risk Management: Position & loss limits                               ║
║     📊 Performance Analytics: Detailed trading reports                       ║
║                                                                              ║
║  📱 Quick API Endpoints:                                                     ║
║     GET  /api/status           - Bot status                                  ║
║     GET  /api/scanner/results  - Scanner results                             ║
║     GET  /api/volatility       - Volatility data                            ║
║     POST /api/strategies/start - Start strategies                            ║
║     POST /api/strategies/stop  - Stop strategies                             ║
║     GET  /api/paper/portfolio  - Paper trading portfolio                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
            print(startup_info)
            logger.info("✅ Startup information displayed")
            
        except Exception as e:
            logger.error(f"❌ Startup info display error: {e}")
    
    def save_startup_info(self):
        """Save startup information to file"""
        try:
            startup_data = {
                'startup_time': self.startup_time.isoformat(),
                'working_directory': os.getcwd(),
                'python_version': sys.version,
                'pid': os.getpid(),
                'components_loaded': list(self.components.keys()),
                'external_ip': self.get_external_ip(),
                'environment': 'production'
            }
            
            with open('data/startup_info.json', 'w') as f:
                json.dump(startup_data, f, indent=4)
            
            logger.info("✅ Startup info saved to data/startup_info.json")
            
        except Exception as e:
            logger.error(f"❌ Startup info save error: {e}")
    
    def graceful_shutdown(self):
        """Graceful shutdown procedure"""
        try:
            if not self.is_running:
                return
            
            logger.info("🔄 Initiating graceful shutdown...")
            self.is_running = False
            
            # Shutdown components
            for name, component in self.components.items():
                try:
                    if hasattr(component, 'shutdown'):
                        logger.info(f"🔄 Shutting down {name}...")
                        component.shutdown()
                    elif hasattr(component, 'stop'):
                        logger.info(f"🔄 Stopping {name}...")
                        component.stop()
                except Exception as e:
                    logger.error(f"❌ Error shutting down {name}: {e}")
            
            # Save final state
            self.save_shutdown_info()
            
            logger.info("✅ Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Graceful shutdown error: {e}")
    
    def save_shutdown_info(self):
        """Save shutdown information"""
        try:
            shutdown_data = {
                'shutdown_time': datetime.now().isoformat(),
                'uptime_seconds': (datetime.now() - self.startup_time).total_seconds(),
                'components_shutdown': list(self.components.keys()),
                'shutdown_reason': 'graceful'
            }
            
            with open('data/shutdown_info.json', 'w') as f:
                json.dump(shutdown_data, f, indent=4)
            
            logger.info("✅ Shutdown info saved")
            
        except Exception as e:
            logger.error(f"❌ Shutdown info save error: {e}")
    
    def run(self) -> bool:
        """Main run method"""
        try:
            logger.info("🚀 Starting FlyingBuddha Trading Bot Production System")
            
            # Pre-flight checks
            if not self.check_files():
                return False
            
            # Setup environment
            self.create_directories()
            
            # Initialize components in order
            initialization_steps = [
                ("Configuration Manager", self.initialize_config_manager),
                ("Security Manager", self.initialize_security_manager),
                ("Database", self.initialize_database),
                ("Data Manager", self.initialize_data_manager),
                ("Goodwill API", self.initialize_goodwill_api),
                ("Dynamic Scanner", self.initialize_dynamic_scanner),
                ("Volatility Analyzer", self.initialize_volatility_analyzer),
                ("Strategy Manager", self.initialize_strategy_manager),
                ("Execution Manager", self.initialize_execution_manager),
                ("Paper Trading", self.initialize_paper_trading),
                ("WebSocket Manager", self.initialize_websocket_manager),
                ("Monitoring", self.initialize_monitoring),
                ("Bot Core", self.initialize_bot_core),
                ("Flask Web App", self.initialize_flask_webapp)
            ]
            
            # Execute initialization steps
            for step_name, step_function in initialization_steps:
                try:
                    if not step_function():
                        logger.warning(f"⚠️ {step_name} initialization failed, continuing...")
                except Exception as e:
                    logger.warning(f"⚠️ {step_name} initialization error: {e}, continuing...")
            
            # Check if we have minimum required components
            if len(self.components) < 3:
                logger.error("❌ Too few components initialized!")
                return False
            
            # Mark as running
            self.is_running = True
            
            # Start background services
            self.start_background_services()
            
            # Save startup info
            self.save_startup_info()
            
            # Start health monitor
            self.start_health_monitor()
            
            # Display startup info
            self.print_startup_info()
            
            # Run Flask app (this blocks)
            if 'webapp' in self.components:
                logger.info("🎯 Starting Flask web application...")
                self.components['webapp'].run(
                    host='0.0.0.0',
                    port=5000,
                    debug=False,
                    use_reloader=False,
                    threaded=True
                )
            else:
                logger.error("❌ Flask webapp not available, entering maintenance mode...")
                # Keep running for background services
                while self.is_running:
                    time.sleep(60)
            
            return True
            
        except KeyboardInterrupt:
            logger.info("🛑 Keyboard interrupt received")
            return True
        except Exception as e:
            logger.error(f"❌ Production runner error: {e}")
            import traceback
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            return False
        finally:
            self.graceful_shutdown()

def install_dependencies():
    """Install required dependencies"""
    try:
        logger.info("📦 Installing dependencies...")
        
        dependencies = [
            'flask>=2.0.0',
            'flask-cors>=3.0.0',
            'flask-socketio>=5.0.0',
            'requests>=2.25.0',
            'websocket-client>=1.0.0',
            'pandas>=1.3.0',
            'numpy>=1.21.0',
            'ta-lib',
            'yfinance>=0.1.70',
            'schedule>=1.1.0',
            'pytz>=2021.1',
            'python-dotenv>=0.19.0'
        ]
        
        import subprocess
        for dep in dependencies:
            logger.info(f"📦 Installing {dep}...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', dep
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"⚠️ Failed to install {dep}: {result.stderr}")
        
        logger.info("✅ Dependencies installation completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dependency installation error: {e}")
        return False

def main():
    """Main entry point"""
    try:
        # Handle command line arguments
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            
            if command == 'install-deps':
                return install_dependencies()
            elif command == 'help':
                print("""
FlyingBuddha Trading Bot - Production Runner

Usage:
  python run_production.py                 # Run the trading bot
  python run_production.py install-deps    # Install dependencies
  python run_production.py help            # Show this help

Environment Variables:
  TRADING_BOT_ENV=production               # Set environment
  TRADING_BOT_PORT=5000                   # Set port (default: 5000)
  TRADING_BOT_DEBUG=false                 # Enable debug mode

The bot will automatically:
- Initialize all 18 components
- Start dynamic scanner (20-50 stocks)
- Run 10 scalping strategies
- Enable paper trading with ₹1,00,000
- Launch web dashboard on port 5000
- Monitor system health
- Handle market hours automation
""")
                return True
        
        # Run the production system
        runner = ProductionRunner()
        return runner.run()
        
    except Exception as e:
        logger.error(f"❌ Main error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
