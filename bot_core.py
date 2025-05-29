#!/usr/bin/env python3
"""
Enhanced Bot Core with Integrated Scheduling
Complete trading bot with market hours scheduling, paper trading, and production features
"""

import os
import sys
import json
import time
import logging
import threading
import signal
import atexit
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, Optional, List, Any, Union
from dataclasses import dataclass, asdict
import pytz
import requests
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradingSession:
    """Trading session information"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_trades: int = 0
    total_pnl: float = 0.0
    strategies_used: List[str] = None
    market_conditions: str = "NORMAL"
    session_status: str = "ACTIVE"  # ACTIVE, COMPLETED, INTERRUPTED

    def __post_init__(self):
        if self.strategies_used is None:
            self.strategies_used = []

class MarketScheduler:
    """Market Hours Scheduler for automated operations"""
    
    def __init__(self):
        """Initialize market scheduler"""
        self.ist_timezone = pytz.timezone('Asia/Kolkata')
        
        # NSE Market Hours (IST)
        self.market_open = dt_time(9, 15)    # 9:15 AM
        self.market_close = dt_time(15, 30)  # 3:30 PM
        
        # VM Operation Hours (1 hour buffer)
        self.vm_start_time = dt_time(8, 15)   # 8:15 AM
        self.vm_stop_time = dt_time(16, 30)   # 4:30 PM
        
        # Market Days (Monday=0, Sunday=6)
        self.market_days = [0, 1, 2, 3, 4]  # Monday to Friday
        
        # Market Holidays (YYYY-MM-DD format)
        self.market_holidays = [
            "2025-01-26", "2025-03-14", "2025-03-31", "2025-04-14", 
            "2025-04-18", "2025-05-01", "2025-08-15", "2025-08-16",
            "2025-10-02", "2025-10-20", "2025-11-05", "2025-12-25"
        ]
    
    def get_current_ist_time(self) -> datetime:
        """Get current time in IST"""
        return datetime.now(self.ist_timezone)
    
    def is_market_day(self, date_obj: datetime = None) -> bool:
        """Check if given date is a market day"""
        if date_obj is None:
            date_obj = self.get_current_ist_time()
        
        if date_obj.weekday() not in self.market_days:
            return False
        
        date_str = date_obj.strftime('%Y-%m-%d')
        return date_str not in self.market_holidays
    
    def is_market_hours(self, check_time: datetime = None) -> bool:
        """Check if current time is within market hours"""
        if check_time is None:
            check_time = self.get_current_ist_time()
        
        if not self.is_market_day(check_time):
            return False
        
        current_time = check_time.time()
        return self.market_open <= current_time <= self.market_close
    
    def is_vm_operation_hours(self, check_time: datetime = None) -> bool:
        """Check if VM should be running (market hours + buffer)"""
        if check_time is None:
            check_time = self.get_current_ist_time()
        
        if not self.is_market_day(check_time):
            return False
        
        current_time = check_time.time()
        return self.vm_start_time <= current_time <= self.vm_stop_time
    
    def time_until_next_vm_start(self) -> Optional[timedelta]:
        """Calculate time until next VM start"""
        now = self.get_current_ist_time()
        
        # Check today first
        today_start = now.replace(hour=8, minute=15, second=0, microsecond=0)
        if now < today_start and self.is_market_day(now):
            return today_start - now
        
        # Check next 7 days
        for days_ahead in range(1, 8):
            next_date = now + timedelta(days=days_ahead)
            if self.is_market_day(next_date):
                next_start = next_date.replace(hour=8, minute=15, second=0, microsecond=0)
                return next_start - now
        
        return None

class APIConnector:
    """API connector for broker integration"""
    
    def __init__(self, config: Dict):
        """Initialize API connector"""
        self.config = config
        self.session = requests.Session()
        self.is_connected = False
        self.last_heartbeat = 0
        
        # Session management
        self.session_token = None
        self.session_expiry = None
        
        logger.info("🔌 API Connector initialized")
    
    def authenticate(self, credentials: Dict) -> bool:
        """Authenticate with broker API"""
        try:
            # Mock authentication - replace with actual broker API
            self.session_token = "mock_token_12345"
            self.session_expiry = datetime.now() + timedelta(hours=8)
            self.is_connected = True
            
            logger.info("✅ API authentication successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ API authentication failed: {e}")
            return False
    
    def is_authenticated(self) -> bool:
        """Check if API session is valid"""
        if not self.session_token:
            return False
        
        if self.session_expiry and datetime.now() > self.session_expiry:
            return False
        
        return self.is_connected
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get market quote for symbol"""
        try:
            if not self.is_authenticated():
                return None
            
            # Mock market data - replace with actual broker API
            import random
            base_prices = {
                'RELIANCE': 2500, 'TCS': 3500, 'INFY': 1500,
                'NIFTY': 19500, 'BANKNIFTY': 45000
            }
            
            base_price = base_prices.get(symbol, 1000)
            current_price = base_price + random.uniform(-50, 50)
            
            return {
                'symbol': symbol,
                'ltp': round(current_price, 2),
                'high': round(current_price * 1.02, 2),
                'low': round(current_price * 0.98, 2),
                'volume': random.randint(100000, 1000000),
                'prev_close': base_price,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Quote fetch error for {symbol}: {e}")
            return None
    
    def place_order(self, order_params: Dict) -> Optional[Dict]:
        """Place order via API"""
        try:
            if not self.is_authenticated():
                logger.error("❌ Cannot place order - not authenticated")
                return None
            
            # Mock order placement - replace with actual broker API
            order_id = f"ORD_{int(time.time())}_{random.randint(1000, 9999)}"
            
            return {
                'order_id': order_id,
                'status': 'PLACED',
                'symbol': order_params.get('symbol'),
                'side': order_params.get('side'),
                'quantity': order_params.get('quantity'),
                'price': order_params.get('price'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Order placement error: {e}")
            return None
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            if not self.is_authenticated():
                return []
            
            # Mock positions - replace with actual broker API
            return []
            
        except Exception as e:
            logger.error(f"❌ Positions fetch error: {e}")
            return []
    
    def get_orders(self) -> List[Dict]:
        """Get order history"""
        try:
            if not self.is_authenticated():
                return []
            
            # Mock orders - replace with actual broker API
            return []
            
        except Exception as e:
            logger.error(f"❌ Orders fetch error: {e}")
            return []

class RiskManager:
    """Risk management system"""
    
    def __init__(self, config: Dict):
        """Initialize risk manager"""
        self.config = config
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_drawdown = 0.0
        self.peak_capital = config.get('initial_capital', 100000)
        
        logger.info("🛡️ Risk Manager initialized")
    
    def check_position_size(self, symbol: str, quantity: int, price: float) -> bool:
        """Check if position size is within limits"""
        try:
            max_position_value = self.config.get('max_position_size', 100000)
            position_value = quantity * price
            
            if position_value > max_position_value:
                logger.warning(f"⚠️ Position size too large: ₹{position_value:,.2f}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Position size check error: {e}")
            return False
    
    def check_daily_limits(self) -> bool:
        """Check daily trading limits"""
        try:
            max_daily_loss = self.config.get('max_daily_loss', 5000)
            max_daily_trades = self.config.get('max_daily_trades', 50)
            
            if self.daily_pnl < -max_daily_loss:
                logger.warning(f"⚠️ Daily loss limit reached: ₹{self.daily_pnl:.2f}")
                return False
            
            if self.daily_trades >= max_daily_trades:
                logger.warning(f"⚠️ Daily trades limit reached: {self.daily_trades}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Daily limits check error: {e}")
            return True
    
    def update_trade_pnl(self, pnl: float):
        """Update daily P&L from trade"""
        self.daily_pnl += pnl
        self.daily_trades += 1
        
        # Update drawdown
        current_capital = self.peak_capital + self.daily_pnl
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
        
        drawdown = self.peak_capital - current_capital
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

class StrategyEngine:
    """Trading strategy execution engine"""
    
    def __init__(self, config: Dict):
        """Initialize strategy engine"""
        self.config = config
        self.active_strategies = []
        self.strategy_performance = {}
        
        logger.info("⚙️ Strategy Engine initialized")
    
    def load_strategies(self):
        """Load and initialize trading strategies"""
        try:
            strategies_config = self.config.get('strategies', {})
            
            for strategy_name, strategy_config in strategies_config.items():
                if strategy_config.get('enabled', False):
                    self.active_strategies.append(strategy_name)
                    self.strategy_performance[strategy_name] = {
                        'trades': 0,
                        'pnl': 0.0,
                        'allocation': strategy_config.get('allocation', 0.1)
                    }
            
            logger.info(f"✅ Loaded strategies: {self.active_strategies}")
            
        except Exception as e:
            logger.error(f"❌ Strategy loading error: {e}")
    
    def generate_signals(self, symbol: str, market_data: Dict) -> List[Dict]:
        """Generate trading signals from active strategies"""
        try:
            signals = []
            
            # Mock signal generation - implement actual strategies
            for strategy in self.active_strategies:
                if strategy == 'momentum':
                    signal = self._momentum_strategy(symbol, market_data)
                elif strategy == 'mean_reversion':
                    signal = self._mean_reversion_strategy(symbol, market_data)
                else:
                    continue
                
                if signal:
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Signal generation error: {e}")
            return []
    
    def _momentum_strategy(self, symbol: str, market_data: Dict) -> Optional[Dict]:
        """Simple momentum strategy"""
        try:
            # Mock momentum logic
            import random
            if random.random() > 0.7:  # 30% chance of signal
                return {
                    'strategy': 'momentum',
                    'symbol': symbol,
                    'side': 'BUY' if random.random() > 0.5 else 'SELL',
                    'confidence': random.uniform(0.6, 0.9),
                    'target_price': market_data['ltp'] * random.uniform(1.01, 1.03)
                }
            return None
            
        except Exception as e:
            logger.debug(f"Momentum strategy error: {e}")
            return None
    
    def _mean_reversion_strategy(self, symbol: str, market_data: Dict) -> Optional[Dict]:
        """Simple mean reversion strategy"""
        try:
            # Mock mean reversion logic
            import random
            if random.random() > 0.8:  # 20% chance of signal
                return {
                    'strategy': 'mean_reversion',
                    'symbol': symbol,
                    'side': 'SELL' if random.random() > 0.5 else 'BUY',
                    'confidence': random.uniform(0.5, 0.8),
                    'target_price': market_data['ltp'] * random.uniform(0.97, 0.99)
                }
            return None
            
        except Exception as e:
            logger.debug(f"Mean reversion strategy error: {e}")
            return None

class TradingBot:
    """
    Main Trading Bot with Integrated Scheduling
    Complete solution with market hours automation, paper trading, and production features
    """
    
    def __init__(self, config_file: str = "bot_config.json"):
        """Initialize the trading bot"""
        self.config_file = config_file
        self.state_file = "bot_state.json"
        self.performance_file = "daily_performance.json"
        
        # Load configuration
        self.config = self.load_configuration()
        
        # Initialize components
        self.market_scheduler = MarketScheduler()
        self.api = APIConnector(self.config.get('api_settings', {}))
        self.risk_manager = RiskManager(self.config.get('risk_management', {}))
        self.strategy_engine = StrategyEngine(self.config.get('strategies', {}))
        
        # Session management
        self.current_session: Optional[TradingSession] = None
        self.session_history: List[TradingSession] = []
        
        # Control flags
        self.is_running = False
        self.auto_mode = True
        self.scheduler_enabled = True
        self.paper_mode = self.config.get('paper_trading', {}).get('enabled', False)
        
        # Threading
        self.scheduler_thread = None
        self.trading_thread = None
        self.health_check_interval = 300  # 5 minutes
        self.last_health_check = time.time()
        
        # Paper trading
        self.paper_engine = None
        if self.paper_mode:
            self.init_paper_trading()
        
        # Load previous state
        self.load_state()
        
        # Setup lifecycle handlers
        self.setup_lifecycle_handlers()
        
        logger.info("🤖 FlyingBuddha Trading Bot initialized")
    
    def load_configuration(self) -> Dict:
        """Load bot configuration from file"""
        try:
            default_config = {
                "trading_settings": {
                    "symbols": ["RELIANCE", "TCS", "INFY"],
                    "max_daily_trades": 50,
                    "max_position_size": 100000,
                    "auto_square_off": True,
                    "square_off_time": "15:15"
                },
                "strategies": {
                    "momentum": {"enabled": True, "allocation": 0.4},
                    "mean_reversion": {"enabled": True, "allocation": 0.3}
                },
                "risk_management": {
                    "max_daily_loss": 5000,
                    "max_position_size": 100000,
                    "max_daily_trades": 50
                },
                "paper_trading": {
                    "enabled": True,
                    "initial_capital": 100000
                },
                "api_settings": {
                    "retry_attempts": 3,
                    "request_timeout": 30
                }
            }
            
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                    # Merge configurations
                    return {**default_config, **user_config}
            else:
                # Save default config
                with open(self.config_file, 'w') as f:
                    json.dump(default_config, f, indent=4)
                return default_config
                
        except Exception as e:
            logger.error(f"❌ Configuration load error: {e}")
            return {}
    
    def save_configuration(self):
        """Save current configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logger.error(f"❌ Configuration save error: {e}")
    
    def init_paper_trading(self):
        """Initialize paper trading engine"""
        try:
            from paper_trading_engine import PaperTradingEngine
            
            initial_capital = self.config.get('paper_trading', {}).get('initial_capital', 100000)
            self.paper_engine = PaperTradingEngine(self.api, initial_capital)
            self.paper_engine.start_realtime_updates()
            
            logger.info(f"📊 Paper trading initialized with ₹{initial_capital:,.2f}")
            
        except ImportError:
            logger.warning("⚠️ Paper trading engine not available")
        except Exception as e:
            logger.error(f"❌ Paper trading initialization error: {e}")
    
    def setup_lifecycle_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown")
            self.graceful_shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        atexit.register(self.graceful_shutdown)
    
    def start(self) -> bool:
        """Start the trading bot"""
        try:
            logger.info("🚀 Starting FlyingBuddha Trading Bot")
            
            # Load strategies
            self.strategy_engine.load_strategies()
            
            # Start scheduled operations
            self.start_scheduled_operations()
            
            logger.info("✅ Trading bot started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Bot startup error: {e}")
            return False
    
    def start_scheduled_operations(self):
        """Start scheduled trading operations"""
        def scheduler_loop():
            logger.info("🕒 Scheduled operations started")
            
            while self.scheduler_enabled:
                try:
                    current_time = self.market_scheduler.get_current_ist_time()
                    
                    # Check if we should start trading
                    if self.should_start_trading():
                        self.start_trading_session()
                    
                    # Check if we should stop trading
                    elif self.should_stop_trading() and self.is_running:
                        self.stop_trading_session()
                    
                    # Health checks
                    self.perform_health_checks()
                    
                    # Log status periodically
                    if current_time.minute % 15 == 0 and current_time.second < 30:
                        self.log_status()
                    
                    # Auto square-off check
                    if self.is_running:
                        self.check_auto_square_off()
                    
                    time.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    logger.error(f"❌ Scheduler loop error: {e}")
                    time.sleep(60)
            
            logger.info("🛑 Scheduled operations stopped")
        
        self.scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self.scheduler_thread.start()
    
    def should_start_trading(self) -> bool:
        """Determine if trading should start"""
        if not self.auto_mode:
            return False
        
        if not self.market_scheduler.is_vm_operation_hours():
            return False
        
        if self.is_running:
            return False
        
        if not self.api.is_authenticated():
            logger.warning("⚠️ Cannot start - not authenticated")
            return False
        
        if not self.risk_manager.check_daily_limits():
            logger.warning("⚠️ Cannot start - daily limits reached")
            return False
        
        return True
    
    def should_stop_trading(self) -> bool:
        """Determine if trading should stop"""
        if not self.auto_mode:
            return False
        
        if not self.market_scheduler.is_market_hours():
            return True
        
        if not self.risk_manager.check_daily_limits():
            return True
        
        # Check auto square-off time
        current_time = self.market_scheduler.get_current_ist_time()
        square_off_time = self.config.get("trading_settings", {}).get("square_off_time", "15:15")
        
        try:
            hour, minute = map(int, square_off_time.split(':'))
            square_off_dt = current_time.replace(hour=hour, minute=minute, second=0)
            
            if current_time >= square_off_dt:
                return True
        except:
            pass
        
        return False
    
    def start_trading_session(self):
        """Start a new trading session"""
        try:
            if self.current_session:
                logger.warning("⚠️ Trading session already active")
                return False
            
            # Create new session
            session_id = f"SESSION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.current_session = TradingSession(
                session_id=session_id,
                start_time=datetime.now(),
                strategies_used=list(self.strategy_engine.active_strategies)
            )
            
            # Start trading
            self.is_running = True
            self.start_trading_loop()
            
            logger.info(f"🚀 Trading session started: {session_id}")
            self.save_state()
            return True
            
        except Exception as e:
            logger.error(f"❌ Trading session start error: {e}")
            self.current_session = None
            return False
    
    def stop_trading_session(self, reason: str = "SCHEDULED"):
        """Stop current trading session"""
        try:
            if not self.current_session:
                return
            
            # Close all positions if auto square-off enabled
            if self.config.get("trading_settings", {}).get("auto_square_off", True):
                self.close_all_positions()
            
            # Stop trading
            self.is_running = False
            
            # Update session
            self.current_session.end_time = datetime.now()
            self.current_session.session_status = "COMPLETED"
            
            # Calculate performance
            if self.paper_engine:
                portfolio = self.paper_engine.get_portfolio_summary()
                self.current_session.total_pnl = portfolio.get('account_summary', {}).get('total_pnl', 0)
                self.current_session.total_trades = portfolio.get('performance_metrics', {}).get('total_trades', 0)
            
            # Save to history
            self.session_history.append(self.current_session)
            
            # Log session summary
            duration = (self.current_session.end_time - self.current_session.start_time).total_seconds() / 3600
            logger.info(f"🏁 Session ended: {self.current_session.session_id}")
            logger.info(f"   Duration: {duration:.1f}h | Trades: {self.current_session.total_trades} | P&L: ₹{self.current_session.total_pnl:.2f}")
            
            # Clear session
            self.current_session = None
            self.save_state()
            
        except Exception as e:
            logger.error(f"❌ Session stop error: {e}")
    
    def start_trading_loop(self):
        """Start the main trading loop"""
        def trading_loop():
            logger.info("📈 Trading loop started")
            
            while self.is_running:
                try:
                    # Process each symbol
                    symbols = self.config.get('trading_settings', {}).get('symbols', [])
                    
                    for symbol in symbols:
                        if not self.is_running:
                            break
                        
                        # Get market data
                        market_data = self.api.get_quote(symbol)
                        if not market_data:
                            continue
                        
                        # Generate signals
                        signals = self.strategy_engine.generate_signals(symbol, market_data)
                        
                        # Process signals
                        for signal in signals:
                            self.process_trading_signal(signal, market_data)
                    
                    time.sleep(5)  # Wait 5 seconds between cycles
                    
                except Exception as e:
                    logger.error(f"❌ Trading loop error: {e}")
                    time.sleep(10)
            
            logger.info("📉 Trading loop stopped")
        
        self.trading_thread = threading.Thread(target=trading_loop, daemon=True)
        self.trading_thread.start()
    
    def process_trading_signal(self, signal: Dict, market_data: Dict):
        """Process a trading signal"""
        try:
            symbol = signal['symbol']
            side = signal['side']
            
            # Risk checks
            if not self.risk_manager.check_daily_limits():
                return
            
            # Calculate position size (simple logic)
            allocation = self.strategy_engine.strategy_performance[signal['strategy']]['allocation']
            capital = self.config.get('paper_trading', {}).get('initial_capital', 100000)
            position_value = capital * allocation * 0.1  # 10% of allocation per trade
            quantity = int(position_value / market_data['ltp'])
            
            if quantity <= 0:
                return
            
            # Position size check
            if not self.risk_manager.check_position_size(symbol, quantity, market_data['ltp']):
                return
            
            # Place order
            if self.paper_mode and self.paper_engine:
                from paper_trading_engine import OrderType
                order_id = self.paper_engine.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=OrderType.MARKET
                )
                
                if order_id:
                    logger.info(f"📝 Paper order placed: {side} {quantity} {symbol}")
            else:
                # Live trading
                order_params = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'order_type': 'MARKET'
                }
                
                result = self.api.place_order(order_params)
                if result:
                    logger.info(f"📝 Live order placed: {side} {quantity} {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Signal processing error: {e}")
    
    def close_all_positions(self):
        """Close all open positions"""
        try:
            if self.paper_mode and self.paper_engine:
                order_ids = self.paper_engine.close_all_positions()
                logger.info(f"🔒 Paper positions closed: {len(order_ids)} orders")
            else:
                positions = self.api.get_positions()
                for position in positions:
                    # Close position logic for live trading
                    pass
                logger.info("🔒 Live positions closed")
                
        except Exception as e:
            logger.error(f"❌ Position closing error: {e}")
    
    def check_auto_square_off(self):
        """Check and execute auto square-off"""
        try:
            if not self.config.get("trading_settings", {}).get("auto_square_off", True):
                return
            
            current_time = self.market_scheduler.get_current_ist_time()
            square_off_time = self.config.get("trading_settings", {}).get("square_off_time", "15:15")
            
            hour, minute = map(int, square_off_time.split(':'))
            square_off_dt = current_time.replace(hour=hour, minute=minute, second=0)
            square_off_trigger = square_off_dt - timedelta(minutes=5)
            
            if current_time >= square_off_trigger and self.is_running:
                logger.info("🔒 Auto square-off triggered")
                self.close_all_positions()
                
        except Exception as e:
            logger.error(f"❌ Auto square-off error: {e}")
    
    def perform_health_checks(self):
        """Perform system health checks"""
        try:
            current_time = time.time()
            if current_time - self.last_health_check < self.health_check_interval:
                return
            
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'api_connected': self.api.is_authenticated(),
                'trading_active': self.is_running,
                'paper_mode': self.paper_mode,
                'memory_usage': self.get_memory_usage(),
                'session_active': self.current_session is not None
            }
            
            # Log critical issues
            if not health_status['api_connected'] and self.is_running:
                logger.error("❌ API disconnected during active trading")
            
            if health_status['memory_usage'] > 80:
                logger.warning(f"⚠️ High memory usage: {health_status['memory_usage']:.1f}%")
            
            self.last_health_check = current_time
            
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0
        except Exception:
            return 0.0
    
    def log_status(self):
        """Log current bot status"""
        try:
            current_time = self.market_scheduler.get_current_ist_time()
            
            status = {
                'time': current_time.strftime('%Y-%m-%d %H:%M:%S IST'),
                'market_hours': self.market_scheduler.is_market_hours(),
                'vm_hours': self.market_scheduler.is_vm_operation_hours(),
                'auto_mode': self.auto_mode,
                'trading_active': self.is_running,
                'paper_mode': self.paper_mode,
                'session_active': self.current_session is not None,
                'authenticated': self.api.is_authenticated()
            }
            
            if self.current_session:
                status['session_id'] = self.current_session.session_id
                status['session_trades'] = self.current_session.total_trades
                status['session_pnl'] = self.current_session.total_pnl
            
            logger.info(f"📊 Bot Status: {status}")
            
        except Exception as e:
            logger.error(f"❌ Status logging error: {e}")
    
    def load_state(self):
        """Load previous state from file"""
        try:
            if not os.path.exists(self.state_file):
                return
            
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)
            
            self.auto_mode = state_data.get('auto_mode', True)
            self.scheduler_enabled = state_data.get('scheduler_enabled', True)
            
            # Load session history
            session_history = state_data.get('session_history', [])
            self.session_history = []
            
            for session_data in session_history:
                session_data['start_time'] = datetime.fromisoformat(session_data['start_time'])
                if session_data['end_time']:
                    session_data['end_time'] = datetime.fromisoformat(session_data['end_time'])
                
                session = TradingSession(**session_data)
                self.session_history.append(session)
            
            logger.info("✅ Previous state loaded")
            
        except Exception as e:
            logger.error(f"❌ State load error: {e}")
    
    def save_state(self):
        """Save current state to file"""
        try:
            state_data = {
                'current_session': asdict(self.current_session) if self.current_session else None,
                'session_history': [asdict(session) for session in self.session_history[-10:]],
                'auto_mode': self.auto_mode,
                'scheduler_enabled': self.scheduler_enabled,
                'last_updated': datetime.now().isoformat()
            }
            
            def datetime_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object {obj} is not JSON serializable")
            
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=4, default=datetime_serializer)
                
        except Exception as e:
            logger.error(f"❌ State save error: {e}")
    
    def save_daily_performance(self):
        """Save daily performance metrics"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            performance_data = {}
            if os.path.exists(self.performance_file):
                with open(self.performance_file, 'r') as f:
                    performance_data = json.load(f)
            
            # Calculate today's metrics
            today_sessions = [s for s in self.session_history 
                            if s.start_time.strftime('%Y-%m-%d') == today]
            
            total_trades = sum(s.total_trades for s in today_sessions)
            total_pnl = sum(s.total_pnl for s in today_sessions)
            total_time = sum((s.end_time - s.start_time).total_seconds() 
                           for s in today_sessions if s.end_time) / 3600
            
            performance_data[today] = {
                'total_sessions': len(today_sessions),
                'total_trades': total_trades,
                'total_pnl': round(total_pnl, 2),
                'total_hours': round(total_time, 2),
                'avg_pnl_per_trade': round(total_pnl / total_trades, 2) if total_trades > 0 else 0,
                'paper_mode': self.paper_mode,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.performance_file, 'w') as f:
                json.dump(performance_data, f, indent=4)
                
        except Exception as e:
            logger.error(f"❌ Performance save error: {e}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        try:
            current_time = self.market_scheduler.get_current_ist_time()
            
            dashboard_data = {
                'system_status': {
                    'current_time': current_time.strftime('%Y-%m-%d %H:%M:%S IST'),
                    'market_day': self.market_scheduler.is_market_day(),
                    'market_hours': self.market_scheduler.is_market_hours(),
                    'vm_operation_hours': self.market_scheduler.is_vm_operation_hours(),
                    'auto_mode': self.auto_mode,
                    'scheduler_enabled': self.scheduler_enabled,
                    'trading_active': self.is_running,
                    'paper_mode': self.paper_mode,
                    'authenticated': self.api.is_authenticated()
                },
                'session_info': {
                    'current_session': asdict(self.current_session) if self.current_session else None,
                    'total_sessions_today': len([s for s in self.session_history 
                                               if s.start_time.date() == datetime.now().date()]),
                    'last_session': asdict(self.session_history[-1]) if self.session_history else None
                },
                'market_schedule': {
                    'next_vm_start': None,
                    'next_vm_stop': None
                },
                'performance': {
                    'today_pnl': 0.0,
                    'today_trades': 0,
                    'session_count': 0
                },
                'risk_status': {
                    'daily_pnl': self.risk_manager.daily_pnl,
                    'daily_trades': self.risk_manager.daily_trades,
                    'max_drawdown': self.risk_manager.max_drawdown,
                    'within_limits': self.risk_manager.check_daily_limits()
                }
            }
            
            # Add timing information
            next_start = self.market_scheduler.time_until_next_vm_start()
            next_stop = self.market_scheduler.time_until_vm_stop()
            
            dashboard_data['market_schedule']['next_vm_start'] = next_start.total_seconds() if next_start else None
            dashboard_data['market_schedule']['next_vm_stop'] = next_stop.total_seconds() if next_stop else None
            
            # Add performance data
            today = datetime.now().date()
            today_sessions = [s for s in self.session_history if s.start_time.date() == today]
            
            dashboard_data['performance'] = {
                'today_pnl': sum(s.total_pnl for s in today_sessions),
                'today_trades': sum(s.total_trades for s in today_sessions),
                'session_count': len(today_sessions)
            }
            
            # Add paper trading data if available
            if self.paper_mode and self.paper_engine:
                portfolio = self.paper_engine.get_portfolio_summary()
                dashboard_data['portfolio'] = portfolio
                
                analytics = self.paper_engine.get_trade_analytics()
                dashboard_data['analytics'] = analytics
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"❌ Dashboard data error: {e}")
            return {'error': str(e)}
    
    def authenticate_api(self, credentials: Dict) -> bool:
        """Authenticate with broker API"""
        try:
            return self.api.authenticate(credentials)
        except Exception as e:
            logger.error(f"❌ Authentication error: {e}")
            return False
    
    def manual_start_session(self) -> bool:
        """Manually start trading session"""
        try:
            if not self.api.is_authenticated():
                logger.error("❌ Cannot start - not authenticated")
                return False
            
            return self.start_trading_session()
            
        except Exception as e:
            logger.error(f"❌ Manual start error: {e}")
            return False
    
    def manual_stop_session(self) -> bool:
        """Manually stop trading session"""
        try:
            self.stop_trading_session("MANUAL")
            return True
            
        except Exception as e:
            logger.error(f"❌ Manual stop error: {e}")
            return False
    
    def emergency_stop(self) -> bool:
        """Emergency stop all trading activities"""
        try:
            logger.warning("🚨 EMERGENCY STOP TRIGGERED")
            
            # Stop trading immediately
            self.is_running = False
            
            # Close all positions
            self.close_all_positions()
            
            # Stop session
            if self.current_session:
                self.stop_trading_session("EMERGENCY")
            
            logger.info("✅ Emergency stop completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Emergency stop error: {e}")
            return False
    
    def toggle_auto_mode(self) -> bool:
        """Toggle automatic mode on/off"""
        try:
            self.auto_mode = not self.auto_mode
            self.save_state()
            
            mode = "enabled" if self.auto_mode else "disabled"
            logger.info(f"🔄 Auto mode {mode}")
            return self.auto_mode
            
        except Exception as e:
            logger.error(f"❌ Auto mode toggle error: {e}")
            return self.auto_mode
    
    def update_config(self, new_config: Dict) -> bool:
        """Update bot configuration"""
        try:
            self.config.update(new_config)
            self.save_configuration()
            
            # Reload components if needed
            if 'strategies' in new_config:
                self.strategy_engine.config = self.config.get('strategies', {})
                self.strategy_engine.load_strategies()
            
            if 'risk_management' in new_config:
                self.risk_manager.config = self.config.get('risk_management', {})
            
            logger.info("✅ Configuration updated")
            return True
            
        except Exception as e:
            logger.error(f"❌ Config update error: {e}")
            return False
    
    def get_performance_report(self) -> Dict:
        """Get detailed performance report"""
        try:
            if self.paper_mode and self.paper_engine:
                return self.paper_engine.get_trade_analytics()
            else:
                # Basic performance for live trading
                today_sessions = [s for s in self.session_history 
                                if s.start_time.date() == datetime.now().date()]
                
                return {
                    'total_sessions': len(today_sessions),
                    'total_trades': sum(s.total_trades for s in today_sessions),
                    'total_pnl': sum(s.total_pnl for s in today_sessions),
                    'active_strategies': self.strategy_engine.active_strategies
                }
                
        except Exception as e:
            logger.error(f"❌ Performance report error: {e}")
            return {'error': str(e)}
    
    def graceful_shutdown(self):
        """Graceful shutdown procedure"""
        try:
            logger.info("🔄 Initiating graceful shutdown...")
            
            # Stop scheduler
            self.scheduler_enabled = False
            
            # Stop trading session
            if self.current_session:
                self.stop_trading_session("SHUTDOWN")
            
            # Stop trading
            self.is_running = False
            
            # Stop paper trading
            if self.paper_engine:
                self.paper_engine.stop_realtime_updates()
            
            # Save all states
            self.save_state()
            self.save_daily_performance()
            self.save_configuration()
            
            logger.info("✅ Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Graceful shutdown error: {e}")


# Flask Integration
def create_flask_app(bot: TradingBot):
    """Create Flask application with bot integration"""
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/api/status')
    def api_status():
        """Get bot status"""
        try:
            return jsonify(bot.get_dashboard_data())
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/authenticate', methods=['POST'])
    def api_authenticate():
        """Authenticate with broker API"""
        try:
            credentials = request.get_json()
            success = bot.authenticate_api(credentials)
            
            return jsonify({
                'success': success,
                'message': 'Authentication successful' if success else 'Authentication failed'
            })
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    
    @app.route('/api/start-session', methods=['POST'])
    def api_start_session():
        """Start trading session"""
        try:
            success = bot.manual_start_session()
            
            return jsonify({
                'success': success,
                'message': 'Session started' if success else 'Failed to start session'
            })
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    
    @app.route('/api/stop-session', methods=['POST'])
    def api_stop_session():
        """Stop trading session"""
        try:
            success = bot.manual_stop_session()
            
            return jsonify({
                'success': success,
                'message': 'Session stopped' if success else 'Failed to stop session'
            })
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    
    @app.route('/api/emergency-stop', methods=['POST'])
    def api_emergency_stop():
        """Emergency stop"""
        try:
            success = bot.emergency_stop()
            
            return jsonify({
                'success': success,
                'message': 'Emergency stop executed' if success else 'Emergency stop failed'
            })
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    
    @app.route('/api/toggle-auto', methods=['POST'])
    def api_toggle_auto():
        """Toggle auto mode"""
        try:
            auto_mode = bot.toggle_auto_mode()
            
            return jsonify({
                'success': True,
                'auto_mode': auto_mode,
                'message': f'Auto mode {"enabled" if auto_mode else "disabled"}'
            })
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    
    @app.route('/api/config', methods=['GET', 'POST'])
    def api_config():
        """Get or update configuration"""
        try:
            if request.method == 'GET':
                return jsonify(bot.config)
            
            elif request.method == 'POST':
                new_config = request.get_json()
                success = bot.update_config(new_config)
                
                return jsonify({
                    'success': success,
                    'message': 'Configuration updated' if success else 'Configuration update failed',
                    'config': bot.config
                })
                
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    
    @app.route('/api/performance')
    def api_performance():
        """Get performance report"""
        try:
            return jsonify(bot.get_performance_report())
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/positions')
    def api_positions():
        """Get current positions"""
        try:
            if bot.paper_mode and bot.paper_engine:
                portfolio = bot.paper_engine.get_portfolio_summary()
                return jsonify(portfolio.get('positions', []))
            else:
                positions = bot.api.get_positions()
                return jsonify(positions)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/orders')
    def api_orders():
        """Get order history"""
        try:
            if bot.paper_mode and bot.paper_engine:
                orders = bot.paper_engine.get_order_history()
                return jsonify(orders)
            else:
                orders = bot.api.get_orders()
                return jsonify(orders)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/close-position/<symbol>', methods=['POST'])
    def api_close_position(symbol):
        """Close specific position"""
        try:
            if bot.paper_mode and bot.paper_engine:
                order_id = bot.paper_engine.close_position(symbol)
                success = bool(order_id)
            else:
                # Live trading position close
                success = False  # Implement live position closing
            
            return jsonify({
                'success': success,
                'message': f'Position {"closed" if success else "close failed"} for {symbol}'
            })
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    
    @app.route('/api/health')
    def api_health():
        """Health check endpoint"""
        try:
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'uptime': time.time() - bot.last_health_check if hasattr(bot, 'last_health_check') else 0
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    # Paper trading specific routes
    if bot.paper_mode:
        @app.route('/api/paper/summary')
        def api_paper_summary():
            """Get paper trading summary"""
            try:
                if bot.paper_engine:
                    return jsonify(bot.paper_engine.get_portfolio_summary())
                else:
                    return jsonify({'error': 'Paper trading not initialized'}), 400
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/paper/analytics')
        def api_paper_analytics():
            """Get paper trading analytics"""
            try:
                if bot.paper_engine:
                    return jsonify(bot.paper_engine.get_trade_analytics())
                else:
                    return jsonify({'error': 'Paper trading not initialized'}), 400
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/paper/reset', methods=['POST'])
        def api_paper_reset():
            """Reset paper trading portfolio"""
            try:
                if bot.paper_engine:
                    data = request.get_json() or {}
                    new_capital = data.get('capital', 100000)
                    bot.paper_engine.reset_portfolio(new_capital)
                    
                    return jsonify({
                        'success': True,
                        'message': f'Portfolio reset with ₹{new_capital:,.2f}'
                    })
                else:
                    return jsonify({'success': False, 'message': 'Paper trading not initialized'}), 400
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)}), 500
    
    return app


# Production Runner
def run_production_bot():
    """Main function to run the bot in production"""
    try:
        logger.info("🚀 Starting FlyingBuddha Trading Bot - Production Mode")
        
        # Initialize bot
        bot = TradingBot()
        
        # Start bot
        if not bot.start():
            logger.error("❌ Failed to start trading bot")
            return False
        
        # Create Flask app
        app = create_flask_app(bot)
        
        # Run Flask app
        logger.info("🌐 Starting web dashboard on port 5000")
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received")
    except Exception as e:
        logger.error(f"❌ Production runner error: {e}")
        logger.error(traceback.format_exc())
    finally:
        if 'bot' in locals():
            bot.graceful_shutdown()


# Development Runner
def run_development_bot():
    """Run bot in development mode with debugging"""
    try:
        logger.info("🔧 Starting FlyingBuddha Trading Bot - Development Mode")
        
        # Set debug logging
        logging.getLogger().setLevel(logging.DEBUG)
        
        # Initialize bot
        bot = TradingBot()
        
        # Enable paper trading for development
        bot.paper_mode = True
        if not bot.paper_engine:
            bot.init_paper_trading()
        
        # Start bot
        if not bot.start():
            logger.error("❌ Failed to start trading bot")
            return False
        
        # Create Flask app
        app = create_flask_app(bot)
        
        # Run Flask app in debug mode
        logger.info("🌐 Starting development server on port 5000")
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
        
    except KeyboardInterrupt:
        logger.info("🛑 Development mode stopped")
    except Exception as e:
        logger.error(f"❌ Development runner error: {e}")
        logger.error(traceback.format_exc())
    finally:
        if 'bot' in locals():
            bot.graceful_shutdown()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        run_development_bot()
    else:
        run_production_bot()
