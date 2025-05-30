#!/usr/bin/env python3
"""
Advanced Multi-Strategy Trading Bot Core with User Risk Management Integration
Market-ready for live trading with Goodwill API integration
Includes automatic scheduling and user-defined risk parameters from webapp
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config_manager import ConfigManager
from strategy_manager import StrategyManager
from data_manager import DataManager
from execution_manager import ExecutionManager
from paper_trading_engine import PaperTradingEngine
from goodwill_api_handler import GoodwillAPIHandler
from volatility_analyzer import VolatilityAnalyzer
from dynamic_scanner import DynamicScanner
from database_setup import DatabaseManager
from monitoring import SystemMonitor
from websocket_manager import WebSocketManager

@dataclass
class TradingSignal:
    symbol: str
    action: str  # BUY, SELL, HOLD
    strategy: str
    confidence: float
    price: float
    quantity: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class UserRiskConfig:
    """User-defined risk management configuration from webapp"""
    user_id: str
    capital: float
    risk_per_trade_percent: float
    daily_loss_limit_percent: float
    max_concurrent_trades: int
    risk_reward_ratio: float
    max_position_size_percent: float
    stop_loss_percent: float
    take_profit_percent: float
    trading_start_time: str
    trading_end_time: str
    auto_square_off: bool
    paper_trading_mode: bool
    last_updated: datetime

class TradingBotCore:
    def __init__(self, config_path: str = "config.json", user_id: str = None):
        """Initialize the trading bot core system with user risk integration"""
        # Core components
        self.config = ConfigManager(config_path)
        self.logger = self._setup_logging()
        
        # User configuration
        self.user_id = user_id or self.config.get('default_user_id', 'default_user')
        self.user_risk_config: Optional[UserRiskConfig] = None
        
        # Trading mode (will be set from user config)
        self.trading_mode = self.config.get('trading_mode', 'paper')  # 'live' or 'paper'
        
        # Initialize components
        self.data_manager = DataManager(self.config)
        self.strategy_manager = StrategyManager(self.config)
        self.execution_manager = ExecutionManager(self.config)
        self.paper_engine = PaperTradingEngine(self.config)
        self.goodwill_api = GoodwillAPIHandler(self.config)
        self.volatility_analyzer = VolatilityAnalyzer(self.config)
        self.scanner = DynamicScanner(self.config)
        self.db_manager = DatabaseManager(self.config)
        self.monitor = SystemMonitor(self.config)
        self.websocket_manager = WebSocketManager(self.config)
        
        # State management
        self.is_running = False
        self.active_positions = {}
        self.pending_orders = {}
        self.trading_signals = queue.Queue()
        
        # Risk management state
        self.daily_pnl = 0.0
        self.daily_trades_count = 0
        self.current_exposure = 0.0
        self.trading_halted_by_risk = False
        self.last_risk_config_update = None
        
        # Performance metrics
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }
        
        # Threading and Auto-Scheduling
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.lock = threading.RLock()
        self.scheduler = AsyncIOScheduler()
        
        # Market hours and auto-schedule times (IST)
        self.market_open_time = "09:15"
        self.market_close_time = "15:30"
        self.pre_market_start = "08:15"   # 1 hour before market open
        self.post_market_stop = "16:30"   # 1 hour after market close
        
        # Auto-scheduling configuration
        self.auto_schedule_enabled = self.config.get('auto_schedule', True)
        self.is_scheduled_session = False
        self.new_positions_allowed = True
        
        self.logger.info(f"Trading Bot initialized for user: {self.user_id}")
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    async def start(self):
        """Start the trading bot system with user risk integration"""
        try:
            self.logger.info(f"Starting Trading Bot Core System for user: {self.user_id}...")
            
            # Load user risk configuration FIRST
            await self.load_user_risk_config()
            
            # Apply user settings to bot configuration
            await self.apply_user_risk_settings()
            
            # Setup auto-scheduling based on user preferences
            if self.auto_schedule_enabled:
                await self._setup_auto_schedule()
            
            # Initialize database
            await self.db_manager.initialize()
            
            # Start monitoring
            await self.monitor.start()
            
            # Initialize API connections based on user preference
            if self.trading_mode == 'live':
                await self.goodwill_api.initialize()
                if not await self.goodwill_api.authenticate():
                    raise Exception("Failed to authenticate with Goodwill API")
                self.logger.info("✅ Live trading mode - Goodwill API connected")
            else:
                await self.paper_engine.initialize()
                self.logger.info("✅ Paper trading mode - Simulation engine ready")
            
            # Start data feeds
            await self.data_manager.start()
            
            # Start WebSocket connections for real-time data
            await self.websocket_manager.start()
            
            # Initialize market scanner
            await self.scanner.start()
            
            # Start main trading loop
            self.is_running = True
            
            # Start scheduler if auto-schedule is enabled
            if self.auto_schedule_enabled and not self.scheduler.running:
                self.scheduler.start()
                self.logger.info("✅ Auto-scheduler started")
            
            # Log user configuration summary
            if self.user_risk_config:
                self.logger.info(f"✅ User Risk Config Applied:")
                self.logger.info(f"  💰 Capital: ₹{self.user_risk_config.capital:,.2f}")
                self.logger.info(f"  📊 Risk per trade: {self.user_risk_config.risk_per_trade_percent}%")
                self.logger.info(f"  🛡️ Daily loss limit: {self.user_risk_config.daily_loss_limit_percent}%")
                self.logger.info(f"  📈 Max concurrent trades: {self.user_risk_config.max_concurrent_trades}")
                self.logger.info(f"  🎯 Trading mode: {'LIVE' if not self.user_risk_config.paper_trading_mode else 'PAPER'}")
            
            # Run concurrent tasks
            tasks = [
                self._market_data_processor(),
                self._strategy_execution_loop(),
                self._order_management_loop(),
                self._user_risk_management_loop(),  # Enhanced risk management with user settings
                self._performance_tracking_loop(),
                self._market_scanner_loop(),
                self._schedule_monitor_loop(),
                self._user_config_monitor_loop()  # Monitor for user config changes
            ]
            
            await asyncio.gather(*tasks)
            
        except Exception as e:
            self.logger.error(f"Failed to start trading bot: {e}")
            await self.stop()
            raise
    
    async def load_user_risk_config(self):
        """Load user's risk management configuration from database"""
        try:
            self.logger.info(f"Loading risk configuration for user: {self.user_id}")
            
            # Get database connection (assuming PostgreSQL as per your earlier setup)
            async with self.db_manager.get_connection() as conn:
                config_row = await conn.fetchrow("""
                    SELECT * FROM user_trading_config WHERE user_id = $1
                    ORDER BY last_updated DESC LIMIT 1
                """, self.user_id)
                
                if config_row:
                    self.user_risk_config = UserRiskConfig(
                        user_id=config_row['user_id'],
                        capital=float(config_row['capital']),
                        risk_per_trade_percent=float(config_row['risk_per_trade_percent']),
                        daily_loss_limit_percent=float(config_row['daily_loss_limit_percent']),
                        max_concurrent_trades=config_row['max_concurrent_trades'],
                        risk_reward_ratio=float(config_row['risk_reward_ratio']),
                        max_position_size_percent=float(config_row['max_position_size_percent']),
                        stop_loss_percent=float(config_row['stop_loss_percent']),
                        take_profit_percent=float(config_row['take_profit_percent']),
                        trading_start_time=config_row['trading_start_time'].strftime('%H:%M'),
                        trading_end_time=config_row['trading_end_time'].strftime('%H:%M'),
                        auto_square_off=config_row['auto_square_off'],
                        paper_trading_mode=config_row['paper_trading_mode'],
                        last_updated=config_row['last_updated']
                    )
                    
                    self.last_risk_config_update = config_row['last_updated']
                    
                    self.logger.info(f"✅ User risk config loaded successfully")
                    return True
                else:
                    self.logger.warning(f"No risk configuration found for user: {self.user_id}")
                    await self._create_default_user_config()
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error loading user risk config: {e}")
            await self._create_default_user_config()
            return False
    
    async def _create_default_user_config(self):
        """Create default user configuration if none exists"""
        try:
            self.logger.info("Creating default user configuration...")
            
            default_config = UserRiskConfig(
                user_id=self.user_id,
                capital=100000.0,  # Default ₹1 lakh
                risk_per_trade_percent=2.0,
                daily_loss_limit_percent=5.0,
                max_concurrent_trades=5,
                risk_reward_ratio=2.0,
                max_position_size_percent=20.0,
                stop_loss_percent=3.0,
                take_profit_percent=6.0,
                trading_start_time="09:15",
                trading_end_time="15:30",
                auto_square_off=True,
                paper_trading_mode=True,
                last_updated=datetime.now()
            )
            
            # Save to database
            async with self.db_manager.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO user_trading_config (
                        user_id, capital, risk_per_trade_percent, daily_loss_limit_percent,
                        max_concurrent_trades, risk_reward_ratio, max_position_size_percent,
                        stop_loss_percent, take_profit_percent, trading_start_time,
                        trading_end_time, auto_square_off, paper_trading_mode, last_updated
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
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
                        last_updated = EXCLUDED.last_updated
                """, 
                default_config.user_id, default_config.capital, default_config.risk_per_trade_percent,
                default_config.daily_loss_limit_percent, default_config.max_concurrent_trades,
                default_config.risk_reward_ratio, default_config.max_position_size_percent,
                default_config.stop_loss_percent, default_config.take_profit_percent,
                datetime.strptime(default_config.trading_start_time, '%H:%M').time(),
                datetime.strptime(default_config.trading_end_time, '%H:%M').time(),
                default_config.auto_square_off, default_config.paper_trading_mode,
                default_config.last_updated)
            
            self.user_risk_config = default_config
            self.logger.info("✅ Default user configuration created")
            
        except Exception as e:
            self.logger.error(f"Error creating default user config: {e}")
    
    async def apply_user_risk_settings(self):
        """Apply user risk settings to bot configuration"""
        try:
            if not self.user_risk_config:
                self.logger.warning("No user risk config to apply")
                return
            
            # Update trading mode based on user preference
            self.trading_mode = 'paper' if self.user_risk_config.paper_trading_mode else 'live'
            
            # Calculate derived risk parameters
            self.max_position_value = (self.user_risk_config.capital * 
                                     self.user_risk_config.max_position_size_percent) / 100
            
            self.daily_loss_limit = (self.user_risk_config.capital * 
                                   self.user_risk_config.daily_loss_limit_percent) / 100
            
            self.risk_per_trade_amount = (self.user_risk_config.capital * 
                                        self.user_risk_config.risk_per_trade_percent) / 100
            
            # Update market hours from user config
            self.market_open_time = self.user_risk_config.trading_start_time
            self.market_close_time = self.user_risk_config.trading_end_time
            
            # Reset daily counters
            self.daily_pnl = 0.0
            self.daily_trades_count = 0
            self.trading_halted_by_risk = False
            
            self.logger.info(f"✅ User risk settings applied:")
            self.logger.info(f"  💰 Max position value: ₹{self.max_position_value:,.2f}")
            self.logger.info(f"  🛡️ Daily loss limit: ₹{self.daily_loss_limit:,.2f}")
            self.logger.info(f"  📊 Risk per trade: ₹{self.risk_per_trade_amount:,.2f}")
            self.logger.info(f"  📈 Max concurrent trades: {self.user_risk_config.max_concurrent_trades}")
            self.logger.info(f"  🕒 Trading hours: {self.market_open_time} - {self.market_close_time}")
            
        except Exception as e:
            self.logger.error(f"Error applying user risk settings: {e}")
            raise
    
    async def _user_config_monitor_loop(self):
        """Monitor for changes in user configuration"""
        while self.is_running:
            try:
                # Check for config updates every 30 seconds
                async with self.db_manager.get_connection() as conn:
                    latest_update = await conn.fetchval("""
                        SELECT last_updated FROM user_trading_config 
                        WHERE user_id = $1
                    """, self.user_id)
                    
                    if (latest_update and self.last_risk_config_update and 
                        latest_update > self.last_risk_config_update):
                        
                        self.logger.info("User configuration updated - reloading...")
                        await self.load_user_risk_config()
                        await self.apply_user_risk_settings()
                        
                        await self.monitor.send_alert(
                            "CONFIG_UPDATED", 
                            f"User {self.user_id} configuration updated and applied"
                        )
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"User config monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _user_risk_management_loop(self):
        """Enhanced risk management loop using user-defined parameters"""
        while self.is_running:
            try:
                # Calculate current portfolio metrics
                portfolio_value = await self._calculate_portfolio_value()
                daily_pnl = await self._calculate_daily_pnl()
                current_positions = len(self.active_positions)
                
                with self.lock:
                    self.daily_pnl = daily_pnl
                    self.current_exposure = sum(pos.get('value', 0) for pos in self.active_positions.values())
                
                # User-defined daily loss limit check
                if self.user_risk_config and abs(daily_pnl) > self.daily_loss_limit and daily_pnl < 0:
                    if not self.trading_halted_by_risk:
                        self.logger.critical(f"🚨 DAILY LOSS LIMIT EXCEEDED: {daily_pnl:,.2f} > {self.daily_loss_limit:,.2f}")
                        self.trading_halted_by_risk = True
                        self.new_positions_allowed = False
                        
                        await self.monitor.send_alert(
                            "DAILY_LOSS_LIMIT", 
                            f"Trading halted - Daily loss: ₹{daily_pnl:,.2f}"
                        )
                        
                        # Auto square off if enabled
                        if self.user_risk_config.auto_square_off:
                            await self._emergency_close_all_positions("Daily loss limit exceeded")
                
                # User-defined max concurrent trades check
                if (self.user_risk_config and 
                    current_positions >= self.user_risk_config.max_concurrent_trades):
                    self.new_positions_allowed = False
                    self.logger.warning(f"Max concurrent trades reached: {current_positions}/{self.user_risk_config.max_concurrent_trades}")
                elif not self.trading_halted_by_risk:
                    self.new_positions_allowed = True
                
                # Check individual position sizes against user limits
                await self._check_user_position_limits()
                
                # Update performance metrics
                with self.lock:
                    self.performance_metrics['daily_pnl'] = daily_pnl
                    self.performance_metrics['total_pnl'] = portfolio_value - self.user_risk_config.capital if self.user_risk_config else 0
                    self.performance_metrics['current_exposure'] = self.current_exposure
                    self.performance_metrics['positions_count'] = current_positions
                    self.performance_metrics['risk_status'] = 'HALTED' if self.trading_halted_by_risk else 'ACTIVE'
                
                await asyncio.sleep(5)  # Check every 5 seconds for real-time risk management
                
            except Exception as e:
                self.logger.error(f"User risk management error: {e}")
                await asyncio.sleep(30)
    
    async def _check_user_position_limits(self):
        """Check individual position sizes against user-defined limits"""
        try:
            if not self.user_risk_config:
                return
            
            for symbol, position in self.active_positions.items():
                position_value = position.get('value', 0)
                
                # Check against max position size
                if position_value > self.max_position_value:
                    self.logger.warning(f"Position {symbol} exceeds max size: ₹{position_value:,.2f} > ₹{self.max_position_value:,.2f}")
                    
                    # Optionally reduce position size
                    if self.user_risk_config.auto_square_off:
                        await self._reduce_position_size(symbol, position_value, self.max_position_value)
        
        except Exception as e:
            self.logger.error(f"Position limit check error: {e}")
    
    async def _reduce_position_size(self, symbol: str, current_value: float, max_value: float):
        """Reduce position size to comply with user limits"""
        try:
            position = self.active_positions.get(symbol, {})
            current_quantity = position.get('quantity', 0)
            current_price = await self.data_manager.get_current_price(symbol)
            
            if current_quantity > 0 and current_price:
                # Calculate quantity to sell to reach max value
                target_quantity = int(max_value / current_price)
                quantity_to_sell = current_quantity - target_quantity
                
                if quantity_to_sell > 0:
                    self.logger.info(f"Reducing {symbol} position by {quantity_to_sell} shares")
                    await self._execute_exit_order(symbol, quantity_to_sell, 'POSITION_SIZE_LIMIT', current_price)
        
        except Exception as e:
            self.logger.error(f"Error reducing position size for {symbol}: {e}")
    
    async def _calculate_position_size_for_signal(self, signal: TradingSignal) -> int:
        """Calculate position size based on user risk parameters"""
        try:
            if not self.user_risk_config:
                return signal.quantity
            
            # Risk-based position sizing
            risk_amount = self.risk_per_trade_amount
            
            # Calculate stop loss distance
            if signal.stop_loss:
                stop_loss_distance = abs(signal.price - signal.stop_loss)
                
                # Position size = Risk Amount / Stop Loss Distance
                calculated_quantity = int(risk_amount / stop_loss_distance)
            else:
                # Default stop loss based on user percentage
                default_stop_distance = signal.price * (self.user_risk_config.stop_loss_percent / 100)
                calculated_quantity = int(risk_amount / default_stop_distance)
            
            # Ensure position doesn't exceed max position value
            max_quantity_by_value = int(self.max_position_value / signal.price)
            
            # Use the smaller of the two
            final_quantity = min(calculated_quantity, max_quantity_by_value, signal.quantity)
            
            self.logger.debug(f"Position sizing for {signal.symbol}: "
                            f"Risk-based: {calculated_quantity}, "
                            f"Value-based: {max_quantity_by_value}, "
                            f"Final: {final_quantity}")
            
            return max(1, final_quantity)  # At least 1 share
            
        except Exception as e:
            self.logger.error(f"Position sizing error: {e}")
            return signal.quantity
    
    async def _validate_signal_with_user_risk(self, signal: TradingSignal) -> Tuple[bool, str]:
        """Enhanced signal validation using user risk parameters"""
        try:
            if not self.user_risk_config:
                return await self._validate_signal_basic(signal)
            
            # Basic validation first
            if not signal.symbol or not signal.action or signal.confidence < 0.6:
                return False, "Basic signal validation failed"
            
            # Check if trading is halted by risk management
            if self.trading_halted_by_risk:
                return False, "Trading halted due to risk limits"
            
            # Check if new positions are allowed
            if signal.action == 'BUY' and not self.new_positions_allowed:
                return False, "New positions not allowed"
            
            # Check trading hours based on user config
            if not self._is_within_user_trading_hours():
                return False, "Outside user-defined trading hours"
            
            # Check max concurrent trades
            if (signal.action == 'BUY' and 
                len(self.active_positions) >= self.user_risk_config.max_concurrent_trades):
                return False, f"Max concurrent trades limit reached ({self.user_risk_config.max_concurrent_trades})"
            
            # Check daily trade count (optional limit)
            daily_trade_limit = self.user_risk_config.max_concurrent_trades * 2  # Example: 2x max positions
            if self.daily_trades_count >= daily_trade_limit:
                return False, f"Daily trade limit reached ({daily_trade_limit})"
            
            # Portfolio exposure check
            position_value = signal.quantity * signal.price
            if self.current_exposure + position_value > (self.user_risk_config.capital * 0.8):  # 80% max exposure
                return False, "Portfolio exposure limit exceeded"
            
            # Symbol concentration check
            symbol_exposure = self.active_positions.get(signal.symbol, {}).get('value', 0)
            if symbol_exposure + position_value > self.max_position_value:
                return False, f"Symbol position size limit exceeded for {signal.symbol}"
            
            # Risk-reward ratio check
            if signal.stop_loss and signal.take_profit:
                risk = abs(signal.price - signal.stop_loss)
                reward = abs(signal.take_profit - signal.price)
                actual_rr_ratio = reward / risk if risk > 0 else 0
                
                if actual_rr_ratio < self.user_risk_config.risk_reward_ratio:
                    return False, f"Risk-reward ratio too low: {actual_rr_ratio:.2f} < {self.user_risk_config.risk_reward_ratio}"
            
            # Volatility check (enhanced)
            volatility = await self.volatility_analyzer.get_symbol_volatility(signal.symbol)
            max_volatility = 0.08  # 8% max daily volatility
            if volatility and volatility > max_volatility:
                return False, f"Symbol volatility too high: {volatility:.2%}"
            
            return True, "Signal validated successfully"
            
        except Exception as e:
            self.logger.error(f"Signal validation error: {e}")
            return False, f"Validation error: {str(e)}"
    
    def _is_within_user_trading_hours(self) -> bool:
        """Check if current time is within user-defined trading hours"""
        try:
            if not self.user_risk_config:
                return self._is_market_open()
            
            now = datetime.now()
            
            # Parse user trading hours
            start_time = datetime.strptime(self.user_risk_config.trading_start_time, '%H:%M').time()
            end_time = datetime.strptime(self.user_risk_config.trading_end_time, '%H:%M').time()
            
            current_time = now.time()
            
            # Check if within user trading hours
            if start_time <= end_time:  # Same day
                return start_time <= current_time <= end_time
            else:  # Crosses midnight
                return current_time >= start_time or current_time <= end_time
                
        except Exception as e:
            self.logger.error(f"Trading hours check error: {e}")
            return self._is_market_open()
    
    async def _apply_user_stop_loss_take_profit(self, signal: TradingSignal) -> TradingSignal:
        """Apply user-defined stop loss and take profit if not set"""
        try:
            if not self.user_risk_config:
                return signal
            
            # Apply default stop loss if not set
            if not signal.stop_loss:
                if signal.action == 'BUY':
                    signal.stop_loss = signal.price * (1 - self.user_risk_config.stop_loss_percent / 100)
                elif signal.action == 'SELL':  # Short position
                    signal.stop_loss = signal.price * (1 + self.user_risk_config.stop_loss_percent / 100)
            
            # Apply default take profit if not set
            if not signal.take_profit:
                if signal.action == 'BUY':
                    signal.take_profit = signal.price * (1 + self.user_risk_config.take_profit_percent / 100)
                elif signal.action == 'SELL':  # Short position
                    signal.take_profit = signal.price * (1 - self.user_risk_config.take_profit_percent / 100)
            
            # Adjust position size based on risk
            signal.quantity = await self._calculate_position_size_for_signal(signal)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error applying user SL/TP: {e}")
            return signal
    
    # Override the existing methods to include user risk integration
    
    async def _validate_signal(self, signal: TradingSignal) -> bool:
        """Enhanced signal validation with user risk parameters"""
        is_valid, reason = await self._validate_signal_with_user_risk(signal)
        if not is_valid:
            self.logger.debug(f"Signal validation failed for {signal.symbol}: {reason}")
        return is_valid
    
    async def _execute_live_signal(self, signal: TradingSignal):
        """Execute signal in live trading mode with user risk integration"""
        try:
            if not self._is_within_user_trading_hours():
                self.logger.warning(f"Outside trading hours - skipping {signal.symbol}")
                return
            
            if not self.new_positions_allowed and signal.action == 'BUY':
                self.logger.warning(f"New positions disabled - skipping BUY {signal.symbol}")
                return
            
            # Apply user-defined stop loss and take profit
            signal = await self._apply_user_stop_loss_take_profit(signal)
            
            if signal.action == 'BUY':
                order_result = await self.goodwill_api.place_buy_order(
                    symbol=signal.symbol,
                    quantity=signal.quantity,
                    price=signal.price,
                    order_type='LIMIT',
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit
                )
            elif signal.action == 'SELL':
                order_result = await self.goodwill_api.place_sell_order(
                    symbol=signal.symbol,
                    quantity=signal.quantity,
                    price=signal.price,
                    order_type='LIMIT'
                )
            
            if order_result and order_result.get('success'):
                await self._log_trade(signal, order_result, 'live')
                self.daily_trades_count += 1
                
                self.logger.info(f"✅ Live: {signal.action} {signal.quantity} {signal.symbol} @ ₹{signal.price} "
                               f"SL: ₹{signal.stop_loss:.2f} TP: ₹{signal.take_profit:.2f}")
                
                # Send notification for successful trade
                await self.monitor.send_alert(
                    "TRADE_EXECUTED", 
                    f"Live trade: {signal.action} {signal.quantity} {signal.symbol} @ ₹{signal.price}"
                )
            else:
                self.logger.error(f"❌ Live order failed: {order_result}")
                
        except Exception as e:
            self.logger.error(f"Live signal execution error: {e}")
    
    async def _execute_paper_signal(self, signal: TradingSignal):
        """Execute signal in paper trading mode with user risk integration"""
        try:
            # Apply user-defined stop loss and take profit
            signal = await self._apply_user_stop_loss_take_profit(signal)
            
            result = await self.paper_engine.execute_trade(
                symbol=signal.symbol,
                action=signal.action,
                quantity=signal.quantity,
                price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                strategy=signal.strategy
            )
            
            if result.get('success'):
                await self._log_trade(signal, result, 'paper')
                self.daily_trades_count += 1
                
                self.logger.info(f"📄 Paper: {signal.action} {signal.quantity} {signal.symbol} @ ₹{signal.price} "
                               f"SL: ₹{signal.stop_loss:.2f} TP: ₹{signal.take_profit:.2f}")
            
        except Exception as e:
            self.logger.error(f"Paper signal execution error: {e}")
    
    async def _validate_signal_basic(self, signal: TradingSignal) -> Tuple[bool, str]:
        """Basic signal validation (fallback when no user config)"""
        try:
            if not signal.symbol or not signal.action or signal.confidence < 0.6:
                return False, "Basic validation failed"
            
            if self.trading_mode == 'live':
                is_tradeable = await self.goodwill_api.is_symbol_tradeable(signal.symbol)
                if not is_tradeable:
                    return False, "Symbol not tradeable"
            
            max_position_size = self.config.get('max_position_size', 50000)
            if signal.quantity * signal.price > max_position_size:
                return False, "Position size too large"
            
            return True, "Basic validation passed"
            
        except Exception as e:
            self.logger.error(f"Basic signal validation error: {e}")
            return False, f"Validation error: {str(e)}"
    
    async def _emergency_close_all_positions(self, reason: str):
        """Emergency close all positions with user risk integration"""
        self.logger.critical(f"🚨 EMERGENCY CLOSE ALL POSITIONS: {reason}")
        
        try:
            positions_to_close = list(self.active_positions.items())
            
            for symbol, position in positions_to_close:
                quantity = position.get('quantity', 0)
                if quantity != 0:
                    current_price = await self.data_manager.get_current_price(symbol)
                    
                    if current_price:
                        if self.trading_mode == 'live':
                            if quantity > 0:  # Long position
                                await self.goodwill_api.place_sell_order(
                                    symbol=symbol,
                                    quantity=quantity,
                                    order_type='MARKET'
                                )
                            else:  # Short position
                                await self.goodwill_api.place_buy_order(
                                    symbol=symbol,
                                    quantity=abs(quantity),
                                    order_type='MARKET'
                                )
                        else:
                            await self.paper_engine.close_position(symbol)
                        
                        self.logger.info(f"Emergency closed position: {symbol} ({quantity} shares)")
            
            await self.monitor.send_alert(
                "EMERGENCY_CLOSE", 
                f"All positions closed due to: {reason}"
            )
            
            # Reset trading state
            self.new_positions_allowed = False
            
        except Exception as e:
            self.logger.error(f"Emergency close error: {e}")
    
    # Enhanced auto-scheduling with user config integration
    async def _setup_auto_schedule(self):
        """Setup automatic start/stop scheduling with user preferences"""
        try:
            self.logger.info("Setting up automatic trading schedule with user preferences...")
            
            if not self.user_risk_config:
                # Use default schedule
                start_hour, start_minute = 8, 15
                end_hour, end_minute = 16, 30
            else:
                # Use user-defined trading hours
                start_time = datetime.strptime(self.user_risk_config.trading_start_time, '%H:%M')
                end_time = datetime.strptime(self.user_risk_config.trading_end_time, '%H:%M')
                
                # Start 1 hour before user trading time
                pre_start = start_time - timedelta(hours=1)
                post_end = end_time + timedelta(hours=1)
                
                start_hour, start_minute = pre_start.hour, pre_start.minute
                end_hour, end_minute = post_end.hour, post_end.minute
            
            # Pre-market start - Monday to Friday
            self.scheduler.add_job(
                self._scheduled_pre_market_start,
                CronTrigger(hour=start_hour, minute=start_minute, day_of_week='0-4'),
                id='pre_market_start',
                replace_existing=True
            )
            
            # Market open preparation (5 minutes before user trading time)
            user_start = datetime.strptime(self.user_risk_config.trading_start_time if self.user_risk_config else "09:15", '%H:%M')
            prep_time = user_start - timedelta(minutes=5)
            
            self.scheduler.add_job(
                self._scheduled_market_open_prep,
                CronTrigger(hour=prep_time.hour, minute=prep_time.minute, day_of_week='0-4'),
                id='market_open_prep',
                replace_existing=True
            )
            
            # Market close preparation (5 minutes before user end time)
            user_end = datetime.strptime(self.user_risk_config.trading_end_time if self.user_risk_config else "15:30", '%H:%M')
            close_prep = user_end - timedelta(minutes=5)
            
            self.scheduler.add_job(
                self._scheduled_market_close_prep,
                CronTrigger(hour=close_prep.hour, minute=close_prep.minute, day_of_week='0-4'),
                id='market_close_prep',
                replace_existing=True
            )
            
            # Post-market stop
            self.scheduler.add_job(
                self._scheduled_post_market_stop,
                CronTrigger(hour=end_hour, minute=end_minute, day_of_week='0-4'),
                id='post_market_stop',
                replace_existing=True
            )
            
            # Weekend maintenance
            self.scheduler.add_job(
                self._scheduled_weekend_maintenance,
                CronTrigger(hour=2, minute=0, day_of_week='5'),
                id='weekend_maintenance',
                replace_existing=True
            )
            
            self.logger.info("✅ Auto-schedule configured with user preferences:")
            self.logger.info(f"  📅 Pre-market: {start_hour:02d}:{start_minute:02d} IST")
            self.logger.info(f"  📅 Trading: {prep_time.hour:02d}:{prep_time.minute:02d} - {close_prep.hour:02d}:{close_prep.minute:02d} IST")
            self.logger.info(f"  📅 Post-market: {end_hour:02d}:{end_minute:02d} IST")
            
        except Exception as e:
            self.logger.error(f"Error setting up auto-schedule: {e}")
    
    async def _scheduled_market_close_prep(self):
        """Market close preparation with user auto-square-off settings"""
        try:
            self.logger.info("🔔 MARKET CLOSE PREP - Preparing for close...")
            
            # Stop new positions
            self.new_positions_allowed = False
            
            # Check user auto-square-off setting
            if self.user_risk_config and self.user_risk_config.auto_square_off:
                self.logger.info("User auto-square-off enabled - closing all positions")
                await self._close_all_positions()
            else:
                # Just close risky positions
                await self._close_risky_positions()
            
            await self.monitor.send_alert("MARKET_CLOSING", "Preparing for market close")
            
        except Exception as e:
            self.logger.error(f"Market close prep error: {e}")
    
    async def _scheduled_post_market_stop(self):
        """Post-market shutdown with user reporting"""
        try:
            self.logger.info("🌙 POST-MARKET SHUTDOWN - Ending trading session...")
            
            # Generate user-specific daily report
            await self._generate_user_daily_report()
            
            # Final position check with user settings
            if self.user_risk_config and self.user_risk_config.auto_square_off:
                await self._close_all_positions()
            
            # Reset daily counters
            self.daily_pnl = 0.0
            self.daily_trades_count = 0
            self.trading_halted_by_risk = False
            
            # Stop trading operations
            self.is_running = False
            self.is_scheduled_session = False
            
            # Backup user data
            await self._backup_user_daily_data()
            
            self.logger.info("✅ Trading session ended successfully")
            await self.monitor.send_alert("SESSION_END", f"Trading session completed for user {self.user_id}")
            
        except Exception as e:
            self.logger.error(f"Post-market shutdown error: {e}")
    
    async def _generate_user_daily_report(self):
        """Generate comprehensive daily report for user"""
        try:
            self.logger.info("Generating user daily trading report...")
            
            # Calculate daily metrics
            daily_pnl = await self._calculate_daily_pnl()
            today_trades = await self.get_today_trades()
            portfolio_value = await self._calculate_portfolio_value()
            
            # Calculate performance metrics
            winning_trades = [t for t in today_trades if t.get('realized_pnl', 0) > 0]
            losing_trades = [t for t in today_trades if t.get('realized_pnl', 0) < 0]
            
            win_rate = len(winning_trades) / len(today_trades) if today_trades else 0
            avg_win = sum(t.get('realized_pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t.get('realized_pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
            
            # Create report
            report = {
                'user_id': self.user_id,
                'date': datetime.now().date().isoformat(),
                'daily_pnl': daily_pnl,
                'portfolio_value': portfolio_value,
                'total_trades': len(today_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'max_concurrent_positions': max([len(self.active_positions)] + [0]),
                'risk_limit_breaches': 1 if self.trading_halted_by_risk else 0,
                'auto_square_off_used': self.user_risk_config.auto_square_off if self.user_risk_config else False,
                'trading_mode': self.trading_mode
            }
            
            # Save to database
            async with self.db_manager.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO daily_trading_reports (
                        user_id, report_date, daily_pnl, portfolio_value, total_trades,
                        winning_trades, losing_trades, win_rate, avg_win, avg_loss,
                        max_concurrent_positions, risk_limit_breaches, auto_square_off_used,
                        trading_mode, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    ON CONFLICT (user_id, report_date) DO UPDATE SET
                        daily_pnl = EXCLUDED.daily_pnl,
                        portfolio_value = EXCLUDED.portfolio_value,
                        total_trades = EXCLUDED.total_trades,
                        winning_trades = EXCLUDED.winning_trades,
                        losing_trades = EXCLUDED.losing_trades,
                        win_rate = EXCLUDED.win_rate,
                        avg_win = EXCLUDED.avg_win,
                        avg_loss = EXCLUDED.avg_loss,
                        max_concurrent_positions = EXCLUDED.max_concurrent_positions,
                        risk_limit_breaches = EXCLUDED.risk_limit_breaches,
                        auto_square_off_used = EXCLUDED.auto_square_off_used,
                        trading_mode = EXCLUDED.trading_mode,
                        updated_at = NOW()
                """, 
                report['user_id'], report['date'], report['daily_pnl'], report['portfolio_value'],
                report['total_trades'], report['winning_trades'], report['losing_trades'],
                report['win_rate'], report['avg_win'], report['avg_loss'],
                report['max_concurrent_positions'], report['risk_limit_breaches'],
                report['auto_square_off_used'], report['trading_mode'], datetime.now())
            
            # Log summary
            self.logger.info(f"📊 Daily Report Summary for {self.user_id}:")
            self.logger.info(f"  💰 P&L: ₹{daily_pnl:,.2f}")
            self.logger.info(f"  📈 Trades: {len(today_trades)} (Win: {len(winning_trades)}, Loss: {len(losing_trades)})")
            self.logger.info(f"  🎯 Win Rate: {win_rate:.1%}")
            self.logger.info(f"  💼 Portfolio Value: ₹{portfolio_value:,.2f}")
            
        except Exception as e:
            self.logger.error(f"Error generating daily report: {e}")
    
    async def _backup_user_daily_data(self):
        """Backup user-specific daily trading data"""
        try:
            self.logger.info(f"Backing up daily data for user {self.user_id}...")
            
            # Export today's trades to JSON
            today_trades = await self.get_today_trades()
            positions = await self.get_positions()
            performance = await self.get_performance_metrics()
            
            backup_data = {
                'user_id': self.user_id,
                'date': datetime.now().date().isoformat(),
                'trades': today_trades,
                'final_positions': positions,
                'performance_metrics': performance,
                'user_config': {
                    'capital': self.user_risk_config.capital if self.user_risk_config else 0,
                    'risk_per_trade_percent': self.user_risk_config.risk_per_trade_percent if self.user_risk_config else 0,
                    'daily_loss_limit_percent': self.user_risk_config.daily_loss_limit_percent if self.user_risk_config else 0,
                    'max_concurrent_trades': self.user_risk_config.max_concurrent_trades if self.user_risk_config else 0,
                    'trading_mode': self.trading_mode
                }
            }
            
            # Save backup file
            backup_filename = f"backup_{self.user_id}_{datetime.now().strftime('%Y%m%d')}.json"
            with open(f"backups/{backup_filename}", 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            self.logger.info(f"✅ Daily data backup completed: {backup_filename}")
            
        except Exception as e:
            self.logger.error(f"Backup error: {e}")
    
    # Enhanced public API methods with user risk integration
    
    async def get_user_status(self) -> dict:
        """Get comprehensive user-specific bot status"""
        base_status = await self.get_status()
        
        user_specific = {
            'user_id': self.user_id,
            'user_risk_config': {
                'capital': self.user_risk_config.capital if self.user_risk_config else 0,
                'risk_per_trade_percent': self.user_risk_config.risk_per_trade_percent if self.user_risk_config else 0,
                'daily_loss_limit_percent': self.user_risk_config.daily_loss_limit_percent if self.user_risk_config else 0,
                'max_concurrent_trades': self.user_risk_config.max_concurrent_trades if self.user_risk_config else 0,
                'paper_trading_mode': self.user_risk_config.paper_trading_mode if self.user_risk_config else True,
                'auto_square_off': self.user_risk_config.auto_square_off if self.user_risk_config else False,
                'trading_start_time': self.user_risk_config.trading_start_time if self.user_risk_config else "09:15",
                'trading_end_time': self.user_risk_config.trading_end_time if self.user_risk_config else "15:30",
                'last_updated': self.user_risk_config.last_updated.isoformat() if self.user_risk_config and self.user_risk_config.last_updated else None
            },
            'risk_status': {
                'daily_pnl': self.daily_pnl,
                'daily_trades_count': self.daily_trades_count,
                'current_exposure': self.current_exposure,
                'trading_halted_by_risk': self.trading_halted_by_risk,
                'daily_loss_limit': self.daily_loss_limit if hasattr(self, 'daily_loss_limit') else 0,
                'max_position_value': self.max_position_value if hasattr(self, 'max_position_value') else 0,
                'risk_per_trade_amount': self.risk_per_trade_amount if hasattr(self, 'risk_per_trade_amount') else 0
            },
            'within_trading_hours': self._is_within_user_trading_hours()
        }
        
        return {**base_status, **user_specific}
    
    async def update_user_risk_config(self, new_config: dict) -> dict:
        """Update user risk configuration"""
        try:
            if not self.user_id:
                return {'success': False, 'message': 'No user ID specified'}
            
            # Update database
            async with self.db_manager.get_connection() as conn:
                await conn.execute("""
                    UPDATE user_trading_config SET
                        capital = $2,
                        risk_per_trade_percent = $3,
                        daily_loss_limit_percent = $4,
                        max_concurrent_trades = $5,
                        risk_reward_ratio = $6,
                        max_position_size_percent = $7,
                        stop_loss_percent = $8,
                        take_profit_percent = $9,
                        trading_start_time = $10,
                        trading_end_time = $11,
                        auto_square_off = $12,
                        paper_trading_mode = $13,
                        last_updated = $14
                    WHERE user_id = $1
                """, 
                self.user_id,
                new_config.get('capital', self.user_risk_config.capital if self.user_risk_config else 100000),
                new_config.get('risk_per_trade_percent', self.user_risk_config.risk_per_trade_percent if self.user_risk_config else 2.0),
                new_config.get('daily_loss_limit_percent', self.user_risk_config.daily_loss_limit_percent if self.user_risk_config else 5.0),
                new_config.get('max_concurrent_trades', self.user_risk_config.max_concurrent_trades if self.user_risk_config else 5),
                new_config.get('risk_reward_ratio', self.user_risk_config.risk_reward_ratio if self.user_risk_config else 2.0),
                new_config.get('max_position_size_percent', self.user_risk_config.max_position_size_percent if self.user_risk_config else 20.0),
                new_config.get('stop_loss_percent', self.user_risk_config.stop_loss_percent if self.user_risk_config else 3.0),
                new_config.get('take_profit_percent', self.user_risk_config.take_profit_percent if self.user_risk_config else 6.0),
                datetime.strptime(new_config.get('trading_start_time', '09:15'), '%H:%M').time(),
                datetime.strptime(new_config.get('trading_end_time', '15:30'), '%H:%M').time(),
                new_config.get('auto_square_off', self.user_risk_config.auto_square_off if self.user_risk_config else True),
                new_config.get('paper_trading_mode', self.user_risk_config.paper_trading_mode if self.user_risk_config else True),
                datetime.now())
            
            # Reload configuration
            await self.load_user_risk_config()
            await self.apply_user_risk_settings()
            
            # Recreate schedule if trading hours changed
            if ('trading_start_time' in new_config or 'trading_end_time' in new_config) and self.auto_schedule_enabled:
                await self._setup_auto_schedule()
            
            self.logger.info(f"User risk configuration updated for {self.user_id}")
            
            return {
                'success': True, 
                'message': 'Risk configuration updated successfully',
                'config': new_config
            }
            
        except Exception as e:
            self.logger.error(f"Error updating user risk config: {e}")
            return {'success': False, 'message': str(e)}
    
    async def get_user_daily_report(self, date: str = None) -> dict:
        """Get daily trading report for user"""
        try:
            report_date = date or datetime.now().date().isoformat()
            
            async with self.db_manager.get_connection() as conn:
                report = await conn.fetchrow("""
                    SELECT * FROM daily_trading_reports 
                    WHERE user_id = $1 AND report_date = $2
                """, self.user_id, report_date)
                
                if report:
                    return dict(report)
                else:
                    return {'message': f'No report found for {report_date}'}
                    
        except Exception as e:
            self.logger.error(f"Error fetching daily report: {e}")
            return {'error': str(e)}
    
    async def reset_daily_risk_limits(self) -> dict:
        """Reset daily risk limits (admin function)"""
        try:
            self.daily_pnl = 0.0
            self.daily_trades_count = 0
            self.trading_halted_by_risk = False
            self.new_positions_allowed = True
            
            self.logger.info(f"Daily risk limits reset for user {self.user_id}")
            
            return {
                'success': True,
                'message': 'Daily risk limits reset successfully'
            }
            
        except Exception as e:
            self.logger.error(f"Error resetting daily limits: {e}")
            return {'success': False, 'message': str(e)}
    
    # Keep all existing methods from the original bot_core.py
    # (The rest of the methods remain the same as in the original file)
    
    def stop(self):
        """Stop the trading bot system"""
        return super().stop()
    
    # ... (all other existing methods remain unchanged)

# Entry point with user ID support
if __name__ == "__main__":
    import sys
    
    async def main():
        """Main execution function with user support"""
        user_id = sys.argv[1] if len(sys.argv) > 1 else 'default_user'
        bot = TradingBotCore(user_id=user_id)
        
        try:
            bot.logger.info(f"🚀 Starting Advanced Trading Bot for User: {user_id}")
            await bot.start()
            
        except KeyboardInterrupt:
            bot.logger.info("👋 Shutdown requested by user")
            await bot.stop()
            
        except Exception as e:
            bot.logger.error(f"💥 Bot crashed: {e}")
            await bot.stop()
            raise

    # Run the bot
    asyncio.run(main())
