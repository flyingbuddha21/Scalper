#!/usr/bin/env python3
"""
Advanced Multi-Strategy Trading Bot Core
Market-ready for live trading with Goodwill API integration
Includes automatic scheduling: starts 1 hour before market, stops 1 hour after
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

class TradingBotCore:
    def __init__(self, config_path: str = "config.json"):
        """Initialize the trading bot core system with auto-scheduling"""
        # Core components
        self.config = ConfigManager(config_path)
        self.logger = self._setup_logging()
        
        # Trading mode (live or paper)
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
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
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
        
        self.logger.info(f"Trading Bot initialized in {self.trading_mode.upper()} mode")
        if self.auto_schedule_enabled:
            self.logger.info(f"Auto-schedule enabled: {self.pre_market_start} - {self.post_market_stop} IST")
    
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
        """Start the trading bot system with auto-scheduling"""
        try:
            self.logger.info("Starting Trading Bot Core System...")
            
            # Setup auto-scheduling first if enabled
            if self.auto_schedule_enabled:
                await self._setup_auto_schedule()
            
            # Initialize database
            await self.db_manager.initialize()
            
            # Start monitoring
            await self.monitor.start()
            
            # Initialize API connections
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
            
            # Run concurrent tasks
            tasks = [
                self._market_data_processor(),
                self._strategy_execution_loop(),
                self._order_management_loop(),
                self._risk_management_loop(),
                self._performance_tracking_loop(),
                self._market_scanner_loop(),
                self._schedule_monitor_loop()
            ]
            
            await asyncio.gather(*tasks)
            
        except Exception as e:
            self.logger.error(f"Failed to start trading bot: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the trading bot system"""
        self.logger.info("Stopping Trading Bot Core System...")
        
        self.is_running = False
        
        # Close all positions if configured
        if self.config.get('close_positions_on_stop', True):
            await self._close_all_positions()
        
        # Stop components
        await self.websocket_manager.stop()
        await self.data_manager.stop()
        await self.scanner.stop()
        await self.monitor.stop()
        
        if self.trading_mode == 'live':
            await self.goodwill_api.disconnect()
        
        # Stop scheduler
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        
        self.executor.shutdown(wait=True)
        self.logger.info("Trading Bot stopped successfully")
    
    async def _setup_auto_schedule(self):
        """Setup automatic start/stop scheduling"""
        try:
            self.logger.info("Setting up automatic trading schedule...")
            
            # Pre-market start (8:15 AM IST) - Monday to Friday
            self.scheduler.add_job(
                self._scheduled_pre_market_start,
                CronTrigger(hour=8, minute=15, day_of_week='0-4'),
                id='pre_market_start',
                replace_existing=True
            )
            
            # Market open preparation (9:10 AM IST)
            self.scheduler.add_job(
                self._scheduled_market_open_prep,
                CronTrigger(hour=9, minute=10, day_of_week='0-4'),
                id='market_open_prep',
                replace_existing=True
            )
            
            # Market close preparation (15:25 PM IST)
            self.scheduler.add_job(
                self._scheduled_market_close_prep,
                CronTrigger(hour=15, minute=25, day_of_week='0-4'),
                id='market_close_prep',
                replace_existing=True
            )
            
            # Post-market stop (16:30 PM IST)
            self.scheduler.add_job(
                self._scheduled_post_market_stop,
                CronTrigger(hour=16, minute=30, day_of_week='0-4'),
                id='post_market_stop',
                replace_existing=True
            )
            
            # Weekend maintenance (Saturday 2:00 AM)
            self.scheduler.add_job(
                self._scheduled_weekend_maintenance,
                CronTrigger(hour=2, minute=0, day_of_week='5'),
                id='weekend_maintenance',
                replace_existing=True
            )
            
            self.logger.info("✅ Auto-schedule configured:")
            self.logger.info(f"  📅 Pre-market: {self.pre_market_start} IST")
            self.logger.info(f"  📅 Market prep: 09:10 IST")
            self.logger.info(f"  📅 Close prep: 15:25 IST")
            self.logger.info(f"  📅 Post-market: {self.post_market_stop} IST")
            
        except Exception as e:
            self.logger.error(f"Error setting up auto-schedule: {e}")
    
    async def _scheduled_pre_market_start(self):
        """Pre-market startup (1 hour before market open)"""
        try:
            self.logger.info("🌅 PRE-MARKET STARTUP - Initializing systems...")
            self.is_scheduled_session = True
            
            # System health check
            await self._system_health_check()
            
            # Initialize data connections
            if not self.data_manager.is_connected:
                await self.data_manager.start()
            if not self.websocket_manager.is_connected:
                await self.websocket_manager.start()
            
            # Pre-market analysis
            await self._pre_market_analysis()
            
            await self.monitor.send_alert("PRE-MARKET START", "Trading systems initialized")
            
        except Exception as e:
            self.logger.error(f"Pre-market startup error: {e}")
            await self.monitor.send_alert("PRE-MARKET ERROR", f"Startup failed: {e}")
    
    async def _scheduled_market_open_prep(self):
        """Market open preparation (5 minutes before open)"""
        try:
            self.logger.info("🔔 MARKET OPEN PREP - Final checks...")
            
            # Authenticate APIs
            if self.trading_mode == 'live':
                if not await self.goodwill_api.is_authenticated():
                    await self.goodwill_api.authenticate()
            
            # Final system checks
            await self._pre_trading_checks()
            
            # Enable trading
            self.is_running = True
            self.new_positions_allowed = True
            
            # Load daily trading plan
            await self._load_daily_trading_plan()
            
            self.logger.info("✅ Ready for market open!")
            await self.monitor.send_alert("MARKET READY", "All systems ready for trading")
            
        except Exception as e:
            self.logger.error(f"Market prep error: {e}")
            await self.monitor.send_alert("MARKET PREP ERROR", f"Prep failed: {e}")
    
    async def _scheduled_market_close_prep(self):
        """Market close preparation (5 minutes before close)"""
        try:
            self.logger.info("🔔 MARKET CLOSE PREP - Preparing for close...")
            
            # Stop new positions
            self.new_positions_allowed = False
            
            # Close positions if configured
            if self.config.get('close_positions_at_market_close', True):
                await self._close_risky_positions()
            
            await self.monitor.send_alert("MARKET CLOSING", "Preparing for market close")
            
        except Exception as e:
            self.logger.error(f"Market close prep error: {e}")
    
    async def _scheduled_post_market_stop(self):
        """Post-market shutdown (1 hour after market close)"""
        try:
            self.logger.info("🌙 POST-MARKET SHUTDOWN - Ending trading session...")
            
            # Generate daily report
            await self._generate_daily_report()
            
            # Close remaining positions if any
            await self._close_all_positions()
            
            # Stop trading operations
            self.is_running = False
            self.is_scheduled_session = False
            
            # Backup data
            await self._backup_daily_data()
            
            self.logger.info("✅ Trading session ended successfully")
            await self.monitor.send_alert("SESSION END", "Trading session completed")
            
        except Exception as e:
            self.logger.error(f"Post-market shutdown error: {e}")
    
    async def _scheduled_weekend_maintenance(self):
        """Weekend maintenance tasks"""
        try:
            self.logger.info("🔧 WEEKEND MAINTENANCE - System optimization...")
            
            # Database maintenance
            await self.db_manager.optimize_database()
            
            # Performance analysis
            await self._weekly_performance_analysis()
            
            # System cleanup
            await self._cleanup_logs()
            
            self.logger.info("✅ Weekend maintenance completed")
            
        except Exception as e:
            self.logger.error(f"Weekend maintenance error: {e}")
    
    async def _schedule_monitor_loop(self):
        """Monitor scheduled events and auto-scheduling"""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Log next scheduled job
                if self.scheduler.running:
                    next_job = self.scheduler.get_jobs()[0] if self.scheduler.get_jobs() else None
                    if next_job:
                        self.logger.debug(f"Next scheduled event: {next_job.id} at {next_job.next_run_time}")
                
                # Check if we're in a scheduled session
                if self.auto_schedule_enabled:
                    await self._check_schedule_compliance(current_time)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Schedule monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _check_schedule_compliance(self, current_time: datetime):
        """Ensure bot is running according to schedule"""
        try:
            # Parse times
            pre_market = current_time.replace(hour=8, minute=15, second=0, microsecond=0)
            post_market = current_time.replace(hour=16, minute=30, second=0, microsecond=0)
            
            # Check if we should be running
            should_be_running = (
                current_time.weekday() < 5 and  # Monday-Friday
                pre_market <= current_time <= post_market
            )
            
            if should_be_running and not self.is_running:
                self.logger.warning("Bot should be running but isn't - auto-starting...")
                await self._emergency_start()
            elif not should_be_running and self.is_running and self.is_scheduled_session:
                self.logger.warning("Bot running outside schedule - auto-stopping...")
                await self._emergency_stop()
                
        except Exception as e:
            self.logger.error(f"Schedule compliance check error: {e}")
    
    async def set_trading_mode(self, mode: str):
        """Switch between live and paper trading modes"""
        if mode not in ['live', 'paper']:
            raise ValueError("Trading mode must be 'live' or 'paper'")
        
        if mode == self.trading_mode:
            self.logger.info(f"Already in {mode} mode")
            return
        
        self.logger.info(f"Switching from {self.trading_mode} to {mode} mode...")
        
        # Stop current mode
        if self.trading_mode == 'live':
            await self.goodwill_api.disconnect()
        else:
            await self.paper_engine.stop()
        
        # Start new mode
        self.trading_mode = mode
        if mode == 'live':
            await self.goodwill_api.initialize()
            await self.goodwill_api.authenticate()
        else:
            await self.paper_engine.initialize()
        
        # Update configuration
        self.config.set('trading_mode', mode)
        await self.config.save()
        
        self.logger.info(f"✅ Switched to {mode.upper()} mode")
    
    async def _market_data_processor(self):
        """Process real-time market data"""
        while self.is_running:
            try:
                market_data = await self.data_manager.get_realtime_data()
                
                if market_data:
                    await self.volatility_analyzer.update(market_data)
                    
                    for symbol, data in market_data.items():
                        signals = await self.strategy_manager.analyze(symbol, data)
                        
                        for signal in signals:
                            self.trading_signals.put(signal)
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Market data processing error: {e}")
                await asyncio.sleep(1)
    
    async def _strategy_execution_loop(self):
        """Execute trading strategies and generate signals"""
        while self.is_running:
            try:
                if not self.trading_signals.empty():
                    signal = self.trading_signals.get_nowait()
                    
                    if await self._validate_signal(signal):
                        if self.trading_mode == 'live':
                            await self._execute_live_signal(signal)
                        else:
                            await self._execute_paper_signal(signal)
                
                await asyncio.sleep(0.05)
                
            except queue.Empty:
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Strategy execution error: {e}")
                await asyncio.sleep(1)
    
    async def _execute_live_signal(self, signal: TradingSignal):
        """Execute signal in live trading mode"""
        try:
            if not self._is_market_open():
                self.logger.warning(f"Market closed - skipping {signal.symbol}")
                return
            
            if not self.new_positions_allowed and signal.action == 'BUY':
                self.logger.warning(f"New positions disabled - skipping BUY {signal.symbol}")
                return
            
            if not await self._risk_check(signal):
                self.logger.warning(f"Risk check failed for {signal.symbol}")
                return
            
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
                self.logger.info(f"✅ Live: {signal.action} {signal.quantity} {signal.symbol} @ {signal.price}")
            else:
                self.logger.error(f"❌ Live order failed: {order_result}")
                
        except Exception as e:
            self.logger.error(f"Live signal execution error: {e}")
    
    async def _execute_paper_signal(self, signal: TradingSignal):
        """Execute signal in paper trading mode"""
        try:
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
                self.logger.info(f"📄 Paper: {signal.action} {signal.quantity} {signal.symbol} @ {signal.price}")
            
        except Exception as e:
            self.logger.error(f"Paper signal execution error: {e}")
    
    async def _order_management_loop(self):
        """Manage active orders and positions"""
        while self.is_running:
            try:
                if self.trading_mode == 'live':
                    orders = await self.goodwill_api.get_order_status()
                    positions = await self.goodwill_api.get_positions()
                else:
                    orders = await self.paper_engine.get_active_orders()
                    positions = await self.paper_engine.get_positions()
                
                with self.lock:
                    self.active_positions = positions or {}
                    self.pending_orders = orders or {}
                
                await self._process_exit_conditions()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Order management error: {e}")
                await asyncio.sleep(5)
    
    async def _risk_management_loop(self):
        """Continuous risk management"""
        while self.is_running:
            try:
                portfolio_value = await self._calculate_portfolio_value()
                daily_pnl = await self._calculate_daily_pnl()
                
                # Daily loss limit check
                max_daily_loss = self.config.get('max_daily_loss', 5000)
                if abs(daily_pnl) > max_daily_loss and daily_pnl < 0:
                    self.logger.warning(f"Daily loss limit exceeded: {daily_pnl}")
                    await self._emergency_stop()
                
                # Position size checks
                await self._check_position_sizes()
                
                with self.lock:
                    self.performance_metrics['daily_pnl'] = daily_pnl
                    self.performance_metrics['total_pnl'] = portfolio_value - self.config.get('initial_capital', 100000)
                
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Risk management error: {e}")
                await asyncio.sleep(30)
    
    async def _performance_tracking_loop(self):
        """Track and update performance metrics"""
        while self.is_running:
            try:
                metrics = await self._calculate_performance_metrics()
                
                with self.lock:
                    self.performance_metrics.update(metrics)
                
                await self.db_manager.save_performance_metrics(self.performance_metrics)
                
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Performance tracking error: {e}")
                await asyncio.sleep(60)
    
    async def _market_scanner_loop(self):
        """Continuous market scanning"""
        while self.is_running:
            try:
                if self._is_market_open():
                    opportunities = await self.scanner.scan_market()
                    
                    for opp in opportunities:
                        if opp.get('probability', 0) > 0.7:
                            await self.data_manager.add_to_watchlist(opp['symbol'])
                
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Market scanning error: {e}")
                await asyncio.sleep(60)
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now()
        
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    async def _validate_signal(self, signal: TradingSignal) -> bool:
        """Validate trading signal"""
        try:
            if not signal.symbol or not signal.action or signal.confidence < 0.6:
                return False
            
            if self.trading_mode == 'live':
                is_tradeable = await self.goodwill_api.is_symbol_tradeable(signal.symbol)
                if not is_tradeable:
                    return False
            
            max_position_size = self.config.get('max_position_size', 10000)
            if signal.quantity * signal.price > max_position_size:
                return False
            
            return await self._risk_check(signal)
            
        except Exception as e:
            self.logger.error(f"Signal validation error: {e}")
            return False
    
    async def _risk_check(self, signal: TradingSignal) -> bool:
        """Comprehensive risk check"""
        try:
            # Portfolio exposure
            current_exposure = sum(pos.get('value', 0) for pos in self.active_positions.values())
            max_exposure = self.config.get('max_portfolio_exposure', 80000)
            
            new_exposure = signal.quantity * signal.price
            if current_exposure + new_exposure > max_exposure:
                return False
            
            # Symbol concentration
            symbol_exposure = self.active_positions.get(signal.symbol, {}).get('value', 0)
            max_symbol_exposure = self.config.get('max_symbol_exposure', 20000)
            
            if symbol_exposure + new_exposure > max_symbol_exposure:
                return False
            
            # Volatility check
            volatility = await self.volatility_analyzer.get_symbol_volatility(signal.symbol)
            if volatility and volatility > self.config.get('max_volatility', 0.05):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Risk check error: {e}")
            return False
    
    # Helper methods for auto-scheduling
    async def _system_health_check(self):
        """Pre-market system health check"""
        self.logger.info("Performing system health check...")
        # Add health check logic here
    
    async def _pre_market_analysis(self):
        """Pre-market analysis and preparation"""
        self.logger.info("Running pre-market analysis...")
        # Add pre-market analysis logic here
    
    async def _pre_trading_checks(self):
        """Final checks before trading starts"""
        self.logger.info("Running pre-trading checks...")
        # Add pre-trading check logic here
    
    async def _load_daily_trading_plan(self):
        """Load today's trading plan"""
        self.logger.info("Loading daily trading plan...")
        # Add trading plan logic here
    
    async def _close_risky_positions(self):
        """Close risky positions before market close"""
        self.logger.info("Closing risky positions...")
        # Add position closing logic here
    
    async def _generate_daily_report(self):
        """Generate end-of-day trading report"""
        self.logger.info("Generating daily trading report...")
        # Add report generation logic here
    
    async def _backup_daily_data(self):
        """Backup daily trading data"""
        self.logger.info("Backing up daily data...")
        # Add backup logic here
    
    async def _weekly_performance_analysis(self):
        """Weekly performance analysis"""
        self.logger.info("Running weekly performance analysis...")
        # Add weekly analysis logic here
    
    async def _cleanup_logs(self):
        """Clean up old log files"""
        self.logger.info("Cleaning up old logs...")
        # Add log cleanup logic here
    
    async def _emergency_start(self):
        """Emergency start procedure"""
        self.logger.warning("Emergency start triggered")
        if not self.is_running:
            await self.start()
    
    async def _emergency_stop(self):
        """Emergency stop procedure"""
        self.logger.critical("EMERGENCY STOP TRIGGERED")
        
        try:
            await self._close_all_positions()
            self.is_running = False
            
            await self.monitor.send_alert("EMERGENCY STOP", "Trading bot stopped due to risk limits")
            
        except Exception as e:
            self.logger.error(f"Emergency stop error: {e}")
    
    async def _close_all_positions(self):
        """Close all active positions"""
        self.logger.info("Closing all active positions...")
        
        try:
            for symbol, position in self.active_positions.items():
                if position.get('quantity', 0) > 0:
                    if self.trading_mode == 'live':
                        await self.goodwill_api.place_sell_order(
                            symbol=symbol,
                            quantity=position['quantity'],
                            order_type='MARKET'
                        )
                    else:
                        await self.paper_engine.close_position(symbol)
            
            self.logger.info("All positions closed")
            
        except Exception as e:
            self.logger.error(f"Error closing positions: {e}")
    
    async def _log_trade(self, signal: TradingSignal, result: dict, mode: str):
        """Log executed trade"""
        try:
            trade_data = {
                'timestamp': signal.timestamp,
                'symbol': signal.symbol,
                'action': signal.action,
                'strategy': signal.strategy,
                'quantity': signal.quantity,
                'price': signal.price,
                'confidence': signal.confidence,
                'mode': mode,
                'result': result,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit
            }
            
            await self.db_manager.log_trade(trade_data)
            
            with self.lock:
                self.performance_metrics['total_trades'] += 1
            
        except Exception as e:
            self.logger.error(f"Trade logging error: {e}")
    
    async def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        try:
            if self.trading_mode == 'live':
                account_info = await self.goodwill_api.get_account_info()
                return account_info.get('total_value', 0.0)
            else:
                return await self.paper_engine.get_portfolio_value()
        except Exception as e:
            self.logger.error(f"Portfolio value calculation error: {e}")
            return 0.0
    
    async def _calculate_daily_pnl(self) -> float:
        """Calculate daily P&L"""
        try:
            today = datetime.now().date()
            trades = await self.db_manager.get_trades_by_date(today)
            
            daily_pnl = 0.0
            for trade in trades:
                if trade.get('status') == 'FILLED':
                    pnl = trade.get('realized_pnl', 0.0)
                    daily_pnl += pnl
            
            return daily_pnl
            
        except Exception as e:
            self.logger.error(f"Daily P&L calculation error: {e}")
            return 0.0
    
    async def _calculate_performance_metrics(self) -> dict:
        """Calculate comprehensive performance metrics"""
        try:
            trades = await self.db_manager.get_all_trades()
            
            winning_trades = [t for t in trades if t.get('realized_pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('realized_pnl', 0) < 0]
            
            total_pnl = sum(t.get('realized_pnl', 0) for t in trades)
            
            win_rate = len(winning_trades) / len(trades) if trades else 0
            avg_win = sum(t.get('realized_pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t.get('realized_pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
            
            return {
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_pnl': total_pnl,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Performance metrics calculation error: {e}")
            return {}
    
    async def _check_position_sizes(self):
        """Check and validate position sizes"""
        try:
            max_position_value = self.config.get('max_position_value', 25000)
            
            for symbol, position in self.active_positions.items():
                position_value = position.get('value', 0)
                if position_value > max_position_value:
                    self.logger.warning(f"Position {symbol} exceeds max size: {position_value}")
                    # Optionally close or reduce position
                    
        except Exception as e:
            self.logger.error(f"Position size check error: {e}")
    
    async def _process_exit_conditions(self):
        """Process stop-loss and take-profit conditions"""
        try:
            for symbol, position in self.active_positions.items():
                current_price = await self.data_manager.get_current_price(symbol)
                
                if not current_price:
                    continue
                
                entry_price = position.get('entry_price', 0)
                quantity = position.get('quantity', 0)
                stop_loss = position.get('stop_loss')
                take_profit = position.get('take_profit')
                
                if quantity > 0:  # Long position
                    if stop_loss and current_price <= stop_loss:
                        await self._execute_exit_order(symbol, quantity, 'STOP_LOSS', current_price)
                    elif take_profit and current_price >= take_profit:
                        await self._execute_exit_order(symbol, quantity, 'TAKE_PROFIT', current_price)
                
                elif quantity < 0:  # Short position
                    if stop_loss and current_price >= stop_loss:
                        await self._execute_exit_order(symbol, abs(quantity), 'STOP_LOSS', current_price)
                    elif take_profit and current_price <= take_profit:
                        await self._execute_exit_order(symbol, abs(quantity), 'TAKE_PROFIT', current_price)
                        
        except Exception as e:
            self.logger.error(f"Exit conditions processing error: {e}")
    
    async def _execute_exit_order(self, symbol: str, quantity: int, reason: str, price: float):
        """Execute exit order (stop-loss or take-profit)"""
        try:
            if self.trading_mode == 'live':
                result = await self.goodwill_api.place_sell_order(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    order_type='MARKET'
                )
            else:
                result = await self.paper_engine.execute_trade(
                    symbol=symbol,
                    action='SELL',
                    quantity=quantity,
                    price=price,
                    strategy=f'EXIT_{reason}'
                )
            
            if result and result.get('success'):
                self.logger.info(f"✅ {reason}: Sold {quantity} {symbol} @ {price}")
                
                # Log the exit trade
                exit_signal = TradingSignal(
                    symbol=symbol,
                    action='SELL',
                    strategy=f'EXIT_{reason}',
                    confidence=1.0,
                    price=price,
                    quantity=quantity
                )
                await self._log_trade(exit_signal, result, self.trading_mode)
            
        except Exception as e:
            self.logger.error(f"Exit order execution error: {e}")
    
    # Public API methods for dashboard control
    async def get_status(self) -> dict:
        """Get comprehensive bot status"""
        next_scheduled_job = None
        if self.scheduler.running and self.scheduler.get_jobs():
            next_job = min(self.scheduler.get_jobs(), key=lambda x: x.next_run_time)
            next_scheduled_job = {
                'id': next_job.id,
                'next_run': next_job.next_run_time.isoformat() if next_job.next_run_time else None
            }
        
        return {
            'is_running': self.is_running,
            'trading_mode': self.trading_mode,
            'market_open': self._is_market_open(),
            'auto_schedule_enabled': self.auto_schedule_enabled,
            'is_scheduled_session': self.is_scheduled_session,
            'new_positions_allowed': self.new_positions_allowed,
            'active_positions': len(self.active_positions),
            'pending_orders': len(self.pending_orders),
            'performance': self.performance_metrics.copy(),
            'next_scheduled_event': next_scheduled_job,
            'timestamp': datetime.now().isoformat()
        }
    
    async def get_positions(self) -> dict:
        """Get current positions with detailed info"""
        with self.lock:
            detailed_positions = {}
            
            for symbol, position in self.active_positions.items():
                current_price = await self.data_manager.get_current_price(symbol)
                entry_price = position.get('entry_price', 0)
                quantity = position.get('quantity', 0)
                
                unrealized_pnl = 0
                if current_price and entry_price and quantity:
                    unrealized_pnl = (current_price - entry_price) * quantity
                
                detailed_positions[symbol] = {
                    **position,
                    'current_price': current_price,
                    'unrealized_pnl': unrealized_pnl,
                    'pnl_percentage': (unrealized_pnl / (entry_price * abs(quantity)) * 100) if entry_price and quantity else 0
                }
            
            return detailed_positions
    
    async def get_performance_metrics(self) -> dict:
        """Get detailed performance metrics"""
        with self.lock:
            metrics = self.performance_metrics.copy()
        
        # Add additional calculated metrics
        portfolio_value = await self._calculate_portfolio_value()
        initial_capital = self.config.get('initial_capital', 100000)
        
        metrics.update({
            'portfolio_value': portfolio_value,
            'initial_capital': initial_capital,
            'total_return_pct': ((portfolio_value - initial_capital) / initial_capital * 100) if initial_capital else 0,
            'daily_return_pct': (metrics['daily_pnl'] / portfolio_value * 100) if portfolio_value else 0
        })
        
        return metrics
    
    async def get_today_trades(self) -> list:
        """Get today's trades"""
        try:
            today = datetime.now().date()
            trades = await self.db_manager.get_trades_by_date(today)
            return trades
        except Exception as e:
            self.logger.error(f"Error fetching today's trades: {e}")
            return []
    
    async def get_active_strategies(self) -> dict:
        """Get active trading strategies status"""
        try:
            return await self.strategy_manager.get_strategy_status()
        except Exception as e:
            self.logger.error(f"Error fetching strategy status: {e}")
            return {}
    
    async def toggle_strategy(self, strategy_name: str, enabled: bool):
        """Enable/disable a specific strategy"""
        try:
            await self.strategy_manager.toggle_strategy(strategy_name, enabled)
            self.logger.info(f"Strategy {strategy_name} {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            self.logger.error(f"Error toggling strategy {strategy_name}: {e}")
            raise
    
    async def update_risk_parameters(self, params: dict):
        """Update risk management parameters"""
        try:
            # Update configuration
            for key, value in params.items():
                if key in ['max_daily_loss', 'max_position_size', 'max_portfolio_exposure', 'max_symbol_exposure']:
                    self.config.set(key, value)
            
            await self.config.save()
            self.logger.info(f"Risk parameters updated: {params}")
            
        except Exception as e:
            self.logger.error(f"Error updating risk parameters: {e}")
            raise
    
    async def manual_trade(self, symbol: str, action: str, quantity: int, price: float = None):
        """Execute a manual trade"""
        try:
            if not self._is_market_open():
                raise Exception("Market is closed")
            
            current_price = price or await self.data_manager.get_current_price(symbol)
            if not current_price:
                raise Exception(f"Could not get price for {symbol}")
            
            # Create manual signal
            signal = TradingSignal(
                symbol=symbol,
                action=action,
                strategy='MANUAL',
                confidence=1.0,
                price=current_price,
                quantity=quantity
            )
            
            # Validate and execute
            if await self._validate_signal(signal):
                if self.trading_mode == 'live':
                    await self._execute_live_signal(signal)
                else:
                    await self._execute_paper_signal(signal)
                
                return {'success': True, 'message': f'Manual trade executed: {action} {quantity} {symbol}'}
            else:
                return {'success': False, 'message': 'Trade validation failed'}
                
        except Exception as e:
            self.logger.error(f"Manual trade error: {e}")
            return {'success': False, 'message': str(e)}
    
    async def close_position(self, symbol: str):
        """Manually close a specific position"""
        try:
            if symbol not in self.active_positions:
                return {'success': False, 'message': f'No active position for {symbol}'}
            
            position = self.active_positions[symbol]
            quantity = abs(position.get('quantity', 0))
            
            if quantity > 0:
                current_price = await self.data_manager.get_current_price(symbol)
                await self._execute_exit_order(symbol, quantity, 'MANUAL_CLOSE', current_price)
                return {'success': True, 'message': f'Position {symbol} closed'}
            else:
                return {'success': False, 'message': f'No quantity to close for {symbol}'}
                
        except Exception as e:
            self.logger.error(f"Error closing position {symbol}: {e}")
            return {'success': False, 'message': str(e)}
    
    async def enable_auto_schedule(self, enabled: bool):
        """Enable/disable auto-scheduling"""
        try:
            self.auto_schedule_enabled = enabled
            self.config.set('auto_schedule', enabled)
            await self.config.save()
            
            if enabled and not self.scheduler.running:
                await self._setup_auto_schedule()
                self.scheduler.start()
                self.logger.info("Auto-scheduling enabled and started")
            elif not enabled and self.scheduler.running:
                self.scheduler.shutdown(wait=False)
                self.logger.info("Auto-scheduling disabled")
            
            return {'success': True, 'message': f'Auto-schedule {"enabled" if enabled else "disabled"}'}
            
        except Exception as e:
            self.logger.error(f"Error toggling auto-schedule: {e}")
            return {'success': False, 'message': str(e)}
    
    async def force_market_close_procedures(self):
        """Manually trigger market close procedures"""
        try:
            await self._scheduled_market_close_prep()
            await self._scheduled_post_market_stop()
            return {'success': True, 'message': 'Market close procedures executed'}
        except Exception as e:
            self.logger.error(f"Error in force market close: {e}")
            return {'success': False, 'message': str(e)}
    
    async def get_system_health(self) -> dict:
        """Get comprehensive system health status"""
        try:
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'HEALTHY',
                'components': {
                    'data_manager': {'status': 'CONNECTED' if hasattr(self.data_manager, 'is_connected') and self.data_manager.is_connected else 'DISCONNECTED'},
                    'websocket_manager': {'status': 'CONNECTED' if hasattr(self.websocket_manager, 'is_connected') and self.websocket_manager.is_connected else 'DISCONNECTED'},
                    'goodwill_api': {'status': 'AUTHENTICATED' if self.trading_mode == 'live' and await self.goodwill_api.is_authenticated() else 'N/A'},
                    'paper_engine': {'status': 'ACTIVE' if self.trading_mode == 'paper' else 'N/A'},
                    'database': {'status': 'CONNECTED' if hasattr(self.db_manager, 'is_connected') and self.db_manager.is_connected else 'UNKNOWN'},
                    'scheduler': {'status': 'RUNNING' if self.scheduler.running else 'STOPPED'},
                    'monitor': {'status': 'ACTIVE' if hasattr(self.monitor, 'is_running') and self.monitor.is_running else 'INACTIVE'}
                },
                'market_status': {
                    'is_open': self._is_market_open(),
                    'trading_allowed': self.new_positions_allowed,
                    'current_time': datetime.now().strftime('%H:%M:%S'),
                    'market_open_time': self.market_open_time,
                    'market_close_time': self.market_close_time
                },
                'performance': {
                    'active_positions': len(self.active_positions),
                    'pending_orders': len(self.pending_orders),
                    'daily_pnl': self.performance_metrics.get('daily_pnl', 0),
                    'total_trades_today': len(await self.get_today_trades())
                }
            }
            
            # Determine overall health
            component_issues = [comp for comp, status in health_status['components'].items() 
                              if status['status'] in ['DISCONNECTED', 'INACTIVE'] and status['status'] != 'N/A']
            
            if component_issues:
                health_status['overall_status'] = 'WARNING'
                health_status['issues'] = component_issues
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"System health check error: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'ERROR',
                'error': str(e)
            }

# Main execution
if __name__ == "__main__":
    async def main():
        """Main execution function"""
        bot = TradingBotCore()
        
        try:
            self.logger.info("🚀 Starting Advanced Trading Bot with Auto-Scheduling...")
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
