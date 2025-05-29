#!/usr/bin/env python3
"""
Core Trading Bot with Paper Trading Integration
Integrates all components for production trading
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from paper_trading_engine import PaperTradingEngine
from goodwill_api_handler import GoodwillAPIHandler, APICredentials

logger = logging.getLogger(__name__)

class TradingBotCore:
    """Main trading bot that orchestrates all components"""
    
    def __init__(self):
        self.running = False
        self.paper_engine = None
        self.api_handler = None
        self.paper_mode = True  # Start in paper mode
        
        # Bot statistics
        self.start_time = None
        self.total_trades = 0
        self.successful_trades = 0
        self.total_pnl = 0.0
        
        # Configuration
        self.config = {
            'initial_capital': 100000.0,
            'max_positions': 10,
            'risk_per_trade': 1.0,  # 1% risk per trade
            'paper_mode': True
        }
        
        logger.info("🤖 Trading Bot Core initialized")
    
    def initialize_api(self, credentials: APICredentials = None):
        """Initialize API connection"""
        try:
            if credentials:
                self.api_handler = GoodwillAPIHandler(credentials)
                if self.api_handler.login():
                    logger.info("✅ API connection established")
                    return True
                else:
                    logger.error("❌ API login failed")
                    return False
            else:
                logger.warning("⚠️ No API credentials provided - using mock mode")
                return True
                
        except Exception as e:
            logger.error(f"❌ API initialization error: {e}")
            return False
    
    def initialize_paper_trading(self, initial_capital: float = 100000.0):
        """Initialize paper trading engine"""
        try:
            self.paper_engine = PaperTradingEngine(
                self.api_handler, 
                initial_capital=initial_capital
            )
            self.paper_engine.start_realtime_updates()
            
            logger.info(f"✅ Paper trading initialized with ₹{initial_capital:,.2f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Paper trading initialization error: {e}")
            return False
    
    def start_bot(self):
        """Start the trading bot"""
        try:
            if self.running:
                logger.warning("⚠️ Bot is already running")
                return
            
            self.running = True
            self.start_time = datetime.now()
            
            # Start main bot thread
            bot_thread = threading.Thread(
                target=self._bot_main_loop,
                daemon=True,
                name="TradingBotMain"
            )
            bot_thread.start()
            
            logger.info("🚀 Trading Bot started")
            
        except Exception as e:
            logger.error(f"❌ Bot start error: {e}")
            self.running = False
    
    def stop_bot(self):
        """Stop the trading bot"""
        try:
            self.running = False
            
            if self.paper_engine:
                self.paper_engine.stop_realtime_updates()
            
            if self.api_handler:
                self.api_handler.logout()
            
            logger.info("⏹️ Trading Bot stopped")
            
        except Exception as e:
            logger.error(f"❌ Bot stop error: {e}")
    
    def _bot_main_loop(self):
        """Main bot execution loop"""
        while self.running:
            try:
                # Update bot statistics
                self._update_statistics()
                
                # Health check
                self._health_check()
                
                # Sleep for 10 seconds
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"❌ Bot main loop error: {e}")
                time.sleep(30)  # Wait before retry
    
    def _update_statistics(self):
        """Update bot statistics"""
        try:
            if self.paper_engine:
                portfolio_summary = self.paper_engine.get_portfolio_summary()
                
                if 'account_summary' in portfolio_summary:
                    account = portfolio_summary['account_summary']
                    self.total_pnl = account.get('total_pnl', 0.0)
                
                if 'performance_metrics' in portfolio_summary:
                    metrics = portfolio_summary['performance_metrics']
                    self.total_trades = metrics.get('total_trades', 0)
                    self.successful_trades = metrics.get('winning_trades', 0)
                    
        except Exception as e:
            logger.debug(f"Statistics update error: {e}")
    
    def _health_check(self):
        """Perform health checks"""
        try:
            # Check API connection
            if self.api_handler and not self.api_handler.is_logged_in():
                logger.warning("⚠️ API connection lost - attempting reconnection")
                self.api_handler.refresh_session()
            
            # Check paper engine
            if self.paper_engine:
                # Basic health check
                pass
                
        except Exception as e:
            logger.debug(f"Health check error: {e}")
    
    def run(self):
        """Run method for thread execution"""
        try:
            # Initialize components
            self.initialize_paper_trading(self.config['initial_capital'])
            
            # Start bot
            self.start_bot()
            
            # Keep running
            while self.running:
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"❌ Bot run error: {e}")
    
    def get_bot_status(self) -> Dict:
        """Get current bot status"""
        try:
            uptime = str(datetime.now() - self.start_time) if self.start_time else "0:00:00"
            
            return {
                'running': self.running,
                'paper_mode': self.paper_mode,
                'uptime': uptime,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'total_trades': self.total_trades,
                'successful_trades': self.successful_trades,
                'win_rate': (self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
                'total_pnl': round(self.total_pnl, 2),
                'api_connected': self.api_handler.is_logged_in() if self.api_handler else False,
                'paper_engine_active': self.paper_engine is not None
            }
            
        except Exception as e:
            logger.error(f"❌ Bot status error: {e}")
            return {'error': str(e)}
    
    def toggle_paper_mode(self):
        """Toggle between paper and live trading"""
        try:
            self.paper_mode = not self.paper_mode
            logger.info(f"📊 Trading mode: {'Paper' if self.paper_mode else 'Live'}")
            return self.paper_mode
            
        except Exception as e:
            logger.error(f"❌ Toggle paper mode error: {e}")
            return self.paper_mode
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        try:
            if self.paper_engine:
                return self.paper_engine.get_portfolio_summary()
            else:
                return {'error': 'Paper engine not initialized'}
                
        except Exception as e:
            logger.error(f"❌ Portfolio summary error: {e}")
            return {'error': str(e)}
