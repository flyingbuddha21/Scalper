#!/usr/bin/env python3
"""
Enhanced Paper Trading Engine with Advanced Risk Management Integration
Production-ready paper trading with realistic execution, slippage simulation, and risk controls
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import sqlite3
import json
import uuid
import asyncio
import random

logger = logging.getLogger(__name__)

@dataclass
class PaperPosition:
    """Enhanced paper trading position"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    
    # Risk management integration
    stop_loss: float = 0.0
    profit_target: float = 0.0
    trailing_stop: float = 0.0
    position_size_method: str = "RISK_BASED"
    risk_amount: float = 0.0
    
    # Performance tracking
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    max_drawdown: float = 0.0
    max_profit: float = 0.0
    
    # Execution details
    entry_slippage: float = 0.0
    entry_commission: float = 0.0
    order_id: str = ""
    
    # Market data tracking
    highest_price: float = 0.0
    lowest_price: float = 0.0
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate current unrealized P&L with realistic factors"""
        self.current_price = current_price
        
        if self.side == 'BUY':
            pnl_per_share = current_price - self.entry_price
            
            # Track highest price for trailing
            if current_price > self.highest_price:
                self.highest_price = current_price
                
        else:  # SELL
            pnl_per_share = self.entry_price - current_price
            
            # Track lowest price for trailing
            if current_price < self.lowest_price or self.lowest_price == 0:
                self.lowest_price = current_price
        
        self.unrealized_pnl = pnl_per_share * self.quantity
        
        # Update max profit and drawdown
        if self.unrealized_pnl > self.max_profit:
            self.max_profit = self.unrealized_pnl
        
        drawdown = self.max_profit - self.unrealized_pnl
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            
        return self.unrealized_pnl
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'current_price': self.current_price,
            'stop_loss': self.stop_loss,
            'profit_target': self.profit_target,
            'trailing_stop': self.trailing_stop,
            'unrealized_pnl': round(self.unrealized_pnl, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'max_profit': round(self.max_profit, 2),
            'position_size_method': self.position_size_method,
            'risk_amount': self.risk_amount,
            'entry_slippage': self.entry_slippage,
            'entry_commission': self.entry_commission,
            'order_id': self.order_id
        }

@dataclass
class PaperTrade:
    """Completed paper trade record"""
    trade_id: str
    symbol: str
    side: str
    quantity: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    commission: float = 0.0
    slippage: float = 0.0
    hold_time_minutes: float = 0.0
    exit_reason: str = "MANUAL"
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat(),
            'pnl': round(self.pnl, 2),
            'commission': self.commission,
            'slippage': self.slippage,
            'hold_time_minutes': round(self.hold_time_minutes, 1),
            'exit_reason': self.exit_reason,
            'max_favorable_excursion': round(self.max_favorable_excursion, 2),
            'max_adverse_excursion': round(self.max_adverse_excursion, 2)
        }

class EnhancedPaperTradingEngine:
    """
    Enhanced Paper Trading Engine with Risk Management Integration
    
    Features:
    - Realistic execution simulation with slippage
    - Risk manager integration for position sizing
    - Advanced performance analytics
    - Kelly Criterion and volatility-based sizing
    - Market impact simulation
    - Commission and cost modeling
    """
    
    def __init__(self, initial_capital: float = 100000, 
                 commission_per_trade: float = 20.0,
                 max_positions: int = 10,
                 risk_manager=None):
        
        # Account settings
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.available_capital = initial_capital
        self.commission_per_trade = commission_per_trade
        self.max_positions = max_positions
        self.risk_manager = risk_manager
        
        # Position tracking
        self.positions = {}  # symbol -> PaperPosition
        self.trade_history = deque(maxlen=1000)
        self.daily_pnl_history = deque(maxlen=365)
        self.pending_orders = {}  # order_id -> order_details
        
        # Execution simulation parameters
        self.base_slippage_bps = 2.0      # 2 basis points base slippage
        self.volatility_slippage_factor = 0.5  # Additional slippage in volatile markets
        self.market_impact_threshold = 50000   # Order value for market impact
        self.max_slippage_bps = 20.0      # Maximum slippage cap
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_rate = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.profit_factor = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_capital = initial_capital
        
        # Kelly Criterion and position sizing
        self.kelly_lookback_trades = 30
        self.recent_trades = deque(maxlen=self.kelly_lookback_trades)
        self.position_sizing_methods = ['FIXED', 'KELLY', 'VOLATILITY_ADJUSTED', 'RISK_BASED']
        
        # Market data for realistic simulation
        self.market_data_cache = {}
        self.volatility_cache = {}
        
        # Database for persistence
        self.db_path = "paper_trading.db"
        self._init_database()
        
        logger.info(f"💰 Enhanced Paper Trading Engine initialized with ₹{initial_capital:,.2f}")
    
    def _init_database(self):
        """Initialize SQLite database for trade history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create enhanced trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    side TEXT,
                    quantity INTEGER,
                    entry_price REAL,
                    exit_price REAL,
                    entry_time TEXT,
                    exit_time TEXT,
                    pnl REAL,
                    commission REAL,
                    slippage REAL,
                    hold_time_minutes REAL,
                    exit_reason TEXT,
                    max_favorable_excursion REAL,
                    max_adverse_excursion REAL,
                    position_size_method TEXT,
                    risk_amount REAL
                )
            ''')
            
            # Create daily performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_performance (
                    date TEXT PRIMARY KEY,
                    capital REAL,
                    daily_pnl REAL,
                    trades_count INTEGER,
                    win_rate REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL
                )
            ''')
            
            # Create positions table for current positions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS current_positions (
                    symbol TEXT PRIMARY KEY,
                    side TEXT,
                    quantity INTEGER,
                    entry_price REAL,
                    entry_time TEXT,
                    stop_loss REAL,
                    profit_target REAL,
                    position_size_method TEXT,
                    risk_amount REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"⚠️ Database initialization warning: {e}")
    
    def update_market_data(self, symbol: str, price: float, volume: int = 0, 
                          bid: float = None, ask: float = None, volatility: float = None):
        """Update market data for realistic execution simulation"""
        try:
            self.market_data_cache[symbol] = {
                'price': price,
                'volume': volume,
                'bid': bid or price * 0.999,
                'ask': ask or price * 1.001,
                'timestamp': datetime.now(),
                'spread': (ask - bid) if (ask and bid) else price * 0.002
            }
            
            if volatility:
                self.volatility_cache[symbol] = volatility
            
            # Update positions with current price
            if symbol in self.positions:
                self.positions[symbol].calculate_pnl(price)
                
        except Exception as e:
            logger.error(f"❌ Market data update error for {symbol}: {e}")
    
    def calculate_optimal_position_size(self, symbol: str, entry_price: float, 
                                      stop_loss_price: float, confidence: float = 50.0,
                                      method: str = "RISK_BASED") -> Dict:
        """Calculate optimal position size using various methods with risk manager integration"""
        try:
            # Use risk manager if available
            if self.risk_manager:
                risk_decision = self.risk_manager.should_enter_trade(
                    symbol=symbol,
                    side="BUY",  # Default side for position sizing
                    confidence=confidence,
                    entry_price=entry_price
                )
                
                if risk_decision['allow_trade']:
                    return {
                        'quantity': risk_decision['position_size'],
                        'position_value': risk_decision['position_size'] * entry_price,
                        'risk_amount': risk_decision['position_size'] * entry_price * (risk_decision['stop_loss_pct'] / 100),
                        'risk_percentage': risk_decision['stop_loss_pct'],
                        'method_used': 'RISK_MANAGER',
                        'confidence_applied': confidence,
                        'max_risk_per_trade': risk_decision.get('stop_loss_pct', 2.0)
                    }
                else:
                    return {
                        'quantity': 0,
                        'position_value': 0,
                        'risk_amount': 0,
                        'risk_percentage': 0,
                        'method_used': 'REJECTED',
                        'rejection_reason': risk_decision['reason']
                    }
            
            # Fallback to internal position sizing if no risk manager
            return self._internal_position_sizing(symbol, entry_price, stop_loss_price, confidence, method)
            
        except Exception as e:
            logger.error(f"❌ Position size calculation error: {e}")
            return self._safe_default_sizing(entry_price)
    
    def _internal_position_sizing(self, symbol: str, entry_price: float, 
                                stop_loss_price: float, confidence: float, method: str) -> Dict:
        """Internal position sizing methods"""
        try:
            base_risk_amount = self.available_capital * 0.02  # 2% base risk
            
            if method == "FIXED":
                position_value = self.available_capital * 0.1
                quantity = int(position_value / entry_price)
                
            elif method == "KELLY":
                kelly_fraction = self._calculate_kelly_fraction()
                if kelly_fraction > 0:
                    position_value = self.available_capital * kelly_fraction * 0.25  # Conservative Kelly
                    quantity = int(position_value / entry_price)
                else:
                    position_value = self.available_capital * 0.05
                    quantity = int(position_value / entry_price)
                    
            elif method == "VOLATILITY_ADJUSTED":
                volatility = self.volatility_cache.get(symbol, 20.0)  # Default 20% volatility
                volatility_factor = max(0.5, min(2.0, 20.0 / volatility))  # Inverse relationship
                base_position_value = self.available_capital * 0.1
                adjusted_position_value = base_position_value * volatility_factor
                quantity = int(adjusted_position_value / entry_price)
                
            else:  # RISK_BASED (default)
                if stop_loss_price > 0:
                    risk_per_share = abs(entry_price - stop_loss_price)
                    quantity = int(base_risk_amount / risk_per_share) if risk_per_share > 0 else 1
                else:
                    quantity = int(base_risk_amount / (entry_price * 0.02))
            
            # Apply confidence adjustment
            confidence_multiplier = confidence / 100.0
            quantity = int(quantity * confidence_multiplier)
            
            # Apply position limits
            max_position_value = self.available_capital * 0.2  # 20% max
            max_quantity = int(max_position_value / entry_price)
            quantity = max(1, min(quantity, max_quantity))
            
            # Calculate final metrics
            position_value = quantity * entry_price
            risk_amount = quantity * abs(entry_price - stop_loss_price) if stop_loss_price > 0 else position_value * 0.02
            
            return {
                'quantity': quantity,
                'position_value': position_value,
                'risk_amount': risk_amount,
                'risk_percentage': (risk_amount / self.available_capital) * 100,
                'method_used': method,
                'confidence_applied': confidence,
                'kelly_fraction': self._calculate_kelly_fraction() if method == "KELLY" else 0.0
            }
            
        except Exception as e:
            logger.error(f"❌ Internal position sizing error: {e}")
            return self._safe_default_sizing(entry_price)
    
    def _safe_default_sizing(self, entry_price: float) -> Dict:
        """Safe default position sizing"""
        quantity = max(1, int(self.available_capital * 0.05 / entry_price))  # 5% of capital
        return {
            'quantity': quantity,
            'position_value': quantity * entry_price,
            'risk_amount': quantity * entry_price * 0.02,
            'risk_percentage': 2.0,
            'method_used': 'SAFE_DEFAULT',
            'confidence_applied': 50.0
        }
    
    def _calculate_kelly_fraction(self) -> float:
        """Calculate Kelly Criterion fraction based on recent trades"""
        try:
            if len(self.recent_trades) < 10:
                return 0.0
            
            wins = [trade for trade in self.recent_trades if trade.pnl > 0]
            losses = [trade for trade in self.recent_trades if trade.pnl < 0]
            
            if len(wins) == 0 or len(losses) == 0:
                return 0.0
            
            win_probability = len(wins) / len(self.recent_trades)
            avg_win_pct = np.mean([trade.pnl / (trade.quantity * trade.entry_price) for trade in wins])
            avg_loss_pct = abs(np.mean([trade.pnl / (trade.quantity * trade.entry_price) for trade in losses]))
            
            if avg_loss_pct > 0:
                b = avg_win_pct / avg_loss_pct
                p = win_probability
                q = 1 - p
                kelly_fraction = (b * p - q) / b
                return max(0.0, min(0.25, kelly_fraction))  # Cap at 25%
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _simulate_execution_slippage(self, symbol: str, side: str, quantity: int, 
                                   price: float, order_type: str = "MARKET") -> Dict:
        """Simulate realistic execution slippage and market impact"""
        try:
            market_data = self.market_data_cache.get(symbol, {})
            
            # Base slippage
            base_slippage = self.base_slippage_bps / 10000  # Convert bps to decimal
            
            # Volatility-based slippage
            volatility = self.volatility_cache.get(symbol, 20.0)
            volatility_slippage = (volatility / 20.0) * self.volatility_slippage_factor / 10000
            
            # Market impact based on order size
            order_value = quantity * price
            if order_value > self.market_impact_threshold:
                impact_factor = min(2.0, order_value / self.market_impact_threshold)
                market_impact = (impact_factor - 1.0) * 0.001  # Additional impact
            else:
                market_impact = 0.0
            
            # Order type impact
            if order_type == "MARKET":
                # Market orders get bid-ask spread impact
                spread = market_data.get('spread', price * 0.002)
                spread_impact = (spread / price) / 2  # Half spread
            else:
                spread_impact = 0.0  # Limit orders avoid spread
            
            # Random execution variation
            random_slippage = random.uniform(-0.0001, 0.0001)  # ±1 bps random
            
            # Total slippage
            total_slippage_pct = base_slippage + volatility_slippage + market_impact + spread_impact + random_slippage
            total_slippage_pct = min(total_slippage_pct, self.max_slippage_bps / 10000)  # Cap slippage
            
            # Apply slippage direction
            if side == 'BUY':
                execution_price = price * (1 + total_slippage_pct)
            else:  # SELL
                execution_price = price * (1 - total_slippage_pct)
            
            return {
                'execution_price': execution_price,
                'slippage_pct': total_slippage_pct * 100,
                'slippage_amount': abs(execution_price - price) * quantity,
                'components': {
                    'base_slippage': base_slippage * 100,
                    'volatility_slippage': volatility_slippage * 100,
                    'market_impact': market_impact * 100,
                    'spread_impact': spread_impact * 100
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Execution simulation error: {e}")
            return {
                'execution_price': price,
                'slippage_pct': 0.0,
                'slippage_amount': 0.0,
                'components': {}
            }
    
    async def execute_trade(self, symbol: str, side: str, quantity: int, 
                           price: float, order_type: str = "MARKET") -> Dict:
        """Execute paper trade with realistic simulation"""
        try:
            # Validate trade parameters
            if side.upper() == 'BUY':
                return await self._execute_entry_trade(symbol, side, quantity, price, order_type)
            else:  # SELL - exit existing position
                return await self._execute_exit_trade(symbol, quantity, price, order_type)
                
        except Exception as e:
            logger.error(f"❌ Trade execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_entry_trade(self, symbol: str, side: str, quantity: int, 
                                  price: float, order_type: str) -> Dict:
        """Execute entry trade (open position)"""
        try:
            # Check if position already exists
            if symbol in self.positions:
                return {
                    'success': False,
                    'error': f'Position already exists for {symbol}'
                }
            
            # Check available capital
            estimated_cost = quantity * price + self.commission_per_trade
            if estimated_cost > self.available_capital:
                return {
                    'success': False,
                    'error': 'Insufficient capital',
                    'required': estimated_cost,
                    'available': self.available_capital
                }
            
            # Simulate execution
            execution_result = self._simulate_execution_slippage(symbol, side, quantity, price, order_type)
            execution_price = execution_result['execution_price']
            slippage_amount = execution_result['slippage_amount']
            
            # Calculate total cost
            position_cost = quantity * execution_price
            total_cost = position_cost + self.commission_per_trade + slippage_amount
            
            # Create position
            order_id = str(uuid.uuid4())
            position = PaperPosition(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=execution_price,
                entry_time=datetime.now(),
                entry_slippage=execution_result['slippage_pct'],
                entry_commission=self.commission_per_trade,
                order_id=order_id,
                highest_price=execution_price,
                lowest_price=execution_price
            )
            
            # Update capital
            self.available_capital -= total_cost
            
            # Store position
            self.positions[symbol] = position
            
            # Save to database
            self._save_position_to_db(position)
            
            logger.info(f"📈 Paper trade executed: {symbol} {side} {quantity}@{execution_price:.2f} "
                       f"(Slippage: {execution_result['slippage_pct']:.3f}%)")
            
            return {
                'success': True,
                'order_id': order_id,
                'execution_price': execution_price,
                'quantity': quantity,
                'total_cost': total_cost,
                'slippage': execution_result['slippage_pct'],
                'commission': self.commission_per_trade,
                'available_capital': self.available_capital
            }
            
        except Exception as e:
            logger.error(f"❌ Entry trade execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_exit_trade(self, symbol: str, quantity: int, 
                                 price: float, order_type: str, exit_reason: str = "MANUAL") -> Dict:
        """Execute exit trade (close position)"""
        try:
            if symbol not in self.positions:
                return {
                    'success': False,
                    'error': f'No position found for {symbol}'
                }
            
            position = self.positions[symbol]
            
            # Validate quantity
            if quantity > position.quantity:
                quantity = position.quantity  # Close partial or full position
            
            # Simulate execution
            execution_result = self._simulate_execution_slippage(symbol, "SELL", quantity, price, order_type)
            execution_price = execution_result['execution_price']
            
            # Calculate P&L
            if position.side == 'BUY':
                pnl_per_share = execution_price - position.entry_price
            else:  # SELL position (short)
                pnl_per_share = position.entry_price - execution_price
            
            gross_pnl = pnl_per_share * quantity
            commission = self.commission_per_trade
            slippage_cost = execution_result['slippage_amount']
            net_pnl = gross_pnl - commission - slippage_cost
            
            # Calculate hold time
            hold_time = (datetime.now() - position.entry_time).total_seconds() / 60
            
            # Create trade record
            trade_id = str(uuid.uuid4())
            trade = PaperTrade(
                trade_id=trade_id,
                symbol=symbol,
                side=position.side,
                quantity=quantity,
                entry_price=position.entry_price,
                exit_price=execution_price,
                entry_time=position.entry_time,
                exit_time=datetime.now(),
                pnl=net_pnl,
                commission=commission,
                slippage=execution_result['slippage_pct'],
                hold_time_minutes=hold_time,
                exit_reason=exit_reason,
                max_favorable_excursion=position.max_profit / quantity if quantity > 0 else 0,
                max_adverse_excursion=position.max_drawdown / quantity if quantity > 0 else 0
            )
            
            # Update capital
            proceeds = quantity * execution_price - commission - slippage_cost
            self.available_capital += proceeds
            self.current_capital += net_pnl
            
            # Update position or remove if fully closed
            if quantity == position.quantity:
                # Full exit
                del self.positions[symbol]
                self._remove_position_from_db(symbol)
            else:
                # Partial exit
                position.quantity -= quantity
                position.realized_pnl += net_pnl
                self._save_position_to_db(position)
            
            # Add to trade history
            self.trade_history.append(trade)
            self.recent_trades.append(trade)
            
            # Update performance metrics
            self._update_performance_metrics(trade)
            
            # Save trade to database
            self._save_trade_to_db(trade)
            
            # Update risk manager if available
            if self.risk_manager:
                self.risk_manager.update_performance(
                    symbol=symbol,
                    entry_price=position.entry_price,
                    exit_price=execution_price,
                    quantity=quantity,
                    side=position.side,
                    exit_reason=exit_reason
                )
            
            logger.info(f"📉 Paper trade closed: {symbol} - P&L: ₹{net_pnl:.2f} ({exit_reason})")
            
            return {
                'success': True,
                'trade_id': trade_id,
                'execution_price': execution_price,
                'pnl': net_pnl,
                'commission': commission,
                'slippage': execution_result['slippage_pct'],
                'hold_time_minutes': hold_time,
                'current_capital': self.current_capital,
                'available_capital': self.available_capital
            }
            
        except Exception as e:
            logger.error(f"❌ Exit trade execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    def enter_position(self, symbol: str, side: str, entry_price: float, 
                      stop_loss_pct: float = 2.0, profit_target_pct: float = 4.0,
                      confidence: float = 50.0, position_size_method: str = "RISK_BASED") -> Dict:
        """Enter position with risk management integration"""
        try:
            # Calculate optimal position size
            stop_loss_price = entry_price * (1 - stop_loss_pct/100) if side == 'BUY' else entry_price * (1 + stop_loss_pct/100)
            
            sizing_result = self.calculate_optimal_position_size(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                confidence=confidence,
                method=position_size_method
            )
            
            if sizing_result['quantity'] == 0:
                return {
                    'success': False,
                    'error': sizing_result.get('rejection_reason', 'Position size calculation failed'),
                    'sizing_result': sizing_result
                }
            
            # Execute the trade
            result = asyncio.run(self.execute_trade(
                symbol=symbol,
                side=side,
                quantity=sizing_result['quantity'],
                price=entry_price,
                order_type="LIMIT"
            ))
            
            if result['success']:
                # Set stop loss and profit target
                position = self.positions[symbol]
                
                if side == 'BUY':
                    position.stop_loss = entry_price * (1 - stop_loss_pct/100)
                    position.profit_target = entry_price * (1 + profit_target_pct/100)
                else:
                    position.stop_loss = entry_price * (1 + stop_loss_pct/100)
                    position.profit_target = entry_price * (1 - profit_target_pct/100)
                
                position.position_size_method = position_size_method
                position.risk_amount = sizing_result['risk_amount']
                
                # Update database
                self._save_position_to_db(position)
                
                result['position_sizing'] = sizing_result
                result['stop_loss'] = position.stop_loss
                result['profit_target'] = position.profit_target
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Enter position error: {e}")
            return {'success': False, 'error': str(e)}
    
    def exit_position(self, symbol: str, exit_reason: str = "MANUAL") -> Dict:
        """Exit position using current market price"""
        try:
            if symbol not in self.positions:
                return {'success': False, 'error': f'No position found for {symbol}'}
            
            position = self.positions[symbol]
            current_price = position.current_price or position.entry_price
            
            return asyncio.run(self._execute_exit_trade(
                symbol=symbol,
                quantity=position.quantity,
                price=current_price,
                order_type="MARKET",
                exit_reason=exit_reason
            ))
            
        except Exception as e:
            logger.error(f"❌ Exit position error: {e}")
            return {'success': False, 'error': str(e)}
    
    def check_exit_conditions(self, symbol: str) -> Optional[Dict]:
        """Check if position should be exited based on stop loss/profit target"""
        try:
            if symbol not in self.positions:
                return None
            
            position = self.positions[symbol]
            current_price = position.current_price
            
            if current_price <= 0:
                return None
            
            # Check stop loss
            if position.stop_loss > 0:
                if position.side == 'BUY' and current_price <= position.stop_loss:
                    return {
                        'should_exit': True,
                        'reason': 'STOP_LOSS',
                        'trigger_price': position.stop_loss,
                        'current_price': current_price
                    }
                elif position.side == 'SELL' and current_price >= position.stop_loss:
                    return {
                        'should_exit': True,
                        'reason': 'STOP_LOSS',
                        'trigger_price': position.stop_loss,
                        'current_price': current_price
                    }
            
            # Check profit target
            if position.profit_target > 0:
                if position.side == 'BUY' and current_price >= position.profit_target:
                    return {
                        'should_exit': True,
                        'reason': 'PROFIT_TARGET',
                        'trigger_price': position.profit_target,
                        'current_price': current_price
                    }
                elif position.side == 'SELL' and current_price <= position.profit_target:
                    return {
                        'should_exit': True,
                        'reason': 'PROFIT_TARGET',
                        'trigger_price': position.profit_target,
                        'current_price': current_price
                    }
            
            # Check trailing stop if available
            if position.trailing_stop > 0:
                if position.side == 'BUY' and current_price <= position.trailing_stop:
                    return {
                        'should_exit': True,
                        'reason': 'TRAILING_STOP',
                        'trigger_price': position.trailing_stop,
                        'current_price': current_price
                    }
                elif position.side == 'SELL' and current_price >= position.trailing_stop:
                    return {
                        'should_exit': True,
                        'reason': 'TRAILING_STOP',
                        'trigger_price': position.trailing_stop,
                        'current_price': current_price
                    }
            
            return {'should_exit': False, 'reason': 'NO_EXIT_CONDITION'}
            
        except Exception as e:
            logger.error(f"❌ Exit condition check error: {e}")
            return None
    
    def update_trailing_stop(self, symbol: str, trailing_distance_pct: float = 2.0) -> bool:
        """Update trailing stop for position"""
        try:
            if symbol not in self.positions:
                return False
            
            position = self.positions[symbol]
            current_price = position.current_price
            
            if current_price <= 0:
                return False
            
            if position.side == 'BUY':
                # Trail below current price
                new_trailing_stop = current_price * (1 - trailing_distance_pct / 100)
                # Only move trailing stop up
                if new_trailing_stop > position.trailing_stop:
                    position.trailing_stop = new_trailing_stop
                    logger.info(f"📈 Updated trailing stop for {symbol}: {new_trailing_stop:.2f}")
                    return True
            else:  # SELL
                # Trail above current price
                new_trailing_stop = current_price * (1 + trailing_distance_pct / 100)
                # Only move trailing stop down
                if position.trailing_stop == 0 or new_trailing_stop < position.trailing_stop:
                    position.trailing_stop = new_trailing_stop
                    logger.info(f"📉 Updated trailing stop for {symbol}: {new_trailing_stop:.2f}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Trailing stop update error: {e}")
            return False
    
    def _update_performance_metrics(self, trade: PaperTrade):
        """Update trading performance metrics"""
        try:
            self.total_trades += 1
            
            if trade.pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Calculate win rate
            self.win_rate = (self.winning_trades / self.total_trades) * 100
            
            # Calculate average win/loss
            recent_trades = list(self.recent_trades)
            wins = [t.pnl for t in recent_trades if t.pnl > 0]
            losses = [t.pnl for t in recent_trades if t.pnl < 0]
            
            self.avg_win = np.mean(wins) if wins else 0.0
            self.avg_loss = abs(np.mean(losses)) if losses else 0.0
            
            # Calculate profit factor
            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 1
            self.profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
            
            # Update drawdown
            if self.current_capital > self.peak_capital:
                self.peak_capital = self.current_capital
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
                if self.current_drawdown > self.max_drawdown:
                    self.max_drawdown = self.current_drawdown
            
            # Calculate Sharpe ratio (simplified)
            if len(recent_trades) >= 10:
                daily_returns = [t.pnl / self.initial_capital for t in recent_trades]
                avg_return = np.mean(daily_returns)
                std_return = np.std(daily_returns)
                self.sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0
            
        except Exception as e:
            logger.error(f"❌ Performance metrics update error: {e}")
    
    def _save_trade_to_db(self, trade: PaperTrade):
        """Save trade to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO paper_trades VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.trade_id, trade.symbol, trade.side, trade.quantity,
                trade.entry_price, trade.exit_price, trade.entry_time.isoformat(),
                trade.exit_time.isoformat(), trade.pnl, trade.commission,
                trade.slippage, trade.hold_time_minutes, trade.exit_reason,
                trade.max_favorable_excursion, trade.max_adverse_excursion,
                "", 0.0  # position_size_method, risk_amount (optional)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"⚠️ Database save warning: {e}")
    
    def _save_position_to_db(self, position: PaperPosition):
        """Save current position to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO current_positions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position.symbol, position.side, position.quantity, position.entry_price,
                position.entry_time.isoformat(), position.stop_loss, position.profit_target,
                position.position_size_method, position.risk_amount
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"⚠️ Position save warning: {e}")
    
    def _remove_position_from_db(self, symbol: str):
        """Remove position from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM current_positions WHERE symbol = ?', (symbol,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"⚠️ Position removal warning: {e}")
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        try:
            # Calculate current portfolio value
            portfolio_value = self.available_capital
            total_unrealized_pnl = 0.0
            
            for position in self.positions.values():
                if position.current_price > 0:
                    portfolio_value += position.quantity * position.current_price
                    total_unrealized_pnl += position.unrealized_pnl
                else:
                    portfolio_value += position.quantity * position.entry_price
            
            # Calculate returns
            total_return = ((portfolio_value - self.initial_capital) / self.initial_capital) * 100
            
            return {
                'account_summary': {
                    'initial_capital': self.initial_capital,
                    'current_capital': self.current_capital,
                    'available_capital': self.available_capital,
                    'portfolio_value': round(portfolio_value, 2),
                    'total_return_pct': round(total_return, 2),
                    'total_unrealized_pnl': round(total_unrealized_pnl, 2)
                },
                'positions': {
                    'count': len(self.positions),
                    'total_value': round(sum(pos.quantity * pos.current_price for pos in self.positions.values()), 2),
                    'unrealized_pnl': round(total_unrealized_pnl, 2),
                    'details': [pos.to_dict() for pos in self.positions.values()]
                },
                'performance': {
                    'total_trades': self.total_trades,
                    'winning_trades': self.winning_trades,
                    'losing_trades': self.losing_trades,
                    'win_rate': round(self.win_rate, 1),
                    'avg_win': round(self.avg_win, 2),
                    'avg_loss': round(self.avg_loss, 2),
                    'profit_factor': round(self.profit_factor, 2),
                    'sharpe_ratio': round(self.sharpe_ratio, 2),
                    'max_drawdown': round(self.max_drawdown * 100, 2),
                    'current_drawdown': round(self.current_drawdown * 100, 2)
                },
                'risk_metrics': {
                    'position_count': len(self.positions),
                    'max_positions': self.max_positions,
                    'capital_utilization': round((1 - self.available_capital / portfolio_value) * 100, 1) if portfolio_value > 0 else 0,
                    'avg_position_size': round(portfolio_value / max(1, len(self.positions)), 2),
                    'total_risk_amount': round(sum(pos.risk_amount for pos in self.positions.values()), 2)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Portfolio summary error: {e}")
            return {'error': str(e)}
    
    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get recent trade history"""
        try:
            recent_trades = list(self.trade_history)[-limit:]
            return [trade.to_dict() for trade in recent_trades]
        except Exception as e:
            logger.error(f"❌ Trade history error: {e}")
            return []
    
    def get_position_details(self, symbol: str) -> Optional[Dict]:
        """Get detailed position information"""
        if symbol in self.positions:
            position = self.positions[symbol]
            
            # Add exit condition check
            exit_check = self.check_exit_conditions(symbol)
            
            details = position.to_dict()
            details['exit_conditions'] = exit_check
            details['days_held'] = (datetime.now() - position.entry_time).days
            details['hours_held'] = (datetime.now() - position.entry_time).total_seconds() / 3600
            
            return details
        return None
    
    def get_performance_analytics(self) -> Dict:
        """Get detailed performance analytics"""
        try:
            if not self.trade_history:
                return {'message': 'No trade history available'}
            
            trades = list(self.trade_history)
            
            # Monthly performance
            monthly_pnl = {}
            for trade in trades:
                month_key = trade.exit_time.strftime('%Y-%m')
                if month_key not in monthly_pnl:
                    monthly_pnl[month_key] = 0.0
                monthly_pnl[month_key] += trade.pnl
            
            # Best and worst trades
            best_trade = max(trades, key=lambda x: x.pnl)
            worst_trade = min(trades, key=lambda x: x.pnl)
            
            # Average hold time
            avg_hold_time = np.mean([trade.hold_time_minutes for trade in trades])
            
            # Win/loss streaks
            current_streak = 0
            max_win_streak = 0
            max_loss_streak = 0
            temp_win_streak = 0
            temp_loss_streak = 0
            
            for trade in trades:
                if trade.pnl > 0:
                    temp_win_streak += 1
                    temp_loss_streak = 0
                    max_win_streak = max(max_win_streak, temp_win_streak)
                else:
                    temp_loss_streak += 1
                    temp_win_streak = 0
                    max_loss_streak = max(max_loss_streak, temp_loss_streak)
            
            # Position sizing method performance
            method_performance = {}
            for trade in trades:
                method = getattr(trade, 'position_size_method', 'UNKNOWN')
                if method not in method_performance:
                    method_performance[method] = {'trades': 0, 'total_pnl': 0.0}
                method_performance[method]['trades'] += 1
                method_performance[method]['total_pnl'] += trade.pnl
            
            return {
                'summary': {
                    'total_trades': len(trades),
                    'avg_hold_time_minutes': round(avg_hold_time, 1),
                    'avg_hold_time_hours': round(avg_hold_time / 60, 1),
                    'best_trade_pnl': round(best_trade.pnl, 2),
                    'worst_trade_pnl': round(worst_trade.pnl, 2),
                    'max_win_streak': max_win_streak,
                    'max_loss_streak': max_loss_streak
                },
                'monthly_performance': {month: round(pnl, 2) for month, pnl in monthly_pnl.items()},
                'method_performance': {
                    method: {
                        'trades': data['trades'],
                        'total_pnl': round(data['total_pnl'], 2),
                        'avg_pnl': round(data['total_pnl'] / data['trades'], 2)
                    }
                    for method, data in method_performance.items()
                },
                'recent_trades': [trade.to_dict() for trade in trades[-10:]]
            }
            
        except Exception as e:
            logger.error(f"❌ Performance analytics error: {e}")
            return {'error': str(e)}
    
    def close_all_positions(self, reason: str = "CLOSE_ALL") -> Dict:
        """Close all open positions"""
        try:
            closed_positions = []
            failed_positions = []
            
            for symbol in list(self.positions.keys()):
                result = self.exit_position(symbol, reason)
                if result['success']:
                    closed_positions.append(symbol)
                else:
                    failed_positions.append(symbol)
            
            return {
                'success': True,
                'closed_positions': closed_positions,
                'failed_positions': failed_positions,
                'total_attempted': len(self.positions)
            }
            
        except Exception as e:
            logger.error(f"❌ Close all positions error: {e}")
            return {'success': False, 'error': str(e)}
    
    def reset_account(self, new_capital: float = None):
        """Reset paper trading account"""
        try:
            if new_capital:
                self.initial_capital = new_capital
            else:
                new_capital = self.initial_capital
            
            self.current_capital = new_capital
            self.available_capital = new_capital
            self.positions.clear()
            self.trade_history.clear()
            self.recent_trades.clear()
            
            # Reset metrics
            self.total_trades = 0
            self.winning_trades = 0
            self.losing_trades = 0
            self.win_rate = 0.0
            self.max_drawdown = 0.0
            self.current_drawdown = 0.0
            self.peak_capital = new_capital
            
            # Clear database
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('DELETE FROM paper_trades')
                cursor.execute('DELETE FROM current_positions')
                cursor.execute('DELETE FROM daily_performance')
                conn.commit()
                conn.close()
            except:
                pass
            
            logger.info(f"🔄 Paper trading account reset with ₹{new_capital:,.2f}")
            
        except Exception as e:
            logger.error(f"❌ Account reset error: {e}")
    
    def save_daily_performance(self):
        """Save daily performance metrics"""
        try:
            today = datetime.now().date()
            portfolio_summary = self.get_portfolio_summary()
            
            daily_pnl = portfolio_summary['account_summary']['total_unrealized_pnl']
            self.daily_pnl_history.append({
                'date': today,
                'pnl': daily_pnl,
                'capital': self.current_capital
            })
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO daily_performance VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                today.isoformat(),
                self.current_capital,
                daily_pnl,
                self.total_trades,
                self.win_rate,
                self.max_drawdown
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"⚠️ Daily performance save warning: {e}")
    
    def load_positions_from_db(self):
        """Load positions from database on startup"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM current_positions')
            rows = cursor.fetchall()
            
            for row in rows:
                symbol, side, quantity, entry_price, entry_time_str, stop_loss, profit_target, method, risk_amount = row
                
                position = PaperPosition(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    entry_price=entry_price,
                    entry_time=datetime.fromisoformat(entry_time_str),
                    stop_loss=stop_loss,
                    profit_target=profit_target,
                    position_size_method=method,
                    risk_amount=risk_amount
                )
                
                self.positions[symbol] = position
            
            conn.close()
            
            if self.positions:
                logger.info(f"📂 Loaded {len(self.positions)} positions from database")
            
        except Exception as e:
            logger.warning(f"⚠️ Position loading warning: {e}")
    
    def get_risk_metrics(self) -> Dict:
        """Get risk metrics for integration with risk manager"""
        try:
            total_risk = sum(pos.risk_amount for pos in self.positions.values())
            portfolio_risk = (total_risk / self.current_capital) * 100 if self.current_capital > 0 else 0
            
            return {
                'total_risk_amount': total_risk,
                'portfolio_risk_pct': portfolio_risk,
                'position_count': len(self.positions),
                'available_capital': self.available_capital,
                'capital_utilization_pct': ((self.current_capital - self.available_capital) / self.current_capital) * 100,
                'max_drawdown_pct': self.max_drawdown * 100,
                'current_drawdown_pct': self.current_drawdown * 100,
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor
            }
            
        except Exception as e:
            logger.error(f"❌ Risk metrics error: {e}")
            return {'error': str(e)}

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_paper_trading():
        """Test the enhanced paper trading engine"""
        
        # Initialize engine
        engine = EnhancedPaperTradingEngine(initial_capital=100000)
        
        # Test market data update
        engine.update_market_data('RELIANCE', 2500.0, 100000, 2499.0, 2501.0, 25.0)
        
        # Test position entry
        result = engine.enter_position(
            symbol='RELIANCE',
            side='BUY',
            entry_price=2500.0,
            stop_loss_pct=2.0,
            profit_target_pct=4.0,
            confidence=75.0,
            position_size_method='RISK_BASED'
        )
        
        print(f"Entry result: {result}")
        
        # Test portfolio summary
        portfolio = engine.get_portfolio_summary()
        print(f"Portfolio: {portfolio}")
        
        # Test price update and exit conditions
        engine.update_market_data('RELIANCE', 2450.0, 120000)  # Price drop
        exit_check = engine.check_exit_conditions('RELIANCE')
        print(f"Exit check: {exit_check}")
        
        # Test performance analytics
        analytics = engine.get_performance_analytics()
        print(f"Analytics: {analytics}")
    
    # Run test
    asyncio.run(test_paper_trading())
