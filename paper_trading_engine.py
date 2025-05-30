"""
Paper Trading Engine for Indian Market Trading Bot
Simulates realistic trading with all services connected except real order execution
Provides identical experience to live trading for testing and validation
"""

import logging
import asyncio
import random
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict, deque
import json
import os

from config_manager import get_config
from utils import (
    calculate_position_size, calculate_dynamic_stop_loss, calculate_portfolio_risk,
    format_currency, format_percentage
)
from strategy_manager import ScalpingSignal, SignalType, get_strategy_manager
from data_manager import get_data_manager
from goodwill_api_handler import get_goodwill_handler

class PaperOrderStatus(Enum):
    """Paper trading order status"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL_FILLED = "PARTIAL_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class PaperOrderType(Enum):
    """Paper trading order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LIMIT = "STOP_LIMIT"
    BRACKET = "BRACKET"

@dataclass
class PaperOrder:
    """Paper trading order structure"""
    order_id: str
    symbol: str
    side: str  # BUY/SELL
    order_type: PaperOrderType
    quantity: int
    price: float
    stop_price: float
    trigger_price: float
    time_in_force: str
    status: PaperOrderStatus
    filled_qty: int
    remaining_qty: int
    avg_price: float
    commission: float
    slippage: float
    timestamp: datetime
    fill_timestamp: Optional[datetime]
    parent_signal_id: Optional[str]
    exchange: str
    product: str
    notes: str

@dataclass
class PaperPosition:
    """Paper trading position structure"""
    symbol: str
    side: str  # LONG/SHORT/FLAT
    quantity: int
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    market_value: float
    cost_basis: float
    day_pnl: float
    timestamp: datetime
    entry_orders: List[str]
    exit_orders: List[str]
    stop_loss_price: float
    take_profit_prices: List[float]
    max_profit: float
    max_loss: float
    hold_time_seconds: int
    commission_paid: float

@dataclass
class PaperTrade:
    """Completed paper trade record"""
    trade_id: str
    symbol: str
    strategy: str
    entry_time: datetime
    exit_time: datetime
    side: str
    quantity: int
    entry_price: float
    exit_price: float
    gross_pnl: float
    commission: float
    slippage_cost: float
    net_pnl: float
    hold_time_seconds: int
    max_profit: float
    max_loss: float
    exit_reason: str
    success: bool
    confidence: float

class RealisticPaperTradingEngine:
    """
    Advanced paper trading engine that simulates realistic market conditions
    Connects with all bot services for authentic trading experience
    """
    
    def __init__(self):
        self.config = get_config()
        self.data_manager = get_data_manager()
        self.strategy_manager = get_strategy_manager()
        self.goodwill_handler = get_goodwill_handler()
        
        # Paper trading state
        self.is_enabled = self.config.trading.paper_trading
        self.starting_capital = 100000.0  # ₹1 Lakh starting capital
        self.current_capital = self.starting_capital
        self.available_cash = self.starting_capital
        self.buying_power = self.starting_capital * 2.0  # 2x margin for intraday
        
        # Order and position tracking
        self.pending_orders: Dict[str, PaperOrder] = {}
        self.filled_orders: Dict[str, PaperOrder] = {}
        self.active_positions: Dict[str, PaperPosition] = {}
        self.completed_trades: List[PaperTrade] = []
        
        # Realistic simulation parameters
        self.simulation_config = {
            'market_slippage': 0.001,      # 0.1% market order slippage
            'limit_slippage': 0.0002,      # 0.02% limit order slippage
            'fill_probability': 0.95,      # 95% fill rate for market orders
            'partial_fill_probability': 0.1,  # 10% chance of partial fills
            'rejection_probability': 0.02,   # 2% order rejection rate
            'latency_simulation': True,     # Simulate order latency
            'min_latency_ms': 50,          # Minimum order latency
            'max_latency_ms': 300,         # Maximum order latency
            'weekend_trading': False,       # No weekend trading
            'after_hours_spreads': 2.0     # 2x spreads in after hours
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'total_commission': 0.0,
            'total_slippage': 0.0,
            'max_drawdown': 0.0,
            'max_portfolio_value': self.starting_capital,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'avg_trade_duration': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }
        
        # Order ID counter
        self.order_counter = 0
        self.trade_counter = 0
        
        # Background processing
        self.processing_enabled = True
        self.processing_task: Optional[asyncio.Task] = None
        
        logging.info("Paper Trading Engine initialized with realistic simulation")
    
    async def start_engine(self):
        """Start the paper trading engine"""
        try:
            if not self.is_enabled:
                logging.warning("Paper trading is disabled in config")
                return
            
            # Load existing paper trading state if available
            await self._load_state()
            
            # Start background processing
            self.processing_task = asyncio.create_task(self._background_processing())
            
            # Connect to all services (same as live trading)
            await self._connect_to_services()
            
            logging.info("✅ Paper Trading Engine started successfully")
            logging.info(f"💰 Starting Capital: ₹{self.starting_capital:,.2f}")
            logging.info(f"💵 Available Cash: ₹{self.available_cash:,.2f}")
            logging.info(f"⚡ Buying Power: ₹{self.buying_power:,.2f}")
            
        except Exception as e:
            logging.error(f"Error starting paper trading engine: {e}")
            raise
    
    async def _connect_to_services(self):
        """Connect to all bot services (same as live trading)"""
        try:
            # Initialize data manager for real market data
            if not self.data_manager.session:
                await self.data_manager.initialize()
            
            # Connect to Goodwill API for market data (not for orders)
            if not self.goodwill_handler.is_authenticated:
                logging.info("📡 Paper trading will use simulated API responses")
                # We can still get real market data even in paper trading mode
            
            logging.info("🔗 All services connected for paper trading")
            
        except Exception as e:
            logging.error(f"Error connecting to services: {e}")
    
    async def place_paper_order(self, signal: ScalpingSignal) -> Optional[str]:
        """
        Place a paper order based on trading signal
        Simulates the complete order lifecycle with realistic behavior
        """
        try:
            # Generate unique order ID
            self.order_counter += 1
            order_id = f"PAPER_{signal.symbol}_{int(time.time())}_{self.order_counter}"
            
            # Determine order side
            side = "BUY" if signal.signal_type == SignalType.BUY else "SELL"
            
            # Get current market data for realistic pricing
            latest_tick = self.data_manager.get_latest_tick(signal.symbol)
            if not latest_tick:
                logging.warning(f"No market data available for {signal.symbol}, rejecting order")
                return None
            
            # Simulate pre-order validations (same as live trading)
            if not await self._validate_paper_order(signal, latest_tick):
                return None
            
            # Create paper order
            paper_order = PaperOrder(
                order_id=order_id,
                symbol=signal.symbol,
                side=side,
                order_type=PaperOrderType.MARKET,  # Start with market orders
                quantity=signal.position_size,
                price=signal.entry_price,
                stop_price=0.0,
                trigger_price=0.0,
                time_in_force="IOC",
                status=PaperOrderStatus.PENDING,
                filled_qty=0,
                remaining_qty=signal.position_size,
                avg_price=0.0,
                commission=0.0,
                slippage=0.0,
                timestamp=datetime.now(),
                fill_timestamp=None,
                parent_signal_id=id(signal),
                exchange="NSE",  # Default to NSE
                product="MIS",   # Intraday for scalping
                notes=f"Strategy: {signal.strategy.value}, Confidence: {signal.confidence:.2f}"
            )
            
            # Add to pending orders
            self.pending_orders[order_id] = paper_order
            
            # Simulate order submission latency
            if self.simulation_config['latency_simulation']:
                latency = random.uniform(
                    self.simulation_config['min_latency_ms'],
                    self.simulation_config['max_latency_ms']
                ) / 1000  # Convert to seconds
                await asyncio.sleep(latency)
            
            # Change status to submitted
            paper_order.status = PaperOrderStatus.SUBMITTED
            
            # Schedule order processing
            asyncio.create_task(self._process_order(order_id))
            
            logging.info(f"📝 Paper order placed: {order_id} for {signal.symbol}")
            logging.info(f"   {side} {signal.position_size} shares at market price")
            logging.info(f"   Strategy: {signal.strategy.value} (Confidence: {signal.confidence:.2f})")
            
            return order_id
            
        except Exception as e:
            logging.error(f"Error placing paper order: {e}")
            return None
    
    async def _validate_paper_order(self, signal: ScalpingSignal, latest_tick) -> bool:
        """Validate paper order (same validations as live trading)"""
        try:
            # Check market hours
            if not self.config.is_trading_hours():
                logging.warning(f"Market is closed, rejecting order for {signal.symbol}")
                return False
            
            # Check available cash
            order_value = signal.entry_price * signal.position_size
            if order_value > self.available_cash:
                logging.warning(f"Insufficient cash for {signal.symbol}: need ₹{order_value:.2f}, have ₹{self.available_cash:.2f}")
                return False
            
            # Check position limits
            if len(self.active_positions) >= self.config.risk.max_positions:
                logging.warning(f"Maximum positions limit reached: {len(self.active_positions)}")
                return False
            
            # Check if symbol already has position
            if signal.symbol in self.active_positions:
                logging.warning(f"Already have position in {signal.symbol}")
                return False
            
            # Check portfolio risk
            portfolio_risk = calculate_portfolio_risk(self.active_positions, self.current_capital)
            if portfolio_risk['risk_percentage'] > self.config.risk.max_portfolio_risk * 100:
                logging.warning(f"Portfolio risk too high: {portfolio_risk['risk_percentage']:.1f}%")
                return False
            
            # Simulate random rejections (market conditions, etc.)
            if random.random() < self.simulation_config['rejection_probability']:
                rejection_reasons = [
                    "Insufficient liquidity",
                    "Price band hit",
                    "Circuit breaker triggered",
                    "Exchange connectivity issue"
                ]
                reason = random.choice(rejection_reasons)
                logging.warning(f"Order rejected: {reason}")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating paper order: {e}")
            return False
    
    async def _process_order(self, order_id: str):
        """Process paper order with realistic market simulation"""
        try:
            order = self.pending_orders.get(order_id)
            if not order:
                return
            
            # Simulate processing delay
            processing_delay = random.uniform(0.1, 0.5)  # 100-500ms
            await asyncio.sleep(processing_delay)
            
            # Get current market data
            latest_tick = self.data_manager.get_latest_tick(order.symbol)
            if not latest_tick:
                # Reject order if no market data
                order.status = PaperOrderStatus.REJECTED
                order.notes += " | Rejected: No market data"
                del self.pending_orders[order_id]
                return
            
            # Determine fill probability
            fill_prob = self.simulation_config['fill_probability']
            
            # Reduce fill probability during high volatility
            latest_l1 = self.data_manager.get_latest_l1(order.symbol)
            if latest_l1 and latest_l1.spread_percent > 0.5:  # High spread
                fill_prob *= 0.8
            
            # Check if order gets filled
            if random.random() > fill_prob:
                # Order rejected
                order.status = PaperOrderStatus.REJECTED
                rejection_reasons = [
                    "Connectivity issue",
                    "Exchange rejection",
                    "Risk management rejection"
                ]
                order.notes += f" | Rejected: {random.choice(rejection_reasons)}"
                del self.pending_orders[order_id]
                logging.warning(f"Paper order rejected: {order_id}")
                return
            
            # Simulate order fill
            await self._fill_paper_order(order_id, latest_tick)
            
        except Exception as e:
            logging.error(f"Error processing paper order {order_id}: {e}")
    
    async def _fill_paper_order(self, order_id: str, latest_tick):
        """Fill paper order with realistic slippage and pricing"""
        try:
            order = self.pending_orders[order_id]
            
            # Calculate realistic fill price with slippage
            fill_price = latest_tick.last_price
            
            # Apply slippage based on order type and market conditions
            if order.order_type == PaperOrderType.MARKET:
                slippage_pct = self.simulation_config['market_slippage']
                
                # Increase slippage during high volatility or low liquidity
                latest_l1 = self.data_manager.get_latest_l1(order.symbol)
                if latest_l1:
                    if latest_l1.spread_percent > 0.3:
                        slippage_pct *= 2  # Double slippage for wide spreads
                
                # Apply slippage
                if order.side == "BUY":
                    fill_price *= (1 + slippage_pct)
                else:
                    fill_price *= (1 - slippage_pct)
                
                order.slippage = abs(fill_price - latest_tick.last_price)
            
            # Calculate commission (realistic Indian brokerage)
            order.commission = self._calculate_realistic_commission(order.quantity, fill_price)
            
            # Check for partial fills
            if random.random() < self.simulation_config['partial_fill_probability']:
                # Partial fill (70-90% of quantity)
                fill_percentage = random.uniform(0.7, 0.9)
                filled_qty = int(order.quantity * fill_percentage)
                order.filled_qty = filled_qty
                order.remaining_qty = order.quantity - filled_qty
                order.status = PaperOrderStatus.PARTIAL_FILLED
            else:
                # Complete fill
                order.filled_qty = order.quantity
                order.remaining_qty = 0
                order.status = PaperOrderStatus.FILLED
            
            order.avg_price = fill_price
            order.fill_timestamp = datetime.now()
            
            # Move to filled orders
            self.filled_orders[order_id] = order
            if order.status == PaperOrderStatus.FILLED:
                del self.pending_orders[order_id]
            
            # Update position
            await self._update_paper_position(order)
            
            # Update available cash
            if order.side == "BUY":
                cost = (order.avg_price * order.filled_qty) + order.commission
                self.available_cash -= cost
            else:
                proceeds = (order.avg_price * order.filled_qty) - order.commission
                self.available_cash += proceeds
            
            # Update buying power
            self.buying_power = self.available_cash + (
                sum(pos.market_value for pos in self.active_positions.values()) * 0.5
            )
            
            logging.info(f"✅ Paper order filled: {order_id}")
            logging.info(f"   {order.side} {order.filled_qty}/{order.quantity} shares of {order.symbol}")
            logging.info(f"   Fill Price: ₹{order.avg_price:.2f} (Slippage: ₹{order.slippage:.3f})")
            logging.info(f"   Commission: ₹{order.commission:.2f}")
            logging.info(f"   Available Cash: ₹{self.available_cash:.2f}")
            
        except Exception as e:
            logging.error(f"Error filling paper order {order_id}: {e}")
    
    def _calculate_realistic_commission(self, quantity: int, price: float) -> float:
        """Calculate realistic commission based on Indian brokerage structure"""
        try:
            order_value = quantity * price
            
            # Discount broker structure (like Zerodha, Upstox)
            brokerage = min(20.0, order_value * 0.0003)  # ₹20 or 0.03% whichever is lower
            
            # STT (Securities Transaction Tax)
            stt = order_value * 0.001  # 0.1% for delivery, 0.025% for intraday
            
            # Exchange charges
            exchange_charges = order_value * 0.0000345  # NSE charges
            
            # SEBI charges
            sebi_charges = order_value * 0.000001  # ₹1 per crore
            
            # GST on brokerage
            gst = brokerage * 0.18
            
            # Stamp duty
            stamp_duty = order_value * 0.00003  # 0.003%
            
            total_commission = brokerage + stt + exchange_charges + sebi_charges + gst + stamp_duty
            
            return round(total_commission, 2)
            
        except Exception as e:
            logging.error(f"Error calculating commission: {e}")
            return 20.0  # Default flat brokerage
    
    async def _update_paper_position(self, filled_order: PaperOrder):
        """Update paper position after order fill"""
        try:
            symbol = filled_order.symbol
            
            if symbol not in self.active_positions:
                # Create new position
                side = "LONG" if filled_order.side == "BUY" else "SHORT"
                qty = filled_order.filled_qty if side == "LONG" else -filled_order.filled_qty
                
                position = PaperPosition(
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    avg_price=filled_order.avg_price,
                    current_price=filled_order.avg_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    total_pnl=0.0,
                    market_value=filled_order.avg_price * abs(qty),
                    cost_basis=filled_order.avg_price * abs(qty),
                    day_pnl=0.0,
                    timestamp=datetime.now(),
                    entry_orders=[filled_order.order_id],
                    exit_orders=[],
                    stop_loss_price=0.0,
                    take_profit_prices=[],
                    max_profit=0.0,
                    max_loss=0.0,
                    hold_time_seconds=0,
                    commission_paid=filled_order.commission
                )
                
                self.active_positions[symbol] = position
                
                logging.info(f"📈 New paper position: {symbol} {side} {abs(qty)} shares @ ₹{filled_order.avg_price:.2f}")
                
            else:
                # Update existing position (averaging, partial exits, etc.)
                position = self.active_positions[symbol]
                
                if filled_order.side == "BUY":
                    if position.side == "LONG" or position.side == "FLAT":
                        # Adding to long position
                        total_cost = (abs(position.quantity) * position.avg_price) + (filled_order.filled_qty * filled_order.avg_price)
                        total_qty = abs(position.quantity) + filled_order.filled_qty
                        position.avg_price = total_cost / total_qty
                        position.quantity = total_qty
                        position.side = "LONG"
                    else:  # SHORT position, covering
                        position.quantity += filled_order.filled_qty
                        if position.quantity >= 0:
                            position.side = "LONG" if position.quantity > 0 else "FLAT"
                else:  # SELL
                    if position.side == "SHORT" or position.side == "FLAT":
                        # Adding to short position
                        total_cost = (abs(position.quantity) * position.avg_price) + (filled_order.filled_qty * filled_order.avg_price)
                        total_qty = abs(position.quantity) + filled_order.filled_qty
                        position.avg_price = total_cost / total_qty
                        position.quantity = -total_qty
                        position.side = "SHORT"
                    else:  # LONG position, selling
                        position.quantity -= filled_order.filled_qty
                        if position.quantity <= 0:
                            position.side = "SHORT" if position.quantity < 0 else "FLAT"
                
                position.market_value = abs(position.quantity) * position.current_price
                position.commission_paid += filled_order.commission
                position.entry_orders.append(filled_order.order_id)
                
                # Remove flat positions
                if position.side == "FLAT" or position.quantity == 0:
                    del self.active_positions[symbol]
                    logging.info(f"📉 Position closed: {symbol}")
            
        except Exception as e:
            logging.error(f"Error updating paper position: {e}")
    
    async def _background_processing(self):
        """Background processing for paper trading engine"""
        while self.processing_enabled:
            try:
                # Update positions with real-time prices
                await self._update_positions_realtime()
                
                # Check exit conditions
                await self._check_paper_exit_conditions()
                
                # Process pending limit orders
                await self._process_pending_limit_orders()
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Save state periodically
                await self._save_state()
                
                await asyncio.sleep(1)  # Process every second
                
            except Exception as e:
                logging.error(f"Error in paper trading background processing: {e}")
                await asyncio.sleep(5)
    
    async def _update_positions_realtime(self):
        """Update all positions with real-time market prices"""
        try:
            for symbol, position in self.active_positions.items():
                latest_tick = self.data_manager.get_latest_tick(symbol)
                if latest_tick:
                    position.current_price = latest_tick.last_price
                    
                    # Calculate unrealized P&L
                    if position.side == "LONG":
                        position.unrealized_pnl = (position.current_price - position.avg_price) * position.quantity
                    elif position.side == "SHORT":
                        position.unrealized_pnl = (position.avg_price - position.current_price) * abs(position.quantity)
                    else:
                        position.unrealized_pnl = 0.0
                    
                    position.total_pnl = position.realized_pnl + position.unrealized_pnl - position.commission_paid
                    position.market_value = abs(position.quantity) * position.current_price
                    
                    # Track max profit/loss
                    if position.unrealized_pnl > position.max_profit:
                        position.max_profit = position.unrealized_pnl
                    if position.unrealized_pnl < position.max_loss:
                        position.max_loss = position.unrealized_pnl
                    
                    # Update hold time
                    position.hold_time_seconds = int((datetime.now() - position.timestamp).total_seconds())
            
            # Update portfolio value
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.active_positions.values())
            total_realized_pnl = sum(pos.realized_pnl for pos in self.active_positions.values())
            total_commission = sum(pos.commission_paid for pos in self.active_positions.values())
            
            self.current_capital = self.available_cash + sum(pos.market_value for pos in self.active_positions.values())
            
            # Track maximum portfolio value for drawdown calculation
            if self.current_capital > self.performance_metrics['max_portfolio_value']:
                self.performance_metrics['max_portfolio_value'] = self.current_capital
            
        except Exception as e:
            logging.error(f"Error updating positions real-time: {e}")
    
    async def _check_paper_exit_conditions(self):
        """Check exit conditions for paper positions (same logic as live trading)"""
        try:
            for symbol, position in list(self.active_positions.items()):
                latest_tick = self.data_manager.get_latest_tick(symbol)
                if not latest_tick:
                    continue
                
                should_exit = False
                exit_reason = ""
                
                # Time-based exit (max 10 minutes for scalping)
                if position.hold_time_seconds > 600:
                    should_exit = True
                    exit_reason = "TIME_STOP"
                
                # Stop-loss exit
                elif position.stop_loss_price > 0:
                    if ((position.side == "LONG" and latest_tick.last_price <= position.stop_loss_price) or
                        (position.side == "SHORT" and latest_tick.last_price >= position.stop_loss_price)):
                        should_exit = True
                        exit_reason = "STOP_LOSS"
                
                # Take-profit exit
                elif position.take_profit_prices:
                    if ((position.side == "LONG" and latest_tick.last_price >= position.take_profit_prices[0]) or
                        (position.side == "SHORT" and latest_tick.last_price <= position.take_profit_prices[0])):
                        should_exit = True
                        exit_reason = "TAKE_PROFIT"
                
                # Trailing stop (if profit > 2%)
                elif position.max_profit > position.cost_basis * 0.02:
                    trailing_stop_pct = 0.015  # 1.5% trailing stop
                    if position.side == "LONG":
                        trailing_stop_price = position.current_price * (1 - trailing_stop_pct)
                        if latest_tick.last_price <= trailing_stop_price:
                            should_exit = True
                            exit_reason = "TRAILING_STOP"
                    else:  # SHORT
                        trailing_stop_price = position.current_price * (1 + trailing_stop_pct)
                        if latest_tick.last_price >= trailing_stop_price:
                            should_exit = True
                            exit_reason = "TRAILING_STOP"
                
                if should_exit:
                    await self._create_paper_exit_order(symbol, exit_reason)
                    
        except Exception as e:
            logging.error(f"Error checking paper exit conditions: {e}")
    
    async def _create_paper_exit_order(self, symbol: str, exit_reason: str):
        """Create paper exit order"""
        try:
            position = self.active_positions.get(symbol)
            if not position:
                return
            
            # Create exit order
            self.order_counter += 1
            order_id = f"PAPER_EXIT_{symbol}_{int(time.time())}_{self.order_counter}"
            
            side = "SELL" if position.side == "LONG" else "BUY"
            
            latest_tick = self.data_manager.get_latest_tick(symbol)
            exit_price = latest_tick.last_price if latest_tick else position.current_price
            
            # Create and immediately fill exit order (market order simulation)
            exit_order = PaperOrder(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=PaperOrderType.MARKET,
                quantity=abs(position.quantity),
                price=exit_price,
                stop_price=0.0,
                trigger_price=0.0,
                time_in_force="IOC",
                status=PaperOrderStatus.FILLED,
                filled_qty=abs(position.quantity),
                remaining_qty=0,
                avg_price=exit_price,
                commission=self._calculate_realistic_commission(abs(position.quantity), exit_price),
                slippage=exit_price * self.simulation_config['market_slippage'],
                timestamp=datetime.now(),
                fill_timestamp=datetime.now(),
                parent_signal_id=None,
                exchange="NSE",
                product="MIS",
                notes=f"Exit: {exit_reason}"
            )
            
            self.filled_orders[order_id] = exit_order
            
            # Calculate final P&L
            if position.side == "LONG":
                gross_pnl = (exit_price - position.avg_price) * position.quantity
            else:  # SHORT
                gross_pnl = (position.avg_price - exit_price) * abs(position.quantity)
            
            net_pnl = gross_pnl - position.commission_paid - exit_order.commission
            
            # Create completed trade record
            self.trade_counter += 1
            trade_record = PaperTrade(
                trade_id=f"TRADE_{self.trade_counter:06d}",
                symbol=symbol,
                strategy="Unknown",  # Would need to track from original signal
                entry_time=position.timestamp,
                exit_time=datetime.now(),
                side=position.side,
                quantity=abs(position.quantity),
                entry_price=position.avg_price,
                exit_price=exit_price,
                gross_pnl=gross_pnl,
                commission=position.commission_paid + exit_order.commission,
                slippage_cost=exit_order.slippage,
                net_pnl=net_pnl,
                hold_time_seconds=position.hold_time_seconds,
                max_profit=position.max_profit,
                max_loss=position.max_loss,
                exit_reason=exit_reason,
                success=net_pnl > 0,
                confidence=0.0  # Would need to track from original signal
            )
            
            self.completed_trades.append(trade_record)
            
            # Update available cash
            if exit_order.side == "SELL":
                proceeds = (exit_price * abs(position.quantity)) - exit_order.commission
                self.available_cash += proceeds
            else:  # Covering short
                cost = (exit_price * abs(position.quantity)) + exit_order.commission
                self.available_cash -= cost
            
            # Remove position
            del self.active_positions[symbol]
            
            logging.info(f"🔚 Paper position closed: {symbol}")
            logging.info(f"   Exit Reason: {exit_reason}")
            logging.info(f"   Entry: ₹{position.avg_price:.2f} | Exit: ₹{exit_price:.2f}")
            logging.info(f"   P&L: ₹{net_pnl:.2f} | Hold Time: {position.hold_time_seconds//60}m {position.hold_time_seconds%60}s")
            
        except Exception as e:
            logging.error(f"Error creating paper exit order for {symbol}: {e}")
    
    async def _process_pending_limit_orders(self):
        """Process pending limit orders to check if they should be filled"""
        try:
            for order_id, order in list(self.pending_orders.items()):
                if order.order_type != PaperOrderType.LIMIT:
                    continue
                
                latest_tick = self.data_manager.get_latest_tick(order.symbol)
                if not latest_tick:
                    continue
                
                should_fill = False
                
                # Check if limit order should be filled
                if order.side == "BUY" and latest_tick.last_price <= order.price:
                    should_fill = True
                elif order.side == "SELL" and latest_tick.last_price >= order.price:
                    should_fill = True
                
                if should_fill:
                    await self._fill_paper_order(order_id, latest_tick)
                    
        except Exception as e:
            logging.error(f"Error processing pending limit orders: {e}")
    
    async def _update_performance_metrics(self):
        """Update comprehensive performance metrics"""
        try:
            if not self.completed_trades:
                return
            
            # Basic trade statistics
            total_trades = len(self.completed_trades)
            winning_trades = len([t for t in self.completed_trades if t.success])
            losing_trades = total_trades - winning_trades
            
            total_pnl = sum(t.net_pnl for t in self.completed_trades)
            total_commission = sum(t.commission for t in self.completed_trades)
            total_slippage = sum(t.slippage_cost for t in self.completed_trades)
            
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Profit/Loss analysis
            winning_pnls = [t.net_pnl for t in self.completed_trades if t.success]
            losing_pnls = [t.net_pnl for t in self.completed_trades if not t.success]
            
            largest_win = max(winning_pnls) if winning_pnls else 0
            largest_loss = min(losing_pnls) if losing_pnls else 0
            
            # Trade duration analysis
            avg_duration = sum(t.hold_time_seconds for t in self.completed_trades) / total_trades if total_trades > 0 else 0
            
            # Drawdown calculation
            portfolio_values = [self.starting_capital]
            running_pnl = 0
            for trade in self.completed_trades:
                running_pnl += trade.net_pnl
                portfolio_values.append(self.starting_capital + running_pnl)
            
            max_portfolio = max(portfolio_values)
            current_portfolio = portfolio_values[-1]
            max_drawdown = ((max_portfolio - min(portfolio_values[portfolio_values.index(max_portfolio):])) / max_portfolio) * 100 if max_portfolio > 0 else 0
            
            # Sharpe ratio calculation (simplified)
            if len(self.completed_trades) > 1:
                returns = [t.net_pnl / self.starting_capital for t in self.completed_trades]
                import numpy as np
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = (avg_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Update metrics
            self.performance_metrics.update({
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'total_pnl': total_pnl,
                'total_commission': total_commission,
                'total_slippage': total_slippage,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_trade_duration': avg_duration,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'sharpe_ratio': sharpe_ratio,
                'current_portfolio_value': self.current_capital,
                'total_return_pct': ((self.current_capital - self.starting_capital) / self.starting_capital) * 100
            })
            
        except Exception as e:
            logging.error(f"Error updating performance metrics: {e}")
    
    async def _save_state(self):
        """Save paper trading state to JSON file"""
        try:
            state_data = {
                'starting_capital': self.starting_capital,
                'current_capital': self.current_capital,
                'available_cash': self.available_cash,
                'buying_power': self.buying_power,
                'pending_orders': {oid: asdict(order) for oid, order in self.pending_orders.items()},
                'filled_orders': {oid: asdict(order) for oid, order in self.filled_orders.items()},
                'active_positions': {symbol: asdict(pos) for symbol, pos in self.active_positions.items()},
                'completed_trades': [asdict(trade) for trade in self.completed_trades],
                'performance_metrics': self.performance_metrics,
                'order_counter': self.order_counter,
                'trade_counter': self.trade_counter,
                'last_saved': datetime.now().isoformat()
            }
            
            # Create state directory
            os.makedirs("paper_trading", exist_ok=True)
            
            # Save to JSON file
            state_file = os.path.join("paper_trading", "paper_trading_state.json")
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
                
        except Exception as e:
            logging.error(f"Error saving paper trading state: {e}")
    
    async def _load_state(self):
        """Load paper trading state from JSON file"""
        try:
            state_file = os.path.join("paper_trading", "paper_trading_state.json")
            
            if not os.path.exists(state_file):
                logging.info("No existing paper trading state found, starting fresh")
                return
            
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            # Restore basic state
            self.starting_capital = state_data.get('starting_capital', 100000.0)
            self.current_capital = state_data.get('current_capital', self.starting_capital)
            self.available_cash = state_data.get('available_cash', self.starting_capital)
            self.buying_power = state_data.get('buying_power', self.starting_capital * 2)
            self.order_counter = state_data.get('order_counter', 0)
            self.trade_counter = state_data.get('trade_counter', 0)
            
            # Restore performance metrics
            self.performance_metrics.update(state_data.get('performance_metrics', {}))
            
            # Restore completed trades
            trades_data = state_data.get('completed_trades', [])
            self.completed_trades = []
            for trade_dict in trades_data:
                trade_dict['entry_time'] = datetime.fromisoformat(trade_dict['entry_time'])
                trade_dict['exit_time'] = datetime.fromisoformat(trade_dict['exit_time'])
                self.completed_trades.append(PaperTrade(**trade_dict))
            
            # Note: We don't restore pending orders and active positions as they may be stale
            # The engine will start fresh for new trading session
            
            logging.info(f"📂 Paper trading state loaded from {state_file}")
            logging.info(f"💰 Portfolio Value: ₹{self.current_capital:,.2f}")
            logging.info(f"📊 Total Trades: {len(self.completed_trades)}")
            
        except Exception as e:
            logging.error(f"Error loading paper trading state: {e}")
    
    # Public API methods
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.active_positions.values())
            total_market_value = sum(pos.market_value for pos in self.active_positions.values())
            
            return {
                'timestamp': datetime.now().isoformat(),
                'starting_capital': self.starting_capital,
                'current_capital': self.current_capital,
                'available_cash': self.available_cash,
                'buying_power': self.buying_power,
                'total_market_value': total_market_value,
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_return': self.current_capital - self.starting_capital,
                'total_return_pct': ((self.current_capital - self.starting_capital) / self.starting_capital) * 100,
                'active_positions': len(self.active_positions),
                'pending_orders': len(self.pending_orders),
                'total_trades': len(self.completed_trades),
                'performance_metrics': self.performance_metrics.copy(),
                'is_paper_trading': True
            }
            
        except Exception as e:
            logging.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def get_active_positions(self) -> List[Dict[str, Any]]:
        """Get all active positions"""
        try:
            positions = []
            for symbol, position in self.active_positions.items():
                latest_tick = self.data_manager.get_latest_tick(symbol)
                
                pos_dict = asdict(position)
                pos_dict.update({
                    'timestamp': position.timestamp.isoformat(),
                    'unrealized_pnl_pct': (position.unrealized_pnl / position.cost_basis) * 100 if position.cost_basis > 0 else 0,
                    'current_bid': latest_tick.bid_price if latest_tick else 0,
                    'current_ask': latest_tick.ask_price if latest_tick else 0,
                    'last_trade_time': latest_tick.timestamp.isoformat() if latest_tick else None
                })
                positions.append(pos_dict)
            
            return positions
            
        except Exception as e:
            logging.error(f"Error getting active positions: {e}")
            return []
    
    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade history"""
        try:
            trades = sorted(self.completed_trades, key=lambda x: x.exit_time, reverse=True)
            limited_trades = trades[:limit]
            
            trade_history = []
            for trade in limited_trades:
                trade_dict = asdict(trade)
                trade_dict['entry_time'] = trade.entry_time.isoformat()
                trade_dict['exit_time'] = trade.exit_time.isoformat()
                trade_dict['return_pct'] = (trade.net_pnl / (trade.entry_price * trade.quantity)) * 100
                trade_history.append(trade_dict)
            
            return trade_history
            
        except Exception as e:
            logging.error(f"Error getting trade history: {e}")
            return []
    
    def get_order_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get order history"""
        try:
            all_orders = list(self.filled_orders.values()) + list(self.pending_orders.values())
            sorted_orders = sorted(all_orders, key=lambda x: x.timestamp, reverse=True)
            limited_orders = sorted_orders[:limit]
            
            order_history = []
            for order in limited_orders:
                order_dict = asdict(order)
                order_dict['timestamp'] = order.timestamp.isoformat()
                order_dict['fill_timestamp'] = order.fill_timestamp.isoformat() if order.fill_timestamp else None
                order_history.append(order_dict)
            
            return order_history
            
        except Exception as e:
            logging.error(f"Error getting order history: {e}")
            return []
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        try:
            # Calculate additional metrics
            if self.completed_trades:
                profit_factor = (
                    sum(t.net_pnl for t in self.completed_trades if t.success) /
                    abs(sum(t.net_pnl for t in self.completed_trades if not t.success))
                    if any(not t.success for t in self.completed_trades) else float('inf')
                )
                
                avg_win = (
                    sum(t.net_pnl for t in self.completed_trades if t.success) /
                    len([t for t in self.completed_trades if t.success])
                    if any(t.success for t in self.completed_trades) else 0
                )
                
                avg_loss = (
                    sum(t.net_pnl for t in self.completed_trades if not t.success) /
                    len([t for t in self.completed_trades if not t.success])
                    if any(not t.success for t in self.completed_trades) else 0
                )
                
                # Strategy performance breakdown
                strategy_performance = {}
                for trade in self.completed_trades:
                    strategy = trade.strategy
                    if strategy not in strategy_performance:
                        strategy_performance[strategy] = {
                            'trades': 0, 'wins': 0, 'total_pnl': 0.0
                        }
                    
                    strategy_performance[strategy]['trades'] += 1
                    if trade.success:
                        strategy_performance[strategy]['wins'] += 1
                    strategy_performance[strategy]['total_pnl'] += trade.net_pnl
                
                # Calculate win rates for each strategy
                for strategy, stats in strategy_performance.items():
                    stats['win_rate'] = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
            
            else:
                profit_factor = 0
                avg_win = 0
                avg_loss = 0
                strategy_performance = {}
            
            return {
                'paper_trading_summary': {
                    'starting_capital': self.starting_capital,
                    'current_capital': self.current_capital,
                    'total_return': self.current_capital - self.starting_capital,
                    'total_return_pct': ((self.current_capital - self.starting_capital) / self.starting_capital) * 100,
                    'trading_period_days': (datetime.now() - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).days + 1
                },
                'trade_statistics': {
                    **self.performance_metrics,
                    'profit_factor': profit_factor,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'expectancy': avg_win * (self.performance_metrics['win_rate'] / 100) + avg_loss * ((100 - self.performance_metrics['win_rate']) / 100)
                },
                'strategy_breakdown': strategy_performance,
                'cost_analysis': {
                    'total_commission': self.performance_metrics['total_commission'],
                    'total_slippage': self.performance_metrics['total_slippage'],
                    'commission_per_trade': self.performance_metrics['total_commission'] / max(1, self.performance_metrics['total_trades']),
                    'slippage_per_trade': self.performance_metrics['total_slippage'] / max(1, self.performance_metrics['total_trades'])
                },
                'risk_metrics': {
                    'max_drawdown_pct': self.performance_metrics['max_drawdown'],
                    'sharpe_ratio': self.performance_metrics['sharpe_ratio'],
                    'largest_win': self.performance_metrics['largest_win'],
                    'largest_loss': self.performance_metrics['largest_loss'],
                    'volatility': self._calculate_return_volatility()
                }
            }
            
        except Exception as e:
            logging.error(f"Error getting performance report: {e}")
            return {}
    
    def _calculate_return_volatility(self) -> float:
        """Calculate return volatility"""
        try:
            if len(self.completed_trades) < 2:
                return 0.0
            
            returns = [(t.net_pnl / self.starting_capital) for t in self.completed_trades]
            import numpy as np
            return float(np.std(returns) * np.sqrt(252))  # Annualized volatility
            
        except Exception as e:
            logging.error(f"Error calculating return volatility: {e}")
            return 0.0
    
    async def reset_paper_trading(self):
        """Reset paper trading to initial state"""
        try:
            self.current_capital = self.starting_capital
            self.available_cash = self.starting_capital
            self.buying_power = self.starting_capital * 2.0
            
            self.pending_orders.clear()
            self.filled_orders.clear()
            self.active_positions.clear()
            self.completed_trades.clear()
            
            self.order_counter = 0
            self.trade_counter = 0
            
            # Reset performance metrics
            self.performance_metrics = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'total_commission': 0.0,
                'total_slippage': 0.0,
                'max_drawdown': 0.0,
                'max_portfolio_value': self.starting_capital,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'avg_trade_duration': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
            
            # Save reset state
            await self._save_state()
            
            logging.info("🔄 Paper trading reset to initial state")
            logging.info(f"💰 Starting Capital: ₹{self.starting_capital:,.2f}")
            
        except Exception as e:
            logging.error(f"Error resetting paper trading: {e}")
    
    async def stop_engine(self):
        """Stop the paper trading engine"""
        try:
            self.processing_enabled = False
            
            if self.processing_task and not self.processing_task.done():
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
            
            # Save final state
            await self._save_state()
            
            logging.info("🛑 Paper Trading Engine stopped")
            
        except Exception as e:
            logging.error(f"Error stopping paper trading engine: {e}")

# Global instance
paper_trading_engine = RealisticPaperTradingEngine()

def get_paper_trading_engine() -> RealisticPaperTradingEngine:
    """Get the global paper trading engine instance"""
    return paper_trading_engine
