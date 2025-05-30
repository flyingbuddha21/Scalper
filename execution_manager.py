"""
Advanced Execution Manager for Indian Market Trading Bot
Handles order execution, position management, and risk controls
100% Market Ready - Routes to live Goodwill API or paper trading based on config
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict

from config_manager import get_config
from utils import (
    calculate_position_size, calculate_dynamic_stop_loss, calculate_portfolio_risk,
    format_currency, format_percentage
)
from strategy_manager import ScalpingSignal, SignalType, get_strategy_manager
from data_manager import get_data_manager
from goodwill_api_handler import get_goodwill_handler

class OrderType(Enum):
    """Order types supported"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LIMIT = "STOP_LIMIT"
    BRACKET = "BRACKET"

class OrderStatus(Enum):
    """Order status types"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL_FILLED = "PARTIAL_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class PositionSide(Enum):
    """Position sides"""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"

@dataclass
class Order:
    """Order data structure"""
    order_id: str
    symbol: str
    side: str  # BUY/SELL
    order_type: OrderType
    quantity: int
    price: float
    stop_price: float
    trigger_price: float
    time_in_force: str  # DAY, IOC, FOK
    status: OrderStatus
    filled_qty: int
    remaining_qty: int
    avg_price: float
    commission: float
    timestamp: datetime
    parent_signal_id: Optional[str]
    exchange_order_id: Optional[str]
    notes: str

@dataclass
class Position:
    """Position data structure"""
    symbol: str
    side: PositionSide
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

@dataclass
class ExecutionReport:
    """Execution report for completed trades"""
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
    net_pnl: float
    hold_time_seconds: int
    max_profit: float
    max_loss: float
    exit_reason: str
    success: bool

class MarketReadyExecutionManager:
    """Market-ready execution manager - routes to live API or paper trading"""
    
    def __init__(self):
        self.config = get_config()
        self.data_manager = get_data_manager()
        self.strategy_manager = get_strategy_manager()
        self.goodwill_handler = get_goodwill_handler()
        
        # Order and position tracking
        self.pending_orders: Dict[str, Order] = {}
        self.filled_orders: Dict[str, Order] = {}
        self.active_positions: Dict[str, Position] = {}
        self.execution_reports: List[ExecutionReport] = []
        
        # Portfolio metrics
        self.portfolio_value = 100000.0  # Will be updated from API
        self.available_cash = 100000.0
        self.buying_power = 100000.0
        self.daily_pnl = 0.0
        self.max_daily_loss_limit = -5000.0  # ₹5,000 daily loss limit
        
        # Execution statistics
        self.execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'avg_fill_time_ms': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_commission': 0.0,
            'total_pnl': 0.0
        }
        
        # Risk controls
        self.risk_controls = {
            'max_position_size': 10000,  # Max shares per position
            'max_portfolio_risk': 0.10,  # 10% max portfolio risk
            'max_single_position_risk': 0.02,  # 2% max risk per position
            'max_daily_trades': 50,  # Max trades per day
            'min_liquidity_check': True,  # Check liquidity before orders
            'pre_market_trading': False,  # Allow pre-market trading
            'after_hours_trading': False  # Allow after-hours trading
        }
        
        # Order management
        self.order_counter = 0
        self.execution_lock = threading.Lock()
        
        logging.info("Market-Ready Execution Manager initialized")
    
    async def execute_signal(self, signal: ScalpingSignal) -> Optional[str]:
        """Execute a trading signal - routes to live API or paper trading"""
        try:
            if self.config.trading.paper_trading:
                # Route to paper trading engine
                from paper_trading_engine import get_paper_trading_engine
                paper_engine = get_paper_trading_engine()
                return await paper_engine.place_paper_order(signal)
            else:
                # Live trading execution
                return await self._execute_live_signal(signal)
                
        except Exception as e:
            logging.error(f"Error executing signal for {signal.symbol}: {e}")
            return None
    
    async def _execute_live_signal(self, signal: ScalpingSignal) -> Optional[str]:
        """Execute live trading signal with comprehensive risk checks"""
        try:
            async with asyncio.Lock():
                # Pre-execution risk checks
                if not await self._pre_execution_risk_check(signal):
                    return None
                
                # Check market conditions
                if not await self._check_market_conditions(signal.symbol):
                    return None
                
                # Validate signal freshness (signals older than 30 seconds are rejected)
                signal_age = (datetime.now() - signal.timestamp).total_seconds()
                if signal_age > 30:
                    logging.warning(f"Signal for {signal.symbol} is {signal_age:.1f}s old, rejecting")
                    return None
                
                # Create primary order
                order_id = await self._create_primary_order(signal)
                if not order_id:
                    return None
                
                # Submit order to Goodwill API
                if await self._submit_order(order_id):
                    logging.info(f"Successfully executed live signal for {signal.symbol}: {order_id}")
                    return order_id
                else:
                    logging.error(f"Failed to submit live order for {signal.symbol}")
                    return None
                    
        except Exception as e:
            logging.error(f"Error executing live signal for {signal.symbol}: {e}")
            return None
    
    async def _pre_execution_risk_check(self, signal: ScalpingSignal) -> bool:
        """Comprehensive pre-execution risk validation"""
        try:
            # Check daily loss limit
            if self.daily_pnl <= self.max_daily_loss_limit:
                logging.warning(f"Daily loss limit reached: ₹{self.daily_pnl:.2f}")
                return False
            
            # Check maximum daily trades
            today_trades = len([r for r in self.execution_reports 
                              if r.entry_time.date() == datetime.now().date()])
            if today_trades >= self.risk_controls['max_daily_trades']:
                logging.warning(f"Daily trade limit reached: {today_trades}")
                return False
            
            # Check position size limits
            if signal.position_size > self.risk_controls['max_position_size']:
                logging.warning(f"Position size {signal.position_size} exceeds limit {self.risk_controls['max_position_size']}")
                return False
            
            # Check portfolio risk
            portfolio_risk = calculate_portfolio_risk(self.active_positions, self.portfolio_value)
            if portfolio_risk['risk_percentage'] > self.risk_controls['max_portfolio_risk'] * 100:
                logging.warning(f"Portfolio risk {portfolio_risk['risk_percentage']:.1f}% exceeds limit")
                return False
            
            # Check individual position risk
            position_risk = abs(signal.entry_price - signal.stop_loss_price) * signal.position_size
            position_risk_pct = position_risk / self.portfolio_value
            if position_risk_pct > self.risk_controls['max_single_position_risk']:
                logging.warning(f"Position risk {position_risk_pct:.2%} exceeds {self.risk_controls['max_single_position_risk']:.2%}")
                return False
            
            # Check available cash
            order_value = signal.entry_price * signal.position_size
            if order_value > self.available_cash:
                logging.warning(f"Insufficient cash: need ₹{order_value:.2f}, have ₹{self.available_cash:.2f}")
                return False
            
            # Check if symbol already has maximum exposure
            existing_position = self.active_positions.get(signal.symbol)
            if existing_position and existing_position.market_value > self.portfolio_value * 0.1:  # 10% max per symbol
                logging.warning(f"Maximum exposure reached for {signal.symbol}")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error in pre-execution risk check: {e}")
            return False
    
    async def _check_market_conditions(self, symbol: str) -> bool:
        """Check market conditions before order placement"""
        try:
            # Check if market is open
            if not self.config.is_trading_hours():
                if not (self.risk_controls['pre_market_trading'] or self.risk_controls['after_hours_trading']):
                    logging.warning(f"Market is closed and extended hours trading is disabled")
                    return False
            
            # Check liquidity
            if self.risk_controls['min_liquidity_check']:
                latest_tick = self.data_manager.get_latest_tick(symbol)
                if not latest_tick or latest_tick.volume < 1000:  # Minimum volume requirement
                    logging.warning(f"Insufficient liquidity for {symbol}")
                    return False
                
                # Check bid-ask spread
                latest_l1 = self.data_manager.get_latest_l1(symbol)
                if latest_l1 and latest_l1.spread_percent > 1.0:  # 1% max spread
                    logging.warning(f"Bid-ask spread too wide for {symbol}: {latest_l1.spread_percent:.2f}%")
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error checking market conditions for {symbol}: {e}")
            return False
    
    async def _create_primary_order(self, signal: ScalpingSignal) -> Optional[str]:
        """Create primary market order from signal"""
        try:
            self.order_counter += 1
            order_id = f"ORD_{signal.symbol}_{int(time.time())}_{self.order_counter}"
            
            # Determine order side
            side = "BUY" if signal.signal_type == SignalType.BUY else "SELL"
            
            order = Order(
                order_id=order_id,
                symbol=signal.symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=signal.position_size,
                price=signal.entry_price,
                stop_price=0.0,
                trigger_price=0.0,
                time_in_force="IOC",  # Immediate or Cancel for market orders
                status=OrderStatus.PENDING,
                filled_qty=0,
                remaining_qty=signal.position_size,
                avg_price=0.0,
                commission=0.0,
                timestamp=datetime.now(),
                parent_signal_id=id(signal),
                exchange_order_id=None,
                notes=f"Strategy: {signal.strategy.value}, Confidence: {signal.confidence:.2f}"
            )
            
            self.pending_orders[order_id] = order
            self.execution_stats['total_orders'] += 1
            
            # Create bracket orders for stop-loss and take-profits
            await self._create_bracket_orders(order_id, signal)
            
            logging.info(f"Created primary order {order_id} for {signal.symbol}: "
                        f"{side} {signal.position_size} shares at market")
            
            return order_id
            
        except Exception as e:
            logging.error(f"Error creating primary order: {e}")
            return None
    
    async def _create_bracket_orders(self, parent_order_id: str, signal: ScalpingSignal):
        """Create stop-loss and take-profit bracket orders"""
        try:
            parent_order = self.pending_orders.get(parent_order_id)
            if not parent_order:
                return
            
            # Stop-loss order
            sl_order_id = f"SL_{parent_order_id}"
            sl_side = "SELL" if parent_order.side == "BUY" else "BUY"
            
            sl_order = Order(
                order_id=sl_order_id,
                symbol=signal.symbol,
                side=sl_side,
                order_type=OrderType.STOP_LOSS,
                quantity=signal.position_size,
                price=signal.stop_loss_price,
                stop_price=signal.stop_loss_price,
                trigger_price=signal.stop_loss_price,
                time_in_force="DAY",
                status=OrderStatus.PENDING,
                filled_qty=0,
                remaining_qty=signal.position_size,
                avg_price=0.0,
                commission=0.0,
                timestamp=datetime.now(),
                parent_signal_id=parent_order_id,
                exchange_order_id=None,
                notes=f"Stop-loss for {parent_order_id}"
            )
            
            self.pending_orders[sl_order_id] = sl_order
            
            # Take-profit orders (3 levels)
            tp_quantities = [
                int(signal.position_size * 0.5),  # 50% at TP1
                int(signal.position_size * 0.3),  # 30% at TP2
                int(signal.position_size * 0.2)   # 20% at TP3
            ]
            
            tp_prices = [signal.take_profit_1, signal.take_profit_2, signal.take_profit_3]
            
            for i, (qty, price) in enumerate(zip(tp_quantities, tp_prices)):
                if qty <= 0:
                    continue
                
                tp_order_id = f"TP{i+1}_{parent_order_id}"
                tp_side = "SELL" if parent_order.side == "BUY" else "BUY"
                
                tp_order = Order(
                    order_id=tp_order_id,
                    symbol=signal.symbol,
                    side=tp_side,
                    order_type=OrderType.LIMIT,
                    quantity=qty,
                    price=price,
                    stop_price=0.0,
                    trigger_price=0.0,
                    time_in_force="DAY",
                    status=OrderStatus.PENDING,
                    filled_qty=0,
                    remaining_qty=qty,
                    avg_price=0.0,
                    commission=0.0,
                    timestamp=datetime.now(),
                    parent_signal_id=parent_order_id,
                    exchange_order_id=None,
                    notes=f"Take-profit {i+1} for {parent_order_id}"
                )
                
                self.pending_orders[tp_order_id] = tp_order
            
            logging.info(f"Created bracket orders for {parent_order_id}: SL + 3 TP levels")
            
        except Exception as e:
            logging.error(f"Error creating bracket orders: {e}")
    
    async def _submit_order(self, order_id: str) -> bool:
        """Submit order - routes to live API or paper trading"""
        try:
            order = self.pending_orders.get(order_id)
            if not order:
                return False
            
            if self.config.trading.paper_trading:
                # Paper trading is handled by paper_trading_engine
                return True
            else:
                # Live trading - submit to real Goodwill API
                return await self._submit_live_order(order_id)
                
        except Exception as e:
            logging.error(f"Error submitting order {order_id}: {e}")
            return False
    
    async def _submit_live_order(self, order_id: str) -> bool:
        """Submit live order to Goodwill API"""
        try:
            order = self.pending_orders[order_id]
            
            # Convert internal order to Goodwill API format
            order_data = {
                "tsym": f"{order.symbol}-EQ",  # Add -EQ suffix for equity
                "exchange": "NSE",  # Default to NSE
                "trantype": order.side,  # BUY/SELL
                "validity": "DAY",
                "pricetype": self._convert_order_type(order.order_type),
                "qty": str(order.quantity),
                "discqty": "0",  # No disclosed quantity for scalping
                "price": "0" if order.order_type == OrderType.MARKET else str(order.price),
                "trgprc": str(order.trigger_price) if order.trigger_price > 0 else "0",
                "product": "MIS",  # Intraday for scalping
                "amo": "NO"  # Not after market order
            }
            
            # Submit to Goodwill API
            response = await self.goodwill_handler.place_order(order_data)
            
            if response and response.status == 'submitted':
                # Update order with exchange order ID
                order.status = OrderStatus.SUBMITTED
                if hasattr(response, 'order_id'):
                    order.exchange_order_id = response.order_id
                    order.notes += f" | Exchange Order ID: {response.order_id}"
                
                logging.info(f"Live order submitted successfully: {order_id}")
                return True
            else:
                # Order rejected
                order.status = OrderStatus.REJECTED
                rejection_reason = response.message if response else "API submission failed"
                order.notes += f" | Rejected: {rejection_reason}"
                
                # Move to filled orders with rejected status
                self.filled_orders[order_id] = order
                del self.pending_orders[order_id]
                self.execution_stats['rejected_orders'] += 1
                
                logging.error(f"Live order rejected: {order_id} - {rejection_reason}")
                return False
                
        except Exception as e:
            logging.error(f"Error submitting live order {order_id}: {e}")
            order = self.pending_orders.get(order_id)
            if order:
                order.status = OrderStatus.REJECTED
                order.notes += f" | Error: {str(e)}"
            return False
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert internal order type to Goodwill API format"""
        mapping = {
            OrderType.MARKET: "MKT",
            OrderType.LIMIT: "L",
            OrderType.STOP_LOSS: "SL-M",
            OrderType.STOP_LIMIT: "SL-L"
        }
        return mapping.get(order_type, "MKT")
    
    async def update_positions_real_time(self):
        """Update all positions with real-time prices"""
        try:
            for symbol, position in self.active_positions.items():
                latest_tick = self.data_manager.get_latest_tick(symbol)
                if latest_tick:
                    position.current_price = latest_tick.last_price
                    
                    # Calculate unrealized P&L
                    if position.side == PositionSide.LONG:
                        position.unrealized_pnl = (position.current_price - position.avg_price) * position.quantity
                    elif position.side == PositionSide.SHORT:
                        position.unrealized_pnl = (position.avg_price - position.current_price) * abs(position.quantity)
                    else:
                        position.unrealized_pnl = 0.0
                    
                    position.total_pnl = position.realized_pnl + position.unrealized_pnl
                    position.market_value = abs(position.quantity) * position.current_price
                    
                    # Track max profit/loss
                    if position.unrealized_pnl > position.max_profit:
                        position.max_profit = position.unrealized_pnl
                    if position.unrealized_pnl < position.max_loss:
                        position.max_loss = position.unrealized_pnl
                    
                    # Update hold time
                    position.hold_time_seconds = int((datetime.now() - position.timestamp).total_seconds())
            
            # Update portfolio metrics
            await self._update_portfolio_metrics()
            
        except Exception as e:
            logging.error(f"Error updating positions real-time: {e}")
    
    async def _update_portfolio_metrics(self):
        """Update overall portfolio metrics"""
        try:
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.active_positions.values())
            total_realized_pnl = sum(pos.realized_pnl for pos in self.active_positions.values())
            total_market_value = sum(pos.market_value for pos in self.active_positions.values())
            
            self.portfolio_value = self.available_cash + total_market_value
            self.daily_pnl = total_unrealized_pnl + total_realized_pnl
            
            # Update buying power (simplified)
            self.buying_power = self.available_cash + (total_market_value * 0.5)  # 50% margin
            
        except Exception as e:
            logging.error(f"Error updating portfolio metrics: {e}")
    
    async def process_order_updates(self):
        """Process order status updates from broker/exchange"""
        try:
            if self.config.trading.paper_trading:
                # Paper trading handles its own order processing
                return
            else:
                # Live trading - get real order updates from Goodwill API
                await self._process_live_order_updates()
                
        except Exception as e:
            logging.error(f"Error processing order updates: {e}")
    
    async def _process_live_order_updates(self):
        """Process live order updates from Goodwill API"""
        try:
            # Get current order book from Goodwill API
            order_book = await self.goodwill_handler.get_order_book()
            
            if not order_book:
                return
            
            # Process each order in the order book
            for api_order in order_book:
                exchange_order_id = api_order.get('nstordno', '')
                api_status = api_order.get('status', '').lower()
                filled_qty = int(api_order.get('fillshares', 0))
                avg_price = float(api_order.get('avgprc', 0))
                
                # Find matching internal order by exchange order ID or symbol/details
                matching_order = None
                for order_id, order in self.pending_orders.items():
                    if (order.exchange_order_id == exchange_order_id or
                        (order.symbol.replace('-EQ', '') in api_order.get('tsym', '') and
                         order.side == api_order.get('trantype', '') and
                         order.quantity == int(api_order.get('qty', 0)))):
                        matching_order = (order_id, order)
                        break
                
                if not matching_order:
                    continue
                
                order_id, order = matching_order
                
                # Update order status based on API response
                if api_status == 'complete':
                    # Order fully filled
                    order.status = OrderStatus.FILLED
                    order.filled_qty = filled_qty
                    order.remaining_qty = 0
                    order.avg_price = avg_price
                    order.commission = self._calculate_commission(filled_qty, avg_price)
                    
                    # Move to filled orders
                    self.filled_orders[order_id] = order
                    del self.pending_orders[order_id]
                    
                    # Update position
                    await self._update_position(order)
                    
                    # Update statistics
                    self.execution_stats['filled_orders'] += 1
                    
                    logging.info(f"Live order filled: {order_id} at ₹{avg_price:.2f}")
                    
                elif api_status == 'rejected':
                    # Order rejected
                    order.status = OrderStatus.REJECTED
                    rejection_reason = api_order.get('rejreason', 'Unknown rejection reason')
                    order.notes += f" | Rejected: {rejection_reason}"
                    
                    # Move to filled orders with rejected status
                    self.filled_orders[order_id] = order
                    del self.pending_orders[order_id]
                    
                    self.execution_stats['rejected_orders'] += 1
                    
                    logging.warning(f"Live order rejected: {order_id} - {rejection_reason}")
                    
                elif api_status in ['open', 'pending']:
                    # Order still pending
                    if filled_qty > 0 and filled_qty < order.quantity:
                        # Partial fill
                        order.status = OrderStatus.PARTIAL_FILLED
                        order.filled_qty = filled_qty
                        order.remaining_qty = order.quantity - filled_qty
                        order.avg_price = avg_price
                        
                        logging.info(f"Live order partially filled: {order_id} - {filled_qty}/{order.quantity}")
                
                # Store exchange order ID for reference
                if exchange_order_id and not order.exchange_order_id:
                    order.exchange_order_id = exchange_order_id
                    order.notes += f" | Exchange Order ID: {exchange_order_id}"
                    
        except Exception as e:
            logging.error(f"Error processing live order updates: {e}")
    
    def _calculate_commission(self, quantity: int, price: float) -> float:
        """Calculate brokerage commission"""
        try:
            # Indian brokerage structure (typical discount broker)
            order_value = quantity * price
            
            # Flat fee per order (₹20) or 0.03% whichever is lower
            flat_fee = 20.0
            percentage_fee = order_value * 0.0003
            
            brokerage = min(flat_fee, percentage_fee)
            
            # Add other charges (STT, transaction charges, GST, etc.)
            stt = order_value * 0.001  # 0.1% STT for delivery
            transaction_charges = order_value * 0.00003  # 0.003%
            gst = brokerage * 0.18  # 18% GST on brokerage
            
            total_commission = brokerage + stt + transaction_charges + gst
            
            return round(total_commission, 2)
            
        except Exception as e:
            logging.error(f"Error calculating commission: {e}")
            return 0.0
    
    async def _update_position(self, filled_order: Order):
        """Update position after order fill"""
        try:
            symbol = filled_order.symbol
            
            if symbol not in self.active_positions:
                # Create new position
                side = PositionSide.LONG if filled_order.side == "BUY" else PositionSide.SHORT
                qty = filled_order.filled_qty if side == PositionSide.LONG else -filled_order.filled_qty
                
                position = Position(
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    avg_price=filled_order.avg_price,
                    current_price=filled_order.avg_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    total_pnl=0.0,
                    market_value=filled_order.avg_price * filled_order.filled_qty,
                    cost_basis=filled_order.avg_price * filled_order.filled_qty,
                    day_pnl=0.0,
                    timestamp=datetime.now(),
                    entry_orders=[filled_order.order_id],
                    exit_orders=[],
                    stop_loss_price=0.0,
                    take_profit_prices=[],
                    max_profit=0.0,
                    max_loss=0.0,
                    hold_time_seconds=0
                )
                
                self.active_positions[symbol] = position
                
            else:
                # Update existing position
                position = self.active_positions[symbol]
                
                if filled_order.side == "BUY":
                    # Adding to long position or reducing short position
                    if position.side == PositionSide.LONG:
                        # Average down/up calculation
                        total_cost = (position.quantity * position.avg_price) + (filled_order.filled_qty * filled_order.avg_price)
                        total_qty = position.quantity + filled_order.filled_qty
                        position.avg_price = total_cost / total_qty
                        position.quantity = total_qty
                    else:  # Covering short position
                        position.quantity += filled_order.filled_qty
                        if position.quantity >= 0:
                            position.side = PositionSide.LONG if position.quantity > 0 else PositionSide.FLAT
                
                else:  # SELL
                    # Adding to short position or reducing long position
                    if position.side == PositionSide.SHORT:
                        # Average down/up calculation for short
                        total_cost = abs(position.quantity * position.avg_price) + (filled_order.filled_qty * filled_order.avg_price)
                        total_qty = abs(position.quantity) + filled_order.filled_qty
                        position.avg_price = total_cost / total_qty
                        position.quantity = -total_qty
                    else:  # Selling long position
                        position.quantity -= filled_order.filled_qty
                        if position.quantity <= 0:
                            position.side = PositionSide.SHORT if position.quantity < 0 else PositionSide.FLAT
                
                # Update other fields
                position.market_value = abs(position.quantity) * position.current_price
                position.entry_orders.append(filled_order.order_id)
            
            # Update available cash
            if filled_order.side == "BUY":
                self.available_cash -= (filled_order.avg_price * filled_order.filled_qty + filled_order.commission)
            else:
                self.available_cash += (filled_order.avg_price * filled_order.filled_qty - filled_order.commission)
            
            logging.info(f"Updated position for {symbol}: {position.quantity} shares @ ₹{position.avg_price:.2f}")
            
        except Exception as e:
            logging.error(f"Error updating position: {e}")
    
    async def check_exit_conditions(self):
        """Check exit conditions for all active positions"""
        try:
            for symbol, position in list(self.active_positions.items()):
                # Check time-based exits
                if position.hold_time_seconds > 600:  # 10 minutes max hold
                    await self._create_exit_order(symbol, "TIME_STOP", position.quantity)
                
                # Check stop-loss conditions
                latest_tick = self.data_manager.get_latest_tick(symbol)
                if latest_tick and position.stop_loss_price > 0:
                    if ((position.side == PositionSide.LONG and latest_tick.last_price <= position.stop_loss_price) or
                        (position.side == PositionSide.SHORT and latest_tick.last_price >= position.stop_loss_price)):
                        await self._create_exit_order(symbol, "STOP_LOSS", position.quantity)
                
                # Check trailing stop
                if position.max_profit > position.avg_price * 0.02:  # 2% profit
                    trailing_stop = position.current_price * 0.98 if position.side == PositionSide.LONG else position.current_price * 1.02
                    if ((position.side == PositionSide.LONG and latest_tick.last_price <= trailing_stop) or
                        (position.side == PositionSide.SHORT and latest_tick.last_price >= trailing_stop)):
                        await self._create_exit_order(symbol, "TRAILING_STOP", position.quantity)
                
        except Exception as e:
            logging.error(f"Error checking exit conditions: {e}")
    
    async def _create_exit_order(self, symbol: str, exit_reason: str, quantity: int) -> Optional[str]:
        """Create exit order for position"""
        try:
            position = self.active_positions.get(symbol)
            if not position:
                return None
            
            self.order_counter += 1
            order_id = f"EXIT_{symbol}_{int(time.time())}_{self.order_counter}"
            
            side = "SELL" if position.side == PositionSide.LONG else "BUY"
            
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=abs(quantity),
                price=position.current_price,
                stop_price=0.0,
                trigger_price=0.0,
                time_in_force="IOC",
                status=OrderStatus.PENDING,
                filled_qty=0,
                remaining_qty=abs(quantity),
                avg_price=0.0,
                commission=0.0,
                timestamp=datetime.now(),
                parent_signal_id=None,
                exchange_order_id=None,
                notes=f"Exit: {exit_reason}"
            )
            
            self.pending_orders[order_id] = order
            
            if await self._submit_order(order_id):
                logging.info(f"Created exit order {order_id} for {symbol}: {exit_reason}")
                return order_id
            
            return None
            
        except Exception as e:
            logging.error(f"Error creating exit order for {symbol}: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        try:
            if order_id not in self.pending_orders:
                logging.warning(f"Order {order_id} not found in pending orders")
                return False
            
            order = self.pending_orders[order_id]
            
            if self.config.trading.paper_trading:
                # Paper trading - simple cancellation
                order.status = OrderStatus.CANCELLED
                del self.pending_orders[order_id]
                self.execution_stats['cancelled_orders'] += 1
                logging.info(f"Paper order cancelled: {order_id}")
                return True
            else:
                # Live trading - cancel via Goodwill API
                # Use exchange order ID if available
                cancel_order_id = order.exchange_order_id if order.exchange_order_id else order_id
                
                success = await self.goodwill_handler.cancel_order(cancel_order_id)
                if success:
                    order.status = OrderStatus.CANCELLED
                    del self.pending_orders[order_id]
                    self.execution_stats['cancelled_orders'] += 1
                    logging.info(f"Live order cancelled: {order_id}")
                    return True
                else:
                    logging.error(f"Failed to cancel live order: {order_id}")
                    return False
                    
        except Exception as e:
            logging.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: str = None) -> int:
        """Cancel all pending orders, optionally for a specific symbol"""
        try:
            orders_to_cancel = []
            
            for order_id, order in self.pending_orders.items():
                if symbol is None or order.symbol == symbol:
                    orders_to_cancel.append(order_id)
            
            cancelled_count = 0
            for order_id in orders_to_cancel:
                if await self.cancel_order(order_id):
                    cancelled_count += 1
            
            logging.info(f"Cancelled {cancelled_count} orders" + (f" for {symbol}" if symbol else ""))
            return cancelled_count
            
        except Exception as e:
            logging.error(f"Error cancelling all orders: {e}")
            return 0
    
    async def close_position(self, symbol: str, reason: str = "MANUAL_CLOSE") -> bool:
        """Close an entire position"""
        try:
            if symbol not in self.active_positions:
                logging.warning(f"No active position found for {symbol}")
                return False
            
            position = self.active_positions[symbol]
            
            if position.side == PositionSide.FLAT:
                logging.info(f"Position for {symbol} is already flat")
                return True
            
            # Cancel any existing orders for this symbol
            await self.cancel_all_orders(symbol)
            
            # Create market order to close position
            exit_order_id = await self._create_exit_order(symbol, reason, abs(position.quantity))
            
            if exit_order_id:
                logging.info(f"Created exit order to close position in {symbol}")
                return True
            else:
                logging.error(f"Failed to create exit order for {symbol}")
                return False
                
        except Exception as e:
            logging.error(f"Error closing position for {symbol}: {e}")
            return False
    
    async def close_all_positions(self, reason: str = "EMERGENCY_CLOSE") -> int:
        """Close all active positions"""
        try:
            positions_to_close = list(self.active_positions.keys())
            closed_count = 0
            
            for symbol in positions_to_close:
                if await self.close_position(symbol, reason):
                    closed_count += 1
            
            logging.info(f"Initiated closure of {closed_count} positions")
            return closed_count
            
        except Exception as e:
            logging.error(f"Error closing all positions: {e}")
            return 0
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            total_positions = len(self.active_positions)
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.active_positions.values())
            total_realized_pnl = sum(pos.realized_pnl for pos in self.active_positions.values())
            total_market_value = sum(pos.market_value for pos in self.active_positions.values())
            
            long_positions = len([p for p in self.active_positions.values() if p.side == PositionSide.LONG])
            short_positions = len([p for p in self.active_positions.values() if p.side == PositionSide.SHORT])
            
            return {
                'timestamp': datetime.now(),
                'trading_mode': 'Paper Trading' if self.config.trading.paper_trading else 'Live Trading',
                'portfolio_value': self.portfolio_value,
                'available_cash': self.available_cash,
                'buying_power': self.buying_power,
                'total_market_value': total_market_value,
                'daily_pnl': self.daily_pnl,
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_realized_pnl': total_realized_pnl,
                'total_positions': total_positions,
                'long_positions': long_positions,
                'short_positions': short_positions,
                'pending_orders': len(self.pending_orders),
                'filled_orders_today': len([o for o in self.filled_orders.values() 
                                          if o.timestamp.date() == datetime.now().date()]),
                'execution_stats': self.execution_stats.copy(),
                'risk_metrics': self._calculate_risk_metrics()
            }
            
        except Exception as e:
            logging.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def _calculate_risk_metrics(self) -> Dict[str, Any]:
        """Calculate portfolio risk metrics"""
        try:
            portfolio_risk = calculate_portfolio_risk(self.active_positions, self.portfolio_value)
            
            # Calculate additional risk metrics
            var_95 = 0.0  # Value at Risk (95% confidence)
            max_drawdown = 0.0
            sharpe_ratio = 0.0
            
            if self.execution_reports:
                returns = [r.net_pnl / self.portfolio_value for r in self.execution_reports]
                if returns:
                    import numpy as np
                    var_95 = np.percentile(returns, 5) * self.portfolio_value
                    
                    # Calculate max drawdown
                    cumulative_returns = np.cumsum(returns)
                    running_max = np.maximum.accumulate(cumulative_returns)
                    drawdown = cumulative_returns - running_max
                    max_drawdown = np.min(drawdown)
                    
                    # Calculate Sharpe ratio (simplified)
                    if len(returns) > 1:
                        avg_return = np.mean(returns)
                        std_return = np.std(returns)
                        if std_return > 0:
                            sharpe_ratio = (avg_return * 252) / (std_return * np.sqrt(252))  # Annualized
            
            return {
                'portfolio_risk_pct': portfolio_risk['risk_percentage'],
                'largest_position_risk': portfolio_risk['largest_position_risk'],
                'value_at_risk_95': var_95,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'risk_free_rate': self.config.risk.risk_free_rate,
                'correlation_risk': self._calculate_correlation_risk()
            }
            
        except Exception as e:
            logging.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_correlation_risk(self) -> float:
        """Calculate portfolio correlation risk"""
        try:
            # Simplified correlation risk calculation
            # In practice, this would use historical correlation matrix
            unique_sectors = len(set([pos.symbol[:3] for pos in self.active_positions.values()]))  # Rough sector proxy
            total_positions = len(self.active_positions)
            
            if total_positions <= 1:
                return 0.0
            
            # Higher correlation risk if positions are concentrated in fewer sectors
            correlation_risk = max(0.0, 1.0 - (unique_sectors / total_positions))
            return correlation_risk
            
        except Exception as e:
            logging.error(f"Error calculating correlation risk: {e}")
            return 0.0
    
    def get_position_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific position"""
        try:
            if symbol not in self.active_positions:
                return None
            
            position = self.active_positions[symbol]
            latest_tick = self.data_manager.get_latest_tick(symbol)
            
            return {
                'symbol': symbol,
                'side': position.side.value,
                'quantity': position.quantity,
                'avg_price': position.avg_price,
                'current_price': position.current_price,
                'market_value': position.market_value,
                'cost_basis': position.cost_basis,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_pct': (position.unrealized_pnl / position.cost_basis) * 100 if position.cost_basis > 0 else 0,
                'realized_pnl': position.realized_pnl,
                'total_pnl': position.total_pnl,
                'max_profit': position.max_profit,
                'max_loss': position.max_loss,
                'hold_time_seconds': position.hold_time_seconds,
                'hold_time_minutes': position.hold_time_seconds / 60,
                'entry_time': position.timestamp,
                'stop_loss_price': position.stop_loss_price,
                'take_profit_prices': position.take_profit_prices,
                'bid_price': latest_tick.bid_price if latest_tick else 0,
                'ask_price': latest_tick.ask_price if latest_tick else 0,
                'last_price': latest_tick.last_price if latest_tick else 0,
                'volume': latest_tick.volume if latest_tick else 0
            }
            
        except Exception as e:
            logging.error(f"Error getting position details for {symbol}: {e}")
            return None
    
    def get_order_history(self, symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get order history"""
        try:
            all_orders = list(self.filled_orders.values()) + list(self.pending_orders.values())
            
            # Filter by symbol if specified
            if symbol:
                all_orders = [o for o in all_orders if o.symbol == symbol]
            
            # Sort by timestamp (newest first)
            all_orders.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Limit results
            all_orders = all_orders[:limit]
            
            # Convert to dictionaries
            order_history = []
            for order in all_orders:
                order_dict = asdict(order)
                order_dict['timestamp'] = order.timestamp.isoformat()
                order_history.append(order_dict)
            
            return order_history
            
        except Exception as e:
            logging.error(f"Error getting order history: {e}")
            return []
    
    def get_trade_history(self, symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get completed trade history"""
        try:
            trade_history = self.execution_reports.copy()
            
            # Filter by symbol if specified
            if symbol:
                trade_history = [t for t in trade_history if t.symbol == symbol]
            
            # Sort by exit time (newest first)
            trade_history.sort(key=lambda x: x.exit_time, reverse=True)
            
            # Limit results
            trade_history = trade_history[:limit]
            
            # Convert to dictionaries
            trades = []
            for trade in trade_history:
                trade_dict = asdict(trade)
                trade_dict['entry_time'] = trade.entry_time.isoformat()
                trade_dict['exit_time'] = trade.exit_time.isoformat()
                trades.append(trade_dict)
            
            return trades
            
        except Exception as e:
            logging.error(f"Error getting trade history: {e}")
            return []
    
    async def emergency_stop_all(self) -> bool:
        """Emergency stop - cancel all orders and close all positions"""
        try:
            logging.warning("EMERGENCY STOP INITIATED")
            
            # Cancel all pending orders
            cancelled_orders = await self.cancel_all_orders()
            
            # Close all positions
            closed_positions = await self.close_all_positions("EMERGENCY_STOP")
            
            logging.warning(f"Emergency stop completed: {cancelled_orders} orders cancelled, {closed_positions} positions closed")
            
            return True
            
        except Exception as e:
            logging.error(f"Error during emergency stop: {e}")
            return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            if not self.execution_reports:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'avg_profit_per_trade': 0,
                    'total_pnl': 0,
                    'max_win': 0,
                    'max_loss': 0,
                    'avg_hold_time_minutes': 0,
                    'sharpe_ratio': 0
                }
            
            winning_trades = [t for t in self.execution_reports if t.net_pnl > 0]
            losing_trades = [t for t in self.execution_reports if t.net_pnl <= 0]
            
            total_trades = len(self.execution_reports)
            win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
            
            total_pnl = sum(t.net_pnl for t in self.execution_reports)
            avg_profit_per_trade = total_pnl / total_trades if total_trades > 0 else 0
            
            max_win = max(t.net_pnl for t in self.execution_reports) if self.execution_reports else 0
            max_loss = min(t.net_pnl for t in self.execution_reports) if self.execution_reports else 0
            
            avg_hold_time = sum(t.hold_time_seconds for t in self.execution_reports) / total_trades / 60 if total_trades > 0 else 0
            
            # Calculate Sharpe ratio
            if len(self.execution_reports) > 1:
                import numpy as np
                returns = [t.net_pnl / self.portfolio_value for t in self.execution_reports]
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = (avg_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0
            else:
                sharpe_ratio = 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_profit_per_trade': avg_profit_per_trade,
                'total_pnl': total_pnl,
                'total_commission': sum(t.commission for t in self.execution_reports),
                'max_win': max_win,
                'max_loss': max_loss,
                'avg_hold_time_minutes': avg_hold_time,
                'sharpe_ratio': sharpe_ratio,
                'profit_factor': sum(t.net_pnl for t in winning_trades) / abs(sum(t.net_pnl for t in losing_trades)) if losing_trades else float('inf')
            }
            
        except Exception as e:
            logging.error(f"Error getting performance summary: {e}")
            return {}

# Global instance
execution_manager = MarketReadyExecutionManager()

def get_execution_manager() -> MarketReadyExecutionManager:
    """Get the global execution manager instance"""
    return execution_manager
