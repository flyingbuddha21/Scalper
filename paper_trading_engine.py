#!/usr/bin/env python3
"""
Advanced Paper Trading System - Real-time Market Simulation
Mimics real trading scenarios with accurate P&L, slippage, and market impact
- Realistic order execution with bid-ask spreads
- Market impact simulation based on order size
- Accurate P&L calculation with all costs
- Real-time position tracking and updates
"""

import json
import time
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import statistics

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LIMIT = "STOP_LIMIT"

@dataclass
class PaperOrder:
    """Paper trading order with realistic execution"""
    order_id: str
    symbol: str
    side: str  # BUY/SELL
    order_type: OrderType
    quantity: int
    price: Optional[float]  # None for market orders
    stop_price: Optional[float]  # For stop orders
    
    # Order state
    status: OrderStatus
    filled_quantity: int
    filled_price: float
    remaining_quantity: int
    
    # Execution details
    placed_time: datetime
    filled_time: Optional[datetime]
    slippage: float
    brokerage: float
    taxes: float
    total_cost: float
    
    # Market conditions at execution
    market_price: float
    bid_price: float
    ask_price: float
    spread_pct: float

@dataclass
class PaperPosition:
    """Paper trading position with real-time P&L"""
    symbol: str
    side: str  # LONG/SHORT
    quantity: int
    avg_entry_price: float
    current_price: float
    
    # P&L calculations
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    
    # Costs
    total_brokerage: float
    total_taxes: float
    net_investment: float
    
    # Position details
    entry_time: datetime
    last_updated: datetime
    orders: List[str]  # Order IDs that created this position
    
    # Risk metrics
    max_profit: float
    max_loss: float
    hold_time_minutes: int

class PaperTradingEngine:
    """
    Advanced paper trading engine with realistic market simulation
    """
    
    def __init__(self, api, initial_capital: float = 100000.0):
        """Initialize paper trading engine"""
        self.api = api
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.available_margin = initial_capital
        
        # Trading state
        self.orders = {}  # order_id -> PaperOrder
        self.positions = {}  # symbol -> PaperPosition
        self.trades = []  # Completed trades history
        
        # Market data cache for realistic execution
        self.market_data_cache = {}
        self.last_market_update = {}
        
        # Trading costs configuration
        self.cost_config = {
            'equity_brokerage_pct': 0.03,      # 0.03% for equity
            'fno_brokerage_flat': 20.0,        # ₹20 per lot for F&O
            'stt_equity_pct': 0.025,           # 0.025% STT for equity delivery
            'stt_fno_pct': 0.0125,             # 0.0125% STT for F&O
            'exchange_charges_pct': 0.00345,   # 0.00345% exchange charges
            'gst_pct': 18.0,                   # 18% GST on brokerage
            'sebi_charges_pct': 0.0001,        # 0.0001% SEBI charges
            'stamp_duty_pct': 0.003,           # 0.003% stamp duty
        }
        
        # Market impact simulation
        self.market_impact_config = {
            'small_order_threshold': 10000,    # Orders < ₹10k have minimal impact
            'large_order_threshold': 100000,   # Orders > ₹1L have significant impact
            'max_impact_pct': 0.1,             # Maximum 0.1% market impact
            'liquidity_recovery_time': 30      # Seconds for liquidity recovery
        }
        
        # Execution simulation
        self.execution_config = {
            'market_order_slippage_pct': 0.02,  # 0.02% average slippage
            'limit_order_fill_probability': 0.8, # 80% fill probability
            'execution_delay_ms': 200,          # 200ms execution delay
            'partial_fill_threshold': 0.3       # 30% chance of partial fills
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'peak_capital': initial_capital,
            'win_rate': 0.0,
            'avg_trade_pnl': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Real-time update thread
        self.update_thread = None
        self.running = False
    
    def start_realtime_updates(self):
        """Start real-time position and P&L updates"""
        if self.running:
            return
        
        self.running = True
        
        def update_loop():
            logger.info("🔄 Paper trading real-time updates started")
            
            while self.running:
                try:
                    # Update market data for all symbols with positions
                    self._update_market_data()
                    
                    # Update position P&L
                    self._update_position_pnl()
                    
                    # Check pending orders
                    self._process_pending_orders()
                    
                    # Update performance metrics
                    self._update_performance_metrics()
                    
                    time.sleep(2)  # Update every 2 seconds
                    
                except Exception as e:
                    logger.error(f"❌ Paper trading update error: {e}")
                    time.sleep(10)
            
            logger.info("🛑 Paper trading updates stopped")
        
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
    
    def stop_realtime_updates(self):
        """Stop real-time updates"""
        self.running = False
    
    def place_order(self, symbol: str, side: str, quantity: int, 
                   order_type: OrderType = OrderType.MARKET, 
                   price: Optional[float] = None,
                   stop_price: Optional[float] = None) -> str:
        """
        Place paper trading order with realistic execution simulation
        """
        try:
            # Generate unique order ID
            order_id = f"PAPER_{uuid.uuid4().hex[:8].upper()}"
            
            # Get current market data
            market_data = self._get_current_market_data(symbol)
            if not market_data:
                logger.error(f"❌ Cannot place order - no market data for {symbol}")
                return None
            
            # Calculate bid-ask prices
            ltp = market_data['ltp']
            spread_pct = self._estimate_bid_ask_spread(symbol, ltp)
            bid_price = ltp * (1 - spread_pct/200)
            ask_price = ltp * (1 + spread_pct/200)
            
            # Create paper order
            paper_order = PaperOrder(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                status=OrderStatus.PENDING,
                filled_quantity=0,
                filled_price=0.0,
                remaining_quantity=quantity,
                placed_time=datetime.now(),
                filled_time=None,
                slippage=0.0,
                brokerage=0.0,
                taxes=0.0,
                total_cost=0.0,
                market_price=ltp,
                bid_price=bid_price,
                ask_price=ask_price,
                spread_pct=spread_pct
            )
            
            # Check margin requirements
            if not self._check_margin_requirements(paper_order):
                paper_order.status = OrderStatus.REJECTED
                logger.warning(f"❌ Order rejected - insufficient margin: {order_id}")
                return None
            
            # Store order
            self.orders[order_id] = paper_order
            
            # Immediate execution for market orders
            if order_type == OrderType.MARKET:
                self._execute_market_order(paper_order)
            
            logger.info(f"📝 Paper order placed: {side} {quantity} {symbol} @ {price or 'MARKET'} | ID: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"❌ Paper order placement error: {e}")
            return None
    
    def _get_current_market_data(self, symbol: str) -> Optional[Dict]:
        """Get current market data with caching"""
        try:
            # Check cache first (avoid too frequent API calls)
            current_time = time.time()
            if (symbol in self.market_data_cache and 
                current_time - self.last_market_update.get(symbol, 0) < 2):
                return self.market_data_cache[symbol]
            
            # Fetch fresh data
            quote = self.api.get_quote(symbol)
            if quote:
                market_data = {
                    'ltp': float(quote.get('ltp', 0)),
                    'high': float(quote.get('high', 0)),
                    'low': float(quote.get('low', 0)),
                    'volume': int(quote.get('volume', 0)),
                    'prev_close': float(quote.get('prev_close', 0))
                }
                
                # Cache the data
                self.market_data_cache[symbol] = market_data
                self.last_market_update[symbol] = current_time
                
                return market_data
            
            return None
            
        except Exception as e:
            logger.debug(f"Market data fetch error for {symbol}: {e}")
            return None
    
    def _estimate_bid_ask_spread(self, symbol: str, ltp: float) -> float:
        """Estimate realistic bid-ask spread percentage"""
        try:
            # Base spread by price level
            if ltp < 50:
                base_spread = 0.3
            elif ltp < 200:
                base_spread = 0.15
            elif ltp < 1000:
                base_spread = 0.08
            elif ltp < 5000:
                base_spread = 0.05
            else:
                base_spread = 0.03
            
            # Adjust for instrument type (if available)
            # Indices typically have tighter spreads
            if symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY']:
                base_spread *= 0.5
            
            return base_spread
            
        except Exception as e:
            logger.debug(f"Spread estimation error: {e}")
            return 0.1
    
    def _check_margin_requirements(self, order: PaperOrder) -> bool:
        """Check if sufficient margin is available"""
        try:
            # Calculate required margin
            required_margin = self._calculate_required_margin(order)
            
            # Check available margin
            if required_margin > self.available_margin:
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Margin check error: {e}")
            return False
    
    def _calculate_required_margin(self, order: PaperOrder) -> float:
        """Calculate margin required for order"""
        try:
            if order.order_type == OrderType.MARKET:
                price = order.ask_price if order.side == 'BUY' else order.bid_price
            else:
                price = order.price or order.market_price
            
            order_value = price * order.quantity
            
            # For equity, full amount required
            # For F&O, margin percentage (simplified)
            if order.symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY']:
                # Index futures - assume 10-15% margin
                return order_value * 0.125
            else:
                # Equity - full amount
                return order_value
                
        except Exception as e:
            logger.debug(f"Margin calculation error: {e}")
            return order.quantity * order.market_price
    
    def _execute_market_order(self, order: PaperOrder):
        """Execute market order with realistic simulation"""
        try:
            # Simulate execution delay
            time.sleep(self.execution_config['execution_delay_ms'] / 1000)
            
            # Determine execution price with slippage
            if order.side == 'BUY':
                base_price = order.ask_price
            else:
                base_price = order.bid_price
            
            # Calculate slippage
            slippage_pct = self._calculate_slippage(order)
            
            if order.side == 'BUY':
                execution_price = base_price * (1 + slippage_pct/100)
            else:
                execution_price = base_price * (1 - slippage_pct/100)
            
            # Simulate partial fills for large orders
            fill_quantity = order.quantity
            if self._should_partial_fill(order):
                fill_quantity = int(order.quantity * (0.3 + 0.7 * statistics.random()))
                order.status = OrderStatus.PARTIAL
            else:
                order.status = OrderStatus.FILLED
            
            # Update order details
            order.filled_quantity = fill_quantity
            order.remaining_quantity = order.quantity - fill_quantity
            order.filled_price = execution_price
            order.filled_time = datetime.now()
            order.slippage = slippage_pct
            
            # Calculate costs
            self._calculate_trading_costs(order)
            
            # Update position
            self._update_position(order)
            
            # Update available margin
            self._update_available_margin()
            
            logger.info(f"✅ Paper order executed: {order.side} {fill_quantity} {order.symbol} @ ₹{execution_price:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Market order execution error: {e}")
            order.status = OrderStatus.REJECTED
    
    def _calculate_slippage(self, order: PaperOrder) -> float:
        """Calculate realistic slippage based on order size and market conditions"""
        try:
            order_value = order.quantity * order.market_price
            base_slippage = self.execution_config['market_order_slippage_pct']
            
            # Market impact based on order size
            if order_value > self.market_impact_config['large_order_threshold']:
                market_impact = min(
                    (order_value / self.market_impact_config['large_order_threshold'] - 1) * 0.02,
                    self.market_impact_config['max_impact_pct']
                )
                total_slippage = base_slippage + market_impact
            elif order_value < self.market_impact_config['small_order_threshold']:
                total_slippage = base_slippage * 0.5  # Reduced slippage for small orders
            else:
                total_slippage = base_slippage
            
            # Add some randomness to simulate market volatility
            volatility_factor = 1 + (statistics.random() - 0.5) * 0.5
            return total_slippage * volatility_factor
            
        except Exception as e:
            logger.debug(f"Slippage calculation error: {e}")
            return self.execution_config['market_order_slippage_pct']
    
    def _should_partial_fill(self, order: PaperOrder) -> bool:
        """Determine if order should be partially filled"""
        try:
            order_value = order.quantity * order.market_price
            
            # Large orders more likely to be partially filled
            if order_value > self.market_impact_config['large_order_threshold']:
                return statistics.random() < 0.6  # 60% chance
            elif order_value > self.market_impact_config['small_order_threshold']:
                return statistics.random() < 0.3  # 30% chance
            else:
                return False  # Small orders usually fill completely
                
        except Exception as e:
            logger.debug(f"Partial fill check error: {e}")
            return False
    
    def _calculate_trading_costs(self, order: PaperOrder):
        """Calculate realistic trading costs"""
        try:
            order_value = order.filled_quantity * order.filled_price
            
            # Brokerage calculation
            if order.symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY']:
                # F&O brokerage - flat per lot
                lots = max(1, order.filled_quantity // 50)  # Assume 50 lot size
                brokerage = lots * self.cost_config['fno_brokerage_flat']
            else:
                # Equity brokerage - percentage
                brokerage = order_value * self.cost_config['equity_brokerage_pct'] / 100
            
            # STT calculation
            if order.symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY']:
                stt = order_value * self.cost_config['stt_fno_pct'] / 100
            else:
                stt = order_value * self.cost_config['stt_equity_pct'] / 100
            
            # Other charges
            exchange_charges = order_value * self.cost_config['exchange_charges_pct'] / 100
            sebi_charges = order_value * self.cost_config['sebi_charges_pct'] / 100
            stamp_duty = order_value * self.cost_config['stamp_duty_pct'] / 100
            
            # GST on brokerage
            gst = brokerage * self.cost_config['gst_pct'] / 100
            
            # Total costs
            order.brokerage = brokerage + gst
            order.taxes = stt + exchange_charges + sebi_charges + stamp_duty
            order.total_cost = order.brokerage + order.taxes
            
        except Exception as e:
            logger.debug(f"Cost calculation error: {e}")
            order.brokerage = 0.0
            order.taxes = 0.0
            order.total_cost = 0.0
    
    def _update_position(self, order: PaperOrder):
        """Update position after order execution"""
        try:
            symbol = order.symbol
            
            if symbol not in self.positions:
                # Create new position
                if order.side == 'BUY':
                    side = 'LONG'
                    quantity = order.filled_quantity
                else:
                    side = 'SHORT'
                    quantity = -order.filled_quantity
                
                position = PaperPosition(
                    symbol=symbol,
                    side=side,
                    quantity=abs(quantity),
                    avg_entry_price=order.filled_price,
                    current_price=order.filled_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    total_pnl=0.0,
                    total_brokerage=order.brokerage,
                    total_taxes=order.taxes,
                    net_investment=order.filled_price * order.filled_quantity,
                    entry_time=order.filled_time,
                    last_updated=datetime.now(),
                    orders=[order.order_id],
                    max_profit=0.0,
                    max_loss=0.0,
                    hold_time_minutes=0
                )
                
                self.positions[symbol] = position
                
            else:
                # Update existing position
                position = self.positions[symbol]
                
                if ((position.side == 'LONG' and order.side == 'BUY') or
                    (position.side == 'SHORT' and order.side == 'SELL')):
                    # Adding to position
                    total_value = (position.quantity * position.avg_entry_price + 
                                 order.filled_quantity * order.filled_price)
                    total_quantity = position.quantity + order.filled_quantity
                    position.avg_entry_price = total_value / total_quantity
                    position.quantity = total_quantity
                    
                else:
                    # Closing/reducing position
                    if order.filled_quantity >= position.quantity:
                        # Position closed completely
                        realized_pnl = self._calculate_realized_pnl(position, order)
                        position.realized_pnl += realized_pnl
                        
                        # Create trade record
                        self._create_trade_record(position, order, realized_pnl)
                        
                        # Remove position if completely closed
                        if order.filled_quantity == position.quantity:
                            del self.positions[symbol]
                            return
                        else:
                            # Reverse position
                            remaining_qty = order.filled_quantity - position.quantity
                            position.quantity = remaining_qty
                            position.side = 'LONG' if order.side == 'BUY' else 'SHORT'
                            position.avg_entry_price = order.filled_price
                    else:
                        # Partial close
                        close_qty = order.filled_quantity
                        realized_pnl = self._calculate_partial_realized_pnl(position, order, close_qty)
                        position.realized_pnl += realized_pnl
                        position.quantity -= close_qty
                
                # Update costs
                position.total_brokerage += order.brokerage
                position.total_taxes += order.taxes
                position.orders.append(order.order_id)
                position.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Position update error: {e}")
    
    def _calculate_realized_pnl(self, position: PaperPosition, order: PaperOrder) -> float:
        """Calculate realized P&L when position is closed"""
        try:
            if position.side == 'LONG':
                # Long position closed by selling
                pnl = (order.filled_price - position.avg_entry_price) * position.quantity
            else:
                # Short position closed by buying
                pnl = (position.avg_entry_price - order.filled_price) * position.quantity
            
            # Subtract costs
            pnl -= (position.total_brokerage + position.total_taxes + order.brokerage + order.taxes)
            
            return pnl
            
        except Exception as e:
            logger.debug(f"Realized P&L calculation error: {e}")
            return 0.0
    
    def _calculate_partial_realized_pnl(self, position: PaperPosition, order: PaperOrder, close_qty: int) -> float:
        """Calculate realized P&L for partial position close"""
        try:
            if position.side == 'LONG':
                pnl = (order.filled_price - position.avg_entry_price) * close_qty
            else:
                pnl = (position.avg_entry_price - order.filled_price) * close_qty
            
            # Subtract proportional costs
            cost_per_share = (position.total_brokerage + position.total_taxes) / position.quantity
            pnl -= (cost_per_share * close_qty + order.total_cost)
            
            return pnl
            
        except Exception as e:
            logger.debug(f"Partial realized P&L calculation error: {e}")
            return 0.0
    
    def _create_trade_record(self, position: PaperPosition, closing_order: PaperOrder, realized_pnl: float):
        """Create completed trade record"""
        try:
            trade_record = {
                'symbol': position.symbol,
                'side': position.side,
                'quantity': position.quantity,
                'entry_price': position.avg_entry_price,
                'exit_price': closing_order.filled_price,
                'entry_time': position.entry_time,
                'exit_time': closing_order.filled_time,
                'hold_time_minutes': (closing_order.filled_time - position.entry_time).total_seconds() / 60,
                'realized_pnl': realized_pnl,
                'total_costs': position.total_brokerage + position.total_taxes + closing_order.total_cost,
                'max_profit': position.max_profit,
                'max_loss': position.max_loss,
                'orders': position.orders + [closing_order.order_id]
            }
            
            self.trades.append(trade_record)
            
        except Exception as e:
            logger.debug(f"Trade record creation error: {e}")
    
    def _update_market_data(self):
        """Update market data for all symbols with positions"""
        try:
            symbols_to_update = list(self.positions.keys())
            
            for symbol in symbols_to_update:
                self._get_current_market_data(symbol)
                
        except Exception as e:
            logger.debug(f"Market data update error: {e}")
    
    def _update_position_pnl(self):
        """Update unrealized P&L for all positions"""
        try:
            for symbol, position in self.positions.items():
                market_data = self.market_data_cache.get(symbol)
                if not market_data:
                    continue
                
                current_price = market_data['ltp']
                position.current_price = current_price
                
                # Calculate unrealized P&L
                if position.side == 'LONG':
                    unrealized_pnl = (current_price - position.avg_entry_price) * position.quantity
                else:
                    unrealized_pnl = (position.avg_entry_price - current_price) * position.quantity
                
                # Subtract costs for net P&L
                unrealized_pnl -= (position.total_brokerage + position.total_taxes)
                
                position.unrealized_pnl = unrealized_pnl
                position.total_pnl = position.realized_pnl + unrealized_pnl
                
                # Update max profit/loss tracking
                if unrealized_pnl > position.max_profit:
                    position.max_profit = unrealized_pnl
                if unrealized_pnl < position.max_loss:
                    position.max_loss = unrealized_pnl
                
                # Update hold time
                position.hold_time_minutes = (datetime.now() - position.entry_time).total_seconds() / 60
                position.last_updated = datetime.now()
                
        except Exception as e:
            logger.debug(f"Position P&L update error: {e}")
    
    def _process_pending_orders(self):
        """Process pending limit and stop orders"""
        try:
            pending_orders = [order for order in self.orders.values() 
                            if order.status == OrderStatus.PENDING]
            
            for order in pending_orders:
                if order.order_type == OrderType.LIMIT:
                    self._check_limit_order_execution(order)
                elif order.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LIMIT]:
                    self._check_stop_order_execution(order)
                    
        except Exception as e:
            logger.debug(f"Pending order processing error: {e}")
    
    def _check_limit_order_execution(self, order: PaperOrder):
        """Check if limit order should be executed"""
        try:
            market_data = self.market_data_cache.get(order.symbol)
            if not market_data:
                return
            
            current_price = market_data['ltp']
            
            # Check if limit price is hit
            should_execute = False
            
            if order.side == 'BUY' and current_price <= order.price:
                should_execute = True
            elif order.side == 'SELL' and current_price >= order.price:
                should_execute = True
            
            if should_execute:
                # Simulate execution probability
                if statistics.random() < self.execution_config['limit_order_fill_probability']:
                    order.filled_price = order.price
                    order.filled_quantity = order.quantity
                    order.remaining_quantity = 0
                    order.status = OrderStatus.FILLED
                    order.filled_time = datetime.now()
                    
                    # Calculate costs and update position
                    self._calculate_trading_costs(order)
                    self._update_position(order)
                    
                    logger.info(f"✅ Limit order executed: {order.order_id}")
                    
        except Exception as e:
            logger.debug(f"Limit order check error: {e}")
    
    def _check_stop_order_execution(self, order: PaperOrder):
        """Check if stop order should be triggered"""
        try:
            market_data = self.market_data_cache.get(order.symbol)
            if not market_data:
                return
            
            current_price = market_data['ltp']
            
            # Check if stop price is hit
            should_trigger = False
            
            if order.side == 'BUY' and current_price >= order.stop_price:
                should_trigger = True
            elif order.side == 'SELL' and current_price <= order.stop_price:
                should_trigger = True
            
            if should_trigger:
                if order.order_type == OrderType.STOP_LOSS:
                    # Convert to market order
                    order.order_type = OrderType.MARKET
                    self._execute_market_order(order)
                    logger.info(f"🛑 Stop loss triggered: {order.order_id}")
                    
        except Exception as e:
            logger.debug(f"Stop order check error: {e}")
    
    def _update_available_margin(self):
        """Update available margin based on current positions"""
        try:
            used_margin = 0.0
            
            for position in self.positions.values():
                # Calculate margin used by position
                position_value = position.quantity * position.current_price
                
                if position.symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY']:
                  # F&O margin requirement
                    used_margin += position_value * 0.125  # 12.5% margin
                else:
                    # Equity - full value
                    used_margin += position_value
            
            self.available_margin = self.current_capital - used_margin
            
        except Exception as e:
            logger.debug(f"Margin update error: {e}")
    
    def _update_performance_metrics(self):
        """Update overall performance metrics"""
        try:
            # Calculate total P&L from all positions and trades
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_realized_pnl = sum(trade['realized_pnl'] for trade in self.trades)
            total_pnl = total_realized_pnl + total_unrealized_pnl
            
            # Update current capital
            self.current_capital = self.initial_capital + total_pnl
            
            # Track peak capital for drawdown calculation
            if self.current_capital > self.performance_metrics['peak_capital']:
                self.performance_metrics['peak_capital'] = self.current_capital
            
            # Calculate drawdown
            drawdown = self.performance_metrics['peak_capital'] - self.current_capital
            if drawdown > self.performance_metrics['max_drawdown']:
                self.performance_metrics['max_drawdown'] = drawdown
            
            # Update metrics
            self.performance_metrics['total_trades'] = len(self.trades)
            self.performance_metrics['total_pnl'] = total_pnl
            
            if len(self.trades) > 0:
                winning_trades = [t for t in self.trades if t['realized_pnl'] > 0]
                self.performance_metrics['winning_trades'] = len(winning_trades)
                self.performance_metrics['win_rate'] = (len(winning_trades) / len(self.trades)) * 100
                self.performance_metrics['avg_trade_pnl'] = total_realized_pnl / len(self.trades)
                
                # Calculate Sharpe ratio (simplified)
                if len(self.trades) > 5:
                    pnl_values = [t['realized_pnl'] for t in self.trades]
                    avg_return = statistics.mean(pnl_values)
                    std_return = statistics.stdev(pnl_values) if len(pnl_values) > 1 else 1
                    self.performance_metrics['sharpe_ratio'] = (avg_return / std_return) if std_return > 0 else 0
            
        except Exception as e:
            logger.debug(f"Performance metrics update error: {e}")
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        try:
            if order_id not in self.orders:
                return False
            
            order = self.orders[order_id]
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                logger.info(f"❌ Order cancelled: {order_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Order cancellation error: {e}")
            return False
    
    def close_position(self, symbol: str, order_type: OrderType = OrderType.MARKET) -> Optional[str]:
        """Close position with market or limit order"""
        try:
            if symbol not in self.positions:
                logger.warning(f"❌ No position found for {symbol}")
                return None
            
            position = self.positions[symbol]
            
            # Determine closing side
            close_side = 'SELL' if position.side == 'LONG' else 'BUY'
            
            # Place closing order
            order_id = self.place_order(
                symbol=symbol,
                side=close_side,
                quantity=position.quantity,
                order_type=order_type
            )
            
            logger.info(f"🔒 Closing position: {symbol} | Order ID: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Position closing error: {e}")
            return None
    
    def close_all_positions(self) -> List[str]:
        """Close all open positions"""
        try:
            order_ids = []
            symbols_to_close = list(self.positions.keys())
            
            for symbol in symbols_to_close:
                order_id = self.close_position(symbol)
                if order_id:
                    order_ids.append(order_id)
            
            logger.info(f"🔒 Closing all positions: {len(order_ids)} orders placed")
            return order_ids
            
        except Exception as e:
            logger.error(f"Close all positions error: {e}")
            return []
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        try:
            # Calculate totals
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_realized_pnl = sum(trade['realized_pnl'] for trade in self.trades)
            total_pnl = total_realized_pnl + total_unrealized_pnl
            
            total_investment = sum(pos.net_investment for pos in self.positions.values())
            
            # Position details
            position_details = []
            for symbol, pos in self.positions.items():
                position_details.append({
                    'symbol': symbol,
                    'side': pos.side,
                    'quantity': pos.quantity,
                    'avg_entry_price': round(pos.avg_entry_price, 2),
                    'current_price': round(pos.current_price, 2),
                    'unrealized_pnl': round(pos.unrealized_pnl, 2),
                    'total_pnl': round(pos.total_pnl, 2),
                    'pnl_percentage': round((pos.unrealized_pnl / pos.net_investment) * 100, 2),
                    'hold_time_minutes': int(pos.hold_time_minutes),
                    'max_profit': round(pos.max_profit, 2),
                    'max_loss': round(pos.max_loss, 2)
                })
            
            # Recent trades
            recent_trades = []
            for trade in self.trades[-10:]:
                recent_trades.append({
                    'symbol': trade['symbol'],
                    'side': trade['side'],
                    'quantity': trade['quantity'],
                    'entry_price': round(trade['entry_price'], 2),
                    'exit_price': round(trade['exit_price'], 2),
                    'realized_pnl': round(trade['realized_pnl'], 2),
                    'hold_time_minutes': round(trade['hold_time_minutes'], 1),
                    'exit_time': trade['exit_time'].strftime('%H:%M:%S')
                })
            
            return {
                'account_summary': {
                    'initial_capital': self.initial_capital,
                    'current_capital': round(self.current_capital, 2),
                    'available_margin': round(self.available_margin, 2),
                    'total_pnl': round(total_pnl, 2),
                    'pnl_percentage': round((total_pnl / self.initial_capital) * 100, 2),
                    'unrealized_pnl': round(total_unrealized_pnl, 2),
                    'realized_pnl': round(total_realized_pnl, 2)
                },
                'positions': position_details,
                'recent_trades': recent_trades,
                'performance_metrics': {
                    'total_trades': self.performance_metrics['total_trades'],
                    'winning_trades': self.performance_metrics['winning_trades'],
                    'win_rate': round(self.performance_metrics['win_rate'], 1),
                    'avg_trade_pnl': round(self.performance_metrics['avg_trade_pnl'], 2),
                    'max_drawdown': round(self.performance_metrics['max_drawdown'], 2),
                    'sharpe_ratio': round(self.performance_metrics['sharpe_ratio'], 3)
                },
                'order_summary': {
                    'total_orders': len(self.orders),
                    'pending_orders': len([o for o in self.orders.values() if o.status == OrderStatus.PENDING]),
                    'filled_orders': len([o for o in self.orders.values() if o.status == OrderStatus.FILLED]),
                    'cancelled_orders': len([o for o in self.orders.values() if o.status == OrderStatus.CANCELLED])
                }
            }
            
        except Exception as e:
            logger.error(f"Portfolio summary error: {e}")
            return {'error': str(e)}
    
    def get_order_history(self) -> List[Dict]:
        """Get complete order history"""
        try:
            order_history = []
            
            for order in self.orders.values():
                order_dict = {
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'order_type': order.order_type.value,
                    'quantity': order.quantity,
                    'price': order.price,
                    'status': order.status.value,
                    'filled_quantity': order.filled_quantity,
                    'filled_price': round(order.filled_price, 2) if order.filled_price else None,
                    'placed_time': order.placed_time.strftime('%H:%M:%S'),
                    'filled_time': order.filled_time.strftime('%H:%M:%S') if order.filled_time else None,
                    'slippage': round(order.slippage, 3),
                    'total_cost': round(order.total_cost, 2),
                    'spread_pct': round(order.spread_pct, 3)
                }
                order_history.append(order_dict)
            
            # Sort by placed time (most recent first)
            order_history.sort(key=lambda x: x['placed_time'], reverse=True)
            
            return order_history
            
        except Exception as e:
            logger.error(f"Order history error: {e}")
            return []
    
    def get_trade_analytics(self) -> Dict:
        """Get detailed trade analytics"""
        try:
            if not self.trades:
                return {'message': 'No completed trades yet'}
            
            # P&L analysis
            pnl_values = [trade['realized_pnl'] for trade in self.trades]
            winning_trades = [pnl for pnl in pnl_values if pnl > 0]
            losing_trades = [pnl for pnl in pnl_values if pnl < 0]
            
            # Hold time analysis
            hold_times = [trade['hold_time_minutes'] for trade in self.trades]
            
            # Symbol analysis
            symbol_performance = {}
            for trade in self.trades:
                symbol = trade['symbol']
                if symbol not in symbol_performance:
                    symbol_performance[symbol] = {'trades': 0, 'pnl': 0, 'wins': 0}
                
                symbol_performance[symbol]['trades'] += 1
                symbol_performance[symbol]['pnl'] += trade['realized_pnl']
                if trade['realized_pnl'] > 0:
                    symbol_performance[symbol]['wins'] += 1
            
            # Calculate win rates for symbols
            for symbol_data in symbol_performance.values():
                symbol_data['win_rate'] = (symbol_data['wins'] / symbol_data['trades']) * 100
            
            return {
                'pnl_analysis': {
                    'total_pnl': round(sum(pnl_values), 2),
                    'avg_pnl': round(statistics.mean(pnl_values), 2),
                    'median_pnl': round(statistics.median(pnl_values), 2),
                    'best_trade': round(max(pnl_values), 2),
                    'worst_trade': round(min(pnl_values), 2),
                    'avg_winner': round(statistics.mean(winning_trades), 2) if winning_trades else 0,
                    'avg_loser': round(statistics.mean(losing_trades), 2) if losing_trades else 0,
                    'profit_factor': round(sum(winning_trades) / abs(sum(losing_trades)), 2) if losing_trades else float('inf'),
                    'pnl_std_dev': round(statistics.stdev(pnl_values), 2) if len(pnl_values) > 1 else 0
                },
                'trade_metrics': {
                    'total_trades': len(self.trades),
                    'winning_trades': len(winning_trades),
                    'losing_trades': len(losing_trades),
                    'win_rate': round((len(winning_trades) / len(self.trades)) * 100, 1),
                    'avg_hold_time': round(statistics.mean(hold_times), 1),
                    'median_hold_time': round(statistics.median(hold_times), 1),
                    'longest_trade': round(max(hold_times), 1),
                    'shortest_trade': round(min(hold_times), 1)
                },
                'symbol_performance': symbol_performance,
                'monthly_pnl': self._calculate_monthly_pnl(),
                'drawdown_analysis': self._analyze_drawdowns()
            }
            
        except Exception as e:
            logger.error(f"Trade analytics error: {e}")
            return {'error': str(e)}
    
    def _calculate_monthly_pnl(self) -> Dict:
        """Calculate P&L by month"""
        try:
            monthly_pnl = {}
            
            for trade in self.trades:
                month_key = trade['exit_time'].strftime('%Y-%m')
                if month_key not in monthly_pnl:
                    monthly_pnl[month_key] = 0
                
                monthly_pnl[month_key] += trade['realized_pnl']
            
            return {month: round(pnl, 2) for month, pnl in monthly_pnl.items()}
            
        except Exception as e:
            logger.debug(f"Monthly P&L calculation error: {e}")
            return {}
    
    def _analyze_drawdowns(self) -> Dict:
        """Analyze drawdown periods"""
        try:
            if len(self.trades) < 2:
                return {}
            
            # Calculate cumulative P&L
            cumulative_pnl = []
            running_total = 0
            
            for trade in self.trades:
                running_total += trade['realized_pnl']
                cumulative_pnl.append(running_total)
            
            # Find drawdown periods
            peak = cumulative_pnl[0]
            max_drawdown = 0
            current_drawdown = 0
            drawdown_periods = []
            
            for i, pnl in enumerate(cumulative_pnl):
                if pnl > peak:
                    peak = pnl
                    current_drawdown = 0
                else:
                    current_drawdown = peak - pnl
                    if current_drawdown > max_drawdown:
                        max_drawdown = current_drawdown
            
            return {
                'max_drawdown': round(max_drawdown, 2),
                'current_drawdown': round(self.performance_metrics['peak_capital'] - self.current_capital, 2),
                'drawdown_percentage': round((max_drawdown / max(peak, 1)) * 100, 2)
            }
            
        except Exception as e:
            logger.debug(f"Drawdown analysis error: {e}")
            return {}
    
    def reset_portfolio(self, new_capital: float = None):
        """Reset portfolio to initial state"""
        try:
            if new_capital:
                self.initial_capital = new_capital
            
            self.current_capital = self.initial_capital
            self.available_margin = self.initial_capital
            
            # Clear all data
            self.orders.clear()
            self.positions.clear()
            self.trades.clear()
            self.market_data_cache.clear()
            self.last_market_update.clear()
            
            # Reset performance metrics
            self.performance_metrics = {
                'total_trades': 0,
                'winning_trades': 0,
                'total_pnl': 0.0,
                'max_drawdown': 0.0,
                'peak_capital': self.initial_capital,
                'win_rate': 0.0,
                'avg_trade_pnl': 0.0,
                'sharpe_ratio': 0.0
            }
            
            logger.info(f"✅ Portfolio reset with capital: ₹{self.initial_capital:,.2f}")
            
        except Exception as e:
            logger.error(f"Portfolio reset error: {e}")
    
    def export_trading_data(self) -> Dict:
        """Export all trading data for analysis"""
        try:
            return {
                'account_info': {
                    'initial_capital': self.initial_capital,
                    'current_capital': self.current_capital,
                    'available_margin': self.available_margin
                },
                'orders': [asdict(order) for order in self.orders.values()],
                'positions': [asdict(pos) for pos in self.positions.values()],
                'trades': self.trades,
                'performance_metrics': self.performance_metrics,
                'cost_config': self.cost_config,
                'export_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Data export error: {e}")
            return {'error': str(e)}


# Integration with Trading Bot
class PaperTradingMixin:
    """Mixin class to integrate paper trading with main trading bot"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paper_engine = None
        self.paper_mode = False
    
    def enable_paper_trading(self, initial_capital: float = 100000.0):
        """Enable paper trading mode"""
        try:
            self.paper_engine = PaperTradingEngine(self.api, initial_capital)
            self.paper_engine.start_realtime_updates()
            self.paper_mode = True
            
            logger.info(f"✅ Paper trading enabled with ₹{initial_capital:,.2f} capital")
            
        except Exception as e:
            logger.error(f"❌ Paper trading enable error: {e}")
    
    def disable_paper_trading(self):
        """Disable paper trading mode"""
        try:
            if self.paper_engine:
                self.paper_engine.stop_realtime_updates()
            
            self.paper_mode = False
            logger.info("❌ Paper trading disabled")
            
        except Exception as e:
            logger.error(f"❌ Paper trading disable error: {e}")
    
    def place_trade_order(self, symbol: str, side: str, quantity: int, 
                         order_type: str = "MARKET", price: float = None):
        """Place order (paper or live based on mode)"""
        try:
            if self.paper_mode and self.paper_engine:
                # Paper trading
                order_type_enum = OrderType.MARKET if order_type == "MARKET" else OrderType.LIMIT
                order_id = self.paper_engine.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=order_type_enum,
                    price=price
                )
                return {'order_id': order_id, 'mode': 'PAPER'}
            else:
                # Live trading
                response = self.api.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    order_type=order_type
                )
                return {'order_id': response.get('order_id'), 'mode': 'LIVE'}
                
        except Exception as e:
            logger.error(f"❌ Trade order placement error: {e}")
            return None
    
    def get_trading_summary(self) -> Dict:
        """Get trading summary (paper or live)"""
        try:
            if self.paper_mode and self.paper_engine:
                return {
                    'mode': 'PAPER',
                    'portfolio': self.paper_engine.get_portfolio_summary(),
                    'analytics': self.paper_engine.get_trade_analytics()
                }
            else:
                # Live trading summary
                return {
                    'mode': 'LIVE',
                    'positions': self.api.get_positions(),
                    'orders': self.api.get_orders()
                }
                
        except Exception as e:
            logger.error(f"❌ Trading summary error: {e}")
            return {'error': str(e)}


# Flask Routes for Paper Trading Dashboard
def add_paper_trading_routes(app, trading_bot):
    """Add paper trading routes to Flask application"""
    
    @app.route('/api/paper-trading/enable', methods=['POST'])
    def api_enable_paper_trading():
        try:
            data = request.get_json()
            capital = data.get('capital', 100000.0)
            
            trading_bot.enable_paper_trading(capital)
            
            return jsonify({
                'success': True,
                'message': f'Paper trading enabled with ₹{capital:,.2f}',
                'mode': 'PAPER'
            })
            
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    
    @app.route('/api/paper-trading/disable', methods=['POST'])
    def api_disable_paper_trading():
        try:
            trading_bot.disable_paper_trading()
            
            return jsonify({
                'success': True,
                'message': 'Paper trading disabled',
                'mode': 'LIVE'
            })
            
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    
    @app.route('/api/paper-trading/summary')
    def api_paper_trading_summary():
        try:
            if not trading_bot.paper_mode or not trading_bot.paper_engine:
                return jsonify({'error': 'Paper trading not enabled'})
            
            summary = trading_bot.paper_engine.get_portfolio_summary()
            return jsonify(summary)
            
        except Exception as e:
            return jsonify({'error': str(e)})
    
    @app.route('/api/paper-trading/analytics')
    def api_paper_trading_analytics():
        try:
            if not trading_bot.paper_mode or not trading_bot.paper_engine:
                return jsonify({'error': 'Paper trading not enabled'})
            
            analytics = trading_bot.paper_engine.get_trade_analytics()
            return jsonify(analytics)
            
        except Exception as e:
            return jsonify({'error': str(e)})
    
    @app.route('/api/paper-trading/close-position/<symbol>', methods=['POST'])
    def api_close_paper_position(symbol):
        try:
            if not trading_bot.paper_mode or not trading_bot.paper_engine:
                return jsonify({'success': False, 'message': 'Paper trading not enabled'})
            
            order_id = trading_bot.paper_engine.close_position(symbol)
            
            if order_id:
                return jsonify({
                    'success': True,
                    'message': f'Position closing order placed for {symbol}',
                    'order_id': order_id
                })
            else:
                return jsonify({
                    'success': False,
                    'message': f'Failed to close position for {symbol}'
                })
                
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    
    @app.route('/api/paper-trading/reset', methods=['POST'])
    def api_reset_paper_portfolio():
        try:
            if not trading_bot.paper_mode or not trading_bot.paper_engine:
                return jsonify({'success': False, 'message': 'Paper trading not enabled'})
            
            data = request.get_json()
            new_capital = data.get('capital', 100000.0)
            
            trading_bot.paper_engine.reset_portfolio(new_capital)
            
            return jsonify({
                'success': True,
                'message': f'Portfolio reset with ₹{new_capital:,.2f}',
                'new_capital': new_capital
            })
            
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})

# Test function
def test_paper_trading():
    """Test paper trading system"""
    print("Testing Paper Trading System...")
    
    # Mock API for testing
    class MockAPI:
        def get_quote(self, symbol):
            import random
            base_prices = {'RELIANCE': 2500, 'TCS': 3500, 'NIFTY': 19500}
            base = base_prices.get(symbol, 1000)
            return {
                'ltp': base + random.uniform(-50, 50),
                'high': base + 100,
                'low': base - 100,
                'volume': random.randint(100000, 1000000),
                'prev_close': base
            }
    
    # Test paper trading engine
    mock_api = MockAPI()
    engine = PaperTradingEngine(mock_api, 100000.0)
    engine.start_realtime_updates()
    
    # Test order placement
    order_id = engine.place_order('RELIANCE', 'BUY', 10, OrderType.MARKET)
    print(f"✅ Market order placed: {order_id}")
    
    # Wait for execution
    time.sleep(1)
    
    # Get portfolio summary
    summary = engine.get_portfolio_summary()
    print(f"✅ Portfolio summary: {summary['account_summary']}")
    
    # Close position
    close_order = engine.close_position('RELIANCE')
    print(f"✅ Position closed: {close_order}")
    
    # Wait and get final summary
    time.sleep(1)
    final_summary = engine.get_portfolio_summary()
    print(f"✅ Final summary: {final_summary['account_summary']}")
    
    engine.stop_realtime_updates()
    print("🎉 Paper trading test completed!")

if __name__ == "__main__":
    test_paper_trading()
                  
