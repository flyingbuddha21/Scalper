#!/usr/bin/env python3
"""
Advanced Trading System Utilities
Complete utility functions with integrated dynamic risk management,
stop-loss logic, and adaptive exit algorithms.

This file provides all utility functions used across the 18-file system,
with built-in risk management logic wherever applicable.
"""

import logging
import json
import uuid
import hashlib
import hmac
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import deque

logger = logging.getLogger(__name__)

# ==========================================
# RISK MANAGEMENT ENUMS AND CONSTANTS
# ==========================================

class RiskLevel(Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"
    CRITICAL = "CRITICAL"

class MarketCondition(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    TRENDING = "TRENDING"

class ExitReason(Enum):
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"
    TIME_EXIT = "TIME_EXIT"
    MOMENTUM_DECLINE = "MOMENTUM_DECLINE"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    PROFIT_PROTECTION = "PROFIT_PROTECTION"
    RISK_LIMIT = "RISK_LIMIT"
    MARKET_CLOSE = "MARKET_CLOSE"
    MANUAL_EXIT = "MANUAL_EXIT"

# Risk management constants
RISK_CONSTANTS = {
    'MAX_POSITION_RISK_PCT': 2.0,           # Max 2% risk per position
    'MAX_PORTFOLIO_RISK_PCT': 10.0,         # Max 10% portfolio risk
    'DEFAULT_STOP_LOSS_PCT': 1.5,           # Default 1.5% stop loss
    'MAX_STOP_LOSS_PCT': 5.0,               # Maximum 5% stop loss
    'MIN_RISK_REWARD_RATIO': 1.5,           # Minimum 1.5:1 risk/reward
    'TRAILING_STOP_ACTIVATION_PCT': 2.0,    # Activate trailing at 2% profit
    'MOMENTUM_EXIT_THRESHOLD': 3,            # Exit after 3 declining bars
    'TIME_EXIT_MINUTES': 30,                # Max holding time for scalping
    'VOLATILITY_EXIT_MULTIPLIER': 2.0,      # Exit if volatility 2x normal
}

# ==========================================
# TECHNICAL ANALYSIS WITH RISK INTEGRATION
# ==========================================

def calculate_rsi(prices: List[float], period: int = 14, risk_adjusted: bool = True) -> float:
    """
    Calculate RSI with optional risk adjustment for dynamic stop loss
    """
    try:
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        # Calculate price changes
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Separate gains and losses
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        # Calculate average gains and losses
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Risk adjustment: Modify RSI based on market volatility
        if risk_adjusted:
            volatility = calculate_volatility(prices[-period:]) if len(prices) >= period else 0.02
            
            # In high volatility, make RSI more conservative
            if volatility > 0.03:  # High volatility (>3%)
                if rsi > 50:
                    rsi = min(rsi * 0.9, 100)  # Reduce overbought signals
                else:
                    rsi = max(rsi * 1.1, 0)    # Enhance oversold signals
        
        return max(0, min(100, rsi))
        
    except Exception as e:
        logger.error(f"❌ RSI calculation error: {e}")
        return 50.0

def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
    """
    Calculate MACD with risk-adjusted signals
    """
    try:
        if len(prices) < slow:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0, 'risk_signal': 'NEUTRAL'}
        
        # Calculate EMAs
        ema_fast = calculate_ema(prices, fast)
        ema_slow = calculate_ema(prices, slow)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line (EMA of MACD)
        macd_values = [macd_line]  # Simplified - would need historical MACD values
        signal_line = calculate_ema(macd_values, signal) if len(macd_values) >= signal else macd_line
        
        # Histogram
        histogram = macd_line - signal_line
        
        # Risk-adjusted signal generation
        risk_signal = "NEUTRAL"
        volatility = calculate_volatility(prices[-20:]) if len(prices) >= 20 else 0.02
        
        # Adjust MACD sensitivity based on volatility
        volatility_multiplier = 1 + volatility * 10  # Scale sensitivity
        
        if histogram > 0.001 * volatility_multiplier:
            risk_signal = "BULLISH"
        elif histogram < -0.001 * volatility_multiplier:
            risk_signal = "BEARISH"
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram,
            'risk_signal': risk_signal,
            'volatility_adjusted': True
        }
        
    except Exception as e:
        logger.error(f"❌ MACD calculation error: {e}")
        return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0, 'risk_signal': 'NEUTRAL'}

def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2) -> Dict[str, float]:
    """
    Calculate Bollinger Bands with dynamic stop loss integration
    """
    try:
        if len(prices) < period:
            current_price = prices[-1] if prices else 100
            return {
                'upper': current_price * 1.02,
                'middle': current_price,
                'lower': current_price * 0.98,
                'width': 0.04,
                'position': 0.5
            }
        
        # Calculate SMA and standard deviation
        recent_prices = prices[-period:]
        sma = sum(recent_prices) / len(recent_prices)
        variance = sum((price - sma) ** 2 for price in recent_prices) / len(recent_prices)
        std = math.sqrt(variance)
        
        # Bollinger Bands
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        current_price = prices[-1]
        
        # Band width (volatility measure)
        band_width = (upper_band - lower_band) / sma
        
        # Price position within bands (0 = lower band, 1 = upper band)
        if upper_band != lower_band:
            position = (current_price - lower_band) / (upper_band - lower_band)
        else:
            position = 0.5
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band,
            'width': band_width,
            'position': max(0, min(1, position)),
            'squeeze': band_width < 0.02,  # Band squeeze indicator
            'breakout_direction': 'UP' if position > 0.8 else 'DOWN' if position < 0.2 else 'NONE'
        }
        
    except Exception as e:
        logger.error(f"❌ Bollinger Bands calculation error: {e}")
        current_price = prices[-1] if prices else 100
        return {
            'upper': current_price * 1.02,
            'middle': current_price,
            'lower': current_price * 0.98,
            'width': 0.04,
            'position': 0.5
        }

def calculate_ema(prices: List[float], period: int) -> float:
    """
    Calculate Exponential Moving Average
    """
    try:
        if not prices or len(prices) < period:
            return prices[-1] if prices else 0.0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]  # Start with first price
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
        
    except Exception as e:
        logger.error(f"❌ EMA calculation error: {e}")
        return prices[-1] if prices else 0.0

def calculate_volatility(prices: List[float], period: int = 20) -> float:
    """
    Calculate price volatility (standard deviation of returns)
    """
    try:
        if len(prices) < 2:
            return 0.02  # Default 2% volatility
        
        # Calculate returns
        returns = []
        for i in range(1, min(len(prices), period + 1)):
            if prices[i-1] != 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if not returns:
            return 0.02
        
        # Calculate standard deviation
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        volatility = math.sqrt(variance)
        
        return max(0.001, min(0.2, volatility))  # Cap between 0.1% and 20%
        
    except Exception as e:
        logger.error(f"❌ Volatility calculation error: {e}")
        return 0.02

def calculate_momentum(prices: List[float], period: int = 10) -> float:
    """
    Calculate price momentum for adaptive exit logic
    """
    try:
        if len(prices) < period:
            return 0.0
        
        # Momentum = (Current Price - Price N periods ago) / Price N periods ago
        current_price = prices[-1]
        past_price = prices[-period]
        
        if past_price == 0:
            return 0.0
        
        momentum = (current_price - past_price) / past_price
        return momentum
        
    except Exception as e:
        logger.error(f"❌ Momentum calculation error: {e}")
        return 0.0

# ==========================================
# DYNAMIC RISK MANAGEMENT FUNCTIONS
# ==========================================

def calculate_position_risk(entry_price: float, current_price: float, stop_loss: float, 
                          quantity: int, account_value: float) -> Dict[str, float]:
    """
    Calculate comprehensive position risk metrics
    """
    try:
        position_value = quantity * current_price
        position_size_pct = (position_value / account_value) * 100 if account_value > 0 else 0
        
        # Risk amount (loss if stop hit)
        if stop_loss > 0:
            risk_per_share = abs(entry_price - stop_loss)
            total_risk = risk_per_share * quantity
            risk_pct = (total_risk / account_value) * 100 if account_value > 0 else 0
        else:
            total_risk = 0
            risk_pct = 0
        
        # Unrealized P&L
        if entry_price > 0:
            unrealized_pnl = (current_price - entry_price) * quantity
            unrealized_pnl_pct = (unrealized_pnl / (entry_price * quantity)) * 100
        else:
            unrealized_pnl = 0
            unrealized_pnl_pct = 0
        
        # Risk level assessment
        if risk_pct > 5:
            risk_level = RiskLevel.CRITICAL
        elif risk_pct > 3:
            risk_level = RiskLevel.VERY_HIGH
        elif risk_pct > 2:
            risk_level = RiskLevel.HIGH
        elif risk_pct > 1:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.LOW
        
        return {
            'position_value': position_value,
            'position_size_pct': position_size_pct,
            'risk_amount': total_risk,
            'risk_pct': risk_pct,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'risk_level': risk_level.value,
            'risk_score': min(100, risk_pct * 20)  # Scale to 0-100
        }
        
    except Exception as e:
        logger.error(f"❌ Position risk calculation error: {e}")
        return {
            'position_value': 0,
            'position_size_pct': 0,
            'risk_amount': 0,
            'risk_pct': 0,
            'unrealized_pnl': 0,
            'unrealized_pnl_pct': 0,
            'risk_level': RiskLevel.LOW.value,
            'risk_score': 0
        }

def calculate_dynamic_stop_loss(entry_price: float, current_price: float, side: str,
                              volatility: float = 0.02, profit_pct: float = 0,
                              bars_held: int = 0) -> Dict[str, float]:
    """
    Calculate dynamic stop loss based on multiple factors
    """
    try:
        base_stop_pct = RISK_CONSTANTS['DEFAULT_STOP_LOSS_PCT'] / 100
        
        # Volatility adjustment
        volatility_multiplier = max(0.5, min(2.0, 1 + (volatility - 0.02) * 10))
        
        # Time decay adjustment (wider stops for longer holds)
        time_multiplier = min(1.5, 1 + (bars_held / 100))
        
        # Profit-based adjustment (tighter stops in profit)
        if profit_pct > 5:
            profit_multiplier = 0.5  # Very tight stop in high profit
        elif profit_pct > 2:
            profit_multiplier = 0.7  # Tight stop in profit
        elif profit_pct > 0:
            profit_multiplier = 0.85  # Slightly tighter
        else:
            profit_multiplier = 1.0  # Normal stop
        
        # Calculate adjusted stop distance
        adjusted_stop_pct = base_stop_pct * volatility_multiplier * time_multiplier * profit_multiplier
        adjusted_stop_pct = max(0.005, min(0.05, adjusted_stop_pct))  # Cap between 0.5% and 5%
        
        # Calculate stop price
        if side.upper() == 'BUY':
            stop_price = current_price * (1 - adjusted_stop_pct)
        else:  # SELL
            stop_price = current_price * (1 + adjusted_stop_pct)
        
        # Trailing stop logic
        trailing_activated = profit_pct > RISK_CONSTANTS['TRAILING_STOP_ACTIVATION_PCT']
        
        return {
            'stop_price': stop_price,
            'stop_distance_pct': adjusted_stop_pct * 100,
            'volatility_multiplier': volatility_multiplier,
            'time_multiplier': time_multiplier,
            'profit_multiplier': profit_multiplier,
            'trailing_activated': trailing_activated,
            'stop_type': 'TRAILING' if trailing_activated else 'FIXED'
        }
        
    except Exception as e:
        logger.error(f"❌ Dynamic stop loss calculation error: {e}")
        fallback_stop = entry_price * 0.98 if side.upper() == 'BUY' else entry_price * 1.02
        return {
            'stop_price': fallback_stop,
            'stop_distance_pct': 2.0,
            'volatility_multiplier': 1.0,
            'time_multiplier': 1.0,
            'profit_multiplier': 1.0,
            'trailing_activated': False,
            'stop_type': 'FIXED'
        }

def calculate_adaptive_exit_signals(prices: List[float], volumes: List[int], 
                                  entry_price: float, side: str,
                                  bars_held: int = 0) -> Dict[str, Any]:
    """
    Calculate adaptive exit signals based on multiple factors
    """
    try:
        if len(prices) < 5:
            return {'should_exit': False, 'reasons': [], 'urgency': 'LOW'}
        
        exit_reasons = []
        urgency = 'LOW'
        current_price = prices[-1]
        
        # 1. Momentum deterioration
        momentum = calculate_momentum(prices, period=5)
        if side.upper() == 'BUY' and momentum < -0.01:  # 1% negative momentum
            exit_reasons.append('MOMENTUM_DECLINE')
            urgency = 'MEDIUM'
        elif side.upper() == 'SELL' and momentum > 0.01:  # 1% positive momentum
            exit_reasons.append('MOMENTUM_DECLINE')
            urgency = 'MEDIUM'
        
        # 2. Time-based exit (scalping)
        if bars_held > RISK_CONSTANTS['TIME_EXIT_MINUTES']:
            exit_reasons.append('TIME_LIMIT')
            if urgency == 'LOW':
                urgency = 'LOW'
        
        # 3. Volatility spike
        volatility = calculate_volatility(prices)
        avg_volatility = 0.02  # Would be calculated from historical data
        if volatility > avg_volatility * RISK_CONSTANTS['VOLATILITY_EXIT_MULTIPLIER']:
            exit_reasons.append('VOLATILITY_SPIKE')
            urgency = 'HIGH'
        
        # 4. Volume decline (if available)
        if volumes and len(volumes) >= 3:
            recent_volume = sum(volumes[-3:]) / 3
            historical_volume = sum(volumes[:-3]) / max(1, len(volumes) - 3) if len(volumes) > 3 else recent_volume
            
            if recent_volume < historical_volume * 0.5:  # 50% volume decline
                exit_reasons.append('VOLUME_DECLINE')
        
        # 5. Profit protection
        if entry_price > 0:
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            if side.upper() == 'SELL':
                profit_pct = -profit_pct
            
            # Calculate max favorable move
            if side.upper() == 'BUY':
                max_price = max(prices[-min(10, len(prices)):])
                max_favorable_pct = ((max_price - entry_price) / entry_price) * 100
            else:
                min_price = min(prices[-min(10, len(prices)):])
                max_favorable_pct = ((entry_price - min_price) / entry_price) * 100
            
            # Profit protection trigger
            if profit_pct > 3 and max_favorable_pct > 5:  # 3% current profit, 5% max move
                if profit_pct < max_favorable_pct * 0.6:  # Profit dropped below 60% of max
                    exit_reasons.append('PROFIT_PROTECTION')
                    urgency = 'MEDIUM'
        
        should_exit = len(exit_reasons) > 0
        
        # Adjust urgency based on number of signals
        if len(exit_reasons) >= 3:
            urgency = 'HIGH'
        elif len(exit_reasons) >= 2:
            urgency = 'MEDIUM'
        
        return {
            'should_exit': should_exit,
            'reasons': exit_reasons,
            'urgency': urgency,
            'exit_score': len(exit_reasons) * 25,  # Scale to 0-100
            'momentum': momentum,
            'volatility': volatility,
            'bars_held': bars_held
        }
        
    except Exception as e:
        logger.error(f"❌ Adaptive exit signals calculation error: {e}")
        return {
            'should_exit': False,
            'reasons': [],
            'urgency': 'LOW',
            'exit_score': 0,
            'momentum': 0,
            'volatility': 0.02,
            'bars_held': bars_held
        }

def calculate_position_size(account_value: float, risk_amount: float, entry_price: float,
                          stop_loss: float, max_position_pct: float = 10.0) -> int:
    """
    Calculate optimal position size based on risk parameters
    """
    try:
        if entry_price <= 0 or stop_loss <= 0 or account_value <= 0:
            return 0
        
        # Risk-based sizing
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0
        
        max_shares_by_risk = int(risk_amount / risk_per_share)
        
        # Position size limit
        max_position_value = account_value * (max_position_pct / 100)
        max_shares_by_size = int(max_position_value / entry_price)
        
        # Take the smaller of the two
        position_size = min(max_shares_by_risk, max_shares_by_size)
        position_size = max(1, position_size)  # Minimum 1 share
        
        return position_size
        
    except Exception as e:
        logger.error(f"❌ Position size calculation error: {e}")
        return 1

# ==========================================
# PORTFOLIO RISK METRICS
# ==========================================

def calculate_portfolio_metrics(positions: List[Dict], account_value: float) -> Dict[str, float]:
    """
    Calculate comprehensive portfolio risk metrics
    """
    try:
        if not positions or account_value <= 0:
            return {
                'total_exposure': 0,
                'net_exposure': 0,
                'long_exposure': 0,
                'short_exposure': 0,
                'portfolio_risk_pct': 0,
                'diversification_score': 100,
                'correlation_risk': 0,
                'concentration_risk': 0
            }
        
        total_long_value = 0
        total_short_value = 0
        total_risk_amount = 0
        position_values = []
        
        for position in positions:
            current_value = position.get('quantity', 0) * position.get('current_price', 0)
            position_values.append(current_value)
            
            if position.get('side') == 'BUY':
                total_long_value += current_value
            else:
                total_short_value += current_value
            
            # Calculate risk for this position
            risk_info = calculate_position_risk(
                entry_price=position.get('entry_price', 0),
                current_price=position.get('current_price', 0),
                stop_loss=position.get('stop_loss', 0),
                quantity=position.get('quantity', 0),
                account_value=account_value
            )
            total_risk_amount += risk_info['risk_amount']
        
        total_exposure = total_long_value + total_short_value
        net_exposure = total_long_value - total_short_value
        
        # Portfolio risk percentage
        portfolio_risk_pct = (total_risk_amount / account_value) * 100 if account_value > 0 else 0
        
        # Concentration risk (largest position as % of portfolio)
        if position_values:
            max_position_value = max(position_values)
            concentration_risk = (max_position_value / account_value) * 100 if account_value > 0 else 0
        else:
            concentration_risk = 0
        
        # Diversification score (simple measure based on number of positions)
        num_positions = len(positions)
        if num_positions >= 5:
            diversification_score = 100
        elif num_positions >= 3:
            diversification_score = 80
        elif num_positions >= 2:
            diversification_score = 60
        else:
            diversification_score = 40
        
        return {
            'total_exposure': total_exposure,
            'net_exposure': net_exposure,
            'long_exposure': total_long_value,
            'short_exposure': total_short_value,
            'portfolio_risk_pct': portfolio_risk_pct,
            'diversification_score': diversification_score,
            'correlation_risk': 0,  # Would need correlation analysis
            'concentration_risk': concentration_risk,
            'total_risk_amount': total_risk_amount,
            'exposure_ratio': (total_exposure / account_value) * 100 if account_value > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"❌ Portfolio metrics calculation error: {e}")
        return {
            'total_exposure': 0,
            'net_exposure': 0,
            'long_exposure': 0,
            'short_exposure': 0,
            'portfolio_risk_pct': 0,
            'diversification_score': 0,
            'correlation_risk': 0,
            'concentration_risk': 0
        }

# ==========================================
# DATA VALIDATION AND QUALITY
# ==========================================

def validate_market_data(data: Dict) -> Dict[str, Any]:
    """
    Validate market data quality with risk implications
    """
    try:
        issues = []
        quality_score = 100
        
        required_fields = ['symbol', 'price', 'timestamp']
        for field in required_fields:
            if field not in data or data[field] is None:
                issues.append(f"Missing {field}")
                quality_score -= 20
        
        # Price validation
        price = data.get('price', 0)
        if price <= 0:
            issues.append("Invalid price (<= 0)")
            quality_score -= 30
        
        # Volume validation
        volume = data.get('volume', 0)
        if volume < 0:
            issues.append("Negative volume")
            quality_score -= 10
        
        # Timestamp validation
        timestamp = data.get('timestamp')
        if timestamp:
            try:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                # Check if data is too old (more than 1 minute for real-time)
                age_seconds = (datetime.now() - timestamp.replace(tzinfo=None)).total_seconds()
                if age_seconds > 60:
                    issues.append(f"Stale data ({age_seconds:.0f}s old)")
                    quality_score -= 15
                    
            except Exception:
                issues.append("Invalid timestamp format")
                quality_score -= 10
        
        # Price change validation (detect potential errors)
        high = data.get('high', price)
        low = data.get('low', price)
        
        if high < low:
            issues.append("High < Low (data error)")
            quality_score -= 25
        
        if price > high * 1.1 or price < low * 0.9:
            issues.append("Price outside high/low range")
            quality_score -= 20
        
        # Calculate quality level
        if quality_score >= 90:
            quality_level = "EXCELLENT"
        elif quality_score >= 75:
            quality_level = "GOOD"
        elif quality_score >= 50:
            quality_level = "FAIR"
        else:
            quality_level = "POOR"
        
        return {
            'is_valid': len(issues) == 0,
            'quality_score': max(0, quality_score),
            'quality_level': quality_level,
            'issues': issues,
            'risk_implications': 'HIGH' if quality_score < 50 else 'MEDIUM' if quality_score < 75 else 'LOW'
        }
        
    except Exception as e:
        logger.error(f"❌ Market data validation error: {e}")
        return {
            'is_valid': False,
            'quality_score': 0,
            'quality_level': "INVALID",
            'issues': [f"Validation error: {str(e)}"],
            'risk_implications': 'CRITICAL'
        }

# ==========================================
# FORMATTING AND DISPLAY UTILITIES
# ==========================================

def format_currency(amount: float, currency: str = "INR", decimal_places: int = 2) -> str:
    """Format currency with proper Indian number formatting"""
    try:
        if currency == "INR":
            # Indian number formatting (Lakh, Crore)
            if abs(amount) >= 10000000:  # 1 Crore
                return f"₹{amount/10000000:.{decimal_places}f} Cr"
            elif abs(amount) >= 100000:  # 1 Lakh
                return f"₹{amount/100000:.{decimal_places}f} L"
            elif abs(amount) >= 1000:   # 1 Thousand
                return f"₹{amount/1000:.{decimal_places}f}k"
            else:
                return f"₹{amount:.{decimal_places}f}"
        else:
            return f"{amount:,.{decimal_places}f}"
            
    except Exception as e:
        logger.error(f"❌ Currency formatting error: {e}")
        return f"₹{amount:.2f}"

def format_percentage(value: float, decimal_places: int = 2, show_sign: bool = True) -> str:
    """Format percentage with proper sign and color coding"""
    try:
        formatted = f"{value:.{decimal_places}f}%"
        
        if show_sign:
            if value > 0:
                formatted = f"+{formatted}"
            # Negative sign is automatic
        
        return formatted
        
    except Exception as e:
        logger.error(f"❌ Percentage formatting error: {e}")
        return f"{value:.2f}%"

def format_datetime(dt: datetime, format_type: str = "standard") -> str:
    """Format datetime for different display purposes"""
    try:
        if format_type == "standard":
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        elif format_type == "time_only":
            return dt.strftime("%H:%M:%S")
        elif format_type == "date_only":
            return dt.strftime("%Y-%m-%d")
        elif format_type == "compact":
            return dt.strftime("%m/%d %H:%M")
        elif format_type == "log":
            return dt.strftime("%Y%m%d_%H%M%S")
        else:
            return dt.isoformat()
            
    except Exception as e:
        logger.error(f"❌ Datetime formatting error: {e}")
        return str(dt)

def format_large_number(number: float, compact: bool = True) -> str:
    """Format large numbers with appropriate suffixes"""
    try:
        if compact:
            if abs(number) >= 1e9:
                return f"{number/1e9:.1f}B"
            elif abs(number) >= 1e6:
                return f"{number/1e6:.1f}M"
            elif abs(number) >= 1e3:
                return f"{number/1e3:.1f}K"
            else:
                return f"{number:.0f}"
        else:
            return f"{number:,.0f}"
            
    except Exception as e:
        logger.error(f"❌ Large number formatting error: {e}")
        return str(number)

# ==========================================
# UNIQUE ID AND HASH GENERATION
# ==========================================

def generate_order_id(prefix: str = "ORD") -> str:
    """Generate unique order ID"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_part = uuid.uuid4().hex[:8].upper()
        return f"{prefix}_{timestamp}_{unique_part}"
        
    except Exception as e:
        logger.error(f"❌ Order ID generation error: {e}")
        return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:6]}"

def generate_session_id() -> str:
    """Generate unique trading session ID"""
    try:
        return f"SESSION_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6].upper()}"
        
    except Exception as e:
        logger.error(f"❌ Session ID generation error: {e}")
        return f"SESSION_{int(time.time())}"

def generate_signal_id() -> str:
    """Generate unique signal ID"""
    try:
        return f"SIG_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:4].upper()}"
        
    except Exception as e:
        logger.error(f"❌ Signal ID generation error: {e}")
        return f"SIG_{int(time.time())}"

def generate_secure_hash(data: str, secret_key: str = "") -> str:
    """Generate secure hash for API authentication"""
    try:
        if secret_key:
            # HMAC hash with secret key
            return hmac.new(
                secret_key.encode('utf-8'),
                data.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
        else:
            # Simple SHA256 hash
            return hashlib.sha256(data.encode('utf-8')).hexdigest()
            
    except Exception as e:
        logger.error(f"❌ Hash generation error: {e}")
        return hashlib.md5(str(time.time()).encode()).hexdigest()

# ==========================================
# TIME AND MARKET UTILITIES
# ==========================================

def is_market_open(current_time: datetime = None, timezone: str = "Asia/Kolkata") -> bool:
    """Check if Indian market is currently open"""
    try:
        if current_time is None:
            current_time = datetime.now()
        
        # Check day of week (Monday=0, Sunday=6)
        if current_time.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Market hours: 9:15 AM to 3:30 PM IST
        market_open = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= current_time <= market_close
        
    except Exception as e:
        logger.error(f"❌ Market open check error: {e}")
        return False

def is_pre_market(current_time: datetime = None) -> bool:
    """Check if it's pre-market hours"""
    try:
        if current_time is None:
            current_time = datetime.now()
        
        if current_time.weekday() >= 5:
            return False
        
        pre_market_start = current_time.replace(hour=8, minute=15, second=0, microsecond=0)
        market_open = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
        
        return pre_market_start <= current_time < market_open
        
    except Exception as e:
        logger.error(f"❌ Pre-market check error: {e}")
        return False

def is_post_market(current_time: datetime = None) -> bool:
    """Check if it's post-market hours"""
    try:
        if current_time is None:
            current_time = datetime.now()
        
        if current_time.weekday() >= 5:
            return False
        
        market_close = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
        post_market_end = current_time.replace(hour=16, minute=30, second=0, microsecond=0)
        
        return market_close < current_time <= post_market_end
        
    except Exception as e:
        logger.error(f"❌ Post-market check error: {e}")
        return False

def get_next_market_open() -> datetime:
    """Get next market opening time"""
    try:
        now = datetime.now()
        
        # If it's weekend, next open is Monday
        if now.weekday() >= 5:  # Saturday or Sunday
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0:  # It's Sunday
                days_until_monday = 1
            next_open = now + timedelta(days=days_until_monday)
        else:
            # If market is closed today, next open is today at 9:15 AM
            market_open_today = now.replace(hour=9, minute=15, second=0, microsecond=0)
            
            if now < market_open_today:
                next_open = market_open_today
            else:
                # Next open is tomorrow (or Monday if it's Friday)
                if now.weekday() == 4:  # Friday
                    next_open = now + timedelta(days=3)  # Monday
                else:
                    next_open = now + timedelta(days=1)  # Tomorrow
                
                next_open = next_open.replace(hour=9, minute=15, second=0, microsecond=0)
        
        return next_open
        
    except Exception as e:
        logger.error(f"❌ Next market open calculation error: {e}")
        return datetime.now() + timedelta(hours=24)

def get_trading_days_between(start_date: datetime, end_date: datetime) -> int:
    """Count trading days between two dates (excluding weekends)"""
    try:
        current_date = start_date.date()
        end_date = end_date.date()
        trading_days = 0
        
        while current_date <= end_date:
            # Monday=0, Sunday=6
            if current_date.weekday() < 5:  # Weekday
                trading_days += 1
            current_date += timedelta(days=1)
        
        return trading_days
        
    except Exception as e:
        logger.error(f"❌ Trading days calculation error: {e}")
        return 0

# ==========================================
# DATA STRUCTURE UTILITIES
# ==========================================

def safe_get(data: Dict, key: str, default: Any = None, data_type: type = None) -> Any:
    """Safely get value from dictionary with type conversion"""
    try:
        value = data.get(key, default)
        
        if value is None:
            return default
        
        if data_type:
            if data_type == float:
                return float(value)
            elif data_type == int:
                return int(float(value))  # Handle string numbers
            elif data_type == str:
                return str(value)
            elif data_type == bool:
                if isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes', 'on')
                return bool(value)
        
        return value
        
    except Exception as e:
        logger.error(f"❌ Safe get error for key '{key}': {e}")
        return default

def merge_dictionaries(*dicts: Dict, deep: bool = True) -> Dict:
    """Merge multiple dictionaries with optional deep merge"""
    try:
        result = {}
        
        for d in dicts:
            if not isinstance(d, dict):
                continue
            
            if deep:
                for key, value in d.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = merge_dictionaries(result[key], value, deep=True)
                    else:
                        result[key] = value
            else:
                result.update(d)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Dictionary merge error: {e}")
        return {}

def flatten_dictionary(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten nested dictionary"""
    try:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dictionary(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
        
    except Exception as e:
        logger.error(f"❌ Dictionary flatten error: {e}")
        return d

def clean_dictionary(d: Dict, remove_none: bool = True, remove_empty_strings: bool = True) -> Dict:
    """Clean dictionary by removing None values and empty strings"""
    try:
        cleaned = {}
        
        for key, value in d.items():
            if remove_none and value is None:
                continue
            if remove_empty_strings and value == "":
                continue
            
            if isinstance(value, dict):
                cleaned_value = clean_dictionary(value, remove_none, remove_empty_strings)
                if cleaned_value:  # Only include non-empty dictionaries
                    cleaned[key] = cleaned_value
            else:
                cleaned[key] = value
        
        return cleaned
        
    except Exception as e:
        logger.error(f"❌ Dictionary cleaning error: {e}")
        return d

# ==========================================
# MATHEMATICAL UTILITIES
# ==========================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
        
    except Exception as e:
        logger.error(f"❌ Safe divide error: {e}")
        return default

def round_to_tick_size(price: float, tick_size: float = 0.05) -> float:
    """Round price to valid tick size"""
    try:
        if tick_size <= 0:
            return round(price, 2)
        
        return round(price / tick_size) * tick_size
        
    except Exception as e:
        logger.error(f"❌ Tick size rounding error: {e}")
        return round(price, 2)

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    try:
        if old_value == 0:
            return 0.0
        
        return ((new_value - old_value) / old_value) * 100
        
    except Exception as e:
        logger.error(f"❌ Percentage change calculation error: {e}")
        return 0.0

def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """Normalize value to 0-1 range"""
    try:
        if max_val == min_val:
            return 0.5
        
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))
        
    except Exception as e:
        logger.error(f"❌ Value normalization error: {e}")
        return 0.5

def calculate_compound_return(returns: List[float]) -> float:
    """Calculate compound return from list of individual returns"""
    try:
        if not returns:
            return 0.0
        
        compound = 1.0
        for ret in returns:
            compound *= (1 + ret / 100)
        
        return (compound - 1) * 100
        
    except Exception as e:
        logger.error(f"❌ Compound return calculation error: {e}")
        return 0.0

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.06) -> float:
    """Calculate Sharpe ratio for returns"""
    try:
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio
        excess_return = mean_return - risk_free_rate / 252  # Daily risk-free rate
        sharpe = (excess_return / std_return) * np.sqrt(252)
        
        return sharpe
        
    except Exception as e:
        logger.error(f"❌ Sharpe ratio calculation error: {e}")
        return 0.0

def calculate_max_drawdown(values: List[float]) -> float:
    """Calculate maximum drawdown from series of values"""
    try:
        if len(values) < 2:
            return 0.0
        
        peak = values[0]
        max_drawdown = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown * 100  # Return as percentage
        
    except Exception as e:
        logger.error(f"❌ Max drawdown calculation error: {e}")
        return 0.0

# ==========================================
# FILE AND PATH UTILITIES
# ==========================================

def ensure_directory_exists(directory_path: str) -> bool:
    """Ensure directory exists, create if not"""
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
        
    except Exception as e:
        logger.error(f"❌ Directory creation error: {e}")
        return False

def get_file_size(file_path: str) -> int:
    """Get file size in bytes"""
    try:
        return Path(file_path).stat().st_size
        
    except Exception as e:
        logger.error(f"❌ File size error: {e}")
        return 0

def backup_file(file_path: str, backup_dir: str = "backups") -> str:
    """Create backup of file with timestamp"""
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            return ""
        
        # Create backup directory
        backup_path = file_path.parent / backup_dir
        backup_path.mkdir(exist_ok=True)
        
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_file_path = backup_path / backup_filename
        
        # Copy file
        import shutil
        shutil.copy2(file_path, backup_file_path)
        
        return str(backup_file_path)
        
    except Exception as e:
        logger.error(f"❌ File backup error: {e}")
        return ""

def cleanup_old_files(directory: str, max_age_days: int = 30, pattern: str = "*") -> int:
    """Cleanup old files in directory"""
    try:
        directory_path = Path(directory)
        if not directory_path.exists():
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        deleted_count = 0
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file():
                file_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_modified < cutoff_date:
                    file_path.unlink()
                    deleted_count += 1
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"❌ File cleanup error: {e}")
        return 0

# ==========================================
# LOGGING UTILITIES
# ==========================================

def setup_logger(name: str, log_file: str = None, level: str = "INFO") -> logging.Logger:
    """Setup logger with proper formatting"""
    try:
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            ensure_directory_exists(Path(log_file).parent)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
        
    except Exception as e:
        print(f"❌ Logger setup error: {e}")
        return logging.getLogger(name)

def log_performance_metric(metric_name: str, value: float, logger_instance: logging.Logger = None):
    """Log performance metric in structured format"""
    try:
        if logger_instance is None:
            logger_instance = logger
        
        metric_data = {
            'metric': metric_name,
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'type': 'performance'
        }
        
        logger_instance.info(f"METRIC: {json.dumps(metric_data)}")
        
    except Exception as e:
        logger.error(f"❌ Performance metric logging error: {e}")

# ==========================================
# ERROR HANDLING UTILITIES
# ==========================================

def handle_api_error(response, operation: str = "API call") -> Dict[str, Any]:
    """Standardized API error handling"""
    try:
        if hasattr(response, 'status_code'):
            if response.status_code == 200:
                return {'success': True, 'data': response.json()}
            elif response.status_code == 401:
                return {'success': False, 'error': 'Authentication failed', 'code': 401}
            elif response.status_code == 403:
                return {'success': False, 'error': 'Access forbidden', 'code': 403}
            elif response.status_code == 429:
                return {'success': False, 'error': 'Rate limit exceeded', 'code': 429}
            elif response.status_code >= 500:
                return {'success': False, 'error': 'Server error', 'code': response.status_code}
            else:
                return {'success': False, 'error': f'{operation} failed', 'code': response.status_code}
        else:
            return {'success': False, 'error': 'Invalid response object'}
            
    except Exception as e:
        logger.error(f"❌ API error handling error: {e}")
        return {'success': False, 'error': f'Error handling failed: {str(e)}'}

def retry_operation(func, max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry operation with exponential backoff"""
    def decorator(*args, **kwargs):
        current_delay = delay
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"❌ Operation failed after {max_retries} retries: {e}")
                    raise
                
                logger.warning(f"⚠️ Attempt {attempt + 1} failed, retrying in {current_delay}s: {e}")
                time.sleep(current_delay)
                current_delay *= backoff
    
    return decorator

# ==========================================
# CONFIGURATION UTILITIES
# ==========================================

def load_json_config(file_path: str, default_config: Dict = None) -> Dict:
    """Load JSON configuration with fallback"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Merge with default config if provided
        if default_config:
            config = merge_dictionaries(default_config, config)
        
        return config
        
    except Exception as e:
        logger.error(f"❌ Config loading error: {e}")
        return default_config or {}

def save_json_config(config: Dict, file_path: str, backup: bool = True) -> bool:
    """Save JSON configuration with optional backup"""
    try:
        # Create backup if requested
        if backup and Path(file_path).exists():
            backup_file(file_path)
        
        # Ensure directory exists
        ensure_directory_exists(Path(file_path).parent)
        
        # Save config
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, default=str)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Config saving error: {e}")
        return False

def validate_config_schema(config: Dict, required_keys: List[str]) -> Dict[str, Any]:
    """Validate configuration against required schema"""
    try:
        errors = []
        warnings = []
        
        # Check required keys
        for key in required_keys:
            if key not in config:
                errors.append(f"Missing required key: {key}")
        
        # Check for empty values
        for key, value in config.items():
            if value is None or value == "":
                warnings.append(f"Empty value for key: {key}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
        
    except Exception as e:
        logger.error(f"❌ Config validation error: {e}")
        return {
            'valid': False,
            'errors': [f"Validation error: {str(e)}"],
            'warnings': []
        }

# ==========================================
# SYSTEM UTILITIES
# ==========================================

def get_system_info() -> Dict[str, Any]:
    """Get system information for monitoring"""
    try:
        import platform
        import psutil
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ System info error: {e}")
        return {
            'platform': 'unknown',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }

def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are available"""
    dependencies = {
        'numpy': False,
        'pandas': False,
        'requests': False,
        'websocket': False,
        'flask': False,
        'sqlalchemy': False,
        'schedule': False
    }
    
    for dependency in dependencies:
        try:
            __import__(dependency)
            dependencies[dependency] = True
        except ImportError:
            dependencies[dependency] = False
    
    return dependencies

# ==========================================
# MODULE INITIALIZATION
# ==========================================

def initialize_utils() -> bool:
    """Initialize utilities module"""
    try:
        # Create necessary directories
        directories = ['logs', 'data', 'config', 'backups', 'reports']
        for directory in directories:
            ensure_directory_exists(directory)
        
        logger.info("✅ Utils module initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Utils initialization error: {e}")
        return False

# Initialize on import
if __name__ != "__main__":
    initialize_utils()

# ==========================================
# USAGE EXAMPLES AND TESTING
# ==========================================

if __name__ == "__main__":
    # Example usage and testing
    print("🔧 Advanced Trading System Utilities")
    print("=" * 50)
    
    # Test technical indicators
    test_prices = [100, 101, 99, 102, 98, 103, 97, 104, 96, 105]
    
    print("📊 Technical Analysis Tests:")
    print(f"RSI: {calculate_rsi(test_prices):.2f}")
    
    macd = calculate_macd(test_prices)
    print(f"MACD: {macd['macd']:.4f}, Signal: {macd['signal']:.4f}")
    
    bb = calculate_bollinger_bands(test_prices)
    print(f"Bollinger Bands: Upper={bb['upper']:.2f}, Lower={bb['lower']:.2f}")
    
    # Test risk calculations
    print("\n💰 Risk Management Tests:")
    risk = calculate_position_risk(100, 105, 98, 100, 100000)
    print(f"Position Risk: {risk['risk_pct']:.2f}% ({risk['risk_level']})")
    
    stop_loss = calculate_dynamic_stop_loss(100, 105, 'BUY', volatility=0.03, profit_pct=5)
    print(f"Dynamic Stop Loss: ₹{stop_loss['stop_price']:.2f} ({stop_loss['stop_type']})")
    
    # Test exit signals
    exit_signals = calculate_adaptive_exit_signals(test_prices, [1000]*10, 100, 'BUY', bars_held=15)
    print(f"Exit Signals: {exit_signals['should_exit']} - {exit_signals['reasons']}")
    
    # Test formatting
    print("\n📄 Formatting Tests:")
    print(f"Currency: {format_currency(123456.78)}")
    print(f"Percentage: {format_percentage(5.67)}")
    print(f"Large Number: {format_large_number(1234567)}")
    
    # Test market timing
    print("\n⏰ Market Timing Tests:")
    print(f"Market Open: {is_market_open()}")
    print(f"Pre-market: {is_pre_market()}")
    print(f"Next Market Open: {get_next_market_open()}")
    
    # Test system info
    print("\n🖥️ System Information:")
    sys_info = get_system_info()
    print(f"Platform: {sys_info['platform']}")
    print(f"Memory Usage: {sys_info.get('memory_percent', 'N/A')}%")
    
    print("\n✅ All utility tests completed!")
