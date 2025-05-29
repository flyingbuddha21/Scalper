#!/usr/bin/env python3
"""
Strategy Manager
Coordinates and manages all trading strategies
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class StrategyManager:
    """Manages and coordinates all trading strategies"""
    
    def __init__(self, config_manager, data_manager):
        self.config_manager = config_manager
        self.data_manager = data_manager
        
        # Strategy management
        self.strategies = {}
        self.strategy_performance = defaultdict(dict)
        self.strategy_signals = defaultdict(list)
        
        # Signal aggregation
        self.signal_weights = {}
        self.signal_buffer = defaultdict(list)
        self.consensus_threshold = 0.7  # 70% consensus required
        
        # Performance tracking
        self.performance_metrics = {
            'total_signals': 0,
            'consensus_signals': 0,
            'executed_signals': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0
        }
        
        # Strategy configuration
        self.strategy_config = self._load_strategy_config()
        
        logger.info("🎯 Strategy Manager initialized")
    
    def _load_strategy_config(self) -> Dict:
        """Load strategy configuration"""
        try:
            return self.config_manager.get_strategy_config()
        except Exception as e:
            logger.error(f"❌ Load strategy config error: {e}")
            return {}
    
    def register_strategy(self, strategy_name: str, strategy_instance: Any, weight: float = 1.0):
        """Register a trading strategy"""
        try:
            self.strategies[strategy_name] = {
                'instance': strategy_instance,
                'weight': weight,
                'enabled': self.strategy_config.get(strategy_name, {}).get('enabled', True),
                'signals_generated': 0,
                'successful_signals': 0,
                'total_pnl': 0.0,
                'last_signal_time': None
            }
            
            self.signal_weights[strategy_name] = weight
            
            logger.info(f"📝 Registered strategy: {strategy_name} (weight: {weight})")
            
        except Exception as e:
            logger.error(f"❌ Register strategy error: {e}")
    
    def unregister_strategy(self, strategy_name: str):
        """Unregister a trading strategy"""
        try:
            if strategy_name in self.strategies:
                del self.strategies[strategy_name]
                del self.signal_weights[strategy_name]
                logger.info(f"🗑️ Unregistered strategy: {strategy_name}")
            
        except Exception as e:
            logger.error(f"❌ Unregister strategy error: {e}")
    
    def enable_strategy(self, strategy_name: str):
        """Enable a strategy"""
        try:
            if strategy_name in self.strategies:
                self.strategies[strategy_name]['enabled'] = True
                self.config_manager.toggle_strategy(strategy_name, True)
                logger.info(f"✅ Enabled strategy: {strategy_name}")
            
        except Exception as e:
            logger.error(f"❌ Enable strategy error: {e}")
    
    def disable_strategy(self, strategy_name: str):
        """Disable a strategy"""
        try:
            if strategy_name in self.strategies:
                self.strategies[strategy_name]['enabled'] = False
                self.config_manager.toggle_strategy(strategy_name, False)
                logger.info(f"❌ Disabled strategy: {strategy_name}")
            
        except Exception as e:
            logger.error(f"❌ Disable strategy error: {e}")
    
    def analyze_symbol(self, symbol: str, market_data: Dict) -> Optional[Dict]:
        """Analyze symbol with all enabled strategies"""
        try:
            signals = []
            strategy_results = {}
            
            for strategy_name, strategy_info in self.strategies.items():
                if not strategy_info['enabled']:
                    continue
                
                try:
                    strategy_instance = strategy_info['instance']
                    
                    # Run strategy analysis
                    signal = strategy_instance.analyze_signal(market_data)
                    
                    if signal:
                        # Convert signal to dict if it's not already
                        if hasattr(signal, 'to_dict'):
                            signal_dict = signal.to_dict()
                        else:
                            signal_dict = signal
                        
                        signal_dict['strategy_name'] = strategy_name
                        signal_dict['weight'] = strategy_info['weight']
                        signals.append(signal_dict)
                        
                        # Update strategy stats
                        strategy_info['signals_generated'] += 1
                        strategy_info['last_signal_time'] = datetime.now()
                        
                        strategy_results[strategy_name] = signal_dict
                    
                except Exception as e:
                    logger.debug(f"Strategy {strategy_name} analysis error: {e}")
                    continue
            
            # Update performance metrics
            self.performance_metrics['total_signals'] += len(signals)
            
            # Store signals for consensus analysis
            if signals:
                self.signal_buffer[symbol].extend(signals)
                self._cleanup_old_signals(symbol)
            
            # Generate consensus signal
            consensus_signal = self._generate_consensus_signal(symbol, signals)
            
            return {
                'symbol': symbol,
                'individual_signals': strategy_results,
                'consensus_signal': consensus_signal,
                'signal_count': len(signals),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Analyze symbol error: {e}")
            return None
    
    def _generate_consensus_signal(self, symbol: str, signals: List[Dict]) -> Optional[Dict]:
        """Generate consensus signal from multiple strategies"""
        try:
            if not signals:
                return None
            
            # Group signals by type
            buy_signals = [s for s in signals if s.get('signal_type') == 'BUY']
            sell_signals = [s for s in signals if s.get('signal_type') == 'SELL']
            
            # Calculate weighted consensus
            buy_weight = sum(s.get('weight', 1.0) * s.get('confidence', 0) for s in buy_signals)
            sell_weight = sum(s.get('weight', 1.0) * s.get('confidence', 0) for s in sell_signals)
            
            total_weight = sum(s.get('weight', 1.0) for s in signals)
            
            if total_weight == 0:
                return None
            
            # Determine consensus
            buy_consensus = buy_weight / total_weight if total_weight > 0 else 0
            sell_consensus = sell_weight / total_weight if total_weight > 0 else 0
            
            # Check if consensus threshold is met
            if buy_consensus >= self.consensus_threshold:
                consensus_signal = {
                    'signal_type': 'BUY',
                    'confidence': buy_consensus,
                    'supporting_strategies': len(buy_signals),
                    'consensus_strength': buy_consensus,
                    'price': self._calculate_consensus_price(buy_signals),
                    'stop_loss': self._calculate_consensus_stop_loss(buy_signals),
                    'take_profit': self._calculate_consensus_take_profit(buy_signals)
                }
            elif sell_consensus >= self.consensus_threshold:
                consensus_signal = {
                    'signal_type': 'SELL',
                    'confidence': sell_consensus,
                    'supporting_strategies': len(sell_signals),
                    'consensus_strength': sell_consensus,
                    'price': self._calculate_consensus_price(sell_signals),
                    'stop_loss': self._calculate_consensus_stop_loss(sell_signals),
                    'take_profit': self._calculate_consensus_take_profit(sell_signals)
                }
            else:
                # No consensus reached
                consensus_signal = {
                    'signal_type': 'NO_CONSENSUS',
                    'confidence': max(buy_consensus, sell_consensus),
                    'buy_strength': buy_consensus,
                    'sell_strength': sell_consensus,
                    'supporting_strategies': len(signals)
                }
            
            if consensus_signal.get('signal_type') in ['BUY', 'SELL']:
                self.performance_metrics['consensus_signals'] += 1
            
            return consensus_signal
            
        except Exception as e:
            logger.error(f"❌ Generate consensus signal error: {e}")
            return None
    
    def _calculate_consensus_price(self, signals: List[Dict]) -> float:
        """Calculate consensus price from signals"""
        try:
            if not signals:
                return 0.0
            
            # Weighted average of prices
            total_weight = sum(s.get('weight', 1.0) for s in signals)
            weighted_price = sum(s.get('price', 0) * s.get('weight', 1.0) for s in signals)
            
            return weighted_price / total_weight if total_weight > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_consensus_stop_loss(self, signals: List[Dict]) -> Optional[float]:
        """Calculate consensus stop loss from signals"""
        try:
            stop_losses = [s.get('stop_loss') for s in signals if s.get('stop_loss')]
            
            if not stop_losses:
                return None
            
            # Use median stop loss
            return float(np.median(stop_losses))
            
        except Exception:
            return None
    
    def _calculate_consensus_take_profit(self, signals: List[Dict]) -> Optional[float]:
        """Calculate consensus take profit from signals"""
        try:
            take_profits = [s.get('take_profit') for s in signals if s.get('take_profit')]
            
            if not take_profits:
                return None
            
            # Use median take profit
            return float(np.median(take_profits))
            
        except Exception:
            return None
    
    def _cleanup_old_signals(self, symbol: str, max_age_minutes: int = 5):
        """Clean up old signals from buffer"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
            
            # Keep only recent signals
            self.signal_buffer[symbol] = [
                s for s in self.signal_buffer[symbol]
                if datetime.fromisoformat(s.get('timestamp', datetime.now().isoformat())) > cutoff_time
            ]
            
        except Exception as e:
            logger.debug(f"Cleanup old signals error: {e}")
    
    def update_strategy_performance(self, strategy_name: str, trade_result: Dict):
        """Update strategy performance based on trade result"""
        try:
            if strategy_name not in self.strategies:
                return
            
            strategy_info = self.strategies[strategy_name]
            trade_pnl = trade_result.get('pnl', 0.0)
            
            # Update strategy performance
            strategy_info['total_pnl'] += trade_pnl
            
            if trade_pnl > 0:
                strategy_info['successful_signals'] += 1
                self.performance_metrics['profitable_trades'] += 1
            
            self.performance_metrics['total_pnl'] += trade_pnl
            self.performance_metrics['executed_signals'] += 1
            
            # Update win rate
            if strategy_info['signals_generated'] > 0:
                strategy_info['win_rate'] = (strategy_info['successful_signals'] / strategy_info['signals_generated']) * 100
            
            logger.info(f"📊 Updated {strategy_name} performance: PnL={trade_pnl:.2f}, Win Rate={strategy_info.get('win_rate', 0):.1f}%")
            
        except Exception as e:
            logger.error(f"❌ Update strategy performance error: {e}")
    
    def get_strategy_rankings(self) -> List[Dict]:
        """Get strategies ranked by performance"""
        try:
            rankings = []
            
            for strategy_name, strategy_info in self.strategies.items():
                if strategy_info['signals_generated'] == 0:
                    continue
                
                win_rate = strategy_info.get('win_rate', 0)
                total_pnl = strategy_info.get('total_pnl', 0)
                signals_count = strategy_info.get('signals_generated', 0)
                
                # Calculate performance score
                performance_score = (win_rate * 0.6) + (min(total_pnl / 1000, 100) * 0.4)
                
                rankings.append({
                    'strategy_name': strategy_name,
                    'performance_score': round(performance_score, 2),
                    'win_rate': round(win_rate, 1),
                    'total_pnl': round(total_pnl, 2),
                    'signals_generated': signals_count,
                    'successful_signals': strategy_info.get('successful_signals', 0),
                    'enabled': strategy_info.get('enabled', False),
                    'weight': strategy_info.get('weight', 1.0)
                })
            
            # Sort by performance score
            rankings.sort(key=lambda x: x['performance_score'], reverse=True)
            
            return rankings
            
        except Exception as e:
            logger.error(f"❌ Get strategy rankings error: {e}")
            return []
    
    def optimize_strategy_weights(self):
        """Optimize strategy weights based on performance"""
        try:
            rankings = self.get_strategy_rankings()
            
            if not rankings:
                return
            
            # Calculate new weights based on performance
            total_score = sum(r['performance_score'] for r in rankings)
            
            if total_score <= 0:
                return
            
            for ranking in rankings:
                strategy_name = ranking['strategy_name']
                if strategy_name in self.strategies:
                    # New weight proportional to performance
                    new_weight = (ranking['performance_score'] / total_score) * len(rankings)
                    new_weight = max(0.1, min(new_weight, 3.0))  # Clamp between 0.1 and 3.0
                    
                    old_weight = self.strategies[strategy_name]['weight']
                    self.strategies[strategy_name]['weight'] = round(new_weight, 2)
                    self.signal_weights[strategy_name] = round(new_weight, 2)
                    
                    logger.info(f"⚖️ {strategy_name} weight: {old_weight:.2f} → {new_weight:.2f}")
            
            logger.info("⚖️ Strategy weights optimized based on performance")
            
        except Exception as e:
            logger.error(f"❌ Optimize strategy weights error: {e}")
    
    def get_strategy_performance_report(self) -> Dict:
        """Get comprehensive strategy performance report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'overall_metrics': self.performance_metrics.copy(),
                'strategy_details': {},
                'rankings': self.get_strategy_rankings(),
                'consensus_stats': self._get_consensus_statistics()
            }
            
            # Add detailed strategy information
            for strategy_name, strategy_info in self.strategies.items():
                report['strategy_details'][strategy_name] = {
                    'enabled': strategy_info.get('enabled', False),
                    'weight': strategy_info.get('weight', 1.0),
                    'signals_generated': strategy_info.get('signals_generated', 0),
                    'successful_signals': strategy_info.get('successful_signals', 0),
                    'win_rate': strategy_info.get('win_rate', 0),
                    'total_pnl': strategy_info.get('total_pnl', 0),
                    'last_signal_time': strategy_info.get('last_signal_time').isoformat() if strategy_info.get('last_signal_time') else None
                }
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Strategy performance report error: {e}")
            return {'error': str(e)}
    
    def _get_consensus_statistics(self) -> Dict:
        """Get consensus signal statistics"""
        try:
            total_signals = self.performance_metrics.get('total_signals', 0)
            consensus_signals = self.performance_metrics.get('consensus_signals', 0)
            
            consensus_rate = (consensus_signals / total_signals * 100) if total_signals > 0 else 0
            
            return {
                'total_individual_signals': total_signals,
                'consensus_signals_generated': consensus_signals,
                'consensus_rate': round(consensus_rate, 1),
                'consensus_threshold': self.consensus_threshold * 100,
                'buffer_sizes': {symbol: len(signals) for symbol, signals in self.signal_buffer.items()}
            }
            
        except Exception as e:
            logger.debug(f"Consensus statistics error: {e}")
            return {}
    
    def adjust_consensus_threshold(self, new_threshold: float):
        """Adjust consensus threshold"""
        try:
            if 0.1 <= new_threshold <= 1.0:
                old_threshold = self.consensus_threshold
                self.consensus_threshold = new_threshold
                logger.info(f"🎯 Consensus threshold: {old_threshold:.1%} → {new_threshold:.1%}")
                return True
            else:
                logger.warning("⚠️ Consensus threshold must be between 0.1 and 1.0")
                return False
                
        except Exception as e:
            logger.error(f"❌ Adjust consensus threshold error: {e}")
            return False
    
    def reset_strategy_performance(self, strategy_name: str = None):
        """Reset strategy performance metrics"""
        try:
            if strategy_name:
                if strategy_name in self.strategies:
                    self.strategies[strategy_name].update({
                        'signals_generated': 0,
                        'successful_signals': 0,
                        'total_pnl': 0.0,
                        'win_rate': 0.0,
                        'last_signal_time': None
                    })
                    logger.info(f"🔄 Reset performance for {strategy_name}")
            else:
                # Reset all strategies
                for strategy_info in self.strategies.values():
                    strategy_info.update({
                        'signals_generated': 0,
                        'successful_signals': 0,
                        'total_pnl': 0.0,
                        'win_rate': 0.0,
                        'last_signal_time': None
                    })
                
                # Reset overall metrics
                self.performance_metrics = {
                    'total_signals': 0,
                    'consensus_signals': 0,
                    'executed_signals': 0,
                    'profitable_trades': 0,
                    'total_pnl': 0.0
                }
                
                logger.info("🔄 Reset performance for all strategies")
            
        except Exception as e:
            logger.error(f"❌ Reset strategy performance error: {e}")
    
    def export_strategy_data(self) -> Dict:
        """Export strategy data for analysis"""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'strategies': {},
                'performance_metrics': self.performance_metrics.copy(),
                'signal_buffer': {},
                'configuration': {
                    'consensus_threshold': self.consensus_threshold,
                    'signal_weights': self.signal_weights.copy()
                }
            }
            
            # Export strategy details
            for strategy_name, strategy_info in self.strategies.items():
                export_data['strategies'][strategy_name] = {
                    'weight': strategy_info.get('weight', 1.0),
                    'enabled': strategy_info.get('enabled', False),
                    'signals_generated': strategy_info.get('signals_generated', 0),
                    'successful_signals': strategy_info.get('successful_signals', 0),
                    'total_pnl': strategy_info.get('total_pnl', 0),
                    'win_rate': strategy_info.get('win_rate', 0),
                    'last_signal_time': strategy_info.get('last_signal_time').isoformat() if strategy_info.get('last_signal_time') else None
                }
            
            # Export signal buffer (recent signals only)
            for symbol, signals in self.signal_buffer.items():
                export_data['signal_buffer'][symbol] = signals[-10:]  # Last 10 signals per symbol
            
            return export_data
            
        except Exception as e:
            logger.error(f"❌ Export strategy data error: {e}")
            return {'error': str(e)}
    
    def get_active_strategies(self) -> List[str]:
        """Get list of active (enabled) strategies"""
        return [
            strategy_name for strategy_name, strategy_info in self.strategies.items()
            if strategy_info.get('enabled', False)
        ]
    
    def get_strategy_status(self) -> Dict:
        """Get current strategy manager status"""
        try:
            active_strategies = self.get_active_strategies()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'total_strategies': len(self.strategies),
                'active_strategies': len(active_strategies),
                'active_strategy_names': active_strategies,
                'consensus_threshold': self.consensus_threshold,
                'performance_summary': {
                    'total_signals': self.performance_metrics.get('total_signals', 0),
                    'consensus_rate': round(
                        (self.performance_metrics.get('consensus_signals', 0) / 
                         max(self.performance_metrics.get('total_signals', 1), 1)) * 100, 1
                    ),
                    'total_pnl': round(self.performance_metrics.get('total_pnl', 0), 2),
                    'win_rate': round(
                        (self.performance_metrics.get('profitable_trades', 0) / 
                         max(self.performance_metrics.get('executed_signals', 1), 1)) * 100, 1
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Get strategy status error: {e}")
            return {'error': str(e)}


# Strategy performance analyzer
class StrategyPerformanceAnalyzer:
    """Analyzes strategy performance and provides insights"""
    
    def __init__(self, strategy_manager: StrategyManager):
        self.strategy_manager = strategy_manager
    
    def analyze_strategy_correlation(self) -> Dict:
        """Analyze correlation between strategies"""
        try:
            strategies = self.strategy_manager.strategies
            correlations = {}
            
            strategy_names = list(strategies.keys())
            
            for i, strategy1 in enumerate(strategy_names):
                for j, strategy2 in enumerate(strategy_names[i+1:], i+1):
                    # Calculate correlation based on signal timing and success
                    correlation = self._calculate_strategy_correlation(strategy1, strategy2)
                    correlations[f"{strategy1}-{strategy2}"] = correlation
            
            return {
                'correlations': correlations,
                'high_correlation_pairs': [
                    pair for pair, corr in correlations.items() if abs(corr) > 0.7
                ],
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Strategy correlation analysis error: {e}")
            return {'error': str(e)}
    
    def _calculate_strategy_correlation(self, strategy1: str, strategy2: str) -> float:
        """Calculate correlation between two strategies"""
        try:
            # Simplified correlation based on performance metrics
            s1_info = self.strategy_manager.strategies.get(strategy1, {})
            s2_info = self.strategy_manager.strategies.get(strategy2, {})
            
            s1_win_rate = s1_info.get('win_rate', 0)
            s2_win_rate = s2_info.get('win_rate', 0)
            
            s1_signals = s1_info.get('signals_generated', 0)
            s2_signals = s2_info.get('signals_generated', 0)
            
            # Simple correlation based on performance similarity
            if s1_signals == 0 or s2_signals == 0:
                return 0.0
            
            signal_ratio = min(s1_signals, s2_signals) / max(s1_signals, s2_signals)
            win_rate_diff = abs(s1_win_rate - s2_win_rate) / 100
            
            correlation = signal_ratio * (1 - win_rate_diff)
            return round(correlation, 3)
            
        except Exception:
            return 0.0
    
    def recommend_strategy_adjustments(self) -> Dict:
        """Recommend strategy adjustments based on performance"""
        try:
            rankings = self.strategy_manager.get_strategy_rankings()
            recommendations = []
            
            if not rankings:
                return {'recommendations': [], 'message': 'No strategy data available'}
            
            # Analyze top and bottom performers
            top_performers = rankings[:3]
            bottom_performers = rankings[-3:]
            
            # Recommendations for top performers
            for strategy in top_performers:
                if strategy['performance_score'] > 80:
                    recommendations.append({
                        'strategy': strategy['strategy_name'],
                        'action': 'increase_weight',
                        'current_weight': strategy['weight'],
                        'suggested_weight': min(strategy['weight'] * 1.2, 3.0),
                        'reason': f"High performance score: {strategy['performance_score']}"
                    })
            
            # Recommendations for bottom performers
            for strategy in bottom_performers:
                if strategy['performance_score'] < 30 and strategy['signals_generated'] > 10:
                    recommendations.append({
                        'strategy': strategy['strategy_name'],
                        'action': 'decrease_weight_or_disable',
                        'current_weight': strategy['weight'],
                        'suggested_weight': max(strategy['weight'] * 0.5, 0.1),
                        'reason': f"Low performance score: {strategy['performance_score']}"
                    })
            
            # Check for inactive strategies
            for strategy_name, strategy_info in self.strategy_manager.strategies.items():
                if strategy_info.get('signals_generated', 0) == 0 and strategy_info.get('enabled', False):
                    recommendations.append({
                        'strategy': strategy_name,
                        'action': 'investigate_inactivity',
                        'reason': 'No signals generated despite being enabled'
                    })
            
            return {
                'recommendations': recommendations,
                'analysis_timestamp': datetime.now().isoformat(),
                'total_strategies_analyzed': len(rankings)
            }
            
        except Exception as e:
            logger.error(f"❌ Strategy recommendations error: {e}")
            return {'error': str(e)}


# Usage example and testing
if __name__ == "__main__":
    import tempfile
    
    # Test strategy manager
    print("🎯 Testing Strategy Manager...")
    
    # Mock dependencies
    class MockConfigManager:
        def get_strategy_config(self):
            return {
                'Strategy1': {'enabled': True, 'weight': 1.0},
                'Strategy2': {'enabled': True, 'weight': 1.5}
            }
        
        def toggle_strategy(self, name, enabled):
            pass
    
    class MockDataManager:
        pass
    
    class MockStrategy:
        def __init__(self, name):
            self.name = name
        
        def analyze_signal(self, market_data):
            import random
            if random.random() > 0.7:  # 30% chance of signal
                return {
                    'signal_type': random.choice(['BUY', 'SELL']),
                    'confidence': random.uniform(70, 95),
                    'price': market_data.get('ltp', 100),
                    'timestamp': datetime.now().isoformat()
                }
            return None
    
    # Initialize strategy manager
    config_manager = MockConfigManager()
    data_manager = MockDataManager()
    strategy_manager = StrategyManager(config_manager, data_manager)
    
    # Register test strategies
    strategy_manager.register_strategy('Strategy1', MockStrategy('Strategy1'), 1.0)
    strategy_manager.register_strategy('Strategy2', MockStrategy('Strategy2'), 1.5)
    
    # Test symbol analysis
    market_data = {'ltp': 2500, 'volume': 10000}
    result = strategy_manager.analyze_symbol('RELIANCE', market_data)
    
    if result:
        print(f"📊 Analysis result: {result['signal_count']} signals generated")
        if result['consensus_signal']:
            print(f"🎯 Consensus: {result['consensus_signal']['signal_type']}")
    
    # Test performance report
    report = strategy_manager.get_strategy_performance_report()
    print(f"📈 Performance report: {len(report['strategy_details'])} strategies")
    
    # Test rankings
    rankings = strategy_manager.get_strategy_rankings()
    print(f"🏆 Strategy rankings: {len(rankings)} strategies ranked")
    
    print("✅ Strategy manager test completed")
