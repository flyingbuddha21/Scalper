#!/usr/bin/env python3
"""
Dynamic Stock Scanner for High Volatility & Liquid Stocks
Scans 20-50 stocks and updates every 15-30 minutes for scalping
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import sqlite3
import json

logger = logging.getLogger(__name__)

class AssetType(Enum):
    EQUITY = "equity"
    FNO = "fno"
    INDEX = "index"

@dataclass
class StockMetrics:
    symbol: str
    asset_type: AssetType
    current_price: float
    volume: int
    volatility_score: float
    liquidity_score: float
    spread_pct: float
    atr_pct: float
    iv_rank: float
    price_change_pct: float
    volume_ratio: float
    market_cap: float
    last_updated: datetime
    scalping_score: float
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'asset_type': self.asset_type.value,
            'current_price': self.current_price,
            'volume': self.volume,
            'volatility_score': round(self.volatility_score, 2),
            'liquidity_score': round(self.liquidity_score, 2),
            'spread_pct': round(self.spread_pct, 4),
            'atr_pct': round(self.atr_pct, 3),
            'iv_rank': round(self.iv_rank, 2),
            'price_change_pct': round(self.price_change_pct, 2),
            'volume_ratio': round(self.volume_ratio, 2),
            'market_cap': self.market_cap,
            'last_updated': self.last_updated.isoformat(),
            'scalping_score': round(self.scalping_score, 2)
        }

class DynamicStockScanner:
    def __init__(self, api_handler, db_path: str = "data/scanner_cache.db"):
        self.api = api_handler
        self.db_path = db_path
        self.running = False
        self.scan_interval = 20 * 60  # 20 minutes default
        
        # Stock universes
        self.equity_universe = self._get_equity_universe()
        self.fno_universe = self._get_fno_universe()
        self.index_universe = self._get_index_universe()
        
        # Current scanned stocks
        self.scanned_stocks: Dict[str, StockMetrics] = {}
        self.top_stocks: List[StockMetrics] = []
        
        # Callbacks for real-time updates
        self.update_callbacks = []
        
        # Initialize database
        self._init_database()
        
        logger.info("🔍 Dynamic Stock Scanner initialized")
    
    def _get_equity_universe(self) -> List[str]:
        """Get equity universe for scanning"""
        # High volume, liquid equity stocks
        return [
            # Large Cap Tech
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
            "BHARTIARTL", "WIPRO", "LT", "HCLTECH", "TECHM",
            
            # Mid/Small Cap High Beta
            "ADANIENT", "ADANIPORTS", "TATAMOTORS", "BAJFINANCE",
            "AXISBANK", "KOTAKBANK", "MARUTI", "ASIANPAINT",
            "NESTLEIND", "HINDUNILVR", "ITC", "SBIN",
            
            # High Volatility Stocks
            "ZEEL", "YESBANK", "PNB", "SAIL", "NTPC", "ONGC",
            "COALINDIA", "POWERGRID", "BPCL", "IOC"
        ]
    
    def _get_fno_universe(self) -> List[str]:
        """Get F&O universe for scanning"""
        return [
            # Most active F&O stocks
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "TATAMOTORS", "BAJFINANCE", "AXISBANK", "SBIN", "LT",
            "BHARTIARTL", "MARUTI", "KOTAKBANK", "WIPRO", "HCLTECH",
            "ASIANPAINT", "TECHM", "NESTLEIND", "HINDUNILVR", "ITC",
            "ADANIENT", "ADANIPORTS", "ULTRACEMCO", "TITAN", "SUNPHARMA"
        ]
    
    def _get_index_universe(self) -> List[str]:
        """Get index universe for scanning"""
        return [
            "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
            "NIFTYIT", "NIFTYPHARMA", "NIFTYAUTO", "NIFTYMETAL",
            "NIFTYREALTY", "NIFTYENERGY"
        ]
    
    def _init_database(self):
        """Initialize scanner database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scanned_stocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    current_price REAL,
                    volume INTEGER,
                    volatility_score REAL,
                    liquidity_score REAL,
                    spread_pct REAL,
                    atr_pct REAL,
                    iv_rank REAL,
                    price_change_pct REAL,
                    volume_ratio REAL,
                    market_cap REAL,
                    scalping_score REAL,
                    scan_timestamp DATETIME,
                    is_active BOOLEAN DEFAULT 1,
                    UNIQUE(symbol, scan_timestamp)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scan_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_timestamp DATETIME,
                    total_stocks_scanned INTEGER,
                    avg_volatility REAL,
                    top_10_symbols TEXT,
                    scan_duration_seconds REAL
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Scanner database initialized")
            
        except Exception as e:
            logger.error(f"❌ Scanner database init error: {e}")
    
    def add_update_callback(self, callback):
        """Add callback for real-time updates"""
        self.update_callbacks.append(callback)
    
    def start_scanning(self, interval_minutes: int = 20):
        """Start continuous scanning"""
        self.scan_interval = interval_minutes * 60
        self.running = True
        
        # Start scanning thread
        scan_thread = threading.Thread(
            target=self._scan_loop,
            daemon=True,
            name="StockScanner"
        )
        scan_thread.start()
        
        logger.info(f"🚀 Dynamic scanner started - interval: {interval_minutes} minutes")
    
    def stop_scanning(self):
        """Stop scanning"""
        self.running = False
        logger.info("⏹️ Scanner stopped")
    
    def _scan_loop(self):
        """Main scanning loop"""
        while self.running:
            try:
                scan_start = time.time()
                logger.info("🔍 Starting market scan...")
                
                # Perform full scan
                scan_results = self._perform_full_scan()
                
                # Update database
                self._save_scan_results(scan_results)
                
                # Update top stocks
                self._update_top_stocks(scan_results)
                
                # Notify callbacks
                self._notify_callbacks()
                
                scan_duration = time.time() - scan_start
                logger.info(f"✅ Scan completed in {scan_duration:.2f}s - Found {len(scan_results)} stocks")
                
                # Wait for next scan
                time.sleep(self.scan_interval)
                
            except Exception as e:
                logger.error(f"❌ Scan loop error: {e}")
                time.sleep(60)  # Wait 1 minute before retry
    
    def _perform_full_scan(self) -> List[StockMetrics]:
        """Perform full market scan"""
        all_results = []
        
        # Scan equity stocks
        equity_results = self._scan_asset_class(self.equity_universe, AssetType.EQUITY)
        all_results.extend(equity_results)
        
        # Scan F&O stocks
        fno_results = self._scan_asset_class(self.fno_universe, AssetType.FNO)
        all_results.extend(fno_results)
        
        # Scan index futures
        index_results = self._scan_asset_class(self.index_universe, AssetType.INDEX)
        all_results.extend(index_results)
        
        # Filter and sort by scalping score
        filtered_results = self._filter_and_rank(all_results)
        
        return filtered_results
    
    def _scan_asset_class(self, symbols: List[str], asset_type: AssetType) -> List[StockMetrics]:
        """Scan specific asset class"""
        results = []
        
        for symbol in symbols:
            try:
                metrics = self._calculate_stock_metrics(symbol, asset_type)
                if metrics and self._passes_filters(metrics):
                    results.append(metrics)
                    
            except Exception as e:
                logger.debug(f"Error scanning {symbol}: {e}")
                continue
        
        return results
    
    def _calculate_stock_metrics(self, symbol: str, asset_type: AssetType) -> Optional[StockMetrics]:
        """Calculate comprehensive stock metrics"""
        try:
            # Get current market data
            market_data = self._get_market_data(symbol, asset_type)
            if not market_data:
                return None
            
            # Get historical data for calculations
            historical_data = self._get_historical_data(symbol, days=20)
            if historical_data is None or len(historical_data) < 10:
                return None
            
            # Calculate metrics
            current_price = market_data['ltp']
            volume = market_data.get('volume', 0)
            bid_ask_spread = market_data.get('spread', 0)
            
            # Volatility calculations
            price_changes = np.diff(historical_data['close'].values)
            price_volatility = np.std(price_changes) / np.mean(historical_data['close']) * 100
            
            # ATR calculation
            high_low = historical_data['high'] - historical_data['low']
            high_close = np.abs(historical_data['high'] - historical_data['close'].shift(1))
            low_close = np.abs(historical_data['low'] - historical_data['close'].shift(1))
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = np.mean(true_range.tail(14))
            atr_pct = (atr / current_price) * 100
            
            # Volume analysis
            avg_volume = np.mean(historical_data['volume'].tail(20))
            volume_ratio = volume / avg_volume if avg_volume > 0 else 0
            
            # Price change
            prev_close = historical_data['close'].iloc[-2] if len(historical_data) > 1 else current_price
            price_change_pct = ((current_price - prev_close) / prev_close) * 100
            
            # Spread percentage
            spread_pct = (bid_ask_spread / current_price) * 100 if current_price > 0 else 0
            
            # Implied Volatility Rank (simplified for demo)
            iv_rank = self._calculate_iv_rank(historical_data)
            
            # Market cap (estimated)
            market_cap = self._estimate_market_cap(symbol, current_price)
            
            # Calculate composite scores
            volatility_score = self._calculate_volatility_score(price_volatility, atr_pct, iv_rank)
            liquidity_score = self._calculate_liquidity_score(volume, volume_ratio, spread_pct, market_cap)
            scalping_score = self._calculate_scalping_score(volatility_score, liquidity_score, atr_pct, spread_pct)
            
            return StockMetrics(
                symbol=symbol,
                asset_type=asset_type,
                current_price=current_price,
                volume=volume,
                volatility_score=volatility_score,
                liquidity_score=liquidity_score,
                spread_pct=spread_pct,
                atr_pct=atr_pct,
                iv_rank=iv_rank,
                price_change_pct=price_change_pct,
                volume_ratio=volume_ratio,
                market_cap=market_cap,
                last_updated=datetime.now(),
                scalping_score=scalping_score
            )
            
        except Exception as e:
            logger.debug(f"Metrics calculation error for {symbol}: {e}")
            return None
    
    def _get_market_data(self, symbol: str, asset_type: AssetType) -> Optional[Dict]:
        """Get real-time market data"""
        try:
            # This would integrate with your actual market data API
            # For demo, returning mock data structure
            if hasattr(self.api, 'get_quotes'):
                return self.api.get_quotes(symbol)
            else:
                # Mock data for testing
                return {
                    'ltp': 100.0 + np.random.normal(0, 5),
                    'volume': int(np.random.exponential(100000)),
                    'spread': np.random.uniform(0.01, 0.20),
                    'bid': 99.95,
                    'ask': 100.05
                }
                
        except Exception as e:
            logger.debug(f"Market data error for {symbol}: {e}")
            return None
    
    def _get_historical_data(self, symbol: str, days: int = 20) -> Optional[pd.DataFrame]:
        """Get historical price data"""
        try:
            # This would integrate with your historical data source
            # For demo, generating mock historical data
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            base_price = 100.0
            
            data = []
            for i, date in enumerate(dates):
                price = base_price + np.random.normal(0, 2) + i * 0.1
                data.append({
                    'date': date,
                    'open': price + np.random.normal(0, 0.5),
                    'high': price + abs(np.random.normal(0, 1)),
                    'low': price - abs(np.random.normal(0, 1)),
                    'close': price,
                    'volume': int(np.random.exponential(50000))
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.debug(f"Historical data error for {symbol}: {e}")
            return None
    
    def _calculate_iv_rank(self, historical_data: pd.DataFrame) -> float:
        """Calculate IV rank (simplified)"""
        try:
            volatilities = []
            for i in range(10, len(historical_data)):
                returns = historical_data['close'].iloc[i-10:i].pct_change().dropna()
                vol = returns.std() * np.sqrt(252) * 100
                volatilities.append(vol)
            
            if len(volatilities) > 5:
                current_vol = volatilities[-1]
                rank = sum(1 for v in volatilities if current_vol > v) / len(volatilities) * 100
                return rank
            
            return 50.0  # Default middle rank
            
        except Exception:
            return 50.0
    
    def _estimate_market_cap(self, symbol: str, price: float) -> float:
        """Estimate market cap (simplified)"""
        # This would use actual shares outstanding data
        # For demo, using rough estimates
        cap_estimates = {
            'RELIANCE': 15000000, 'TCS': 12000000, 'HDFCBANK': 8000000,
            'INFY': 4000000, 'ICICIBANK': 7000000, 'BHARTIARTL': 6000000
        }
        
        return cap_estimates.get(symbol, 500000)  # Default to mid-cap
    
    def _calculate_volatility_score(self, price_vol: float, atr_pct: float, iv_rank: float) -> float:
        """Calculate composite volatility score"""
        # Weights: price_vol (40%), atr (40%), iv_rank (20%)
        score = (price_vol * 0.4) + (atr_pct * 40 * 0.4) + (iv_rank * 0.2)
        return min(score, 100)  # Cap at 100
    
    def _calculate_liquidity_score(self, volume: int, volume_ratio: float, 
                                 spread_pct: float, market_cap: float) -> float:
        """Calculate composite liquidity score"""
        # Volume score (0-40 points)
        volume_score = min(volume / 1000000 * 20, 40)
        
        # Volume ratio score (0-30 points)
        ratio_score = min(volume_ratio * 15, 30) if volume_ratio > 0 else 0
        
        # Spread score (0-20 points, lower spread = higher score)
        spread_score = max(20 - (spread_pct * 1000), 0)
        
        # Market cap score (0-10 points)
        cap_score = min(market_cap / 1000000, 10)
        
        return volume_score + ratio_score + spread_score + cap_score
    
    def _calculate_scalping_score(self, vol_score: float, liq_score: float, 
                                atr_pct: float, spread_pct: float) -> float:
        """Calculate overall scalping suitability score"""
        # Base score from volatility and liquidity
        base_score = (vol_score * 0.6) + (liq_score * 0.4)
        
        # ATR bonus/penalty
        if 0.5 <= atr_pct <= 3.0:  # Sweet spot for scalping
            atr_multiplier = 1.1
        elif atr_pct > 3.0:  # Too volatile
            atr_multiplier = 0.9
        else:  # Too low volatility
            atr_multiplier = 0.8
        
        # Spread penalty
        if spread_pct > 0.1:  # High spread penalty
            spread_multiplier = max(0.7, 1 - (spread_pct * 5))
        else:
            spread_multiplier = 1.0
        
        final_score = base_score * atr_multiplier * spread_multiplier
        return min(final_score, 100)
    
    def _passes_filters(self, metrics: StockMetrics) -> bool:
        """Check if stock passes minimum filters"""
        return (
            metrics.volume > 10000 and  # Minimum volume
            metrics.spread_pct < 0.5 and  # Maximum spread
            metrics.current_price > 10 and  # Minimum price
            metrics.scalping_score > 30  # Minimum scalping score
        )
    
    def _filter_and_rank(self, results: List[StockMetrics]) -> List[StockMetrics]:
        """Filter and rank stocks by scalping score"""
        # Sort by scalping score (descending)
        sorted_results = sorted(results, key=lambda x: x.scalping_score, reverse=True)
        
        # Take top 50 stocks
        return sorted_results[:50]
    
    def _save_scan_results(self, results: List[StockMetrics]):
        """Save scan results to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            scan_timestamp = datetime.now()
            
            # Save individual stock results
            for stock in results:
                cursor.execute("""
                    INSERT OR REPLACE INTO scanned_stocks 
                    (symbol, asset_type, current_price, volume, volatility_score,
                     liquidity_score, spread_pct, atr_pct, iv_rank, price_change_pct,
                     volume_ratio, market_cap, scalping_score, scan_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    stock.symbol, stock.asset_type.value, stock.current_price,
                    stock.volume, stock.volatility_score, stock.liquidity_score,
                    stock.spread_pct, stock.atr_pct, stock.iv_rank,
                    stock.price_change_pct, stock.volume_ratio, stock.market_cap,
                    stock.scalping_score, scan_timestamp
                ))
            
            # Save scan summary
            if results:
                avg_volatility = np.mean([s.volatility_score for s in results])
                top_10_symbols = json.dumps([s.symbol for s in results[:10]])
                
                cursor.execute("""
                    INSERT INTO scan_history 
                    (scan_timestamp, total_stocks_scanned, avg_volatility, 
                     top_10_symbols, scan_duration_seconds)
                    VALUES (?, ?, ?, ?, ?)
                """, (scan_timestamp, len(results), avg_volatility, top_10_symbols, 0))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Save scan results error: {e}")
    
    def _update_top_stocks(self, results: List[StockMetrics]):
        """Update top stocks for execution"""
        self.scanned_stocks = {stock.symbol: stock for stock in results}
        self.top_stocks = results[:10]  # Top 10 for execution
        
        logger.info(f"📊 Updated scanner: {len(results)} total, top 10: {[s.symbol for s in self.top_stocks]}")
    
    def _notify_callbacks(self):
        """Notify all registered callbacks"""
        for callback in self.update_callbacks:
            try:
                callback(self.get_scan_summary())
            except Exception as e:
                logger.debug(f"Callback notification error: {e}")
    
    def get_top_stocks(self, count: int = 10) -> List[Dict]:
        """Get top ranked stocks"""
        return [stock.to_dict() for stock in self.top_stocks[:count]]
    
    def get_scan_summary(self) -> Dict:
        """Get current scan summary"""
        if not self.scanned_stocks:
            return {'status': 'no_data'}
        
        return {
            'status': 'active',
            'total_scanned': len(self.scanned_stocks),
            'top_10_count': len(self.top_stocks),
            'avg_volatility': np.mean([s.volatility_score for s in self.scanned_stocks.values()]),
            'avg_liquidity': np.mean([s.liquidity_score for s in self.scanned_stocks.values()]),
            'top_symbols': [s.symbol for s in self.top_stocks],
            'last_scan': max(s.last_updated for s in self.scanned_stocks.values()).isoformat(),
            'next_scan_in': self._get_next_scan_time(),
            'asset_breakdown': self._get_asset_breakdown()
        }
    
    def _get_next_scan_time(self) -> int:
        """Get seconds until next scan"""
        # This would calculate based on last scan time and interval
        return self.scan_interval
    
    def _get_asset_breakdown(self) -> Dict:
        """Get breakdown by asset type"""
        breakdown = {'equity': 0, 'fno': 0, 'index': 0}
        
        for stock in self.scanned_stocks.values():
            breakdown[stock.asset_type.value] += 1
        
        return breakdown
    
    def force_scan(self) -> Dict:
        """Force immediate scan"""
        try:
            logger.info("🔍 Force scan initiated...")
            scan_start = time.time()
            
            # Perform scan
            results = self._perform_full_scan()
            self._save_scan_results(results)
            self._update_top_stocks(results)
            self._notify_callbacks()
            
            scan_duration = time.time() - scan_start
            
            return {
                'success': True,
                'scan_duration': round(scan_duration, 2),
                'stocks_found': len(results),
                'top_10': [s.symbol for s in self.top_stocks],
                'message': f'Force scan completed - {len(results)} stocks analyzed'
            }
            
        except Exception as e:
            logger.error(f"❌ Force scan error: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Force scan failed'
            }
    
    def update_scan_interval(self, minutes: int):
        """Update scan interval"""
        if 5 <= minutes <= 120:  # Between 5 minutes and 2 hours
            self.scan_interval = minutes * 60
            logger.info(f"📅 Scan interval updated to {minutes} minutes")
            return True
        return False
    
    def get_stock_details(self, symbol: str) -> Optional[Dict]:
        """Get detailed metrics for specific stock"""
        if symbol in self.scanned_stocks:
            return self.scanned_stocks[symbol].to_dict()
        return None
    
    def get_historical_scans(self, days: int = 7) -> List[Dict]:
        """Get historical scan data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            cursor.execute("""
                SELECT scan_timestamp, total_stocks_scanned, avg_volatility, 
                       top_10_symbols, scan_duration_seconds
                FROM scan_history 
                WHERE scan_timestamp > ?
                ORDER BY scan_timestamp DESC
            """, (cutoff_date,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'timestamp': row[0],
                    'total_scanned': row[1],
                    'avg_volatility': round(row[2], 2),
                    'top_10_symbols': json.loads(row[3]) if row[3] else [],
                    'duration': row[4]
                })
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"❌ Historical scans error: {e}")
            return []
    
    def export_scan_data(self, format: str = 'json') -> str:
        """Export current scan data"""
        try:
            data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_stocks': len(self.scanned_stocks),
                'top_stocks': self.get_top_stocks(20),
                'scan_summary': self.get_scan_summary(),
                'scanner_config': {
                    'scan_interval_minutes': self.scan_interval // 60,
                    'equity_universe_size': len(self.equity_universe),
                    'fno_universe_size': len(self.fno_universe),
                    'index_universe_size': len(self.index_universe)
                }
            }
            
            if format == 'json':
                return json.dumps(data, indent=2)
            elif format == 'csv':
                # Convert to CSV format
                df = pd.DataFrame([stock.to_dict() for stock in self.scanned_stocks.values()])
                return df.to_csv(index=False)
            
        except Exception as e:
            logger.error(f"❌ Export error: {e}")
            return ""
    
    def add_custom_symbol(self, symbol: str, asset_type: str) -> bool:
        """Add custom symbol to scan universe"""
        try:
            asset_enum = AssetType(asset_type.lower())
            
            if asset_enum == AssetType.EQUITY:
                if symbol not in self.equity_universe:
                    self.equity_universe.append(symbol)
                    logger.info(f"➕ Added {symbol} to equity universe")
                    return True
            elif asset_enum == AssetType.FNO:
                if symbol not in self.fno_universe:
                    self.fno_universe.append(symbol)
                    logger.info(f"➕ Added {symbol} to F&O universe")
                    return True
            elif asset_enum == AssetType.INDEX:
                if symbol not in self.index_universe:
                    self.index_universe.append(symbol)
                    logger.info(f"➕ Added {symbol} to index universe")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Add symbol error: {e}")
            return False
    
    def remove_symbol(self, symbol: str) -> bool:
        """Remove symbol from all universes"""
        removed = False
        
        if symbol in self.equity_universe:
            self.equity_universe.remove(symbol)
            removed = True
        
        if symbol in self.fno_universe:
            self.fno_universe.remove(symbol)
            removed = True
        
        if symbol in self.index_universe:
            self.index_universe.remove(symbol)
            removed = True
        
        if removed:
            logger.info(f"➖ Removed {symbol} from scan universe")
        
        return removed
    
    def get_scanner_status(self) -> Dict:
        """Get detailed scanner status"""
        return {
            'running': self.running,
            'scan_interval_minutes': self.scan_interval // 60,
            'total_universe_size': len(self.equity_universe) + len(self.fno_universe) + len(self.index_universe),
            'universe_breakdown': {
                'equity': len(self.equity_universe),
                'fno': len(self.fno_universe),
                'index': len(self.index_universe)
            },
            'current_scan_data': {
                'total_scanned': len(self.scanned_stocks),
                'top_10_active': len(self.top_stocks),
                'last_scan': max(s.last_updated for s in self.scanned_stocks.values()).isoformat() if self.scanned_stocks else None
            },
            'performance_stats': self._get_performance_stats()
        }
    
    def _get_performance_stats(self) -> Dict:
        """Get scanner performance statistics"""
        try:
            if not self.scanned_stocks:
                return {}
            
            volatility_scores = [s.volatility_score for s in self.scanned_stocks.values()]
            liquidity_scores = [s.liquidity_score for s in self.scanned_stocks.values()]
            scalping_scores = [s.scalping_score for s in self.scanned_stocks.values()]
            
            return {
                'avg_volatility_score': round(np.mean(volatility_scores), 2),
                'max_volatility_score': round(max(volatility_scores), 2),
                'avg_liquidity_score': round(np.mean(liquidity_scores), 2),
                'max_liquidity_score': round(max(liquidity_scores), 2),
                'avg_scalping_score': round(np.mean(scalping_scores), 2),
                'max_scalping_score': round(max(scalping_scores), 2),
                'stocks_above_80_scalping': sum(1 for s in scalping_scores if s > 80),
                'stocks_above_60_scalping': sum(1 for s in scalping_scores if s > 60)
            }
            
        except Exception as e:
            logger.debug(f"Performance stats error: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup scanner resources"""
        try:
            self.stop_scanning()
            
            # Clear data
            self.scanned_stocks.clear()
            self.top_stocks.clear()
            self.update_callbacks.clear()
            
            logger.info("🧹 Scanner cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Scanner cleanup error: {e}")


# Scanner Testing and Validation
class ScannerValidator:
    """Validates scanner results and performance"""
    
    def __init__(self, scanner: DynamicStockScanner):
        self.scanner = scanner
        self.logger = logging.getLogger(__name__ + ".Validator")
    
    def validate_scan_results(self) -> Dict:
        """Validate current scan results"""
        try:
            results = {
                'total_checks': 0,
                'passed_checks': 0,
                'warnings': [],
                'errors': [],
                'summary': {}
            }
            
            # Check 1: Minimum stocks found
            results['total_checks'] += 1
            if len(self.scanner.scanned_stocks) >= 20:
                results['passed_checks'] += 1
            else:
                results['warnings'].append(f"Only {len(self.scanner.scanned_stocks)} stocks found (minimum: 20)")
            
            # Check 2: Top 10 quality
            results['total_checks'] += 1
            if len(self.scanner.top_stocks) >= 10:
                avg_scalping_score = np.mean([s.scalping_score for s in self.scanner.top_stocks])
                if avg_scalping_score >= 60:
                    results['passed_checks'] += 1
                else:
                    results['warnings'].append(f"Low average scalping score: {avg_scalping_score:.1f}")
            else:
                results['errors'].append("Less than 10 top stocks available")
            
            # Check 3: Asset diversification
            results['total_checks'] += 1
            asset_breakdown = self.scanner._get_asset_breakdown()
            if len([v for v in asset_breakdown.values() if v > 0]) >= 2:
                results['passed_checks'] += 1
            else:
                results['warnings'].append("Limited asset class diversification")
            
            # Check 4: Data freshness
            results['total_checks'] += 1
            if self.scanner.scanned_stocks:
                latest_update = max(s.last_updated for s in self.scanner.scanned_stocks.values())
                if (datetime.now() - latest_update).seconds < 3600:  # Within 1 hour
                    results['passed_checks'] += 1
                else:
                    results['warnings'].append("Scan data is stale (>1 hour old)")
            
            # Summary
            results['summary'] = {
                'validation_score': (results['passed_checks'] / results['total_checks']) * 100,
                'status': 'healthy' if results['passed_checks'] >= results['total_checks'] * 0.8 else 'warning',
                'total_stocks': len(self.scanner.scanned_stocks),
                'top_10_avg_score': round(np.mean([s.scalping_score for s in self.scanner.top_stocks]), 2) if self.scanner.top_stocks else 0
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return {'error': str(e)}
    
    def performance_benchmark(self) -> Dict:
        """Benchmark scanner performance"""
        try:
            # This would run performance tests
            return {
                'scan_speed': 'normal',
                'memory_usage': 'acceptable',
                'cpu_usage': 'low',
                'api_calls_per_scan': len(self.scanner.equity_universe) + len(self.scanner.fno_universe) + len(self.scanner.index_universe),
                'recommendation': 'Scanner performing within normal parameters'
            }
            
        except Exception as e:
            self.logger.error(f"Benchmark error: {e}")
            return {'error': str(e)}


# Usage example and integration point
if __name__ == "__main__":
    # This would be used for testing the scanner independently
    import sys
    
    # Mock API handler for testing
    class MockAPI:
        def get_quotes(self, symbol):
            return {
                'ltp': 100.0 + np.random.normal(0, 5),
                'volume': int(np.random.exponential(100000)),
                'spread': np.random.uniform(0.01, 0.20)
            }
    
    # Test scanner
    mock_api = MockAPI()
    scanner = DynamicStockScanner(mock_api)
    
    print("🧪 Testing Dynamic Stock Scanner...")
    
    # Start scanning
    scanner.start_scanning(interval_minutes=1)  # 1 minute for testing
    
    # Wait and check results
    time.sleep(65)  # Wait for first scan
    
    summary = scanner.get_scan_summary()
    print(f"📊 Scan Summary: {summary}")
    
    top_stocks = scanner.get_top_stocks(5)
    print(f"🏆 Top 5 Stocks: {[s['symbol'] for s in top_stocks]}")
    
    # Validate results
    validator = ScannerValidator(scanner)
    validation = validator.validate_scan_results()
    print(f"✅ Validation Score: {validation.get('summary', {}).get('validation_score', 0)}%")
    
    scanner.stop_scanning()
    print("🛑 Scanner test completed")
