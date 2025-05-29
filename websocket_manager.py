#!/usr/bin/env python3
"""
WebSocket Manager for Real-time Communication
Handles WebSocket connections and real-time data streaming
"""

import asyncio
import websockets
import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Set, Any
import queue

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self, host='localhost', port=9001):
        self.host = host
        self.port = port
        self.running = False
        
        # Connected clients
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # Data queues for different types of updates
        self.market_data_queue = queue.Queue(maxsize=1000)
        self.signal_queue = queue.Queue(maxsize=500)
        self.portfolio_queue = queue.Queue(maxsize=100)
        self.alert_queue = queue.Queue(maxsize=200)
        
        # External components
        self.bot_core = None
        self.scanner = None
        self.execution_manager = None
        self.volatility_analyzer = None
        
        # Background threads
        self.server_thread = None
        self.broadcast_thread = None
        
        logger.info(f"🔌 WebSocket Manager initialized on {host}:{port}")
    
    def set_components(self, bot_core=None, scanner=None, execution_manager=None, volatility_analyzer=None):
        """Set external components for data access"""
        self.bot_core = bot_core
        self.scanner = scanner
        self.execution_manager = execution_manager
        self.volatility_analyzer = volatility_analyzer
    
    async def register_client(self, websocket):
        """Register new WebSocket client"""
        try:
            self.clients.add(websocket)
            logger.info(f"🔌 Client connected: {websocket.remote_address}")
            
            # Send welcome message
            welcome_msg = {
                'type': 'welcome',
                'message': 'Connected to Trading Bot WebSocket',
                'timestamp': datetime.now().isoformat(),
                'client_count': len(self.clients)
            }
            await websocket.send(json.dumps(welcome_msg))
            
        except Exception as e:
            logger.error(f"❌ Client registration error: {e}")
    
    async def unregister_client(self, websocket):
        """Unregister WebSocket client"""
        try:
            self.clients.discard(websocket)
            logger.info(f"🔌 Client disconnected: {websocket.remote_address}")
            
        except Exception as e:
            logger.error(f"❌ Client unregistration error: {e}")
    
    async def handle_client_message(self, websocket, message):
        """Handle incoming client messages"""
        try:
            data = json.loads(message)
            msg_type = data.get('type', '')
            
            if msg_type == 'ping':
                # Respond to ping
                pong_msg = {
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                }
                await websocket.send(json.dumps(pong_msg))
                
            elif msg_type == 'request_status':
                # Send current status
                status_data = await self._get_current_status()
                await websocket.send(json.dumps(status_data))
                
            elif msg_type == 'request_portfolio':
                # Send portfolio data
                portfolio_data = await self._get_portfolio_data()
                await websocket.send(json.dumps(portfolio_data))
                
            elif msg_type == 'request_top_stocks':
                # Send top stocks
                stocks_data = await self._get_top_stocks_data()
                await websocket.send(json.dumps(stocks_data))
                
            elif msg_type == 'force_scan':
                # Force scanner update
                result = await self._force_scanner_update()
                await websocket.send(json.dumps(result))
                
            else:
                error_msg = {
                    'type': 'error',
                    'message': f'Unknown message type: {msg_type}',
                    'timestamp': datetime.now().isoformat()
                }
                await websocket.send(json.dumps(error_msg))
                
        except json.JSONDecodeError:
            error_msg = {
                'type': 'error',
                'message': 'Invalid JSON format',
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(error_msg))
            
        except Exception as e:
            logger.error(f"❌ Handle client message error: {e}")
            error_msg = {
                'type': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(error_msg))
    
    async def _get_current_status(self) -> Dict:
        """Get current system status"""
        try:
            status = {
                'type': 'status_update',
                'timestamp': datetime.now().isoformat(),
                'bot_status': self.bot_core.get_bot_status() if self.bot_core else {'running': False},
                'scanner_status': self.scanner.get_scanner_status() if self.scanner else {'running': False},
                'execution_status': self.execution_manager.get_execution_summary() if self.execution_manager else {'running': False},
                'client_count': len(self.clients)
            }
            return status
            
        except Exception as e:
            logger.error(f"❌ Get current status error: {e}")
            return {
                'type': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _get_portfolio_data(self) -> Dict:
        """Get portfolio data"""
        try:
            if self.bot_core and self.bot_core.paper_engine:
                portfolio = self.bot_core.paper_engine.get_portfolio_summary()
                return {
                    'type': 'portfolio_update',
                    'timestamp': datetime.now().isoformat(),
                    'data': portfolio
                }
            else:
                return {
                    'type': 'error',
                    'message': 'Portfolio data not available',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Get portfolio data error: {e}")
            return {
                'type': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _get_top_stocks_data(self) -> Dict:
        """Get top stocks data"""
        try:
            if self.scanner:
                top_stocks = self.scanner.get_top_stocks(10)
                return {
                    'type': 'top_stocks_update',
                    'timestamp': datetime.now().isoformat(),
                    'data': top_stocks
                }
            else:
                return {
                    'type': 'error',
                    'message': 'Scanner data not available',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Get top stocks data error: {e}")
            return {
                'type': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _force_scanner_update(self) -> Dict:
        """Force scanner update"""
        try:
            if self.scanner:
                result = self.scanner.force_scan()
                return {
                    'type': 'scan_result',
                    'timestamp': datetime.now().isoformat(),
                    'data': result
                }
            else:
                return {
                    'type': 'error',
                    'message': 'Scanner not available',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Force scanner update error: {e}")
            return {
                'type': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def broadcast_message(self, message: Dict):
        """Broadcast message to all connected clients"""
        if not self.clients:
            return
        
        message_str = json.dumps(message)
        disconnected_clients = set()
        
        for client in self.clients.copy():
            try:
                await client.send(message_str)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.debug(f"Broadcast error to {client.remote_address}: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.clients.discard(client)
    
    async def client_handler(self, websocket, path):
        """Handle individual client connection"""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Client {websocket.remote_address} connection closed")
        except Exception as e:
            logger.error(f"❌ Client handler error: {e}")
        finally:
            await self.unregister_client(websocket)
    
    def start_server(self):
        """Start WebSocket server"""
        try:
            self.running = True
            
            # Start asyncio event loop in separate thread
            def run_server():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                start_server = websockets.serve(
                    self.client_handler,
                    self.host,
                    self.port,
                    ping_interval=20,
                    ping_timeout=10
                )
                
                loop.run_until_complete(start_server)
                loop.run_forever()
            
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            
            # Start broadcast thread
            self.broadcast_thread = threading.Thread(target=self._broadcast_loop, daemon=True)
            self.broadcast_thread.start()
            
            logger.info(f"🚀 WebSocket server started on ws://{self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"❌ Start server error: {e}")
            self.running = False
    
    def stop_server(self):
        """Stop WebSocket server"""
        try:
            self.running = False
            logger.info("⏹️ WebSocket server stopped")
            
        except Exception as e:
            logger.error(f"❌ Stop server error: {e}")
    
    def _broadcast_loop(self):
        """Background loop for broadcasting updates"""
        while self.running:
            try:
                # Broadcast status updates every 5 seconds
                if self.clients:
                    status_update = asyncio.run(self._get_current_status())
                    asyncio.run(self.broadcast_message(status_update))
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.debug(f"Broadcast loop error: {e}")
                time.sleep(5)
    
    def send_market_data_update(self, symbol: str, data: Dict):
        """Send market data update"""
        try:
            message = {
                'type': 'market_data',
                'symbol': symbol,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            
            if not self.market_data_queue.full():
                self.market_data_queue.put(message)
            
            # Immediate broadcast for critical updates
            if self.clients:
                asyncio.run(self.broadcast_message(message))
                
        except Exception as e:
            logger.debug(f"Send market data error: {e}")
    
    def send_signal_update(self, signal_data: Dict):
        """Send trading signal update"""
        try:
            message = {
                'type': 'trading_signal',
                'data': signal_data,
                'timestamp': datetime.now().isoformat()
            }
            
            if not self.signal_queue.full():
                self.signal_queue.put(message)
            
            # Immediate broadcast for signals
            if self.clients:
                asyncio.run(self.broadcast_message(message))
                
        except Exception as e:
            logger.debug(f"Send signal error: {e}")
    
    def send_portfolio_update(self, portfolio_data: Dict):
        """Send portfolio update"""
        try:
            message = {
                'type': 'portfolio_update',
                'data': portfolio_data,
                'timestamp': datetime.now().isoformat()
            }
            
            if not self.portfolio_queue.full():
                self.portfolio_queue.put(message)
            
            # Broadcast portfolio updates
            if self.clients:
                asyncio.run(self.broadcast_message(message))
                
        except Exception as e:
            logger.debug(f"Send portfolio error: {e}")
    
    def send_alert(self, alert_type: str, message: str, data: Dict = None):
        """Send alert message"""
        try:
            alert_message = {
                'type': 'alert',
                'alert_type': alert_type,
                'message': message,
                'data': data or {},
                'timestamp': datetime.now().isoformat()
            }
            
            if not self.alert_queue.full():
                self.alert_queue.put(alert_message)
            
            # Immediate broadcast for alerts
            if self.clients:
                asyncio.run(self.broadcast_message(alert_message))
                
        except Exception as e:
            logger.debug(f"Send alert error: {e}")
    
    def get_connection_stats(self) -> Dict:
        """Get WebSocket connection statistics"""
        try:
            return {
                'connected_clients': len(self.clients),
                'server_running': self.running,
                'host': self.host,
                'port': self.port,
                'queue_sizes': {
                    'market_data': self.market_data_queue.qsize(),
                    'signals': self.signal_queue.qsize(),
                    'portfolio': self.portfolio_queue.qsize(),
                    'alerts': self.alert_queue.qsize()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Connection stats error: {e}")
            return {'error': str(e)}


# Usage example and testing
if __name__ == "__main__":
    # Test WebSocket server
    import random
    
    ws_manager = WebSocketManager()
    ws_manager.start_server()
    
    print("🔌 WebSocket server running on ws://localhost:9001")
    print("📱 Connect with a WebSocket client to test")
    print("💬 Send JSON messages like: {'type': 'ping'}")
    
    # Simulate some data updates
    def simulate_updates():
        time.sleep(2)  # Wait for server to start
        
        for i in range(10):
            # Simulate market data
            ws_manager.send_market_data_update('RELIANCE', {
                'ltp': 2500 + random.uniform(-10, 10),
                'volume': random.randint(1000, 10000),
                'bid': 2499.95,
                'ask': 2500.05
            })
            
            # Simulate trading signal
            if random.random() > 0.7:  # 30% chance
                ws_manager.send_signal_update({
                    'symbol': 'RELIANCE',
                    'signal': 'BUY',
                    'confidence': random.uniform(70, 95),
                    'strategy': 'Strategy1_BidAskScalping'
                })
            
            # Simulate alert
            if random.random() > 0.9:  # 10% chance
                ws_manager.send_alert('HIGH_VOLATILITY', 'RELIANCE volatility spike detected')
            
            time.sleep(3)
    
    # Start simulation in background
    sim_thread = threading.Thread(target=simulate_updates, daemon=True)
    sim_thread.start()
    
    try:
        # Keep running
        while True:
            time.sleep(1)
            
            # Print stats every 10 seconds
            if int(time.time()) % 10 == 0:
                stats = ws_manager.get_connection_stats()
                print(f"📊 Connection Stats: {stats}")
                
    except KeyboardInterrupt:
        print("\n⏹️ Stopping WebSocket server...")
        ws_manager.stop_server()
