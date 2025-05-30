#!/usr/bin/env python3
"""
Production Monitoring System for Trading Database
Comprehensive monitoring, alerting, and health checks
"""

import asyncio
import logging
import time
import psutil
import asyncpg
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import schedule
from pathlib import Path

# Import system components
from database_setup import TradingDatabase
from config_manager import ConfigManager
from notification_system import NotificationManager
from utils import Logger, ErrorHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

class ServiceStatus(Enum):
    """Service status enumeration"""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    DOWN = "DOWN"

@dataclass
class HealthCheck:
    """Health check result"""
    service: str
    status: ServiceStatus
    message: str
    timestamp: datetime
    metrics: Dict[str, Any]
    response_time: float

@dataclass
class Alert:
    """Alert notification"""
    level: AlertLevel
    service: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] =     async def initialize(self):
        """Initialize database connections"""
        try:
            # Use existing trading

class DatabaseMonitor:
    """PostgreSQL and SQLite database monitoring"""
    
    def __init__(self, trading_db: TradingDatabase, config_manager: ConfigManager):
        self.trading_db = trading_db
        self.config = config_manager.get_config()['monitoring']
        self.pg_pool = trading_db.pg_pool
        self.sqlite_path = trading_db.sqlite_path
        self.alert_thresholds = {
            'connection_usage': 80,  # % of max connections
            'query_time': 5.0,       # seconds
            'disk_space': 85,        # % used
            'cache_hit_ratio': 90,   # % minimum
            'deadlocks': 5,          # per hour
            'replication_lag': 60    # seconds
        }
    
    async def initialize(self):
        """Initialize database connections"""
        try:
            self.connection_pool = await asyncpg.create_pool(
                **self.pg_config,
                min_size=2,
                max_size=10
            )
            logger.info("Database monitor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database monitor: {e}")
            raise
    
    async def check_postgresql_health(self) -> HealthCheck:
        """Comprehensive PostgreSQL health check"""
        start_time = time.time()
        metrics = {}
        
        try:
            async with self.connection_pool.acquire() as conn:
                # Basic connectivity
                await conn.execute("SELECT 1")
                
                # Connection statistics
                conn_stats = await conn.fetchrow("""
                    SELECT 
                        numbackends as active_connections,
                        xact_commit as transactions_committed,
                        xact_rollback as transactions_rolled_back,
                        blks_read as blocks_read,
                        blks_hit as blocks_hit,
                        tup_returned as tuples_returned,
                        tup_fetched as tuples_fetched,
                        tup_inserted as tuples_inserted,
                        tup_updated as tuples_updated,
                        tup_deleted as tuples_deleted
                    FROM pg_stat_database 
                    WHERE datname = current_database()
                """)
                
                # Cache hit ratio
                cache_hit_ratio = (conn_stats['blocks_hit'] / 
                                 max(conn_stats['blocks_hit'] + conn_stats['blocks_read'], 1)) * 100
                
                # Long running queries
                long_queries = await conn.fetch("""
                    SELECT pid, query, state, query_start
                    FROM pg_stat_activity 
                    WHERE state = 'active' 
                    AND query_start < NOW() - INTERVAL '30 seconds'
                    AND query NOT LIKE '%pg_stat_activity%'
                """)
                
                # Lock statistics
                locks = await conn.fetchrow("""
                    SELECT COUNT(*) as total_locks,
                           COUNT(CASE WHEN NOT granted THEN 1 END) as waiting_locks
                    FROM pg_locks
                """)
                
                # Database size
                db_size = await conn.fetchrow("""
                    SELECT pg_size_pretty(pg_database_size(current_database())) as size,
                           pg_database_size(current_database()) as size_bytes
                """)
                
                # Replication lag (if applicable)
                replication_lag = await conn.fetchrow("""
                    SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) as lag_seconds
                """)
                
                metrics.update({
                    'active_connections': conn_stats['active_connections'],
                    'cache_hit_ratio': round(cache_hit_ratio, 2),
                    'long_running_queries': len(long_queries),
                    'waiting_locks': locks['waiting_locks'],
                    'database_size_bytes': db_size['size_bytes'],
                    'database_size_pretty': db_size['size'],
                    'transactions_per_second': conn_stats['transactions_committed'],
                    'replication_lag_seconds': replication_lag['lag_seconds'] if replication_lag['lag_seconds'] else 0
                })
                
                # Determine health status
                status = ServiceStatus.HEALTHY
                issues = []
                
                if cache_hit_ratio < self.alert_thresholds['cache_hit_ratio']:
                    status = ServiceStatus.DEGRADED
                    issues.append(f"Low cache hit ratio: {cache_hit_ratio:.1f}%")
                
                if len(long_queries) > 5:
                    status = ServiceStatus.DEGRADED
                    issues.append(f"Too many long-running queries: {len(long_queries)}")
                
                if locks['waiting_locks'] > 10:
                    status = ServiceStatus.DEGRADED
                    issues.append(f"High lock contention: {locks['waiting_locks']} waiting locks")
                
                message = "PostgreSQL healthy" if not issues else "; ".join(issues)
                
        except Exception as e:
            status = ServiceStatus.DOWN
            message = f"PostgreSQL connection failed: {str(e)}"
            logger.error(message)
        
        response_time = time.time() - start_time
        
        return HealthCheck(
            service="postgresql",
            status=status,
            message=message,
            timestamp=datetime.now(),
            metrics=metrics,
            response_time=response_time
        )
    
    async def check_sqlite_health(self) -> HealthCheck:
        """SQLite database health check"""
        start_time = time.time()
        metrics = {}
        
        try:
            # Check file exists and is accessible
            sqlite_file = Path(self.sqlite_path)
            if not sqlite_file.exists():
                raise FileNotFoundError(f"SQLite file not found: {self.sqlite_path}")
            
            # Get file size
            file_size = sqlite_file.stat().st_size
            
            # Connect and run diagnostics
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            # Basic connectivity
            cursor.execute("SELECT 1")
            
            # Get database info
            cursor.execute("PRAGMA database_list")
            db_info = cursor.fetchall()
            
            # Check integrity
            cursor.execute("PRAGMA integrity_check")
            integrity = cursor.fetchone()[0]
            
            # Get page count and size
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            
            # Get table statistics
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            table_stats = {}
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                table_stats[table] = count
            
            conn.close()
            
            metrics.update({
                'file_size_bytes': file_size,
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'page_count': page_count,
                'page_size': page_size,
                'total_pages_size': page_count * page_size,
                'table_counts': table_stats,
                'integrity_check': integrity
            })
            
            # Determine status
            status = ServiceStatus.HEALTHY
            message = "SQLite healthy"
            
            if integrity != "ok":
                status = ServiceStatus.UNHEALTHY
                message = f"SQLite integrity check failed: {integrity}"
            
        except Exception as e:
            status = ServiceStatus.DOWN
            message = f"SQLite check failed: {str(e)}"
            logger.error(message)
        
        response_time = time.time() - start_time
        
        return HealthCheck(
            service="sqlite",
            status=status,
            message=message,
            timestamp=datetime.now(),
            metrics=metrics,
            response_time=response_time
        )

class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self):
        self.alert_thresholds = {
            'cpu_usage': 80.0,      # %
            'memory_usage': 85.0,   # %
            'disk_usage': 90.0,     # %
            'network_errors': 100,  # per minute
            'load_average': 4.0     # 1-minute load average
        }
    
    def check_system_health(self) -> HealthCheck:
        """Comprehensive system health check"""
        start_time = time.time()
        metrics = {}
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg()[0]  # 1-minute load average
            
            # Memory usage
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network statistics
            network = psutil.net_io_counters()
            
            # Process information
            process_count = len(psutil.pids())
            
            metrics.update({
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'load_average_1min': load_avg,
                'memory_total_gb': round(memory.total / (1024**3), 2),
                'memory_used_gb': round(memory.used / (1024**3), 2),
                'memory_percent': memory.percent,
                'swap_percent': swap.percent,
                'disk_total_gb': round(disk.total / (1024**3), 2),
                'disk_used_gb': round(disk.used / (1024**3), 2),
                'disk_percent': (disk.used / disk.total) * 100,
                'disk_read_mb': round(disk_io.read_bytes / (1024**2), 2),
                'disk_write_mb': round(disk_io.write_bytes / (1024**2), 2),
                'network_sent_mb': round(network.bytes_sent / (1024**2), 2),
                'network_recv_mb': round(network.bytes_recv / (1024**2), 2),
                'network_errors': network.errin + network.errout,
                'process_count': process_count
            })
            
            # Determine health status
            status = ServiceStatus.HEALTHY
            issues = []
            
            if cpu_percent > self.alert_thresholds['cpu_usage']:
                status = ServiceStatus.DEGRADED
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory.percent > self.alert_thresholds['memory_usage']:
                status = ServiceStatus.DEGRADED
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            
            if (disk.used / disk.total) * 100 > self.alert_thresholds['disk_usage']:
                status = ServiceStatus.CRITICAL
                issues.append(f"High disk usage: {(disk.used / disk.total) * 100:.1f}%")
            
            if load_avg > self.alert_thresholds['load_average']:
                status = ServiceStatus.DEGRADED
                issues.append(f"High load average: {load_avg:.2f}")
            
            message = "System healthy" if not issues else "; ".join(issues)
            
        except Exception as e:
            status = ServiceStatus.DOWN
            message = f"System check failed: {str(e)}"
            logger.error(message)
        
        response_time = time.time() - start_time
        
        return HealthCheck(
            service="system",
            status=status,
            message=message,
            timestamp=datetime.now(),
            metrics=metrics,
            response_time=response_time
        )

class AlertManager:
    """Alert management and notification system"""
    
    def __init__(self, config: Dict, notification_manager: NotificationManager):
        self.config = config
        self.notification_manager = notification_manager
        self.alerts: List[Alert] = []
        self.logger = Logger(__name__)
        
        # Setup notification channels
        self.notification_channels = []
        if config.get('email'):
            self.notification_channels.append(self._send_email_alert)
        if config.get('webhook'):
            self.notification_channels.append(self._send_webhook_alert)
    
    async def process_health_check(self, health_check: HealthCheck):
        """Process health check and generate alerts if needed"""
        alert_level = self._determine_alert_level(health_check.status)
        
        if alert_level:
            alert = Alert(
                level=alert_level,
                service=health_check.service,
                message=f"{health_check.service}: {health_check.message}",
                timestamp=health_check.timestamp
            )
            
            await self._send_alert(alert)
            self.alerts.append(alert)
            
            # Also send through notification manager
            await self.notification_manager.send_alert(
                f"System Alert: {alert.service}",
                alert.message,
                alert.level.value
            )
    
    def _determine_alert_level(self, status: ServiceStatus) -> Optional[AlertLevel]:
        """Determine alert level based on service status"""
        status_to_alert = {
            ServiceStatus.HEALTHY: None,
            ServiceStatus.DEGRADED: AlertLevel.WARNING,
            ServiceStatus.UNHEALTHY: AlertLevel.CRITICAL,
            ServiceStatus.DOWN: AlertLevel.EMERGENCY
        }
        return status_to_alert.get(status)
    
    async def _send_alert(self, alert: Alert):
        """Send alert through all configured channels"""
        for channel in self.notification_channels:
            try:
                await channel(alert)
            except Exception as e:
                self.logger.error(f"Failed to send alert via {channel.__name__}: {e}")
    
    async def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        if not self.config.get('email'):
            return
        
        email_config = self.config['email']
        
        msg = MIMEMultipart()
        msg['From'] = email_config['from']
        msg['To'] = ', '.join(email_config['to'])
        msg['Subject'] = f"[{alert.level.value}] Trading System Alert - {alert.service}"
        
        body = f"""
        Alert Level: {alert.level.value}
        Service: {alert.service}
        Message: {alert.message}
        Timestamp: {alert.timestamp}
        
        Please investigate immediately.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
            if email_config.get('use_tls'):
                server.starttls()
            if email_config.get('username'):
                server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
    
    async def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert"""
        if not self.config.get('webhook'):
            return
        
        webhook_config = self.config['webhook']
        
        payload = {
            'level': alert.level.value,
            'service': alert.service,
            'message': alert.message,
            'timestamp': alert.timestamp.isoformat(),
            'resolved': alert.resolved
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                webhook_config['url'],
                json=payload,
                headers=webhook_config.get('headers', {}),
                timeout=10
            ) as response:
                if response.status >= 300:
                    logger.error(f"Webhook alert failed: {response.status}")

class TradingSystemMonitor:
    """Main trading system monitoring orchestrator"""
    
    def __init__(self, trading_db: TradingDatabase, config_manager: ConfigManager, notification_manager: NotificationManager):
        self.trading_db = trading_db
        self.config_manager = config_manager
        self.config = config_manager.get_config()['monitoring']
        self.notification_manager = notification_manager
        
        self.db_monitor = DatabaseMonitor(trading_db, config_manager)
        self.system_monitor = SystemMonitor()
        self.alert_manager = AlertManager(self.config.get('alerts', {}), notification_manager)
        self.monitoring_active = False
        
        # Initialize logger and error handler
        self.logger = Logger(__name__)
        self.error_handler = ErrorHandler()
        
        # Health check results storage
        self.health_history: List[HealthCheck] = []
        self.max_history = 1000  # Keep last 1000 health checks
    
    async def initialize(self):
        """Initialize monitoring system"""
        await self.db_monitor.initialize()
        self.logger.info("Trading system monitor initialized")
        
        # Register with the main trading bot for health callbacks
        if hasattr(self.trading_db, 'register_monitor'):
            self.trading_db.register_monitor(self)
    
    async def run_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all health checks"""
        results = {}
        
        # Database health checks
        pg_health = await self.db_monitor.check_postgresql_health()
        sqlite_health = await self.db_monitor.check_sqlite_health()
        
        # System health check
        system_health = self.system_monitor.check_system_health()
        
        results = {
            'postgresql': pg_health,
            'sqlite': sqlite_health,
            'system': system_health
        }
        
        # Process alerts
        for health_check in results.values():
            await self.alert_manager.process_health_check(health_check)
        
        # Store in history
        for health_check in results.values():
            self.health_history.append(health_check)
        
        # Trim history
        if len(self.health_history) > self.max_history:
            self.health_history = self.health_history[-self.max_history:]
        
        return results
    
    async def start_monitoring(self, interval: int = 60):
        """Start continuous monitoring"""
        self.monitoring_active = True
        logger.info(f"Starting continuous monitoring (interval: {interval}s)")
        
        while self.monitoring_active:
            try:
                results = await self.run_health_checks()
                
                # Log summary
                healthy_services = sum(1 for r in results.values() if r.status == ServiceStatus.HEALTHY)
                total_services = len(results)
                
                logger.info(f"Health check completed: {healthy_services}/{total_services} services healthy")
                
                # Wait for next interval
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error during monitoring cycle: {e}")
                await asyncio.sleep(10)  # Short sleep before retry
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        logger.info("Monitoring stopped")
    
    def get_system_status(self) -> Dict:
        """Get current system status summary"""
        if not self.health_history:
            return {"status": "No data", "services": {}}
        
        # Get latest health checks
        latest_checks = {}
        for check in reversed(self.health_history):
            if check.service not in latest_checks:
                latest_checks[check.service] = check
        
        # Calculate overall status
        statuses = [check.status for check in latest_checks.values()]
        if ServiceStatus.DOWN in statuses:
            overall_status = "DOWN"
        elif ServiceStatus.UNHEALTHY in statuses:
            overall_status = "UNHEALTHY"
        elif ServiceStatus.DEGRADED in statuses:
            overall_status = "DEGRADED"
        else:
            overall_status = "HEALTHY"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "services": {
                service: {
                    "status": check.status.value,
                    "message": check.message,
                    "response_time": check.response_time,
                    "timestamp": check.timestamp.isoformat(),
                    "metrics": check.metrics
                }
                for service, check in latest_checks.items()
            },
            "active_alerts": len([a for a in self.alert_manager.alerts if not a.resolved]),
            "total_health_checks": len(self.health_history)
        }

# Example configuration
EXAMPLE_CONFIG = {
    "postgresql": {
        "host": "localhost",
        "port": 5432,
        "database": "trading_db",
        "user": "trading_user",
        "password": "secure_password"
    },
    "sqlite_path": "data/realtime_cache.db",
    "alerts": {
        "email": {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "use_tls": True,
            "from": "alerts@trading-system.com",
            "to": ["admin@trading-system.com"],
            "username": "alerts@trading-system.com",
            "password": "app_password"
        },
        "webhook": {
            "url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
            "headers": {
                "Content-Type": "application/json"
            }
        }
    }
}

async def main():
    """Example usage with integrated components"""
    # Initialize system components
    config_manager = ConfigManager("config/config.yaml")
    trading_db = TradingDatabase(config_manager)
    await trading_db.initialize()
    
    # Initialize notification system (placeholder - will be implemented)
    notification_manager = NotificationManager(config_manager)
    
    # Initialize monitoring system
    monitor = TradingSystemMonitor(trading_db, config_manager, notification_manager)
    await monitor.initialize()
    
    # Run single health check
    results = await monitor.run_health_checks()
    
    print("Health Check Results:")
    print("=" * 50)
    for service, result in results.items():
        print(f"{service.upper()}: {result.status.value}")
        print(f"  Message: {result.message}")
        print(f"  Response Time: {result.response_time:.3f}s")
        print(f"  Key Metrics: {json.dumps(result.metrics, indent=2)}")
        print()
    
    # Print system status
    status = monitor.get_system_status()
    print(f"Overall System Status: {status['status']}")
    
    # Start continuous monitoring (uncomment to run)
    # await monitor.start_monitoring(interval=30)

if __name__ == "__main__":
    asyncio.run(main())
