#!/usr/bin/env python3
"""
System Monitoring and Health Checks
Monitors all system components and provides alerts
"""

import time
import threading
import logging
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class SystemMonitor:
    """Monitors system health and performance"""
    
    def __init__(self, services: Dict):
        self.services = services
        self.running = False
        self.monitoring_interval = 60  # 1 minute
        
        # Health status
        self.health_status = {
            'overall': 'unknown',
            'services': {},
            'system': {},
            'alerts': []
        }
        
        # Performance metrics
        self.metrics = {
            'system': [],
            'services': [],
            'alerts': []
        }
        
        # Alert thresholds
        self.thresholds = {
            'cpu_usage': 80.0,  # Percent
            'memory_usage': 85.0,  # Percent
            'disk_usage': 90.0,  # Percent
            'response_time': 5.0,  # Seconds
            'error_rate': 10.0  # Percent
        }
        
        logger.info("📊 System Monitor initialized")
    
    def start_monitoring(self):
        """Start system monitoring"""
        try:
            self.running = True
            
            logger.info("🚀 System monitoring started")
            
            while self.running:
                try:
                    # Check system health
                    self._check_system_health()
                    
                    # Check service health
                    self._check_service_health()
                    
                    # Update overall status
                    self._update_overall_status()
                    
                    # Clean old metrics
                    self._cleanup_old_metrics()
                    
                    time.sleep(self.monitoring_interval)
                    
                except Exception as e:
                    logger.error(f"❌ Monitoring loop error: {e}")
                    time.sleep(30)  # Wait before retry
                    
        except Exception as e:
            logger.error(f"❌ Monitoring start error: {e}")
            self.running = False
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.running = False
        logger.info("⏹️ System monitoring stopped")
    
    def _check_system_health(self):
        """Check system resource health"""
        try:
            timestamp = datetime.now()
            
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            # System load
            load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
            
            system_metrics = {
                'timestamp': timestamp.isoformat(),
                'cpu_usage': round(cpu_usage, 2),
                'memory_usage': round(memory_usage, 2),
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'disk_usage': round(disk_usage, 2),
                'disk_free_gb': round(disk.free / (1024**3), 2),
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'process_count': process_count,
                'load_average': round(load_avg, 2)
            }
            
            # Add to metrics history
            self.metrics['system'].append(system_metrics)
            
            # Update health status
            self.health_status['system'] = system_metrics
            
            # Check thresholds and create alerts
            self._check_system_thresholds(system_metrics)
            
        except Exception as e:
            logger.error(f"❌ System health check error: {e}")
    
    def _check_service_health(self):
        """Check health of all services"""
        try:
            timestamp = datetime.now()
            service_status = {}
            
            for service_name, service_data in self.services.items():
                try:
                    status = self._check_individual_service(service_name, service_data)
                    service_status[service_name] = status
                    
                except Exception as e:
                    logger.error(f"❌ Service check error for {service_name}: {e}")
                    service_status[service_name] = {
                        'status': 'error',
                        'error': str(e),
                        'last_check': timestamp.isoformat()
                    }
            
            # Update health status
            self.health_status['services'] = service_status
            
            # Add to metrics history
            service_metrics = {
                'timestamp': timestamp.isoformat(),
                'services': service_status
            }
            self.metrics['services'].append(service_metrics)
            
        except Exception as e:
            logger.error(f"❌ Service health check error: {e}")
    
    def _check_individual_service(self, service_name: str, service_data: Dict) -> Dict:
        """Check health of individual service"""
        try:
            status = {
                'status': 'unknown',
                'last_check': datetime.now().isoformat(),
                'details': {}
            }
            
            if service_name == 'bot':
                # Check trading bot
                if 'core' in service_data:
                    bot_core = service_data['core']
                    if hasattr(bot_core, 'running') and bot_core.running:
                        status['status'] = 'healthy'
                        status['details'] = {
                            'running': True,
                            'paper_mode': getattr(bot_core, 'paper_mode', True)
                        }
                    else:
                        status['status'] = 'stopped'
                        
            elif service_name == 'webapp':
                # Check Flask webapp
                if 'thread' in service_data:
                    thread = service_data['thread']
                    if thread.is_alive():
                        status['status'] = 'healthy'
                        status['details'] = {'thread_alive': True}
                    else:
                        status['status'] = 'stopped'
                        
            elif service_name == 'websocket':
                # Check WebSocket server
                if 'manager' in service_data:
                    ws_manager = service_data['manager']
                    if hasattr(ws_manager, 'running') and ws_manager.running:
                        status['status'] = 'healthy'
                        status['details'] = {
                            'running': True,
                            'connected_clients': len(getattr(ws_manager, 'clients', set()))
                        }
                    else:
                        status['status'] = 'stopped'
                        
            elif service_name == 'scanner':
                # Check dynamic scanner
                if 'scanner' in service_data:
                    scanner = service_data['scanner']
                    if hasattr(scanner, 'running') and scanner.running:
                        status['status'] = 'healthy'
                        status['details'] = {
                            'running': True,
                            'scanned_stocks': len(getattr(scanner, 'scanned_stocks', {}))
                        }
                    else:
                        status['status'] = 'stopped'
                        
            elif service_name == 'execution':
                # Check execution manager
                if 'manager' in service_data:
                    exec_manager = service_data['manager']
                    if hasattr(exec_manager, 'running') and exec_manager.running:
                        status['status'] = 'healthy'
                        status['details'] = {
                            'running': True,
                            'active_stocks': len(getattr(exec_manager, 'active_stocks', {}))
                        }
                    else:
                        status['status'] = 'stopped'
                        
            else:
                # Generic service check
                if service_data.get('status') == 'running':
                    status['status'] = 'healthy'
                else:
                    status['status'] = 'stopped'
            
            return status
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
    
    def _check_system_thresholds(self, metrics: Dict):
        """Check system metrics against thresholds"""
        try:
            timestamp = datetime.now()
            
            # CPU threshold
            if metrics['cpu_usage'] > self.thresholds['cpu_usage']:
                self._create_alert(
                    'HIGH_CPU_USAGE',
                    f"CPU usage is {metrics['cpu_usage']}% (threshold: {self.thresholds['cpu_usage']}%)",
                    'warning',
                    {'cpu_usage': metrics['cpu_usage']}
                )
            
            # Memory threshold
            if metrics['memory_usage'] > self.thresholds['memory_usage']:
                self._create_alert(
                    'HIGH_MEMORY_USAGE',
                    f"Memory usage is {metrics['memory_usage']}% (threshold: {self.thresholds['memory_usage']}%)",
                    'warning',
                    {'memory_usage': metrics['memory_usage']}
                )
            
            # Disk threshold
            if metrics['disk_usage'] > self.thresholds['disk_usage']:
                self._create_alert(
                    'HIGH_DISK_USAGE',
                    f"Disk usage is {metrics['disk_usage']}% (threshold: {self.thresholds['disk_usage']}%)",
                    'critical',
                    {'disk_usage': metrics['disk_usage']}
                )
            
        except Exception as e:
            logger.error(f"❌ Threshold check error: {e}")
    
    def _create_alert(self, alert_type: str, message: str, severity: str, data: Dict = None):
        """Create system alert"""
        try:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'type': alert_type,
                'message': message,
                'severity': severity,
                'data': data or {},
                'resolved': False
            }
            
            # Add to alerts
            self.health_status['alerts'].append(alert)
            self.metrics['alerts'].append(alert)
            
            # Log alert
            if severity == 'critical':
                logger.critical(f"🚨 CRITICAL ALERT: {message}")
            elif severity == 'warning':
                logger.warning(f"⚠️ WARNING: {message}")
            else:
                logger.info(f"ℹ️ INFO: {message}")
                
        except Exception as e:
            logger.error(f"❌ Create alert error: {e}")
    
    def _update_overall_status(self):
        """Update overall system status"""
        try:
            # Check if any critical alerts exist
            critical_alerts = [a for a in self.health_status['alerts'] 
                             if a['severity'] == 'critical' and not a['resolved']]
            
            # Check service statuses
            service_issues = []
            for service_name, service_status in self.health_status['services'].items():
                if service_status['status'] in ['error', 'stopped']:
                    service_issues.append(service_name)
            
            # Determine overall status
            if critical_alerts:
                self.health_status['overall'] = 'critical'
            elif service_issues:
                self.health_status['overall'] = 'warning'
            elif self.health_status['system'].get('cpu_usage', 0) > 90:
                self.health_status['overall'] = 'warning'
            else:
                self.health_status['overall'] = 'healthy'
                
        except Exception as e:
            logger.error(f"❌ Update overall status error: {e}")
            self.health_status['overall'] = 'error'
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory issues"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)  # Keep 24 hours
            
            # Clean system metrics
            self.metrics['system'] = [
                m for m in self.metrics['system'] 
                if datetime.fromisoformat(m['timestamp']) > cutoff_time
            ]
            
            # Clean service metrics
            self.metrics['services'] = [
                m for m in self.metrics['services']
                if datetime.fromisoformat(m['timestamp']) > cutoff_time
            ]
            
            # Clean old alerts (keep for 7 days)
            alert_cutoff = datetime.now() - timedelta(days=7)
            self.metrics['alerts'] = [
                a for a in self.metrics['alerts']
                if datetime.fromisoformat(a['timestamp']) > alert_cutoff
            ]
            
            # Clean resolved alerts from health status (keep only last 10)
            resolved_alerts = [a for a in self.health_status['alerts'] if a['resolved']]
            unresolved_alerts = [a for a in self.health_status['alerts'] if not a['resolved']]
            
            self.health_status['alerts'] = unresolved_alerts + resolved_alerts[-10:]
            
        except Exception as e:
            logger.error(f"❌ Cleanup metrics error: {e}")
    
    def get_health_status(self) -> Dict:
        """Get current health status"""
        return self.health_status.copy()
    
    def get_system_metrics(self, hours: int = 1) -> List[Dict]:
        """Get system metrics for specified hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            return [
                m for m in self.metrics['system']
                if datetime.fromisoformat(m['timestamp']) > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"❌ Get system metrics error: {e}")
            return []
    
    def get_service_metrics(self, hours: int = 1) -> List[Dict]:
        """Get service metrics for specified hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            return [
                m for m in self.metrics['services']
                if datetime.fromisoformat(m['timestamp']) > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"❌ Get service metrics error: {e}")
            return []
    
    def get_alerts(self, resolved: bool = None, hours: int = 24) -> List[Dict]:
        """Get alerts with optional filtering"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            alerts = [
                a for a in self.metrics['alerts']
                if datetime.fromisoformat(a['timestamp']) > cutoff_time
            ]
            
            if resolved is not None:
                alerts = [a for a in alerts if a['resolved'] == resolved]
            
            # Sort by timestamp (newest first)
            alerts.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Get alerts error: {e}")
            return []
    
    def resolve_alert(self, alert_timestamp: str) -> bool:
        """Mark alert as resolved"""
        try:
            # Find and resolve alert in health status
            for alert in self.health_status['alerts']:
                if alert['timestamp'] == alert_timestamp:
                    alert['resolved'] = True
                    alert['resolved_at'] = datetime.now().isoformat()
                    break
            
            # Find and resolve alert in metrics
            for alert in self.metrics['alerts']:
                if alert['timestamp'] == alert_timestamp:
                    alert['resolved'] = True
                    alert['resolved_at'] = datetime.now().isoformat()
                    break
            
            logger.info(f"✅ Alert resolved: {alert_timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Resolve alert error: {e}")
            return False
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        try:
            # Get recent metrics
            recent_system = self.get_system_metrics(1)  # Last hour
            recent_services = self.get_service_metrics(1)
            recent_alerts = self.get_alerts(resolved=False, hours=24)
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': self.health_status['overall'],
                'uptime_hours': self._calculate_uptime(),
                'system_performance': self._calculate_system_performance(recent_system),
                'service_performance': self._calculate_service_performance(recent_services),
                'alert_summary': {
                    'critical_count': len([a for a in recent_alerts if a['severity'] == 'critical']),
                    'warning_count': len([a for a in recent_alerts if a['severity'] == 'warning']),
                    'total_unresolved': len(recent_alerts)
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Performance summary error: {e}")
            return {'error': str(e)}
    
    def _calculate_uptime(self) -> float:
        """Calculate system uptime in hours"""
        try:
            # Get system boot time
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.now() - boot_time
            return round(uptime.total_seconds() / 3600, 2)
            
        except Exception:
            return 0.0
    
    def _calculate_system_performance(self, metrics: List[Dict]) -> Dict:
        """Calculate system performance statistics"""
        try:
            if not metrics:
                return {}
            
            cpu_values = [m['cpu_usage'] for m in metrics]
            memory_values = [m['memory_usage'] for m in metrics]
            disk_values = [m['disk_usage'] for m in metrics]
            
            return {
                'cpu': {
                    'current': cpu_values[-1] if cpu_values else 0,
                    'average': round(sum(cpu_values) / len(cpu_values), 2),
                    'max': max(cpu_values),
                    'min': min(cpu_values)
                },
                'memory': {
                    'current': memory_values[-1] if memory_values else 0,
                    'average': round(sum(memory_values) / len(memory_values), 2),
                    'max': max(memory_values),
                    'min': min(memory_values)
                },
                'disk': {
                    'current': disk_values[-1] if disk_values else 0,
                    'average': round(sum(disk_values) / len(disk_values), 2),
                    'max': max(disk_values),
                    'min': min(disk_values)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ System performance calculation error: {e}")
            return {}
    
    def _calculate_service_performance(self, metrics: List[Dict]) -> Dict:
        """Calculate service performance statistics"""
        try:
            if not metrics:
                return {}
            
            performance = {}
            
            # Analyze each service
            for service_name in self.services.keys():
                service_statuses = []
                
                for metric in metrics:
                    service_data = metric.get('services', {}).get(service_name, {})
                    if service_data:
                        service_statuses.append(service_data['status'])
                
                if service_statuses:
                    healthy_count = service_statuses.count('healthy')
                    total_count = len(service_statuses)
                    
                    performance[service_name] = {
                        'availability_percent': round((healthy_count / total_count) * 100, 2),
                        'current_status': service_statuses[-1],
                        'total_checks': total_count,
                        'healthy_checks': healthy_count
                    }
            
            return performance
            
        except Exception as e:
            logger.error(f"❌ Service performance calculation error: {e}")
            return {}
    
    def export_metrics(self, hours: int = 24) -> Dict:
        """Export metrics for analysis"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'period_hours': hours,
                'system_metrics': [
                    m for m in self.metrics['system']
                    if datetime.fromisoformat(m['timestamp']) > cutoff_time
                ],
                'service_metrics': [
                    m for m in self.metrics['services']
                    if datetime.fromisoformat(m['timestamp']) > cutoff_time
                ],
                'alerts': [
                    a for a in self.metrics['alerts']
                    if datetime.fromisoformat(a['timestamp']) > cutoff_time
                ],
                'health_status': self.health_status,
                'performance_summary': self.get_performance_summary()
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"❌ Export metrics error: {e}")
            return {'error': str(e)}
    
    def update_thresholds(self, new_thresholds: Dict) -> bool:
        """Update monitoring thresholds"""
        try:
            for key, value in new_thresholds.items():
                if key in self.thresholds:
                    self.thresholds[key] = value
                    logger.info(f"✅ Updated threshold {key}: {value}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Update thresholds error: {e}")
            return False
    
    def get_thresholds(self) -> Dict:
        """Get current monitoring thresholds"""
        return self.thresholds.copy()
    
    def force_health_check(self) -> Dict:
        """Force immediate health check"""
        try:
            logger.info("🔍 Force health check initiated")
            
            self._check_system_health()
            self._check_service_health()
            self._update_overall_status()
            
            return {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'health_status': self.get_health_status()
            }
            
        except Exception as e:
            logger.error(f"❌ Force health check error: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Health check utilities
class HealthChecker:
    """Utility class for health checks"""
    
    @staticmethod
    def check_database_connection(db_path: str) -> bool:
        """Check database connection"""
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            return True
        except Exception:
            return False
    
    @staticmethod
    def check_network_connectivity(host: str = "8.8.8.8", port: int = 53, timeout: int = 3) -> bool:
        """Check network connectivity"""
        try:
            import socket
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except Exception:
            return False
    
    @staticmethod
    def check_file_permissions(file_path: str) -> Dict:
        """Check file permissions"""
        try:
            path = Path(file_path)
            return {
                'exists': path.exists(),
                'readable': os.access(path, os.R_OK) if path.exists() else False,
                'writable': os.access(path, os.W_OK) if path.exists() else False,
                'executable': os.access(path, os.X_OK) if path.exists() else False
            }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def check_disk_space(path: str = "/", min_free_gb: float = 1.0) -> Dict:
        """Check available disk space"""
        try:
            disk_usage = psutil.disk_usage(path)
            free_gb = disk_usage.free / (1024**3)
            
            return {
                'free_gb': round(free_gb, 2),
                'total_gb': round(disk_usage.total / (1024**3), 2),
                'used_percent': round((disk_usage.used / disk_usage.total) * 100, 2),
                'sufficient_space': free_gb >= min_free_gb
            }
        except Exception as e:
            return {'error': str(e)}


# Usage example and testing
if __name__ == "__main__":
    import time
    
    # Test system monitor
    print("📊 Testing System Monitor...")
    
    # Mock services for testing
    mock_services = {
        'bot': {
            'core': type('MockBot', (), {'running': True, 'paper_mode': True})(),
            'status': 'running'
        },
        'webapp': {
            'thread': threading.current_thread(),
            'status': 'running'
        }
    }
    
    monitor = SystemMonitor(mock_services)
    
    # Test health checks
    print("🔍 Running health checks...")
    
    # Force health check
    result = monitor.force_health_check()
    print(f"Health check result: {result['success']}")
    
    # Get health status
    health = monitor.get_health_status()
    print(f"Overall status: {health['overall']}")
    print(f"System CPU: {health['system'].get('cpu_usage', 0)}%")
    
    # Test performance summary
    summary = monitor.get_performance_summary()
    print(f"Uptime: {summary.get('uptime_hours', 0)} hours")
    
    # Test thresholds
    monitor.update_thresholds({'cpu_usage': 75.0})
    thresholds = monitor.get_thresholds()
    print(f"CPU threshold: {thresholds['cpu_usage']}%")
    
    # Test export
    export_data = monitor.export_metrics(1)
    print(f"Exported metrics for 1 hour: {len(export_data.get('system_metrics', []))} data points")
    
    print("✅ System monitor test completed")
