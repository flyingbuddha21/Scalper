#!/usr/bin/env python3
"""
Configuration Manager
Handles all application configuration and settings
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages application configuration"""
    
    def __init__(self, config_path: Path = None):
        if config_path is None:
            config_path = Path.cwd() / "config"
        
        self.config_path = config_path
        self.config_path.mkdir(exist_ok=True)
        
        # Configuration files
        self.config_files = {
            'app': self.config_path / 'app_config.json',
            'trading': self.config_path / 'trading_config.json',
            'gcp': self.config_path / 'gcp_config.json'
        }
        
        # Loaded configurations
        self.configs = {}
        
        # Load all configurations
        self.load_all_configs()
        
        logger.info("⚙️ Configuration Manager initialized")
    
    def load_all_configs(self):
        """Load all configuration files"""
        for config_name, config_file in self.config_files.items():
            try:
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        self.configs[config_name] = json.load(f)
                    logger.info(f"✅ Loaded {config_name} configuration")
                else:
                    logger.warning(f"⚠️ Configuration file not found: {config_file}")
                    self.configs[config_name] = {}
                    
            except Exception as e:
                logger.error(f"❌ Failed to load {config_name} config: {e}")
                self.configs[config_name] = {}
    
    def get_config(self, config_name: str, key_path: str = None, default: Any = None) -> Any:
        """Get configuration value"""
        try:
            if config_name not in self.configs:
                return default
            
            config = self.configs[config_name]
            
            if key_path is None:
                return config
            
            # Navigate nested keys
            keys = key_path.split('.')
            value = config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            
            return value
            
        except Exception as e:
            logger.debug(f"Config get error: {e}")
            return default
    
    def set_config(self, config_name: str, key_path: str, value: Any) -> bool:
        """Set configuration value"""
        try:
            if config_name not in self.configs:
                self.configs[config_name] = {}
            
            config = self.configs[config_name]
            keys = key_path.split('.')
            
            # Navigate to parent of target key
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set the value
            current[keys[-1]] = value
            
            # Save to file
            return self.save_config(config_name)
            
        except Exception as e:
            logger.error(f"❌ Config set error: {e}")
            return False
    
    def save_config(self, config_name: str) -> bool:
        """Save configuration to file"""
        try:
            if config_name not in self.configs:
                return False
            
            config_file = self.config_files[config_name]
            
            with open(config_file, 'w') as f:
                json.dump(self.configs[config_name], f, indent=2)
            
            logger.info(f"✅ Saved {config_name} configuration")
            return True
            
        except Exception as e:
            logger.error(f"❌ Config save error: {e}")
            return False
    
    def save_all_configs(self) -> Dict[str, bool]:
        """Save all configurations"""
        results = {}
        
        for config_name in self.configs.keys():
            results[config_name] = self.save_config(config_name)
        
        return results
    
    def get_app_config(self) -> Dict:
        """Get application configuration"""
        return self.get_config('app', default={})
    
    def get_trading_config(self) -> Dict:
        """Get trading configuration"""
        return self.get_config('trading', default={})
    
    def get_gcp_config(self) -> Dict:
        """Get GCP configuration"""
        return self.get_config('gcp', default={})
    
    def get_database_config(self) -> Dict:
        """Get database configuration"""
        return self.get_config('app', 'database', default={})
    
    def get_server_config(self) -> Dict:
        """Get server configuration"""
        return self.get_config('app', 'server', default={})
    
    def get_scanner_config(self) -> Dict:
        """Get scanner configuration"""
        return self.get_config('trading', 'scanner', default={})
    
    def get_risk_config(self) -> Dict:
        """Get risk management configuration"""
        return self.get_config('trading', 'risk_management', default={})
    
    def get_strategy_config(self) -> Dict:
        """Get strategy configuration"""
        return self.get_config('trading', 'strategies', default={})
    
    def update_scanner_interval(self, minutes: int) -> bool:
        """Update scanner interval"""
        return self.set_config('trading', 'scanner.scan_interval_minutes', minutes)
    
    def update_risk_settings(self, max_positions: int = None, 
                           max_daily_loss: float = None,
                           stop_loss_pct: float = None) -> bool:
        """Update risk management settings"""
        try:
            success = True
            
            if max_positions is not None:
                success &= self.set_config('trading', 'risk_management.max_positions', max_positions)
            
            if max_daily_loss is not None:
                success &= self.set_config('trading', 'risk_management.max_daily_loss', max_daily_loss)
            
            if stop_loss_pct is not None:
                success &= self.set_config('trading', 'risk_management.stop_loss_percentage', stop_loss_pct)
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Risk settings update error: {e}")
            return False
    
    def toggle_strategy(self, strategy_name: str, enabled: bool) -> bool:
        """Enable/disable a trading strategy"""
        return self.set_config('trading', f'strategies.{strategy_name}.enabled', enabled)
    
    def get_strategy_status(self, strategy_name: str) -> bool:
        """Check if strategy is enabled"""
        return self.get_config('trading', f'strategies.{strategy_name}.enabled', default=True)
    
    def get_enabled_strategies(self) -> Dict[str, Dict]:
        """Get all enabled strategies"""
        strategies = self.get_strategy_config()
        return {name: config for name, config in strategies.items() 
                if config.get('enabled', True)}
    
    def reset_to_defaults(self, config_name: str) -> bool:
        """Reset configuration to defaults"""
        try:
            if config_name == 'app':
                self.configs['app'] = self._get_default_app_config()
            elif config_name == 'trading':
                self.configs['trading'] = self._get_default_trading_config()
            elif config_name == 'gcp':
                self.configs['gcp'] = self._get_default_gcp_config()
            else:
                return False
            
            return self.save_config(config_name)
            
        except Exception as e:
            logger.error(f"❌ Reset config error: {e}")
            return False
    
    def _get_default_app_config(self) -> Dict:
        """Get default application configuration"""
        return {
            "app": {
                "name": "Scalping Trading Bot",
                "version": "1.0.0",
                "debug": False,
                "secret_key": "trading_bot_secret_key_2024"
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8080,
                "websocket_port": 9001
            },
            "database": {
                "trading_bot": "data/trading_bot.db",
                "paper_trades": "data/paper_trades.db",
                "market_data": "data/market_data.db",
                "scanner_cache": "data/scanner_cache.db",
                "volatility_data": "data/volatility_data.db",
                "execution_queue": "data/execution_queue.db"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/app.log"
            }
        }
    
    def _get_default_trading_config(self) -> Dict:
        """Get default trading configuration"""
        return {
            "paper_trading": {
                "initial_capital": 100000.0,
                "enable_slippage": True,
                "slippage_percentage": 0.05,
                "commission_per_trade": 20.0
            },
            "risk_management": {
                "max_positions": 10,
                "max_position_size": 1000,
                "max_daily_loss": 5000.0,
                "max_daily_trades": 100,
                "stop_loss_percentage": 0.5,
                "take_profit_percentage": 1.0
            },
            "scanner": {
                "scan_interval_minutes": 20,
                "max_stocks_to_scan": 50,
                "min_volatility_score": 30,
                "min_liquidity_score": 40,
                "top_stocks_count": 10
            },
            "execution": {
                "max_active_stocks": 10,
                "min_confidence_threshold": 70,
                "order_type": "MARKET",
                "enable_all_strategies": True
            },
            "strategies": {
                "Strategy1_BidAskScalping": {"enabled": True, "weight": 1.0},
                "Strategy2_VolumeSpike": {"enabled": True, "weight": 1.0},
                "Strategy3_TickMomentum": {"enabled": True, "weight": 1.0},
                "Strategy4_OrderBookImbalance": {"enabled": True, "weight": 1.0},
                "Strategy5_MicroTrend": {"enabled": True, "weight": 1.0},
                "Strategy6_SpreadCompression": {"enabled": True, "weight": 1.0},
                "Strategy7_PriceAction": {"enabled": True, "weight": 1.0},
                "Strategy8_VolumeProfile": {"enabled": True, "weight": 1.0},
                "Strategy9_TimeBasedMomentum": {"enabled": True, "weight": 1.0},
                "Strategy10_MultiTimeframe": {"enabled": True, "weight": 1.0}
            }
        }
    
    def _get_default_gcp_config(self) -> Dict:
        """Get default GCP configuration"""
        return {
            "gcp": {
                "project_id": "your-gcp-project-id",
                "region": "us-central1",
                "app_engine": {
                    "service": "default",
                    "runtime": "python39",
                    "instance_class": "F2"
                },
                "cloud_sql": {
                    "instance_name": "trading-bot-db",
                    "database_name": "trading_data",
                    "user": "trading_bot",
                    "region": "us-central1"
                }
            },
            "monitoring": {
                "enable_logging": True,
                "enable_metrics": True,
                "log_level": "INFO"
            }
        }
    
    def validate_config(self, config_name: str) -> Dict[str, Any]:
        """Validate configuration"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            config = self.configs.get(config_name, {})
            
            if config_name == 'trading':
                validation_result = self._validate_trading_config(config)
            elif config_name == 'app':
                validation_result = self._validate_app_config(config)
            elif config_name == 'gcp':
                validation_result = self._validate_gcp_config(config)
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {e}")
        
        return validation_result
    
    def _validate_trading_config(self, config: Dict) -> Dict:
        """Validate trading configuration"""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        # Validate paper trading settings
        paper_trading = config.get('paper_trading', {})
        initial_capital = paper_trading.get('initial_capital', 0)
        if initial_capital <= 0:
            result['errors'].append("Initial capital must be positive")
            result['valid'] = False
        
        # Validate risk management
        risk_mgmt = config.get('risk_management', {})
        max_positions = risk_mgmt.get('max_positions', 0)
        if max_positions <= 0 or max_positions > 50:
            result['errors'].append("Max positions must be between 1 and 50")
            result['valid'] = False
        
        stop_loss = risk_mgmt.get('stop_loss_percentage', 0)
        if stop_loss <= 0 or stop_loss > 10:
            result['warnings'].append("Stop loss percentage should be between 0.1% and 10%")
        
        # Validate scanner settings
        scanner = config.get('scanner', {})
        scan_interval = scanner.get('scan_interval_minutes', 0)
        if scan_interval < 5 or scan_interval > 120:
            result['warnings'].append("Scan interval should be between 5 and 120 minutes")
        
        return result
    
    def _validate_app_config(self, config: Dict) -> Dict:
        """Validate application configuration"""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        # Validate server settings
        server = config.get('server', {})
        port = server.get('port', 0)
        if port <= 0 or port > 65535:
            result['errors'].append("Server port must be between 1 and 65535")
            result['valid'] = False
        
        ws_port = server.get('websocket_port', 0)
        if ws_port <= 0 or ws_port > 65535:
            result['errors'].append("WebSocket port must be between 1 and 65535")
            result['valid'] = False
        
        if port == ws_port:
            result['errors'].append("Server and WebSocket ports must be different")
            result['valid'] = False
        
        return result
    
    def _validate_gcp_config(self, config: Dict) -> Dict:
        """Validate GCP configuration"""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        gcp = config.get('gcp', {})
        project_id = gcp.get('project_id', '')
        if not project_id or project_id == 'your-gcp-project-id':
            result['warnings'].append("GCP project ID should be configured for deployment")
        
        return result
    
    def export_config(self, config_name: str = None) -> Dict:
        """Export configuration for backup"""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'exported_by': 'config_manager'
            }
            
            if config_name:
                if config_name in self.configs:
                    export_data['configs'] = {config_name: self.configs[config_name]}
                else:
                    export_data['error'] = f"Configuration '{config_name}' not found"
            else:
                export_data['configs'] = self.configs.copy()
            
            return export_data
            
        except Exception as e:
            logger.error(f"❌ Config export error: {e}")
            return {'error': str(e)}
    
    def import_config(self, import_data: Dict, overwrite: bool = False) -> bool:
        """Import configuration from backup"""
        try:
            if 'configs' not in import_data:
                logger.error("❌ Invalid import data format")
                return False
            
            imported_configs = import_data['configs']
            
            for config_name, config_data in imported_configs.items():
                if config_name in self.configs and not overwrite:
                    logger.warning(f"⚠️ Skipping {config_name} (already exists, use overwrite=True)")
                    continue
                
                self.configs[config_name] = config_data
                self.save_config(config_name)
                logger.info(f"✅ Imported {config_name} configuration")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Config import error: {e}")
            return False
    
    def get_config_summary(self) -> Dict:
        """Get summary of all configurations"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'config_files': {},
                'validation_status': {},
                'total_configs': len(self.configs)
            }
            
            for config_name, config_file in self.config_files.items():
                summary['config_files'][config_name] = {
                    'file_path': str(config_file),
                    'exists': config_file.exists(),
                    'loaded': config_name in self.configs,
                    'size_kb': round(config_file.stat().st_size / 1024, 2) if config_file.exists() else 0
                }
                
                # Validate each config
                validation = self.validate_config(config_name)
                summary['validation_status'][config_name] = {
                    'valid': validation['valid'],
                    'error_count': len(validation['errors']),
                    'warning_count': len(validation['warnings'])
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Config summary error: {e}")
            return {'error': str(e)}
    
    def reload_config(self, config_name: str = None) -> bool:
        """Reload configuration from file"""
        try:
            if config_name:
                if config_name in self.config_files:
                    config_file = self.config_files[config_name]
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            self.configs[config_name] = json.load(f)
                        logger.info(f"✅ Reloaded {config_name} configuration")
                        return True
                    else:
                        logger.error(f"❌ Config file not found: {config_file}")
                        return False
                else:
                    logger.error(f"❌ Unknown config name: {config_name}")
                    return False
            else:
                # Reload all configs
                self.load_all_configs()
                logger.info("✅ Reloaded all configurations")
                return True
                
        except Exception as e:
            logger.error(f"❌ Config reload error: {e}")
            return False
    
    def create_backup(self, backup_path: Path = None) -> bool:
        """Create backup of all configuration files"""
        try:
            if backup_path is None:
                backup_path = Path.cwd() / "backups" / "config"
            
            backup_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            backup_info = {
                'timestamp': timestamp,
                'backed_up_files': [],
                'errors': []
            }
            
            for config_name, config_file in self.config_files.items():
                try:
                    if config_file.exists():
                        backup_file = backup_path / f"{config_name}_config_{timestamp}.json"
                        
                        import shutil
                        shutil.copy2(config_file, backup_file)
                        
                        backup_info['backed_up_files'].append({
                            'config': config_name,
                            'original': str(config_file),
                            'backup': str(backup_file)
                        })
                        
                        logger.info(f"✅ Backed up {config_name} config to {backup_file}")
                
                except Exception as e:
                    backup_info['errors'].append({
                        'config': config_name,
                        'error': str(e)
                    })
                    logger.error(f"❌ Backup failed for {config_name}: {e}")
            
            # Save backup manifest
            manifest_file = backup_path / f"config_backup_manifest_{timestamp}.json"
            with open(manifest_file, 'w') as f:
                json.dump(backup_info, f, indent=2)
            
            logger.info(f"📦 Configuration backup completed: {len(backup_info['backed_up_files'])} files")
            return True
            
        except Exception as e:
            logger.error(f"❌ Config backup error: {e}")
            return False


# Configuration validation utilities
class ConfigValidator:
    """Utility class for configuration validation"""
    
    @staticmethod
    def validate_port(port: int) -> bool:
        """Validate port number"""
        return 1 <= port <= 65535
    
    @staticmethod
    def validate_percentage(value: float, min_val: float = 0.0, max_val: float = 100.0) -> bool:
        """Validate percentage value"""
        return min_val <= value <= max_val
    
    @staticmethod
    def validate_positive_number(value: float) -> bool:
        """Validate positive number"""
        return value > 0
    
    @staticmethod
    def validate_string_not_empty(value: str) -> bool:
        """Validate non-empty string"""
        return isinstance(value, str) and len(value.strip()) > 0
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format"""
        import re
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return re.match(pattern, url) is not None


# Configuration templates
class ConfigTemplates:
    """Predefined configuration templates"""
    
    @staticmethod
    def get_development_config() -> Dict:
        """Get development configuration template"""
        return {
            "app": {
                "debug": True,
                "log_level": "DEBUG"
            },
            "server": {
                "host": "localhost",
                "port": 8080
            },
            "trading": {
                "paper_trading": {
                    "initial_capital": 50000.0
                },
                "risk_management": {
                    "max_positions": 5,
                    "max_daily_loss": 1000.0
                }
            }
        }
    
    @staticmethod
    def get_production_config() -> Dict:
        """Get production configuration template"""
        return {
            "app": {
                "debug": False,
                "log_level": "INFO"
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8080
            },
            "trading": {
                "paper_trading": {
                    "initial_capital": 100000.0
                },
                "risk_management": {
                    "max_positions": 10,
                    "max_daily_loss": 5000.0
                }
            }
        }
    
    @staticmethod
    def get_conservative_trading_config() -> Dict:
        """Get conservative trading configuration"""
        return {
            "risk_management": {
                "max_positions": 5,
                "max_daily_loss": 2000.0,
                "stop_loss_percentage": 0.3,
                "take_profit_percentage": 0.6
            },
            "execution": {
                "min_confidence_threshold": 80
            }
        }
    
    @staticmethod
    def get_aggressive_trading_config() -> Dict:
        """Get aggressive trading configuration"""
        return {
            "risk_management": {
                "max_positions": 15,
                "max_daily_loss": 10000.0,
                "stop_loss_percentage": 0.8,
                "take_profit_percentage": 1.5
            },
            "execution": {
                "min_confidence_threshold": 65
            }
        }


# Usage example and testing
if __name__ == "__main__":
    import tempfile
    
    # Test configuration manager
    print("⚙️ Testing Configuration Manager...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config"
        config_manager = ConfigManager(config_path)
        
        # Test setting and getting configuration
        config_manager.set_config('trading', 'scanner.scan_interval_minutes', 25)
        interval = config_manager.get_config('trading', 'scanner.scan_interval_minutes')
        print(f"📊 Scan interval: {interval} minutes")
        
        # Test strategy toggle
        config_manager.toggle_strategy('Strategy1_BidAskScalping', False)
        enabled = config_manager.get_strategy_status('Strategy1_BidAskScalping')
        print(f"🎯 Strategy 1 enabled: {enabled}")
        
        # Test configuration validation
        validation = config_manager.validate_config('trading')
        print(f"✅ Trading config valid: {validation['valid']}")
        
        # Test configuration summary
        summary = config_manager.get_config_summary()
        print(f"📋 Total configs: {summary['total_configs']}")
        
        # Test backup
        backup_success = config_manager.create_backup()
        print(f"📦 Backup created: {backup_success}")
        
        print("✅ Configuration manager test completed")
