#!/usr/bin/env python3
"""
Complete Files Generator for Trading Bot Production System
Auto-generates all HTML, JS, CSS, config files and directory structure
Integrated with user risk management and database system
"""

import os
import json
from pathlib import Path
from datetime import datetime
import secrets

class ProductionFilesGenerator:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.generated_files = []
        print(f"🚀 Initializing Production Files Generator at: {self.project_root}")
    
    def create_directory_structure(self):
        """Create the complete directory structure"""
        directories = [
            "templates",
            "static/css", 
            "static/js",
            "static/images",
            "config",
            "data",
            "logs", 
            "backups",
            "strategies",
            "indicators",
            "utils",
            "tests",
            "docs"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")
    
    def generate_config_files(self):
        """Generate all configuration files"""
        
        # Main configuration
        config = {
            "database": {
                "postgresql": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "trading_bot",
                    "user": "trading_user",
                    "password": "change_this_password"
                },
                "sqlite_path": "data/realtime_cache.db",
                "connection_pool_size": 20,
                "enable_logging": True,
                "backup_enabled": True,
                "backup_schedule": "daily"
            },
            "trading": {
                "trading_mode": "paper",
                "auto_schedule": True,
                "market_hours_ist": {
                    "market_open": "09:15",
                    "market_close": "15:30",
                    "pre_market_start": "09:00",
                    "after_market_end": "16:00"
                },
                "enable_premarket": False,
                "enable_afterhours": False,
                "default_user_id": "default_user",
                "max_users_concurrent": 10,
                "session_timeout_minutes": 480,
                "default_timeframe": "1min",
                "supported_timeframes": ["1min", "5min", "15min", "1h", "1d"]
            },
            "default_risk": {
                "capital": 100000.0,
                "risk_per_trade_percent": 2.0,
                "daily_loss_limit_percent": 5.0,
                "max_concurrent_trades": 5,
                "risk_reward_ratio": 2.0,
                "max_position_size_percent": 20.0,
                "stop_loss_percent": 3.0,
                "take_profit_percent": 6.0,
                "trading_start_time": "09:15",
                "trading_end_time": "15:30",
                "auto_square_off": True,
                "paper_trading_mode": True,
                "max_daily_trades": 50,
                "trailing_stop_loss": False,
                "bracket_order_enabled": True
            },
            "api": {
                "goodwill": {
                    "base_url": "https://api.goodwill.com"
                },
                "rate_limit_requests": 300,
                "rate_limit_window": 60,
                "timeout": 30,
                "max_retries": 3,
                "retry_delay": 1.0
            },
            "flask": {
                "debug": False,
                "host": "0.0.0.0",
                "port": 5000,
                "max_content_length": 16777216,
                "session_timeout": 3600,
                "csrf_enabled": True
            },
            "notifications": {
                "email_enabled": True,
                "sms_enabled": False,
                "webhook_enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587
            },
            "logging": {
                "level": "INFO",
                "file_path": "logs/trading_bot.log",
                "max_file_size_mb": 100,
                "backup_count": 7,
                "format_string": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                "enable_console": True,
                "enable_file": True
            }
        }
        
        config_path = self.project_root / "config" / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        self.generated_files.append(str(config_path))
        print(f"✅ Generated: {config_path}")
        
        # Environment variables template
        env_content = f"""# Trading Bot Environment Variables
DATABASE_URL=postgresql://trading_user:change_this_password@localhost:5432/trading_bot
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_bot
DB_USER=trading_user
DB_PASSWORD=change_this_password

FLASK_SECRET_KEY={secrets.token_hex(32)}
FLASK_DEBUG=False
FLASK_PORT=5000

GOODWILL_API_KEY=your_goodwill_api_key
GOODWILL_SECRET_KEY=your_goodwill_secret_key
GOODWILL_USER_ID=your_goodwill_user_id
GOODWILL_BASE_URL=https://api.goodwill.com

TRADING_MODE=paper
AUTO_SCHEDULE=true

EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
WEBHOOK_URL=your_webhook_url

LOG_LEVEL=INFO
ENVIRONMENT=production
"""
        
        env_path = self.project_root / ".env"
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        self.generated_files.append(str(env_path))
        print(f"✅ Generated: {env_path}")
    
    def generate_html_templates(self):
        """Generate all HTML templates"""
        
        # Base template
        base_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Advanced Trading Bot{% endblock %}</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Font Awesome -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- Custom CSS -->
    <link href="{{ url_for('static', filename='css/style.css') }}" rel="stylesheet">
    
    {% block extra_head %}{% endblock %}
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container-fluid">
            <a class="navbar-brand" href="{{ url_for('dashboard') }}">
                <i class="fas fa-chart-line"></i> Trading Bot
            </a>
            
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav me-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('dashboard') }}">
                            <i class="fas fa-tachometer-alt"></i> Dashboard
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('portfolio') }}">
                            <i class="fas fa-briefcase"></i> Portfolio
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('trades') }}">
                            <i class="fas fa-exchange-alt"></i> Trades
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('risk_management') }}">
                            <i class="fas fa-shield-alt"></i> Risk Management
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('strategies') }}">
                            <i class="fas fa-cogs"></i> Strategies
                        </a>
                    </li>
                </ul>
                
                <ul class="navbar-nav">
                    <li class="nav-item dropdown">
                        <a class="nav-link dropdown-toggle" href="#" id="userDropdown" role="button" data-bs-toggle="dropdown">
                            <i class="fas fa-user"></i> {{ session.get('username', 'User') }}
                        </a>
                        <ul class="dropdown-menu">
                            <li><a class="dropdown-item" href="{{ url_for('profile') }}">Profile</a></li>
                            <li><a class="dropdown-item" href="{{ url_for('settings') }}">Settings</a></li>
                            <li><hr class="dropdown-divider"></li>
                            <li><a class="dropdown-item" href="{{ url_for('logout') }}">Logout</a></li>
                        </ul>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <!-- Alert Messages -->
    <div class="container-fluid mt-3">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ 'danger' if category == 'error' else category }} alert-dismissible fade show" role="alert">
                        {{ message }}
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}
    </div>

    <!-- Main Content -->
    <main class="container-fluid">
        {% block content %}{% endblock %}
    </main>

    <!-- Footer -->
    <footer class="bg-dark text-light text-center py-3 mt-5">
        <div class="container">
            <p>&copy; {{ datetime.now().year }} Advanced Trading Bot. All rights reserved.</p>
            <p><small>Last updated: <span id="lastUpdate">{{ datetime.now().strftime('%Y-%m-%d %H:%M:%S') }}</span></small></p>
        </div>
    </footer>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <!-- Custom JS -->
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
    
    {% block extra_scripts %}{% endblock %}
</body>
</html>"""
        
        # Login template
        login_template = """{% extends "base.html" %}

{% block title %}Login - Trading Bot{% endblock %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-6 col-lg-4">
        <div class="card shadow">
            <div class="card-header text-center">
                <h4><i class="fas fa-sign-in-alt"></i> Login</h4>
            </div>
            <div class="card-body">
                <form method="POST">
                    <div class="mb-3">
                        <label for="username" class="form-label">Username or Email</label>
                        <input type="text" class="form-control" id="username" name="username" required>
                    </div>
                    <div class="mb-3">
                        <label for="password" class="form-label">Password</label>
                        <input type="password" class="form-control" id="password" name="password" required>
                    </div>
                    <div class="mb-3 form-check">
                        <input type="checkbox" class="form-check-input" id="remember" name="remember">
                        <label class="form-check-label" for="remember">Remember me</label>
                    </div>
                    <button type="submit" class="btn btn-primary w-100">Login</button>
                </form>
                <hr>
                <div class="text-center">
                    <a href="{{ url_for('register') }}" class="text-decoration-none">Don't have an account? Register</a>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}"""

        # Dashboard template
        dashboard_template = """{% extends "base.html" %}

{% block title %}Dashboard - Trading Bot{% endblock %}

{% block extra_head %}
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .status-badge {
        font-size: 0.9rem;
        padding: 5px 10px;
    }
</style>
{% endblock %}

{% block content %}
<div class="row">
    <div class="col-12">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h2><i class="fas fa-tachometer-alt"></i> Trading Dashboard</h2>
            <div>
                <span class="badge bg-success status-badge me-2" id="botStatus">Bot Active</span>
                <span class="badge bg-info status-badge" id="tradingMode">Paper Trading</span>
            </div>
        </div>
    </div>
</div>

<!-- Key Metrics Row -->
<div class="row">
    <div class="col-md-3">
        <div class="metric-card text-center">
            <div class="metric-value" id="portfolioValue">₹1,00,000</div>
            <div>Portfolio Value</div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="metric-card text-center">
            <div class="metric-value" id="dailyPnL">₹0</div>
            <div>Daily P&L</div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="metric-card text-center">
            <div class="metric-value" id="totalTrades">0</div>
            <div>Total Trades</div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="metric-card text-center">
            <div class="metric-value" id="winRate">0%</div>
            <div>Win Rate</div>
        </div>
    </div>
</div>

<!-- Charts and Tables Row -->
<div class="row">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-chart-line"></i> P&L Chart</h5>
            </div>
            <div class="card-body">
                <canvas id="pnlChart" height="300"></canvas>
            </div>
        </div>
    </div>
    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-exclamation-triangle"></i> Risk Alerts</h5>
            </div>
            <div class="card-body" id="riskAlerts">
                <div class="text-muted text-center">No active alerts</div>
            </div>
        </div>
    </div>
</div>

<!-- Recent Trades -->
<div class="row mt-4">
    <div class="col-12">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-history"></i> Recent Trades</h5>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-hover" id="recentTradesTable">
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Symbol</th>
                                <th>Action</th>
                                <th>Quantity</th>
                                <th>Price</th>
                                <th>P&L</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody id="recentTradesBody">
                            <tr>
                                <td colspan="7" class="text-center text-muted">No trades yet</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_scripts %}
<script src="{{ url_for('static', filename='js/dashboard.js') }}"></script>
{% endblock %}"""

        # Risk Management template
        risk_management_template = """{% extends "base.html" %}

{% block title %}Risk Management - Trading Bot{% endblock %}

{% block content %}
<div class="row">
    <div class="col-12">
        <h2><i class="fas fa-shield-alt"></i> Risk Management</h2>
        <p class="text-muted">Configure your trading risk parameters</p>
    </div>
</div>

<div class="row">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <h5>Risk Parameters</h5>
            </div>
            <div class="card-body">
                <form id="riskConfigForm">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label for="capital" class="form-label">Capital (₹)</label>
                                <input type="number" class="form-control" id="capital" name="capital" value="100000" min="1000" step="1000" required>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label for="riskPerTrade" class="form-label">Risk per Trade (%)</label>
                                <input type="number" class="form-control" id="riskPerTrade" name="risk_per_trade_percent" value="2.0" min="0.1" max="10" step="0.1" required>
                            </div>
                        </div>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label for="dailyLossLimit" class="form-label">Daily Loss Limit (%)</label>
                                <input type="number" class="form-control" id="dailyLossLimit" name="daily_loss_limit_percent" value="5.0" min="1" max="20" step="0.5" required>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label for="maxTrades" class="form-label">Max Concurrent Trades</label>
                                <input type="number" class="form-control" id="maxTrades" name="max_concurrent_trades" value="5" min="1" max="50" required>
                            </div>
                        </div>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label for="stopLoss" class="form-label">Stop Loss (%)</label>
                                <input type="number" class="form-control" id="stopLoss" name="stop_loss_percent" value="3.0" min="0.5" max="20" step="0.1" required>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label for="takeProfit" class="form-label">Take Profit (%)</label>
                                <input type="number" class="form-control" id="takeProfit" name="take_profit_percent" value="6.0" min="1" max="50" step="0.1" required>
                            </div>
                        </div>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label for="tradingStart" class="form-label">Trading Start Time</label>
                                <input type="time" class="form-control" id="tradingStart" name="trading_start_time" value="09:15" required>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label for="tradingEnd" class="form-label">Trading End Time</label>
                                <input type="time" class="form-control" id="tradingEnd" name="trading_end_time" value="15:30" required>
                            </div>
                        </div>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <div class="form-check mb-3">
                                <input class="form-check-input" type="checkbox" id="autoSquareOff" name="auto_square_off" checked>
                                <label class="form-check-label" for="autoSquareOff">
                                    Auto Square Off at Market Close
                                </label>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="form-check mb-3">
                                <input class="form-check-input" type="checkbox" id="paperTrading" name="paper_trading_mode" checked>
                                <label class="form-check-label" for="paperTrading">
                                    Paper Trading Mode
                                </label>
                            </div>
                        </div>
                    </div>
                    
                    <button type="submit" class="btn btn-primary">
                        <i class="fas fa-save"></i> Save Configuration
                    </button>
                </form>
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <h5>Risk Summary</h5>
            </div>
            <div class="card-body">
                <div class="mb-3">
                    <label class="form-label">Max Position Size</label>
                    <div class="form-control-plaintext" id="maxPositionSize">₹20,000</div>
                </div>
                <div class="mb-3">
                    <label class="form-label">Daily Loss Limit</label>
                    <div class="form-control-plaintext" id="dailyLossAmount">₹5,000</div>
                </div>
                <div class="mb-3">
                    <label class="form-label">Risk per Trade</label>
                    <div class="form-control-plaintext" id="riskPerTradeAmount">₹2,000</div>
                </div>
                <div class="mb-3">
                    <label class="form-label">Risk-Reward Ratio</label>
                    <div class="form-control-plaintext" id="riskRewardRatio">1:2</div>
                </div>
            </div>
        </div>
        
        <div class="card mt-3">
            <div class="card-header">
                <h5>Current Status</h5>
            </div>
            <div class="card-body">
                <div class="d-flex justify-content-between mb-2">
                    <span>Daily P&L:</span>
                    <span id="currentDailyPnL" class="fw-bold">₹0</span>
                </div>
                <div class="d-flex justify-content-between mb-2">
                    <span>Active Trades:</span>
                    <span id="activeTrades">0</span>
                </div>
                <div class="d-flex justify-content-between mb-2">
                    <span>Risk Status:</span>
                    <span id="riskStatus" class="badge bg-success">Safe</span>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_scripts %}
<script src="{{ url_for('static', filename='js/risk-management.js') }}"></script>
{% endblock %}"""

        # Save templates
        templates = {
            "base.html": base_template,
            "login.html": login_template,
            "dashboard.html": dashboard_template,
            "risk_management.html": risk_management_template
        }
        
        for filename, content in templates.items():
            template_path = self.project_root / "templates" / filename
            with open(template_path, 'w') as f:
                f.write(content)
            self.generated_files.append(str(template_path))
            print(f"✅ Generated template: {filename}")
    
    def generate_css_files(self):
        """Generate CSS files"""
        
        main_css = """/* Advanced Trading Bot Custom Styles */
:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --success-color: #28a745;
    --danger-color: #dc3545;
    --warning-color: #ffc107;
    --info-color: #17a2b8;
    --dark-color: #343a40;
    --light-color: #f8f9fa;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f5f7fa;
}

.navbar-brand {
    font-weight: bold;
    font-size: 1.5rem;
}

.metric-card {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    color: white;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    margin-bottom: 5px;
}

.card {
    border: none;
    border-radius: 15px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}

.card-header {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-bottom: 1px solid #dee2e6;
    border-radius: 15px 15px 0 0 !important;
    padding: 15px 20px;
}

.btn {
    border-radius: 10px;
    font-weight: 500;
    transition: all 0.3s ease;
}

.btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
}

.status-badge {
    font-size: 0.9rem;
    padding: 8px 15px;
    border-radius: 20px;
    font-weight: 500;
}

.table {
    border-radius: 10px;
    overflow: hidden;
}

.table thead th {
    background-color: var(--dark-color);
    color: white;
    border: none;
    font-weight: 600;
}

.table tbody tr:hover {
    background-color: rgba(102, 126, 234, 0.1);
}

.form-control, .form-select {
    border-radius: 10px;
    border: 2px solid #e9ecef;
    transition: border-color 0.3s ease;
}

.form-control:focus, .form-select:focus {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
}

.alert {
    border: none;
    border-radius: 10px;
    font-weight: 500;
}

.nav-link {
    font-weight: 500;
    transition: color 0.3s ease;
}

.nav-link:hover {
    color: var(--primary-color) !important;
}

.dropdown-menu {
    border: none;
    border-radius: 10px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

footer {
    margin-top: auto;
}

/* Custom animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.fade-in-up {
    animation: fadeInUp 0.6s ease;
}

/* Responsive design */
@media (max-width: 768px) {
    .metric-card {
        margin-bottom: 15px;
    }
    
    .metric-value {
        font-size: 1.5rem;
    }
    
    .card-body {
        padding: 15px;
    }
}

/* Loading spinner */
.spinner {
    width: 40px;
    height: 40px;
    border: 4px solid #f3f3f3;
    border-top: 4px solid var(--primary-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 20px auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Chart container */
.chart-container {
    position: relative;
    height: 300px;
    width: 100%;
}

/* Risk status indicators */
.risk-safe { color: var(--success-color); }
.risk-warning { color: var(--warning-color); }
.risk-danger { color: var(--danger-color); }

/* Trading status indicators */
.status-active { background-color: var(--success-color); }
.status-inactive { background-color: var(--danger-color); }
.status-paper { background-color: var(--info-color); }
.status-live { background-color: var(--warning-color); }
"""
        
        css_path = self.project_root / "static" / "css" / "style.css"
        with open(css_path, 'w') as f:
            f.write(main_css)
        
        self.generated_files.append(str(css_path))
        print(f"✅ Generated: static/css/style.css")
    
    def generate_js_files(self):
        """Generate JavaScript files"""
        
        # Main JavaScript file
        main_js = """// Advanced Trading Bot Main JavaScript
class TradingBotApp {
    constructor() {
        this.apiBase = '/api';
        this.updateInterval = 5000; // 5 seconds
        this.charts = {};
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.startDataUpdates();
        console.log('Trading Bot App initialized');
    }
    
    setupEventListeners() {
        // Update last update time
        this.updateLastUpdateTime();
        setInterval(() => this.updateLastUpdateTime(), 1000);
        
        // Handle form submissions
        document.addEventListener('submit', (e) => {
            if (e.target.id === 'riskConfigForm') {
                e.preventDefault();
                this.saveRiskConfiguration(e.target);
            }
        });
        
        // Handle bot control buttons
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('bot-control')) {
                this.handleBotControl(e.target);
            }
        });
    }
    
    updateLastUpdateTime() {
        const lastUpdateElement = document.getElementById('lastUpdate');
        if (lastUpdateElement) {
            lastUpdateElement.textContent = new Date().toLocaleString();
        }
    }
    
    startDataUpdates() {
        // Start periodic updates for dashboard data
        if (window.location.pathname === '/dashboard' || window.location.pathname === '/') {
            this.updateDashboard();
            setInterval(() => this.updateDashboard(), this.updateInterval);
        }
        
        // Start updates for risk management page
        if (window.location.pathname === '/risk-management') {
            this.updateRiskStatus();
            setInterval(() => this.updateRiskStatus(), this.updateInterval);
        }
    }
    
    async updateDashboard() {
        try {
            const response = await fetch(`${this.apiBase}/dashboard-data`);
            const data = await response.json();
            
            if (data.success) {
                this.updateDashboardMetrics(data.data);
                this.updateRecentTrades(data.data.recent_trades);
                this.updateRiskAlerts(data.data.alerts);
                this.updatePnLChart(data.data.pnl_data);
            }
        } catch (error) {
            console.error('Error updating dashboard:', error);
        }
    }
    
    updateDashboardMetrics(data) {
        // Update portfolio value
        const portfolioValue = document.getElementById('portfolioValue');
        if (portfolioValue && data.portfolio_value !== undefined) {
            portfolioValue.textContent = this.formatCurrency(data.portfolio_value);
        }
        
        // Update daily P&L
        const dailyPnL = document.getElementById('dailyPnL');
        if (dailyPnL && data.daily_pnl !== undefined) {
            dailyPnL.textContent = this.formatCurrency(data.daily_pnl);
            dailyPnL.className = `metric-value ${data.daily_pnl >= 0 ? 'text-success' : 'text-danger'}`;
        }
        
        // Update total trades
        const totalTrades = document.getElementById('totalTrades');
        if (totalTrades && data.total_trades !== undefined) {
            totalTrades.textContent = data.total_trades;
        }
        
        // Update win rate
        const winRate = document.getElementById('winRate');
        if (winRate && data.win_rate !== undefined) {
            winRate.textContent = `${data.win_rate.toFixed(1)}%`;
        }
        
        // Update bot status
        const botStatus = document.getElementById('botStatus');
        if (botStatus && data.bot_status) {
            botStatus.textContent = data.bot_status.is_running ? 'Bot Active' : 'Bot Stopped';
            botStatus.className = `badge status-badge me-2 ${data.bot_status.is_running ? 'bg-success' : 'bg-danger'}`;
        }
        
        // Update trading mode
        const tradingMode = document.getElementById('tradingMode');
        if (tradingMode && data.bot_status) {
            tradingMode.textContent = data.bot_status.trading_mode === 'paper' ? 'Paper Trading' : 'Live Trading';
            tradingMode.className = `badge status-badge ${data.bot_status.trading_mode === 'paper' ? 'bg-info' : 'bg-warning'}`;
        }
    }
    
    updateRecentTrades(trades) {
        const tbody = document.getElementById('recentTradesBody');
        if (!tbody || !trades) return;
        
        if (trades.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No trades yet</td></tr>';
            return;
        }
        
        tbody.innerHTML = trades.map(trade => `
            <tr>
                <td>${new Date(trade.entry_time).toLocaleTimeString()}</td>
                <td>${trade.symbol}</td>
                <td><span class="badge ${trade.action === 'BUY' ? 'bg-success' : 'bg-danger'}">${trade.action}</span></td>
                <td>${trade.quantity}</td>
                <td>₹${trade.entry_price.toFixed(2)}</td>
                <td class="${trade.realized_pnl >= 0 ? 'text-success' : 'text-danger'}">
                    ${trade.realized_pnl ? this.formatCurrency(trade.realized_pnl) : '-'}
                </td>
                <td><span class="badge bg-${this.getStatusColor(trade.status)}">${trade.status}</span></td>
            </tr>
        `).join('');
    }
    
    updateRiskAlerts(alerts) {
        const alertsContainer = document.getElementById('riskAlerts');
        if (!alertsContainer || !alerts) return;
        
        if (alerts.length === 0) {
            alertsContainer.innerHTML = '<div class="text-muted text-center">No active alerts</div>';
            return;
        }
        
        alertsContainer.innerHTML = alerts.map(alert => `
            <div class="alert alert-${this.getAlertColor(alert.alert_level)} alert-sm mb-2">
                <strong>${alert.alert_type}:</strong> ${alert.message}
                <button type="button" class="btn-close btn-sm" onclick="app.acknowledgeAlert('${alert.alert_id}')"></button>
            </div>
        `).join('');
    }
    
    updatePnLChart(pnlData) {
        if (!pnlData || !document.getElementById('pnlChart')) return;
        
        const ctx = document.getElementById('pnlChart').getContext('2d');
        
        if (this.charts.pnl) {
            this.charts.pnl.destroy();
        }
        
        this.charts.pnl = new Chart(ctx, {
            type: 'line',
            data: {
                labels: pnlData.labels || [],
                datasets: [{
                    label: 'Daily P&L',
                    data: pnlData.values || [],
                    borderColor: 'rgb(102, 126, 234)',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.1,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return '₹' + value.toLocaleString();
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
    
    async updateRiskStatus() {
        try {
            const response = await fetch(`${this.apiBase}/risk-status`);
            const data = await response.json();
            
            if (data.success) {
                this.updateRiskMetrics(data.data);
            }
        } catch (error) {
            console.error('Error updating risk status:', error);
        }
    }
    
    updateRiskMetrics(data) {
        // Update current daily P&L
        const currentDailyPnL = document.getElementById('currentDailyPnL');
        if (currentDailyPnL && data.daily_pnl !== undefined) {
            currentDailyPnL.textContent = this.formatCurrency(data.daily_pnl);
            currentDailyPnL.className = `fw-bold ${data.daily_pnl >= 0 ? 'text-success' : 'text-danger'}`;
        }
        
        // Update active trades
        const activeTrades = document.getElementById('activeTrades');
        if (activeTrades && data.active_trades !== undefined) {
            activeTrades.textContent = data.active_trades;
        }
        
        // Update risk status
        const riskStatus = document.getElementById('riskStatus');
        if (riskStatus && data.risk_status) {
            riskStatus.textContent = data.risk_status;
            riskStatus.className = `badge ${this.getRiskStatusColor(data.risk_status)}`;
        }
        
        // Update calculated risk amounts
        if (data.risk_calculations) {
            const maxPositionSize = document.getElementById('maxPositionSize');
            if (maxPositionSize) {
                maxPositionSize.textContent = this.formatCurrency(data.risk_calculations.max_position_size);
            }
            
            const dailyLossAmount = document.getElementById('dailyLossAmount');
            if (dailyLossAmount) {
                dailyLossAmount.textContent = this.formatCurrency(data.risk_calculations.daily_loss_limit);
            }
            
            const riskPerTradeAmount = document.getElementById('riskPerTradeAmount');
            if (riskPerTradeAmount) {
                riskPerTradeAmount.textContent = this.formatCurrency(data.risk_calculations.risk_per_trade);
            }
        }
    }
    
    async saveRiskConfiguration(form) {
        try {
            const formData = new FormData(form);
            const configData = {};
            
            for (let [key, value] of formData.entries()) {
                if (value === 'on') {
                    configData[key] = true;
                } else if (key.includes('percent') || key === 'capital' || key === 'ratio') {
                    configData[key] = parseFloat(value);
                } else if (key.includes('trades') || key.includes('daily')) {
                    configData[key] = parseInt(value);
                } else {
                    configData[key] = value;
                }
            }
            
            // Handle checkboxes that might not be in FormData if unchecked
            configData.auto_square_off = formData.has('auto_square_off');
            configData.paper_trading_mode = formData.has('paper_trading_mode');
            
            const response = await fetch(`${this.apiBase}/update-risk-config`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(configData)
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showAlert('Risk configuration saved successfully!', 'success');
                this.updateRiskSummary(configData);
            } else {
                this.showAlert('Error saving configuration: ' + result.message, 'danger');
            }
        } catch (error) {
            console.error('Error saving risk configuration:', error);
            this.showAlert('Error saving configuration', 'danger');
        }
    }
    
    updateRiskSummary(config) {
        // Update risk summary with calculated values
        const capital = config.capital || 100000;
        const maxPositionSize = capital * (config.max_position_size_percent || 20) / 100;
        const dailyLossAmount = capital * (config.daily_loss_limit_percent || 5) / 100;
        const riskPerTradeAmount = capital * (config.risk_per_trade_percent || 2) / 100;
        const riskRewardRatio = `1:${config.risk_reward_ratio || 2}`;
        
        document.getElementById('maxPositionSize').textContent = this.formatCurrency(maxPositionSize);
        document.getElementById('dailyLossAmount').textContent = this.formatCurrency(dailyLossAmount);
        document.getElementById('riskPerTradeAmount').textContent = this.formatCurrency(riskPerTradeAmount);
        document.getElementById('riskRewardRatio').textContent = riskRewardRatio;
    }
    
    async handleBotControl(button) {
        const action = button.dataset.action;
        
        try {
            const response = await fetch(`${this.apiBase}/bot-control`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ action })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showAlert(`Bot ${action} successful!`, 'success');
            } else {
                this.showAlert(`Error: ${result.message}`, 'danger');
            }
        } catch (error) {
            console.error('Error controlling bot:', error);
            this.showAlert('Error controlling bot', 'danger');
        }
    }
    
    async acknowledgeAlert(alertId) {
        try {
            const response = await fetch(`${this.apiBase}/acknowledge-alert/${alertId}`, {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                // Remove alert from UI
                document.querySelector(`[onclick="app.acknowledgeAlert('${alertId}')"]`)?.closest('.alert')?.remove();
            }
        } catch (error) {
            console.error('Error acknowledging alert:', error);
        }
    }
    
    formatCurrency(amount) {
        return new Intl.NumberFormat('en-IN', {
            style: 'currency',
            currency: 'INR',
            minimumFractionDigits: 0,
            maximumFractionDigits: 0
        }).format(amount);
    }
    
    getStatusColor(status) {
        const colors = {
            'FILLED': 'success',
            'PENDING': 'warning',
            'CANCELLED': 'danger',
            'REJECTED': 'danger',
            'PARTIAL': 'info'
        };
        return colors[status] || 'secondary';
    }
    
    getAlertColor(level) {
        const colors = {
            'INFO': 'info',
            'WARNING': 'warning',
            'CRITICAL': 'danger'
        };
        return colors[level] || 'secondary';
    }
    
    getRiskStatusColor(status) {
        const colors = {
            'Safe': 'bg-success',
            'Warning': 'bg-warning',
            'Critical': 'bg-danger',
            'HALTED': 'bg-danger'
        };
        return colors[status] || 'bg-secondary';
    }
    
    showAlert(message, type = 'info') {
        const alertHtml = `
            <div class="alert alert-${type} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        const container = document.querySelector('.container-fluid');
        const firstChild = container.firstElementChild;
        firstChild.insertAdjacentHTML('beforebegin', alertHtml);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            const alert = container.querySelector('.alert');
            if (alert) {
                alert.remove();
            }
        }, 5000);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new TradingBotApp();
});

// Utility functions for risk management form
document.addEventListener('DOMContentLoaded', () => {
    const riskForm = document.getElementById('riskConfigForm');
    if (riskForm) {
        // Add real-time calculation updates
        const inputs = riskForm.querySelectorAll('input[type="number"]');
        inputs.forEach(input => {
            input.addEventListener('input', updateRiskCalculations);
        });
        
        // Initial calculation
        updateRiskCalculations();
    }
});

function updateRiskCalculations() {
    const capital = parseFloat(document.getElementById('capital')?.value) || 100000;
    const riskPerTrade = parseFloat(document.getElementById('riskPerTrade')?.value) || 2;
    const dailyLossLimit = parseFloat(document.getElementById('dailyLossLimit')?.value) || 5;
    const maxPositionSizePercent = 20; // Default 20%
    
    // Calculate and update displays
    const maxPositionSize = capital * maxPositionSizePercent / 100;
    const dailyLossAmount = capital * dailyLossLimit / 100;
    const riskPerTradeAmount = capital * riskPerTrade / 100;
    
    if (document.getElementById('maxPositionSize')) {
        document.getElementById('maxPositionSize').textContent = formatCurrency(maxPositionSize);
    }
    if (document.getElementById('dailyLossAmount')) {
        document.getElementById('dailyLossAmount').textContent = formatCurrency(dailyLossAmount);
    }
    if (document.getElementById('riskPerTradeAmount')) {
        document.getElementById('riskPerTradeAmount').textContent = formatCurrency(riskPerTradeAmount);
    }
}

function formatCurrency(amount) {
    return new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(amount);
}
"""
        
        # Dashboard specific JavaScript
        dashboard_js = """// Dashboard specific functionality
class DashboardManager {
    constructor() {
        this.charts = {};
        this.init();
    }
    
    init() {
        this.initializeCharts();
        this.loadInitialData();
    }
    
    initializeCharts() {
        // Initialize P&L Chart
        const pnlCtx = document.getElementById('pnlChart');
        if (pnlCtx) {
            this.charts.pnl = new Chart(pnlCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Daily P&L',
                        data: [],
                        borderColor: 'rgb(102, 126, 234)',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.1,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                callback: function(value) {
                                    return '₹' + value.toLocaleString();
                                }
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return 'P&L: ₹' + context.parsed.y.toLocaleString();
                                }
                            }
                        }
                    }
                }
            });
        }
    }
    
    async loadInitialData() {
        try {
            // Load recent performance data for chart
            const response = await fetch('/api/performance-history?days=30');
            const data = await response.json();
            
            if (data.success && this.charts.pnl) {
                this.charts.pnl.data.labels = data.data.labels;
                this.charts.pnl.data.datasets[0].data = data.data.values;
                this.charts.pnl.update();
            }
        } catch (error) {
            console.error('Error loading initial dashboard data:', error);
        }
    }
}

// Initialize dashboard manager
if (document.getElementById('pnlChart')) {
    document.addEventListener('DOMContentLoaded', () => {
        window.dashboardManager = new DashboardManager();
    });
}
"""
        
        # Risk management specific JavaScript
        risk_management_js = """// Risk Management specific functionality
class RiskManagementManager {
    constructor() {
        this.init();
    }
    
    init() {
        this.loadCurrentConfig();
        this.setupFormValidation();
        this.setupRealTimeUpdates();
    }
    
    async loadCurrentConfig() {
        try {
            const response = await fetch('/api/user-config');
            const data = await response.json();
            
            if (data.success) {
                this.populateForm(data.data);
                this.updateRiskSummary(data.data);
            }
        } catch (error) {
            console.error('Error loading current config:', error);
        }
    }
    
    populateForm(config) {
        // Populate form fields with current configuration
        Object.keys(config).forEach(key => {
            const element = document.getElementById(this.getElementId(key));
            if (element) {
                if (element.type === 'checkbox') {
                    element.checked = config[key];
                } else {
                    element.value = config[key];
                }
            }
        });
    }
    
    getElementId(configKey) {
        // Map config keys to form element IDs
        const mapping = {
            'capital': 'capital',
            'risk_per_trade_percent': 'riskPerTrade',
            'daily_loss_limit_percent': 'dailyLossLimit',
            'max_concurrent_trades': 'maxTrades',
            'stop_loss_percent': 'stopLoss',
            'take_profit_percent': 'takeProfit',
            'trading_start_time': 'tradingStart',
            'trading_end_time': 'tradingEnd',
            'auto_square_off': 'autoSquareOff',
            'paper_trading_mode': 'paperTrading'
        };
        return mapping[configKey] || configKey;
    }
    
    setupFormValidation() {
        const form = document.getElementById('riskConfigForm');
        if (!form) return;
        
        // Add validation for risk parameters
        const riskPerTrade = document.getElementById('riskPerTrade');
        const dailyLossLimit = document.getElementById('dailyLossLimit');
        const capital = document.getElementById('capital');
        
        if (riskPerTrade) {
            riskPerTrade.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                if (value > 10) {
                    e.target.setCustomValidity('Risk per trade should not exceed 10%');
                } else {
                    e.target.setCustomValidity('');
                }
            });
        }
        
        if (dailyLossLimit) {
            dailyLossLimit.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                if (value > 20) {
                    e.target.setCustomValidity('Daily loss limit should not exceed 20%');
                } else {
                    e.target.setCustomValidity('');
                }
            });
        }
        
        if (capital) {
            capital.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                if (value < 1000) {
                    e.target.setCustomValidity('Minimum capital is ₹1,000');
                } else {
                    e.target.setCustomValidity('');
                }
            });
        }
    }
    
    setupRealTimeUpdates() {
        // Update risk calculations in real-time as user types
        const form = document.getElementById('riskConfigForm');
        if (!form) return;
        
        const inputs = form.querySelectorAll('input[type="number"]');
        inputs.forEach(input => {
            input.addEventListener('input', () => {
                this.updateRiskCalculationsRealTime();
            });
        });
    }
    
    updateRiskCalculationsRealTime() {
        const capital = parseFloat(document.getElementById('capital')?.value) || 100000;
        const riskPerTrade = parseFloat(document.getElementById('riskPerTrade')?.value) || 2;
        const dailyLossLimit = parseFloat(document.getElementById('dailyLossLimit')?.value) || 5;
        const stopLoss = parseFloat(document.getElementById('stopLoss')?.value) || 3;
        const takeProfit = parseFloat(document.getElementById('takeProfit')?.value) || 6;
        
        // Calculate derived values
        const maxPositionSize = capital * 0.2; // 20% default
        const dailyLossAmount = capital * dailyLossLimit / 100;
        const riskPerTradeAmount = capital * riskPerTrade / 100;
        const riskRewardRatio = takeProfit / stopLoss;
        
        // Update displays
        this.updateDisplay('maxPositionSize', this.formatCurrency(maxPositionSize));
        this.updateDisplay('dailyLossAmount', this.formatCurrency(dailyLossAmount));
        this.updateDisplay('riskPerTradeAmount', this.formatCurrency(riskPerTradeAmount));
        this.updateDisplay('riskRewardRatio', `1:${riskRewardRatio.toFixed(2)}`);
        
        // Validate risk levels and show warnings
        this.validateRiskLevels(riskPerTrade, dailyLossLimit, riskRewardRatio);
    }
    
    validateRiskLevels(riskPerTrade, dailyLossLimit, riskRewardRatio) {
        const warnings = [];
        
        if (riskPerTrade > 5) {
            warnings.push('High risk per trade (>5%)');
        }
        
        if (dailyLossLimit > 10) {
            warnings.push('High daily loss limit (>10%)');
        }
        
        if (riskRewardRatio < 1.5) {
            warnings.push('Low risk-reward ratio (<1.5)');
        }
        
        // Display warnings
        this.displayRiskWarnings(warnings);
    }
    
    displayRiskWarnings(warnings) {
        // Remove existing warnings
        const existingWarnings = document.querySelectorAll('.risk-warning');
        existingWarnings.forEach(warning => warning.remove());
        
        if (warnings.length > 0) {
            const warningHtml = `
                <div class="alert alert-warning risk-warning mt-3">
                    <strong>Risk Warnings:</strong>
                    <ul class="mb-0 mt-2">
                        ${warnings.map(warning => `<li>${warning}</li>`).join('')}
                    </ul>
                </div>
            `;
            
            const form = document.getElementById('riskConfigForm');
            form.insertAdjacentHTML('afterend', warningHtml);
        }
    }
    
    updateDisplay(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = value;
        }
    }
    
    formatCurrency(amount) {
        return new Intl.NumberFormat('en-IN', {
            style: 'currency',
            currency: 'INR',
            minimumFractionDigits: 0,
            maximumFractionDigits: 0
        }).format(amount);
    }
    
    updateRiskSummary(config) {
        const capital = config.capital || 100000;
        const maxPositionSize = capital * 0.2; // 20% default
        const dailyLossAmount = capital * (config.daily_loss_limit_percent || 5) / 100;
        const riskPerTradeAmount = capital * (config.risk_per_trade_percent || 2) / 100;
        const riskRewardRatio = (config.take_profit_percent || 6) / (config.stop_loss_percent || 3);
        
        this.updateDisplay('maxPositionSize', this.formatCurrency(maxPositionSize));
        this.updateDisplay('dailyLossAmount', this.formatCurrency(dailyLossAmount));
        this.updateDisplay('riskPerTradeAmount', this.formatCurrency(riskPerTradeAmount));
        this.updateDisplay('riskRewardRatio', `1:${riskRewardRatio.toFixed(2)}`);
    }
}

// Initialize risk management manager
if (document.getElementById('riskConfigForm')) {
    document.addEventListener('DOMContentLoaded', () => {
        window.riskManager = new RiskManagementManager();
    });
}
"""
        
        # Save JavaScript files
        js_files = {
            "main.js": main_js,
            "dashboard.js": dashboard_js,
            "risk-management.js": risk_management_js
        }
        
        for filename, content in js_files.items():
            js_path = self.project_root / "static" / "js" / filename
            with open(js_path, 'w') as f:
                f.write(content)
                self.generated_files.append(str(js_path))
            print(f"✅ Generated: static/js/{filename}")
    
    def generate_additional_templates(self):
        """Generate additional HTML templates"""
        
        # Register template
        register_template = """{% extends "base.html" %}

{% block title %}Register - Trading Bot{% endblock %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-6 col-lg-5">
        <div class="card shadow">
            <div class="card-header text-center">
                <h4><i class="fas fa-user-plus"></i> Create Account</h4>
            </div>
            <div class="card-body">
                <form method="POST" id="registerForm">
                    <div class="mb-3">
                        <label for="username" class="form-label">Username</label>
                        <input type="text" class="form-control" id="username" name="username" required minlength="3">
                        <div class="form-text">At least 3 characters</div>
                    </div>
                    <div class="mb-3">
                        <label for="email" class="form-label">Email</label>
                        <input type="email" class="form-control" id="email" name="email" required>
                    </div>
                    <div class="mb-3">
                        <label for="password" class="form-label">Password</label>
                        <input type="password" class="form-control" id="password" name="password" required minlength="6">
                        <div class="form-text">At least 6 characters</div>
                    </div>
                    <div class="mb-3">
                        <label for="confirm_password" class="form-label">Confirm Password</label>
                        <input type="password" class="form-control" id="confirm_password" name="confirm_password" required>
                    </div>
                    <div class="mb-3 form-check">
                        <input type="checkbox" class="form-check-input" id="terms" name="terms" required>
                        <label class="form-check-label" for="terms">
                            I agree to the <a href="#" data-bs-toggle="modal" data-bs-target="#termsModal">Terms and Conditions</a>
                        </label>
                    </div>
                    <button type="submit" class="btn btn-primary w-100">Create Account</button>
                </form>
                <hr>
                <div class="text-center">
                    <a href="{{ url_for('login') }}" class="text-decoration-none">Already have an account? Login</a>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Terms Modal -->
<div class="modal fade" id="termsModal" tabindex="-1">
    <div class="modal-dialog modal-lg">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">Terms and Conditions</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
            </div>
            <div class="modal-body">
                <h6>Trading Bot Terms of Service</h6>
                <p>By using this trading bot, you acknowledge and agree to the following:</p>
                <ul>
                    <li>Trading involves risk and you may lose money</li>
                    <li>Past performance does not guarantee future results</li>
                    <li>You are responsible for your trading decisions</li>
                    <li>This software is provided for educational purposes</li>
                    <li>Always start with paper trading before live trading</li>
                </ul>
                <p><strong>Risk Disclaimer:</strong> Trading in financial markets carries significant risk. Only trade with money you can afford to lose.</p>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_scripts %}
<script>
document.getElementById('registerForm').addEventListener('submit', function(e) {
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirm_password').value;
    
    if (password !== confirmPassword) {
        e.preventDefault();
        alert('Passwords do not match');
    }
});
</script>
{% endblock %}"""

        # Portfolio template
        portfolio_template = """{% extends "base.html" %}

{% block title %}Portfolio - Trading Bot{% endblock %}

{% block content %}
<div class="row">
    <div class="col-12">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h2><i class="fas fa-briefcase"></i> Portfolio</h2>
            <div>
                <button class="btn btn-outline-primary" onclick="refreshPortfolio()">
                    <i class="fas fa-sync-alt"></i> Refresh
                </button>
            </div>
        </div>
    </div>
</div>

<!-- Portfolio Summary -->
<div class="row">
    <div class="col-md-3">
        <div class="card">
            <div class="card-body text-center">
                <h5 class="card-title">Total Value</h5>
                <h3 class="text-primary" id="totalValue">₹1,00,000</h3>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card">
            <div class="card-body text-center">
                <h5 class="card-title">Invested Amount</h5>
                <h3 class="text-info" id="investedAmount">₹95,000</h3>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card">
            <div class="card-body text-center">
                <h5 class="card-title">Total P&L</h5>
                <h3 class="text-success" id="totalPnL">₹5,000</h3>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card">
            <div class="card-body text-center">
                <h5 class="card-title">Total Return</h5>
                <h3 class="text-success" id="totalReturn">+5.26%</h3>
            </div>
        </div>
    </div>
</div>

<!-- Holdings Table -->
<div class="row mt-4">
    <div class="col-12">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-list"></i> Holdings</h5>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-hover" id="holdingsTable">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Company</th>
                                <th>Quantity</th>
                                <th>Avg Price</th>
                                <th>LTP</th>
                                <th>Invested</th>
                                <th>Current Value</th>
                                <th>P&L</th>
                                <th>P&L %</th>
                                <th>Day P&L</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="holdingsBody">
                            <tr>
                                <td colspan="11" class="text-center text-muted">No holdings</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_scripts %}
<script>
async function refreshPortfolio() {
    try {
        const response = await fetch('/api/portfolio');
        const data = await response.json();
        
        if (data.success) {
            updatePortfolioData(data.data);
        }
    } catch (error) {
        console.error('Error refreshing portfolio:', error);
    }
}

function updatePortfolioData(data) {
    // Update summary cards
    document.getElementById('totalValue').textContent = formatCurrency(data.total_value);
    document.getElementById('investedAmount').textContent = formatCurrency(data.invested_amount);
    document.getElementById('totalPnL').textContent = formatCurrency(data.total_pnl);
    document.getElementById('totalReturn').textContent = data.total_return_percent.toFixed(2) + '%';
    
    // Update P&L colors
    const pnlElement = document.getElementById('totalPnL');
    const returnElement = document.getElementById('totalReturn');
    
    if (data.total_pnl >= 0) {
        pnlElement.className = 'text-success';
        returnElement.className = 'text-success';
    } else {
        pnlElement.className = 'text-danger';
        returnElement.className = 'text-danger';
    }
    
    // Update holdings table
    updateHoldingsTable(data.holdings);
}

function updateHoldingsTable(holdings) {
    const tbody = document.getElementById('holdingsBody');
    
    if (!holdings || holdings.length === 0) {
        tbody.innerHTML = '<tr><td colspan="11" class="text-center text-muted">No holdings</td></tr>';
        return;
    }
    
    tbody.innerHTML = holdings.map(holding => `
        <tr>
            <td><strong>${holding.symbol}</strong></td>
            <td>${holding.company_name}</td>
            <td>${holding.quantity}</td>
            <td>₹${holding.avg_price.toFixed(2)}</td>
            <td>₹${holding.current_price.toFixed(2)}</td>
            <td>₹${holding.invested_amount.toFixed(0)}</td>
            <td>₹${holding.current_value.toFixed(0)}</td>
            <td class="${holding.unrealized_pnl >= 0 ? 'text-success' : 'text-danger'}">
                ₹${holding.unrealized_pnl.toFixed(0)}
            </td>
            <td class="${holding.unrealized_pnl_percent >= 0 ? 'text-success' : 'text-danger'}">
                ${holding.unrealized_pnl_percent.toFixed(2)}%
            </td>
            <td class="${holding.day_pnl >= 0 ? 'text-success' : 'text-danger'}">
                ₹${holding.day_pnl.toFixed(0)}
            </td>
            <td>
                <button class="btn btn-sm btn-danger" onclick="sellPosition('${holding.symbol}')">
                    <i class="fas fa-minus"></i> Sell
                </button>
            </td>
        </tr>
    `).join('');
}

async function sellPosition(symbol) {
    if (confirm(`Are you sure you want to sell all ${symbol} positions?`)) {
        try {
            const response = await fetch('/api/close-position', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol })
            });
            
            const result = await response.json();
            
            if (result.success) {
                app.showAlert(`Sell order placed for ${symbol}`, 'success');
                refreshPortfolio();
            } else {
                app.showAlert(`Error: ${result.message}`, 'danger');
            }
        } catch (error) {
            console.error('Error selling position:', error);
            app.showAlert('Error placing sell order', 'danger');
        }
    }
}

function formatCurrency(amount) {
    return new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(amount);
}

// Auto-refresh every 30 seconds
setInterval(refreshPortfolio, 30000);

// Load initial data
document.addEventListener('DOMContentLoaded', refreshPortfolio);
</script>
{% endblock %}"""

        # Trades template
        trades_template = """{% extends "base.html" %}

{% block title %}Trades - Trading Bot{% endblock %}

{% block content %}
<div class="row">
    <div class="col-12">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h2><i class="fas fa-exchange-alt"></i> Trade History</h2>
            <div>
                <select class="form-select" id="dateFilter" onchange="filterTrades()">
                    <option value="today">Today</option>
                    <option value="week">This Week</option>
                    <option value="month">This Month</option>
                    <option value="all">All Time</option>
                </select>
            </div>
        </div>
    </div>
</div>

<!-- Trade Summary -->
<div class="row">
    <div class="col-md-2">
        <div class="card">
            <div class="card-body text-center">
                <h6>Total Trades</h6>
                <h4 id="totalTrades">0</h4>
            </div>
        </div>
    </div>
    <div class="col-md-2">
        <div class="card">
            <div class="card-body text-center">
                <h6>Winning</h6>
                <h4 class="text-success" id="winningTrades">0</h4>
            </div>
        </div>
    </div>
    <div class="col-md-2">
        <div class="card">
            <div class="card-body text-center">
                <h6>Losing</h6>
                <h4 class="text-danger" id="losingTrades">0</h4>
            </div>
        </div>
    </div>
    <div class="col-md-2">
        <div class="card">
            <div class="card-body text-center">
                <h6>Win Rate</h6>
                <h4 id="winRate">0%</h4>
            </div>
        </div>
    </div>
    <div class="col-md-2">
        <div class="card">
            <div class="card-body text-center">
                <h6>Total P&L</h6>
                <h4 id="totalPnL">₹0</h4>
            </div>
        </div>
    </div>
    <div class="col-md-2">
        <div class="card">
            <div class="card-body text-center">
                <h6>Avg P&L</h6>
                <h4 id="avgPnL">₹0</h4>
            </div>
        </div>
    </div>
</div>

<!-- Trades Table -->
<div class="row mt-4">
    <div class="col-12">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-history"></i> Trade Details</h5>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-hover" id="tradesTable">
                        <thead>
                            <tr>
                                <th>Date/Time</th>
                                <th>Symbol</th>
                                <th>Strategy</th>
                                <th>Action</th>
                                <th>Quantity</th>
                                <th>Entry Price</th>
                                <th>Exit Price</th>
                                <th>P&L</th>
                                <th>P&L %</th>
                                <th>Status</th>
                                <th>Holding Period</th>
                            </tr>
                        </thead>
                        <tbody id="tradesBody">
                            <tr>
                                <td colspan="11" class="text-center text-muted">No trades found</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_scripts %}
<script>
async function loadTrades(filter = 'today') {
    try {
        const response = await fetch(`/api/trades?filter=${filter}`);
        const data = await response.json();
        
        if (data.success) {
            updateTradesSummary(data.data.summary);
            updateTradesTable(data.data.trades);
        }
    } catch (error) {
        console.error('Error loading trades:', error);
    }
}

function updateTradesSummary(summary) {
    document.getElementById('totalTrades').textContent = summary.total_trades;
    document.getElementById('winningTrades').textContent = summary.winning_trades;
    document.getElementById('losingTrades').textContent = summary.losing_trades;
    document.getElementById('winRate').textContent = summary.win_rate.toFixed(1) + '%';
    
    const pnlElement = document.getElementById('totalPnL');
    const avgPnlElement = document.getElementById('avgPnL');
    
    pnlElement.textContent = formatCurrency(summary.total_pnl);
    avgPnlElement.textContent = formatCurrency(summary.avg_pnl);
    
    // Set colors based on P&L
    const pnlClass = summary.total_pnl >= 0 ? 'text-success' : 'text-danger';
    const avgPnlClass = summary.avg_pnl >= 0 ? 'text-success' : 'text-danger';
    
    pnlElement.className = pnlClass;
    avgPnlElement.className = avgPnlClass;
}

function updateTradesTable(trades) {
    const tbody = document.getElementById('tradesBody');
    
    if (!trades || trades.length === 0) {
        tbody.innerHTML = '<tr><td colspan="11" class="text-center text-muted">No trades found</td></tr>';
        return;
    }
    
    tbody.innerHTML = trades.map(trade => `
        <tr>
            <td>${new Date(trade.entry_time).toLocaleString()}</td>
            <td><strong>${trade.symbol}</strong></td>
            <td>${trade.strategy}</td>
            <td><span class="badge ${trade.action === 'BUY' ? 'bg-success' : 'bg-danger'}">${trade.action}</span></td>
            <td>${trade.quantity}</td>
            <td>₹${trade.entry_price.toFixed(2)}</td>
            <td>${trade.exit_price ? '₹' + trade.exit_price.toFixed(2) : '-'}</td>
            <td class="${trade.realized_pnl >= 0 ? 'text-success' : 'text-danger'}">
                ${trade.realized_pnl ? formatCurrency(trade.realized_pnl) : '-'}
            </td>
            <td class="${trade.realized_pnl >= 0 ? 'text-success' : 'text-danger'}">
                ${trade.realized_pnl && trade.entry_price ? 
                  ((trade.realized_pnl / (trade.entry_price * trade.quantity)) * 100).toFixed(2) + '%' : '-'}
            </td>
            <td><span class="badge bg-${getStatusColor(trade.status)}">${trade.status}</span></td>
            <td>${trade.holding_period || '-'}</td>
        </tr>
    `).join('');
}

function getStatusColor(status) {
    const colors = {
        'FILLED': 'success',
        'PENDING': 'warning',
        'CANCELLED': 'danger',
        'REJECTED': 'danger',
        'PARTIAL': 'info'
    };
    return colors[status] || 'secondary';
}

function filterTrades() {
    const filter = document.getElementById('dateFilter').value;
    loadTrades(filter);
}

function formatCurrency(amount) {
    return new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(amount);
}

// Load initial data
document.addEventListener('DOMContentLoaded', () => loadTrades());
</script>
{% endblock %}"""

        # Strategies template
        strategies_template = """{% extends "base.html" %}

{% block title %}Strategies - Trading Bot{% endblock %}

{% block content %}
<div class="row">
    <div class="col-12">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h2><i class="fas fa-cogs"></i> Trading Strategies</h2>
            <div>
                <button class="btn btn-primary" data-bs-toggle="modal" data-bs-target="#addStrategyModal">
                    <i class="fas fa-plus"></i> Add Strategy
                </button>
            </div>
        </div>
    </div>
</div>

<!-- Active Strategies -->
<div class="row">
    <div class="col-12">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-play"></i> Active Strategies</h5>
            </div>
            <div class="card-body">
                <div class="row" id="strategiesContainer">
                    <div class="col-md-4">
                        <div class="card strategy-card">
                            <div class="card-body">
                                <div class="d-flex justify-content-between align-items-center mb-3">
                                    <h6 class="card-title">Momentum Breakout</h6>
                                    <div class="form-check form-switch">
                                        <input class="form-check-input" type="checkbox" id="strategy1" checked>
                                    </div>
                                </div>
                                <p class="card-text text-muted">Trades on momentum breakouts with volume confirmation</p>
                                <div class="row text-center">
                                    <div class="col-4">
                                        <small>Trades</small>
                                        <div class="fw-bold">15</div>
                                    </div>
                                    <div class="col-4">
                                        <small>Win Rate</small>
                                        <div class="fw-bold text-success">73%</div>
                                    </div>
                                    <div class="col-4">
                                        <small>P&L</small>
                                        <div class="fw-bold text-success">+₹2,450</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-4">
                        <div class="card strategy-card">
                            <div class="card-body">
                                <div class="d-flex justify-content-between align-items-center mb-3">
                                    <h6 class="card-title">Mean Reversion</h6>
                                    <div class="form-check form-switch">
                                        <input class="form-check-input" type="checkbox" id="strategy2">
                                    </div>
                                </div>
                                <p class="card-text text-muted">Trades on oversold/overbought conditions</p>
                                <div class="row text-center">
                                    <div class="col-4">
                                        <small>Trades</small>
                                        <div class="fw-bold">8</div>
                                    </div>
                                    <div class="col-4">
                                        <small>Win Rate</small>
                                        <div class="fw-bold text-warning">50%</div>
                                    </div>
                                    <div class="col-4">
                                        <small>P&L</small>
                                        <div class="fw-bold text-danger">-₹320</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Strategy Performance -->
<div class="row mt-4">
    <div class="col-12">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-chart-bar"></i> Strategy Performance</h5>
            </div>
            <div class="card-body">
                <canvas id="strategyChart" height="300"></canvas>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_scripts %}
<script>
// Initialize strategy performance chart
document.addEventListener('DOMContentLoaded', () => {
    const ctx = document.getElementById('strategyChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Momentum Breakout', 'Mean Reversion', 'Trend Following', 'Scalping'],
            datasets: [{
                label: 'P&L (₹)',
                data: [2450, -320, 1800, 650],
                backgroundColor: [
                    'rgba(40, 167, 69, 0.8)',
                    'rgba(220, 53, 69, 0.8)',
                    'rgba(40, 167, 69, 0.8)',
                    'rgba(40, 167, 69, 0.8)'
                ],
                borderColor: [
                    'rgba(40, 167, 69, 1)',
                    'rgba(220, 53, 69, 1)',
                    'rgba(40, 167, 69, 1)',
                    'rgba(40, 167, 69, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return '₹' + value;
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
});

// Handle strategy toggle
document.addEventListener('change', (e) => {
    if (e.target.type === 'checkbox' && e.target.id.startsWith('strategy')) {
        const strategyName = e.target.closest('.card-body').querySelector('.card-title').textContent;
        const isEnabled = e.target.checked;
        
        toggleStrategy(strategyName, isEnabled);
    }
});

async function toggleStrategy(strategyName, enabled) {
    try {
        const response = await fetch('/api/toggle-strategy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ strategy: strategyName, enabled })
        });
        
        const result = await response.json();
        
        if (result.success) {
            app.showAlert(`${strategyName} ${enabled ? 'enabled' : 'disabled'}`, 'success');
        } else {
            app.showAlert(`Error: ${result.message}`, 'danger');
        }
    } catch (error) {
        console.error('Error toggling strategy:', error);
        app.showAlert('Error updating strategy', 'danger');
    }
}
</script>
{% endblock %}"""

        # Settings template
        settings_template = """{% extends "base.html" %}

{% block title %}Settings - Trading Bot{% endblock %}

{% block content %}
<div class="row">
    <div class="col-12">
        <h2><i class="fas fa-cog"></i> Settings</h2>
        <p class="text-muted">Configure your trading bot preferences</p>
    </div>
</div>

<div class="row">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5>General Settings</h5>
            </div>
            <div class="card-body">
                <form id="generalSettingsForm">
                    <div class="mb-3">
                        <label class="form-label">Trading Mode</label>
                        <select class="form-select" name="trading_mode">
                            <option value="paper">Paper Trading</option>
                            <option value="live">Live Trading</option>
                        </select>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Auto Schedule</label>
                        <div class="form-check form-switch">
                            <input class="form-check-input" type="checkbox" name="auto_schedule" checked>
                            <label class="form-check-label">Enable automatic start/stop</label>
                        </div>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Session Timeout (minutes)</label>
                        <input type="number" class="form-control" name="session_timeout" value="480" min="60" max="1440">
                    </div>
                    <button type="submit" class="btn btn-primary">Save General Settings</button>
                </form>
            </div>
        </div>
    </div>
    
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5>Notification Settings</h5>
            </div>
            <div class="card-body">
                <form id="notificationSettingsForm">
                    <div class="mb-3">
                        <div class="form-check form-switch">
                            <input class="form-check-input" type="checkbox" name="email_notifications" checked>
                            <label class="form-check-label">Email Notifications</label>
                        </div>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Email Address</label>
                        <input type="email" class="form-control" name="email_address" placeholder="your@email.com">
                    </div>
                    <div class="mb-3">
                        <div class="form-check form-switch">
                            <input class="form-check-input" type="checkbox" name="webhook_notifications">
                            <label class="form-check-label">Webhook Notifications</label>
                        </div>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Webhook URL</label>
                        <input type="url" class="form-control" name="webhook_url" placeholder="https://your-webhook-url.com">
                    </div>
                    <button type="submit" class="btn btn-primary">Save Notification Settings</button>
                </form>
            </div>
        </div>
    </div>
</div>

<div class="row mt-4">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5>API Configuration</h5>
            </div>
            <div class="card-body">
                <form id="apiSettingsForm">
                    <div class="mb-3">
                        <label class="form-label">Goodwill API Key</label>
                        <input type="password" class="form-control" name="goodwill_api_key" placeholder="Enter API Key">
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Goodwill Secret Key</label>
                        <input type="password" class="form-control" name="goodwill_secret_key" placeholder="Enter Secret Key">
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Goodwill User ID</label>
                        <input type="text" class="form-control" name="goodwill_user_id" placeholder="Enter User ID">
                    </div>
                    <button type="submit" class="btn btn-primary">Save API Configuration</button>
                </form>
            </div>
        </div>
    </div>
    
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5>System Status</h5>
            </div>
            <div class="card-body">
                <div class="mb-3">
                    <label class="form-label">Bot Status</label>
                    <div class="d-flex align-items-center">
                        <span class="badge bg-success me-2" id="systemBotStatus">Running</span>
                        <button class="btn btn-sm btn-outline-secondary" onclick="checkSystemStatus()">
                            <i class="fas fa-sync-alt"></i> Refresh
                        </button>
                    </div>
                </div>
                <div class="mb-3">
                    <label class="form-label">Database Status</label>
                    <div><span class="badge bg-success" id="systemDbStatus">Connected</span></div>
                </div>
                <div class="mb-3">
                    <label class="form-label">API Status</label>
                    <div><span class="badge bg-warning" id="systemApiStatus">Not Configured</span></div>
                </div>
                <div class="mb-3">
                    <label class="form-label">Last Update</label>
                    <div class="text-muted" id="systemLastUpdate">{{ datetime.now().strftime('%Y-%m-%d %H:%M:%S') }}</div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_scripts %}
<script>
async function checkSystemStatus() {
    try {
        const response = await fetch('/api/system-status');
        const data = await response.json();
        
        if (data.success) {
            updateSystemStatus(data.data);
        }
    } catch (error) {
        console.error('Error checking system status:', error);
    }
}

function updateSystemStatus(status) {
    document.getElementById('systemBotStatus').textContent = status.bot_running ? 'Running' : 'Stopped';
    document.getElementById('systemBotStatus').className = `badge me-2 ${status.bot_running ? 'bg-success' : 'bg-danger'}`;
    
    document.getElementById('systemDbStatus').textContent = status.database_connected ? 'Connected' : 'Disconnected';
    document.getElementById('systemDbStatus').className = `badge ${status.database_connected ? 'bg-success' : 'bg-danger'}`;
    
    document.getElementById('systemApiStatus').textContent = status.api_configured ? 'Configured' : 'Not Configured';
    document.getElementById('systemApiStatus').className = `badge ${status.api_configured ? 'bg-success' : 'bg-warning'}`;
    
    document.getElementById('systemLastUpdate').textContent = new Date().toLocaleString();
}

// Handle form submissions
document.getElementById('generalSettingsForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    await saveSettings('general', new FormData(e.target));
});

document.getElementById('notificationSettingsForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    await saveSettings('notifications', new FormData(e.target));
});

document.getElementById('apiSettingsForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    await saveSettings('api', new FormData(e.target));
});

async function saveSettings(category, formData) {
    try {
        const data = {};
        for (let [key, value] of formData.entries()) {
            if (value === 'on') {
                data[key] = true;
            } else {
                data[key] = value;
            }
        }
        
        const response = await fetch(`/api/save-settings/${category}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.success) {
            app.showAlert(`${category} settings saved successfully!`, 'success');
        } else {
            app.showAlert(`Error saving ${category} settings: ${result.message}`, 'danger');
        }
    } catch (error) {
        console.error(`Error saving ${category} settings:`, error);
        app.showAlert(`Error saving ${category} settings`, 'danger');
    }
}

// Load initial system status
document.addEventListener('DOMContentLoaded', checkSystemStatus);
</script>
{% endblock %}"""

        # Save additional templates
        additional_templates = {
            "register.html": register_template,
            "portfolio.html": portfolio_template,
            "trades.html": trades_template,
            "strategies.html": strategies_template,
            "settings.html": settings_template
        }
        
        for filename, content in additional_templates.items():
            template_path = self.project_root / "templates" / filename
            with open(template_path, 'w') as f:
                f.write(content)
            self.generated_files.append(str(template_path))
            print(f"✅ Generated template: {filename}")
    
    def generate_requirements_file(self):
        """Generate requirements.txt file"""
        
        requirements = """# Trading Bot Requirements
# Core Framework
Flask==2.3.2
Flask-Session==0.5.0
Flask-CORS==4.0.0

# Database
asyncpg==0.28.0
psycopg2-binary==2.9.6
SQLAlchemy==2.0.19

# Data Processing
pandas==2.0.3
numpy==1.24.3
scipy==1.11.1

# Financial Data
yfinance==0.2.18
ta-lib==0.4.27
ccxt==4.0.77

# API and Web
requests==2.31.0
aiohttp==3.8.5
websockets==11.0.3

# Configuration and Environment
python-dotenv==1.0.0
PyYAML==6.0.1

# Scheduling and Tasks
APScheduler==3.10.1

# Utilities
pytz==2023.3
python-dateutil==2.8.2
cryptography==41.0.3

# Logging and Monitoring
structlog==23.1.0

# Development and Testing
pytest==7.4.0
pytest-asyncio==0.21.1
black==23.7.0
flake8==6.0.0

# Production
gunicorn==21.2.0
eventlet==0.33.3
"""
        
        req_path = self.project_root / "requirements.txt"
        with open(req_path, 'w') as f:
            f.write(requirements)
        
        self.generated_files.append(str(req_path))
        print(f"✅ Generated: requirements.txt")
    
    def generate_dockerfile(self):
        """Generate Dockerfile for containerization"""
        
        dockerfile_content = """FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    postgresql-client \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data backups

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
  CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--worker-class", "eventlet", "app:app"]
"""
        
        dockerfile_path = self.project_root / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        self.generated_files.append(str(dockerfile_path))
        print(f"✅ Generated: Dockerfile")
    
    def generate_docker_compose(self):
        """Generate docker-compose.yml for easy deployment"""
        
        docker_compose_content = """version: '3.8'

services:
  trading-bot:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://trading_user:trading_pass@postgres:5432/trading_bot
      - FLASK_ENV=production
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./backups:/app/backups
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=trading_bot
      - POSTGRES_USER=trading_user
      - POSTGRES_PASSWORD=trading_pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - trading-bot
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
"""
        
        compose_path = self.project_root / "docker-compose.yml"
        with open(compose_path, 'w') as f:
            f.write(docker_compose_content)
        
        self.generated_files.append(str(compose_path))
        print(f"✅ Generated: docker-compose.yml")
    
    def generate_init_sql(self):
        """Generate database initialization SQL"""
        
        init_sql = """-- Database initialization script
-- This script sets up the basic database structure

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create default user if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'trading_user') THEN
        CREATE USER trading_user WITH PASSWORD 'trading_pass';
    END IF;
END
$$;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE trading_bot TO trading_user;
GRANT ALL ON SCHEMA public TO trading_user;
GRANT ALL ON ALL TABLES IN SCHEMA public TO trading_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO trading_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO trading_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO trading_user;

-- Create basic indexes for performance
-- These will be created by the application, but having them here ensures they exist
"""
        
        init_sql_path = self.project_root / "init.sql"
        with open(init_sql_path, 'w') as f:
            f.write(init_sql)
        
        self.generated_files.append(str(init_sql_path))
        print(f"✅ Generated: init.sql")
    
    def generate_nginx_config(self):
        """Generate Nginx configuration"""
        
        nginx_config = """events {
    worker_connections 1024;
}

http {
    upstream trading_bot {
        server trading-bot:5000;
    }

    server {
        listen 80;
        server_name localhost;

        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name localhost;

        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header Referrer-Policy "no-referrer-when-downgrade" always;
        add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;

        # Gzip compression
        gzip on;
        gzip_vary on;
        gzip_min_length 1024;
        gzip_proxied expired no-cache no-store private must-revalidate auth;
        gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml+rss;

        # Rate limiting
        limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
        limit_req_zone $binary_remote_addr zone=login:10m rate=1r/s;

        location / {
            proxy_pass http://trading_bot;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # WebSocket support
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://trading_bot;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /login {
            limit_req zone=login burst=5 nodelay;
            proxy_pass http://trading_bot;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Static files caching
        location /static/ {
            proxy_pass http://trading_bot;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
"""
        
        nginx_path = self.project_root / "nginx.conf"
        with open(nginx_path, 'w') as f:
            f.write(nginx_config)
        
        self.generated_files.append(str(nginx_path))
        print(f"✅ Generated: nginx.conf")
    
    def generate_startup_scripts(self):
        """Generate startup and utility scripts"""
        
        # Main startup script
        startup_script = """#!/bin/bash
# Trading Bot Startup Script

set -e

echo "🚀 Starting Trading Bot Production System..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please create one from .env.example"
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '#' | awk '/=/ {print $1}')

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Run database migrations if needed
echo "🗄️ Setting up database..."
python -c "
import asyncio
from database_setup import DatabaseManager
from config_manager import ConfigManager

async def setup_db():
    config_manager = ConfigManager()
    db_manager = DatabaseManager(config_manager)
    await db_manager.initialize()
    print('✅ Database setup complete')

asyncio.run(setup_db())
"

# Start the application
echo "🎯 Starting Flask application..."
if [ "$FLASK_ENV" = "production" ]; then
    gunicorn --bind 0.0.0.0:5000 --workers 4 --worker-class eventlet app:app
else
    python app.py
fi
"""
        
        # Development script
        dev_script = """#!/bin/bash
# Development startup script

echo "🛠️ Starting Trading Bot in Development Mode..."

# Set development environment
export FLASK_ENV=development
export FLASK_DEBUG=True

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start development server
python app.py
"""
        
        # Setup script
        setup_script = """#!/bin/bash
# Setup script for fresh installation

echo "🔧 Setting up Trading Bot..."

# Update system packages
sudo apt-get update

# Install system dependencies
sudo apt-get install -y python3 python3-pip python3-venv postgresql postgresql-contrib

# Create database user and database
sudo -u postgres createuser --interactive trading_user
sudo -u postgres createdb trading_bot
sudo -u postgres psql -c "ALTER USER trading_user PASSWORD 'change_this_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE trading_bot TO trading_user;"

# Create project directories
mkdir -p logs data backups

# Set permissions
chmod +x start.sh
chmod +x dev.sh
chmod +x setup.sh

echo "✅ Setup complete! Please:"
echo "1. Copy .env.example to .env and configure your settings"
echo "2. Run ./start.sh to start the application"
"""
        
        # Save scripts
        scripts = {
            "start.sh": startup_script,
            "dev.sh": dev_script,
            "setup.sh": setup_script
        }
        
        for filename, content in scripts.items():
            script_path = self.project_root / filename
            with open(script_path, 'w') as f:
                f.write(content)
            
            # Make executable
            script_path.chmod(0o755)
            
            self.generated_files.append(str(script_path))
            print(f"✅ Generated executable script: {filename}")
    
    def generate_readme(self):
        """Generate comprehensive README.md"""
        
        readme_content = f"""# Advanced Trading Bot

A comprehensive algorithmic trading bot with user risk management, real-time monitoring, and multi-strategy support.

## Features

🚀 **Multi-Strategy Trading**
- Momentum breakout strategies
- Mean reversion algorithms  
- Custom strategy support
- Real-time signal generation

🛡️ **Advanced Risk Management**
- User-configurable risk parameters
- Real-time risk monitoring
- Automatic position sizing
- Dynamic stop-loss and take-profit

📊 **Professional Dashboard**
- Real-time portfolio monitoring
- Performance analytics
- Trade history and reporting
- Risk alerts and notifications

🔧 **Production Ready**
- Containerized deployment
- Database-backed persistence
- RESTful API
- Comprehensive logging

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 13+
- Redis (optional, for caching)

### Installation

1. **Clone and setup:**
```bash
git clone <repository-url>
cd trading-bot
chmod +x setup.sh
./setup.sh
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your settings
```

3. **Start the application:**
```bash
./start.sh
```

### Development Mode

```bash
./dev.sh
```

### Docker Deployment

```bash
docker-compose up -d
```

## Configuration

### Environment Variables

Key environment variables in `.env`:

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/trading_bot

# Flask
FLASK_SECRET_KEY=your-secret-key
FLASK_PORT=5000

# Trading APIs
GOODWILL_API_KEY=your-api-key
GOODWILL_SECRET_KEY=your-secret-key

# Trading Settings
TRADING_MODE=paper  # or 'live'
AUTO_SCHEDULE=true
```

### User Risk Parameters

Configure via the web interface:

- **Capital**: Total trading capital
- **Risk per Trade**: Maximum % of capital to risk per trade
- **Daily Loss Limit**: Maximum daily loss percentage
- **Position Sizing**: Maximum position size as % of capital
- **Stop Loss/Take Profit**: Default percentages
- **Trading Hours**: Custom trading session times

## API Documentation

### Authentication

All API endpoints require authentication via session cookies.

### Key Endpoints

- `GET /api/dashboard-data` - Dashboard metrics
- `GET /api/portfolio` - Portfolio holdings
- `GET /api/trades` - Trade history
- `POST /api/update-risk-config` - Update risk parameters
- `POST /api/bot-control` - Start/stop bot

## Architecture

```
├── app.py                 # Flask application entry point
├── bot_core.py           # Main trading bot engine
├── config_manager.py     # Configuration management
├── database_setup.py     # Database models and operations
├── api_routes.py         # REST API endpoints
├── strategies/           # Trading strategy modules
├── indicators/          # Technical indicators
├── templates/           # HTML templates
├── static/             # CSS, JS, images
└── utils/              # Utility functions
```

## Trading Strategies

### Built-in Strategies

1. **Momentum Breakout**
   - Identifies breakout patterns
   - Volume confirmation
   - Configurable parameters

2. **Mean Reversion**
   - Oversold/overbought detection
   - Statistical analysis
   - Risk-adjusted entries

### Adding Custom Strategies

Create a new strategy class:

```python
from strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def analyze(self, data):
        # Your strategy logic
        return signals
```

## Risk Management

### User-Configurable Parameters

- **Capital Management**: Set total trading capital
- **Position Sizing**: Automatic calculation based on risk tolerance
- **Stop Loss**: Configurable percentage or ATR-based
- **Take Profit**: Risk-reward ratio based targets
- **Daily Limits**: Maximum daily loss protection
- **Concurrent Trades**: Limit number of simultaneous positions

### Real-time Monitoring

- Portfolio exposure tracking
- Risk limit enforcement
- Automatic position closure
- Alert notifications

## Monitoring and Alerts

### Built-in Monitoring

- Real-time P&L tracking
- Risk metric calculations
- System health monitoring
- Performance analytics

### Notification Channels

- Email alerts
- Webhook notifications
- Dashboard alerts

## Security

### Best Practices

- Environment variable configuration
- API key encryption
- Session management
- Input validation
- SQL injection prevention

### Production Security

- HTTPS encryption
- Rate limiting
- Authentication required
- Secure headers

## Deployment

### Local Development

```bash
python app.py
```

### Production with Gunicorn

```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
```

### Docker Deployment

```bash
docker-compose up -d
```

### Cloud Deployment

Supports deployment on:
- AWS EC2/ECS
- Google Cloud Run
- Azure Container Instances
- Digital Ocean Droplets

## Testing

Run the test suite:

```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Disclaimer

⚠️ **Important**: This software is for educational purposes only. Trading involves substantial risk of loss. Only trade with money you can afford to lose. Past performance does not guarantee future results.

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the examples

---

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        readme_path = self.project_root / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        self.generated_files.append(str(readme_path))
        print(f"✅ Generated: README.md")
    
    def generate_all_files(self):
        """Generate all production files"""
        print("🏗️ Generating complete production file structure...")
        
        # Create directory structure
        self.create_directory_structure()
        
        # Generate configuration files
        self.generate_config_files()
        
        # Generate HTML templates
        self.generate_html_templates()
        self.generate_additional_templates()
        
        # Generate CSS and JS files
        self.generate_css_files()
        self.generate_js_files()
        
        # Generate deployment files
        self.generate_requirements_file()
        self.generate_dockerfile()
        self.generate_docker_compose()
        self.generate_init_sql()
        self.generate_nginx_config()
        
        # Generate scripts
        self.generate_startup_scripts()
        
        # Generate documentation
        self.generate_readme()
        
        print(f"\n🎉 Production file generation complete!")
        print(f"📁 Generated {len(self.generated_files)} files")
        print(f"📍 Project root: {self.project_root}")
        
        print("\n📋 Next steps:")
        print("1. Copy .env.example to .env and configure your settings")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Setup database: ./setup.sh")
        print("4. Start application: ./start.sh")
        
        return self.generated_files

def main():
    """Main function to generate all files"""
    import sys
    
    project_root = sys.argv[1] if len(sys.argv) > 1 else "."
    
    generator = ProductionFilesGenerator(project_root)
    generated_files = generator.generate_all_files()
    
    print(f"\n✅ All files generated successfully in: {project_root}")
    
    return generated_files

if __name__ == "__main__":
    main()
