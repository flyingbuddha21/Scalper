#!/bin/bash
# PostgreSQL Setup Script for Trading Bot

echo "🚀 Setting up PostgreSQL for Trading Bot..."

# Update system
sudo apt update

# Install PostgreSQL
echo "📦 Installing PostgreSQL..."
sudo apt install -y postgresql postgresql-contrib

# Start and enable PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
echo "🗃️ Creating database and user..."
sudo -u postgres psql << EOF
-- Create database
CREATE DATABASE trading_bot;

-- Create user with password
CREATE USER trading_bot WITH PASSWORD 'TradingBot2024!';

-- Grant all privileges on database
GRANT ALL PRIVILEGES ON DATABASE trading_bot TO trading_bot;

-- Grant schema privileges
\c trading_bot;
GRANT ALL ON SCHEMA public TO trading_bot;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_bot;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_bot;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO trading_bot;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO trading_bot;

-- Show created database and user
\l
\du

\q
EOF

# Configure PostgreSQL for trading bot
echo "⚙️ Configuring PostgreSQL..."

# Update postgresql.conf for performance
PG_VERSION=$(sudo -u postgres psql -t -c "SELECT version();" | grep -oP '\d+\.\d+' | head -1)
PG_CONFIG="/etc/postgresql/$PG_VERSION/main/postgresql.conf"

sudo cp $PG_CONFIG $PG_CONFIG.backup

# Performance tuning for trading bot
sudo tee -a $PG_CONFIG << EOF

# Trading Bot Performance Settings
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200

# Connection settings
max_connections = 100
listen_addresses = 'localhost'

# Logging
log_statement = 'all'
log_min_duration_statement = 1000
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
EOF

# Update pg_hba.conf for authentication
PG_HBA="/etc/postgresql/$PG_VERSION/main/pg_hba.conf"
sudo cp $PG_HBA $PG_HBA.backup

# Add trading bot user authentication
sudo tee -a $PG_HBA << EOF

# Trading Bot Authentication
local   trading_bot     trading_bot                     md5
host    trading_bot     trading_bot     127.0.0.1/32    md5
host    trading_bot     trading_bot     ::1/128         md5
EOF

# Restart PostgreSQL to apply changes
echo "🔄 Restarting PostgreSQL..."
sudo systemctl restart postgresql

# Test connection
echo "🧪 Testing database connection..."
PGPASSWORD='TradingBot2024!' psql -h localhost -U trading_bot -d trading_bot -c "SELECT 'Connection successful!' as status;"

if [ $? -eq 0 ]; then
    echo "✅ PostgreSQL setup completed successfully!"
    echo ""
    echo "📝 Database Configuration:"
    echo "   Host: localhost"
    echo "   Port: 5432"
    echo "   Database: trading_bot"
    echo "   Username: trading_bot"
    echo "   Password: TradingBot2024!"
    echo ""
    echo "🔧 Update your bot configuration:"
    echo "   'database_type': 'postgresql'"
    echo "   'postgresql': {"
    echo "     'host': 'localhost',"
    echo "     'port': 5432,"
    echo "     'user': 'trading_bot',"
    echo "     'password': 'TradingBot2024!',"
    echo "     'database': 'trading_bot'"
    echo "   }"
else
    echo "❌ Database connection test failed!"
    echo "Please check the setup and try again."
fi

echo ""
echo "🔍 Useful PostgreSQL commands:"
echo "   Connect: psql -h localhost -U trading_bot -d trading_bot"
echo "   Check status: sudo systemctl status postgresql"
echo "   View logs: sudo tail -f /var/log/postgresql/postgresql-$PG_VERSION-main.log"
echo "   Restart: sudo systemctl restart postgresql"
