#!/bin/bash

# MLOps Platform Startup Script
# Starts all components of the comprehensive MLOps platform

echo "🚀 STARTING MLOPS PLATFORM"
echo "=================================="
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
port_in_use() {
    lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists docker; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker."
    exit 1
fi

if ! command_exists python3; then
    echo "❌ Python 3 is not installed."
    exit 1
fi

echo "✅ Prerequisites checked"
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p ~/mlops_backups
mkdir -p /tmp/prometheus
mkdir -p /tmp/neo4j
mkdir -p /tmp/postgres
echo "✅ Directories created"
echo ""

# Start core services with Docker
echo "🐳 Starting Docker services..."

# Start Redis
if ! port_in_use 6379; then
    echo "  Starting Redis..."
    docker run -d --name mlops_redis -p 6379:6379 redis:alpine
else
    echo "  Redis already running on port 6379"
fi

# Start Neo4j
if ! port_in_use 7474; then
    echo "  Starting Neo4j..."
    docker run -d --name mlops_neo4j \
        -p 7474:7474 -p 7687:7687 \
        -e NEO4J_AUTH=neo4j/password \
        -v /tmp/neo4j:/data \
        neo4j:latest
else
    echo "  Neo4j already running on port 7474"
fi

# Start PostgreSQL
if ! port_in_use 5432; then
    echo "  Starting PostgreSQL..."
    docker run -d --name mlops_postgres \
        -p 5432:5432 \
        -e POSTGRES_USER=admin \
        -e POSTGRES_PASSWORD=password \
        -e POSTGRES_DB=mlops \
        -v /tmp/postgres:/var/lib/postgresql/data \
        postgres:14
else
    echo "  PostgreSQL already running on port 5432"
fi

echo "✅ Docker services started"
echo ""

# Start monitoring services
echo "📊 Starting monitoring services..."

# Start WebSocket monitoring server
if ! port_in_use 8765; then
    echo "  Starting WebSocket monitoring server..."
    python3 realtime_monitoring_websocket_server.py > /tmp/websocket_monitor.log 2>&1 &
    echo "  WebSocket server started on port 8765"
else
    echo "  WebSocket server already running on port 8765"
fi

echo "✅ Monitoring services started"
echo ""

# Start Airbyte (optional - takes time)
read -p "Do you want to start Airbyte? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔄 Starting Airbyte..."
    if [ -f airbyte_setup.sh ]; then
        chmod +x airbyte_setup.sh
        ./airbyte_setup.sh
    else
        echo "  Airbyte setup script not found"
    fi
else
    echo "  Skipping Airbyte startup"
fi
echo ""

# Start the master control panel
echo "🎛️ Starting Master Control Panel..."
python3 master_control_panel.py &
CONTROL_PANEL_PID=$!
echo "✅ Control Panel started (PID: $CONTROL_PANEL_PID)"
echo ""

# Display access information
echo "=================================="
echo "🎉 MLOPS PLATFORM IS READY!"
echo "=================================="
echo ""
echo "📍 Access Points:"
echo "  • Master Control Panel: http://localhost:5000"
echo "  • Monitoring Dashboard: file://$(pwd)/advanced_monitoring_dashboard.html"
echo "  • WebSocket Monitor: ws://localhost:8765"
echo "  • Neo4j Browser: http://localhost:7474"
echo "  • PostgreSQL: localhost:5432"
echo "  • Redis: localhost:6379"

if [ -f ~/airbyte/docker-compose.yaml ]; then
    echo "  • Airbyte UI: http://localhost:8000"
fi

echo ""
echo "🔧 Quick Commands:"
echo "  • View logs: tail -f /tmp/*.log"
echo "  • Stop all: ./stop_mlops_platform.sh"
echo "  • Run cleanup: python3 cleanup_maintenance_utility.py"
echo "  • Check health: python3 service_health_auto_recovery.py"
echo ""
echo "Press Ctrl+C to stop the control panel"
echo ""

# Keep the script running
wait $CONTROL_PANEL_PID