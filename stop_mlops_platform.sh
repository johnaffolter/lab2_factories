#!/bin/bash

# MLOps Platform Shutdown Script
# Stops all components of the MLOps platform

echo "🛑 STOPPING MLOPS PLATFORM"
echo "=================================="
echo ""

# Stop Python processes
echo "Stopping Python services..."
pkill -f "master_control_panel.py"
pkill -f "realtime_monitoring_websocket_server.py"
pkill -f "service_health_auto_recovery.py"
echo "✅ Python services stopped"
echo ""

# Stop Docker containers
echo "Stopping Docker containers..."
docker stop mlops_redis 2>/dev/null
docker stop mlops_neo4j 2>/dev/null
docker stop mlops_postgres 2>/dev/null
docker stop mlops_airflow 2>/dev/null
docker stop mlops_monitoring 2>/dev/null
echo "✅ Docker containers stopped"
echo ""

# Remove Docker containers (optional)
read -p "Remove Docker containers? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker rm mlops_redis 2>/dev/null
    docker rm mlops_neo4j 2>/dev/null
    docker rm mlops_postgres 2>/dev/null
    docker rm mlops_airflow 2>/dev/null
    docker rm mlops_monitoring 2>/dev/null
    echo "✅ Docker containers removed"
fi
echo ""

# Stop Airbyte if running
if [ -d ~/airbyte ]; then
    echo "Stopping Airbyte..."
    cd ~/airbyte && docker-compose down
    cd - > /dev/null
    echo "✅ Airbyte stopped"
fi
echo ""

echo "=================================="
echo "✅ MLOPS PLATFORM STOPPED"
echo "==================================