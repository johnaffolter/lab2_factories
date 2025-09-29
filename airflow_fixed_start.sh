#!/bin/bash

echo "🚀 Starting Airflow with Browser-Friendly Configuration"
echo "======================================================="
echo ""

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker stop airflow-standalone 2>/dev/null
docker rm airflow-standalone 2>/dev/null

# Start Airflow with optimized settings for browser access
echo "📦 Starting Airflow with optimized configuration..."
docker run -d \
    --name airflow-standalone \
    -p 8080:8080 \
    -e AIRFLOW__WEBSERVER__WEB_SERVER_WORKER_TIMEOUT=120 \
    -e AIRFLOW__WEBSERVER__WORKER_REFRESH_INTERVAL=30 \
    -e AIRFLOW__WEBSERVER__WORKER_CLASS=sync \
    -e AIRFLOW__WEBSERVER__WORKERS=2 \
    -e AIRFLOW__WEBSERVER__ACCESS_LOGFILE=- \
    -e AIRFLOW__WEBSERVER__ERROR_LOGFILE=- \
    -e AIRFLOW__WEBSERVER__EXPOSE_CONFIG=true \
    -e AIRFLOW__WEBSERVER__NAVBAR_COLOR=#667eea \
    -e AIRFLOW__WEBSERVER__INSTANCE_NAME="MLOps Lab2 Airflow" \
    -e AIRFLOW__CORE__EXECUTOR=LocalExecutor \
    -e AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@host.docker.internal/airflow \
    -e AIRFLOW__CORE__LOAD_EXAMPLES=false \
    -e _AIRFLOW_DB_MIGRATE=true \
    -e _AIRFLOW_WWW_USER_CREATE=true \
    -e _AIRFLOW_WWW_USER_USERNAME=admin \
    -e _AIRFLOW_WWW_USER_PASSWORD=admin \
    -e AIRFLOW__API__AUTH_BACKENDS=airflow.api.auth.backend.basic_auth \
    -e AIRFLOW__WEBSERVER__ENABLE_PROXY_FIX=true \
    -e AIRFLOW__WEBSERVER__PROXY_FIX_X_FOR=1 \
    -e AIRFLOW__WEBSERVER__PROXY_FIX_X_PROTO=1 \
    -e AIRFLOW__WEBSERVER__PROXY_FIX_X_HOST=1 \
    -e AIRFLOW__WEBSERVER__PROXY_FIX_X_PORT=1 \
    -e AIRFLOW__WEBSERVER__PROXY_FIX_X_PREFIX=1 \
    -v ${PWD}/dags:/opt/airflow/dags \
    -v ${PWD}/logs:/opt/airflow/logs \
    apache/airflow:2.7.0 standalone

echo "✅ Airflow container started"
echo ""

# Wait for initialization
echo "⏳ Waiting for Airflow to initialize (30 seconds)..."
sleep 30

# Check if Airflow is healthy
echo "🔍 Checking Airflow health..."
if curl -s -f http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Airflow is healthy and running!"
else
    echo "⚠️ Airflow health check failed, but it may still be initializing..."
fi

echo ""
echo "======================================================="
echo "📋 ACCESS OPTIONS:"
echo "======================================================="
echo ""
echo "1️⃣ LIGHTWEIGHT WEB INTERFACE (Recommended):"
echo "   python3 airflow_web_interface.py"
echo "   Then access: http://localhost:5555"
echo "   ✅ No browser header issues!"
echo ""
echo "2️⃣ COMMAND LINE INTERFACE:"
echo "   python3 airflow_cli_interface.py"
echo "   ✅ Interactive menu system"
echo ""
echo "3️⃣ DIRECT BROWSER ACCESS:"
echo "   URL: http://localhost:8080"
echo "   Username: admin"
echo "   Password: admin"
echo "   ⚠️ If you get header errors:"
echo "      • Clear browser cache/cookies"
echo "      • Use incognito/private mode"
echo "      • Try a different browser"
echo ""
echo "4️⃣ DIRECT DOCKER COMMANDS:"
echo "   List DAGs:     docker exec airflow-standalone airflow dags list"
echo "   Trigger DAG:   docker exec airflow-standalone airflow dags trigger <dag_id>"
echo "   View logs:     docker logs -f airflow-standalone"
echo ""
echo "======================================================="
echo ""

# Show running containers
echo "🐳 Running containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(NAMES|airflow|postgres)"

echo ""
echo "✅ Setup complete! Use one of the access methods above."