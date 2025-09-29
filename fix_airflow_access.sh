#!/bin/bash

echo "🔧 Fixing Airflow Access Issues"
echo "================================"
echo ""

# Option 1: Restart Airflow with increased header limits
echo "1️⃣ Restarting Airflow with increased limits..."
docker stop airflow-standalone
docker rm airflow-standalone

docker run -d \
    --name airflow-standalone \
    -p 8080:8080 \
    -e AIRFLOW__WEBSERVER__WEB_SERVER_WORKER_TIMEOUT=120 \
    -e AIRFLOW__WEBSERVER__WORKER_REFRESH_INTERVAL=30 \
    -e AIRFLOW__WEBSERVER__WORKER_CLASS=sync \
    -e AIRFLOW__CORE__EXECUTOR=LocalExecutor \
    -e AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@host.docker.internal/airflow \
    -e AIRFLOW__CORE__LOAD_EXAMPLES=true \
    -e _AIRFLOW_DB_MIGRATE=true \
    -e _AIRFLOW_WWW_USER_CREATE=true \
    -e _AIRFLOW_WWW_USER_USERNAME=admin \
    -e _AIRFLOW_WWW_USER_PASSWORD=admin \
    -v ${PWD}/dags:/opt/airflow/dags \
    -v ${PWD}/logs:/opt/airflow/logs \
    apache/airflow:2.7.0 standalone

echo "✅ Airflow restarted with optimized settings"
echo ""

# Option 2: Clear browser cache
echo "2️⃣ Browser Troubleshooting:"
echo "   • Clear your browser cache and cookies"
echo "   • Try accessing in Incognito/Private mode"
echo "   • Try a different browser"
echo "   • Direct link: http://localhost:8080/login/"
echo ""

# Option 3: Use curl to test
echo "3️⃣ Testing with curl..."
curl -s -X GET http://localhost:8080/login/ \
    -H "Accept: text/html" \
    -H "User-Agent: Mozilla/5.0" \
    --max-time 5 | grep -q "Airflow" && echo "✅ Airflow is accessible via curl" || echo "❌ Connection issue"

echo ""

# Option 4: Alternative access method
echo "4️⃣ Alternative Access via Port Forward:"
echo "   If browser access fails, use command line:"
echo ""
echo "   List DAGs:"
echo "   docker exec airflow-standalone airflow dags list"
echo ""
echo "   Trigger DAG:"
echo "   docker exec airflow-standalone airflow dags trigger mlops_data_pipeline"
echo ""

# Wait for Airflow to be ready
echo "⏳ Waiting for Airflow to initialize..."
sleep 30

# Check final status
echo "📊 Final Status:"
docker ps | grep airflow && echo "✅ Airflow container running" || echo "❌ Container not running"

echo ""
echo "================================"
echo "Try accessing: http://localhost:8080"
echo "Username: admin"
echo "Password: admin"