#!/bin/bash

# Quick Airflow Startup Script (Simplified Version)
# Uses docker run for immediate startup

echo "🚀 Quick Start: Apache Airflow"
echo "=============================="

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if PostgreSQL is running for Airflow backend
if ! docker ps | grep -q mlops_postgres; then
    echo "📦 Starting PostgreSQL for Airflow backend..."
    docker run -d \
        --name mlops_postgres \
        -e POSTGRES_USER=airflow \
        -e POSTGRES_PASSWORD=airflow \
        -e POSTGRES_DB=airflow \
        -p 5432:5432 \
        postgres:13

    echo "⏳ Waiting for PostgreSQL to be ready..."
    sleep 10
fi

# Start Airflow standalone in a single container
echo "🐳 Starting Airflow in standalone mode..."

# Stop existing Airflow container if running
docker stop airflow-standalone 2>/dev/null
docker rm airflow-standalone 2>/dev/null

# Run Airflow standalone
docker run -d \
    --name airflow-standalone \
    -p 8080:8080 \
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

echo "⏳ Waiting for Airflow to initialize (this takes about 60 seconds)..."

# Wait for Airflow to be ready
for i in {1..12}; do
    if docker logs airflow-standalone 2>&1 | grep -q "Airflow is ready"; then
        echo "✅ Airflow is ready!"
        break
    fi
    echo "   Still initializing... ($i/12)"
    sleep 10
done

# Create sample DAGs directory
mkdir -p dags

# Create a simple example DAG
cat > dags/simple_mlops_dag.py << 'EOF'
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'mlops',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def process_data():
    print("Processing MLOps data...")
    return "Data processed successfully"

def train_model():
    print("Training ML model...")
    return "Model trained successfully"

def evaluate_model():
    print("Evaluating model performance...")
    return "Model evaluation complete"

with DAG(
    'simple_mlops_pipeline',
    default_args=default_args,
    description='Simple MLOps Pipeline',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    # Task 1: Process data
    process = PythonOperator(
        task_id='process_data',
        python_callable=process_data,
    )

    # Task 2: Train model
    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    # Task 3: Evaluate model
    evaluate = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
    )

    # Task 4: Deploy notification
    notify = BashOperator(
        task_id='deployment_notification',
        bash_command='echo "Model ready for deployment!"',
    )

    # Set task dependencies
    process >> train >> evaluate >> notify
EOF

echo "✅ Sample DAG created in ./dags/"

# Display access information
echo ""
echo "=============================="
echo "✅ AIRFLOW IS RUNNING!"
echo "=============================="
echo ""
echo "📊 Access Airflow UI:"
echo "   URL: http://localhost:8080"
echo "   Username: admin"
echo "   Password: admin"
echo ""
echo "🔧 Useful Commands:"
echo "   View logs: docker logs -f airflow-standalone"
echo "   Stop: docker stop airflow-standalone"
echo "   Restart: docker restart airflow-standalone"
echo "   Shell: docker exec -it airflow-standalone bash"
echo ""
echo "📁 DAGs Directory: ./dags"
echo "   Place your DAG files here and they'll appear in the UI"
echo ""

# Show initial logs
echo "📝 Showing Airflow logs..."
docker logs --tail=20 airflow-standalone

echo ""
echo "🎉 Airflow is initializing. The UI will be available at http://localhost:8080 in about 1 minute."
echo ""
echo "Press Ctrl+C to exit (Airflow will continue running in background)"