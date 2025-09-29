#!/bin/bash

# Airflow Setup and Startup Script
# Sets up Apache Airflow with Docker and configures for MLOps platform

echo "🚀 Setting up Apache Airflow for MLOps Platform"
echo "=============================================="

# Configuration
AIRFLOW_HOME="${HOME}/airflow"
AIRFLOW_VERSION="2.7.0"
PYTHON_VERSION="3.8"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Create Airflow directory
echo "📁 Creating Airflow directories..."
mkdir -p ${AIRFLOW_HOME}/dags
mkdir -p ${AIRFLOW_HOME}/logs
mkdir -p ${AIRFLOW_HOME}/plugins
mkdir -p ${AIRFLOW_HOME}/config

# Option 1: Docker Compose Setup (Recommended)
echo "🐳 Setting up Airflow with Docker Compose..."

# Download docker-compose.yaml for Airflow
cd ${AIRFLOW_HOME}

if [ ! -f "docker-compose.yaml" ]; then
    echo "📥 Downloading Airflow Docker Compose configuration..."
    curl -LfO 'https://airflow.apache.org/docs/apache-airflow/stable/docker-compose.yaml'

    # Create .env file with configuration
    cat > .env << EOF
AIRFLOW_UID=$(id -u)
AIRFLOW_GID=0
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=mlops2024
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__CORE__LOAD_EXAMPLES=false
AIRFLOW__API__AUTH_BACKENDS=airflow.api.auth.backend.basic_auth
EOF

    echo "✅ Configuration files created"
fi

# Initialize Airflow database
echo "🗄️ Initializing Airflow database..."
docker-compose up airflow-init

# Start Airflow services
echo "🚀 Starting Airflow services..."
docker-compose up -d

# Wait for Airflow to be ready
echo "⏳ Waiting for Airflow to start (this may take a minute)..."
sleep 30

# Check if Airflow is running
AIRFLOW_RUNNING=false
for i in {1..10}; do
    if curl -s http://localhost:8080/health | grep -q "healthy"; then
        AIRFLOW_RUNNING=true
        break
    fi
    echo "   Waiting... (attempt $i/10)"
    sleep 10
done

if [ "$AIRFLOW_RUNNING" = true ]; then
    echo "✅ Airflow is running!"
    echo ""
    echo "📊 Airflow Web UI: http://localhost:8080"
    echo "   Username: admin"
    echo "   Password: mlops2024"
else
    echo "⚠️ Airflow may still be starting. Check http://localhost:8080 in a few minutes."
fi

# Show running containers
echo ""
echo "🐳 Airflow containers:"
docker-compose ps

# Create a sample DAG
echo ""
echo "📝 Creating sample MLOps DAG..."

cat > ${AIRFLOW_HOME}/dags/mlops_pipeline.py << 'EOF'
"""
MLOps Pipeline DAG
Orchestrates data processing, model training, and deployment
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.task_group import TaskGroup

# Default arguments
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define DAG
dag = DAG(
    'mlops_pipeline',
    default_args=default_args,
    description='MLOps data processing and model deployment pipeline',
    schedule_interval='@daily',
    catchup=False,
    tags=['mlops', 'production'],
)

def extract_data(**context):
    """Extract data from sources"""
    print("Extracting data from S3, APIs, and databases...")
    # Implementation here
    return {'records_extracted': 1000}

def transform_data(**context):
    """Transform and clean data"""
    print("Transforming data...")
    # Implementation here
    return {'records_transformed': 950}

def validate_data(**context):
    """Validate data quality"""
    print("Validating data quality...")
    # Implementation here
    return {'validation_passed': True}

def train_model(**context):
    """Train ML model"""
    print("Training model...")
    # Implementation here
    return {'model_accuracy': 0.95}

def deploy_model(**context):
    """Deploy model to production"""
    print("Deploying model...")
    # Implementation here
    return {'deployment_status': 'success'}

# Define tasks
with dag:
    # Data ingestion task group
    with TaskGroup('data_ingestion') as data_ingestion:
        extract_task = PythonOperator(
            task_id='extract_data',
            python_callable=extract_data,
            provide_context=True,
        )

        transform_task = PythonOperator(
            task_id='transform_data',
            python_callable=transform_data,
            provide_context=True,
        )

        validate_task = PythonOperator(
            task_id='validate_data',
            python_callable=validate_data,
            provide_context=True,
        )

        extract_task >> transform_task >> validate_task

    # Model training task
    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        provide_context=True,
    )

    # Model deployment task
    deploy_task = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model,
        provide_context=True,
    )

    # Health check
    health_check = BashOperator(
        task_id='health_check',
        bash_command='echo "Pipeline completed successfully"',
    )

    # Define dependencies
    data_ingestion >> train_task >> deploy_task >> health_check

EOF

echo "✅ Sample DAG created at ${AIRFLOW_HOME}/dags/mlops_pipeline.py"

echo ""
echo "=============================================="
echo "✅ Airflow setup complete!"
echo "=============================================="
echo ""
echo "📌 Important Information:"
echo "   Web UI: http://localhost:8080"
echo "   Username: admin"
echo "   Password: mlops2024"
echo ""
echo "📁 Directories:"
echo "   DAGs: ${AIRFLOW_HOME}/dags"
echo "   Logs: ${AIRFLOW_HOME}/logs"
echo "   Plugins: ${AIRFLOW_HOME}/plugins"
echo ""
echo "🔧 Useful Commands:"
echo "   View logs: cd ${AIRFLOW_HOME} && docker-compose logs -f"
echo "   Stop Airflow: cd ${AIRFLOW_HOME} && docker-compose down"
echo "   Restart: cd ${AIRFLOW_HOME} && docker-compose restart"
echo "   Shell access: docker exec -it airflow-airflow-webserver-1 bash"
echo ""

# Keep the script running to show logs
echo "📝 Showing Airflow logs (Ctrl+C to exit)..."
cd ${AIRFLOW_HOME}
docker-compose logs -f --tail=50