"""
MLOps Data Pipeline DAG
Complete data pipeline integrating S3, databases, and ML processing
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.amazon.aws.operators.s3 import S3ListOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
import json

# Default arguments for the DAG
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['mlops@company.com'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'mlops_data_pipeline',
    default_args=default_args,
    description='Complete MLOps data pipeline with S3, databases, and ML',
    schedule_interval='0 2 * * *',  # Run at 2 AM daily
    catchup=False,
    tags=['mlops', 'data', 'production'],
)

def check_data_quality(**context):
    """Check data quality metrics"""
    import random

    # Simulate data quality check
    quality_score = random.uniform(0.85, 0.99)

    # Push to XCom for downstream tasks
    context['task_instance'].xcom_push(key='quality_score', value=quality_score)

    if quality_score < 0.9:
        raise ValueError(f"Data quality too low: {quality_score}")

    print(f"Data quality check passed: {quality_score}")
    return quality_score

def extract_from_s3(**context):
    """Extract data from S3 buckets"""
    print("Extracting data from S3...")

    # Simulate S3 extraction
    data = {
        'bucket': 'mlops-data',
        'files_processed': 42,
        'total_size_mb': 256.7,
        'timestamp': datetime.now().isoformat()
    }

    context['task_instance'].xcom_push(key='s3_data', value=json.dumps(data))
    print(f"Extracted {data['files_processed']} files from S3")
    return data

def load_to_postgres(**context):
    """Load processed data to PostgreSQL"""
    # Get upstream data from XCom
    s3_data = json.loads(context['task_instance'].xcom_pull(key='s3_data'))

    print(f"Loading data to PostgreSQL...")
    print(f"Processing {s3_data['files_processed']} files")

    # Simulate database load
    records_inserted = s3_data['files_processed'] * 100

    return {
        'records_inserted': records_inserted,
        'table': 'mlops_metrics',
        'timestamp': datetime.now().isoformat()
    }

def sync_to_neo4j(**context):
    """Sync graph data to Neo4j"""
    print("Syncing data to Neo4j graph database...")

    # Simulate Neo4j sync
    nodes_created = 150
    relationships_created = 450

    return {
        'nodes': nodes_created,
        'relationships': relationships_created
    }

def trigger_ml_training(**context):
    """Trigger ML model training"""
    quality_score = context['task_instance'].xcom_pull(key='quality_score')

    print(f"Triggering ML training with quality score: {quality_score}")

    # Simulate model training trigger
    training_config = {
        'model_type': 'gradient_boosting',
        'hyperparameters': {
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 5
        },
        'data_quality': quality_score,
        'triggered_at': datetime.now().isoformat()
    }

    context['task_instance'].xcom_push(key='training_config', value=json.dumps(training_config))
    return training_config

def evaluate_model(**context):
    """Evaluate trained model performance"""
    import random

    training_config = json.loads(context['task_instance'].xcom_pull(key='training_config'))

    print("Evaluating model performance...")

    # Simulate model evaluation
    metrics = {
        'accuracy': random.uniform(0.92, 0.98),
        'precision': random.uniform(0.90, 0.97),
        'recall': random.uniform(0.91, 0.96),
        'f1_score': random.uniform(0.91, 0.97),
        'auc_roc': random.uniform(0.93, 0.99)
    }

    print(f"Model metrics: {metrics}")

    # Decide on deployment based on performance
    if metrics['accuracy'] > 0.95:
        context['task_instance'].xcom_push(key='deploy_model', value=True)
    else:
        context['task_instance'].xcom_push(key='deploy_model', value=False)

    return metrics

def deploy_to_production(**context):
    """Deploy model to production if performance criteria met"""
    deploy = context['task_instance'].xcom_pull(key='deploy_model')

    if deploy:
        print("✅ Deploying model to production...")
        deployment_info = {
            'status': 'deployed',
            'endpoint': 'https://api.mlops.company.com/v1/predict',
            'version': f"v{datetime.now().strftime('%Y%m%d_%H%M')}",
            'deployed_at': datetime.now().isoformat()
        }
        print(f"Model deployed: {deployment_info['version']}")
        return deployment_info
    else:
        print("❌ Model performance below threshold, skipping deployment")
        return {'status': 'skipped'}

def cleanup_temp_data(**context):
    """Clean up temporary data and resources"""
    print("Cleaning up temporary data...")

    # Simulate cleanup
    cleaned_items = {
        'temp_files': 25,
        'cache_cleared': True,
        'space_freed_mb': 512
    }

    print(f"Cleanup complete: {cleaned_items}")
    return cleaned_items

def send_notification(**context):
    """Send pipeline completion notification"""
    # Gather all results
    deploy_info = context['task_instance'].xcom_pull(task_ids='deploy_to_production')

    print("Sending pipeline completion notification...")

    notification = {
        'pipeline': 'mlops_data_pipeline',
        'status': 'completed',
        'deployment': deploy_info,
        'timestamp': datetime.now().isoformat()
    }

    print(f"Notification sent: {notification}")
    return notification

# Define tasks with the DAG
with dag:

    # Check if services are healthy
    check_api_health = HttpSensor(
        task_id='check_api_health',
        http_conn_id='mlops_api',
        endpoint='/health',
        poke_interval=30,
        timeout=300,
        soft_fail=False,
    )

    # Data ingestion task group
    with TaskGroup('data_ingestion') as data_ingestion:

        # Extract from S3
        s3_extract = PythonOperator(
            task_id='extract_from_s3',
            python_callable=extract_from_s3,
        )

        # Data quality check
        quality_check = PythonOperator(
            task_id='check_data_quality',
            python_callable=check_data_quality,
        )

        # Load to PostgreSQL
        postgres_load = PythonOperator(
            task_id='load_to_postgres',
            python_callable=load_to_postgres,
        )

        # Sync to Neo4j
        neo4j_sync = PythonOperator(
            task_id='sync_to_neo4j',
            python_callable=sync_to_neo4j,
        )

        s3_extract >> quality_check >> [postgres_load, neo4j_sync]

    # ML pipeline task group
    with TaskGroup('ml_pipeline') as ml_pipeline:

        # Trigger training
        trigger_training = PythonOperator(
            task_id='trigger_ml_training',
            python_callable=trigger_ml_training,
        )

        # Evaluate model
        evaluate = PythonOperator(
            task_id='evaluate_model',
            python_callable=evaluate_model,
        )

        # Deploy to production
        deploy = PythonOperator(
            task_id='deploy_to_production',
            python_callable=deploy_to_production,
        )

        trigger_training >> evaluate >> deploy

    # Cleanup task
    cleanup = PythonOperator(
        task_id='cleanup_temp_data',
        python_callable=cleanup_temp_data,
        trigger_rule='all_done',  # Run regardless of upstream success/failure
    )

    # Notification task
    notify = PythonOperator(
        task_id='send_notification',
        python_callable=send_notification,
        trigger_rule='all_done',
    )

    # Create table if not exists (runs first)
    create_tables = PostgresOperator(
        task_id='create_tables',
        postgres_conn_id='mlops_postgres',
        sql="""
        CREATE TABLE IF NOT EXISTS mlops_metrics (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
            metric_name VARCHAR(255),
            metric_value FLOAT,
            metadata JSONB
        );
        """,
    )

    # Define task dependencies
    create_tables >> check_api_health >> data_ingestion >> ml_pipeline >> [cleanup, notify]