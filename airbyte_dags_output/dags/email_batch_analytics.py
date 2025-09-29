
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from datetime import datetime, timedelta
import sys
sys.path.append('/opt/airflow/dags/includes')
from email_processing_functions import *

# DAG Configuration
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1, 0, 0, 0),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5.0),
}

dag = DAG(
    'email_batch_analytics',
    default_args=default_args,
    description='Daily batch analytics and ML model training',
    schedule_interval='0 2 * * *',
    catchup=True,
    max_active_runs=1,
    tags=['analytics', 'batch', 'ml-training', 'reporting']
)

# Task Definitions

ingest_database_extract = BashOperator(
    task_id='ingest_database_extract',
    bash_command='airbyte sync --connection-id {{ params.connection_id }} --workspace-id {{ params.workspace_id }}',
    params={'source_type': 'database_extract', 'connection_config': {'source': {'name': 'Gmail', 'connector_type': 'source-gmail', 'configuration': {'credentials': {'auth_type': 'oauth2', 'client_id': '{{ var.value.gmail_client_id }}', 'client_secret': '{{ var.value.gmail_client_secret }}', 'refresh_token': '{{ var.value.gmail_refresh_token }}'}, 'start_date': '2024-01-01T00:00:00Z', 'query': 'is:unread', 'batch_size': 100}}, 'destination': {'name': 'PostgreSQL', 'connector_type': 'destination-postgres', 'configuration': {'host': '{{ var.value.postgres_host }}', 'port': 5432, 'database': 'email_analytics', 'username': '{{ var.value.postgres_user }}', 'password': '{{ var.value.postgres_password }}', 'schema': 'raw_emails'}}}, 'sync_mode': 'incremental'},
    dag=dag
)

validate_ingested_data = PythonOperator(
    task_id='validate_ingested_data',
    python_callable=validate_data_quality,
    op_kwargs={'validation_rules': ['email_format_validation', 'duplicate_detection', 'completeness_check']},
    dag=dag
)

extract_email_features = PythonOperator(
    task_id='extract_email_features',
    python_callable=extract_emails_from_source,
    op_kwargs={'feature_extractors': ['subject', 'body', 'sender', 'attachments'], 'preprocessing_steps': ['clean_text', 'extract_metadata']},
    dag=dag
)

analyze_pattern_recognition = PythonOperator(
    task_id='analyze_pattern_recognition',
    python_callable=perform_advanced_analysis,
    op_kwargs={'analysis_type': 'pattern_recognition', 'batch_processing': True, 'output_format': 'structured_json'},
    dag=dag
)

analyze_thread_analysis = PythonOperator(
    task_id='analyze_thread_analysis',
    python_callable=perform_advanced_analysis,
    op_kwargs={'analysis_type': 'thread_analysis', 'batch_processing': True, 'output_format': 'structured_json'},
    dag=dag
)

analyze_anomaly_detection = PythonOperator(
    task_id='analyze_anomaly_detection',
    python_callable=perform_advanced_analysis,
    op_kwargs={'analysis_type': 'anomaly_detection', 'batch_processing': True, 'output_format': 'structured_json'},
    dag=dag
)

enrich_analysis_results = DummyOperator(
    task_id='enrich_analysis_results',
    dag=dag
)

store_in_data_warehouse = DummyOperator(
    task_id='store_in_data_warehouse',
    dag=dag
)

store_in_neo4j = DummyOperator(
    task_id='store_in_neo4j',
    dag=dag
)

# Task Dependencies
ingest_database_extract >> validate_ingested_data
validate_ingested_data >> extract_email_features
extract_email_features >> analyze_pattern_recognition
extract_email_features >> analyze_thread_analysis
extract_email_features >> analyze_anomaly_detection
analyze_pattern_recognition >> analyze_thread_analysis >> analyze_anomaly_detection >> enrich_analysis_results
enrich_analysis_results >> store_in_data_warehouse
enrich_analysis_results >> store_in_neo4j
