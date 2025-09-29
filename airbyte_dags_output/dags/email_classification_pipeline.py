
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
    'email_on_failure': True,
    'email_on_retry': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=5.0),
}

dag = DAG(
    'email_classification_pipeline',
    default_args=default_args,
    description='Advanced email classification with ML and analytics',
    schedule_interval=datetime.timedelta(seconds=3600),
    catchup=False,
    max_active_runs=1,
    tags=['email', 'classification', 'ml', 'airbyte']
)

# Task Definitions

ingest_gmail_api = BashOperator(
    task_id='ingest_gmail_api',
    bash_command='airbyte sync --connection-id {{ params.connection_id }} --workspace-id {{ params.workspace_id }}',
    params={'source_type': 'gmail_api', 'connection_config': {'source': {'name': 'Gmail', 'connector_type': 'source-gmail', 'configuration': {'credentials': {'auth_type': 'oauth2', 'client_id': '{{ var.value.gmail_client_id }}', 'client_secret': '{{ var.value.gmail_client_secret }}', 'refresh_token': '{{ var.value.gmail_refresh_token }}'}, 'start_date': '2024-01-01T00:00:00Z', 'query': 'is:unread', 'batch_size': 100}}, 'destination': {'name': 'PostgreSQL', 'connector_type': 'destination-postgres', 'configuration': {'host': '{{ var.value.postgres_host }}', 'port': 5432, 'database': 'email_analytics', 'username': '{{ var.value.postgres_user }}', 'password': '{{ var.value.postgres_password }}', 'schema': 'raw_emails'}}}, 'sync_mode': 'incremental'},
    dag=dag
)

ingest_outlook_api = BashOperator(
    task_id='ingest_outlook_api',
    bash_command='airbyte sync --connection-id {{ params.connection_id }} --workspace-id {{ params.workspace_id }}',
    params={'source_type': 'outlook_api', 'connection_config': {'source': {'name': 'Outlook', 'connector_type': 'source-microsoft-outlook', 'configuration': {'tenant_id': '{{ var.value.outlook_tenant_id }}', 'client_id': '{{ var.value.outlook_client_id }}', 'client_secret': '{{ var.value.outlook_client_secret }}', 'start_date': '2024-01-01T00:00:00Z'}}, 'destination': {'name': 'Snowflake', 'connector_type': 'destination-snowflake', 'configuration': {'host': '{{ var.value.snowflake_account }}.snowflakecomputing.com', 'role': 'AIRBYTE_ROLE', 'warehouse': 'AIRBYTE_WAREHOUSE', 'database': 'EMAIL_ANALYTICS', 'schema': 'RAW_DATA'}}}, 'sync_mode': 'incremental'},
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

analyze_email_classification = PythonOperator(
    task_id='analyze_email_classification',
    python_callable=perform_advanced_analysis,
    op_kwargs={'analysis_type': 'email_classification', 'batch_processing': True, 'output_format': 'structured_json'},
    dag=dag
)

analyze_sentiment_analysis = PythonOperator(
    task_id='analyze_sentiment_analysis',
    python_callable=perform_advanced_analysis,
    op_kwargs={'analysis_type': 'sentiment_analysis', 'batch_processing': True, 'output_format': 'structured_json'},
    dag=dag
)

analyze_security_scan = PythonOperator(
    task_id='analyze_security_scan',
    python_callable=perform_advanced_analysis,
    op_kwargs={'analysis_type': 'security_scan', 'batch_processing': True, 'output_format': 'structured_json'},
    dag=dag
)

analyze_grammar_check = PythonOperator(
    task_id='analyze_grammar_check',
    python_callable=perform_advanced_analysis,
    op_kwargs={'analysis_type': 'grammar_check', 'batch_processing': True, 'output_format': 'structured_json'},
    dag=dag
)

ml_email_classification = PythonOperator(
    task_id='ml_email_classification',
    python_callable=classify_emails_ml,
    op_kwargs={'model_ensemble': True, 'confidence_reporting': True, 'fallback_rules': True},
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

send_completion_notification = SlackWebhookOperator(
    task_id='send_completion_notification',
    http_conn_id='slack_default',
    webhook_token='{{ var.value.slack_webhook }}',
    message='Pipeline email_classification_pipeline - {{ task.task_id }} completed',
    channel='#data-pipeline',
    dag=dag
)

# Task Dependencies
ingest_gmail_api >> ingest_outlook_api >> validate_ingested_data
validate_ingested_data >> extract_email_features
extract_email_features >> analyze_email_classification
extract_email_features >> analyze_sentiment_analysis
extract_email_features >> analyze_security_scan
extract_email_features >> analyze_grammar_check
extract_email_features >> ml_email_classification
analyze_email_classification >> analyze_sentiment_analysis >> analyze_security_scan >> analyze_grammar_check >> enrich_analysis_results
enrich_analysis_results >> store_in_data_warehouse
enrich_analysis_results >> store_in_neo4j
store_in_data_warehouse >> store_in_neo4j >> send_completion_notification
