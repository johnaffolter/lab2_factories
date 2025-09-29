#!/usr/bin/env python3
"""
Airbyte Modular DAGs for Advanced Analysis
Sophisticated data pipeline orchestration with ML integration and advanced analytics
"""

import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import logging
from pathlib import Path

# Airflow imports (if available)
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
    from airflow.operators.dummy import DummyOperator
    from airflow.operators.email import EmailOperator
    from airflow.sensors.filesystem import FileSensor
    from airflow.providers.postgres.operators.postgres import PostgresOperator
    from airflow.providers.http.operators.http import SimpleHttpOperator
    from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
    from airflow.models import Variable
    from airflow.hooks.base import BaseHook
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    print("Airflow not available - DAG definitions will be created as configuration only")

class PipelineStage(Enum):
    """Pipeline execution stages"""
    INGESTION = "ingestion"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    ENRICHMENT = "enrichment"
    ANALYSIS = "analysis"
    ML_INFERENCE = "ml_inference"
    STORAGE = "storage"
    NOTIFICATION = "notification"

class DataSource(Enum):
    """Supported data sources"""
    GMAIL_API = "gmail_api"
    OUTLOOK_API = "outlook_api"
    IMAP_EMAIL = "imap_email"
    WEBHOOK_EMAIL = "webhook_email"
    FILE_UPLOAD = "file_upload"
    DATABASE_EXTRACT = "database_extract"
    API_ENDPOINT = "api_endpoint"
    STREAMING_SOURCE = "streaming_source"

class AnalysisType(Enum):
    """Types of analysis to perform"""
    EMAIL_CLASSIFICATION = "email_classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    GRAMMAR_CHECK = "grammar_check"
    SECURITY_SCAN = "security_scan"
    ATTACHMENT_ANALYSIS = "attachment_analysis"
    THREAD_ANALYSIS = "thread_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    PATTERN_RECOGNITION = "pattern_recognition"

@dataclass
class DAGConfiguration:
    """Configuration for DAG generation"""
    dag_id: str
    description: str
    schedule_interval: Union[str, timedelta]
    start_date: datetime
    catchup: bool = False
    max_active_runs: int = 1
    tags: List[str] = field(default_factory=list)
    default_args: Dict[str, Any] = field(default_factory=dict)
    data_sources: List[DataSource] = field(default_factory=list)
    analysis_types: List[AnalysisType] = field(default_factory=list)
    notifications: Dict[str, Any] = field(default_factory=dict)
    retries: int = 2
    retry_delay: timedelta = timedelta(minutes=5)

@dataclass
class TaskConfiguration:
    """Configuration for individual tasks"""
    task_id: str
    stage: PipelineStage
    operator_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)

class ModularDAGBuilder:
    """Builder for creating modular Airbyte-integrated DAGs"""

    def __init__(self):
        self.configs: Dict[str, DAGConfiguration] = {}
        self.task_templates: Dict[str, Dict[str, Any]] = self._load_task_templates()
        self.connection_configs: Dict[str, Dict[str, Any]] = self._load_connection_configs()

    def _load_task_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load reusable task templates"""
        return {
            "airbyte_sync": {
                "operator": "BashOperator",
                "template": "airbyte sync --connection-id {{params.connection_id}} --workspace-id {{params.workspace_id}}",
                "parameters": {
                    "connection_id": None,
                    "workspace_id": None,
                    "sync_mode": "full_refresh",
                    "destination_namespace": "raw_data"
                }
            },
            "email_extraction": {
                "operator": "PythonOperator",
                "template": "extract_emails_from_source",
                "parameters": {
                    "source_type": "gmail",
                    "batch_size": 100,
                    "filters": {},
                    "output_format": "json"
                }
            },
            "ml_classification": {
                "operator": "PythonOperator",
                "template": "classify_emails_ml",
                "parameters": {
                    "model_path": "/models/email_classifier.pkl",
                    "confidence_threshold": 0.8,
                    "batch_processing": True,
                    "feature_extractors": ["spam", "word_length", "non_text", "embeddings"]
                }
            },
            "data_quality_check": {
                "operator": "PythonOperator",
                "template": "validate_data_quality",
                "parameters": {
                    "quality_rules": ["completeness", "uniqueness", "validity"],
                    "tolerance_threshold": 0.95,
                    "fail_on_error": False
                }
            },
            "advanced_analysis": {
                "operator": "PythonOperator",
                "template": "perform_advanced_analysis",
                "parameters": {
                    "analysis_types": ["grammar", "tone", "security"],
                    "output_detailed_report": True,
                    "generate_suggestions": True
                }
            },
            "neo4j_storage": {
                "operator": "PythonOperator",
                "template": "store_in_neo4j",
                "parameters": {
                    "neo4j_uri": "bolt://localhost:7687",
                    "batch_size": 1000,
                    "create_relationships": True,
                    "node_types": ["Email", "Person", "Organization"]
                }
            },
            "slack_notification": {
                "operator": "SlackWebhookOperator",
                "template": None,
                "parameters": {
                    "webhook_token": "{{ var.value.slack_webhook }}",
                    "message": "Pipeline {{dag.dag_id}} completed successfully",
                    "channel": "#data-pipeline"
                }
            }
        }

    def _load_connection_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load Airbyte connection configurations"""
        return {
            "gmail_to_postgres": {
                "source": {
                    "name": "Gmail",
                    "connector_type": "source-gmail",
                    "configuration": {
                        "credentials": {
                            "auth_type": "oauth2",
                            "client_id": "{{ var.value.gmail_client_id }}",
                            "client_secret": "{{ var.value.gmail_client_secret }}",
                            "refresh_token": "{{ var.value.gmail_refresh_token }}"
                        },
                        "start_date": "2024-01-01T00:00:00Z",
                        "query": "is:unread",
                        "batch_size": 100
                    }
                },
                "destination": {
                    "name": "PostgreSQL",
                    "connector_type": "destination-postgres",
                    "configuration": {
                        "host": "{{ var.value.postgres_host }}",
                        "port": 5432,
                        "database": "email_analytics",
                        "username": "{{ var.value.postgres_user }}",
                        "password": "{{ var.value.postgres_password }}",
                        "schema": "raw_emails"
                    }
                }
            },
            "outlook_to_snowflake": {
                "source": {
                    "name": "Outlook",
                    "connector_type": "source-microsoft-outlook",
                    "configuration": {
                        "tenant_id": "{{ var.value.outlook_tenant_id }}",
                        "client_id": "{{ var.value.outlook_client_id }}",
                        "client_secret": "{{ var.value.outlook_client_secret }}",
                        "start_date": "2024-01-01T00:00:00Z"
                    }
                },
                "destination": {
                    "name": "Snowflake",
                    "connector_type": "destination-snowflake",
                    "configuration": {
                        "host": "{{ var.value.snowflake_account }}.snowflakecomputing.com",
                        "role": "AIRBYTE_ROLE",
                        "warehouse": "AIRBYTE_WAREHOUSE",
                        "database": "EMAIL_ANALYTICS",
                        "schema": "RAW_DATA"
                    }
                }
            },
            "files_to_s3": {
                "source": {
                    "name": "Local Files",
                    "connector_type": "source-file",
                    "configuration": {
                        "dataset_name": "email_exports",
                        "format": "jsonl",
                        "url": "/data/email_exports/*.jsonl",
                        "provider": {
                            "storage": "local"
                        }
                    }
                },
                "destination": {
                    "name": "Amazon S3",
                    "connector_type": "destination-s3",
                    "configuration": {
                        "s3_bucket_name": "{{ var.value.s3_bucket }}",
                        "s3_bucket_path": "email-data",
                        "s3_bucket_region": "us-west-2",
                        "format": {
                            "format_type": "JSONL"
                        }
                    }
                }
            }
        }

    def create_email_classification_dag(self) -> DAGConfiguration:
        """Create comprehensive email classification DAG"""

        config = DAGConfiguration(
            dag_id="email_classification_pipeline",
            description="Advanced email classification with ML and analytics",
            schedule_interval=timedelta(hours=1),
            start_date=datetime(2024, 1, 1),
            catchup=False,
            max_active_runs=1,
            tags=["email", "classification", "ml", "airbyte"],
            data_sources=[DataSource.GMAIL_API, DataSource.OUTLOOK_API],
            analysis_types=[
                AnalysisType.EMAIL_CLASSIFICATION,
                AnalysisType.SENTIMENT_ANALYSIS,
                AnalysisType.SECURITY_SCAN,
                AnalysisType.GRAMMAR_CHECK
            ],
            notifications={
                "on_failure": ["slack", "email"],
                "on_success": ["slack"],
                "on_retry": ["email"]
            }
        )

        self.configs[config.dag_id] = config
        return config

    def create_security_analysis_dag(self) -> DAGConfiguration:
        """Create security-focused email analysis DAG"""

        config = DAGConfiguration(
            dag_id="email_security_analysis",
            description="Security analysis and threat detection for emails",
            schedule_interval=timedelta(minutes=30),
            start_date=datetime(2024, 1, 1),
            catchup=False,
            max_active_runs=2,
            tags=["security", "threat-detection", "email", "airbyte"],
            data_sources=[DataSource.GMAIL_API, DataSource.IMAP_EMAIL],
            analysis_types=[
                AnalysisType.SECURITY_SCAN,
                AnalysisType.ATTACHMENT_ANALYSIS,
                AnalysisType.ANOMALY_DETECTION
            ],
            notifications={
                "on_failure": ["slack", "email", "pagerduty"],
                "on_success": ["slack"],
                "on_security_alert": ["slack", "email", "sms"]
            }
        )

        self.configs[config.dag_id] = config
        return config

    def create_real_time_analysis_dag(self) -> DAGConfiguration:
        """Create real-time email analysis DAG"""

        config = DAGConfiguration(
            dag_id="realtime_email_analysis",
            description="Real-time email processing and classification",
            schedule_interval=None,  # Triggered by external events
            start_date=datetime(2024, 1, 1),
            catchup=False,
            max_active_runs=10,
            tags=["realtime", "streaming", "email", "airbyte"],
            data_sources=[DataSource.WEBHOOK_EMAIL, DataSource.STREAMING_SOURCE],
            analysis_types=[
                AnalysisType.EMAIL_CLASSIFICATION,
                AnalysisType.SENTIMENT_ANALYSIS,
                AnalysisType.GRAMMAR_CHECK
            ]
        )

        self.configs[config.dag_id] = config
        return config

    def create_batch_analytics_dag(self) -> DAGConfiguration:
        """Create batch analytics and reporting DAG"""

        config = DAGConfiguration(
            dag_id="email_batch_analytics",
            description="Daily batch analytics and ML model training",
            schedule_interval="0 2 * * *",  # Daily at 2 AM
            start_date=datetime(2024, 1, 1),
            catchup=True,
            max_active_runs=1,
            tags=["analytics", "batch", "ml-training", "reporting"],
            data_sources=[DataSource.DATABASE_EXTRACT],
            analysis_types=[
                AnalysisType.PATTERN_RECOGNITION,
                AnalysisType.THREAD_ANALYSIS,
                AnalysisType.ANOMALY_DETECTION
            ]
        )

        self.configs[config.dag_id] = config
        return config

    def generate_dag_tasks(self, config: DAGConfiguration) -> List[TaskConfiguration]:
        """Generate tasks for a DAG configuration"""
        tasks = []

        # 1. Ingestion tasks
        for source in config.data_sources:
            tasks.append(TaskConfiguration(
                task_id=f"ingest_{source.value}",
                stage=PipelineStage.INGESTION,
                operator_type="airbyte_sync",
                parameters={
                    "source_type": source.value,
                    "connection_config": self._get_connection_config(source),
                    "sync_mode": "incremental"
                }
            ))

        # 2. Data validation
        tasks.append(TaskConfiguration(
            task_id="validate_ingested_data",
            stage=PipelineStage.VALIDATION,
            operator_type="data_quality_check",
            dependencies=[f"ingest_{source.value}" for source in config.data_sources],
            parameters={
                "validation_rules": [
                    "email_format_validation",
                    "duplicate_detection",
                    "completeness_check"
                ]
            }
        ))

        # 3. Email extraction and preprocessing
        tasks.append(TaskConfiguration(
            task_id="extract_email_features",
            stage=PipelineStage.TRANSFORMATION,
            operator_type="email_extraction",
            dependencies=["validate_ingested_data"],
            parameters={
                "feature_extractors": ["subject", "body", "sender", "attachments"],
                "preprocessing_steps": ["clean_text", "extract_metadata"]
            }
        ))

        # 4. Analysis tasks based on configuration
        for analysis_type in config.analysis_types:
            tasks.append(TaskConfiguration(
                task_id=f"analyze_{analysis_type.value}",
                stage=PipelineStage.ANALYSIS,
                operator_type="advanced_analysis",
                dependencies=["extract_email_features"],
                parameters={
                    "analysis_type": analysis_type.value,
                    "batch_processing": True,
                    "output_format": "structured_json"
                }
            ))

        # 5. ML inference (if classification is enabled)
        if AnalysisType.EMAIL_CLASSIFICATION in config.analysis_types:
            tasks.append(TaskConfiguration(
                task_id="ml_email_classification",
                stage=PipelineStage.ML_INFERENCE,
                operator_type="ml_classification",
                dependencies=["extract_email_features"],
                parameters={
                    "model_ensemble": True,
                    "confidence_reporting": True,
                    "fallback_rules": True
                }
            ))

        # 6. Data enrichment
        tasks.append(TaskConfiguration(
            task_id="enrich_analysis_results",
            stage=PipelineStage.ENRICHMENT,
            operator_type="data_enrichment",
            dependencies=[f"analyze_{at.value}" for at in config.analysis_types],
            parameters={
                "enrichment_sources": ["external_apis", "knowledge_base"],
                "entity_resolution": True
            }
        ))

        # 7. Storage tasks
        tasks.append(TaskConfiguration(
            task_id="store_in_data_warehouse",
            stage=PipelineStage.STORAGE,
            operator_type="data_warehouse_load",
            dependencies=["enrich_analysis_results"],
            parameters={
                "warehouse_type": "snowflake",
                "schema": "processed_emails",
                "partition_by": "date"
            }
        ))

        tasks.append(TaskConfiguration(
            task_id="store_in_neo4j",
            stage=PipelineStage.STORAGE,
            operator_type="neo4j_storage",
            dependencies=["enrich_analysis_results"],
            parameters={
                "graph_schema": "email_network",
                "relationship_types": ["SENT_TO", "REPLIED_TO", "FORWARDED"]
            }
        ))

        # 8. Notifications
        if config.notifications:
            tasks.append(TaskConfiguration(
                task_id="send_completion_notification",
                stage=PipelineStage.NOTIFICATION,
                operator_type="slack_notification",
                dependencies=["store_in_data_warehouse", "store_in_neo4j"],
                parameters={
                    "notification_channels": config.notifications
                }
            ))

        return tasks

    def _get_connection_config(self, source: DataSource) -> Dict[str, Any]:
        """Get connection configuration for data source"""
        source_mapping = {
            DataSource.GMAIL_API: "gmail_to_postgres",
            DataSource.OUTLOOK_API: "outlook_to_snowflake",
            DataSource.FILE_UPLOAD: "files_to_s3"
        }

        return self.connection_configs.get(
            source_mapping.get(source, "gmail_to_postgres"),
            {}
        )

    def generate_airflow_dag_code(self, config: DAGConfiguration) -> str:
        """Generate Airflow DAG Python code"""

        tasks = self.generate_dag_tasks(config)

        dag_code = f'''
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
default_args = {{
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime{config.start_date.timetuple()[:6]},
    'email_on_failure': {'True' if 'email' in config.notifications.get('on_failure', []) else 'False'},
    'email_on_retry': {'True' if 'email' in config.notifications.get('on_retry', []) else 'False'},
    'retries': {config.retries},
    'retry_delay': timedelta(minutes={config.retry_delay.total_seconds()//60}),
}}

dag = DAG(
    '{config.dag_id}',
    default_args=default_args,
    description='{config.description}',
    schedule_interval={'None' if config.schedule_interval is None else repr(config.schedule_interval)},
    catchup={config.catchup},
    max_active_runs={config.max_active_runs},
    tags={config.tags}
)

# Task Definitions
'''

        # Generate task definitions
        task_objects = {}

        for task in tasks:
            template = self.task_templates.get(task.operator_type, {})

            if task.operator_type == "airbyte_sync":
                task_code = f'''
{task.task_id} = BashOperator(
    task_id='{task.task_id}',
    bash_command='airbyte sync --connection-id {{{{ params.connection_id }}}} --workspace-id {{{{ params.workspace_id }}}}',
    params={task.parameters},
    dag=dag
)
'''
            elif task.operator_type in ["email_extraction", "ml_classification", "advanced_analysis", "data_quality_check"]:
                task_code = f'''
{task.task_id} = PythonOperator(
    task_id='{task.task_id}',
    python_callable={template.get('template', 'default_function')},
    op_kwargs={task.parameters},
    dag=dag
)
'''
            elif task.operator_type == "slack_notification":
                task_code = f'''
{task.task_id} = SlackWebhookOperator(
    task_id='{task.task_id}',
    http_conn_id='slack_default',
    webhook_token='{{{{ var.value.slack_webhook }}}}',
    message='Pipeline {config.dag_id} - {{{{ task.task_id }}}} completed',
    channel='#data-pipeline',
    dag=dag
)
'''
            else:
                task_code = f'''
{task.task_id} = DummyOperator(
    task_id='{task.task_id}',
    dag=dag
)
'''

            dag_code += task_code
            task_objects[task.task_id] = task

        # Generate dependencies
        dag_code += "\n# Task Dependencies\n"
        for task in tasks:
            if task.dependencies:
                dependencies_str = " >> ".join(task.dependencies)
                dag_code += f"{dependencies_str} >> {task.task_id}\n"

        return dag_code

    def generate_airbyte_config_yaml(self, config: DAGConfiguration) -> str:
        """Generate Airbyte configuration YAML"""

        airbyte_config = {
            "version": "0.50.0",
            "project_name": f"email_analytics_{config.dag_id}",
            "connections": []
        }

        for source in config.data_sources:
            connection_config = self._get_connection_config(source)
            if connection_config:
                airbyte_config["connections"].append({
                    "name": f"{source.value}_connection",
                    "source": connection_config["source"],
                    "destination": connection_config["destination"],
                    "sync_mode": "incremental",
                    "schedule": {
                        "schedule_type": "cron",
                        "cron_expression": "0 */1 * * *"  # Every hour
                    }
                })

        return yaml.dump(airbyte_config, default_flow_style=False, indent=2)

    def generate_docker_compose(self) -> str:
        """Generate Docker Compose for the complete stack"""

        docker_compose = """
version: '3.8'

services:
  # Airbyte Services
  airbyte-server:
    image: airbyte/server:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://airbyte:password@airbyte-db:5432/airbyte
      - CONFIG_DATABASE_URL=postgresql://airbyte:password@airbyte-db:5432/airbyte_config
    depends_on:
      - airbyte-db
    volumes:
      - airbyte_workspace:/tmp/airbyte_local
      - airbyte_data:/tmp/airbyte_local_data
      - airbyte_logs:/tmp/airbyte_local_logs

  airbyte-worker:
    image: airbyte/worker:latest
    environment:
      - DATABASE_URL=postgresql://airbyte:password@airbyte-db:5432/airbyte
      - CONFIG_DATABASE_URL=postgresql://airbyte:password@airbyte-db:5432/airbyte_config
    depends_on:
      - airbyte-db
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - airbyte_workspace:/tmp/airbyte_local
      - airbyte_data:/tmp/airbyte_local_data
      - airbyte_logs:/tmp/airbyte_local_logs

  airbyte-webapp:
    image: airbyte/webapp:latest
    ports:
      - "8001:80"
    environment:
      - API_URL=http://airbyte-server:8000/api/v1/

  airbyte-db:
    image: postgres:13
    environment:
      - POSTGRES_DB=airbyte
      - POSTGRES_USER=airbyte
      - POSTGRES_PASSWORD=password
    volumes:
      - airbyte_db_data:/var/lib/postgresql/data

  # Airflow Services
  airflow-webserver:
    image: apache/airflow:2.5.0
    ports:
      - "8080:8080"
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@airflow-db:5432/airflow
      - AIRFLOW__CORE__FERNET_KEY=your_fernet_key_here
      - AIRFLOW__WEBSERVER__SECRET_KEY=your_secret_key_here
    depends_on:
      - airflow-db
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
      - ./includes:/opt/airflow/dags/includes
    command: webserver

  airflow-scheduler:
    image: apache/airflow:2.5.0
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@airflow-db:5432/airflow
      - AIRFLOW__CORE__FERNET_KEY=your_fernet_key_here
    depends_on:
      - airflow-db
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
      - ./includes:/opt/airflow/dags/includes
    command: scheduler

  airflow-db:
    image: postgres:13
    environment:
      - POSTGRES_DB=airflow
      - POSTGRES_USER=airflow
      - POSTGRES_PASSWORD=airflow
    volumes:
      - airflow_db_data:/var/lib/postgresql/data

  # Data Stores
  neo4j:
    image: neo4j:5.0
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc", "graph-data-science"]
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs

  postgres-analytics:
    image: postgres:14
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=email_analytics
      - POSTGRES_USER=analytics
      - POSTGRES_PASSWORD=analytics
    volumes:
      - postgres_analytics_data:/var/lib/postgresql/data

  # ML Services
  mlflow:
    image: python:3.9
    ports:
      - "5000:5000"
    command: >
      bash -c "pip install mlflow psycopg2-binary &&
               mlflow server --host 0.0.0.0 --port 5000
               --backend-store-uri postgresql://mlflow:mlflow@mlflow-db:5432/mlflow
               --default-artifact-root /mlflow/artifacts"
    depends_on:
      - mlflow-db
    volumes:
      - mlflow_artifacts:/mlflow/artifacts

  mlflow-db:
    image: postgres:13
    environment:
      - POSTGRES_DB=mlflow
      - POSTGRES_USER=mlflow
      - POSTGRES_PASSWORD=mlflow
    volumes:
      - mlflow_db_data:/var/lib/postgresql/data

  # Monitoring
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  airbyte_workspace:
  airbyte_data:
  airbyte_logs:
  airbyte_db_data:
  airflow_db_data:
  neo4j_data:
  neo4j_logs:
  postgres_analytics_data:
  mlflow_artifacts:
  mlflow_db_data:
  grafana_data:
"""
        return docker_compose.strip()

    def export_all_configurations(self, output_dir: str = "airbyte_dags_output"):
        """Export all DAG configurations and supporting files"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Create DAG configurations
        configs = [
            self.create_email_classification_dag(),
            self.create_security_analysis_dag(),
            self.create_real_time_analysis_dag(),
            self.create_batch_analytics_dag()
        ]

        # Generate and save DAG files
        dags_dir = output_path / "dags"
        dags_dir.mkdir(exist_ok=True)

        for config in configs:
            # Generate Airflow DAG
            dag_code = self.generate_airflow_dag_code(config)
            with open(dags_dir / f"{config.dag_id}.py", "w") as f:
                f.write(dag_code)

            # Generate Airbyte config
            airbyte_config = self.generate_airbyte_config_yaml(config)
            with open(output_path / f"{config.dag_id}_airbyte_config.yaml", "w") as f:
                f.write(airbyte_config)

        # Generate Docker Compose
        docker_compose = self.generate_docker_compose()
        with open(output_path / "docker-compose.yml", "w") as f:
            f.write(docker_compose)

        # Generate requirements file
        requirements = """
apache-airflow==2.5.0
apache-airflow-providers-postgres
apache-airflow-providers-slack
apache-airflow-providers-http
psycopg2-binary
sqlalchemy
pandas
numpy
scikit-learn
spacy
neo4j
requests
pyyaml
mlflow
"""
        with open(output_path / "requirements.txt", "w") as f:
            f.write(requirements.strip())

        # Generate environment variables template
        env_template = """
# Airbyte
AIRBYTE_API_URL=http://localhost:8000/api/v1
AIRBYTE_WORKSPACE_ID=your_workspace_id

# Gmail API
GMAIL_CLIENT_ID=your_gmail_client_id
GMAIL_CLIENT_SECRET=your_gmail_client_secret
GMAIL_REFRESH_TOKEN=your_gmail_refresh_token

# Outlook API
OUTLOOK_TENANT_ID=your_outlook_tenant_id
OUTLOOK_CLIENT_ID=your_outlook_client_id
OUTLOOK_CLIENT_SECRET=your_outlook_client_secret

# Database Connections
POSTGRES_HOST=localhost
POSTGRES_USER=analytics
POSTGRES_PASSWORD=analytics
POSTGRES_DB=email_analytics

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Snowflake
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password

# S3
S3_BUCKET=your_s3_bucket
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Notifications
SLACK_WEBHOOK=your_slack_webhook_url
"""
        with open(output_path / ".env.template", "w") as f:
            f.write(env_template.strip())

        # Generate README
        readme = f"""
# Airbyte Modular DAGs for Email Analysis

This directory contains modular DAGs for advanced email analysis using Airbyte and Airflow.

## Generated DAGs

{chr(10).join([f"- `{config.dag_id}`: {config.description}" for config in configs])}

## Setup Instructions

1. **Environment Setup**:
   ```bash
   cp .env.template .env
   # Edit .env with your actual credentials
   ```

2. **Start Services**:
   ```bash
   docker-compose up -d
   ```

3. **Access Services**:
   - Airbyte UI: http://localhost:8001
   - Airflow UI: http://localhost:8080
   - Neo4j Browser: http://localhost:7474
   - Grafana: http://localhost:3000

4. **Configure Airbyte Connections**:
   - Import the generated YAML configurations
   - Set up your data source credentials
   - Test connections before running DAGs

## DAG Features

- **Modular Design**: Each DAG is built from reusable components
- **Multiple Data Sources**: Gmail, Outlook, Files, APIs
- **Advanced Analytics**: ML classification, security analysis, grammar checking
- **Scalable Storage**: PostgreSQL, Neo4j, Snowflake support
- **Monitoring**: Integrated logging and notifications

## Customization

Modify the DAG configurations in the Python files to:
- Add new data sources
- Change analysis types
- Adjust scheduling
- Add custom processing steps

## Monitoring

Each DAG includes:
- Data quality checks
- Error handling and retries
- Slack notifications
- Comprehensive logging
"""
        with open(output_path / "README.md", "w") as f:
            f.write(readme.strip())

        return output_path

def main():
    """Demonstrate the modular DAG builder"""
    print("🏗️ AIRBYTE MODULAR DAG SYSTEM")
    print("Creating sophisticated data pipelines for email analysis")
    print("=" * 70)

    # Initialize builder
    builder = ModularDAGBuilder()

    # Export all configurations
    output_dir = builder.export_all_configurations()

    print(f"\n📁 Generated DAG System:")
    print(f"   Output Directory: {output_dir}")
    print(f"   DAG Files: {len(list((output_dir / 'dags').glob('*.py')))}")
    print(f"   Airbyte Configs: {len(list(output_dir.glob('*_airbyte_config.yaml')))}")

    print(f"\n🚀 Features Included:")
    print(f"   ✓ Email classification with ML")
    print(f"   ✓ Security analysis and threat detection")
    print(f"   ✓ Real-time processing capabilities")
    print(f"   ✓ Batch analytics and reporting")
    print(f"   ✓ Multi-source data ingestion")
    print(f"   ✓ Advanced analytics (grammar, tone, sentiment)")
    print(f"   ✓ Graph database integration (Neo4j)")
    print(f"   ✓ Data warehouse support")
    print(f"   ✓ Monitoring and notifications")

    print(f"\n💡 Next Steps:")
    print(f"   1. Review generated configurations")
    print(f"   2. Set up environment variables")
    print(f"   3. Start services with docker-compose")
    print(f"   4. Configure Airbyte connections")
    print(f"   5. Deploy DAGs to Airflow")

    return builder, output_dir

if __name__ == "__main__":
    main()