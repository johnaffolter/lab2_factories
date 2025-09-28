"""
Airflow DAG for Email Classification Pipeline
This DAG automates the email classification workflow

To use:
1. Copy this file to your Airflow DAGs folder
2. Configure the API_URL variable
3. Set up email sources (IMAP, files, etc.)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.http_operator import SimpleHttpOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.models import Variable
import json
import requests
import logging

# Configuration
API_URL = Variable.get("EMAIL_CLASSIFIER_API_URL", default_var="http://localhost:8000")

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'email_classification_pipeline',
    default_args=default_args,
    description='Automated email classification and topic management',
    schedule_interval=timedelta(hours=1),  # Run hourly
    catchup=False,
    tags=['email', 'classification', 'ml']
)

def check_api_health():
    """Check if the API is healthy"""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            logging.info("API is healthy")
            return True
        else:
            raise Exception(f"API unhealthy: {response.status_code}")
    except Exception as e:
        logging.error(f"API health check failed: {e}")
        raise

def fetch_new_emails(**context):
    """
    Fetch new emails from source (customize this for your email source)
    This is a placeholder - implement your email fetching logic
    """
    # Example: Fetch from IMAP, database, or file system
    sample_emails = [
        {
            "subject": "Quarterly Report Due",
            "body": "Please submit your quarterly reports by end of day Friday.",
            "source": "internal"
        },
        {
            "subject": "Special Offer - 50% Off",
            "body": "Limited time offer on all products. Shop now!",
            "source": "marketing"
        }
    ]

    # Store in XCom for next task
    context['ti'].xcom_push(key='new_emails', value=sample_emails)
    logging.info(f"Fetched {len(sample_emails)} new emails")
    return len(sample_emails)

def classify_emails(**context):
    """Classify fetched emails"""
    emails = context['ti'].xcom_pull(key='new_emails')
    classified_results = []

    for email in emails:
        try:
            response = requests.post(
                f"{API_URL}/emails/classify",
                json={
                    "subject": email['subject'],
                    "body": email['body'],
                    "use_email_similarity": False  # Configure based on needs
                }
            )

            if response.status_code == 200:
                result = response.json()
                classified_results.append({
                    "email": email,
                    "prediction": result['predicted_topic'],
                    "scores": result['topic_scores']
                })
                logging.info(f"Classified email as: {result['predicted_topic']}")
            else:
                logging.error(f"Classification failed: {response.text}")

        except Exception as e:
            logging.error(f"Error classifying email: {e}")

    context['ti'].xcom_push(key='classified_emails', value=classified_results)
    return len(classified_results)

def store_high_confidence_emails(**context):
    """Store emails with high confidence classifications as training data"""
    classified_emails = context['ti'].xcom_pull(key='classified_emails')
    stored_count = 0

    for item in classified_emails:
        email = item['email']
        prediction = item['prediction']
        scores = item['scores']

        # Only store if confidence is high (>0.8)
        max_score = max(scores.values())
        if max_score > 0.8:
            try:
                response = requests.post(
                    f"{API_URL}/emails",
                    json={
                        "subject": email['subject'],
                        "body": email['body'],
                        "ground_truth": prediction
                    }
                )

                if response.status_code == 200:
                    stored_count += 1
                    logging.info(f"Stored email with ground truth: {prediction}")

            except Exception as e:
                logging.error(f"Error storing email: {e}")

    logging.info(f"Stored {stored_count} high-confidence emails")
    return stored_count

def update_topic_statistics(**context):
    """Update statistics and potentially add new topics based on patterns"""
    classified_emails = context['ti'].xcom_pull(key='classified_emails')

    # Count classifications
    topic_counts = {}
    for item in classified_emails:
        prediction = item['prediction']
        topic_counts[prediction] = topic_counts.get(prediction, 0) + 1

    # Log statistics
    for topic, count in topic_counts.items():
        logging.info(f"Topic '{topic}': {count} emails")

    # Store statistics for monitoring
    context['ti'].xcom_push(key='topic_statistics', value=topic_counts)

    # Could add logic here to create new topics based on patterns
    return topic_counts

def generate_classification_report(**context):
    """Generate a summary report of the classification run"""
    stats = context['ti'].xcom_pull(key='topic_statistics')
    classified_count = context['ti'].xcom_pull(task_ids='classify_emails')
    stored_count = context['ti'].xcom_pull(task_ids='store_high_confidence')

    report = {
        "timestamp": datetime.now().isoformat(),
        "emails_processed": classified_count,
        "emails_stored": stored_count,
        "topic_distribution": stats
    }

    logging.info(f"Classification Report: {json.dumps(report, indent=2)}")

    # Could send this report via email, store in database, etc.
    return report

# Task definitions
health_check = PythonOperator(
    task_id='check_api_health',
    python_callable=check_api_health,
    dag=dag
)

fetch_emails = PythonOperator(
    task_id='fetch_new_emails',
    python_callable=fetch_new_emails,
    provide_context=True,
    dag=dag
)

classify = PythonOperator(
    task_id='classify_emails',
    python_callable=classify_emails,
    provide_context=True,
    dag=dag
)

store_emails = PythonOperator(
    task_id='store_high_confidence',
    python_callable=store_high_confidence_emails,
    provide_context=True,
    dag=dag
)

update_stats = PythonOperator(
    task_id='update_statistics',
    python_callable=update_topic_statistics,
    provide_context=True,
    dag=dag
)

generate_report = PythonOperator(
    task_id='generate_report',
    python_callable=generate_classification_report,
    provide_context=True,
    dag=dag
)

# Define task dependencies
health_check >> fetch_emails >> classify >> [store_emails, update_stats] >> generate_report

# Additional DAGs for maintenance tasks

maintenance_dag = DAG(
    'email_classification_maintenance',
    default_args=default_args,
    description='Maintenance tasks for email classification system',
    schedule_interval=timedelta(days=1),  # Daily
    catchup=False,
    tags=['email', 'maintenance']
)

def cleanup_old_emails():
    """Remove emails older than 30 days"""
    # Implementation depends on your storage strategy
    logging.info("Cleaning up old emails")
    pass

def retrain_model():
    """Trigger model retraining if needed"""
    # Check if enough new labeled data exists
    # Trigger retraining pipeline if threshold met
    logging.info("Checking if retraining needed")
    pass

def backup_data():
    """Backup email and topic data"""
    try:
        # Get current data
        topics_response = requests.get(f"{API_URL}/topics")
        emails_response = requests.get(f"{API_URL}/emails")

        if topics_response.status_code == 200 and emails_response.status_code == 200:
            backup = {
                "timestamp": datetime.now().isoformat(),
                "topics": topics_response.json(),
                "emails": emails_response.json()
            }

            # Save backup (implement your backup strategy)
            logging.info(f"Backup completed: {len(backup['emails']['emails'])} emails")
            return backup
        else:
            raise Exception("Failed to fetch data for backup")

    except Exception as e:
        logging.error(f"Backup failed: {e}")
        raise

cleanup_task = PythonOperator(
    task_id='cleanup_old_emails',
    python_callable=cleanup_old_emails,
    dag=maintenance_dag
)

retrain_task = PythonOperator(
    task_id='check_retrain',
    python_callable=retrain_model,
    dag=maintenance_dag
)

backup_task = PythonOperator(
    task_id='backup_data',
    python_callable=backup_data,
    dag=maintenance_dag
)

# Maintenance task dependencies
backup_task >> cleanup_task >> retrain_task