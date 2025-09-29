"""
Simplified MLOps Pipeline with Neo4j Integration
Works within Docker Airflow environment without external dependencies
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import json
import hashlib

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'mlops_neo4j_simple',
    default_args=default_args,
    description='Simple MLOps pipeline with Neo4j integration',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'neo4j', 'email-classification'],
)


def generate_sample_data(**context):
    """Generate sample email data"""
    emails = [
        {
            "id": "email_001",
            "subject": "Q4 Financial Report Ready",
            "body": "The quarterly financial report is ready for your review.",
            "label": "work",
            "features": {
                "word_count": 10,
                "has_urgent": False,
                "char_count": 56
            }
        },
        {
            "id": "email_002",
            "subject": "FREE iPhone - CLICK NOW!!!",
            "body": "You won a FREE iPhone! Click to claim your prize NOW!!!",
            "label": "spam",
            "features": {
                "word_count": 11,
                "has_urgent": True,
                "char_count": 57
            }
        },
        {
            "id": "email_003",
            "subject": "Meeting rescheduled",
            "body": "Team meeting moved to 3pm today in conference room B.",
            "label": "work",
            "features": {
                "word_count": 10,
                "has_urgent": False,
                "char_count": 55
            }
        }
    ]

    context['task_instance'].xcom_push(key='emails', value=emails)
    print(f"✅ Generated {len(emails)} sample emails")
    return len(emails)


def simple_classify(**context):
    """Simple rule-based classification"""
    emails = context['task_instance'].xcom_pull(task_ids='generate_data', key='emails')

    classified = []
    for email in emails:
        # Simple classification logic
        subject_lower = email['subject'].lower()
        body_lower = email['body'].lower()

        if 'free' in subject_lower or 'click now' in subject_lower:
            predicted = 'spam'
            confidence = 0.95
        elif 'meeting' in subject_lower or 'report' in subject_lower:
            predicted = 'work'
            confidence = 0.85
        else:
            predicted = 'personal'
            confidence = 0.60

        email['predicted_topic'] = predicted
        email['confidence'] = confidence
        classified.append(email)

        print(f"✅ Classified: {email['subject'][:40]} → {predicted} ({confidence:.0%})")

    context['task_instance'].xcom_push(key='classified', value=classified)
    return len(classified)


def connect_neo4j(**context):
    """Connect to Neo4j and store results"""
    import os

    # Check if Neo4j credentials are available
    neo4j_uri = os.getenv('NEO4J_URI')
    neo4j_user = os.getenv('NEO4J_USERNAME')
    neo4j_pass = os.getenv('NEO4J_PASSWORD')

    if not all([neo4j_uri, neo4j_user, neo4j_pass]):
        print("⚠️ Neo4j credentials not found in environment")
        print("   Required: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD")
        print("   Skipping Neo4j storage...")
        return {"stored": 0, "status": "skipped"}

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))

        classified = context['task_instance'].xcom_pull(task_ids='classify', key='classified')

        stored_count = 0
        with driver.session() as session:
            for email in classified:
                # Generate email ID
                content = f"{email['subject']}_{email['body']}"
                email_id = hashlib.md5(content.encode()).hexdigest()

                # Store email
                query = """
                MERGE (e:Email {id: $email_id})
                SET e.subject = $subject,
                    e.body = $body,
                    e.timestamp = datetime()
                WITH e
                MERGE (t:Topic {name: $predicted_topic})
                MERGE (e)-[r:CLASSIFIED_AS]->(t)
                SET r.confidence = $confidence,
                    r.timestamp = datetime()
                """

                session.run(
                    query,
                    email_id=email_id,
                    subject=email['subject'],
                    body=email['body'],
                    predicted_topic=email['predicted_topic'],
                    confidence=email['confidence']
                )

                # Store ground truth if available
                if 'label' in email:
                    gt_query = """
                    MATCH (e:Email {id: $email_id})
                    MERGE (t:Topic {name: $true_label})
                    MERGE (e)-[r:HAS_GROUND_TRUTH]->(t)
                    SET r.annotator = 'dataset',
                        r.timestamp = datetime()
                    """
                    session.run(gt_query, email_id=email_id, true_label=email['label'])

                stored_count += 1
                print(f"✅ Stored in Neo4j: {email_id[:12]}... - {email['subject'][:40]}")

        driver.close()
        print(f"\n✅ Successfully stored {stored_count} emails in Neo4j")
        return {"stored": stored_count, "status": "success"}

    except Exception as e:
        print(f"❌ Error storing in Neo4j: {e}")
        return {"stored": 0, "status": "error", "error": str(e)}


def generate_summary(**context):
    """Generate pipeline summary"""
    email_count = context['task_instance'].xcom_pull(task_ids='generate_data')
    classified_count = context['task_instance'].xcom_pull(task_ids='classify')
    neo4j_result = context['task_instance'].xcom_pull(task_ids='store_neo4j')

    print("\n" + "="*60)
    print("📊 MLOPS PIPELINE SUMMARY")
    print("="*60)
    print(f"Emails Generated: {email_count}")
    print(f"Emails Classified: {classified_count}")
    print(f"Neo4j Status: {neo4j_result.get('status', 'unknown')}")
    print(f"Emails Stored: {neo4j_result.get('stored', 0)}")
    print("="*60 + "\n")

    return {
        "pipeline_status": "completed",
        "emails_processed": email_count,
        "neo4j_stored": neo4j_result.get('stored', 0)
    }


# Define tasks
generate_data = PythonOperator(
    task_id='generate_data',
    python_callable=generate_sample_data,
    dag=dag,
)

classify = PythonOperator(
    task_id='classify',
    python_callable=simple_classify,
    dag=dag,
)

store_neo4j = PythonOperator(
    task_id='store_neo4j',
    python_callable=connect_neo4j,
    dag=dag,
)

summary = PythonOperator(
    task_id='summary',
    python_callable=generate_summary,
    dag=dag,
)

# Set dependencies
generate_data >> classify >> store_neo4j >> summary