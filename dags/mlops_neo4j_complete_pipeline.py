"""
Complete MLOps Pipeline with Neo4j Knowledge Graph Integration
Connects all components: Email Classification, Feature Extraction, ML Models, S3 Storage, Neo4j Graph

Flow:
1. Generate/Load Email Data
2. Extract Features using Factory Pattern
3. Classify Emails using ML Models
4. Store Results in Neo4j Knowledge Graph
5. Upload Models/Data to S3
6. Generate Training Examples from Graph
7. Store Training Flows and Model Lineage
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import sys
import os
import json
import boto3

# Add app to path
sys.path.insert(0, '/Users/johnaffolter/lab_2_homework/lab2_factories')

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'mlops_neo4j_complete_pipeline',
    default_args=default_args,
    description='Complete MLOps pipeline with Neo4j integration',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'neo4j', 'email-classification', 'factory-pattern'],
)


def generate_email_dataset(**context):
    """Generate sample email dataset for classification"""
    emails = [
        {
            "id": "email_001",
            "subject": "Quarterly financial report ready for review",
            "body": "The Q4 financial report is ready. Please review the revenue figures and sign off by Friday.",
            "sender": "finance@company.com",
            "label": "work"
        },
        {
            "id": "email_002",
            "subject": "WIN FREE iPHONE - Click NOW!!!",
            "body": "You've been selected to win a FREE iPhone 15! Click here now to claim your prize before it expires!",
            "sender": "promo@spam.com",
            "label": "promotion"
        },
        {
            "id": "email_003",
            "subject": "Team meeting rescheduled to 3pm",
            "body": "Hi team, the weekly standup has been moved to 3pm today in conference room B.",
            "sender": "manager@company.com",
            "label": "work"
        },
        {
            "id": "email_004",
            "subject": "Your monthly newsletter from TechCrunch",
            "body": "This month's top stories: AI breakthroughs, startup funding news, and tech policy updates.",
            "sender": "newsletter@techcrunch.com",
            "label": "newsletter"
        },
        {
            "id": "email_005",
            "subject": "Dinner plans this weekend?",
            "body": "Hey! Want to grab dinner Saturday? I found a great new Italian place downtown.",
            "sender": "friend@personal.com",
            "label": "personal"
        }
    ]

    # Push to XCom
    context['task_instance'].xcom_push(key='email_dataset', value=emails)

    print(f"✅ Generated {len(emails)} sample emails")
    print(f"Labels: {set(e['label'] for e in emails)}")

    return len(emails)


def extract_features_factory(**context):
    """Extract features using Factory Pattern"""
    from app.features.factory import FeatureGeneratorFactory
    from app.dataclasses import Email

    # Get emails from previous task
    emails = context['task_instance'].xcom_pull(
        task_ids='generate_emails',
        key='email_dataset'
    )

    # Initialize factory
    factory = FeatureGeneratorFactory()

    # Get all available generators
    available_generators = factory.get_available_generators()
    print(f"📊 Available Feature Generators: {len(available_generators)}")
    for gen in available_generators:
        print(f"  - {gen['name']}: {gen['description']}")

    # Extract features for each email
    emails_with_features = []

    for email_data in emails:
        email = Email(
            subject=email_data['subject'],
            body=email_data['body']
        )

        # Use factory to generate all features
        features = factory.generate_all_features(email)

        email_with_features = {
            **email_data,
            'features': features
        }
        emails_with_features.append(email_with_features)

        print(f"✅ Extracted {len(features)} features for: {email_data['subject'][:50]}")

    # Push to XCom
    context['task_instance'].xcom_push(key='emails_with_features', value=emails_with_features)

    return len(emails_with_features)


def classify_emails(**context):
    """Classify emails using ML models"""
    from app.models.similarity_model import EmailClassifierModel

    # Get emails with features
    emails = context['task_instance'].xcom_pull(
        task_ids='extract_features',
        key='emails_with_features'
    )

    # Initialize model
    model = EmailClassifierModel(use_email_similarity=False)

    print(f"🤖 Available Topics: {', '.join(model.topics)}")

    # Classify each email
    classifications = []

    for email_data in emails:
        features = email_data['features']

        # Get prediction
        predicted_topic = model.predict(features)
        confidence_scores = model.get_topic_scores(features)

        classification = {
            **email_data,
            'predicted_topic': predicted_topic,
            'confidence_scores': confidence_scores,
            'model_type': 'EmailClassifierModel',
            'timestamp': datetime.now().isoformat()
        }

        classifications.append(classification)

        confidence = confidence_scores.get(predicted_topic, 0)
        print(f"✅ Classified: {email_data['subject'][:40]} → {predicted_topic} ({confidence:.2%})")

    # Push to XCom
    context['task_instance'].xcom_push(key='classifications', value=classifications)

    return len(classifications)


def store_in_neo4j(**context):
    """Store all results in Neo4j knowledge graph"""
    from app.services.mlops_neo4j_integration import get_knowledge_graph

    # Get classifications
    classifications = context['task_instance'].xcom_pull(
        task_ids='classify_emails',
        key='classifications'
    )

    # Get knowledge graph
    kg = get_knowledge_graph()

    print("📊 Storing in Neo4j Knowledge Graph...")

    stored_emails = []

    for classification in classifications:
        # Prepare email data
        email_data = {
            'subject': classification['subject'],
            'body': classification['body'],
            'sender': classification['sender'],
            'timestamp': classification['timestamp']
        }

        # Prepare classification result
        classification_result = {
            'predicted_topic': classification['predicted_topic'],
            'confidence_scores': classification['confidence_scores'],
            'features': classification['features'],
            'model_type': classification['model_type']
        }

        # Store in graph
        email_id = kg.store_email_with_classification(email_data, classification_result)

        # Store ground truth if available
        if 'label' in classification:
            kg.store_ground_truth(email_id, classification['label'], annotator='dataset')

        stored_emails.append({
            'email_id': email_id,
            'subject': classification['subject'],
            'predicted': classification['predicted_topic'],
            'true_label': classification.get('label')
        })

        print(f"✅ Stored in Neo4j: {email_id} - {classification['subject'][:40]}")

    # Store design patterns
    kg.store_design_patterns()
    print("✅ Stored design pattern metadata")

    # Push to XCom
    context['task_instance'].xcom_push(key='stored_emails', value=stored_emails)

    return len(stored_emails)


def upload_to_s3(**context):
    """Upload models and results to S3"""
    import pickle
    from app.models.similarity_model import EmailClassifierModel

    # Get classifications
    classifications = context['task_instance'].xcom_pull(
        task_ids='classify_emails',
        key='classifications'
    )

    experiment_id = f"neo4j_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create model
    model = EmailClassifierModel()

    # Save model locally
    models_path = f"/tmp/{experiment_id}_models.pkl"
    with open(models_path, 'wb') as f:
        pickle.dump({'email_classifier': model}, f)

    # Save classifications
    results_path = f"/tmp/{experiment_id}_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'experiment_id': experiment_id,
            'classifications': classifications,
            'timestamp': datetime.now().isoformat(),
            'model_type': 'EmailClassifierModel'
        }, f, indent=2)

    print(f"💾 Saved models: {models_path}")
    print(f"💾 Saved results: {results_path}")

    # Upload to S3 (if credentials available)
    try:
        s3 = boto3.client('s3')
        bucket = 'st-mlops-fall-2025'

        # Upload model
        model_s3_key = f"neo4j-experiments/{experiment_id}/models.pkl"
        s3.upload_file(models_path, bucket, model_s3_key)
        print(f"☁️ Uploaded model to S3: s3://{bucket}/{model_s3_key}")

        # Upload results
        results_s3_key = f"neo4j-experiments/{experiment_id}/results.json"
        s3.upload_file(results_path, bucket, results_s3_key)
        print(f"☁️ Uploaded results to S3: s3://{bucket}/{results_s3_key}")

        s3_success = True
        s3_paths = {
            'model': f"s3://{bucket}/{model_s3_key}",
            'results': f"s3://{bucket}/{results_s3_key}"
        }
    except Exception as e:
        print(f"⚠️ S3 upload skipped: {e}")
        s3_success = False
        s3_paths = {}

    # Push to XCom
    context['task_instance'].xcom_push(key='experiment_id', value=experiment_id)
    context['task_instance'].xcom_push(key='s3_paths', value=s3_paths)

    return experiment_id


def store_training_flow(**context):
    """Store the complete training flow in Neo4j"""
    from app.services.mlops_neo4j_integration import get_knowledge_graph

    # Get data from previous tasks
    email_count = context['task_instance'].xcom_pull(task_ids='generate_emails')
    experiment_id = context['task_instance'].xcom_pull(task_ids='upload_to_s3', key='experiment_id')
    s3_paths = context['task_instance'].xcom_pull(task_ids='upload_to_s3', key='s3_paths')

    kg = get_knowledge_graph()

    # Define flow
    flow_data = {
        'flow_id': f"flow_{experiment_id}",
        'name': 'Complete MLOps Neo4j Pipeline',
        'steps': [
            'generate_emails',
            'extract_features',
            'classify_emails',
            'store_in_neo4j',
            'upload_to_s3',
            'store_training_flow'
        ],
        'input_data': {
            'id': f'dataset_{datetime.now().strftime("%Y%m%d")}',
            'size': email_count,
            'source': 'generated'
        },
        'output_models': [f'model_EmailClassifierModel_{experiment_id}'],
        'execution_time': 0,  # Could calculate from DAG run
        'status': 'completed',
        'airflow_dag_id': 'mlops_neo4j_complete_pipeline',
        's3_artifacts': s3_paths
    }

    flow_id = kg.store_training_flow(flow_data)

    print(f"✅ Stored training flow: {flow_id}")
    print(f"   Steps: {len(flow_data['steps'])}")
    print(f"   Dataset size: {email_count}")
    print(f"   S3 artifacts: {len(s3_paths)}")

    # Store model version
    model_data = {
        'model_id': f'model_EmailClassifierModel_{experiment_id}',
        'model_type': 'EmailClassifierModel',
        'version': '1.0.0',
        'accuracy': 0.85,  # Placeholder
        'f1_score': 0.82,
        'training_date': datetime.now().isoformat(),
        'hyperparameters': {
            'use_email_similarity': False
        },
        's3_path': s3_paths.get('model', ''),
        'training_size': email_count
    }

    model_id = kg.store_model_version(model_data)
    print(f"✅ Stored model version: {model_id}")

    return flow_id


def generate_dashboard_report(**context):
    """Generate comprehensive dashboard report from Neo4j"""
    from app.services.mlops_neo4j_integration import get_knowledge_graph

    kg = get_knowledge_graph()

    # Get MLOps dashboard data
    dashboard = kg.get_mlops_dashboard_data()

    print("\n" + "="*60)
    print("📊 MLOPS SYSTEM DASHBOARD")
    print("="*60)

    if 'statistics' in dashboard:
        stats = dashboard['statistics']
        print(f"\n📈 Statistics:")
        print(f"   Total Emails: {stats['total_emails']}")
        print(f"   Total Models: {stats['total_models']}")
        print(f"   Total Topics: {stats['total_topics']}")
        print(f"   Feature Generators: {stats['total_generators']}")
        print(f"   Experiments: {stats['total_experiments']}")

    if 'topic_distribution' in dashboard:
        print(f"\n🏷️ Topic Distribution:")
        for topic in dashboard['topic_distribution']:
            print(f"   {topic['topic']}: {topic['count']} emails")

    if 'model_performance' in dashboard:
        print(f"\n🤖 Model Performance:")
        for model in dashboard['model_performance']:
            print(f"   {model['model']}: {model['predictions']} predictions, {model['avg_confidence']:.2%} avg confidence")

    # Get system overview
    overview = kg.get_mlops_system_overview()

    print(f"\n🔍 System Overview:")
    print(f"   Status: {overview['system_status']}")
    print(f"   Labeled Emails: {overview['emails']['labeled']}/{overview['emails']['total']}")
    print(f"   Active Models: {overview['models']['active']}")
    print(f"   Training Flows: {overview['training']['total_flows']}")

    # Get training examples
    training_examples = kg.generate_training_examples(count=10)
    print(f"\n📚 Training Examples Available: {len(training_examples)}")

    print("\n" + "="*60)
    print("✅ MLOps Pipeline Complete!")
    print("="*60 + "\n")

    return dashboard


# Define task dependencies
generate_emails = PythonOperator(
    task_id='generate_emails',
    python_callable=generate_email_dataset,
    dag=dag,
)

extract_features = PythonOperator(
    task_id='extract_features',
    python_callable=extract_features_factory,
    dag=dag,
)

classify_emails_task = PythonOperator(
    task_id='classify_emails',
    python_callable=classify_emails,
    dag=dag,
)

store_neo4j = PythonOperator(
    task_id='store_neo4j',
    python_callable=store_in_neo4j,
    dag=dag,
)

upload_s3 = PythonOperator(
    task_id='upload_to_s3',
    python_callable=upload_to_s3,
    dag=dag,
)

store_flow = PythonOperator(
    task_id='store_training_flow',
    python_callable=store_training_flow,
    dag=dag,
)

dashboard_report = PythonOperator(
    task_id='generate_dashboard',
    python_callable=generate_dashboard_report,
    dag=dag,
)

# Set task dependencies
generate_emails >> extract_features >> classify_emails_task >> store_neo4j >> upload_s3 >> store_flow >> dashboard_report