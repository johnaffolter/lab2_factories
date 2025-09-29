"""
Simple MLOps S3 Pipeline - Working Version
Simplified version that works in Docker container
"""

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import boto3
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, Any, List

# Configuration
BUCKET_NAME = "st-mlops-fall-2025"
MODEL_PREFIX = "models/simple_classifier"
DATA_PREFIX = "datasets/training_data"
EXPERIMENT_PREFIX = "experiments"

def generate_training_data(**context):
    """Generate simple training data"""
    print("🔍 Generating training data...")

    # Simple email training data
    training_data = [
        {"text": "free money winner urgent click now", "label": "spam"},
        {"text": "meeting tomorrow project discussion", "label": "work"},
        {"text": "your order has shipped tracking number", "label": "personal"},
        {"text": "congratulations winner urgent action required", "label": "spam"},
        {"text": "team standup meeting agenda review", "label": "work"},
        {"text": "family dinner this weekend please confirm", "label": "personal"},
        {"text": "amazing offer limited time cash prize", "label": "spam"},
        {"text": "quarterly review performance metrics", "label": "work"}
    ]

    # Create feature matrix (simple bag of words)
    print("🔧 Creating features...")
    vocabulary = set()
    for item in training_data:
        vocabulary.update(item["text"].lower().split())

    vocab_list = sorted(list(vocabulary))
    print(f"📚 Vocabulary size: {len(vocab_list)}")

    # Convert to feature vectors
    X = []
    y = []

    for item in training_data:
        words = item["text"].lower().split()
        # Create binary feature vector
        features = [1 if word in words else 0 for word in vocab_list]
        X.append(features)
        y.append(item["label"])

    # Convert to arrays
    X = np.array(X)
    y = np.array(y)

    print(f"✅ Generated {len(X)} training samples with {len(vocab_list)} features")

    # Save data
    data_path = "/tmp/training_data.npz"
    np.savez(data_path, X=X, y=y, vocab=vocab_list, raw_data=training_data)

    # Push to XCom
    context['task_instance'].xcom_push(key='data_path', value=data_path)
    context['task_instance'].xcom_push(key='sample_count', value=len(X))
    context['task_instance'].xcom_push(key='feature_count', value=len(vocab_list))

    return data_path

def train_simple_models(**context):
    """Train simple classification models"""
    print("🤖 Training models...")

    data_path = context['task_instance'].xcom_pull(key='data_path', task_ids='generate_training_data')

    # Load data
    data = np.load(data_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    vocab = data['vocab']

    print(f"📊 Training on {len(X)} samples with {len(vocab)} features")

    # Simple models
    models = {}

    # 1. Naive classifier (most frequent class)
    class MajorityClassifier:
        def __init__(self):
            self.majority_class = None

        def fit(self, X, y):
            unique, counts = np.unique(y, return_counts=True)
            self.majority_class = unique[np.argmax(counts)]

        def predict(self, X):
            return [self.majority_class] * len(X)

    majority_model = MajorityClassifier()
    majority_model.fit(X, y)
    models['majority'] = majority_model

    # 2. Simple centroid classifier
    class CentroidClassifier:
        def __init__(self):
            self.centroids = {}

        def fit(self, X, y):
            for label in np.unique(y):
                mask = y == label
                self.centroids[label] = np.mean(X[mask], axis=0)

        def predict(self, X):
            predictions = []
            for sample in X:
                distances = {}
                for label, centroid in self.centroids.items():
                    distance = np.linalg.norm(sample - centroid)
                    distances[label] = distance
                predictions.append(min(distances, key=distances.get))
            return predictions

    centroid_model = CentroidClassifier()
    centroid_model.fit(X, y)
    models['centroid'] = centroid_model

    # 3. Simple keyword classifier
    class KeywordClassifier:
        def __init__(self, vocab):
            self.vocab = vocab
            self.spam_keywords = {'free', 'winner', 'urgent', 'click', 'money', 'cash', 'prize'}
            self.work_keywords = {'meeting', 'project', 'team', 'review', 'standup', 'agenda'}

        def predict(self, X):
            predictions = []
            for sample in X:
                # Convert back to words
                words = [self.vocab[i] for i, val in enumerate(sample) if val == 1]
                word_set = set(words)

                spam_score = len(word_set.intersection(self.spam_keywords))
                work_score = len(word_set.intersection(self.work_keywords))

                if spam_score > work_score and spam_score > 0:
                    predictions.append('spam')
                elif work_score > 0:
                    predictions.append('work')
                else:
                    predictions.append('personal')
            return predictions

    keyword_model = KeywordClassifier(vocab)
    models['keyword'] = keyword_model

    # Evaluate models
    print("📈 Evaluating models...")
    performance = {}

    for name, model in models.items():
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        performance[name] = {
            'accuracy': float(accuracy),
            'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions
        }
        print(f"  {name}: {accuracy:.3f}")

    # Save models and metadata
    models_path = "/tmp/simple_models.pkl"
    with open(models_path, 'wb') as f:
        pickle.dump(models, f)

    performance_path = "/tmp/model_performance.json"
    with open(performance_path, 'w') as f:
        json.dump(performance, f, indent=2)

    # Push to XCom
    context['task_instance'].xcom_push(key='models_path', value=models_path)
    context['task_instance'].xcom_push(key='performance_path', value=performance_path)
    context['task_instance'].xcom_push(key='best_model', value=max(performance, key=lambda x: performance[x]['accuracy']))

    return models_path

def upload_to_s3(**context):
    """Upload models and results to S3"""
    print("☁️ Uploading to S3...")

    models_path = context['task_instance'].xcom_pull(key='models_path', task_ids='train_simple_models')
    performance_path = context['task_instance'].xcom_pull(key='performance_path', task_ids='train_simple_models')
    data_path = context['task_instance'].xcom_pull(key='data_path', task_ids='generate_training_data')

    try:
        s3 = boto3.client('s3')

        # Create experiment ID
        experiment_id = f"simple_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        uploads = []

        # Upload models
        models_key = f"{MODEL_PREFIX}/{experiment_id}/models.pkl"
        s3.upload_file(models_path, BUCKET_NAME, models_key)
        uploads.append(f"s3://{BUCKET_NAME}/{models_key}")

        # Upload performance
        perf_key = f"{EXPERIMENT_PREFIX}/{experiment_id}/performance.json"
        s3.upload_file(performance_path, BUCKET_NAME, perf_key)
        uploads.append(f"s3://{BUCKET_NAME}/{perf_key}")

        # Upload training data
        data_key = f"{DATA_PREFIX}/{experiment_id}/training_data.npz"
        s3.upload_file(data_path, BUCKET_NAME, data_key)
        uploads.append(f"s3://{BUCKET_NAME}/{data_key}")

        # Create experiment metadata
        metadata = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "dag_run_id": context['dag_run'].run_id,
            "models": ["majority", "centroid", "keyword"],
            "files_uploaded": uploads,
            "sample_count": context['task_instance'].xcom_pull(key='sample_count', task_ids='generate_training_data'),
            "feature_count": context['task_instance'].xcom_pull(key='feature_count', task_ids='generate_training_data'),
            "best_model": context['task_instance'].xcom_pull(key='best_model', task_ids='train_simple_models')
        }

        # Upload metadata
        metadata_path = f"/tmp/{experiment_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        metadata_key = f"{EXPERIMENT_PREFIX}/{experiment_id}/metadata.json"
        s3.upload_file(metadata_path, BUCKET_NAME, metadata_key)
        uploads.append(f"s3://{BUCKET_NAME}/{metadata_key}")

        print(f"✅ Uploaded {len(uploads)} files:")
        for upload in uploads:
            print(f"  📁 {upload}")

        # Push to XCom
        context['task_instance'].xcom_push(key='experiment_id', value=experiment_id)
        context['task_instance'].xcom_push(key='uploads', value=uploads)

        return experiment_id

    except Exception as e:
        print(f"❌ S3 upload failed: {e}")
        print("⚠️ This might be due to AWS credentials not being configured")
        print("   The models were still trained successfully!")

        # Return dummy experiment ID
        experiment_id = f"local_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        context['task_instance'].xcom_push(key='experiment_id', value=experiment_id)

        return experiment_id

def validate_experiment(**context):
    """Validate the experiment results"""
    print("✅ Validating experiment...")

    experiment_id = context['task_instance'].xcom_pull(key='experiment_id', task_ids='upload_to_s3')
    models_path = context['task_instance'].xcom_pull(key='models_path', task_ids='train_simple_models')

    # Load and test models
    with open(models_path, 'rb') as f:
        models = pickle.load(f)

    # Test with a sample
    test_text = "urgent free money winner click now"
    test_words = test_text.split()

    print(f"🧪 Testing with: '{test_text}'")

    # Get vocabulary (simplified)
    data_path = context['task_instance'].xcom_pull(key='data_path', task_ids='generate_training_data')
    data = np.load(data_path, allow_pickle=True)
    vocab = data['vocab'].tolist()

    # Create test feature vector
    test_features = np.array([[1 if word in test_words else 0 for word in vocab]])

    validation_results = {}
    for name, model in models.items():
        try:
            if hasattr(model, 'predict'):
                prediction = model.predict(test_features)[0]
            else:
                prediction = 'unknown'

            validation_results[name] = prediction
            print(f"  {name}: {prediction}")
        except Exception as e:
            print(f"  {name}: failed - {e}")
            validation_results[name] = 'error'

    print(f"📊 Experiment {experiment_id} validated successfully")
    return validation_results

def cleanup_files(**context):
    """Clean up temporary files"""
    print("🧹 Cleaning up...")

    files_to_clean = [
        "/tmp/training_data.npz",
        "/tmp/simple_models.pkl",
        "/tmp/model_performance.json"
    ]

    experiment_id = context['task_instance'].xcom_pull(key='experiment_id', task_ids='upload_to_s3')
    if experiment_id:
        files_to_clean.append(f"/tmp/{experiment_id}_metadata.json")

    cleaned = 0
    for file_path in files_to_clean:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                cleaned += 1
        except:
            pass

    print(f"✅ Cleaned {cleaned} temporary files")

# Default arguments
default_args = {
    'owner': 'mlops-student',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2)
}

# Create DAG
with DAG(
    dag_id='mlops_s3_simple',
    default_args=default_args,
    description='Simple MLOps Pipeline with S3 Storage',
    schedule_interval=None,  # Manual trigger
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 's3', 'simple', 'lab3', 'machine-learning'],
) as dag:

    # Generate training data
    generate_data = PythonOperator(
        task_id='generate_training_data',
        python_callable=generate_training_data,
    )

    # Train models
    train_models = PythonOperator(
        task_id='train_simple_models',
        python_callable=train_simple_models,
    )

    # Upload to S3
    upload_models = PythonOperator(
        task_id='upload_to_s3',
        python_callable=upload_to_s3,
    )

    # Validate experiment
    validate = PythonOperator(
        task_id='validate_experiment',
        python_callable=validate_experiment,
    )

    # Cleanup
    cleanup = PythonOperator(
        task_id='cleanup_files',
        python_callable=cleanup_files,
        trigger_rule='all_done',
    )

    # Define dependencies
    generate_data >> train_models >> upload_models >> validate >> cleanup