"""
Lab 3 - S3 Upload and Download DAG
MLOps Course - St. Thomas University
"""

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import boto3
import json
import os
from datetime import datetime

# Configuration
BUCKET_NAME = "st-mlops-fall-2025"  # Update with your bucket
UPLOAD_KEY = "lab3/uploads/sample_data.json"
LAB3_DIR = "/Users/johnaffolter/lab3_airflow_s3"

def check_aws_credentials(**context):
    """Check if AWS credentials are configured"""
    try:
        # Try to create S3 client
        s3 = boto3.client('s3')

        # List buckets to verify credentials work
        response = s3.list_buckets()
        buckets = [b['Name'] for b in response.get('Buckets', [])]

        print(f"✅ AWS credentials are configured")
        print(f"📦 Found {len(buckets)} buckets")

        if buckets:
            print("Available buckets:")
            for bucket in buckets[:5]:  # Show first 5
                print(f"  - {bucket}")

        # Push bucket info to XCom
        context['task_instance'].xcom_push(key='bucket_count', value=len(buckets))
        context['task_instance'].xcom_push(key='buckets', value=buckets)

        return True

    except Exception as e:
        print(f"❌ AWS credentials not configured: {e}")
        print("\n📝 To configure AWS:")
        print("1. Set environment variables:")
        print("   export AWS_ACCESS_KEY_ID=your_key")
        print("   export AWS_SECRET_ACCESS_KEY=your_secret")
        print("2. Or configure in Airflow UI:")
        print("   Admin > Connections > aws_default")
        return False

def upload_file_to_s3(**context):
    """Upload sample data file to S3"""
    file_path = f"{LAB3_DIR}/data/sample_data.json"

    try:
        # Read the file
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Add upload metadata
        data['upload_metadata'] = {
            'uploaded_at': datetime.now().isoformat(),
            'dag_run_id': context['dag_run'].run_id,
            'task_id': context['task'].task_id
        }

        # Create S3 client
        s3 = boto3.client('s3')

        # Upload file
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=UPLOAD_KEY,
            Body=json.dumps(data, indent=2),
            ContentType='application/json'
        )

        print(f"✅ Uploaded file to s3://{BUCKET_NAME}/{UPLOAD_KEY}")

        # Push S3 location to XCom
        s3_url = f"s3://{BUCKET_NAME}/{UPLOAD_KEY}"
        context['task_instance'].xcom_push(key='s3_location', value=s3_url)

        return s3_url

    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        raise
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        raise

def download_file_from_s3(**context):
    """Download file from S3 and display contents"""
    try:
        s3 = boto3.client('s3')

        # Get object from S3
        response = s3.get_object(Bucket=BUCKET_NAME, Key=UPLOAD_KEY)
        content = response['Body'].read().decode('utf-8')

        # Parse JSON
        data = json.loads(content)

        print("✅ Downloaded file from S3")
        print("📄 File contents:")
        print(json.dumps(data, indent=2))

        # Extract metrics
        if 'data' in data and 'metrics' in data['data']:
            metrics = data['data']['metrics']
            print("\n📊 Model Metrics:")
            for key, value in metrics.items():
                print(f"  - {key}: {value}")

        # Save to local file
        download_path = f"{LAB3_DIR}/data/downloaded_data.json"
        with open(download_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n💾 Saved to: {download_path}")

        return data

    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise

def list_s3_objects(**context):
    """List objects in the S3 bucket"""
    try:
        s3 = boto3.client('s3')

        # List objects
        response = s3.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix='lab3/',
            MaxKeys=10
        )

        objects = response.get('Contents', [])

        print(f"📦 Objects in s3://{BUCKET_NAME}/lab3/")
        if objects:
            for obj in objects:
                size = obj['Size']
                modified = obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')
                print(f"  - {obj['Key']} ({size} bytes, modified: {modified})")
        else:
            print("  No objects found")

        return len(objects)

    except Exception as e:
        print(f"❌ List failed: {e}")
        return 0

def cleanup_s3(**context):
    """Optional: Clean up test files from S3"""
    try:
        s3 = boto3.client('s3')

        # Delete the uploaded file
        s3.delete_object(Bucket=BUCKET_NAME, Key=UPLOAD_KEY)

        print(f"🧹 Cleaned up s3://{BUCKET_NAME}/{UPLOAD_KEY}")

    except Exception as e:
        print(f"⚠️ Cleanup failed (non-critical): {e}")

# Default arguments
default_args = {
    'owner': 'mlops-student',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1
}

# Create DAG
with DAG(
    dag_id='lab3_s3_operations',
    default_args=default_args,
    description='Lab 3 - S3 Upload and Download Operations',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['lab3', 's3', 'teaching', 'mlops'],
) as dag:

    # Task 1: Check AWS credentials
    check_credentials = PythonOperator(
        task_id='check_aws_credentials',
        python_callable=check_aws_credentials,
    )

    # Task 2: Create sample data (if needed)
    create_data = BashOperator(
        task_id='ensure_sample_data',
        bash_command=f"""
        if [ ! -f {LAB3_DIR}/data/sample_data.json ]; then
            echo "Creating sample data..."
            mkdir -p {LAB3_DIR}/data
            echo '{{
                "test": "data",
                "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            }}' > {LAB3_DIR}/data/sample_data.json
        fi
        echo "Sample data ready"
        """,
    )

    # Task 3: Upload to S3
    upload_task = PythonOperator(
        task_id='upload_to_s3',
        python_callable=upload_file_to_s3,
    )

    # Task 4: List S3 objects
    list_task = PythonOperator(
        task_id='list_s3_objects',
        python_callable=list_s3_objects,
    )

    # Task 5: Download from S3
    download_task = PythonOperator(
        task_id='download_from_s3',
        python_callable=download_file_from_s3,
    )

    # Task 6: Optional cleanup
    cleanup_task = PythonOperator(
        task_id='cleanup_s3',
        python_callable=cleanup_s3,
        trigger_rule='none_failed',  # Run even if some tasks fail
    )

    # Define task dependencies
    check_credentials >> create_data >> upload_task >> list_task >> download_task >> cleanup_task
