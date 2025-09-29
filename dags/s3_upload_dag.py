from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
import boto3

BUCKET_NAME = "st-mlops-fall-2025"   # change me
UPLOAD_KEY = "./upload/sample_data.json"

def upload_file_to_s3():
    file_name = "./data/sample_data.json"

    s3 = boto3.client("s3")
    s3.upload_file(file_name, BUCKET_NAME, UPLOAD_KEY)

    print(f"✅ Uploaded {file_name} to s3://{BUCKET_NAME}/{UPLOAD_KEY}")

def download_file_from_s3():
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=BUCKET_NAME, Key=UPLOAD_KEY)

    content = response["Body"].read().decode("utf-8")
    print("✅ Downloaded object content:")
    print(content)

default_args = {"owner": "airflow"}

with DAG(
    dag_id="s3_upload_download_dag",
    start_date=days_ago(1),
    schedule_interval=None,  # run manually
    catchup=False,
    tags=["teaching", "s3"],
    default_args=default_args,
) as dag:

    upload_task = PythonOperator(
        task_id="upload_to_s3",
        python_callable=upload_file_to_s3,
    )
    
    download_task = PythonOperator(
        task_id="download_from_s3",
        python_callable=download_file_from_s3,
    )
    
    upload_task >> download_task  # ensure download happens after upload
