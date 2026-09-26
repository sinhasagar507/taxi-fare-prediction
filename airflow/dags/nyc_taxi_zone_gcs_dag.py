import os
import logging
import subprocess
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from google.cloud import storage

# === Configuration ===
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
BUCKET = os.environ.get("GCP_GCS_BUCKET")
PATH_TO_LOCAL_HOME = os.environ.get("AIRFLOW_HOME", "/opt/airflow/")
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
ZONE_LOOKUP_FILE = "taxi_zone_lookup.csv"


# === Task: Download the zone lookup CSV ===
def download_file(**context):
    local_path = os.path.join(PATH_TO_LOCAL_HOME, ZONE_LOOKUP_FILE)

    logging.info(f"📥 Downloading using curl -L: {ZONE_LOOKUP_URL}")
    try:
        subprocess.run(
            ["curl", "-L", "-o", local_path, ZONE_LOOKUP_URL],
            check=True
        )
        logging.info(f"✅ Downloaded to: {local_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ curl failed with error: {e}")
        raise


# === Task: Upload to GCS ===
def upload_to_gcs(bucket, **context):
    local_file = os.path.join(PATH_TO_LOCAL_HOME, ZONE_LOOKUP_FILE)
    object_name = f"nyc_taxi_data/taxi_lookup_data/{ZONE_LOOKUP_FILE}"

    # GCS Upload Workaround
    storage.blob._MAX_MULTIPART_SIZE = 5 * 1024 * 1024
    storage.blob._DEFAULT_CHUNKSIZE = 5 * 1024 * 1024

    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    blob = bucket_obj.blob(object_name)
    blob.upload_from_filename(local_file)

    logging.info(f"📤 Uploaded {ZONE_LOOKUP_FILE} to gs://{bucket}/{object_name}")


# === Task: Remove local file ===
def remove_local_file(**context):
    file_path = os.path.join(PATH_TO_LOCAL_HOME, ZONE_LOOKUP_FILE)

    if os.path.exists(file_path):
        os.remove(file_path)
        logging.info(f"🗑️ Deleted local file: {file_path}")
    else:
        logging.warning(f"⚠️ File not found: {file_path}")


# === DAG Definition ===
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}

# Zone lookup is static reference metadata — run once on demand, no backfill.
with DAG(
    dag_id="nyc_taxi_zone_ingestion_dag",
    default_args=default_args,
    start_date=datetime(2015, 1, 1),
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    tags=['nyc-taxi'],
) as dag:

    download_dataset_task = PythonOperator(
        task_id="download_dataset_task",
        python_callable=download_file,
        provide_context=True
    )

    local_to_gcs_task = PythonOperator(
        task_id="local_to_gcs_task",
        python_callable=upload_to_gcs,
        op_kwargs={"bucket": BUCKET},
        provide_context=True
    )

    cleanup_local_file_task = PythonOperator(
        task_id="cleanup_local_file_task",
        python_callable=remove_local_file,
        provide_context=True
    )

    trigger_external_table_task = TriggerDagRunOperator(
        task_id="trigger_external_table",
        trigger_dag_id="create_external_table_taxi_zone",
        wait_for_completion=False,
    )

    # Sequential flow: Download → Upload → Cleanup → Register external table
    download_dataset_task >> local_to_gcs_task >> cleanup_local_file_task >> trigger_external_table_task
