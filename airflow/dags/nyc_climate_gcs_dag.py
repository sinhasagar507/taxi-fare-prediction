import os
import pandas as pd
import logging
from datetime import datetime

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from google.cloud import storage
from airflow.providers.google.cloud.operators.bigquery import BigQueryCreateExternalTableOperator
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
BUCKET = os.environ.get("GCP_GCS_BUCKET")
CLIMATE_DATA_DIRECTORY = "nyc_climate" # folder path within the docker taxi data directory
PATH_TO_LOCAL_HOME = os.environ.get("AIRFLOW_HOME", "/opt/airflow/") # the root path within the DOCKER environment
CLIMATE_DATA_SOURCE_URL = "https://gist.githubusercontent.com/adrn/6455c48cb556d2f6f939c1e55b2308f8/raw/dedf6fd1989ab46baeab51058c788cca8aba4ca3/weather_cache_sm.csv"

# The CSV keeps the upstream gist's own filename — it is a local intermediate and
# never reaches GCS, so the name documents where the data came from.
CLIMATE_DATA_TARGET_CSV = "weather_cache_sm.csv"
# The parquet DOES reach GCS, so it carries the project's vocabulary, not the
# upstream one. README.md, notes/gcp-reference.md and
# tests/integration/test_gcs.py all name `climate_data.parquet`; only this DAG
# said `weather_cache_sm.parquet`, so test_climate_parquet_exists failed even
# after a successful ingest.
CLIMATE_DATA_TARGET_PARQUET = "climate_data.parquet"

CLIMATE_DATA_TARGET_CSV_DIRECTORY = os.path.join(PATH_TO_LOCAL_HOME, CLIMATE_DATA_DIRECTORY, CLIMATE_DATA_TARGET_CSV)
CLIMATE_DATA_TARGET_PARQUET_DIRECTORY = os.path.join(PATH_TO_LOCAL_HOME, CLIMATE_DATA_DIRECTORY, CLIMATE_DATA_TARGET_PARQUET)

CLIMATE_BIGQUERY_DATASET_ID = os.environ.get("BIGQUERY_DATASET", "nyc_climate_data")
CLIMATE_BIGQUERY_TABLE_ID = os.environ.get("CLIMATE_TABLE", "climate_table")


# DATASET_FILE = "yellow_tripdata_2015-01.parquet"
# DATASET_URL = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{DATASET_FILE}"
# PATH_TO_LOCAL_HOME = os.environ.get("AIRFLOW_HOME", "/opt/airflow/")
# Function to format the source file to parquet if not already in the format
def format_to_parquet(src_file):
    """
    Converts a CSV file to Parquet format, ensuring that the output file
    is created in the same directory as the source file.

    :param src_file: Path to the source CSV file.
    """
    try:
        if not src_file.endswith('.csv'):
            logging.error("Invalid file format. Only CSV files are supported for conversion to Parquet.")
            return

        logging.info(f"Loading CSV file: {src_file}")
        df = pd.read_csv(src_file)  # Load CSV file

        if df.empty:
            logging.warning("The CSV file is empty. An empty Parquet file will be created.")

        # Extract directory from source file
        output_file = CLIMATE_DATA_TARGET_PARQUET_DIRECTORY

        logging.info(f"Saving as Parquet in the same directory: {output_file}")

        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_file)

        logging.info(f"File successfully saved as Parquet: {output_file}")
    except Exception as e:
        logging.error(f"Error converting {src_file} to Parquet: {e}")

# NOTE: takes 20 mins, at an upload speed of 800kbps. Faster if your internet has a better upload speed
def upload_to_gcs(bucket, object_name, local_file):
    """
    Ref: https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python
    :param bucket: GCS bucket name
    :param object_name: target path & file-name
    :param local_file: source path & file-name
    :return:
    """
    # WORKAROUND to prevent timeout for files > 6 MB on 800 kbps upload speed.
    # (Ref: https://github.com/googleapis/python-storage/issues/74)
    storage.blob._MAX_MULTIPART_SIZE = 5 * 1024 * 1024  # 5 MB
    storage.blob._DEFAULT_CHUNKSIZE = 5 * 1024 * 1024  # 5 MB
    # End of Workaround

    client = storage.Client()
    bucket = client.bucket(bucket)

    blob = bucket.blob(object_name)
    blob.upload_from_filename(local_file)

def remove_local_files(file_paths):
    """
    Deletes multiple files from the local Airflow directory after upload.
    """
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Deleted local file: {file_path}")
        else:
            logging.warning(f"File not found: {file_path}")

default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
    "depends_on_past": False,
    "retries": 1,
}

# NOTE: DAG declaration - using a Context Manager (an implicit way)
with DAG(
    dag_id="nyc_climate_data_ingestion_dag",
    default_args=default_args,
    schedule_interval='0 6 1 * *',   # 6 AM on the 1st of every month
    catchup=False,                   # don't process historical data
    max_active_runs=1,               # Limit parallel runs
    tags=['nyc-taxi'],
) as dag:
    
    download_dataset_task = BashOperator(
        task_id="download_dataset_task",
        bash_command=f"mkdir -p {PATH_TO_LOCAL_HOME}/{CLIMATE_DATA_DIRECTORY} && curl -sSL {CLIMATE_DATA_SOURCE_URL} > {PATH_TO_LOCAL_HOME}/{CLIMATE_DATA_DIRECTORY}/{CLIMATE_DATA_TARGET_CSV}"
   )
    
    convert_to_parquet_task = PythonOperator(
        task_id="convert_to_parquet", 
        python_callable=format_to_parquet, 
        op_kwargs={
            "src_file": CLIMATE_DATA_TARGET_CSV_DIRECTORY,
        }
    )

    local_to_gcs_task = PythonOperator(
        task_id="local_to_gcs_task",
        python_callable=upload_to_gcs,
        op_kwargs={
            "bucket": BUCKET,
            "object_name": f"nyc_climate_data/{CLIMATE_DATA_TARGET_PARQUET}",
            "local_file": f"{PATH_TO_LOCAL_HOME}/{CLIMATE_DATA_DIRECTORY}/{CLIMATE_DATA_TARGET_PARQUET}",
        },
    )

    # bigquery_external_table_task = BigQueryCreateExternalTableOperator(
    #     task_id="bigquery_external_table_task",
    #     table_resource={
    #         "tableReference": {
    #             "projectId": PROJECT_ID,
    #             "datasetId": CLIMATE_BIGQUERY_DATASET_ID,
    #             "tableId": CLIMATE_BIGQUERY_TABLE_ID,
    #         },
    #         "externalDataConfiguration": {
    #             "sourceFormat": "PARQUET",
    #             "sourceUris": [f"gs://{BUCKET}/raw/nyc_climate_data/{CLIMATE_DATA_TARGET_PARQUET}"],
    #         },
    #     },
    # )
    
    cleanup_local_file_task = PythonOperator(
    task_id="cleanup_local_file_task",
    python_callable=remove_local_files,
    op_kwargs={
        "file_paths": [
            f"{PATH_TO_LOCAL_HOME}/{CLIMATE_DATA_DIRECTORY}/{CLIMATE_DATA_TARGET_CSV}",
            f"{PATH_TO_LOCAL_HOME}/{CLIMATE_DATA_DIRECTORY}/{CLIMATE_DATA_TARGET_PARQUET}",
        ]
    },
  )

    trigger_external_table_task = TriggerDagRunOperator(
        task_id="trigger_external_table",
        trigger_dag_id="create_external_table_climate_data",
        wait_for_completion=False,
    )

    download_dataset_task >> convert_to_parquet_task >> local_to_gcs_task >> cleanup_local_file_task >> trigger_external_table_task