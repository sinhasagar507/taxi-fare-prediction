"""
Test-Driven Development tests for NYC Taxi Yellow Data Ingestion DAG.

This module tests the yellow taxi data ingestion pipeline following TDD principles:
1. Write tests first (defining expected behavior)
2. Run tests (they should fail initially)
3. Write minimal code to pass tests
4. Refactor while keeping tests green
"""

import pytest
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, call
from airflow.models import DagBag
from airflow.utils.state import State


class TestNYCTaxiYellowDAG:
    """Test suite for NYC Taxi Yellow Data Ingestion DAG."""
    
    DAG_ID = "nyc_taxi_data_ingestion_dag"
    
    def test_dag_loaded(self, dagbag):
        """Test that the DAG is loaded without errors."""
        # GIVEN: A DAG bag with all DAGs loaded
        # WHEN: We check if our DAG exists
        dag = dagbag.get_dag(self.DAG_ID)
        
        # THEN: The DAG should be loaded successfully
        assert dag is not None
        assert dag.dag_id == self.DAG_ID
        assert len(dagbag.import_errors) == 0, f"DAG import errors: {dagbag.import_errors}"
    
    def test_dag_has_correct_structure(self, dagbag):
        """Test that the DAG has the expected tasks and dependencies."""
        # GIVEN: A loaded DAG
        dag = dagbag.get_dag(self.DAG_ID)
        
        # WHEN: We examine the DAG structure
        task_ids = list(dag.task_dict.keys())
        
        # THEN: It should have the expected tasks
        expected_tasks = [
            "download_dataset_task",
            "local_to_gcs_task", 
            "cleanup_local_file_task"
        ]
        assert set(task_ids) == set(expected_tasks)
        
        # AND: Tasks should have correct dependencies
        download_task = dag.get_task("download_dataset_task")
        upload_task = dag.get_task("local_to_gcs_task")
        cleanup_task = dag.get_task("cleanup_local_file_task")
        
        # Download -> Upload -> Cleanup
        assert upload_task in download_task.downstream_list
        assert cleanup_task in upload_task.downstream_list
    
    def test_dag_has_correct_configuration(self, dagbag):
        """Test that the DAG has the correct configuration."""
        # GIVEN: A loaded DAG
        dag = dagbag.get_dag(self.DAG_ID)
        
        # THEN: It should have correct configuration
        assert dag.schedule_interval == '0 6 1 * *'  # Monthly at 6 AM
        assert dag.max_active_runs == 1
        assert dag.catchup is True
        assert 'nyc-taxi' in dag.tags
        
        # AND: Default args should be set correctly
        assert dag.default_args['owner'] == 'airflow'
        assert dag.default_args['retries'] == 1
    
    @patch('subprocess.run')
    def test_download_file_function_success(self, mock_subprocess, mock_gcp_environment, sample_execution_date):
        """Test the download_file function with successful download."""
        # GIVEN: A mocked successful subprocess call
        mock_subprocess.return_value.returncode = 0
        
        # WHEN: We import and call the download function
        from dags.nyc_taxi_gcs_yellow_dag import download_file
        
        # THEN: It should execute without errors
        try:
            download_file(execution_date=sample_execution_date)
            # Verify subprocess was called with correct parameters
            expected_filename = f"yellow_tripdata_{sample_execution_date.strftime('%Y-%m')}.parquet"
            expected_url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{expected_filename}"
            expected_local_path = os.path.join("/tmp/airflow", expected_filename)
            
            mock_subprocess.assert_called_once_with(
                ["curl", "-L", "-o", expected_local_path, expected_url],
                check=True
            )
        except Exception as e:
            pytest.fail(f"download_file should not raise an exception: {e}")
    
    @patch('subprocess.run')
    def test_download_file_function_failure(self, mock_subprocess, mock_gcp_environment, sample_execution_date):
        """Test the download_file function with failed download."""
        # GIVEN: A mocked failed subprocess call
        from subprocess import CalledProcessError
        mock_subprocess.side_effect = CalledProcessError(1, 'curl')
        
        # WHEN: We call the download function
        from dags.nyc_taxi_gcs_yellow_dag import download_file
        
        # THEN: It should raise an exception
        with pytest.raises(CalledProcessError):
            download_file(execution_date=sample_execution_date)
    
    def test_upload_to_gcs_function(self, mock_gcs_client, mock_gcp_environment, sample_execution_date):
        """Test the upload_to_gcs function."""
        # GIVEN: A mocked GCS client
        # WHEN: We call the upload function
        from dags.nyc_taxi_gcs_yellow_dag import upload_to_gcs
        
        # THEN: It should execute without errors
        try:
            upload_to_gcs(bucket="test-bucket", execution_date=sample_execution_date)
            
            # Verify GCS client was called correctly
            mock_gcs_client.assert_called_once()
            mock_client_instance = mock_gcs_client.return_value
            mock_client_instance.bucket.assert_called_once_with("test-bucket")
        except Exception as e:
            pytest.fail(f"upload_to_gcs should not raise an exception: {e}")
    
    @patch('os.path.exists')
    @patch('os.remove')
    def test_remove_local_file_function_file_exists(self, mock_remove, mock_exists, mock_gcp_environment, sample_execution_date):
        """Test the remove_local_file function when file exists."""
        # GIVEN: A file that exists
        mock_exists.return_value = True
        
        # WHEN: We call the remove function
        from dags.nyc_taxi_gcs_yellow_dag import remove_local_file
        remove_local_file(execution_date=sample_execution_date)
        
        # THEN: The file should be removed
        expected_filename = f"yellow_tripdata_{sample_execution_date.strftime('%Y-%m')}.parquet"
        expected_path = os.path.join("/tmp/airflow", expected_filename)
        
        mock_exists.assert_called_once_with(expected_path)
        mock_remove.assert_called_once_with(expected_path)
    
    @patch('os.path.exists')
    @patch('os.remove')
    def test_remove_local_file_function_file_not_exists(self, mock_remove, mock_exists, mock_gcp_environment, sample_execution_date):
        """Test the remove_local_file function when file doesn't exist."""
        # GIVEN: A file that doesn't exist
        mock_exists.return_value = False
        
        # WHEN: We call the remove function
        from dags.nyc_taxi_gcs_yellow_dag import remove_local_file
        remove_local_file(execution_date=sample_execution_date)
        
        # THEN: Remove should not be called
        mock_remove.assert_not_called()
    
    def test_dag_tasks_have_correct_operators(self, dagbag):
        """Test that tasks use the correct operators."""
        # GIVEN: A loaded DAG
        dag = dagbag.get_dag(self.DAG_ID)
        
        # THEN: Each task should use the correct operator
        download_task = dag.get_task("download_dataset_task")
        upload_task = dag.get_task("local_to_gcs_task")
        cleanup_task = dag.get_task("cleanup_local_file_task")
        
        assert download_task.task_type == "PythonOperator"
        assert upload_task.task_type == "PythonOperator"
        assert cleanup_task.task_type == "PythonOperator"
    
    @pytest.mark.integration
    def test_dag_runs_without_errors(self, dagbag, sample_execution_date):
        """Integration test: Test that the DAG can be instantiated and run."""
        # GIVEN: A loaded DAG
        dag = dagbag.get_dag(self.DAG_ID)
        
        # WHEN: We create a DAG run
        dag_run = dag.create_dagrun(
            run_id=f"test_run_{sample_execution_date}",
            execution_date=sample_execution_date,
            state=State.RUNNING
        )
        
        # THEN: The DAG run should be created successfully
        assert dag_run is not None
        assert dag_run.dag_id == self.DAG_ID
        assert dag_run.execution_date == sample_execution_date


class TestDataQuality:
    """Test suite for data quality validation."""
    
    def test_filename_generation_logic(self, sample_execution_date):
        """Test that filenames are generated correctly."""
        # GIVEN: An execution date
        # WHEN: We generate a filename
        expected_filename = f"yellow_tripdata_{sample_execution_date.strftime('%Y-%m')}.parquet"
        
        # THEN: The filename should match the expected format
        assert expected_filename == "yellow_tripdata_2024-01.parquet"
    
    def test_gcs_object_name_generation(self, sample_execution_date):
        """Test that GCS object names are generated correctly."""
        # GIVEN: An execution date
        filename = f"yellow_tripdata_{sample_execution_date.strftime('%Y-%m')}.parquet"
        
        # WHEN: We generate the GCS object name
        object_name = f"nyc_taxi_data/yellow_taxi_data/{filename}"
        
        # THEN: The object name should follow the expected pattern
        assert object_name == "nyc_taxi_data/yellow_taxi_data/yellow_tripdata_2024-01.parquet"
        assert object_name.startswith("nyc_taxi_data/yellow_taxi_data/")
        assert object_name.endswith(".parquet")
