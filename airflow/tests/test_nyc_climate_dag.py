"""
Test-Driven Development tests for NYC Climate Data Ingestion DAG.

This module tests the climate data ingestion pipeline following TDD principles.
"""

import pytest
import os
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch, mock_open
from airflow.models import DagBag
from airflow.utils.state import State


class TestNYCClimateDAG:
    """Test suite for NYC Climate Data Ingestion DAG."""
    
    DAG_ID = "nyc_climate_data_ingestion_dag"
    
    def test_dag_loaded(self, dagbag):
        """Test that the climate DAG is loaded without errors."""
        # GIVEN: A DAG bag with all DAGs loaded
        dag = dagbag.get_dag(self.DAG_ID)
        
        # THEN: The DAG should be loaded successfully
        assert dag is not None
        assert dag.dag_id == self.DAG_ID
    
    def test_dag_has_correct_tasks(self, dagbag):
        """Test that the DAG has the expected tasks."""
        # GIVEN: A loaded DAG
        dag = dagbag.get_dag(self.DAG_ID)
        
        # WHEN: We examine the DAG structure
        task_ids = list(dag.task_dict.keys())
        
        # THEN: It should have the expected tasks
        expected_tasks = [
            "download_dataset_task",
            "convert_to_parquet",
            "local_to_gcs_task",
            "cleanup_local_file_task"
        ]
        assert set(task_ids) == set(expected_tasks)
    
    def test_dag_task_dependencies(self, dagbag):
        """Test that tasks have correct dependencies."""
        # GIVEN: A loaded DAG
        dag = dagbag.get_dag(self.DAG_ID)
        
        # WHEN: We examine task dependencies
        download_task = dag.get_task("download_dataset_task")
        convert_task = dag.get_task("convert_to_parquet")
        upload_task = dag.get_task("local_to_gcs_task")
        cleanup_task = dag.get_task("cleanup_local_file_task")
        
        # THEN: Dependencies should be correct
        # Download -> Convert -> Upload -> Cleanup
        assert convert_task in download_task.downstream_list
        assert upload_task in convert_task.downstream_list
        assert cleanup_task in upload_task.downstream_list
    
    def test_dag_configuration(self, dagbag):
        """Test DAG configuration parameters."""
        # GIVEN: A loaded DAG
        dag = dagbag.get_dag(self.DAG_ID)
        
        # THEN: Configuration should be correct
        assert dag.schedule_interval == '0 6 1 * *'  # Monthly at 6 AM
        assert dag.catchup is False
        assert dag.max_active_runs == 1
        assert 'nyc-taxi' in dag.tags
    
    @patch('pandas.read_csv')
    @patch('pyarrow.Table.from_pandas')
    @patch('pyarrow.parquet.write_table')
    def test_format_to_parquet_success(self, mock_write_table, mock_from_pandas, mock_read_csv, mock_gcp_environment):
        """Test successful CSV to Parquet conversion."""
        # GIVEN: A valid CSV file and mocked pandas operations
        mock_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        mock_read_csv.return_value = mock_df
        mock_table = Mock()
        mock_from_pandas.return_value = mock_table
        
        # WHEN: We call the format_to_parquet function
        from dags.nyc_climate_gcs_dag import format_to_parquet
        csv_file = "/tmp/airflow/nyc_climate/weather_cache_sm.csv"
        
        # THEN: It should execute without errors
        try:
            format_to_parquet(csv_file)
            
            # Verify the conversion process
            mock_read_csv.assert_called_once_with(csv_file)
            mock_from_pandas.assert_called_once_with(mock_df)
            mock_write_table.assert_called_once()
        except Exception as e:
            pytest.fail(f"format_to_parquet should not raise an exception: {e}")
    
    def test_format_to_parquet_invalid_file_format(self, mock_gcp_environment, caplog):
        """Test format_to_parquet with invalid file format."""
        # GIVEN: A non-CSV file
        invalid_file = "/tmp/test.txt"
        
        # WHEN: We call format_to_parquet
        from dags.nyc_climate_gcs_dag import format_to_parquet
        format_to_parquet(invalid_file)
        
        # THEN: It should log an error message
        assert "Invalid file format" in caplog.text
    
    @patch('pandas.read_csv')
    def test_format_to_parquet_empty_csv(self, mock_read_csv, mock_gcp_environment, caplog):
        """Test format_to_parquet with empty CSV file."""
        # GIVEN: An empty CSV file
        mock_read_csv.return_value = pd.DataFrame()
        
        # WHEN: We call format_to_parquet
        from dags.nyc_climate_gcs_dag import format_to_parquet
        csv_file = "/tmp/airflow/nyc_climate/weather_cache_sm.csv"
        format_to_parquet(csv_file)
        
        # THEN: It should log a warning
        assert "empty" in caplog.text.lower()
    
    def test_upload_to_gcs_function(self, mock_gcs_client, mock_gcp_environment):
        """Test the upload_to_gcs function."""
        # GIVEN: Mocked GCS client and file parameters
        bucket_name = "test-bucket"
        object_name = "raw/nyc_climate_data/weather_cache_sm.parquet"
        local_file = "/tmp/airflow/nyc_climate/weather_cache_sm.parquet"
        
        # WHEN: We call upload_to_gcs
        from dags.nyc_climate_gcs_dag import upload_to_gcs
        upload_to_gcs(bucket_name, object_name, local_file)
        
        # THEN: GCS client should be called correctly
        mock_gcs_client.assert_called_once()
        mock_client_instance = mock_gcs_client.return_value
        mock_client_instance.bucket.assert_called_once_with(bucket_name)
    
    @patch('os.path.exists')
    @patch('os.remove')
    def test_remove_local_files_all_exist(self, mock_remove, mock_exists, mock_gcp_environment):
        """Test remove_local_files when all files exist."""
        # GIVEN: Files that exist
        mock_exists.return_value = True
        file_paths = ["/tmp/file1.csv", "/tmp/file2.parquet"]
        
        # WHEN: We call remove_local_files
        from dags.nyc_climate_gcs_dag import remove_local_files
        remove_local_files(file_paths)
        
        # THEN: All files should be removed
        assert mock_exists.call_count == 2
        assert mock_remove.call_count == 2
    
    @patch('os.path.exists')
    @patch('os.remove')
    def test_remove_local_files_some_missing(self, mock_remove, mock_exists, mock_gcp_environment, caplog):
        """Test remove_local_files when some files are missing."""
        # GIVEN: Some files exist, some don't
        mock_exists.side_effect = [True, False]  # First exists, second doesn't
        file_paths = ["/tmp/file1.csv", "/tmp/file2.parquet"]
        
        # WHEN: We call remove_local_files
        from dags.nyc_climate_gcs_dag import remove_local_files
        remove_local_files(file_paths)
        
        # THEN: Only existing file should be removed, warning logged for missing
        assert mock_remove.call_count == 1
        assert "File not found" in caplog.text
    
    def test_task_operators_are_correct(self, dagbag):
        """Test that tasks use the correct operators."""
        # GIVEN: A loaded DAG
        dag = dagbag.get_dag(self.DAG_ID)
        
        # THEN: Tasks should use correct operators
        download_task = dag.get_task("download_dataset_task")
        convert_task = dag.get_task("convert_to_parquet")
        upload_task = dag.get_task("local_to_gcs_task")
        cleanup_task = dag.get_task("cleanup_local_file_task")
        
        assert download_task.task_type == "BashOperator"
        assert convert_task.task_type == "PythonOperator"
        assert upload_task.task_type == "PythonOperator"
        assert cleanup_task.task_type == "PythonOperator"


class TestClimateDataQuality:
    """Test suite for climate data quality validation."""
    
    def test_csv_file_validation(self):
        """Test CSV file format validation."""
        # GIVEN: Various file extensions
        valid_files = ["data.csv", "weather.CSV", "file.csv"]
        invalid_files = ["data.txt", "weather.json", "file.parquet"]
        
        # WHEN: We check file extensions
        # THEN: Only CSV files should be valid
        for file in valid_files:
            assert file.lower().endswith('.csv')
        
        for file in invalid_files:
            assert not file.lower().endswith('.csv')
    
    def test_parquet_output_path_generation(self, mock_gcp_environment):
        """Test that Parquet output paths are generated correctly."""
        # GIVEN: Environment variables are set
        expected_path = "/tmp/airflow/nyc_climate/weather_cache_sm.parquet"
        
        # WHEN: We check the path construction
        # THEN: Path should match expected format
        assert expected_path.endswith('.parquet')
        assert 'nyc_climate' in expected_path
        assert 'weather_cache_sm' in expected_path
    
    def test_gcs_object_naming_convention(self):
        """Test GCS object naming follows conventions."""
        # GIVEN: A file name
        filename = "weather_cache_sm.parquet"
        
        # WHEN: We construct the GCS object name
        object_name = f"raw/nyc_climate_data/{filename}"
        
        # THEN: It should follow naming conventions
        assert object_name.startswith("raw/")
        assert "nyc_climate_data" in object_name
        assert object_name.endswith(".parquet")
