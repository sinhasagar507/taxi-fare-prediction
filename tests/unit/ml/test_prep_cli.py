"""The prep's command line: `--source` is required (owner decision 4, 2026-09-13).

`00_prep_spark.py` used to default `--source` to the local parquet backup. Since
D-012 that backup holds pre-fix data, and the other source is a 300M-row
BigQuery read. Neither should start by accident, so every run names its source.

The parser is reached through `build_parser()`, so these tests need no
SparkSession. The `main()` test replaces `build_spark` with a failure, so a
regression cannot launch a real prep from the test suite.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyspark", reason="00_prep_spark.py imports pyspark at module level")

REPO_ROOT = Path(__file__).resolve().parents[3]
PREP_PATH = REPO_ROOT / "spark" / "ml" / "00_prep_spark.py"

BIGQUERY_TABLE = "dtc-de-project-506916.dbt_prod.fact_trips"


def _load_prep():
    """Import `00_prep_spark.py` by path — the leading digit blocks `import`."""
    spec = importlib.util.spec_from_file_location("prep_spark_cli", PREP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prep():
    return _load_prep()


def test_a_run_without_source_is_refused(prep):
    with pytest.raises(SystemExit) as exc:
        prep.build_parser().parse_args([])
    assert exc.value.code == 2


def test_the_refusal_names_the_missing_flag(prep, capsys):
    with pytest.raises(SystemExit):
        prep.build_parser().parse_args([])
    assert "--source" in capsys.readouterr().err


def test_the_local_backup_is_accepted_when_named(prep):
    args = prep.build_parser().parse_args(["--source", prep.LOCAL_SOURCE])
    assert args.source == prep.LOCAL_SOURCE


def test_a_bigquery_table_is_accepted_when_named(prep):
    args = prep.build_parser().parse_args(["--source", BIGQUERY_TABLE])
    assert args.source == BIGQUERY_TABLE


def test_the_other_defaults_are_unchanged(prep):
    """Only `--source` loses its default; the sampling contract stays put."""
    args = prep.build_parser().parse_args(["--source", prep.LOCAL_SOURCE])
    assert args.full_frac == 0.10
    assert args.work_frac == 0.06
    assert args.seed == 42
    assert args.limit_files is None
    assert args.output == prep.DEFAULT_OUTPUT
    assert args.cluster is False


def test_main_refuses_before_any_spark_session_starts(prep, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["00_prep_spark.py"])
    monkeypatch.setattr(
        prep, "build_spark",
        lambda *a, **k: pytest.fail("main() reached Spark without a --source"),
    )
    with pytest.raises(SystemExit) as exc:
        prep.main()
    assert exc.value.code == 2
