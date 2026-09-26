"""Unit tests for spark/ml/src/paths.py — where the fact_trips backup lives.

Audit item 10 moved `migration_backup/` (7.1 GB) out of the working tree. The
prep script used to hard-code `REPO_ROOT / "migration_backup"`, so the move
would have broken it silently: Spark reports a missing input directory the same
way whether the path is wrong or the data is gone.

The resolution rule is one line of policy and worth a test of its own:

  1. `MIGRATION_BACKUP_DIR`, if set, wins outright — that is how a different
     machine, an external disk, or a container mount points the prep somewhere
     else without editing code.
  2. Otherwise the default is the repo's **sibling** directory
     `nyc_taxi_migration_backup/`, which is where item 10 put it.

Both branches are tested because the fallback is the one that runs unattended
and the override is the one that runs on someone else's machine.
"""

from pathlib import Path

from spark.ml.src import paths


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_default_backup_dir_is_the_repo_sibling():
    """With no env var set, the backup sits beside the repo, not inside it."""
    resolved = paths.resolve_backup_dir(env={})

    assert resolved == REPO_ROOT.parent / "nyc_taxi_migration_backup"


def test_default_backup_dir_is_outside_the_working_tree():
    """The whole point of item 10: the default must not be under the repo."""
    resolved = paths.resolve_backup_dir(env={})

    assert not resolved.is_relative_to(REPO_ROOT)


def test_env_var_overrides_the_default(tmp_path):
    """MIGRATION_BACKUP_DIR wins, so a mount or an external disk needs no edit."""
    resolved = paths.resolve_backup_dir(env={"MIGRATION_BACKUP_DIR": str(tmp_path)})

    assert resolved == tmp_path


def test_env_var_is_expanded_and_absolute(monkeypatch, tmp_path):
    """`~` and a relative path both resolve — a half-resolved path fails late."""
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = paths.resolve_backup_dir(env={"MIGRATION_BACKUP_DIR": "~/backup"})

    assert resolved == tmp_path / "backup"
    assert resolved.is_absolute()


def test_blank_env_var_falls_back_to_the_default():
    """An exported-but-empty var is not a path. Treat it as unset."""
    resolved = paths.resolve_backup_dir(env={"MIGRATION_BACKUP_DIR": "   "})

    assert resolved == REPO_ROOT.parent / "nyc_taxi_migration_backup"


def test_fact_trips_dir_hangs_off_the_resolved_backup(tmp_path):
    """The prep reads one subdirectory of the backup; keep the join in one place."""
    resolved = paths.resolve_fact_trips_dir(env={"MIGRATION_BACKUP_DIR": str(tmp_path)})

    assert resolved == tmp_path / "fact_trips"


def test_repo_root_points_at_the_repository():
    """paths.REPO_ROOT anchors every other path; a wrong anchor breaks all of them."""
    assert (paths.REPO_ROOT / "CLAUDE.md").exists()
    assert paths.REPO_ROOT == REPO_ROOT


# ---------------------------------------------------------------------------
# Where the prep reads from — the local backup, or BigQuery through the connector
# ---------------------------------------------------------------------------

class TestSourceKind:
    """Migration plan M4 runs the same prep against two different sources.

    Locally it reads the 7.1 GB parquet backup. On Dataproc it reads
    `dbt_prod.fact_trips` through the BigQuery connector, and the point of that
    run is to prove the two agree — 128,408,323 rows and the same p99 caps. One
    `--source` flag therefore has to carry both, and the script has to be able
    to tell which it was handed before it builds a SparkSession.

    The project id is **required**, not optional. The connector would happily
    accept `dbt_prod.fact_trips` and resolve the project from the environment,
    which is how a run reads the wrong project's mart and reports the number
    with total confidence. Three segments or it is not a table.
    """

    def test_fully_qualified_table_is_bigquery(self):
        assert paths.is_bigquery_table(
            "dtc-de-project-506916.dbt_prod.fact_trips"
        )

    def test_the_local_sentinel_is_not_bigquery(self):
        assert not paths.is_bigquery_table(paths.LOCAL_SOURCE)

    def test_absolute_path_is_not_bigquery(self):
        assert not paths.is_bigquery_table("/data/nyc_taxi_backup/fact_trips")

    def test_bucket_uri_is_not_bigquery(self):
        """A gs:// URI has dots in the object name and must not be mistaken for
        a table id."""
        assert not paths.is_bigquery_table("gs://primary-data/ml/sample.parquet")

    def test_two_segments_is_rejected_so_the_project_stays_explicit(self):
        """`dbt_prod.fact_trips` is a valid table reference to the connector,
        which resolves the project from the environment. That is exactly how a
        run silently reads a different project's mart, so it is refused here."""
        assert not paths.is_bigquery_table("dbt_prod.fact_trips")

    def test_a_filename_with_an_extension_is_not_a_table(self):
        assert not paths.is_bigquery_table("sample_full.parquet")

    def test_four_segments_is_rejected(self):
        assert not paths.is_bigquery_table("a.b.c.d")

    def test_empty_segment_is_rejected(self):
        assert not paths.is_bigquery_table("project..table")

    def test_blank_source_is_rejected(self):
        assert not paths.is_bigquery_table("")

    def test_whitespace_is_not_a_table(self):
        assert not paths.is_bigquery_table("   ")
