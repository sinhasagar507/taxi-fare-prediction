"""Where the data lives — resolved in one place, not hard-coded per script.

The 128.78M-row `fact_trips` backup is 7.1 GB. Audit item 10 moved it **out of
the working tree** (2026-09-01): a git repository is a poor container for 7 GB
of parquet, it slowed every status and archive operation, and one bad
`.dockerignore` edit would have shipped the whole thing into an image.

Moving it meant the prep script could no longer say `REPO_ROOT /
"migration_backup"`. So the location becomes policy, stated once:

  1. `MIGRATION_BACKUP_DIR` wins if it is set to a non-blank value. That is the
     hook for another machine, an external disk, or a container mount.
  2. Otherwise the default is the repo's **sibling** `nyc_taxi_migration_backup/`.

The functions take `env` as an argument rather than reading `os.environ`
directly, so a test can state the environment it means instead of mutating the
process and hoping to restore it.

Deliberately free of any `pyspark` import: `00_prep_spark.py` cannot be imported
by a test (its name starts with a digit), so the part worth testing lives here.
"""

from __future__ import annotations

import os
from pathlib import Path

# spark/ml/src/paths.py -> spark/ml/src -> spark/ml -> spark -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]

BACKUP_DIR_ENV = "MIGRATION_BACKUP_DIR"

# Beside the repo, not inside it. Item 10's whole point.
DEFAULT_BACKUP_DIR = REPO_ROOT.parent / "nyc_taxi_migration_backup"

# The one subdirectory the prep reads: the dbt-built fact table, exported from
# BigQuery.
FACT_TRIPS_SUBDIR = "fact_trips"


def resolve_backup_dir(env: dict | None = None) -> Path:
    """Absolute path to the migration backup root.

    A blank value is treated as unset. An exported-but-empty variable is a
    common shell accident (`export MIGRATION_BACKUP_DIR=$SOMETHING_UNSET`), and
    honouring it would resolve to the current directory — a wrong answer that
    looks like a right one.

    `~` is expanded and the result is made absolute here rather than at the
    point of use, because a half-resolved path fails deep inside Spark with a
    message about a missing directory rather than about a misconfigured
    environment.
    """
    env = os.environ if env is None else env
    raw = env.get(BACKUP_DIR_ENV, "")
    if raw.strip():
        return Path(raw.strip()).expanduser().resolve()
    return DEFAULT_BACKUP_DIR


def resolve_fact_trips_dir(env: dict | None = None) -> Path:
    """Absolute path to the parquet `fact_trips` directory the prep reads."""
    return resolve_backup_dir(env) / FACT_TRIPS_SUBDIR


# --- which source the prep reads --------------------------------------------
# Migration plan M4 runs the same prep twice: against the local parquet backup,
# and against `dbt_prod.fact_trips` through the BigQuery connector. The whole
# point of the second run is that the two agree, so one flag carries both and
# the script asks this function which it was handed.
LOCAL_SOURCE = "local"

# project.dataset.table. Exactly three.
_TABLE_SEGMENTS = 3


def is_bigquery_table(source: str) -> bool:
    """True when `source` names a BigQuery table rather than a local backup.

    **The project id is required.** The connector accepts a bare
    `dataset.table` and resolves the project from the environment, which is the
    mechanism by which a run reads a different project's mart and then reports
    the row count with complete confidence. This milestone reads `dbt_prod` and
    must be able to say which `dbt_prod`. Two segments are therefore refused,
    not helpfully completed.

    Legacy `project:dataset.table` is refused for the same reason a typo is:
    one accepted spelling means one thing to check. A path or a bucket URI is
    refused by the same rule, since `sample_full.parquet` and
    `gs://bucket/a.b.parquet` both carry dots that are not separators.
    """
    text = str(source).strip()
    if not text or text == LOCAL_SOURCE:
        return False
    if "/" in text or ":" in text:
        return False
    segments = text.split(".")
    return len(segments) == _TABLE_SEGMENTS and all(segments)
