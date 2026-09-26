"""Unit tests for the Airflow compose override that runs the stack on a GCE VM (M5).

`airflow/docker-compose.yaml` authenticates with a keyfile three ways: the
`GOOGLE_APPLICATION_CREDENTIALS` env var, the `key_path` inside
`AIRFLOW_CONN_GOOGLE_CLOUD_DEFAULT`, and the `../secrets` bind mount. On the VM the
attached service account supplies Application Default Credentials instead, and the
Deployment split rule says no keyfile reaches the VM or an image, ever.

`airflow/docker-compose.vm.yaml` removes all three. These tests fail if any of the three
comes back for any Airflow service, or if the override drops something the DAGs need.

Two layers:

* A pure-YAML model of Compose's merge (`!reset` removes a key, `!override` replaces a
  list). No Docker needed, so it runs on the host, in CI and in the dev container.
* `docker compose config` on the real pair of files, which proves Compose itself resolves
  the merge the same way. It skips where the Docker CLI is absent (the dev container).
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
AIRFLOW_DIR = PROJECT_ROOT / "airflow"
BASE = AIRFLOW_DIR / "docker-compose.yaml"
OVERRIDE = AIRFLOW_DIR / "docker-compose.vm.yaml"

KEYFILE_ENV = "GOOGLE_APPLICATION_CREDENTIALS"
CONN_ENV = "AIRFLOW_CONN_GOOGLE_CLOUD_DEFAULT"
SECRETS_TARGET = "/.google/credentials"

# Env the DAGs read through os.environ; the override must leave it alone.
KEPT_ENV = ["GCP_PROJECT_ID", "GCP_GCS_BUCKET", "INGEST_START_DATE", "INGEST_END_DATE"]


class _Reset:
    """Stands for Compose's `!reset` tag: the key is removed from the merged result."""


class _OverrideLoader(yaml.SafeLoader):
    pass


_OverrideLoader.add_constructor("!reset", lambda loader, node: _Reset())
_OverrideLoader.add_constructor(
    "!override", lambda loader, node: loader.construct_sequence(node)
)


def _load_override():
    return yaml.load(OVERRIDE.read_text(), Loader=_OverrideLoader)


def _airflow_services(base):
    """Services that inherit the keyfile from `x-airflow-common`."""
    return {
        name: svc
        for name, svc in base["services"].items()
        if KEYFILE_ENV in (svc.get("environment") or {})
    }


def _merge(base_svc, over_svc):
    """Compose's merge for the two keys the override touches."""
    env = dict(base_svc.get("environment") or {})
    for key, value in (over_svc.get("environment") or {}).items():
        if isinstance(value, _Reset):
            env.pop(key, None)
        else:
            env[key] = value
    volumes = over_svc.get("volumes", base_svc.get("volumes", []))
    return env, volumes


def _volume_target(volume):
    if isinstance(volume, dict):
        return volume.get("target")
    return volume.split(":")[1]


@pytest.fixture(scope="module")
def merged():
    base = yaml.safe_load(BASE.read_text())
    override = _load_override()
    services = _airflow_services(base)
    assert services, "the base file no longer carries the keyfile env var"
    return {
        name: _merge(svc, override["services"][name])
        for name, svc in services.items()
    }


def test_override_file_exists():
    assert OVERRIDE.is_file()


def test_override_names_every_airflow_service():
    base = yaml.safe_load(BASE.read_text())
    missing = set(_airflow_services(base)) - set(_load_override()["services"])
    assert not missing, f"services still carrying the keyfile: {sorted(missing)}"


def test_no_keyfile_env_var(merged):
    for name, (env, _) in merged.items():
        assert KEYFILE_ENV not in env, name


def test_connection_uses_adc(merged):
    for name, (env, _) in merged.items():
        conn = env[CONN_ENV]
        assert conn.startswith("google-cloud-platform://"), name
        assert "key_path" not in conn and "keyfile" not in conn, name


def test_no_secrets_mount(merged):
    for name, (_, volumes) in merged.items():
        for volume in volumes:
            text = volume if isinstance(volume, str) else json.dumps(volume)
            assert "secrets" not in text, (name, volume)
            assert _volume_target(volume) != SECRETS_TARGET, (name, volume)


def test_dag_env_is_kept(merged):
    for name, (env, _) in merged.items():
        for key in KEPT_ENV:
            assert key in env, (name, key)


def test_dags_mount_is_kept(merged):
    """The workers still need the DAG folder; the init service mounts /sources instead."""
    for name, (_, volumes) in merged.items():
        if name == "airflow-init":
            continue
        assert "/opt/airflow/dags" in [_volume_target(v) for v in volumes], name


@pytest.mark.skipif(shutil.which("docker") is None, reason="needs the Docker CLI")
def test_compose_resolves_the_pair_without_a_keyfile():
    result = subprocess.run(
        [
            "docker", "compose",
            "-f", str(BASE), "-f", str(OVERRIDE),
            "--profile", "debug", "--profile", "flower",
            "config", "--format", "json",
        ],
        capture_output=True, text=True, cwd=AIRFLOW_DIR, check=False,
    )
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    for name, svc in config["services"].items():
        env = svc.get("environment") or {}
        if not name.startswith("airflow") and name != "flower":
            continue
        assert KEYFILE_ENV not in env, name
        assert "key_path" not in env.get(CONN_ENV, ""), name
        for volume in svc.get("volumes") or []:
            assert "secrets" not in str(volume.get("source", "")), (name, volume)
            assert volume.get("target") != SECRETS_TARGET, (name, volume)
