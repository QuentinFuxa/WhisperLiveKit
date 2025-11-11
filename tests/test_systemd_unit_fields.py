from pathlib import Path


def _read(path: str) -> str:
    data = Path(path)
    assert data.exists(), f"{path} missing"
    return data.read_text(encoding="utf-8")


def test_api_unit_fields_and_execstart():
    contents = _read("infra/systemd/daymind-api.service")
    for needle in (
        "WorkingDirectory=/opt/daymind",
        "EnvironmentFile=/etc/default/daymind",
        "Environment=PYTHONPATH=/opt/daymind",
        "ExecStart=/opt/daymind/venv/bin/uvicorn",
    ):
        assert needle in contents


def test_fava_unit_fields_and_execstart():
    contents = _read("infra/systemd/daymind-fava.service")
    for needle in (
        "WorkingDirectory=/opt/daymind",
        "EnvironmentFile=/etc/default/daymind",
        "Environment=PYTHONPATH=/opt/daymind",
    ):
        assert needle in contents
    assert "ExecStart=/opt/daymind/scripts/start_fava.sh" in contents
