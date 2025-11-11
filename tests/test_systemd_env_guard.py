from pathlib import Path


def _contents(path: str) -> str:
    data = Path(path)
    assert data.exists(), f"{path} does not exist"
    return data.read_text(encoding="utf-8")


def test_api_unit_has_env_and_execstartpre():
    contents = _contents("infra/systemd/daymind-api.service")
    assert "WorkingDirectory=/opt/daymind" in contents
    assert "EnvironmentFile=/etc/default/daymind" in contents
    assert "Environment=PYTHONPATH=/opt/daymind" in contents
    assert "ExecStartPre=/usr/bin/test -x /opt/daymind/venv/bin/uvicorn" in contents


def test_fava_unit_has_env_and_execstartpre():
    contents = _contents("infra/systemd/daymind-fava.service")
    assert "WorkingDirectory=/opt/daymind" in contents
    assert "EnvironmentFile=/etc/default/daymind" in contents
    assert "Environment=PYTHONPATH=/opt/daymind" in contents
    assert "ExecStartPre=/usr/bin/test -x /opt/daymind/venv/bin/fava" in contents
