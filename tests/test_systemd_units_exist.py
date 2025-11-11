from pathlib import Path


def _read(path: str) -> str:
    file_path = Path(path)
    assert file_path.exists(), f"{path} missing"
    return file_path.read_text(encoding="utf-8")


def test_api_unit_contains_uvicorn_exec():
    contents = _read("infra/systemd/daymind-api.service")
    assert "ExecStart=/opt/daymind/venv/bin/uvicorn src.api.main:app" in contents


def test_fava_unit_calls_wrapper():
    contents = _read("infra/systemd/daymind-fava.service")
    assert "ExecStart=/opt/daymind/scripts/start_fava.sh" in contents
