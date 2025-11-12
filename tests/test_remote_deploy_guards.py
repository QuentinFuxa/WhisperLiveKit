from pathlib import Path


REMOTE_DEPLOY = Path("scripts/remote_deploy.sh").read_text(encoding="utf-8")


def test_remote_deploy_performs_editable_install():
    assert "pip install -e . --no-deps" in REMOTE_DEPLOY


def test_remote_deploy_writes_env_defaults():
    for needle in (
        "APP_HOST=127.0.0.1",
        "APP_PORT=8000",
        "FAVA_PORT=8010",
        "REDIS_URL=redis://127.0.0.1:6379",
        "PYTHONPATH=/opt/daymind",
    ):
        assert needle in REMOTE_DEPLOY


def test_remote_deploy_enables_services():
    assert "systemctl enable daymind-api.service daymind-fava.service" in REMOTE_DEPLOY
