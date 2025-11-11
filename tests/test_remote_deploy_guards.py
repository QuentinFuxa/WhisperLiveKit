from pathlib import Path


REMOTE_DEPLOY = Path("scripts/remote_deploy.sh").read_text(encoding="utf-8")


def test_remote_deploy_performs_editable_install():
    assert "pip install -e ." in REMOTE_DEPLOY


def test_remote_deploy_writes_env_defaults():
    assert "APP_PORT=8000" in REMOTE_DEPLOY
    assert "PYTHONPATH=/opt/daymind" in REMOTE_DEPLOY
    assert "LEDGER_FILE=/opt/daymind/runtime/ledger.beancount" in REMOTE_DEPLOY


def test_remote_deploy_enables_services():
    assert "systemctl enable --now daymind-api daymind-fava" in REMOTE_DEPLOY
