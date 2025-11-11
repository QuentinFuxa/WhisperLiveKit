import os


def test_healthcheck_script_is_executable():
    script_path = os.path.join("infra", "systemd", "checks", "healthcheck.sh")
    assert os.path.isfile(script_path), f"{script_path} must exist"
    assert os.access(script_path, os.X_OK), f"{script_path} must be executable"
