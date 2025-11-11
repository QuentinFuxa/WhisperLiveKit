def test_ci_workflow_has_manual_trigger():
    path = ".github/workflows/ci_cd.yml"
    with open(path, "r", encoding="utf-8") as fh:
        contents = fh.read()
    assert "workflow_dispatch:" in contents, "ci_cd.yml must expose a manual workflow_dispatch trigger"
