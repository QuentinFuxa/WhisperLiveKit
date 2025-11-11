from pathlib import Path


def test_android_workflow_has_manual_trigger_and_debug_build():
    workflow = Path(".github/workflows/android_build.yml")
    assert workflow.exists(), "android_build.yml must exist"
    contents = workflow.read_text(encoding="utf-8")
    assert "workflow_dispatch" in contents, "android_build.yml should support manual dispatch"
    assert "assembleDebug" in contents, "android_build.yml should assemble debug builds"
