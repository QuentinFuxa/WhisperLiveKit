import importlib


def test_import_src_guard():
    module = importlib.import_module("src")
    assert hasattr(module, "__file__")
