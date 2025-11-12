def test_import_main():
  import importlib

  module = importlib.import_module("src.api.main")
  assert hasattr(module, "app")
