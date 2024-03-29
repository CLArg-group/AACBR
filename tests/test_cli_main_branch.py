### Testing command line interface

## currently this works for 'main', the original implementation

import pytest
import os
from shutil import copy
from pathlib import Path
import subprocess

pytest_plugins = ["pytester"]

CLI_FILE="aacbr.py"

@pytest.fixture
def run():
  original_dir = os.getcwd()
  # raise(Exception(original_dir))
  # raise(Exception([child for child in Path(original_dir).iterdir()]))
  def do_run(*args):
    args = ["python", f"{original_dir}/"+CLI_FILE] + list(args)
    return subprocess.run(args, capture_output=True)
  return do_run

def _test_run_aacbr_cli_default_arguments(tmp_path, run):
  data_files = ["cb.json", "new_5.json", "cb_to_new_5.json"]
  data_dir = "."

  copy(Path(f"{data_dir}/cb.json"), f"{tmp_path}/cb.json")
  copy(Path(f"{data_dir}/new_5.json"), f"{tmp_path}/new.json")
  # copy(Path(f"{data_dir}/cb_to_new_5.json"), f"{tmp_path}/cb_to_new.json")

  with open(f"{data_dir}/cb_to_new_5.json") as f:
    expected = f.read()
  
  # raise(Exception(tmp_path))
  # raise(Exception([child for child in Path(tmp_path).iterdir()]))
  os.chdir(tmp_path)

  result = run()
  # raise(Exception(result.stdout, result.stderr))
  with open("cb_to_new.json") as f:
    result = f.read()
  assert result == expected
