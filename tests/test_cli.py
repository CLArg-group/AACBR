### Testing command line interface

import pytest
import os
from shutil import copy


CLI_FILE="aacbr/cli.py"

@pytest.fixture
def run(pytester):
  original_dir = os.getcwd()
  def do_run(*args):
    args = [f"{original_dir}/python", CLI_FILE] + list(args)
    return pytester._run(*args)
  return do_run

def test_run_aacbr_cli_default_arguments(tmp_path, run):
  data_files = ["cb.json", "new_5.json", "cb_to_new_5.json"]
  os.chdir(tmp_path)
  
  for f in data_files:
    copy(f"data/{f}", f"{f}")  
  
  with open("cb_to_new_5.json") as f:
    expected = f.read()
  tmp_path.unlink("cb_to_new_5.json") # to make sure it is now empty

  result = run()
  with open("cb_to_new_5.json") as f:
    # run() changes this file
    result = f.read()
  assert result == expected
