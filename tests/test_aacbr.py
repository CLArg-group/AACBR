import json
import pytest
from aacbr import Caacbr, Case
from test_aux import prepare_tests

TEST_FILES = ["test_cautious_monotonicity.json",
              "test_topological_sort.json",
              "test_incoherence_and_order.json"]
# AACBR_TYPES = ["non_cautious", "cautious"]
AACBR_TYPES = ["non_cautious"]
AACBR_TYPES_FUNCTION = {
  "non_cautious":"give_casebase",
  "cautious": "give_cautious_subset_of_casebase"
}

TESTS = prepare_tests(TEST_FILES)

def prepare_tests(test_files):
  tests = []
  for myfilepath in test_files:
    with open(myfilepath) as myfile:
      loaded = json.load(myfile)
      if type(loaded) == dict:
        test = loaded
        tests.append(test)
      elif type(loaded) == list:
        for test in loaded:
          tests.append(test)
      else:
        raise(Exception("loaded should be either a json object representing one test setup or a list of such json objects"))
  return tests

@pytest.fixture(params=TESTS)
def setup(request):
  test = request.param
  setup_result = Caacbr(test["outcomes"]["default"], test["outcomes"]["nondefault"], test["outcomes"]["undecided"])
  return test, setup_result
  

def test_files(setup):
  test, scenario = setup
  for aacbr_type in AACBR_TYPES:
    casebase = scenario.load_cases(test["casebase"])
    casebase_prepared = getattr(scenario, AACBR_TYPES_FUNCTION[aacbr_type])(casebase)
    casebase_ids = set(map(lambda x: getattr(x, "id"), casebase_prepared))
    assert set(casebase_ids) == set(test["casebase_expected"][aacbr_type])
    
    for newcase_spec in test["newcases"]:
      newcase = Case(id=newcase_spec["id"], factors=set(newcase_spec["factors"]))
      newcase_prepared = scenario.give_new_cases(casebase_prepared, [newcase])
      result = scenario.give_predictions(casebase_prepared, newcase_prepared)
      prediction = result[1][0]["Prediction"]
      assert prediction == newcase_spec["outcome_expected"][aacbr_type], f"Failed for {newcase_spec}, in type {aacbr_type}"

