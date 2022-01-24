### Tests the model/classifier itself, predictions, argumentation framework, etc

from context import aacbr
import pytest

def test_import():
  from aacbr import Aacbr
  pass

@pytest.mark.skip(reason="Undefined tests")
@pytest.mark.usefixtures("test_import")
class TestAacbr:
  def test_argumentation_framework():
    pass
  def test_predictions():
    pass
  def test_grounded_extension():
    pass
  def scikit_learning_like_api():
    # It would be nice to have it compatible with the scikit-learn API:
    # https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects
    # import data...
    train_data = None
    test_data = None
    expected_output = None
    clf = Aacbr()
    predicted_output = clf.fit(train_data).predict(test_data)
    assert expected_output == predicted_output

@pytest.mark.skip(reason="Undefined tests")
@pytest.mark.usefixtures("test_import")
class TestAacbr:
  def test_argumentation_framework_cautious():
    pass
  def test_predictions_cautious():
    pass
  def test_grounded_extension_cautious():
    pass
  def scikit_learning_like_api():
    # import data...
    train_data = None
    test_data = None
    expected_output = None
    clf = Aacbr()
    predicted_output = clf.fit(train_data).predict(test_data)
    assert expected_output == predicted_output

#### old tests
import json
from aacbr.aacbr import Aacbr, Case
import sys
TEST_PATH_PREFIX = "../tests/data/"


def prepare_tests(test_files):
  tests = []
  for myfilepath in test_files:
    with open(TEST_PATH_PREFIX + myfilepath) as myfile:
      # expected to run from root directory
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


TEST_FILES = ["test_cautious_monotonicity.json",
              "test_topological_sort.json",
              "test_incoherence_and_order.json"]
# AACBR_TYPES = ["non_cautious", "cautious"]
# AACBR_TYPES = ["non_cautious"]
AACBR_TYPES_FUNCTION = {
  "non_cautious":"give_casebase",
  "cautious": "give_cautious_subset_of_casebase"
}

TESTS = prepare_tests(TEST_FILES)


@pytest.fixture(params=TESTS)
def setup(request):
  test = request.param
  setup_result = Aacbr(test["outcomes"]["default"], test["outcomes"]["nondefault"], test["outcomes"]["undecided"])
  return test, setup_result
  
  
def run_test_with_aacbr_type(aacbr_type, setup):
  test, scenario = setup
  casebase = scenario.load_cases(TEST_PATH_PREFIX + test["casebase"])
  casebase_prepared = getattr(scenario, AACBR_TYPES_FUNCTION[aacbr_type])(casebase)
  casebase_ids = set(map(lambda x: getattr(x, "id"), casebase_prepared))
  assert set(casebase_ids) == set(test["casebase_expected"][aacbr_type])

  for newcase_spec in test["newcases"]:
    newcase = Case(id=newcase_spec["id"], factors=set(newcase_spec["factors"]))
    newcase_prepared = scenario.give_new_cases(casebase_prepared, [newcase])
    result = scenario.give_predictions(casebase_prepared, newcase_prepared)
    prediction = result[1][0]["Prediction"]
    assert prediction == newcase_spec["outcome_expected"][aacbr_type], f"Failed for {newcase_spec}, in type {aacbr_type}" f"Failed on test {test}"

def test_files_non_cautious(setup):
  run_test_with_aacbr_type("non_cautious", setup)

@pytest.mark.skip(reason="Cautious is currently bugged.")
def test_files_cautious(setup):
  run_test_with_aacbr_type("cautious", setup)
