### Tests the model/classifier itself, predictions, argumentation framework, etc

from context import aacbr
import pytest
from itertools import product

@pytest.fixture(autouse=True)
def test_import():
  from aacbr import Aacbr, Case
  pass

from aacbr import Aacbr, Case

@pytest.mark.usefixtures("test_import")
class TestAacbr:
  default = Case('default', set(), outcome=0)
  case1 = Case('1', {'a'}, outcome=1)
  case2 = Case('2', {'a','b'}, outcome=0)
  example_cb = (default, case1, case2)
  
  case3 = Case('3', {'a','b','c'}, outcome=0)
  example_cb2 = tuple(list(example_cb) + [case3])
  example_cbs = [example_cb, example_cb2]
    
  @pytest.mark.parametrize("cb", example_cbs)
  def test_initialisation(self, cb):
    clf = Aacbr(cb)
    assert isinstance(clf, Aacbr)

  @pytest.mark.parametrize("cb", example_cbs)
  def test_aacbr_methods(self, cb):
    clf = Aacbr(cb)
    assert clf.casebase == cb
    
  # def test_attack(self):
  #   default = Case('default', set(), outcome=0)
  #   case1 = Case('1', {'a'}, outcome=1)
  #   case2 = Case('2', {'a','b'}, outcome=0)
  #   case3 = Case('3', {'a','b','c'}, outcome=0)
  #   cb = (default, case1, case2)
  #   clf = Aacbr(cb)
  #   list_of_attacks = ((case1, default), (case2,case1))
  #   for pair in product(cb, repeat=2):
  #     assert clf.attacks(pair[0],pair[1]) == pair in list_of_attacks
      
  #   cb = (default, case1, case2, case3)
  #   clf = Aacbr(cb)
  #   for pair in product(cb, repeat=2):
  #     assert clf.attacks(pair[0],pair[1]) == pair in list_of_attacks

  @pytest.mark.parametrize("cb", example_cbs)
  def test_attack(self, cb):
    clf = Aacbr(cb)
    if cb == self.example_cb or cb == self.example_cb2:
      list_of_attacks = ((self.case1, self.default), (self.case2, self.case1))
    else:
      raise(Exception("Undefined test"))
    for pair in product(cb, repeat=2):
      assert clf.attacks(pair[0],pair[1]) == pair in list_of_attacks   
      
  @pytest.mark.skip(reason="Undefined tests")      
  def test_argumentation_framework():
    pass
  @pytest.mark.skip(reason="Undefined tests")
  def test_predictions():
    pass
  @pytest.mark.skip(reason="Undefined tests")
  def test_grounded_extension():
    pass
  @pytest.mark.skip(reason="Undefined tests")
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
class TestCaacbr:
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
