### Tests the model/classifier itself, predictions, argumentation framework, etc
import pytest
from itertools import product

@pytest.fixture(autouse=True)
def test_import():
  from aacbr import Aacbr, Case
  pass

from aacbr import Aacbr, Case
from aacbr.cases import load_cases

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
    clf = Aacbr().fit(cb)
    assert isinstance(clf, Aacbr)

  @pytest.mark.parametrize("cb", example_cbs)
  def test_aacbr_methods(self, cb):
    clf = Aacbr().fit(cb)
    assert clf.casebase_initial == cb
    
  # def test_attack(self):
  #   default = Case('default', set(), outcome=0)
  #   case1 = Case('1', {'a'}, outcome=1)
  #   case2 = Case('2', {'a','b'}, outcome=0)
  #   case3 = Case('3', {'a','b','c'}, outcome=0)
  #   cb = (default, case1, case2)
  #   clf = Aacbr(cb)
  #   list_of_attacks = ((case1, default), (case2,case1))
  #   for pair in product(cb, repeat=2):
  #     assert clf.attacks(pair[0],pair[1]) == (pair in list_of_attacks)
      
  #   cb = (default, case1, case2, case3)
  #   clf = Aacbr(cb)
  #   for pair in product(cb, repeat=2):
  #     assert clf.attacks(pair[0],pair[1]) == (pair in list_of_attacks)

  @pytest.mark.parametrize("cb", example_cbs)
  def test_attack(self, cb):
    clf = Aacbr()
    clf.fit(cb)
    if cb == self.example_cb or cb == self.example_cb2:
      list_of_attacks = ((self.case1, self.default), (self.case2, self.case1))
    else:
      raise(Exception("Undefined test"))
    for pair in product(cb, repeat=2):
      assert ((clf.past_case_attacks(pair[0],pair[1])) == (pair in list_of_attacks)), f"Violated by pair {pair}. Expected {pair in list_of_attacks}."

  @pytest.mark.parametrize("cb", example_cbs)
  def test_attack_new_case(self, cb):
    new = Case('new', {'a', 'd'})
    clf = Aacbr()
    clf.fit(cb)
    assert not clf.new_case_attacks(new, self.default)
    if self.case1 in cb:
      assert not clf.new_case_attacks(new, self.case1)
    if self.case2 in cb:
      assert clf.new_case_attacks(new, self.case2)
    if self.case3 in cb:
      assert clf.new_case_attacks(new, self.case3)

  def test_conciseness(self):
    cb = self.example_cb2
    clf = Aacbr()
    clf.fit(cb)
    assert clf.past_case_attacks(self.case2, self.case1)
    assert not clf.past_case_attacks(self.case3, self.case1), "case3 is attacking case1 even if case2 already does so. Violating conciseness."

  def test_inconsistent(self):
    # Even if already tested in test_cases.py, Aacbr should have its
    # own interface to it.
    case1 = Case('1', {'a','b'}, outcome=0)
    case2 = Case('2', {'a','b'}, outcome=1)
    cb = [case1, case2]
    clf = Aacbr().fit(cb)
    assert clf.past_case_attacks(case1, case2)
    assert clf.past_case_attacks(case2, case1)
    assert clf.inconsistent_attacks(case1, case2)
    assert clf.inconsistent_attacks(case2, case1)
      
  @pytest.mark.skip(reason="For visualising what is happening with AF, but not a proper test.") # this is because this AF implementation is to be changed
  def test_argumentation_framework(self):
    cb = self.example_cb
    # newcase = self.case3
    newcase = Case('4', {'a', 'd'}, outcome=1)
    expected_output = newcase.outcome
    clf = Aacbr().fit(cb)
    aa_framework, format_mapping = clf.format_aaframework(clf.casebase_active, newcase)    
    result_string = f"{aa_framework}\n{format_mapping}"
    output = clf.predict([newcase])
    result_string += f"\n{output}, {expected_output}"
    raise Exception(result_string)
  
  @pytest.mark.skip(reason="Undefined tests -- see integration tests such as 'test_files_non_cautious'")
  def test_predictions():
    pass
  @pytest.mark.skip(reason="Undefined tests -- see integration tests such as 'test_files_non_cautious'")
  def test_grounded_extension():
    pass

  def test_scikit_learning_like_api_with_case_input(self):
    # It would be nice to have it compatible with the scikit-learn API:
    # https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects
    train_data = self.example_cb2
    test_data = [Case('new1', {'a'}),
                 Case('new2', {'a', 'b'}),
                 Case('new3', {'a', 'c'}),
                 Case('new4', {'a', 'b', 'c', 'd'}),
                 Case('new5', set())]    
    expected_output = [1, 0, 1, 0, 0]
    clf = Aacbr()
    predicted_output = clf.fit(train_data).predict(test_data)
    assert expected_output == predicted_output

  def test_scikit_learning_like_api_with_case_input_cautious(self):
    default = Case('default', set(), outcome=0)
    case1 = Case('1', {'a'}, outcome=1)
    case2 = Case('2', {'a','b'}, outcome=0)
    case3 = Case('3', {'c'}, outcome=1)
    case4 = Case('4', {'c','d'}, outcome=0)
    case5 = Case('5', {'a','b','c'}, outcome=1)
    cb = [default, case1, case2, case3, case4, case5]
    train_data = cb
    test_data = [Case('new1', {'a'}),
                 Case('new2', {'a', 'b'}),
                 Case('new3', {'a', 'c'}),
                 Case('new4', {'a', 'b', 'c', 'd'}),
                 Case('new5', set()),
                 Case('new6', {'a','c','d'})]
    expected_output = [1, 0, 1, 0, 0, 1]
    clf = Aacbr(cautious=True)
    predicted_output = clf.fit(train_data).predict(test_data)
    assert expected_output == predicted_output
    #
    clf_noncautious = Aacbr(cautious=False)
    expected_output = [1, 0, 1, 1, 0, 1]
    predicted_output = clf_noncautious.fit(train_data).predict(test_data)
    assert expected_output == predicted_output

  def test_inconsistent_IO(self):
    default = Case('default', set(), outcome=0)
    case1 = Case('1', {'a'}, outcome=1)
    case2 = Case('2', {'a'}, outcome=0)
    case3 = Case('3', {'a', 'b'}, outcome=0)
    case4 = Case('4', {'a', 'b'}, outcome=1)
    cb = [default, case1, case2, case3, case4]
    train_data = cb
    test_data = [Case('new1', {'a'}),
                 Case('new2', {'a', 'b'}),
                 Case('new3', {'a', 'c'}),
                 Case('new4', {'a', 'b', 'c', 'd'}),
                 Case('new5', set()),
                 Case('new6', {'a','c','d'})]
    expected_output = [1, 1, 1, 1, 0, 1]
    clf = Aacbr(cautious=False)
    predicted_output = clf.fit(train_data).predict(test_data)
    assert expected_output == predicted_output

  def test_inconsistent_cautious_IO(self):
    default = Case('default', set(), outcome=0)
    case1 = Case('1', {'a'}, outcome=1)
    case2 = Case('2', {'a'}, outcome=0)
    case3 = Case('3', {'a', 'b'}, outcome=0)
    case4 = Case('4', {'a', 'b'}, outcome=1)
    cb = [default, case1, case2, case3, case4]
    train_data = cb
    test_data = [Case('new1', {'a'}),
                 Case('new2', {'a', 'b'}),
                 Case('new3', {'a', 'c'}),
                 Case('new4', {'a', 'b', 'c', 'd'}),
                 Case('new5', set()),
                 Case('new6', {'a','c','d'})]
    expected_output = [1, 0, 1, 0, 0, 1]
    clf = Aacbr(cautious=True)
    predicted_output = clf.fit(train_data).predict(test_data)
    assert expected_output == predicted_output    

  @pytest.mark.skip(reason="not implemented")
  def test_scikit_learning_like_api_with_characterisation_input(self):
    train_data = self.example_cb2
    train_X = [c.factors for c in cb]
    train_Y = [c.outcome for c in cb]
    test_data = [Case('new1', {'a'}),
                 Case('new2', {'a', 'b'}),
                 Case('new3', {'a', 'c'}),
                 Case('new4', {'a', 'b', 'c', 'd'}),
                 Case('new5', set())]
    test_X = [c.factors for c in test_data]
    expected_output = [1, 0, 1, 0, 0]
    clf = Aacbr()
    predicted_output = clf.fit(train_X, train_Y).predict(test_X)
    assert expected_output == predicted_output

@pytest.mark.skip(reason="Undefined tests")
@pytest.mark.usefixtures("test_import")
class TestCaacbr:
  # TODO: perhaps delete this class, pass as parameter
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

#### json-defined tests
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

def run_test_from_files(aacbr_type, test):  
  # TODO: change this interface in line below in the future
  # -- it should not be inside Aacbr
  cautious = True if aacbr_type == "cautious" else False
  casebase = load_cases(TEST_PATH_PREFIX + test["casebase"])
  clf = Aacbr(outcome_def=test["outcomes"]["default"],
              outcome_nondef=test["outcomes"]["nondefault"],
              outcome_unknown=test["outcomes"]["undecided"],
              cautious=cautious)
  clf.fit(casebase)
  casebase_active_ids = set(map(lambda x: getattr(x, "id"), clf.casebase_active))
  assert set(casebase_active_ids) == set(test["casebase_expected"][aacbr_type])

  for newcase_spec in test["newcases"]:
    newcase = Case(id=newcase_spec["id"], factors=set(newcase_spec["factors"]))
    result = clf.predict([newcase])
    # prediction = result[1][0]["Prediction"]
    prediction = result[0]
    assert prediction == newcase_spec["outcome_expected"][aacbr_type], f"Failed for {newcase_spec}, in type {aacbr_type}" f"Failed on test {test}"

# @pytest.mark.xfail(reason="New interface not yet implemented.")
@pytest.mark.parametrize("test", TESTS)
def test_files_non_cautious(test):
  run_test_from_files("non_cautious", test)

# @pytest.mark.xfail(reason="Cautious is currently bugged.")
@pytest.mark.parametrize("test", TESTS)
def test_files_cautious(test):
  run_test_from_files("cautious", test)

@pytest.fixture(params=TESTS)
def setup(request):
  test = request.param
  setup_result = Aacbr(test["outcomes"]["default"], test["outcomes"]["nondefault"], test["outcomes"]["undecided"])
  return test, setup_result  

def run_test_from_files_old_interface(aacbr_type, setup):
  """Kept for reference, not expected to work.
  """
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

@pytest.mark.skip(reason="Deprecated interface.")
def test_files_non_cautious_old(setup):
  run_test_from_files_old_interface("non_cautious", setup)

@pytest.mark.skip(reason="Deprecated interface.")
def test_files_cautious_old(setup):
  run_test_from_files_old_interface("cautious", setup)
