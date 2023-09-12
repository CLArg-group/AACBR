### Tests the model/classifier itself, predictions, argumentation framework, etc
import pytest
import pickle
from pathlib import Path
from itertools import product
from collections.abc import Sequence
from random import random, randint, shuffle

from logging import debug, info, warning, error

@pytest.fixture(autouse=True)
def test_import():
  from aacbr import Aacbr, Case
  pass

from aacbr import Aacbr, ArbitratedDisputeTree, Case
from aacbr.aacbr import tsorted
from aacbr.cases import load_cases
from aacbr.argumentation import compute_grounded

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
    assert clf.casebase_initial_ == cb
    
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

  def test_default_case_implict(self):
    case1 = Case('1', {'a','b'}, outcome=0)
    case2 = Case('2', {'a','b'}, outcome=1)
    cb = [case1, case2]
    clf = Aacbr().fit(cb)
    assert clf.default_case == Case("default", set(), clf.outcome_def)

  def test_default_case_different_outcome(self):
    default = Case('default', set(), outcome=1)
    case1 = Case('1', {'a'}, outcome=0)
    case2 = Case('2', {'a', 'b'}, outcome=1)
    case3 = Case('3', {'a', 'b'}, outcome=0)
    cb = [default, case1, case2, case3]
    clf = Aacbr()
    assert clf.outcome_def == 0
    assert clf.outcome_nondef == 1
    clf.fit(cb)
    assert clf.default_case == Case("default", set(), clf.outcome_def)
    assert clf.outcome_def == 1
    assert clf.outcome_nondef == 0
    test_data = [Case('new', {'a', 'c'}),
                Case('new2', {'a', 'b'})]
    expected_output = [0, 0]
    predicted_output = clf.predict(test_data)
    assert expected_output == predicted_output
    
  def test_default_case_different_outcome_cautious(self):
    default = Case('default', set(), outcome=1)
    case1 = Case('1', {'a'}, outcome=0)
    case2 = Case('2', {'a', 'b'}, outcome=1)
    case3 = Case('3', {'a', 'b'}, outcome=0)
    cb = [default, case1, case2, case3]
    clf = Aacbr(cautious=True)
    assert clf.outcome_def == 0
    assert clf.outcome_nondef == 1
    clf.fit(cb)
    assert clf.default_case == Case("default", set(), clf.outcome_def)
    assert clf.outcome_def == 1
    assert clf.outcome_nondef == 0
    test_data = [Case('new', {'a', 'c'}),
                Case('new2', {'a', 'b'})]
    expected_output = [0, 1]
    predicted_output = clf.predict(test_data)
    assert expected_output == predicted_output
    
  def test_cautious_basic_example(self):
    default = Case('default', {'a'}, outcome=0)
    case1 = Case('1', {'a','b'}, outcome=0)
    case2 = Case('2', {'a','b'}, outcome=1)
    cb = [default, case1, case2]
    clf = Aacbr(cautious=True).fit(cb)
    assert clf.casebase_active_ == [default, case2]
    
  def test_default_case_in_casebase(self):
    default = Case('default', {'a'}, outcome=0)
    case1 = Case('1', {'a','b'}, outcome=0)
    case2 = Case('2', {'a','b'}, outcome=1)
    cb = [default, case1, case2]
    clf = Aacbr().fit(cb)
    assert clf.default_case == default

  def test_default_case_in_arguments(self):
    default = Case('default', {'a'}, outcome=0)
    case1 = Case('1', {'a','b'}, outcome=0)
    case2 = Case('2', {'a','b'}, outcome=1)
    cb = [case1, case2]
    clf = Aacbr(default_case=default).fit(cb)
    assert clf.default_case == default    
  
  def test_inconsistent(self):
    # Even if already tested in test_cases.py, Aacbr should have its
    # own interface to it.
    case1 = Case('1', {'a','b'}, outcome=0)
    case2 = Case('2', {'a','b'}, outcome=1)
    case3 = Case('3', {'a','b'}, outcome=0)
    cb = [case1, case2, case3]
    clf = Aacbr().fit(cb)
    assert clf.past_case_attacks(case1, case2)
    assert clf.past_case_attacks(case2, case1)
    assert clf.inconsistent_pair(case1, case2)
    assert clf.inconsistent_pair(case2, case1)
    assert set([case2]) == set(clf.attacked_by_[case1])
    assert set([case2]) == set(clf.attacked_by_[case3])
    assert set([case1,case3,clf.default_case]) == set(clf.attacked_by_[case2])

  @pytest.mark.parametrize("remove_spikes", (False, True))
  def test_inconsistent_with_default(self, remove_spikes):
    default = Case('default', set(), outcome=0)
    case1 = Case('1', set(), outcome=1)
    case2 = Case('2', {'a'}, outcome=1)
    case3 = Case('3', {'a','b'}, outcome=0)
    cb = [default, case1, case2, case3]
    clf = Aacbr(remove_spikes=remove_spikes).fit(cb)
    assert clf.default_case == default
    assert set(clf.attacked_by_[default]) == set()
    assert set(clf.attacked_by_[case1]) == set([default])
    assert set(clf.attacked_by_[case2]) == set([default])
    assert set(clf.attacked_by_[case3]) == set([case1,case2])
    
  # DONE: make default not attack anyone. For now, intended behaviour.
  # @pytest.mark.xfail(reason="Current bug: inconsistent with default not added.")
  def test_inconsistent_with_default_cautious(self):
    default = Case('default', set(), outcome=0)
    case1 = Case('1', set(), outcome=1)
    case2 = Case('2', {'a'}, outcome=1)
    case3 = Case('3', {'a','b'}, outcome=0)
    cb = [default, case1, case2, case3]
    clf = Aacbr(cautious=True).fit(cb)
    assert clf.default_case == default
    assert set(clf.attacked_by_[default]) == set()
    assert set(clf.attacked_by_[case1]) == set([default])
    assert set(clf.attacked_by_[case2]) == set([])
    assert set(clf.attacked_by_[case3]) == set([case1])
    assert set(clf.casebase_active_) == set([default, case1,case3])

  def test_depth_calculation(self):
    default = Case('default', set(), outcome=0)
    case1 = Case('1', {'a'}, outcome=1)
    case2 = Case('2', {'b'}, outcome=1)
    case3 = Case('3', {'a','a2'}, outcome=0)
    case4 = Case('4', {'a','a2','a3'}, outcome=1)
    case5 = Case('5', {'a','a2','a3','b'}, outcome=0)
    case6 = Case('6', {'d'}, outcome=0)
    cb = [default, case1, case2, case3, case4, case5, case6]
    clf = Aacbr(cautious=False, remove_spikes=True).fit(cb)
    assert clf.default_case == default
    expected_depths = {
      default: 0,
      case1: 1,
      case2: 1,
      case3: 2,
      case4: 3,
      case5: 2,
      # case6: None
      }
    expected_stats = {"maximum": 3,
                      "average": round(1.5, ndigits=4),
                      "stdev (population)": round(0.95742, ndigits=4)}
    calculated = clf._framework_depth_statistics()
    stats = ["maximum", "average", "stdev (population)"]
    assert clf.get_casebase_depths() == expected_depths
    for stat in stats:
      assert pytest.approx(calculated[stat], 0.001) == expected_stats[stat]
    
    
  def test_argumentation_framework(self):
    cb = self.example_cb
    # newcase = self.case3
    newcase = Case('4', {'a', 'd'}, outcome=1)
    expected_output = newcase.outcome
    clf = Aacbr().fit(cb)
    framework = clf.give_argumentation_framework(newcase)
    arguments, attacks = framework
    assert arguments == set(cb + (newcase,))
    expected_attacks = \
      {(self.case2, self.case1),
       (self.case1, self.default)}
    assert attacks == expected_attacks
  
  def test_argumentation_grounded_extension(self):
    """For inner working of grounded extension
    """
    default = Case('default', set(), outcome=0)
    case1 = Case('1', {'a'}, outcome=1)
    case2 = Case('2', {'a','b'}, outcome=0)
    case3 = Case('3', {'a','b','c'}, outcome=0)
    new1 = Case('new1', {'a', 'b'})
    
    arguments = {default, case1, case2, case3, new1}
    attacks = {(case1, default),
               (case2, case1),
               (new1, case3)}

    labels = compute_grounded(arguments, attacks)
    assert labels['in'] == {case2, new1, default}
    assert labels['out'] == {case3, case1}
    assert labels['undec'] == set()

    new2 = Case('new2', {'a'})
    arguments = {default, case1, case2, case3, new2}
    attacks = {(case1, default),
               (case2, case1),
               (new2, case2),
               (new2, case3)}
    labels = compute_grounded(arguments, attacks)
    assert labels['in'] == {case1, new2}
    assert labels['out'] == {default, case2, case3}
    assert labels['undec'] == set()
    pass
  
  def test_grounded_extension(self):
    default = Case('default', set(), outcome=0)
    case1 = Case('1', {'a'}, outcome=1)
    case2 = Case('2', {'a','b'}, outcome=0)
    case3 = Case('3', {'a','b','c'}, outcome=0)
    case4 = Case('4', {'c'}, outcome=1)
    example_cb = (default, case1, case2, case3, case4)

    new = Case('new', {'a', 'c'})
    new2 = Case('new2', {'a', 'b'})
    ge = {case1, case4, new}
    ge2 = {case2, default, new2}
    
    clf = Aacbr()
    clf.fit(example_cb)
    assert clf.grounded_extension(new_case=new) == ge
    assert clf.grounded_extension(new_case=new2) == ge2


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

  @pytest.mark.xfail(reason="Currently incompatible.")
  def test_scikit_learn_check_estimator(self):
    from sklearn.utils.estimator_checks import check_estimator
    assert check_estimator(Aacbr())    
    
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
    case3 = Case('3', {'a', 'b'}, outcome=1)
    case4 = Case('4', {'a', 'b'}, outcome=0)
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
    assert set(clf.casebase_active_) == set([default, case1, case4])
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
    assert set(clf.casebase_active_) == set([default, case1, case2, case3, case4])
    #
    clf_noncautious = Aacbr(cautious=False)
    expected_output = [1, 0, 1, 1, 0, 1]
    predicted_output = clf_noncautious.fit(train_data).predict(test_data)
    assert expected_output == predicted_output, "Non-cautious is not giving expected result!"

    
  # @pytest.mark.xfail(reason="not implemented")
  def test_scikit_learn_like_api_with_characterisation_input(self):
    case0 = Case("0", set(), outcome=0)
    cb = [case0, self.case1, self.case2, self.case3]
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
    clf.fit(train_X, train_Y)
    assert set(clf.casebase_active_) == set(cb + [self.default])
    predicted_output = clf.predict(test_X)
    assert expected_output == predicted_output

  def test_scikit_learn_like_api_with_characterisation_input2(self):
    train_X = [set(),
               {'a'},
               {'a','b'},
               {'a','b','c'}]
    train_Y = [0,
               1,
               0,
               0]
    test_X = [{'a'},
              {'a', 'b'},
              {'a', 'c'},
              {'a', 'b', 'c', 'd'},
              set()]
    expected_output = [1, 0, 1, 0, 0]
    clf = Aacbr()
    clf.fit(train_X, train_Y)
    
    default = Case('default', set(), outcome=0)
    case0 = Case("0", set(), outcome=0)
    case1 = Case('1', {'a'}, outcome=1)
    case2 = Case('2', {'a','b'}, outcome=0)
    case3 = Case('3', {'a','b','c'}, outcome=0)
    cb = [case0, case1, case2, case3]
    
    assert set(clf.casebase_active_) == set(cb + [default])
    
    predicted_output = clf.predict(test_X)
    assert expected_output == predicted_output
    
  def test_remove_spikes(self):
    default = Case('default', set(), outcome=0)
    case1 = Case('1', {'a'}, outcome=1)
    case2 = Case('2', {'a','b'}, outcome=0)
    case3 = Case('3', {'b'}, outcome=0)
    case4 = Case('4', {'c'}, outcome=0)
    case5 = Case('5', {'a','c'}, outcome=1)
    case6 = Case('6', {'a','b','c'}, outcome=0)
    cb = (default, case1, case2, case3, case4, case5, case6)
    filtered_cb = {default, case1, case2}
    clf = Aacbr(remove_spikes=True).fit(cb)
    assert set(clf.casebase_active_) == filtered_cb
    
  @pytest.mark.parametrize("remove_spikes", (False, True))
  @pytest.mark.parametrize("cautious", (False, True))
  def test_sort(self, remove_spikes, cautious):
    if cautious and remove_spikes:
      pytest.skip()
    default = Case('default', set(), outcome=0)
    case1 = Case('1', {'a'}, outcome=1)
    case2 = Case('2', {'a','b'}, outcome=0)
    case3 = Case('3', {'b'}, outcome=0)
    case4 = Case('4', {'c'}, outcome=0)
    case5 = Case('5', {'a','c'}, outcome=1)
    case6 = Case('6', {'a','b','c'}, outcome=0)
    cb = [default, case1, case2, case3, case4, case5, case6]
    shuffle(cb)
    sorted_cb = tsorted(cb)
    for idx,case in enumerate(sorted_cb):
      for idx2,othercase in enumerate(sorted_cb[idx:]):
        assert not case > othercase, (idx, idx2)

  @pytest.mark.parametrize("remove_spikes", (False, True))
  @pytest.mark.parametrize("cautious", (False, True))
  def test_sort_with_cycles(self, remove_spikes, cautious):
    if cautious and remove_spikes:
      pytest.skip()
    default = Case('default', set(), outcome=0)
    case1 = Case('1', {'a'}, outcome=1)
    case2 = Case('2', {'a','b'}, outcome=0)
    case3 = Case('2', {'a','b'}, outcome=0)
    case4 = Case('4', {'c'}, outcome=0)
    case5 = Case('5', {'a','c'}, outcome=1)
    case6 = Case('6', {'a','b','c'}, outcome=0)
    cb = [default, case1, case2, case3, case4, case5, case6]
    shuffle(cb)
    sorted_cb = tsorted(cb)
    for idx,case in enumerate(sorted_cb):
      for idx2,othercase in enumerate(sorted_cb[idx:]):
        assert not case > othercase, (idx, idx2)

  @pytest.mark.parametrize("remove_spikes", (False, True))
  @pytest.mark.parametrize("cautious", (False, True))
  def test_sort_larger_cb(self, remove_spikes, cautious):
    if cautious and remove_spikes:
      pytest.skip()
    cb = generate_orderedsequence_casebase(n=100, dim=3)
    shuffle(cb)
    sorted_cb = tsorted(cb)
    for idx,case in enumerate(sorted_cb):
      for idx2,othercase in enumerate(sorted_cb[idx:]):
        assert not case > othercase, (idx, idx2)

  def test_remove_spikes_larger_cb(self):
    cb = generate_orderedsequence_casebase(n=100, dim=5)
    clf = Aacbr(remove_spikes=True, cautious=False).fit(cb)
    non_attacking_cases = [case for case in clf.casebase_active_
                           if clf.attacked_by_[case] == []]
    assert clf.outcome_def == clf.default_case.outcome
    assert non_attacking_cases == [clf.default_case]
    
  @pytest.mark.parametrize("remove_spikes", (False, True))
  @pytest.mark.parametrize("cautious", (False, True))
  def test_attacked_by_attackers_of_consistency(self, remove_spikes, cautious):
    if cautious and remove_spikes:
      pytest.skip()
    cb = generate_orderedsequence_casebase(n=50, dim=3)
    clf = Aacbr(remove_spikes=remove_spikes, cautious=cautious).fit(cb)
    assert set(clf.attacked_by_.keys()).issubset(set(clf.casebase_active_))
    assert set(clf.attackers_of_.keys()).issubset(set(clf.casebase_active_))
    # assert set(clf.attacked_by_.keys()) == set(clf.attackers_of_.keys()) # not necessarily true, since it is a defaultdict
    for case in clf.attacked_by_.keys():
      for othercase in clf.attacked_by_[case]:
        assert case in clf.attackers_of_[othercase]
    for case in clf.attackers_of_.keys():
      for othercase in clf.attackers_of_[case]:
        assert case in clf.attacked_by_[othercase]
    
  def test_alternative_partial_order(self):
    class OrderedPair:
      """Pair (a,b) where (a,b) are natural numbers.
      Partial order is defined by (a,b) <= (c,d) iff a<=c and b<=d."""

      def __init__(self, x, y):
        self.x: int = x
        self.y: int = y

      def __eq__(self, other):
        return self.x == other.x and self.y == other.y
      def __le__(self, other):
        return self.x <= other.x and self.y <= other.y
      def __hash__(self):
        return hash((self.x, self.y))

    default = Case('default', OrderedPair(0,0), outcome=0)
    case1 = Case('1', OrderedPair(1,0), outcome=1)
    case2 = Case('2', OrderedPair(0,1), outcome=0)
    case3 = Case('3', OrderedPair(2,1), outcome=0)
    cb = (case1, case2, case3)
    clf = Aacbr(default_case=default)
    clf.fit(cb)
    assert set(clf.casebase_active_) == set(cb + (default,))
    test = [OrderedPair(2,0),
            OrderedPair(0,2),
            OrderedPair(20,20),
            OrderedPair(1,1),
            OrderedPair(0,0)]
    expected_output = [1, 0, 0, 1, 0]
    predictions = clf.predict(test)
    assert expected_output == predictions
    
  def test_graph_drawing(self, tmp_path):
    "Checks if a graph is created"
    cb = self.example_cb2
    clf = Aacbr().fit(cb)
    clf.draw_graph(output_dir = tmp_path)
    output_path = tmp_path / "graph.png"
    info(f"Graph in {output_path=}")
    assert output_path.exists()
    assert output_path.is_file()
    
  def test_graph_drawing_invalid_input(self, tmp_path):
    "Tests wrong arguments for graph drawing"
    cb = self.example_cb2
    clf = Aacbr().fit(cb)
    with pytest.raises(Exception) as exc_info:
      clf.draw_graph(new_case=tmp_path)
      # info(f"Graph in {output_path=}")
      # assert output_path.exists()
      # assert output_path.is_file()
    info(f"Successfully raised exception: {exc_info.value}")
    clf.draw_graph(tmp_path)
    output_path = tmp_path / "graph.png"
    assert output_path.exists()
    assert output_path.is_file()
    
  
  @pytest.mark.parametrize("engine", ["networkx", "graphviz"])
  def test_graph_drawing_engines(self, tmp_path, engine):
   """Checks if a graph is created using two different engines:
    networkx and graphviz.
   """
   cb = self.example_cb2
   clf = Aacbr().fit(cb)
   clf.draw_graph(output_dir = tmp_path, engine=engine)
   output_path = tmp_path / "graph.png"
   info(f"Graph in {output_path=}")
   assert output_path.exists()
   assert output_path.is_file()
   
  @pytest.mark.parametrize("node_formatter",
                           [None, "full",
                            "id", "factors",
                            lambda x: str(x),
                            lambda x: str(x.outcome),
                            lambda x: str(x.id),
                            lambda x: str(x.factors),
                            lambda x: str(x).upper()])
  def test_graph_drawing_node_formatters(self, tmp_path, node_formatter):
   """Checks if a graph is created using different node_formatters.
   """
   cb = self.example_cb2
   clf = Aacbr().fit(cb)
   clf.draw_graph(output_dir=tmp_path, engine="graphviz",
                  node_formatter=node_formatter)
   output_path = tmp_path / "graph.png"
   info(f"Graph in {output_path=}")
   assert output_path.exists()
   assert output_path.is_file()
    
  def test_if_pickable(self, tmp_path):
    "Checks if aacbr is pickable."
    cb = self.example_cb2
    clf = Aacbr().fit(cb)
    output_path = tmp_path / "aacbr_object"
    output_path.write_bytes(pickle.dumps(clf))
    info(f"Pickle in {output_path=}")
    assert output_path.exists()
    assert output_path.is_file()
    loaded = pickle.loads(output_path.read_bytes())
    assert set(clf.casebase_active_) == set(loaded.casebase_active_)

class TestDisputeTrees:
  def test_basic_dispute_tree(self):
    default = Case('default', set(), outcome=0)
    case1 = Case('1', {'a'}, outcome=1)
    case2 = Case('2', {'b'}, outcome=1)
    case3 = Case('3', {'a', 'b','c'}, outcome=0)
    cb = [default, case1, case2, case3]
    clf = Aacbr()
    clf.fit(cb)
    test_data = [Case('new', {'a', 'c'})]
    expected_output = [1]
    predicted_output = clf.predict(test_data)
    assert expected_output == predicted_output
    adt = clf.adt_explain(test_data[0])
    assert type(adt) is ArbitratedDisputeTree
    debug(f"{adt.nodes=}\n{adt.edges=}")
    pass
  
  def test_incoherent_af_dispute_tree(self):
    default = Case('default', set(), outcome=0)
    case1 = Case('1', {'a'}, outcome=1)
    case2 = Case('2', {'a', 'b'}, outcome=0)
    case3 = Case('3', {'a', 'b'}, outcome=1)
    cb = [default, case1, case2, case3]
    clf = Aacbr()
    clf.fit(cb)
    test_data = [Case('new', {'a', 'b', 'c'})]
    expected_output = [1]
    predicted_output = clf.predict(test_data)
    assert expected_output == predicted_output
    adt = clf.adt_explain(test_data[0])
    assert adt is None

  def test_acceptable_incoherent_af1(self):
    default = Case('default', set(), outcome=0)
    case1 = Case('1', {'a'}, outcome=1)
    case2 = Case('2', {'a', 'b'}, outcome=0)
    case3 = Case('3', {'a', 'b'}, outcome=1)
    case4 = Case('4', {'a', 'b', 'c'}, outcome=0)
    cb = [default, case1, case2, case3, case4]
    clf = Aacbr()
    clf.fit(cb)
    test_data = [Case('new', {'a', 'b', 'c'})]
    expected_output = [0]
    predicted_output = clf.predict(test_data)
    assert expected_output == predicted_output
    adt = clf.adt_explain(test_data[0])
    debug(f"{adt.nodes=}\n{adt.edges=}")
    assert type(adt) is ArbitratedDisputeTree

  def test_acceptable_incoherent_af2(self):
    default = Case('default', set(), outcome=0)
    case1 = Case('1', {'a'}, outcome=1)
    case2 = Case('2', {'a'}, outcome=0)
    case3 = Case('3', {'a', 'b'}, outcome=0)
    cb = [default, case1, case2, case3]
    clf = Aacbr()
    clf.fit(cb)
    test_data = [Case('new', {'a', 'b', 'c'})]
    expected_output = [0]
    predicted_output = clf.predict(test_data)
    assert expected_output == predicted_output
    adt = clf.adt_explain(test_data[0])
    debug(f"{adt.nodes=}\n{adt.edges=}")
    assert type(adt) is ArbitratedDisputeTree

  def test_branch_selection(self):
    default = Case('default', set(), outcome=0)
    case1 = Case('1', {'a'}, outcome=1)
    case2 = Case('2', {'b'}, outcome=1)
    case3 = Case('3', {'a', 'b'}, outcome=0)
    case4 = Case('4', {'b', 'd'}, outcome=0)
    case5 = Case('5', {'b', 'd', 'e'}, outcome=1)
    case6 = Case('6', {'b', 'd', 'e', 'f'}, outcome=0)
    case7 = Case('7', {'b', 'd', 'e', 'f', 'g'}, outcome=1)
    case8 = Case('8', {'b', 'd', 'e', 'f', 'g', 'h'}, outcome=0)
    cb = [default, case1, case2, case4, case5, case6, case7, case8, case3]
    clf = Aacbr()
    clf.fit(cb)
    test_case = Case('new', {'a', 'b', 'c'})
    expected_output = [0]
    predicted_output = clf.predict([test_case])
    assert expected_output == predicted_output
    
    adt = clf.adt_explain(test_case)
    debug(f"{adt.nodes=}\n{adt.edges=}")
    assert type(adt) is ArbitratedDisputeTree
    assert case4 not in adt.get_winning_cases()
    assert case4 not in adt.get_losing_cases()
    pass

  def test_branch_selection2(self):
    default = Case('default', set(), outcome=0)
    case1 = Case('1', {'a'}, outcome=1)
    case2 = Case('2', {'b'}, outcome=1)
    case3 = Case('3', {'c'}, outcome=1)
    othercases1 = ([
      Case('a1', {'a', 'a1'}, outcome=0),
      Case('a2', {'a', 'a2'}, outcome=0),
      Case('a3', {'a', 'a3'}, outcome=0),
      Case('b1', {'b', 'b1'}, outcome=0),
      Case('b2', {'b', 'b2'}, outcome=0),
      ])
    caseb3 = Case('b3', {'b', 'b3'}, outcome=0)
    othercases2 = ([
      Case('c1', {'c', 'c1'}, outcome=0),
      Case('c2', {'c', 'c2'}, outcome=0),
      Case('c3', {'c', 'c3'}, outcome=0),
      Case('a11', {'a', 'a1', 'a11'}, outcome=1),
      Case('a11', {'a', 'a1', 'a12'}, outcome=1),
      Case('a11', {'a', 'a1', 'a13'}, outcome=1),
      Case('a224', {'a', 'a2', 'a24'}, outcome=1),
      Case('a225', {'a', 'a2', 'a25'}, outcome=1),
      Case('a226', {'a', 'a2', 'a26'}, outcome=1),
      Case('a37', {'a', 'a3', 'a37'}, outcome=1),
      Case('a38', {'a', 'a3', 'a38'}, outcome=1),
      Case('a39', {'a', 'a3', 'a39'}, outcome=1),
      Case('b110', {'b', 'b1', 'b110'}, outcome=1),
      Case('b111', {'b', 'b1', 'b111'}, outcome=1),
      Case('b112', {'b', 'b1', 'b112'}, outcome=1),
      Case('b213', {'b', 'b2', 'b213'}, outcome=1),
      Case('b214', {'b', 'b2', 'b214'}, outcome=1),
      Case('b215', {'b', 'b2', 'b215'}, outcome=1),
    ])
    caseb313 = Case('b313', {'b', 'b3', 'b313'}, outcome=1)
    othercases3 = ([
      Case('b314', {'b', 'b3', 'b314'}, outcome=1),
      Case('b315', {'b', 'b3', 'b315'}, outcome=1),
      ])
    caseb313extra = Case('b313extra', {'b', 'b3', 'b313', 'extra'}, outcome=0)
    othercases4 = ([
      Case('c110', {'c', 'c1', 'c110'}, outcome=1),
      Case('c111', {'c', 'c1', 'c111'}, outcome=1),
      Case('c112', {'c', 'c1', 'c112'}, outcome=1),
      Case('c213', {'c', 'c2', 'c213'}, outcome=1),
      Case('c214', {'c', 'c2', 'c214'}, outcome=1),
      Case('c215', {'c', 'c2', 'c215'}, outcome=1),
      Case('c31', {'c', 'c3', 'c313'}, outcome=1),
      Case('c314', {'c', 'c3', 'c314'}, outcome=1),
      Case('c315', {'c', 'c3', 'c315'}, outcome=1),
      ])
    othercases = []
    for l in ([othercases1, [caseb3], othercases2, [caseb313],
               othercases3, [caseb313extra], othercases4]):
      othercases.extend(l)
    cb = [default, case1, case2, case3] + othercases
    clf = Aacbr()
    clf.fit(cb)
    test_case = Case('new', {'b', 'b3', 'b313', 'extra', 'extra2'})
    expected_output = [0]
    predicted_output = clf.predict([test_case])
    assert expected_output == predicted_output
    
    adt = clf.adt_explain(test_case)
    debug(f"{adt.nodes=}\n{adt.edges=}")
    assert type(adt) is ArbitratedDisputeTree
    expected_winning = set([default, caseb3,
                            caseb313extra, test_case])
    assert expected_winning == set(adt.get_winning_cases())

  
  def test_newcase_priority(self):
    """lower_priority case cannot be in ADT since test_case should occur instead.
    """
    default = Case('default', set(), outcome=0)
    case1 = Case('1', {'a'}, outcome=1)
    case2 = Case('2', {'b'}, outcome=1)
    case3 = Case('3', {'a', 'b'}, outcome=0)
    case4 = Case('4', {'b', 'd'}, outcome=0)
    case5 = Case('5', {'b', 'd', 'e'}, outcome=1)
    case6 = Case('6', {'b', 'd', 'e', 'f'}, outcome=0)
    case7 = Case('7', {'b', 'd', 'e', 'f', 'g'}, outcome=1)
    case8 = Case('8', {'b', 'd', 'e', 'f', 'g', 'h'}, outcome=0)
    cb = [default, case1, case2, case3, case4, case5, case6, case7, case8]
    clf = Aacbr()
    clf.fit(cb)
    test_case = Case('new', {'a', 'b', 'c', 'd'})
    expected_output = [0]
    predicted_output = clf.predict([test_case])
    assert expected_output == predicted_output
    
    adt = clf.adt_explain(test_case)
    debug(f"{adt.nodes=}\n{adt.edges=}")
    assert type(adt) is ArbitratedDisputeTree
    lower_priority = [case6, case7, case8]    
    for case in lower_priority:
      assert len(adt.get_nodes_labelled_by(case)) == 0
    pass
  
  # @pytest.mark.xfail(reason="Currently not implemented.")
  # def test_minimality(self):
  #   pass
    
class OrderedSequence(tuple):    
  def __sub__(self, other):
    if len(self) != len(other):
      raise(Exception("Arguments have different lengths!"))
    return OrderedSequence([self[i]-other[i] for i in range(len(self))])
  def __eq__(self, other):
    return isinstance(other, OrderedSequence) and \
      all([self[i]==other[i] for i in range(len(self))])
  def __le__(self, other):
    if len(self) != len(other):
      raise(Exception("Arguments have different lengths!"))
    return all([self[i]<=other[i] for i in range(len(self))])
  def __lt__(self, other):
    return self.__le__(other) and not self.__eq__(other)
  def __hash__(self):
    return tuple.__hash__(self)
  
  # def __lt__(self, other):
  #   return self <= other and self != other
  # def __hash__(self):
  #   return hash((self.x, self.y))
def generate_orderedsequence_casebase(n=100, dim=5):
  cb = [Case(i, OrderedSequence([random() for _ in range(dim)]),
             outcome=randint(0,1)) for i in range(n)]
  default_case = Case("default", OrderedSequence([0 for _ in range(dim)]),
                      outcome=0)
  cb = [default_case] + cb
  return cb

  
@pytest.mark.speed
class TestCautiousAacbrEfficiency:
  casebase = staticmethod(generate_orderedsequence_casebase)
  # @pytest.mark.parametrize("n",(10, 100, 200))
  # @pytest.mark.parametrize("d",(2, 5, 10))
  @pytest.mark.parametrize("n",(100,200))
  @pytest.mark.parametrize("d",(5,10))
  @pytest.mark.parametrize("cautious",(False,True))
  def test_speed(self, benchmark, cautious, n, d):
    cb = self.casebase(n, d)
    clf = Aacbr(cautious=cautious)
    benchmark(clf.fit, cb)
    pass
    
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
  # casebase2 = casebase[:-1]
  # clf.fit(casebase2)
  # print(casebase[-1], clf.predict([casebase[-1]]))
  # print(cautious)
  # print(sorted(casebase))
  clf.fit(casebase)
  casebase_active_ids = set(map(lambda x: getattr(x, "id"), clf.casebase_active_))
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
    result = scenario.give_predictions(newcase_prepared)
    prediction = result[1][0]["Prediction"]
    assert prediction == newcase_spec["outcome_expected"][aacbr_type], f"Failed for {newcase_spec}, in type {aacbr_type}" f"Failed on test {test}"

@pytest.mark.skip(reason="Deprecated interface.")
def test_files_non_cautious_old(setup):
  run_test_from_files_old_interface("non_cautious", setup)

@pytest.mark.skip(reason="Deprecated interface.")
def test_files_cautious_old(setup):
  run_test_from_files_old_interface("cautious", setup)
