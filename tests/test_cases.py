### Tests cases input, partial order, relevance, etc.
from context import aacbr

import pytest
from aacbr.cases import Case, differentOutcomes, moreSpecific #, mostConcise, attacks, newcaseattacks, 

def test_create_case():
  empty_case = Case('empty', set())
  print(empty_case)
  pass

def test_specificity():
  case1 = Case('1', {'a'})
  case2 = Case('2', {'a','b'})
  assert moreSpecific(case2, case1)
  assert not moreSpecific(case1, case2)
  assert not moreSpecific(case1, case1)
  assert not moreSpecific(case2, case2)

def test_different_outcomes():
  case1 = Case('1', {'a'}, outcome=0)
  case2 = Case('2', {'a','b'}, outcome=1)
  assert differentOutcomes(case1, case2)
  assert differentOutcomes(case2, case1)
  assert not differentOutcomes(case1, case1)
  assert not differentOutcomes(case2, case2)
  
# def test_conciseness():
#   case1 = Case('1', {'a'})
#   case2 = Case('2', {'a','b'})
#   case3 = Case('3', {'a','b','c'})
#   cases = [case1, case2, case3]
#   assert mostConcise(cases, case2, case1)
#   assert not mostConcise(cases, case3, case1)
#   assert not mostConcise(cases, case1, case2)

# def test_newcaseattack():
#   newcase = Case('new', {'a'})
#   case1 = Case('1', {'a'}, outcome=0)
#   case2 = Case('2', {'a','b'}, outcome=1)
#   default = Case('default', set(), outcome=0)
#   cases = (default, case1, case2, newcase)
#   assert newcaseattacks(newcase, case2)
#   assert not newcaseattacks(newcase, case1)
#   assert not newcaseattacks(newcase, default)
  
@pytest.mark.xfail(reason="Inconsistent cases not yet solved here.")
def test_inconsistent():
  case1 = Case('1', {'a','b'}, outcome=0)
  case2 = Case('2', {'a','b'}, outcome=1)
  cases = [case1, case2]
  assert attacks(cases, case1, case2)
  assert attacks(cases, case2, case1)
  assert inconsistentattacks(case1, case2)

# @pytest.mark.xfail(reason="No uniform notation yet.")
# [2022-01-27 Thu 22:35]: Removing this test since new cases will not be part of the casebase itself.
# def test_uniform_attack_notation():
#   # Perhaps this test should not exist and the notion of attack should be left to the model, not to case/data.
#   default = Case('default', set(), outcome=0)
#   case1 = Case('1', {'a'}, outcome=1)
#   case2 = Case('2', {'a','b'}, outcome=0)
#   case3 = Case('3', {'a'}, outcome=1)
#   case4 = Case('4', {'a','b'}, outcome=0)
#   case5 = Case('5', {'a'}, outcome=1)
#   newcase = Case('new', {'a'})
#   cases = (default, case1, case2, newcase)
#   assert attacks(cases, case1, default)
#   assert attacks(cases, case2, case1)
#   assert attacks(cases, newcase, case2)
  
def test_load_cases():
  # TODO: implement
  pass

def test_list_of_numbers_partial_order():
  # TODO: implement
  pass

def test_arbitrary_partial_order():
  # TODO: implement
  pass
