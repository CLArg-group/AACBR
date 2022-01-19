### Tests cases input, partial order, relevance, etc.
from context import aacbr

import pytest
from aacbr.cases import Case, moreSpecific, attacks, newcaseattacks

def test_create_case():
  empty_case = Case('empty', [])
  print(empty_case)
  pass

def test_specificity():
  case1 = Case('1', ['a'])
  case2 = Case('2', ['a','b'])
  assert moreSpecific(case2, case1)

def test_attack():
  case1 = Case('1', ['a'], outcome=0)
  case2 = Case('2', ['a','b'], outcome=1)
  assert attacks([case1, case2], case2, case1)
  assert not attacks([case1, case2], case1, case2)
  assert not attacks([case1, case2], case1, case1)

def test_newcaseattack():
  newcase = Case('new', ['a'])
  case1 = Case('1', ['a'], outcome=0)
  case2 = Case('2', ['a','b'], outcome=1)
  default = Case('default', [], outcome=0)
  cases = (default, case1, case2, newcase)
  assert newcaseattacks(newcase, case2)
  assert not newcaseattacks(newcase, case1)
  assert not newcaseattacks(newcase, default)

def test_inconsistent():
  case1 = Case('1', ['a','b'], outcome=0)
  case2 = Case('2', ['a','b'], outcome=1)
  cases = [case1, case2]
  assert attacks(cases, case1, case2)
  assert attacks(cases, case2, case1)
  assert inconsistentattacks(case1, case2)
  
def test_load_cases():
  # TODO: implement
  pass
