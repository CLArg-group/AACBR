import json

from .variables import *
from functools import cache
from collections.abc import Hashable

class Case:
  '''Defines a case to comprise id, factors, outcome, 
    lists of attackees and attackers.

  Partial order is calculated by the "<=" operator of the `factors`
  attribute, i.e., by its __le__ method.

  The type of `factors` should also be hashable. In case it is a set,
  frozenset is used instead for hashing.'''
  
  def __init__(self, id, factors, outcome=None):
    self.id = id
    if type(factors) == set:
      self.factors = frozenset(factors)
    elif not isinstance(factors, Hashable):
      self.factors = tuple(factors)
    else:
      self.factors = factors
    self.outcome = outcome
    self._hash = hash((self.id, self.factors, self.outcome))
  def __str__(self):
    return f'Case("id": {self.id}, "factors": {self.factors}, "outcome": {self.outcome})'
  def __repr__(self):
    # return f'Case("id": {self.id}, "factors": {self.factors}, "outcome": {self.outcome})'
    return self.__str__()
  def __eq__(self, other):
    if not isinstance(other, Case):
      return NotImplemented
    # raise(Exception(f"Trying equality for {self} and {other}.\nHash for self is: {hash(self)}.\nHash for other is: {hash(other)}"))
    # return all([self.id == other.id, self.factors == other.factors, self.outcome == other.outcome]) # slow
    return self._hash == other._hash
  def __hash__(self):
    return self._hash
  @cache
  def __le__(self, other):
    return self.factors <= other.factors
    # return self.factors < other.factors or self.factors == other.factors
  def __lt__(self, other):
    return self <= other and self.factors != other.factors
    
def different_outcomes(A, B):
  return A.outcome != B.outcome

# def more_specific_weakly(A, B):
#   """Partial order >= for two cases."""
#   # return B.factors.issubset(A.factors)
#   return A.factors >= B.factors

def inconsistent_pair(A, B):
  return A.factors == B.factors and different_outcomes(A, B)

def load_cases(file, nr_defaults=1) -> list:
  '''Loads cases form a file'''
  
  # global ID_DEFAULT, OUTCOME_DEFAULT, ID_NON_DEFAULT, OUTCOME_NON_DEFAULT 
  
  cases = []
  with open(file, encoding='utf-8') as json_file:
    entries = json.load(json_file)
    for entry in entries:
      if entry['id'] == ID_DEFAULT:
        default_case = Case(ID_DEFAULT, set(), OUTCOME_DEFAULT)
        cases.insert(0, default_case)
        
        if nr_defaults == 2:
          non_default_case = Case(ID_NON_DEFAULT, set(), OUTCOME_NON_DEFAULT, [], [], 0)
          cases.insert(0, non_default_case)
      else:
        factors = entry['factors']
        case = Case(entry['id'], set(factors), entry['outcome'])
        cases.append(case)
  json_file.close()

  return cases
