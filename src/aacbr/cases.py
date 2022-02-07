import json

from .variables import *

class Case:
  '''Defines a case to comprise id, factors, outcome, 
    lists of attackees and attackers.'''
  
  def __init__(self, id, factors, outcome=None, weight=None):
    self.id = id
    self.factors = set(factors)
    self.outcome = outcome
    if weight: # currently unused
      self.weight = weight
  def __str__(self):
    return f'Case("id": {self.id}, "factors": {self.factors}, "outcome": {self.outcome})'
  def __repr__(self):
    # return f'Case("id": {self.id}, "factors": {self.factors}, "outcome": {self.outcome})'
    return self.__str__()
  def __eq__(self, other):
    if not isinstance(other, Case):
      return NotImplemented
    return all([self.id == other.id, set(self.factors) == set(other.factors), self.outcome == other.outcome])
  def __hash__(self):
    return hash((self.id, frozenset(self.factors), self.outcome))
  def __ge__(self, other):
    return more_specific_weakly(self, other)
  def __gt__(self, other):
    return self.__ge__(other) and self.factors != other.factors
    
def different_outcomes(A, B):
  return A.outcome != B.outcome

def more_specific_weakly(A, B):
  """Partial order >= for two cases."""
  # Important to check whether correct partial order is being used
  # return sum(A.weight) > sum(B.weight)
  # return B.factors.issubset(A.factors) and B.factors != A.factors # this disallows incoherence
  # return B.factors.issubset(A.factors) and (B.factors != A.factors or B.outcome != A.outcome)
  return B.factors.issubset(A.factors)

def inconsistent_pair(A, B):
  return A.factors == B.factors and different_outcomes(A, B)


# should be in aacbr
# def mostConcise(cases, A, B):
#   return moreSpecific(A,B) and not any((moreSpecific(A, case) and moreSpecific(case, B) and not(differentOutcomes(A, case))) for case in cases)

# def attacks(cases, A, B):
#   return differentOutcomes(A, B) and mostConcise(cases, A, B)

# def newcaseattacks(newcase, targetcase):
#   return not newcase.factors.issuperset(targetcase.factors)
    
# def inconsistentattacks(A, B):
#   return differentOutcomes(A, B) and B.factors == A.factors

def load_cases(file: str) -> list:
  '''Loads cases form a file'''

  global ID_DEFAULT, OUTCOME_DEFAULT, ID_NON_DEFAULT, OUTCOME_NON_DEFAULT 

  cases = []
  with open(file, encoding = 'utf-8') as json_file:
    entries = json.load(json_file)
    for entry in entries:
      if entry['id'] == ID_DEFAULT:
        default_case = Case(ID_DEFAULT, set(), OUTCOME_DEFAULT)
        cases.insert(0, default_case)
        # uncomment the following 2 lines for two defaults
#        non_default_case = Case(ID_NON_DEFAULT, set(), OUTCOME_NON_DEFAULT, [], [], 0)
#        cases.insert(0, non_default_case)
      else:
        case = Case(entry['id'], set(entry['factors']), entry['outcome'])
        cases.append(case)
  json_file.close()
  
  return cases
