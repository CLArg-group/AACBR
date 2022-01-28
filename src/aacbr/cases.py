import json

from .variables import *

class Case:
  '''Defines a case to comprise id, factors, outcome, 
    lists of attackees and attackers.'''
  
  def __init__(self, id, factors, outcome=None, attackees=None, attackers=None, weight=None):
    self.id = id
    self.factors = set(factors)
    self.outcome = outcome
    self.attackees = attackees if attackees else []
    self.attackers = attackers if attackers else []
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
    
def differentOutcomes(A, B):
  return A.outcome != B.outcome

def moreSpecific(A, B, partial_order=set.issubset):
  return B.factors.issubset(A.factors) and B.factors != A.factors

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
        default_case = Case(ID_DEFAULT, set(), OUTCOME_DEFAULT, [], [], 0)
        cases.insert(0, default_case)
        # uncomment the following 2 lines for two defaults
#        non_default_case = Case(ID_NON_DEFAULT, set(), OUTCOME_NON_DEFAULT, [], [], 0)
#        cases.insert(0, non_default_case)
      else:
        case = Case(entry['id'], set(entry['factors']), entry['outcome'], [], [], 0)
        cases.append(case)
  json_file.close()
  
  return cases
  
  
def give_casebase(cases: list) -> list:
  '''Returns a casebase given a list of cases'''

  casebase = []
  for candidate_case in cases:
    duplicate = False
    for case in casebase:
      if candidate_case.id != case.id and candidate_case.factors == case.factors and candidate_case.outcome == case.outcome:
        # What if two cases have the same id (and same factors and outcome)? Why not marked as duplicate as well?
        duplicate = True
    if not duplicate:
      casebase.append(candidate_case)  
  
  for case in casebase:
    for othercase in casebase:
      if attacks(casebase, case, othercase) or inconsistentattacks(case, othercase):
        case.attackees.append(othercase)
        othercase.attackers.append(case)
        
  return casebase


def give_newcases(casebase: list, cases: list) -> list:
  '''Returns a a list of new cases given a list of cases and a casebase'''

  newcases = []
  for newcase in cases:
    for case in casebase:
      if newcaseattacks(newcase, case):
        newcase.attackees.append(case)
    newcases.append(newcase)
    
  return newcases

