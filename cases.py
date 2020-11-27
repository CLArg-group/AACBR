
import json

from variables import *


class Case:
	'''Defines a case to comprise id, factors, outcome, 
		lists of attackees and attackers and a label'''

	def __init__(self, id, factors, outcome, attackees: list, attackers: list, label: int):
		self.id = id
		self.factors = factors
		self.outcome = outcome
		self.attackees = attackees
		self.attackers = attackers
		self.label = label

		
def differentOutcomes(A, B):
	return A.outcome != B.outcome


def moreSpecific(A, B):
	return B.factors.issubset(A.factors) and B.factors != A.factors


def mostConcise(cases, A, B):
	return not any((moreSpecific(A, case) and moreSpecific(case, B) and not(differentOutcomes(A, case))) for case in cases)


def attacks(cases, A, B):
	return differentOutcomes(A, B) and moreSpecific(A, B) and mostConcise(cases, A, B)


def newcaseattacks(newcase, targetcase):
	return not newcase.factors.issuperset(targetcase.factors)
	
	
def inconsistentattacks(A, B):
	return differentOutcomes(A, B) and B.factors == A.factors


def loadCases(file: str) -> list:
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
#				non_default_case = Case(ID_NON_DEFAULT, set(), OUTCOME_NON_DEFAULT, [], [], 0)
#				cases.insert(0, non_default_case)
			else:
				case = Case(entry['id'], set(entry['factors']), entry['outcome'], [], [], 0)
				cases.append(case)
	json_file.close()
	
	return cases
	
	
def giveCasebase(cases: list) -> list:
	'''Returns a casebase given a list of cases'''

	casebase = []
	for candidate_case in cases:
		duplicate = False
		for case in casebase:
			if candidate_case.id != case.id and candidate_case.factors == case.factors and candidate_case.outcome == case.outcome:
				duplicate = True
		if not duplicate:
			casebase.append(candidate_case)	
	
	for case in casebase:
		for othercase in casebase:
			if attacks(casebase, case, othercase) or inconsistentattacks(case, othercase):
				case.attackees.append(othercase)
				othercase.attackers.append(case)
				
	return casebase


def giveNewcases(casebase: list, cases: list) -> list:
	'''Returns a a list of new cases given a list of cases and a casebase'''

	newcases = []
	for newcase in cases:
		for case in casebase:
			if newcaseattacks(newcase, case):
				newcase.attackees.append(case)
		newcases.append(newcase)
		
	return newcases
	
	
