
import copy


def giveAAframework(casebase: list, newcase) -> dict:
  '''Returns an abstract argumentation framework given a casebase and a new case'''

  arguments = set()
  attacks = set()
  for case in casebase:
    arguments.add('argument({})'.format(case.id))
    for attackee in case.attackees:
      attacks.add(('argument({attacker_argument})'.format(attacker_argument = case.id), 'argument({attacked_argument})'.format(attacked_argument = attackee.id)))
    for attacker in case.attackers:
      attacks.add(('argument({attacker_argument})'.format(attacker_argument = attacker.id), 'argument({attacked_argument})'.format(attacked_argument = case.id)))
  arguments.add('argument({})'.format(newcase.id))
  for attackee in newcase.attackees:
    attacks.add(('argument({attacker_argument})'.format(attacker_argument = newcase.id), 'argument({attacked_argument})'.format(attacked_argument = attackee.id)))
  
  return {'arguments': arguments, 'attacks': attacks}


def computeGrounded(args: set, att: set) -> dict:
  '''Returns the grounded labelling given an abstract argumentation framework'''

  remaining = copy.copy(args)
  IN = set()
  OUT = set()
  UNDEC = set()
  while remaining:
    attacked = set()
    for (arg1, arg2) in att:
      if (arg1 in remaining) and (arg2 in remaining):
        attacked.add(arg2)
    IN = IN | (remaining - attacked)
    
    for (arg1, arg2) in att:
      if (arg1 in IN) and (arg2 in attacked) and (arg2 not in OUT):
        OUT.add(arg2)
    
    remaining = remaining - (IN | OUT)
    if attacked == remaining:
      break
  UNDEC = args - (IN | OUT)
    
  return {'in': IN, 'out': OUT, 'undec': UNDEC}
  
