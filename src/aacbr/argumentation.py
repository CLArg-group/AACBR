import copy

def compute_grounded(args: set, att: set):
  '''
  Returns the grounded labelling given an abstract argumentation
  framework.
  '''
  remaining = copy.copy(args)
  IN = set()
  OUT = set()
  UNDEC = set()
  unattacked = set()
  computed_unattacked = False
  while remaining:
    attacked = set()
    for (arg1, arg2) in att:
      if (arg1 in remaining) and (arg2 in remaining):
        attacked.add(arg2)
    IN = IN | (remaining - attacked)

    if computed_unattacked is False:
      unattacked.update(IN)
      computed_unattacked = True

    for (arg1, arg2) in att:
      if (arg1 in IN) and (arg2 in attacked) and (arg2 not in OUT):
        OUT.add(arg2)

    remaining = remaining - (IN | OUT)

    if attacked == remaining:
      break
  UNDEC = args - (IN | OUT)

  # | operator means union of 2 sets; IN computes the unattacked
  # arguments mainly G0(in the first step); OUT means the arguments
  # not in the grounding. It computes the unattacked args first and
  # afterwards it removes them from the remaing ones and continues.
  # raise(Exception(f"{args}\n{att}\n{ {'in': IN, 'out': OUT, 'undec': UNDEC}}"))
  return {'in': IN, 'out': OUT, 'undec': UNDEC}
