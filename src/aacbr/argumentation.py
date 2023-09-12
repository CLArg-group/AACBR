import copy
from warnings import warn

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

class ArbitratedDisputeTree:
  win_label = "W"
  lose_label = "L"
  def __init__(self, nodes, edges):
    self.nodes = nodes
    self.edges = edges
  def get_nodes_labelled_by(self, case):
    return tuple(node for node in self.nodes if node[1] == case)
  def get_winning_nodes(self):
    return tuple(node for node in self.nodes if node[0] == self.win_label)
  def get_losing_nodes(self):
    return tuple(node for node in self.nodes if node[0] == self.lose_label)
  def get_winning_cases(self):
    return tuple(node[1] for node in self.get_winning_nodes())
  def get_losing_cases(self):
    return tuple(node[1] for node in self.get_losing_nodes())
  def get_cases(self):
    return tuple(node[1] for node in self.nodes)
      
def _compute_adt(clf, new_case, grounded, mode="arbitrary"):
  """
    - mode=arbitrary: returns an arbitrated dispute tree, with no guarantees except that it is one.
    - mode=all: returns all possible arbitrated dispute trees.
    - mode=minimal: only returns a minimal arbitrated dispute tree, in number of nodes.

    Even in mode=all, not _every_ ADT is generated. We do not consider
    ADTs in which an irrelevant case attacks another (since those are
    unnecessarily longer).
  """
  nodes = []
  edges = []
  current_adt = ArbitratedDisputeTree(nodes, edges)
  root_node = _create_root_node(current_adt, clf, grounded)
  if root_node is None:
    return None
  current_adt.nodes.append(root_node)
  stack = [root_node]
  explored = set()
  while stack != []:
    current_node = stack.pop()
    current_adt, to_explore = _explore_node(current_node, current_adt,
                                            clf, new_case, grounded)
    explored.add(current_node)
    _cycle_check(explored, to_explore, current_adt)
    stack.extend(to_explore)
  return current_adt

def _cycle_check(explored, to_explore, adt):
  for node in to_explore:
    if node in explored:
      return Exception(f"Unexpected error: cycle longer than 2 nodes found.\n{node=} was added, but it was already explored.")

def _create_root_node(adt, clf, grounded):
  if clf.default_case in grounded['undec']:
    warn(f"ArbitratedDisputeTree for {clf} was requested, but this is impossible, since the grounded labelling for the default case {clf.default_case} is UNDEC (undecided). This implies there is a cycle in {clf}.")
    return None
  elif clf.default_case in grounded['in']:
    root_node = (adt.win_label, clf.default_case)
  elif clf.default_case in grounded['out']:
    root_node = (adt.lose_label, clf.default_case)
  else:
    raise Exception(f"Unexpected error: {clf.default_case=} is not labelled by the grounded labelling.")
  return root_node

def _explore_node(node, adt, clf, new_case, grounded):
  node_case = node[1]
  if node[0] == adt.win_label:
    # all attackers included as lose nodes
    attackers = clf.attackers_of_[node_case]
    children = [(adt.lose_label, attacker) for attacker in attackers]
  else: # node[0] == adt.lose_label
    # exactly one of the attackers included as win node
    if node_case in clf.attacked_by_[new_case]:
      # we prioritise the new_case as attacker
      # new_case does not get into clf.attackers_of_, only clf.attacked_by_
      ### this is also not a problem elsewhere since no case attacks the newcase.
      attacker = new_case
    elif len(clf.attackers_of_[node_case]) == 0:
      raise(Exception(f"Unexpected error: {node=} has no attackers in {clf}, but it is labelled as losing. Make sure {clf} was correctly generated."))
    else:
      # an arbitrary attacker
      # the restriction is that it cannot be an incoherent attacker, otherwise it would create a loop
      for attacker in clf.attackers_of_[node_case]:
        if not clf.inconsistent_pair(node_case, attacker):
          # acceptable attacker found
          break
        else:
          continue
      else:
        if clf.inconsistent_pair(node_case, attacker):
          raise(Exception(f"Unexpected error: {node=} has as its only valid attacker its incoherent pair {attacker}. Absurd."))
        else:
          raise(Exception(f"Unexpected error: {node=} no attackers labelled IN. Some of its attackers are: {clf.attackers_of_[node_case][:5]}"))
    child = (adt.win_label, attacker)
    children = [child]
  adt.nodes.extend(children)
  adt.edges.extend((child, node) for child in children)
  return adt, children
