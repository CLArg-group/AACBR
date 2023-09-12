import copy
import operator
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
  """
  - mode=arbitrary: returns an arbitrated dispute tree, with no
    guarantees except that it is one.
  - mode=minimal: returns a minimal arbitrated dispute tree, in
    number of nodes.
  - mode=all: returns all possible arbitrated dispute trees.
  
  Even in mode=all, not _every_ ADT is generated. We do not consider
  ADTs in which an irrelevant case attacks another (since those are
  unnecessarily longer).
      """
  win_label = "W"
  lose_label = "L"
  def __init__(self, clf, new_case, grounded, root_node, mode="arbitrary"):
    "Do not use this method! Create via create_adt."
    self.nodes = []
    self.edges = []
    self.clf = clf
    self.new_case = new_case
    self._compute_adt(grounded, root_node, mode=mode)

  @classmethod
  def create_adt(cls, clf, new_case, grounded, mode="arbitrary"):
    root_node = cls._create_root_node(clf, grounded)
    if root_node is None:
      return None
    else:
      return cls(clf, new_case, grounded, root_node, mode=mode)
  
  @classmethod
  def _create_root_node(cls, clf, grounded):
    if clf.default_case in grounded['undec']:
      warn(f"""ArbitratedDisputeTree for {clf} was requested, but this is
      impossible, since the grounded labelling for the default case
      {clf.default_case} is UNDEC (undecided). This implies there is a
      cycle in {clf}.""")
      return None
    elif clf.default_case in grounded['in']:
      root_node = (cls.win_label, clf.default_case)
    elif clf.default_case in grounded['out']:
      root_node = (cls.lose_label, clf.default_case)
    else:
      raise Exception(f"""Unexpected error: {clf.default_case=} is not
      labelled by the grounded labelling.""")
    return root_node
    
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
  
  def _compute_adt(self, grounded, root_node, mode="arbitrary"):
    if mode == "minimal":
      self._calculate_ranks()
    self.nodes.append(root_node)
    grounded_label_of = _get_node_labelling_dict(grounded)
    stack = [root_node]
    explored = set()
    while stack != []:
      current_node = stack.pop()
      to_explore = self._explore_node(current_node,
                                      grounded_label_of)
      explored.add(current_node)
      self._cycle_check(explored, to_explore)
      stack.extend(to_explore)
    return self

  def _cycle_check(self, explored, to_explore):
    for node in to_explore:
      if node in explored:
        return Exception(f"""Unexpected error: cycle longer than 2 nodes
        found.\n{node=} was added, but it was already explored.""")

  def _explore_node(self, node, grounded_label_of):
    """Changes self in-place.
    
    Returns the children node, which are added but are yet to be
    explored."""
    clf = self.clf
    new_case = self.new_case
    node_case = node[1]
    if node[0] == self.win_label:
      # all attackers included as lose nodes
      attackers = clf.attackers_of_[node_case]
      children = [(self.lose_label, attacker) for attacker in attackers]
    else: # node[0] == self.lose_label
      # exactly one of the attackers included as win node
      if node_case in clf.attacked_by_[new_case]:
        # we prioritise the new_case as attacker
        # new_case does not get into clf.attackers_of_, only clf.attacked_by_
        ### this is also not a problem elsewhere since no case attacks the newcase.
        attacker = new_case
      elif len(clf.attackers_of_[node_case]) == 0:
        raise(Exception(f"""Unexpected error: {node=} has no attackers in {clf}, but it is labelled as losing. Make sure {clf} was correctly generated."""))
      else:
        # an arbitrary attacker
        # the restriction is that it cannot be an incoherent attacker, otherwise it would create a loop
        for attacker in clf.attackers_of_[node_case]:
          if not clf.inconsistent_pair(node_case, attacker) \
             and (grounded_label_of[attacker] == 'in'):
            # acceptable attacker found
            break
          else:
            continue
        else:
          if clf.inconsistent_pair(node_case, attacker):
            raise(Exception(f"Unexpected error: {node=} has as its only valid attacker its incoherent pair {attacker}. Absurd."))
          else:
            raise(Exception(f"Unexpected error: {node=} no attackers labelled IN. Some of its attackers are: {clf.attackers_of_[node_case][:5]}"))
      child = (self.win_label, attacker)
      children = [child]
    self.nodes.extend(children)
    self.edges.extend((child, node) for child in children)
    return children
  
  def _calculate_ranks(self):
    """Returns a dict containing the rank of each node.

    This is used for building a minimal ADT.
    
    The rank is calculated as follows:
    If a node is a leaf (unattacked), the rank is 0.
    Else, if a node is 
    """
    # Not hard to implement changing criterion to depth or width.
    # Here implemented as number of nodes in subtree.
    # Essentially a form of max-min (for depth)
    # or of min-sum (tropical geometry) (for number of nodes).
    # This is basically based on semirings. :)
    criterion = "number_of_nodes"
    if criterion == "number_of_nodes":
      add = min
      prod = operator.add
      initial_value = 1
    clf = self.clf
    cb = clf.casebase_active_
    new_case = clf.new_case
    unattacked = tuple(c for c in cb if
                       c not in clf.attacked_by_[new_case] and
                       len(clf.attackers_of_[c]) == 0)
    ranks = {c:initial_value for c in unattacked}
    stack = list(unattacked)
    while stack != []:
      current = stack.pop()
      to_add = self._get_rank_of_attacked_by(current, ranks, add, prod)
      stack.extend(to_add)
    return ranks

  def _get_rank_of_attacked_by(self, current, ranks, add, prod):
    attacked = self.clf.attacked_by_[current]
    rank = ranks[current]+1
    rank = ...
    raise(Exception("Not yet implemented"))
    

def _get_node_labelling_dict(labels):
  """Receives a dict of the form
    labels = {'in': NODES_IN, 'out': NODES_OUT, 'undec': NODES_UNDEC}
  and returns a dict of the form
    {node:label[node]}
  """
  result = {}
  for key in labels.keys():
    for node in labels[key]:
      result[node] = key
  return result
