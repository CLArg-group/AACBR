import copy
import operator
from collections import deque
from warnings import warn
from logging import debug, info, warning, error

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
  We do not consider ADTs in which an irrelevant case attacks another
  (since those are unnecessarily longer).

  To create an instance, use ArbitratedDisputeTree.create_adt

  TODO:
  - mode=all (TODO): returns all possible arbitrated dispute trees.
    Even in mode=all, not _every_ ADT would be generated (since those
    could be arbitrarily long with a 2-cycle, and due to the bias for
    the irrelevant case attack.
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
    return tuple(set(node[1] for node in self.get_winning_nodes()))
  def get_losing_cases(self):
    return tuple(set(node[1] for node in self.get_losing_nodes()))
  def get_cases(self):
    return tuple(set(node[1] for node in self.nodes))
  def get_depth(self):
    return self.depth
  
  def _compute_adt(self, grounded, root_node, mode="arbitrary"):
    grounded_label_of = _get_node_labelling_dict(grounded)
    if mode == "minimal":      
      ranks = self._calculate_ranks(grounded_label_of)
    else:
      ranks = None
    self.nodes.append(root_node)
    node_depths, max_depth = {root_node: 1}, 1
    stack = [root_node]
    while stack != []:
      current_node = stack.pop()
      to_explore = self._explore_node(current_node,
                                      grounded_label_of,
                                      mode=mode, ranks=ranks)
      stack.extend(to_explore)
      current_depth = node_depths[current_node]
      for node in to_explore:
        node_depth = current_depth + 1
        node_depths[node] = node_depth
        if node_depth > max_depth:
          max_depth = node_depth
    self.depth = max_depth
    return self

  def _explore_node(self, node, grounded_label_of,
                    mode="arbitrary", ranks=None):
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
        if mode == "arbitrary":
          # the restriction is that it cannot be an incoherent attacker,
          # otherwise it would create a loop
          for attacker in clf.attackers_of_[node_case]:
            if (grounded_label_of[attacker] == 'in') \
               and (not clf.inconsistent_pair(node_case, attacker) \
                    or clf.has_default_characterisation(attacker)):
              # acceptable attacker found
              break
            else:
              continue
          else:
            if clf.inconsistent_pair(node_case, attacker):
              raise(Exception(f"Unexpected error: {node=} has as its only valid attacker its incoherent pair {attacker}. Absurd."))
            else:
              raise(Exception(f"Unexpected error: {node=} no attackers labelled IN. Some of its attackers are: {clf.attackers_of_[node_case][:5]}"))
        elif mode == "minimal":
          candidates = [candidate for candidate in clf.attackers_of_[node_case] if \
                        (grounded_label_of[candidate] == 'in') \
                        and \
                        (not clf.inconsistent_pair(node_case, candidate) \
                        or clf.has_default_characterisation(candidate))]
          try:
            attacker = min(candidates, key=ranks.get)
          except:
            exception_string = f"{node=} which is a losing node could not find minimal attacker.\nSome {candidates[:10]=}\nranks: {[ranks.get(c,None) for c in candidates[:10]]}\n"
            exception_string += f"{clf.attackers_of_[node_case]=}"
            raise(Exception(exception_string))
        else:
          raise(Exception(f"Unknown {mode=}"))
      child = (self.win_label, attacker)
      children = [child]
    self.nodes.extend(children)
    self.edges.extend((child, node) for child in children)
    return children
  
  def _calculate_ranks(self, grounded_label_of):
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

    # Can be optimised with some kind of branch-and-bound. For now I
    # will not do it.
    criterion = "number_of_nodes"
    if criterion == "number_of_nodes":
      add = min
      prod = operator.add
      initial_value = 1
    clf = self.clf
    cb = clf.casebase_active_
    new_case = self.new_case
    unattacked_cb = tuple(c for c in cb if
                          c not in clf.attacked_by_[new_case] and
                          len(clf.attackers_of_[c]) == 0)
    unattacked = unattacked_cb + (new_case,)
    ranks = {c:initial_value for c in unattacked}
    queue = deque(unattacked)
    explored = set()
    while queue != deque():
      current = queue.popleft()
      to_add = self._get_rank_of_attacked_by(current, ranks,
                                             grounded_label_of,
                                             add, prod)
      explored.add(current)
      to_add = tuple(x for x in to_add if x not in explored)
      queue.extend(to_add)
    return ranks

  def _get_rank_of_attacked_by(self, current, ranks,
                               grounded_label_of,
                               add, prod):
    all_attacked = self.clf.attacked_by_[current]
    for attacked in all_attacked:
      rank = ranks.get(attacked)
      if rank is None:
        ranks[attacked] = ranks[current]+1
      else:
        match grounded_label_of[attacked]:
          case "undec":
            raise(Exception(f"Unexpected {grounded_label_of[attacked]=}"))
          case "in": # winning, thus consider all
            ranks[attacked] = prod(rank, ranks[current])
          case "out": # losing, thus consider only best
            ranks[attacked] = add(rank, ranks[current]+1)
            # one can do better storing somewhere which is the maxarg
            # for now I will keep like this
    return all_attacked
    

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
