### This encapsulates AA-CBR as a model.

import sys
import os

from datetime import datetime
from datetime import timedelta
import time
import json
import networkx as nx

from functools import cmp_to_key, reduce
from operator import lt, add
from collections import deque, defaultdict, Counter
from warnings import warn
from logging import debug, info, warning, error
from numbers import Real
from collections.abc import Sequence
from graphlib import TopologicalSorter
from statistics import mean, pstdev

from .argumentation import ArbitratedDisputeTree, compute_grounded
from .cases import Case, different_outcomes
from .graphs import giveGraph, getPath, drawGraph
from .variables import OUTCOME_DEFAULT, OUTCOME_NON_DEFAULT, OUTCOME_UNKNOWN, ID_DEFAULT, ID_NON_DEFAULT

def tsorted(case_list):
  ts = TopologicalSorter({i:[j for j in case_list if i>j]
                          for i in case_list})
  return tuple(ts.static_order())


class Aacbr:
  def __init__(self, outcome_def=OUTCOME_DEFAULT,
               outcome_nondef=OUTCOME_NON_DEFAULT, outcome_unknown=OUTCOME_UNKNOWN,
               default_case=None, cautious=False,
               remove_spikes=False):
    self.outcome_def = outcome_def
    self.outcome_nondef = outcome_nondef
    self.outcome_unknown = outcome_unknown
    self.default_case = default_case
    self.cautious = cautious
    self.remove_spikes = remove_spikes
    # See fit method for attributes set only during training
    
  def get_params(self, deep=True):
    # for sklearn.utils.estimator_checks.check_estimator
    return {
      "outcome_def": self.outcome_def,
      "outcome_nondef": self.outcome_nondef,
      "outcome_unknown": self.outcome_unknown,
      "default_case": self.default_case,
      "cautious": self.cautious,
      "remove_spikes": self.remove_spikes
    }
  
  def set_params(self, **kwargs):
  # for sklearn.utils.estimator_checks.check_estimator
    for parameter, value in parameters.items():
      setattr(self, parameter, value)
    return self
  
  
    
  # def attacked_by(self, case):
  #   """List of cases that `case' attacks.
  #   Gets in memory."""
  #   if not case in self.casebase_active:
  #     raise Exception(f"{case} not in active casebase of {self}")
  #   return self._attacks[case]

  # def attackers_of(self, case):
  #   """List of cases that attacks `case'.
  #   Gets in memory."""
  #   if not case in self.casebase_active:
  #     raise Exception(f"{case} not in active casebase of {self}")
  #   return self._attacked[case]
  
  def fit(self, X=None, y=None, casebase=None, outcomes=None):
    """X and casebase are aliases, as are y and outcomes.
    """
    if X is not None:
      if casebase is not None:
        raise(Exception("Only one of 'casebase' and 'X' should be define, as they are aliases."))
      else:
        casebase = X
    if y is not None:
      if outcomes is not None:
        raise(Exception("Only one of 'outcomes' and 'y' should be define, as they are aliases."))
      else:
        outcomes = y
    
    if all((type(x) == Case for x in casebase)):
      cb_input = tuple(casebase)
    elif outcomes is not None:
      if len(outcomes) != len(casebase):
        raise(RuntimeError("Length of casebase argument is not the same as outcomes!"))
      else:
        cb_input = [Case(str(i), x, y)
                    for (i,(x,y)) in enumerate(zip(casebase, outcomes))]

    # unnecessary since it is now a method of the class
    if not isinstance(self, Aacbr):
      raise(Exception(f"{self} is not an instance of {Aacbr}"))

    self.casebase_initial_ = None
    self.casebase_active_ = None
    self.attacked_by_ = defaultdict(list) # for storing inside an Aacbr instance the attacks, not in the cases # attacked_by[a] = {b such that a attacks b}
    self.attackers_of_ = defaultdict(list)  # attackers_of[a] = {b such that b attacks a}
    self._nondefault_case_ = None # the default for opposite outcome

    self.casebase_initial_ = cb_input
    self.infer_default(cb_input)
    if self.default_case not in cb_input:
      cb_input += (self.default_case,)
    # self.partial_order = partial_order
    if not self.cautious:
      if not self.remove_spikes:
        # self.casebase_active_ = cb_input
        self.casebase_active_ = self.give_casebase(cb_input) # adding attacks
      else:
        # self.casebase_active_ = cb_input
        self.casebase_active_ = self.give_casebase_without_spikes(cb_input)
      
      # command 1: filter which cases to use -- but if this depends on attack, do I have to fit in order to define attack? this is strange
      # -- but this is no problem for typical aacbr, as long as we separate saving attack state from filtering
      # command 2: save attacks state
    else:
      if self.remove_spikes:
        warning("remove_spikes argument is ignored for cautious AA-CBR, since there will be no spikes by construction.")
      self.casebase_active_ = []
      self.casebase_active_ = self.give_cautious_subset_of_casebase(cb_input)
      # self.give_casebase(self.casebase_active_)
      # raise(Exception("Cautious case not implemented"))
    return self
  
  def infer_default(self, casebase):
    info("Inferring default")
    default_in_input = self.default_in_casebase(casebase)
    if default_in_input:
      self.default_case = default_in_input
      if default_in_input.outcome is not None and \
         self.outcome_def != default_in_input.outcome:
        warn(f"Overriding {self}.outcome_def of {self.outcome_def} with default_case outcome {default_in_input.outcome}")
        aux = self.outcome_def
        self.outcome_def = default_in_input.outcome
        self.outcome_nondef = aux
        assert self.outcome_unknown != self.outcome_def
        assert self.outcome_unknown != self.outcome_nondef
      
    elif type(self.default_case) == Case:
      if self.outcome_def != self.default_case.outcome:
        raise(RuntimeError(f"Default case outcome is not the same as as passed default outcome: {self.default_case.outcome} != {self.outcome_def}"))
      self.outcome_def = self.default_case.outcome
      if self.casebase_active_:
        self.casebase_active_ += [self.default_case]
      
    elif all([type(case.factors) == frozenset for case in casebase]):
      self.default_case = Case("default", set(), outcome=self.outcome_def)
      if self.casebase_active_:
        self.casebase_active_ += [self.default_case]
        
    elif all([(isinstance(case.factors, Sequence)
               and
               all(isinstance(element, Real)
                   for element in case.factors))
              for case in casebase]):
      # infers a sequence of same type all being negative infinity
      default_factors = type(casebase[0].factors)([float('-inf') for x in casebase[0].factors])
      self.default_case = Case("default", default_factors, outcome=self.outcome_def)
      if self.casebase_active_:
        self.casebase_active_ += [self.default_case]
      
    else:
      raise(RuntimeError("No default case to use!"))

  @staticmethod
  def default_in_casebase(casebase):
    candidates = [case for case in casebase if case.id == ID_DEFAULT]
    if len(candidates) > 1:
      raise(Exception(f"More than one case named 'default' in the casebase: {candidates}"))
    elif len(candidates) == 1:
      return candidates[0]
    else:
      return False

  def reset_attack_relations(self, casebase, completely=False):
    # self.attackers_of.clear()
    # self.attacked_by.clear()
    # TODO: does not work if I am removing just some cases from the AF, I should make a different interface for new cases, which I add and remove
    if completely:
      self.attacked_by_ = defaultdict(list)
      self.attackers_of_ = defaultdict(list)
      
    else:
      for case in casebase:
        self.attackers_of_.pop(case, None)
        self.attacked_by_.pop(case, None)
        # for othercase in self.attacked_by_.keys():
        #   self.attacked_by_[othercase].remove(case)
        # for othercase in self.attackers_of_.keys():
        #   self.attackers_of_[othercase].remove(case)
        pass
  
  # not something in the set of all possible cases in the middle
  def _old_minimal(self, A, B, cases):
    return not any((B < case and
                    case < A and
                    not (different_outcomes(A, case)))
                   for case in cases)
  
  def minimal(self, A, B):
    """Checks whether there is a case larger than A that attacks B,
    AND is strictly greater in the partial order.
    Assumes that A and B are in the casebase and their attackers are
    defined or being defined.
    """
    SAFETY_CHECK = False
    if SAFETY_CHECK:
      assert A in self.casebase_active_
      assert B in self.casebase_active_
    try:
      greater_attackers = [case for case in self.attackers_of_[B]
                           if not B.factors == case.factors]
    except KeyError:
      raise(Exception(f"{B} is not in the casebase!"))
    except NameError as E:
      raise(E)
    # Should we fall back to the usual minimal test? Currently we do not.
    return not any(((case < A) for case in greater_attackers))


  # attack relation defined
  def past_case_attacks(self, A, B):
    """Checks whether A should attack B by the past case rule.
    It assumes A and B are in the active casebase."""
    SAFETY_CHECK = False    
    # for performance, set SAFETY_CHECK to False
    # for safety, to True
    # if True, checks whether both cases are actually in the
    # casebase_active. Since this adds a big cost on performance, we
    # deactivated this check by default.
    if SAFETY_CHECK:
      if not all(x in self.casebase_active_ for x in (A,B)):
        raise(Exception(f"Arguments {(A,B)} are not both in the active casebase."))
    if A == self.default_case:
      return False
    else:
      return (different_outcomes(A, B) and
              B <= A and
              self.minimal(A, B))

  # unlabbled datapoint new_case
  @staticmethod
  def new_case_attacks(new_case, target_case):
    return not target_case.factors <= new_case.factors

  # noisy points
  def inconsistent_pair(self, A, B):
    return different_outcomes(A, B) and B.factors == A.factors

  def has_default_characterisation(self, case):
    return case.factors == self.default_case.factors

  def predict(self, new_cases, _keep_newcase_attacks=False):
    if all([type(nc) == Case for nc in new_cases]):
      new_cases = [Case(f"new{nc.id}", nc.factors, outcome=None) for nc in new_cases]
    else:
      new_cases = [Case(f"new{i}", x) for i,x in enumerate(new_cases)]
    predictions = self.give_predictions(new_cases,
                                        _keep_newcase_attacks=_keep_newcase_attacks)
    # return [prediction_dict["Prediction"] for prediction_dict in predictions]
    return predictions
  
  # predictions for multiple points
  def give_predictions(self, new_cases, nr_defaults=1,
                       _keep_newcase_attacks=False):
    # casebase = self.casebase_active
    # new_cases = self.give_new_cases(casebase, new_cases)
    predictions = []
    for new_case in new_cases:
      new_case_prediction = dict()
      prediction = self.give_prediction(new_case, nr_defaults)
      predictions.append(prediction)
      if not _keep_newcase_attacks:
        self.reset_attack_relations([new_case])  # clean new cases
        # unnecessary, since only self.attacked_by_ is updated by the new_cases
    # formatted = self.format_predictions(new_cases, predictions)
    return predictions

  @staticmethod
  def format_predictions(new_cases, predictions):
    return [{'No.': number,
             'ID': new_case.id,
             'Prediction': prediction}
            for (number,(new_case, prediction))
                 in enumerate(zip(new_cases, predictions))]
  
  
  def give_prediction(self, new_case, nr_defaults: int = 1):
    '''Returns an AA-CBR prediction given a casebase and a new case'''
    grounded = self.grounded_extension(new_case, output_type="labelling")
    def_arg = self.default_case
    non_def_arg = self._nondefault_case_
    prediction = None

    if nr_defaults == 1:
      if def_arg in grounded['in']:
        prediction = self.outcome_def
        sink = def_arg
      else:
        prediction = self.outcome_nondef
        sink = def_arg        
      # elif def_arg not in grounded['in']:
      #   prediction = self.outcome_nondef
      #   sink = def_arg
      # else:
      #   prediction = self.outcome_unknown
      #   sink = None
    else:
      raise(Exception("Unsupported nr_defaults: {nr_defaults}"))
    return prediction

  
  def grounded_extension(self, new_case, output_type="subset"):
    if not output_type in ("subset", "labelling"):
      raise RuntimeError(f"output_type argument should be either 'subset' or 'labelling'")
    if not type(new_case) == Case:
      raise RuntimeError(f"new_case argument needs to be of type {Case}, but is {type(new_case)}")
    casebase = self.casebase_active_
    new_case = self.give_new_case(casebase, new_case)
    arguments, attacks = self.give_argumentation_framework(new_case)
    grounded = compute_grounded(arguments, attacks)
    if output_type == "subset":
      return grounded["in"]
    else:
      return grounded
    
  def give_argumentation_framework(self, new_case = None) -> tuple:
    '''Returns an abstract argumentation framework (AAF) given a
    casebase and, optionally, a new case.
    The AAF is returned as a pair (arguments, attacks).'''
    if new_case is not None and type(new_case) is not Case:
      raise(TypeError(f"{new_case=} is not of type {Case}"))
    casebase = self.casebase_active_
    arguments = set()
    attacks = set()
    arguments = {case for case in casebase}
    for case in casebase:
      for attacker in self.attackers_of_[case]:
        attacks.add((attacker, case))
    if new_case != None:
      arguments.add(new_case)
      for attacked in self.attacked_by_[new_case]:
        attacks.add((new_case, attacked))

    return (arguments, attacks)


  def give_cautious_subset_of_casebase(self, casebase):
    info("Preparing attack relations in the casebase (cautious)")
    self.reset_attack_relations(casebase, completely=True)
    debug("Sorting casebase")
    default = self.default_case
    casebase_without_default = [case for case in casebase if not case == default]
    cases_without_default = tsorted(casebase_without_default)
    ## guaranteeing default in the beginning,
    ## in case there is a case with minimal factorset
    debug("Sorted")
    
    self.casebase_active_ = [default]
    # non-parallel implementation, one by one
    stratum = []
    for next_case in cases_without_default:
      debug(f"Next case is {next_case}. Stratum is {stratum}")
      if any(case < next_case for case in stratum):        
        ### idea: if there is a smaller scase, then case is not minimal
        ### so it should be part of another stratum
        # debug("stratum closed, processing it")
        self._process_stratum(stratum)
        stratum = [] # new stratum
        debug("beginning new stratum")
      stratum = stratum + [next_case]
    else:
      debug(f"Processing last stratum: {stratum}")
      self._process_stratum(stratum)
    return self.casebase_active_
  
  def _select_from_stratum(self, stratum):
    to_add = []
    for case in stratum:
      predicted_outcome = self.predict([case])[0]
      if case.outcome != predicted_outcome:
        to_add = to_add + [case]
    return to_add

  def _process_stratum(self, stratum):
    to_add = self._select_from_stratum(stratum)
    for case in to_add:
      debug(f"simple_add case: {case.id}")
      self._simple_add(case)
    pass
    
  def incoherent_case_in_casebase(self, case):
    return any(self.inconsistent_pair(case, anothercase)
               for anothercase in self.casebase_active_)
  
  def _simple_add(self, newcase):
    # info(f"Adding {newcase}")
    # try:
    attacked = set(case for case in self.casebase_active_
                   if self.new_case_attacks(newcase, case))
    # except KeyError:
    #   raise(Exception(f"_simple_add failed since newcase {newcase} is not a key in attacked_by_."))
    to_attack = [case for case in self.casebase_active_
                 if case not in attacked # since those are not excluded below
                 and all((attacker_of_case in attacked
                         for attacker_of_case in self.attackers_of_[case]))  # defended
                 and different_outcomes(case, newcase)]
    self.casebase_active_.append(newcase)
    self.attacked_by_[newcase] = tsorted(to_attack)  # sorted might be unnecessary
    for case in to_attack:
      self.attackers_of_[case].append(newcase)
    pass
    
  # cautious casebase computed + resetting the attack relations
  def _old_give_cautious_subset_of_casebase(self, casebase):
    self.reset_attack_relations(casebase) # gpp
    ordered_casebase = self.topological_sort(casebase)
    info("Creating cautious casebase")
    # print(ordered_casebase)
    self.reset_attack_relations(ordered_casebase)
    
    current_casebase_set = [ordered_casebase[0]]  # default is the minimal one
    base_clf = Aacbr(outcome_def=self.outcome_def,
                     outcome_nondef=self.outcome_nondef,
                     outcome_unknown=self.outcome_unknown,
                     default_case=self.default_case,
                     cautious=False)
    current_casebase = base_clf.fit(current_casebase_set).casebase_active_
    unprocessed = ordered_casebase[1:]
    
    while unprocessed:
      stratum = [unprocessed[0]]
      # print("Current casebase is: {}".format(current_casebase))
      # print("Stratum is {}".format(stratum))
      unprocessed.pop(0)
      
      new_cases = base_clf.give_new_cases(current_casebase, stratum)
      
      # box, predicted_outcomes = base_clf.give_predictions(new_cases)
      predicted_outcomes = base_clf.predict(new_cases)
      to_add = []
      
      for index in range(len(stratum)):
        # print(f"Current case is {stratum[index]}")
        if stratum[index].outcome != predicted_outcomes[index]:
          inconsistent = False
          for case in current_casebase:
            if base_clf.inconsistent_pair(stratum[index], case):
              print(f"Incoherence found between:\n{stratum[index]} and {case}")
              inconsistent = True
              pass
              # GPP: Since cases are processed one by one,
              # we need to check for the incoherence. The
              # order is not a problem since only of the two
              # will pass the predicted_outcome test above.
              # That is, incoherence will happen iff the correct case
              # was already added, so the other should not be.
          if not inconsistent:
            self.reset_attack_relations([stratum[index]])
            to_add.append(stratum[index])

      # print("Adding the list of cases: {}".format(to_add))
      current_casebase_set.extend(to_add)
      # current_casebase = self.give_casebase(current_casebase_set)
      current_casebase = base_clf.fit(current_casebase_set).casebase_active_
  
    return current_casebase

  def give_new_cases(self, casebase, new_cases):
    """Calculates which cases are attacked by the new case.
    """
    for new_case in new_cases:
      self.give_new_case(casebase, new_case)
    return new_cases

  def give_new_case(self, casebase, new_case):
    self.reset_attack_relations([new_case])
    for case in casebase:
      if self.new_case_attacks(new_case, case):
        self.attacked_by_[new_case].append(case)
        # TODO?: we are not adding new_case to self.attackers_of[case], is
        # this an issue?
        # we do not do it since cleaning it afterwards would be harder
    return new_case
    
  def _old_give_casebase(self, cases):
    """Computes and stores the attack relation.
    """
    info("Preparing attack relations in the casebase")
    self.reset_attack_relations(cases, completely=True)
    casebase = []
    for candidate_case in cases:
      casebase.append(candidate_case)
      # GPP: why check for duplicates? I removed this.
      # duplicate = False
      # for case in casebase:
      #   if candidate_case.id != case.id and candidate_case.factors == case.factors and candidate_case.outcome == case.outcome:
      #     duplicate = True
      # if not duplicate:
      #   casebase.append(candidate_case)

    for case in casebase:
      for othercase in casebase:
        if self.past_case_attacks(case, othercase):
          self.attacked_by_[case].append(othercase)
          self.attackers_of_[othercase].append(case)

    return casebase

  def give_casebase(self, cases):
    """Computes and stores the attack relation.
    """
    info("Preparing attack relations in the casebase")
    self.reset_attack_relations(cases, completely=True)
    debug("Sorting casebase")
    default = self.default_case
    cases_without_default = tsorted([case for case in cases if not case == default])
    casebase = [default] + list(cases_without_default)
    ## guaranteeing default in the beginning,
    ## in case there is a case with minimal factorset
    debug("Sorted")
    self.casebase_active_ = casebase
    # Since cases are sorted, no case that comes before than attack
    # one that comes later.
    for idx,case in enumerate(casebase):
      for othercase in casebase[idx:]:
        debug(f"Considering pair:\n\t - case {case}\n\t - othercase {othercase}")
        if self.past_case_attacks(othercase, case):
          debug(f"othercase attacks case")
          self.attacked_by_[othercase].append(case)
          self.attackers_of_[case].append(othercase)
          if case.factors == othercase.factors and case != self.default_case:            
            # then they are the an incoherent pair, so we add directly
            # the attacks. default case does not attack.
            debug("case attacks othercase")
            self.attacked_by_[case].append(othercase)
            self.attackers_of_[othercase].append(case)

    return casebase

  def give_casebase_without_spikes(self, cases):
    """Gives casebase without "spikes", that is, without nodes that do not reach the default argument.
    This makes the comparison between cAACBR and AACBR much cleaner."""
    info("Preparing attack relations in the casebase (without spikes)")
    casebase = self.give_casebase(cases)
    to_keep = set([self.default_case])
    verified = set([self.default_case])
    unchecked = self.attackers_of_[self.default_case]
    while unchecked != []:
      case = unchecked.pop()
      to_keep.add(case)
      verified.add(case)
      unchecked += [x for x in self.attackers_of_[case] if x not in verified]
        
    clean_casebase = self.give_casebase(list(to_keep))
    
    SAFETY_CHECK=False
    if SAFETY_CHECK:
      assert to_keep == set(self.casebase_active_)
      non_attacking_cases = [case for case in self.casebase_active_
                             if self.attacked_by_[case] == []]
      # It is ok if the default attacks someone, it means there is an
      # inconsistency at the minimal characterisation.
      # The check here is whether there is no other attack, since that would be a spike.
      assert set(non_attacking_cases).issubset([self.default_case]), \
        f"Some in non_attacking_cases: {non_attacking_cases[:10]}"
    return clean_casebase
  
  def _old_give_casebase_without_spikes(self, cases):
    """Gives casebase without "spikes", that is, without nodes that do not reach the default argument.
    This makes the comparison between cAACBR and AACBR much cleaner."""
    info("Preparing attack relations in the casebase (without spikes)")
    casebase = self.give_casebase(cases)
    aaf = self.give_argumentation_framework()
    # print(set(cases).difference(set(mapping.keys())))
    nodes, edges = aaf
    graph = nx.DiGraph()
    if edges:
      graph.add_nodes_from(nodes)
      graph.add_edges_from(edges)
    else:
      nx.add_path(graph, nodes)
      
    new_nodes = set()
    def_arg = self.default_case
    for node in nodes:
      # print(f"node is {node}")
      if node not in new_nodes:
        try:
          path = nx.shortest_path(graph, source=node, target=def_arg)
          # print(f"path is {path}")
          for arg in list(path):
            new_nodes.add(arg)
        except nx.NetworkXNoPath:
          pass
    new_cases = [case for case in cases if case in new_nodes]
    clean_casebase = self.give_casebase(new_cases)
    return clean_casebase

  def topological_sort(self, casebase):
    info("Topological sorting")
    order_dag = self.build_order_dag(casebase, lt)
    output = self.topological_sort_graph(*order_dag)
    info("Topological sorting: done.")
    return output

  def build_order_dag(self, nodes, compare):
    """Returns a directed graph in which (a,b) is an edge iff
    compare(b,a), that is a is smaller than b in the partial
    order."""
    # Currently a very inefficient implementation, O(n^2)
    # I am sure this could be optimised
    edges = tuple((a,b) for a in nodes for b in nodes if compare(a,b))
    return (tuple(nodes), edges)
  
  def topological_sort_graph(self, nodes, edges):
    """Returns the list of nodes in topological sort order. Raises
    an error if graph is not a DAG. This uses Kahn's algorithm."""
    assert type(nodes) == tuple
    assert type(nodes[0]) != tuple
    assert type(edges) == tuple
    assert type(edges[0]) == tuple
    sorted_nodes = []
    # reversed_edges = [(b,a) for (a,b) in edges]
    points_to = {a:[b for b in nodes if (a,b) in edges] for a in nodes}
    pointed_by = {a:[b for b in nodes if (b,a) in edges] for a in nodes}
    stack = [a for a in nodes if pointed_by[a] == []]
    while stack != []:
      current = stack.pop()
      sorted_nodes.append(current)
      for node in points_to[current]:  # current points_to node
        pointed_by[node].remove(current)
        if pointed_by[node] == []:
          stack.append(node)
    remaining_nodes = set.difference(set(nodes), set(sorted_nodes))
    # now searches for a cycle
    if remaining_nodes != set():
      remaining_nodes = list(remaining_nodes)
      explored = []
      stack = deque([remaining_nodes[0]])
      cycle_found = False
      while stack != deque([]):
        current = stack.pop()
        explored.append(current)
        for node in pointed_by[current]:
          if node in explored:
            # cycle was found! "explored" represents it
            cycle_found = True
            break
          else:
            stack.append(node)
        if cycle_found:
          cycle_start = explored.index(node)
          explored.append(node)
          explored = explored[cycle_start:]
          break
      raise Exception(f"This graph is not a dag!\nThe remaining nodes are {remaining_nodes}.\nThis contains a cycle: {explored}")
    return sorted_nodes

  def draw_graph(self, output_dir=None, new_case=None, graph_name="graph",
                 engine="networkx", node_formatter=None):
    arguments, attacks = self.give_argumentation_framework(new_case)
    match engine:
      case "networkx":
        graph = giveGraph(arguments, attacks)
      case "graphviz":
        graph = arguments, attacks
      case _:
        raise(Exception(f"Unsupported {engine=}"))
    # strange, with commented out code below this draws a path from an
    # arbitrary leaf to the default
    # unclear why this was the implementation
    # sink = self.default_case
    # path = getPath(graph, [sink])
    # directed_path = giveGraph(path)
    # drawGraph(directed_path, graph_name, output_dir)
    drawGraph(graph, graph_name, output_dir, engine=engine,
              node_formatter=node_formatter)
    pass  
  
  def give_coherent_dataset(self, cases):
    # Returns a coherent casebase, but no guarantees on how.
    # Use cautious AA-CBR (cAACBR) for a coherent casebase for a
    # principled treatment. 
    casebase = []
    for candidate_case in cases:
      inconsistent = False
      for case in casebase:
        if self.inconsistent_pair(candidate_case, case):
          inconsistent = True
      if not inconsistent:
        casebase.append(candidate_case)
        
    return casebase

  def statistics(self):
    results = {}
    results["number of nodes"] = len(self.casebase_active_)
    results["total number of attacks"] = reduce(add, map(lambda x: len(self.attacked_by_[x]), self.casebase_active_), 0)
    double_count = reduce(add, map(lambda x: len(self.attackers_of_[x]), self.casebase_active_), 0)
    if not double_count == results["total number of attacks"]:
      warn("Double_count test failed, storing of attacks is wrong!")
      results["total number of attacks (double count)"] = double_count
    non_attacking = [str(x) for x in self.casebase_active_ if len(self.attacked_by_[x]) == 0][:10]
    info(f"Some of the non-attacking are {non_attacking}")
    unattacked = [str(x) for x in self.casebase_active_ if len(self.attackers_of_[x]) == 0][:10]
    info(f"Some of the unattacked are {unattacked}")
    results["distribution of attacks (out-going edges)"] = Counter((len(self.attacked_by_[case]) for case in self.casebase_active_))
    results["number of inconsistent cases"] = [self.incoherent_case_in_casebase(case) for case in self.casebase_active_].count(True)
    results["depth"] = self._framework_depth_statistics()
    return results

  def _framework_depth_statistics(self):
    """Filters "None" results."""
    depths = self.get_casebase_depths()
    depths = [d for d in depths.values() if d is not None]
    results = {"maximum": max(depths),
               "average": mean(depths),
               "stdev (population)": pstdev(depths)}
    return results
    
  def get_casebase_depths(self):
    """Distance to default. If no path from default to it, None."""
    depth = {c:None for c in self.casebase_active_}
    to_explore = deque([(self.default_case,-1)])
    while len(to_explore) != 0:
      current, previous_depth = to_explore.pop()
      current_depth = previous_depth + 1
      depth[current] = current_depth
      for next_case in self.attackers_of_[current]:
        if depth[next_case] is None:
          to_explore.appendleft((next_case, current_depth))
    return depth

  def adt_explain(self, new_case, mode="minimal"):
    """Returns explanation based on ArbitratedDisputeTree (ADT).

    Depends on mode:
    - mode=arbitrary: returns an arbitrated dispute tree, with no
      guarantees except that it is one.
    - mode=minimal: returns a minimal arbitrated dispute tree, in
      number of nodes.
    We do not consider ADTs in which an irrelevant case attacks another
    (since those are unnecessarily longer).

    TODO:
    - mode=all (TODO): returns all possible arbitrated dispute trees.
    Even in mode=all, not _every_ ADT would be generated (since those
    could be arbitrarily long with a 2-cycle, and due to the bias for
    the irrelevant case attack.
    """    # OK to implement in the feature _really every_ ADT, but irrelevant right now.
    accepted_modes = ["arbitrary", "minimal"]
    if mode not in accepted_modes:
      raise(Exception(f"{mode=} not implemented! Use one of {accepted_modes=}"))
    
    if type(new_case) == Case:
      # new_case = Case(f"new{new_case.id}", new_case.factors, outcome=None)
      new_case = new_case
    else:
      new_case = Case(f"new_case", new_case)

    grounded = self.grounded_extension(new_case, output_type="labelling")
    result = ArbitratedDisputeTree.create_adt(self, new_case, grounded, mode=mode)
    self.reset_attack_relations([new_case])
    return result
  
### Untested code below (legacy, kept "hidden" via underscore name)  
  def _compute_dialectically_box(self, graph, graph_level_map, prediction, root):
    ordered_graph_level_map = sorted(graph_level_map.items(), key=lambda node_level: node_level[1])
    str = ""

    for (node, level) in ordered_graph_level_map:
      if prediction == self.outcome_def:
        if level % 2:
          is_proposer = "L"
        else:
          is_proposer = "W"
      else:
        if level % 2:
          is_proposer = "W"
        else:
          is_proposer = "L"
      parent_mapping = nx.predecessor(graph, root)
      if node == root:
        str += is_proposer + ": " + node + "<br>"
      else:

        parents = parent_mapping[node]
        for parent in parents:
          str += is_proposer + ": " + node + " " + "attacks " + parent + "<br>"

    return str

  @staticmethod
  def _compute_excess_features(new_case, graph, format_mapping):
    excess_feature_map = {}
    excess_features = set()
    new_case_format = format_mapping[new_case]

    for (arg1, arg2) in graph.edges:
      attackee = None
      if new_case_format == arg1:
        attackee = arg2
      elif new_case_format == arg2:
        attackee = arg1

      if attackee:
        cases = [key for (key, value) in format_mapping.items() if value == attackee]
        for case in cases:
          for feature in case.factors:
            if feature not in new_case.factors:
              excess_features.add(feature)

    excess_feature_map.update({str(new_case.id): str(excess_features)})
    return excess_feature_map
  
  # remove arguments that do not attack to reduce the number of nodes and edges in the dispute tree
  def _remove_arguments_not_attackee(self, arguments, attacks):
    attackees = set()
    for (arg1, arg2) in attacks:
      if (arg1 in arguments) and (arg2 in arguments):
        attackees.add(arg1)
    if len(attackees) == len(arguments) - 1:
      return arguments, attacks
    else:
      filtered_arguments = set([args for args in arguments if args in attackees or "argument(default)" in args])
      filtered_attacks = set([(args1, args2) for (args1, args2) in attacks if
                  args2 in filtered_arguments])
      return self._remove_arguments_not_attackee(filtered_arguments, filtered_attacks)
  
  @staticmethod
  def _give_accuracy(predictions, new_cases):
    accurate_prediction = 0

    for index in range(len(new_cases)):
      if new_cases[index].outcome == predictions[index]['Prediction']:
        accurate_prediction += 1

    return accurate_prediction / float(len(predictions))

  @staticmethod
  def _check_coherent_predictions(predictions, new_cases):
    not_coherency = 0

    for index in range(len(predictions)):
      for other_index in range(len(predictions)):
        if new_cases[index].factors == new_cases[other_index].factors and predictions[index]['Prediction'] != \
            predictions[other_index]['Prediction']:
          not_coherency += 1

    return not_coherency

  # drawing the graph if needed
  def _draw_graph_old(self, graph, gname, outcome_map, graph_level_map, excess_feature_map, dialectical_box):
    graph_dir = os.path.join(os.getcwd(), 'graphsCAACBR')
    if not os.path.isdir(graph_dir):
      os.makedirs(graph_dir)
    graph_name = os.path.join(graph_dir, '{}.html'.format(gname))
    relative_positions = nx.fruchterman_reingold_layout(graph)

    x_nodes_position = [relative_positions[node][0] for node in graph.nodes]
    y_nodes_position = [relative_positions[node][1] for node in graph.nodes]
    x_edge_position = []
    y_edge_position = []

    for edge in graph.edges:
      x_edge_position.extend(
        [relative_positions[edge[0]][0], relative_positions[edge[1]][0], None])
      y_edge_position.extend(
        [relative_positions[edge[0]][1], relative_positions[edge[1]][1], None])

    figure = go.Figure()
    figure.add_traces([go.Scatter(x=x_edge_position,
                    y=y_edge_position,
                    mode='lines',
                    line=dict(color='rgb(210,210,210)', width=1),
                    hoverinfo='none'
                    ),
               go.Scatter(x=x_nodes_position,
                    y=y_nodes_position,
                    mode='markers',
                    name='bla',
                    marker=dict(symbol='circle-dot',
                          size=35,
                          color='#6175c1',
                          line=dict(color='rgb(50,50,50)', width=1)
                          ),
                    text=list(graph.nodes),
                    hoverinfo='text',
                    opacity=0.9
                    )])
    axis = dict(showline=False,
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          )

    figure.layout.update(title='Arbitrated Tree',
               annotations=self._make_annotations_old(relative_positions, list(graph.nodes), outcome_map,
                                 graph_level_map, graph.nodes, gname, excess_feature_map),
               font_size=12,
               showlegend=False,
               xaxis=axis,
               yaxis=axis,
               margin=dict(l=40, r=40, b=85, t=100),
               hovermode='closest',
               plot_bgcolor='rgb(248,248,248)'
               )

    figure.write_html(graph_name)

  def _make_annotations_old(self, pos, text, outcome_map, graph_level_map, nodes_graph, gname, excess_feature_map,
             font_size=12,
             font_color='rgb(26,11,11)'):
    # TODO: check relevance
    annotations = []
    index = 0

    for node in nodes_graph:
      argument_number = text[index].split(' ')[0].split('(')[1].split(')')[0]
      outcome = outcome_map[argument_number]
      max_level = 0
      level = graph_level_map.get(node, max_level)

      if outcome_map[gname] == self.outcome_def:
        if level % 2:
          is_proposer = "L"
        else:
          is_proposer = "W"
      else:
        if level % 2:
          is_proposer = "W"
        else:
          is_proposer = "L"

      annotations.append(
        dict(
          text=argument_number + ", outcome:" + outcome + ", " + is_proposer,
          x=pos[node][0], y=pos[node][1],
          xref='x1', yref='y1',
          font=dict(color=font_color, size=font_size),
          showarrow=True, arrowhead=3
        )
      )
      index += 1
    annotations.append(dict(
      x=0.5,
      y=1.15,
      showarrow=False,
      # arrowhead=1,
      text=excess_feature_map[gname],
      xref="paper",
      yref="paper",
      align='left',
      bordercolor='black',
      borderwidth=1
    ))


    return annotations

  # drawing the graph if needed

  # graph drawing
  @staticmethod
  def _give_graph_old(nodes, root, edges, grounded, new_case, unattacked):
    graph = nx.DiGraph()
    if edges:
      graph.add_nodes_from(nodes)
      graph.add_edges_from(edges)
    else:
      nx.add_path(graph, nodes)

    not_def = False
    edges = set()
    nodes = set()
    grounded1 = grounded.copy()
    nodes.add(root)

    for index in range(len(grounded)):
      arguments = grounded.pop()
      nodes.add(arguments)

      if arguments not in unattacked:
        try:
          paths = nx.all_shortest_paths(graph, source=new_case, target=arguments)
          if arguments == root:
            not_def = True
          for path in map(nx.utils.pairwise, paths):
            edges.update(set(path))
            for (arg1, arg2) in list(path):
              nodes.add(arg1)
              nodes.add(arg2)
        except:
          nodes.add(new_case)
          nodes.add(arguments)
      else:
        if new_case != arguments:
          try:
            paths = nx.all_shortest_paths(graph, source=arguments, target=root)
            if arguments == root:
              not_def = True
            for path in map(nx.utils.pairwise, paths):
              edges.update(set(path))
              for (arg1, arg2) in list(path):
                nodes.add(arg1)
                nodes.add(arg2)
          except:
            nodes.add(root)
            nodes.add(arguments)

    if not_def is False:
      for index in range(len(grounded1)):
        arguments = grounded1.pop()
        if arguments != new_case:
          try:
            paths = nx.all_shortest_paths(graph, source=root, target=arguments)
            for path in map(nx.utils.pairwise, paths):
              edges.update(set(path))
              for (arg1, arg2) in list(path):
                nodes.add(arg1)
                nodes.add(arg2)
          except:
            nodes.add(root)
            nodes.add(arguments)

    graph1 = nx.DiGraph()
    graph1.add_nodes_from(nodes)
    graph1.add_edges_from(edges)
    graph_level_map = nx.single_source_shortest_path_length(graph1, root)

    return graph_level_map, graph1
