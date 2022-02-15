### This encapsulates AA-CBR as a model.

import sys
import os
import copy

from datetime import datetime
from datetime import timedelta
import time
import json

from functools import cmp_to_key
from operator import lt
from collections import deque, defaultdict

from .cases import Case, more_specific_weakly, different_outcomes
from .graphs import giveGraph, getPath, drawGraph
from .variables import OUTCOME_DEFAULT, OUTCOME_NON_DEFAULT, OUTCOME_UNKNOWN, ID_DEFAULT, ID_NON_DEFAULT


class Aacbr:
  def __init__(self, outcome_def=OUTCOME_DEFAULT, outcome_nondef=OUTCOME_NON_DEFAULT, outcome_unknown=OUTCOME_UNKNOWN, default_case=None, cautious=False):
    self.outcome_def = outcome_def
    self.outcome_nondef = outcome_nondef
    self.outcome_unknown = outcome_unknown
    self.default_case = default_case
    self._nondefault_case = None # the default for opposite outcome
    self.cautious = cautious
    # self.partial_order = None
    self.casebase_initial = None
    self.casebase_active = None
    self.attacked_by = defaultdict(list) # for storing inside an Aacbr instance the attacks, not in the cases # attacked_by[a] = {b such that a attacks b}
    self.attackers_of = defaultdict(list)  # attackers_of[a] = {b such that b attacks a}

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

  # attacks(self, case1, case2)
  # is_attacked(self, case1, case2)
  
  def fit(self, casebase=set()):
    if not isinstance(self, Aacbr):
      raise(Exception(f"{self} is not an instance of {Aacbr}"))
    
    self.casebase_initial = casebase
    self.infer_default(casebase)
    # self.partial_order = partial_order
    if not self.cautious:
      self.casebase_active = casebase
      self.casebase_active = self.give_casebase(casebase) # adding attacks
      # command 1: filter which cases to use -- but if this depends on attack, do I have to fit in order to define attack? this is strange
      # -- but this is no problem for typical aacbr, as long as we separate saving attack state from filtering
      # command 2: save attacks state
    else:
      self.casebase_active = []
      self.casebase_active = self.give_cautious_subset_of_casebase(casebase)
      self.give_casebase(self.casebase_active)
      # raise(Exception("Cautious case not implemented"))
    return self
  def infer_default(self, casebase):
    default_in_input = self.default_in_casebase(casebase)
    if default_in_input:
      self.default_case = default_in_input
    elif type(self.default_case) == Case:
      if self.outcome_def != self.default_case.outcome:
        raise(RuntimeError(f"Default case outcome is not the same as as passed default outcome: {self.default_case.outcome} != {self.outcome_def}"))
      self.outcome_def = self.default_case.outcome
    elif all([type(case.factors) == set for case in casebase]):
      self.default_case = Case("default", set(), outcome=self.outcome_def)
    else:
      raise(RuntimeError("No default case to use!"))

  @staticmethod
  def default_in_casebase(casebase):
    try:
      idx = [case.id for case in casebase].index(ID_DEFAULT)
      return casebase[idx]
    except ValueError:
      return False

  def reset_attack_relations(self, casebase):
    # self.attackers_of.clear()
    # self.attacked_by.clear()
    # TODO: does not work if I am removing just some cases from the AF, I should make a different interface for new cases, which I add and remove
    for case in casebase:
      self.attackers_of[case] = []
      self.attacked_by[case] = []
  
  # not something in the set of all possible cases in the middle
  def most_concise(self, cases, A, B):
    return not any((B < case and
                    case < A and
                    not (different_outcomes(A, case)))
                   for case in cases)

  # attack relation defined
  def past_case_attacks(self, A, B):
    if not all(x in self.casebase_active for x in (A,B)):
      raise RuntimeError(f"Arguments {(A,B)} are not both in the active casebase.")
    
    return (different_outcomes(A, B) and
            B <= A and
            self.most_concise(self.casebase_active, A, B))

  # unlabbled datapoint new_case
  @staticmethod
  def new_case_attacks(new_case, targetcase):
    # return sum(targetcase.weight) > sum(new_case.weight)
    return not new_case.factors.issuperset(targetcase.factors)

  # noisy points
  def inconsistent_attacks(self, A, B):
    return different_outcomes(A, B) and B.factors == A.factors

  def predict(self, new_cases):
    predictions = self.give_predictions(new_cases)
    # return [prediction_dict["Prediction"] for prediction_dict in predictions]
    return predictions
  
  # predictions for multiple points
  def give_predictions(self, new_cases, nr_defaults=1):
    casebase = self.casebase_active
    # new_cases = self.give_new_cases(casebase, new_cases)
    predictions = []
    for new_case in new_cases:
      new_case_prediction = dict()
      number = new_cases.index(new_case)
      prediction = self.give_prediction(new_case, nr_defaults, number)
      predictions.append(prediction)
    formatted = self.format_predictions(new_cases, predictions)
    # return dialectical_box, predictions
    return predictions

  @staticmethod
  def format_predictions(new_cases, predictions):
    return [{'No.': number,
             'ID': new_case.id,
             'Prediction': prediction}
            for (number,(new_case, prediction))
                 in enumerate(zip(new_cases, predictions))]
  
  
  def give_prediction(self, new_case, nr_defaults: int = 1, number: int = 0):
    '''Returns an AA-CBR prediction given a casebase and a new case'''
    grounded = self.grounded_extension(new_case, output_type="labelling")
    def_arg = self.default_case
    non_def_arg = self._nondefault_case
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

    # dialectical_box = None
    # # graph drawing part
    # if sink and cautious and outcome_map is False:
    #   new_case_format = format_mapping[new_case]
    #   graph_level_map, graph = self.give_graph(arguments, def_arg, attacks, grounded['in'], new_case_format, unattacked)
    #   outcome_map.update({str(new_case.id): prediction})
    #   excess_feature_map = self.compute_excess_features(new_case, graph, format_mapping)
    #   dialectical_box = self.compute_dialectically_box(graph, graph_level_map, prediction, def_arg)
    #   self.draw_graph(graph, new_case.id, outcome_map, graph_level_map, excess_feature_map, dialectical_box)

    return prediction

  
  def grounded_extension(self, new_case, output_type="subset"):
    if not output_type in ("subset", "labelling"):
      raise RuntimeError(f"output_type argument should be either 'subset' or 'labelling'")
    if not type(new_case) == Case:
      raise RuntimeError(f"new_case argument needs to be of type {Case}, but is {type(new_case)}")
    casebase = self.casebase_active
    new_case = self.give_new_case(casebase, new_case)
    arguments, attacks = self.give_argumentation_framework(new_case)
    grounded = self._compute_grounded(arguments, attacks)
    if output_type == "subset":
      return grounded["in"]
    else:
      return grounded

  @staticmethod
  def _compute_grounded(args: set, att: set):
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
    
    
  def compute_dialectically_box(self, graph, graph_level_map, prediction, root):
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
  def compute_excess_features(new_case, graph, format_mapping):
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

  def give_argumentation_framework(self, new_case = None) -> tuple:
    '''Returns an abstract argumentation framework (AAF) given a
    casebase and, optionally, a new case.
    The AAF is returned as a pair (arguments, attacks).'''
    casebase = self.casebase_active
    arguments = set()
    attacks = set()
    arguments = {case for case in casebase}
    for case in casebase:
      for attacker in self.attackers_of[case]:
        attacks.add((attacker, case))
    if new_case != None:
      arguments.add(new_case)
      for attacked in self.attacked_by[new_case]:
        attacks.add((new_case, attacked))

    return (arguments, attacks)
  

  # remove arguments that do not attack to reduce the number of nodes and edges in the dispute tree
  def remove_arguments_not_attackee(self, arguments, attacks):
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
      return self.remove_arguments_not_attackee(filtered_arguments, filtered_attacks)

  # cautious casebase computed + resetting the attack relations
  def give_cautious_subset_of_casebase(self, casebase):
    self.reset_attack_relations(casebase) # gpp
    ordered_casebase = self.topological_sort(casebase)
    # print(ordered_casebase)
    self.reset_attack_relations(ordered_casebase)
    
    current_casebase_set = [ordered_casebase[0]]  # default is the minimal one
    base_clf = Aacbr(outcome_def=self.outcome_def,
                     outcome_nondef=self.outcome_nondef,
                     outcome_unknown=self.outcome_unknown,
                     default_case=self.default_case,
                     cautious=False)
    current_casebase = base_clf.fit(current_casebase_set).casebase_active
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
            if base_clf.inconsistent_attacks(stratum[index], case):
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
      current_casebase = base_clf.fit(current_casebase_set).casebase_active
  
    return current_casebase

  #comment out if the partial order is subset
  # def give_cautious_subset_of_casebase(self, casebase):
  #   current_casebase_set = [casebase[0]]
  #   current_casebase = self.give_casebase(current_casebase_set)
  #   unprocessed = casebase[1:]
  #
  #   while unprocessed:
  #
  #     min_len = len(unprocessed[0].factors)
  #     stratum = list(filter(lambda case: len(case.factors) == min_len, unprocessed))
  #     unprocessed = [item for item in unprocessed if item not in stratum]
  #     new_cases = self.give_new_cases(current_casebase, stratum)
  #
  #     predicted_outcomes = self.give_predictions(current_casebase, new_cases, 1)
  #     to_add = []
  #
  #     for index in range(len(stratum)):
  #       if stratum[index].outcome != predicted_outcomes[index]['Prediction']:
  #         self.attacked_by[stratum[index]] = []
  #         self.attackers_of[stratum[index]] = []
  #         to_add.append(stratum[index])
  #
  #     current_casebase_set.extend(to_add)
  #     current_casebase = self.give_casebase(current_casebase_set)
  #
  #   return current_casebase

  # calculate which cases are attacked by the new case
  def give_new_cases(self, casebase, new_cases):
    for new_case in new_cases:
      self.give_new_case(casebase, new_case)
    return new_cases

  def give_new_case(self, casebase, new_case):
    self.reset_attack_relations([new_case])
    for case in casebase:
      if self.new_case_attacks(new_case, case):
        self.attacked_by[new_case].append(case)
        # TODO?: we are not adding new_case to self.attackers_of[case], is
        # this an issue?
        # we do not do it since cleaning it afterwards would be harder
    return new_case

  # give casebase-training dataset # (remove the duplicates) -- no!
  # compute the attackees and attackers set
  def give_casebase(self, cases):
    self.reset_attack_relations(cases) # gpp
    casebase = []
    for candidate_case in cases:
      casebase.append(candidate_case)
      # duplicate = False
      # for case in casebase:
      #   if candidate_case.id != case.id and candidate_case.factors == case.factors and candidate_case.outcome == case.outcome:
      #     duplicate = True
      # if not duplicate:
      #   casebase.append(candidate_case)

    for case in casebase:
      for othercase in casebase:
        if self.past_case_attacks(case, othercase):
          self.attacked_by[case].append(othercase)
          self.attackers_of[othercase].append(case)

    return casebase

  def give_casebase_without_spikes(self, cases):
    """Gives casebase without "spikes", that is, without nodes that do not reach the default argument.
    This makes the comparison between cAACBR and AACBR much cleaner."""
    casebase = self.give_casebase(cases)
    aaf, mapping = self.give_argumentation_framework()
    # print(set(cases).difference(set(mapping.keys())))
    nodes, edges = tuple(aaf["arguments"]), tuple(aaf["attacks"])
    graph = nx.DiGraph()
    if edges:
      graph.add_nodes_from(nodes)
      graph.add_edges_from(edges)
    else:
      nx.add_path(graph, nodes)

    new_nodes = set()
    [def_arg] = [x for x in nodes if "default" in x]
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
    new_cases = [case for case in cases if mapping[case] in new_nodes]
    clean_casebase = self.give_casebase(new_cases)
    return clean_casebase

  # def topological_sort(self, casebase):
  #   no_attackers = {}

  #   for case in casebase:
  #     no_attackers.update({case: len(case.attackers)})

  #   unattacked = [case for case in casebase if no_attackers[case] == 0]
  #   order_casebase = []
  #   while unattacked:
  #     first_case = unattacked.pop(0)
  #     order_casebase.append(first_case)

  #     for attackee in self.attacked_by[first_case]:
  #       no_attackers[attackee] -= 1
  #       if no_attackers[attackee] == 0:
  #         unattacked.append(attackee)

  #   return order_casebase
  def topological_sort(self, casebase):
    # compare = lambda x,y: 1 if more_specific(x,y) else (0 if x.factors == y.factors else -1)
    order_dag = self.build_order_dag(casebase, lt)
    output = self.topological_sort_graph(*order_dag)
    return output
    # return sorted(casebase, key=cmp_to_key(comparison_function))
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

  # partial order = subset; order the arguments with respect to partial order # gp18: wrong
  # @staticmethod
  # def min_ordering_casebase(casebase):
  #   sorted_cards = sorted(casebase, key=lambda case: len(case.factors))
  #   return sorted_cards

  # read the cases from the JSON file
  def load_cases(self, file, nr_defaults=1, lime=None, mapping_name=None):
    cases = []
    with open(file, encoding='utf-8') as json_file:
      entries = json.load(json_file)
      for entry in entries:
        if entry['id'] == self.ID_DEFAULT:
          default_case = Case(self.ID_DEFAULT, set(), self.outcome_def)
          cases.insert(0, default_case)

          if nr_defaults == 2:
            non_default_case = Case(self.ID_NON_DEFAULT, set(), self.outcome_nondef, [], [], 0)
            cases.insert(0, non_default_case)
        else:
          if lime:
            factors = self.lime_preprocessing(entry['factors'], mapping_name)
          else:
            factors = entry['factors']
          case = Case(entry['id'], set(factors), entry['outcome'])
          cases.append(case)
    json_file.close()

    return cases

  def give_coherent_dataset(self, cases):
    casebase = []
    for candidate_case in cases:
      inconsistent = False
      for case in casebase:
        if self.inconsistent_attacks(candidate_case, case):
          inconsistent = True
      if not inconsistent:
        casebase.append(candidate_case)

    return casebase

  @staticmethod
  def give_accuracy(predictions, new_cases):
    accurate_prediction = 0

    for index in range(len(new_cases)):
      if new_cases[index].outcome == predictions[index]['Prediction']:
        accurate_prediction += 1

    return accurate_prediction / float(len(predictions))

  @staticmethod
  def check_coherent_predictions(predictions, new_cases):
    not_coherency = 0

    for index in range(len(predictions)):
      for other_index in range(len(predictions)):
        if new_cases[index].factors == new_cases[other_index].factors and predictions[index]['Prediction'] != \
            predictions[other_index]['Prediction']:
          not_coherency += 1

    return not_coherency

  def draw_graph(self, new_case=None, graph_name="graph", output_dir=None):
    sink = self.default_case
    arguments, attacks = self.give_argumentation_framework(new_case)
    graph = giveGraph(arguments, attacks)
    path = getPath(graph, [sink])
    directed_path = giveGraph(path)
    drawGraph(directed_path, graph_name, output_dir)
    pass

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
