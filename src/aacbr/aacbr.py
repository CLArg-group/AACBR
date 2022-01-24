### This encapsulates AA-CBR as a model.

import sys
import os
import copy

from datetime import datetime
from datetime import timedelta
import time
import json

from .cases import Case
import numpy as np

from functools import cmp_to_key
from collections import deque

class Aacbr:
  ID_DEFAULT = 'default'
  ID_NON_DEFAULT = 'non_default'

  def __init__(self, outcome_def, outcome_nondef, outcome_unknown):
    self.outcome_def = outcome_def
    self.outcome_nondef = outcome_nondef
    self.outcome_unknown = outcome_unknown

  @staticmethod
  def different_outcomes(A, B):
    return A.outcome != B.outcome

  # A is a superset of B -> partial order
  @staticmethod
  def more_specific(A, B):
    # return sum(A.weight) > sum(B.weight)
    # return B.factors.issubset(A.factors) and B.factors != A.factors # this disallows incoherence
    return B.factors.issubset(A.factors) and (B.factors != A.factors or B.outcome != A.outcome)

  # not something in the set of all possible cases in the middle
  def most_concise(self, cases, A, B):
    return not any(
      (self.more_specific(A, case) and self.more_specific(case, B) and not (self.different_outcomes(A, case))) for
      case in cases)

  # attack relation defined
  def attacks(self, cases, A, B):
    return self.different_outcomes(A, B) and self.more_specific(A, B) and self.most_concise(cases, A, B)

  # unlabbled datapoint newcase
  @staticmethod
  def new_case_attacks(newcase, targetcase):
    # return sum(targetcase.weight) > sum(newcase.weight)
    return not newcase.factors.issuperset(targetcase.factors)

  # noisy points

  def inconsistent_attacks(self, A, B):
    return self.different_outcomes(A, B) and B.factors == A.factors

  # predictions for multiple points
  def give_predictions(self, casebase, newcases, nr_defaults=1, lime=None, outcome_map=None, cautious=None):
    predictions = []
    for newcase in newcases:
      newcase_prediction = dict()
      number = newcases.index(newcase)
      aa_framework, format_mapping = self.format_aaframework(casebase, newcase)
      dialectical_box, prediction = self.give_prediction(aa_framework, nr_defaults, number, newcase, outcome_map,
                                 format_mapping,
                                 cautious, lime)
      newcase_prediction.update({'No.': number, 'ID': newcase.id, 'Prediction': prediction})
      predictions.append(newcase_prediction)

    return dialectical_box, predictions

  def give_prediction(self, framework, nr_defaults, number, newcase, outcome_map, format_mapping, cautious, lime):
    arguments = framework['arguments']
    attacks = framework['attacks']
    grounded, unattacked = self.compute_grounded(arguments, attacks)
    def_arg = 'argument({})'.format(self.ID_DEFAULT) + " " + 'factors:{}'.format('set()')
    non_def_arg = 'argument({})'.format(self.ID_NON_DEFAULT) + " " + 'factors:{}'.format('set()')
    prediction = None
    dialectical_box = None

    if nr_defaults == 1:

      if def_arg in grounded['in'] and non_def_arg not in grounded['in']:
        prediction = self.outcome_def
        sink = def_arg
      elif def_arg not in grounded['in']:
        prediction = self.outcome_nondef
        sink = def_arg
      else:
        prediction = self.outcome_unknown
        sink = None

    # graph drawing part
    if sink and cautious and outcome_map and lime is False:
      newcase_format = format_mapping[newcase]
      graph_level_map, graph = self.give_graph(arguments, def_arg, attacks, grounded['in'], newcase_format,
                           unattacked)
      outcome_map.update({str(newcase.id): prediction})
      excess_feature_map = self.compute_excess_features(newcase, graph, format_mapping)
      dialectical_box = self.compute_dialectically_box(graph, graph_level_map, prediction, def_arg)
      self.draw_graph(graph, newcase.id, outcome_map, graph_level_map, excess_feature_map, dialectical_box)

    return dialectical_box, prediction

  # | operator means union of 2 sets; IN computes the unattacked arguments mainly G0(in the first step); OUT means the
  # arguments not in
  # the grounding. It computes the unattacked args first and afterwards it removes the form the remaing ones and i
  # continues.

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
  def compute_grounded(args: set, att: set):
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

    return {'in': IN, 'out': OUT, 'undec': UNDEC}, unattacked

  @staticmethod
  def compute_excess_features(newcase, graph, format_mapping):
    excess_feature_map = {}
    excess_features = set()
    newcase_format = format_mapping[newcase]

    for (arg1, arg2) in graph.edges:
      attackee = None
      if newcase_format == arg1:
        attackee = arg2
      elif newcase_format == arg2:
        attackee = arg1

      if attackee:
        cases = [key for (key, value) in format_mapping.items() if value == attackee]
        for case in cases:
          for feature in case.factors:
            if feature not in newcase.factors:
              excess_features.add(feature)

    excess_feature_map.update({str(newcase.id): str(excess_features)})
    return excess_feature_map

  # abstract argumentation framework formatted
  # arguments and attacks
  def format_aaframework(self, casebase, newcase=None):
    arguments = set()
    attacks = set()
    format_mapping = {}
    for case in casebase:
      arguments.add('argument({})'.format(str(case.id)) + " " + 'factors:{}'.format(str(case.factors)))
      format_mapping.update(\
        {case: 'argument({})'.format(str(case.id)) + " " + 'factors:{}'.format(str(case.factors))})
      for attackee in case.attackees:
        attacks.add(('argument({attacker_argument})'.format(attacker_argument=str(case.id)) + " " +
               'factors:{}'.format(str(case.factors)),
               'argument({attacked_argument})'.format(attacked_argument=str(attackee.id)) + " " +
               'factors:{}'.format(str(attackee.factors))))
      for attacker in case.attackers:
        attacks.add(('argument({attacker_argument})'.format(attacker_argument=str(attacker.id)) + " " +
               'factors:{}'.format(str(attacker.factors)),
               'argument({attacked_argument})'.format(attacked_argument=str(case.id)) + " " +
               'factors:{}'.format(str(case.factors))))
    if newcase != None:
      newcase_format = 'argument({})'.format(str(newcase.id)) + " " + 'factors:{}'.format(str(newcase.factors))
      # arguments1, attacks1 = self.remove_arguments_not_attackee(arguments, attacks)
      arguments.add(newcase_format)
      format_mapping.update({newcase: newcase_format})
      for attackee in newcase.attackees:
        attacks.add((newcase_format,
               'argument({attacked_argument})'.format(attacked_argument=str(attackee.id)) + " " +
               'factors:{}'.format(str(attackee.factors))))

    return {'arguments': arguments, 'attacks': attacks}, format_mapping

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
    current_casebase = self.give_casebase(current_casebase_set)
    unprocessed = ordered_casebase[1:]
    
    while unprocessed:
      stratum = [unprocessed[0]]
      # print("Current casebase is: {}".format(current_casebase))
      # print("Stratum is {}".format(stratum))
      unprocessed.pop(0)
      
      newcases = self.give_new_cases(current_casebase, stratum)
      
      box, predicted_outcomes = self.give_predictions(current_casebase, newcases, 1, lime=False)
      to_add = []
      
      for index in range(len(stratum)):
        # print(f"Current case is {stratum[index]}")
        if stratum[index].outcome != predicted_outcomes[index]['Prediction']:
          inconsistent = False
          for case in current_casebase:
            if self.inconsistent_attacks(stratum[index], case):
              print(f"Incoherence found between:\n{stratum[index]} and {case}")
              inconsistent = True
          if not inconsistent:
            # GPP: Since cases are processed one by one,
            # we need to check for the incoherence. The
            # order is not a problem since only of the two
            # will pass the predicted_outcome test above.
            stratum[index].attackees = []
            stratum[index].attackers = []
            to_add.append(stratum[index])      

      # print("Adding the list of cases: {}".format(to_add))
      current_casebase_set.extend(to_add)
      current_casebase = self.give_casebase(current_casebase_set)

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
  #     newcases = self.give_new_cases(current_casebase, stratum)
  #
  #     predicted_outcomes = self.give_predictions(current_casebase, newcases, 1)
  #     to_add = []
  #
  #     for index in range(len(stratum)):
  #       if stratum[index].outcome != predicted_outcomes[index]['Prediction']:
  #         stratum[index].attackees = []
  #         stratum[index].attackers = []
  #         to_add.append(stratum[index])
  #
  #     current_casebase_set.extend(to_add)
  #     current_casebase = self.give_casebase(current_casebase_set)
  #
  #   return current_casebase

  # calculate which cases are attacked by the new case
  def give_new_cases(self, casebase, cases):
    newcases = []

    for newcase in cases:
      for case in casebase:
        if self.new_case_attacks(newcase, case):
          newcase.attackees.append(case)
      newcases.append(newcase)

    return newcases

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
        if self.attacks(casebase, case, othercase):
          case.attackees.append(othercase)
          othercase.attackers.append(case)

    return casebase

  def give_casebase_without_spikes(self, cases):
    """Gives casebase without "spikes", that is, without nodes that do not reach the default argument.
    This makes the comparison between cAACBR and AACBR much cleaner."""
    casebase = self.give_casebase(cases)
    aaf, mapping = self.format_aaframework(casebase)
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

  #     for attackee in first_case.attackees:
  #       no_attackers[attackee] -= 1
  #       if no_attackers[attackee] == 0:
  #         unattacked.append(attackee)

  #   return order_casebase
  def topological_sort(self, casebase):
    # compare = lambda x,y: 1 if self.more_specific(x,y) else (0 if x.factors == y.factors else -1)
    order_dag = self.build_order_dag(casebase, self.more_specific)
    output = self.topological_sort_graph(*order_dag)
    return output
    # return sorted(casebase, key=cmp_to_key(comparison_function))
  def build_order_dag(self, nodes, compare):
    """Returns a directed graph in which (a,b) is an edge iff
    compare(b,a), that is a is smaller than b in the partial
    order."""
    # Currently a very inefficient implementation, O(n^2)
    # I am sure this could be optimised
    edges = tuple((a,b) for a in nodes for b in nodes if compare(b,a))
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
          default_case = Case(self.ID_DEFAULT, set(), self.outcome_def, [], [], 0)
          cases.insert(0, default_case)

          if nr_defaults == 2:
            non_default_case = Case(self.ID_NON_DEFAULT, set(), self.outcome_nondef, [], [], 0)
            cases.insert(0, non_default_case)
        else:
          if lime:
            factors = self.lime_preprocessing(entry['factors'], mapping_name)
          else:
            factors = entry['factors']
          case = Case(entry['id'], set(factors), entry['outcome'], [], [], 0)
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
  def give_accuracy(predictions, newcases):
    accurate_prediction = 0

    for index in range(len(newcases)):
      if newcases[index].outcome == predictions[index]['Prediction']:
        accurate_prediction += 1

    return accurate_prediction / float(len(predictions))

  @staticmethod
  def check_coherent_predictions(predictions, newcases):
    not_coherency = 0

    for index in range(len(predictions)):
      for other_index in range(len(predictions)):
        if newcases[index].factors == newcases[other_index].factors and predictions[index]['Prediction'] != \
            predictions[other_index]['Prediction']:
          not_coherency += 1

    return not_coherency

  # drawing the graph if needed
  def draw_graph(self, graph, gname, outcome_map, graph_level_map, excess_feature_map, dialectical_box):
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
               annotations=self.make_annotations(relative_positions, list(graph.nodes), outcome_map,
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

  def make_annotations(self, pos, text, outcome_map, graph_level_map, nodes_graph, gname, excess_feature_map,
             font_size=12,
             font_color='rgb(26,11,11)'):
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
  def give_graph(nodes, root, edges, grounded, newcase, unattacked):
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
          paths = nx.all_shortest_paths(graph, source=newcase, target=arguments)
          if arguments == root:
            not_def = True
          for path in map(nx.utils.pairwise, paths):
            edges.update(set(path))
            for (arg1, arg2) in list(path):
              nodes.add(arg1)
              nodes.add(arg2)
        except:
          nodes.add(newcase)
          nodes.add(arguments)
      else:
        if newcase != arguments:
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
        if arguments != newcase:
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

  @staticmethod
  def get_outcome_map(cases):
    outcome_map = {}
    for case in cases:
      outcome_map.update({str(case.id): str(case.outcome)})
    return outcome_map

  def reset_attack_relations(self, casebase):
    for case in casebase:
      case.attackers = []
      case.attackees = []

  # @staticmethod
  # def lime_preprocessing(factors, mapping_name):
  #   reverse_encoded_factors = []
  #   for index in range(len(factors)):
  #     key = mapping_name[index][int(factors[index])]
  #     if key != 'Native American' and key != '25 - 45':
  #       reverse_encoded_factors = np.append(reverse_encoded_factors, key)

  #   return reverse_encoded_factors

