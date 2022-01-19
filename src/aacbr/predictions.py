
from .argumentation import giveAAframework, computeGrounded
from .graphs import giveGraph, getPath, drawGraph
from .variables import *


def givePredictions(casebase: list, newcases: list) -> list:
  '''Returns a list of predictions given a casebase and new cases'''

  predictions = []
  for newcase in newcases:
    newcase_prediction = dict()
    number = newcases.index(newcase)
    Prediction = givePrediction(casebase, newcase, number)
    newcase_prediction.update({'No.': number, 'ID': newcase.id, 'Prediction': Prediction})
    predictions.append(newcase_prediction)
    
  return predictions
    

def givePrediction(casebase: list, newcase, number: int) -> dict:
  '''Returns an AA-CBR prediction given a casebase and a new case'''

  global ID_DEFAULT, ID_NON_DEFAULT, OUTCOME_DEFAULT, OUTCOME_NON_DEFAULT, OUTCOME_UNKNOWN

  prediction = None  
  aa_framework = giveAAframework(casebase, newcase)
  arguments = aa_framework['arguments']
  attacks = aa_framework['attacks']
  grounded = computeGrounded(arguments, attacks)
  def_arg = 'argument({})'.format(ID_DEFAULT)
  non_def_arg = 'argument({})'.format(ID_NON_DEFAULT)
  if def_arg in grounded['in'] and non_def_arg not in grounded['in']:
    prediction = OUTCOME_DEFAULT
    # comment the following line for one default; uncomment for two defaults
#    sink = non_def_arg
    # uncomment the following line for one default; comment for two defaults
    sink = def_arg
  # comment the following line for one default; uncomment for two defaults
#  elif non_def_arg in grounded['in'] and def_arg not in grounded['in']: 
  # uncomment the following line for one default; comment for two defaults
  elif def_arg not in grounded['in']: 
    prediction = OUTCOME_NON_DEFAULT
    sink = def_arg  
  else:
    prediction = OUTCOME_UNKNOWN
    sink = None

  # comment the following 5 lines to not produce the graphs (images)
  if sink:
    graph = giveGraph(arguments, attacks)
    path = getPath(graph, [sink])
    directed_path = giveGraph(path)
    drawGraph(directed_path, number)
  
  return prediction
