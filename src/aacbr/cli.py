#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import sys
import os

from datetime import datetime
from datetime import timedelta
import time

import json

from .cases import load_cases, give_casebase, give_newcases
from .predictions import givePredictions

from .variables import *


def interact(cases_filename: str, newcases_filename: str):
  '''Asks the user to provide files with cases and new cases
    and outputs AA-CBR predictions in .json format'''

  print('{now}: Started'.format(now = datetime.now().strftime('%Y-%m-%d %H.%M.%S')))
  cases_file = os.path.join(os.getcwd(), '{}.json'.format(cases_filename))
  if not os.path.isfile(cases_file):
    return print('Casebase file {} not found. Try again.\n'.format(cases_filename))  
  else:      
    cases = load_cases(cases_file)
    casebase = give_casebase(cases)   
  newcases_file = os.path.join(os.getcwd(), '{}.json'.format(newcases_filename))
  if not os.path.isfile(newcases_file):
    return print('New cases file {} not found. Try again.\n'.format(newcases_filename))  
  else:
    new_cases = load_cases(newcases_file)
    newcases = give_newcases(casebase, new_cases)
  
  Predictions = givePredictions(casebase, newcases)

  predictions_output_filename = '{c}_to_{n}.json'.format(c = cases_filename, n = newcases_filename)
  with open(os.path.join(os.getcwd(), predictions_output_filename), 'w', newline = '', encoding = 'utf-8') as output:
    json.dump(Predictions, output, indent = 4, ensure_ascii = False)
  output.close()
    
  return print('{now}: Done. Predictions dumped to {file}'.format(now = datetime.now().strftime('%Y-%m-%d %H.%M.%S'), file = predictions_output_filename))

def main():
        interact('cb', 'new')

if __name__ == "__main__":
        main()

