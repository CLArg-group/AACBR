# AA-CBR Basic Implementation

This provides a basic implementation of (a slightly extended variant of) the AA-CBR formalism [**Kristijonas Cyras, Ken Satoh, Francesca Toni**: *Abstract Argumentation for Case-Based Reasoning*. KR 2016: 549-552](https://dl.acm.org/doi/10.5555/3032027.3032100)

## Project structure

'aacbr.py'
main file to run AA-CBR

'argumentation.py'
for constructing abstract argumentation frameworks and computing grounded labellings

'cases.py'
for constructing cases

'graphs.py'
for drawing directed graphs

'predictions.py' 
for generating AA-CBR predictions

'variables.py'
contains global variables for constructing cases and making predictions

There are a few '.json' files for testing functionality

## Requirements

Tested with **Python 3.8.1**. 

Needs the following libraries for drawing graphs (optional)
- networkx
- matplotlib

## Execution

aacbr.py executes AA-CBR, e.g.:

> python aacbr.py

- As an input, it takes two .json files - one with a casebase, another one with new cases - where each case is a dictionary with ID (default case should have ID 'default'), factors and outcome.

- Default file names are *cb.json* and *new.json*, as in *interact('cb', 'new')*

- Output containing predictions is dumped to *cb_to_new.json*

- Graphs for each new case visualising a single random directed path within AF are (optionally) dumped to .\graphs

***

