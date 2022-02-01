# AA-CBR Basic Implementation

This provides a basic implementation of (a slightly extended variant of) the AA-CBR formalism [**Kristijonas Cyras, Ken Satoh, Francesca Toni**: *Abstract Argumentation for Case-Based Reasoning*. KR 2016: 549-552](https://dl.acm.org/doi/10.5555/3032027.3032100)

## Project structure

- `src`
  Main location for package.
  - `aacbr`
  Directory for aacbr package.

- `tests`
  Directory for testing, using Pytest.
<!-- `aacbr.py` -->
<!-- main file to run AA-CBR -->

<!-- `argumentation.py` -->
<!-- for constructing abstract argumentation frameworks and computing grounded labellings -->

<!-- `cases.py` -->
<!-- for constructing cases -->

<!-- `graphs.py` -->
<!-- for drawing directed graphs -->

<!-- `predictions.py`  -->
<!-- for generating AA-CBR predictions -->

<!-- `variables.py` -->
<!-- contains global variables for constructing cases and making predictions -->

<!-- There are a few `.json` files for testing functionality -->

## Requirements

Tested with **Python 3.9**. 

<!-- Needs the following libraries for drawing graphs (optional) -->
<!-- - networkx -->
<!-- - matplotlib -->

## Installing
The recommended way of using this repository is via [pipenv](https://pipenv.pypa.io). In this repository, simply use
```
pipenv shell
```
You will be able to check whether you are using the correct environment by `make check_python_version`.

Then, from `src` you will be able to run `import aacbr`.

## Running

Currently the supported and recommended interface is by defining cases via the `Case` class, as in the example below.

```python
  from aacbr import Aacbr, Case
  default = Case('default', set(), outcome=0)
  case1 = Case('1', {'a'}, outcome=1)
  case2 = Case('2', {'a','b'}, outcome=0)
  case3 = Case('3', {'a','b','c'}, outcome=0)
  example_cb = (default, case1, case2, case3)

  train_data = example_cb
  test_data = [Case('new1', {'a'}),
               Case('new2', {'a', 'b'}),
               Case('new3', {'a', 'c'}),
               Case('new4', {'a', 'b', 'c', 'd'}),
               Case('new5', set())]    
  expected_output = [1, 0, 1, 0, 0]
  clf = Aacbr()
  predicted_output = clf.fit(train_data).predict(test_data)
  assert expected_output == predicted_output
```

See `tests/` for more examples of input/output behaviour.

You can run tests by using `make test`.


## Roadmap (refactoring and debugging)
This project is under heavy refactoring and expected to change. We expect the API in the previous section to be stable, but it is subject to change.
 - [ ] Add a proper installation mechanism (via setup.py / setup.cfg)
 - [ ] Adapt to a more sklearn API style
 - [ ] (Re-)allow CLI usage via json files
 - [ ] Draw graphs
 - [ ] Allow cautious AA-CBR

<!-- ## Execution -->

<!-- aacbr.py executes AA-CBR, e.g.: -->

<!-- > python aacbr.py -->

<!-- - As an input, it takes two .json files - one with a casebase, another one with new cases - where each case is a dictionary with ID (default case should have ID 'default'), factors and outcome. -->

<!-- - Default file names are *cb.json* and *new.json*, as in *interact('cb', 'new')* -->

<!-- - Output containing predictions is dumped to *cb_to_new.json* -->

<!-- - Graphs for each new case visualising a single random directed path within AF are (optionally) dumped to .\graphs -->

***

