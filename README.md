# AA-CBR Basic Implementation

This provides an implementation of the AA-CBR formalism [Abstract Argumentation for Case-Based Reasoning][1].

This includes an implementation of cautiously monotonic AA-CBR (cAA-CBR), presented in [Monotonicity and Noise-Tolerance in Case-Based Reasoning with Abstract Argumentation][2], and supports general partial orders, as originally introduced in [Data-Empowered Argumentation for Dialectically Explainable Predictions][3], but **not** arbitrary indifference relations (i.e. AA-CBR is assumed to be regular).

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

Alternatively, you may also install with pip:
```
pip install .
```
This will install the package `aacbr`.

## Running

Currently the supported and recommended interface is by defining cases via the `Case` class, as in the example below.

```python
  from aacbr import Aacbr, Case
  cb = [Case('default', set(), outcome=0),
        Case('1', {'a'}, outcome=1),
        Case('2', {'a','b'}, outcome=0),
        Case('3', {'c'}, outcome=1),
        Case('4', {'c','d'}, outcome=0),
        Case('5', {'a','b','c'}, outcome=1)]
  
  train_data = cb
  test_data = [Case('new1', {'a'}),
               Case('new2', {'a', 'b', 'e'}),
               Case('new3', {'a', 'c'}),
               Case('new4', {'a', 'b', 'c'}),
               Case('new5', set())]
  expected_output = [1, 0, 1, 1, 0]
  clf = Aacbr()
  clf.fit(train_data) # trains the classifier, that is, adds the casebase
  predicted_output = clf.predict(test_data)
  assert expected_output == predicted_output
```

By default, this is the traditional (non-cautiously monotonic) AA-CBR:
```python
  test_data = [Case('new6', {'a', 'b', 'c', 'd'})]
  predicted_output = clf.predict(test_data)
  assert predicted_output == [1]
```
One can define the cautiously monotonic AA-CBR by the `cautious` flag:
```python
  test_data = [Case('new6', {'a', 'b', 'c', 'd'})]
  clf = Aacbr(cautious=True)
  clf.fit(train_data)
  predicted_output = clf.predict(test_data)
  assert predicted_output == [0]
```
It is also possible to get AA-CBR without "spikes", that is, arguments from which there is no path to the default argument (in a graph-theoretic sense):
```python
   default = Case('default', set(), outcome=0)
   case1 = Case('1', {'a'}, outcome=1)
   case2 = Case('2', {'a','b'}, outcome=0)
   case3 = Case('3', {'b'}, outcome=0)
   case4 = Case('4', {'c'}, outcome=0)
   case5 = Case('5', {'a','c'}, outcome=1)
   case6 = Case('6', {'a','b','c'}, outcome=0)
   cb = (default, case1, case2, case3, case4, case5, case6)
   filtered_cb = {default, case1, case2}
   clf = Aacbr().fit(cb, remove_spikes=True)
   assert set(clf.casebase_active) == filtered_cb
```

A more "scikit-learn-style" interface is also available:
```python
   train_X = [set(),
              {'a'},
              {'a','b'},
              {'a','b','c'}]
   train_Y = [0,
              1,
              0,
              0]
   test_X = [{'a'},
             {'a', 'b'},
             {'a', 'c'},
             {'a', 'b', 'c', 'd'},
             set()]
   expected_output = [1, 0, 1, 0, 0]
   clf = Aacbr()
   clf.fit(train_X, train_Y)
   
   default = Case('default', set(), outcome=0)
   case0 = Case("0", set(), outcome=0)
   case1 = Case('1', {'a'}, outcome=1)
   case2 = Case('2', {'a','b'}, outcome=0)
   case3 = Case('3', {'a','b','c'}, outcome=0)
   cb = [case0, case1, case2, case3]
   
   assert set(clf.casebase_active) == set(cb + [default])
   
   predicted_output = clf.predict(test_X)
   assert expected_output == predicted_output
```

### Partial orders
Different partial orders may be used, but they are not implemented upfront. It is only required that case factors (characterisations) are defined by a class that supports [comparisons](https://docs.python.org/3/reference/datamodel.html#object.__lt__) and is [hashable](https://docs.python.org/3/glossary.html#term-hashable). Having only `__eq__` and `__le__` is sufficient for comparison, since, e.g. `__lt__` is ignored, but be careful for non-intuitive behaviour if this is not a partial order or comparisons are defined in superclasses.

```python
   class OrderedPair:
     """Pair (a,b) where (a,b) are natural numbers.
     Partial order is defined by (a,b) <= (c,d) iff a<=c and b<=d."""

     def __init__(self, x, y):
       self.x: int = x
       self.y: int = y

     def __eq__(self, other):
       return self.x == other.x and self.y == other.y
     def __le__(self, other):
       return self.x <= other.x and self.y <= other.y
     def __hash__(self):
       return hash((self.x, self.y))
     
   default = Case('default', OrderedPair(0,0), outcome=0)
   case1 = Case('1', OrderedPair(1,0), outcome=1)
   case2 = Case('2', OrderedPair(0,1), outcome=0)
   case3 = Case('3', OrderedPair(2,1), outcome=0)
   cb = (case1, case2, case3)
   clf = Aacbr(default_case=default)
   clf.fit(cb)
   assert set(clf.casebase_active) == set(cb + (default,))
   test = [OrderedPair(2,0),
           OrderedPair(0,2),
           OrderedPair(20,20),
           OrderedPair(1,1),
           OrderedPair(0,0)]
   expected_output = [1, 0, 0, 1, 0]
   predictions = clf.predict(test)
   assert expected_output == predictions
```

### Basic CLI
You may also define a `cb.json` file in [a casebase file format](./tests/data/cb_basic.json), as well as a `new.json` file in a [new cases file format](tests/data/new.json), and simply run:
```bash
python -m aacbr
```
This will generate a `cb_to_new.json` file with the output labels.

###

See `tests/` for more examples of input/output behaviour.
You can run all tests by using `make test`.


## Roadmap (refactoring and debugging)
This project is under heavy refactoring and expected to change. We expect the API in the previous section to be stable, but it is subject to change.
 - [ ] Support graphviz for drawing graphs
 - [X] Adapt to a more sklearn API style
 - [X] Draw graphs
 - [X] (Re-)allow CLI usage via json files
 - [X] Allow cautious AA-CBR
 - [X] Add a proper installation mechanism (via setup.py / setup.cfg)

## Acknowledgements
Code in this repository was originally adapted from work by Kristijonas Čyras and Ruxandra Istrate.

## References

AA-CBR is a result of research carried out by the [Computational Logic and Argumentation group](https://clarg.doc.ic.ac.uk/), at Imperial College London. This repository is based on the following publications:

[1]: https://dl.acm.org/doi/10.5555/3032027.3032100 (Kristijonas Cyras, Ken Satoh, Francesca Toni: Abstract Argumentation for Case-Based Reasoning. KR 2016: 549-552)
> **Kristijonas Cyras, Ken Satoh, Francesca Toni**: *Abstract Argumentation for Case-Based Reasoning*. KR 2016: 549-552
([text](https://dl.acm.org/doi/10.5555/3032027.3032100), [bib](https://dblp.org/rec/conf/kr/CyrasST16.html?view=bibtex))

[2]: https://doi.org/10.24963/kr.2021/48 (Guilherme Paulino-Passos and Francesca Toni: Monotonicity and Noise-Tolerance in Case-Based Reasoning with Abstract Argumentation. KR 2021)
>**Guilherme Paulino-Passos and Francesca Toni**: *Monotonicity and Noise-Tolerance in Case-Based Reasoning with Abstract Argumentation*. KR 2021
([text](https://doi.org/10.24963/kr.2021/48), [bib](https://dblp.org/rec/conf/kr/Paulino-PassosT21.html?view=bibtex))

[3]: https://doi.org/10.3233/FAIA200377 (Oana Cocarascu, Andria Stylianou, Kristijonas Čyras and Francesca Toni: Data-Empowered Argumentation for Dialectically Explainable Predictions. ECAI 2020)
>**Oana Cocarascu, Andria Stylianou, Kristijonas Čyras and Francesca Toni**: *Data-Empowered Argumentation for Dialectically Explainable Predictions*. ECAI 2020
([text](https://doi.org/10.3233/FAIA200377), [bib](https://dblp.org/rec/conf/ecai/CocarascuSCT20.html?view=bibtex))

<!-- ## Execution -->

<!-- aacbr.py executes AA-CBR, e.g.: -->

<!-- > python aacbr.py -->

<!-- - As an input, it takes two .json files - one with a casebase, another one with new cases - where each case is a dictionary with ID (default case should have ID 'default'), factors and outcome. -->

<!-- - Default file names are *cb.json* and *new.json*, as in *interact('cb', 'new')* -->

<!-- - Output containing predictions is dumped to *cb_to_new.json* -->

<!-- - Graphs for each new case visualising a single random directed path within AF are (optionally) dumped to .\graphs -->

***

