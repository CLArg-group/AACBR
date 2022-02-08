# AA-CBR Basic Implementation

This provides an implementation of the AA-CBR formalism [Abstract Argumentation for Case-Based Reasoning][1].

This includes an implementation of cautiously monotonic AA-CBR (cAA-CBR), presented in [Monotonicity and Noise-Tolerance in Case-Based Reasoning with Abstract Argumentation][2], and supports general partial orders, as originally introduced in [Data-Empowered Argumentation for Dialectically Explainable Predictions][3].

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
 - [ ] Draw graphs
 - [ ] Adapt to a more sklearn API style
 - [X] (Re-)allow CLI usage via json files
 - [X] Allow cautious AA-CBR
 - [X] Add a proper installation mechanism (via setup.py / setup.cfg)

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

