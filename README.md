# AA-CBR Basic Implementation

This provides an implementation of the AA-CBR formalism [**Kristijonas Cyras, Ken Satoh, Francesca Toni**: *Abstract Argumentation for Case-Based Reasoning*. KR 2016: 549-552][1].

This includes an implementation of cautiously monotonic AA-CBR (cAA-CBR), presented in [2], and supports general partial orders, as originally introduced in [3].

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

See `tests/` for more examples of input/output behaviour.

You can run tests by using `make test`.


## Roadmap (refactoring and debugging)
This project is under heavy refactoring and expected to change. We expect the API in the previous section to be stable, but it is subject to change.
 - [ ] Draw graphs
 - [ ] Adapt to a more sklearn API style
 - [X] (Re-)allow CLI usage via json files
 - [X] Allow cautious AA-CBR
 - [X] Add a proper installation mechanism (via setup.py / setup.cfg)

## References

[1]: https://dl.acm.org/doi/10.5555/3032027.3032100 (**Kristijonas Cyras, Ken Satoh, Francesca Toni**: *Abstract Argumentation for Case-Based Reasoning*. KR 2016: 549-552)
> **Kristijonas Cyras, Ken Satoh, Francesca Toni**: *Abstract Argumentation for Case-Based Reasoning*. KR 2016: 549-552
```bibtex
@inproceedings{DBLP:conf/kr/CyrasST16,
  author    = {Kristijonas Cyras and
               Ken Satoh and
               Francesca Toni},
  editor    = {Chitta Baral and
               James P. Delgrande and
               Frank Wolter},
  title     = {Abstract Argumentation for Case-Based Reasoning},
  booktitle = {Principles of Knowledge Representation and Reasoning: Proceedings
               of the Fifteenth International Conference, {KR} 2016, Cape Town, South
               Africa, April 25-29, 2016},
  pages     = {549--552},
  publisher = {{AAAI} Press},
  year      = {2016},
  url       = {http://www.aaai.org/ocs/index.php/KR/KR16/paper/view/12879},
  timestamp = {Tue, 09 Feb 2021 08:33:50 +0100},
  biburl    = {https://dblp.org/rec/conf/kr/CyrasST16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
[2]: https://proceedings.kr.org/2021/48/ (**Guilherme Paulino-Passos and Francesca Toni**: *Monotonicity and Noise-Tolerance in Case-Based Reasoning with Abstract Argumentation*. KR 2021)
>**Guilherme Paulino-Passos and Francesca Toni**: *Monotonicity and Noise-Tolerance in Case-Based Reasoning with Abstract Argumentation*
```bibtex
@inproceedings{KR2021-48,
    title     = {{Monotonicity and Noise-Tolerance in Case-Based Reasoning with Abstract Argumentation}},
    author    = {Paulino-Passos, Guilherme and Toni, Francesca},
    booktitle = {{Proceedings of the 18th International Conference on Principles of Knowledge Representation and Reasoning}},
    pages     = {508--518},
    year      = {2021},
    month     = {11},
    doi       = {10.24963/kr.2021/48},
    url       = {https://doi.org/10.24963/kr.2021/48},
  }
```
[3]: https://doi.org/10.3233/FAIA200377 (**Oana Cocarascu, Andria Stylianou, Kristijonas Čyras and Francesca Toni**: *Data-Empowered Argumentation for Dialectically Explainable Predictions*)
>**Oana Cocarascu, Andria Stylianou, Kristijonas Čyras and Francesca Toni**: *Data-Empowered Argumentation for Dialectically Explainable Predictions*. ECAI 2020
```bibtex
@InProceedings{DBLP:conf/ecai/CocarascuSCT20,
  author       = {Oana Cocarascu and Andria Stylianou and Kristijonas
                  Čyras and Francesca Toni},
  title        = {Data-Empowered Argumentation for Dialectically
                  Explainable Predictions},
  year         = 2020,
  booktitle    = {{ECAI} 2020 - 24th European Conference on Artificial
                  Intelligence, 29 August-8 September 2020, Santiago
                  de Compostela, Spain, August 29 - September 8, 2020
                  - Including 10th Conference on Prestigious
                  Applications of Artificial Intelligence {(PAIS}
                  2020)},
  pages        = {2449-2456},
  doi          = {10.3233/FAIA200377},
  url          = {https://doi.org/10.3233/FAIA200377},
  crossref     = {DBLP:conf/ecai/2020},
  timestamp    = {Tue, 29 Dec 2020 18:37:52 +0100},
  biburl       = {https://dblp.org/rec/conf/ecai/CocarascuSCT20.bib},
  bibsource    = {dblp computer science bibliography,
                  https://dblp.org}
}
```

<!-- ## Execution -->

<!-- aacbr.py executes AA-CBR, e.g.: -->

<!-- > python aacbr.py -->

<!-- - As an input, it takes two .json files - one with a casebase, another one with new cases - where each case is a dictionary with ID (default case should have ID 'default'), factors and outcome. -->

<!-- - Default file names are *cb.json* and *new.json*, as in *interact('cb', 'new')* -->

<!-- - Output containing predictions is dumped to *cb_to_new.json* -->

<!-- - Graphs for each new case visualising a single random directed path within AF are (optionally) dumped to .\graphs -->

***

