# NEWS / Release Notes

## Unreleased

## [0.4.2]
  * [FIXED] Bugs with sorting fixed, solving issues with remove_spikes not working properly, among others.

## [0.4.1]
  * [FIXED] When the default case is passed as argument and has a different outcome than originally set in the Aacbr instance as default outcome, now this is taken intou account, respecting the explicitly sent default case.

## [0.4.0]
  API is kept, but massive speed-ups.
  * [CHANGED] Fitting, for both cautious and non-cautious, is now much faster.
    In the slowest scenarios in benchmarking contained in `tests/`, 85% speed-up in non-cautious, and 99.89% speed-up in cautious.

## [0.3.0]
  More compatible with the sklearn API, although still not passing sklearn automated testing.
  * [CHANGED] `X` and `y` arguments can be used in `fit` instead of `casebase` and `outcomes`
  * [CHANGED] Attributes only (properly) set after training are now defined in `fit`, not in initialisation.

## [0.2.0]
  * [CHANGED] When the argument `factors` of `Case` is a `set`, it is converted to `frozenset` instead. This is done in order to guarantee that cases are hashable. If modifying a case, create a new one with different values.
  * [CHANGED] Hashes for `Case` instances are now cached via an attribute.
  * [CHANGED] `past_case_attacks` now does not check whether its arguments are in the active casebase. This is now assumed, instead of verified.
