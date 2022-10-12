# NEWS / Release Notes

## Unreleased

## [0.2.0]
  * [CHANGED] When the argument `factors` of `Case` is a `set`, it is converted to `frozenset` instead. This is done in order to guarantee that cases are hashable. If modifying a case, create a new one with different values.
  * [CHANGED] Hashes for `Case` instances are now cached via an attribute.
  * [CHANGED] `past_case_attacks` now does not check whether its arguments are in the active casebase. This is now assumed, instead of verified.
