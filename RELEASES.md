# NEWS / Release Notes

## Unreleased

## [0.2.0]
  * [CHANGED] When the argument `factors` of `Case` is a `set`, it is converted to `frozenset` instead. This is done in order to guarantee that cases are hashable. If modifying a case, create a new one with different values.
