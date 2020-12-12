# Changelog

## v1.0.3
### Fixed
- Serious memory bug around gradient computation

## v1.0.2
### Fixed
- Wrong documentation of `Graph::slice`

## v1.0.1
### Fixed
- Critical memory issue when the number of nodes in a graph exceeded a certain threshold [#28](https://github.com/raskr/rust-autograd/issues/28)
- Misuse of given indexes in `Graph::slice` [#30](https://github.com/raskr/rust-autograd/pull/30)
- Undisposed dependencies

## v1.0.0
Introduced `Graph` and basic API was fixed.
