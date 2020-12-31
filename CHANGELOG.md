# Changelog

## v1.1.0
### Fixed
- Wrong gradient computation in the case that a target variable is scalar and its broadcast is involved: `Graph::grad(&[scalar * tensor], &[scalar])` [#38](https://github.com/raskr/rust-autograd/issues/38)
### Added
- New `autograd::run` API which is more useful version of `autograd::with` [#34](https://github.com/raskr/rust-autograd/pull/34)
### Other improvements
- Fixed some error messages that are not user-friendly (`Tensor::access_elem`, `Graph::sparse_softmax_crossentropy`)
- Updated dependencies: `num`, `rand_distr`

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
