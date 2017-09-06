# autograd

[![Build Status](https://travis-ci.org/perrier1034/rust-autograd.svg?branch=master)](https://travis-ci.org/perrier1034/rust-autograd)

A library to run the computation graphs, whose backend is 
[rust-ndarray](https://github.com/bluss/rust-ndarray).


## Overview
* (Higher order) automatic differentiation
* Neural net first APIs
* Pure Rust
* Dynamic/static graph construction with shared variables

## Example
```rust

extern crate autograd as ag;

let ref x = ag::placeholder(&[1]);
let ref y = ag::variable(ag::init::zeros(&[1]));
// `z` is a target of partial differentiation.
let ref z = 2*x*x + 3*y + 1;
// dz/dy
let ref g1 = ag::gradients(z, &[y], None)[0];
// dz/dx (necessary to fill the placeholder `x`)
let ref g2 = ag::gradients(z, &[x], None)[0];
// ddz/dx (second order derivative)
let ref gg = ag::gradients(g2, &[x], None)[0];

// evaluation of symbolic gradients
assert_eq!(3., g1.eval()[0]);
let feed_dict = ag::Input::new().add(x, ag::init::from_scalar(2.));
assert_eq!(8., g2.eval_with_input(feed_dict)[0]);
assert_eq!(4., gg.eval()[0]);
```

For more, see 
[examples](https://github.com/perrier1034/rust-autograd/tree/master/examples) or
[tests](https://github.com/perrier1034/rust-autograd/tree/master/tests). 

## Documentation
WIP

## License
MIT
