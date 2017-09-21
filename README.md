# autograd

[![Build Status](https://travis-ci.org/perrier1034/rust-autograd.svg?branch=master)](https://travis-ci.org/perrier1034/rust-autograd)

A library to run the computation graphs whose backend is 
[rust-ndarray](https://github.com/bluss/rust-ndarray).

Documentation: https://docs.rs/autograd/


## Overview
* Automatic differentiation
* Pure Rust
* Neural net first APIs
* Dynamic/static graph construction with shared variables

## Example
Here we are computing partial derivatives of `z = 2x^2 + 3y + 1`.

```rust

extern crate ndarray;
extern crate autograd as ag;

let ref x = ag::placeholder(&[1]);
let ref y = ag::variable(ndarray::arr1(&[0]));
let ref z = 2*x*x + 3*y + 1;

// dz/dy
let ref g1 = ag::gradients(z, &[y], None)[0];

// dz/dx
let ref g2 = ag::gradients(z, &[x], None)[0];

// ddz/dx (differentiates `z` again)
let ref gg = ag::gradients(g2, &[x], None)[0];

// evaluation of symbolic gradients
assert_eq!(3., g1.eval()[0]);
assert_eq!(4., gg.eval()[0]);

// dz/dx requires to fill the placeholder `x`
let feed = ag::Feed::new().add(x, ndarray::arr1(&[2.]));
assert_eq!(8., g2.eval_with_input(feed)[0]);

```

For more, see 
[examples](https://github.com/perrier1034/rust-autograd/tree/master/examples) or
[tests](https://github.com/perrier1034/rust-autograd/tree/master/tests). 

## License
MIT
