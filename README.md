# autograd

[![Build Status](https://travis-ci.org/perrier1034/rust-autograd.svg?branch=master)](https://travis-ci.org/perrier1034/rust-autograd)
[![](http://meritbadge.herokuapp.com/autograd)](https://crates.io/crates/autograd)

A library to run the computation graphs, whose current backend is 
[rust-ndarray](https://github.com/bluss/rust-ndarray).

Documentation: https://docs.rs/autograd/


## Overview
* Automatic differentiation
* Pure Rust
* Neural net first APIs
* Dynamic/static graph construction with shared variables

## Examples
Here we are computing partial derivatives of `z = 2x^2 + 3y + 1`.

```rust

extern crate ndarray;
extern crate autograd as ag;

let mut graph = ag::Graph::new();
let ref x = graph.placeholder();
let ref y = graph.variable(ndarray::arr1(&[0]));
let ref z = 2*x*x + 3*y + 1;

// dz/dy
let ref g1 = ag::gradients(&[z], &[y], None)[0];

// dz/dx
let ref g2 = ag::gradients(&[z], &[x], None)[0];

// ddz/dx (differentiates `z` again)
let ref gg = ag::gradients(&[g2], &[x], None)[0];

// evaluation of symbolic gradients
assert_eq!(3., g1.eval(&mut graph)[0]);
assert_eq!(4., gg.eval(&mut graph)[0]);

// dz/dx requires to fill the placeholder `x`
graph.feed(x, ndarray::arr1(&[2.]));
assert_eq!(8., g2.eval(&mut graph)[0]);
```

Another example: multi layer perceptron for MNIST.

```rust
// -- graph def --
let mut g = ag::Graph::new();

let ref x = g.placeholder();
let ref y = g.placeholder();
let ref w = g.variable(ag::ndarray_ext::glorot_uniform(&[28 * 28, 10]));
let ref b = g.variable(ag::ndarray_ext::zeros(&[1, 10]));
let ref z = ag::matmul(x, w) + b;
let ref loss = ag::sparse_softmax_cross_entropy(z, y);
let ref grads = ag::gradients(loss, &[w, b], None);
let ref predictions = ag::argmax(z, -1, true);
let ref accuracy = ag::reduce_mean(&ag::equals(predictions, y), 0, false);

// -- dataset --
let ((x_train, y_train), (x_test, y_test)) = dataset::load();

// -- training method --
let mut optimizer = ag::sgd::optimizers::Adam { ..Default::default() };

// -- training loop --
for epoch in 0..max_epoch {
    ...
}

```
Available operations in rust-autograd are listed [here](https://docs.rs/autograd/0.4.7/autograd/ops/index.html)

For more, see 
[examples](https://github.com/perrier1034/rust-autograd/tree/master/examples) or
[tests](https://github.com/perrier1034/rust-autograd/tree/master/tests). 

## License
MIT
