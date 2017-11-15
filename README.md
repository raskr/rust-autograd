# autograd

[![Build Status](https://travis-ci.org/perrier1034/rust-autograd.svg?branch=master)](https://travis-ci.org/perrier1034/rust-autograd)
[![](http://meritbadge.herokuapp.com/autograd)](https://crates.io/crates/autograd)

A library to run the computation graphs, whose current backend is 
[rust-ndarray](https://github.com/bluss/rust-ndarray).


## Overview
* Tensors with automatic differentiation
* Pure Rust
* Neural net first APIs

## Examples
Here we are computing partial derivatives of `z = 2x^2 + 3y + 1`.

```rust

extern crate ndarray;
extern crate autograd as ag;

let mut ctx = ag::Context::new();
let ref x = ctx.placeholder();
let ref y = ctx.variable(ndarray::arr1(&[0]));
let ref z = 2*x*x + 3*y + 1;

// dz/dy
let ref g1 = ag::gradients(&[z], &[y], None)[0];

// dz/dx
let ref g2 = ag::gradients(&[z], &[x], None)[0];

// ddz/dx (differentiates `z` again)
let ref gg = ag::gradients(&[g2], &[x], None)[0];

// evaluation of symbolic gradients
assert_eq!(3., g1.eval(&mut ctx)[0]);
assert_eq!(4., gg.eval(&mut ctx)[0]);

// dz/dx requires to fill the placeholder `x`
graph.feed(x, ndarray::arr1(&[2.]));
assert_eq!(8., g2.eval(&mut ctx)[0]);
```

Another example: multi layer perceptron for MNIST.

```rust
// -- graph def --
let mut ctx = ag::Context::new();

let ref x = ctx.placeholder();
let ref y = ctx.placeholder();
let ref w = ctx.variable(ag::ndarray_ext::glorot_uniform(&[28 * 28, 10]));
let ref b = ctx.variable(ag::ndarray_ext::zeros(&[1, 10]));
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
Available operations in rust-autograd are listed [here](https://docs.rs/autograd/0.5.0/autograd/ops/index.html)

For more, see 
[examples](https://github.com/perrier1034/rust-autograd/tree/master/examples) or
[tests](https://github.com/perrier1034/rust-autograd/tree/master/tests). 

## License
MIT
