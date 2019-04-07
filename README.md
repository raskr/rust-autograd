# autograd

[![Build Status](https://travis-ci.org/raskr/rust-autograd.svg?branch=master)](https://travis-ci.org/raskr/rust-autograd)
[![](http://meritbadge.herokuapp.com/autograd)](https://crates.io/crates/autograd)

Provides differentiable operations and tensors.

## Features
* **Lazy, side-effect-free tensors.**
`autograd::Tensor<T>` itself doesn't have its value basically.
It realizes graphs that are immutable and eagerly executable at any timing, 
that is, it supports both *run-by-define* and *define-by-run* naturally
in the context of neural networks.

* **Reverse-mode automatic differentiation.**
There are a lot of [built-in operations](https://docs.rs/autograd/0.9.1/autograd/ops/index.html)
that support *higher-order* derivatives, and
you can [define your own ops](https://docs.rs/autograd/0.9.1/autograd/op/trait.Op.html) with ndarrays easily.

* **Pure Rust.**
The graph execution engine is implemented in pure Rust, so it's compilable to WebAssembly.

## Installation
```
[dependencies]
autograd = { version = "0.9.1", features = ["mkl"] }
```
`mkl` feature is recommended to speedup gemm operations.


## Examples
Here we are computing partial derivatives of `z = 2x^2 + 3y + 1`.

```rust
extern crate autograd as ag;

let ref x = ag::placeholder(&[]);
let ref y = ag::placeholder(&[]);
let ref z = 2.*x*x + 3.*y + 1.;

// dz/dy
let gy = &ag::grad(&[z], &[y])[0];
println!("{:?}", gy.eval(&[]));   // => Some(3.)

// dz/dx (requires to fill the placeholder `x`)
let gx = &ag::grad(&[z], &[x])[0];
println!("{:?}", gx.eval(&[(x, &ag::ndarray::arr0(2.).into_dyn())]));  // => Some(8.)

// ddz/dx (differentiates `z` again)
let ggx = &ag::grad(&[gx], &[x])[0];
println!("{:?}", ggx.eval(&[]));  // => Some(4.)
```

Another example: softmax regression for MNIST digits classification with Adam.

```rust
// This achieves 0.918 test accuracy after 3 epochs, 0.14 sec/epoch on 2.7GHz Intel Core i5


let ref w = ag::variable(ag::ndarray_ext::glorot_uniform::<f32>(&[28*28, 10]));
let ref b = ag::variable(ag::ndarray_ext::zeros::<f32>(&[1, 10]));
let ref x = ag::placeholder(&[-1, 28*28]);
let ref y = ag::placeholder(&[-1]);
let ref z = ag::matmul(x, w) + b;
let ref loss = ag::reduce_mean(&ag::sparse_softmax_cross_entropy(z, y), &[0, 1], false);
let ref params = [w, b];
let ref grads = ag::grad(&[loss], params);
let ref predictions = ag::argmax(z, -1, true);
let ref accuracy = ag::reduce_mean(&ag::equal(predictions, y), &[0], false);
let ref adam = ag::gradient_descent_ops::Adam::default();
let mut stateful_params = ag::gradient_descent_ops::Adam::vars_with_states(params);
let ref update_ops = adam.compute_updates(&stateful_params, grads);

// -- dataset --
let ((x_train, y_train), (x_test, y_test)) = dataset::load();

// -- training loop --
for epoch in 0..max_epoch {
    ...
    ag::eval(update_ops, &[(x, &x_batch), (y, &y_batch)]);
}

```
For more, see [documentation](https://docs.rs/autograd/) or
[examples](https://github.com/raskr/rust-autograd/tree/master/examples)
