# autograd

[![Build Status](https://travis-ci.org/raskr/rust-autograd.svg?branch=master)](https://travis-ci.org/raskr/rust-autograd)
[![](http://meritbadge.herokuapp.com/autograd)](https://crates.io/crates/autograd)

Differentiable operations and tensors backed by [ndarray](https://github.com/rust-ndarray/ndarray).

## Installation
```
[dependencies]
autograd = { version = "0.9.8", features = ["mkl"] }
```
`mkl` feature is recommended to speedup gemm operations using [Intel MKL](https://software.intel.com/en-us/mkl).

## Features
### Lazy, zero-copy tensor evaluation
Computation graphs are created on the fly (a.k.a define-by-run), but are not evaluated until `Tensor::eval` or `ag::eval` is called.
This mechanism balances better performance and flexibility.
```rust
extern crate autograd as ag;

let a: ag::Tensor<f32> = ag::ones(&[60]);
let b: ag::Tensor<f32> = ag::ones(&[24]);
let c: ag::Tensor<f32> = ag::reshape(a, &[3, 4, 5]);
let d: ag::Tensor<f32> = ag::reshape(b, &[4, 3, 2]);
let e: ag::Tensor<f32> = ag::tensordot(c, d, &[1, 0], &[0, 1]);
e.eval(&[]);  // Getting `ndarray::Array` here.
```

### Reverse-mode automatic differentiation
There are a lot of [built-in operations](https://docs.rs/autograd/0.9.8/autograd/ops/index.html)
that support *higher-order* derivatives, and
you can also [define your own differentiable ops](https://docs.rs/autograd/0.9.8/autograd/op/trait.Op.html) with ndarrays easily.

Here we are just computing partial derivatives of `z = 2x^2 + 3y + 1`.

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
println!("{:?}", gx.eval(&[ag::Feed(x, ag::ndarray::arr0(2.).into_dyn().view())]));  // => Some(8.)

// ddz/dx (differentiates `z` again)
let ggx = &ag::grad(&[gx], &[x])[0];
println!("{:?}", ggx.eval(&[]));  // => Some(4.)
```

### Neural networks
This crate has various low-level features inspired by tensorflow/theano to train neural networks.
```rust
// This is a softmax regression for MNIST digits classification with Adam.
// This achieves 0.918 test accuracy after 3 epochs (0.11 sec/epoch on 2.7GHz Intel Core i5).
let ref w = ag::variable(ag::ndarray_ext::glorot_uniform::<f32>(&[28*28, 10]));
let ref b = ag::variable(ag::ndarray_ext::zeros::<f32>(&[1, 10]));
let ref x = ag::placeholder(&[-1, 28*28]);
let ref y = ag::placeholder(&[-1]);
let ref z = ag::matmul(x, w) + b;
let ref loss = ag::sparse_softmax_cross_entropy(z, y);
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
    ag::eval(update_ops, &[ag::Feed(x, x_batch), ag::Feed(y, y_batch)]);
}
```

ConvNet, LSTM example can be found in [examples](https://github.com/raskr/rust-autograd/tree/master/examples)

### Hooks
You can register hooks on `ag::Tensor` objects for debugging.
```rust
extern crate autograd as ag;

// `.p()` is a shorthand for `.with(ag::Hook::Print)`.
let a: ag::Tensor<f32> = ag::zeros(&[4, 2]).p();
let b: ag::Tensor<f32> = ag::ones(&[2, 3]);
let c = ag::matmul(a, b);

c.eval(&[]);
// Zeros:
// [[0.0, 0.0],
// [0.0, 0.0],
// [0.0, 0.0],
// [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
```

## Why Rust?

- **No need for bridges for fast languages.**
The entire logic including hotspots (kernels etc) is implemented in pure Rust,
without compromising performance.

- **Memory safety.** For example, Rust's lifetime checker makes it possible to implement zero-copy computation graphs without GC.

For more, see [documentation](https://docs.rs/autograd/) or
[examples](https://github.com/raskr/rust-autograd/tree/master/examples)
