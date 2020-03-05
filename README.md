# autograd

[![Build Status](https://travis-ci.org/raskr/rust-autograd.svg?branch=master)](https://travis-ci.org/raskr/rust-autograd)
[![Crates.io version](http://meritbadge.herokuapp.com/autograd)](https://crates.io/crates/autograd)
[![docs.rs](https://docs.rs/autograd/badge.svg)](https://docs.rs/autograd/)

Differentiable operations and tensors backed by [ndarray](https://github.com/rust-ndarray/ndarray).

## Installation
```
[dependencies]
autograd = { version = "0.9.8", features = ["mkl"] }
```
`mkl` feature is recommended to speedup gemm operations using [Intel MKL](https://software.intel.com/en-us/mkl).

## Features
### Lazy, zero-copy tensor evaluation
Computation graphs are created on the fly (a.k.a. *define-by-run*), but are not evaluated until `Tensor::eval` or `ag::eval` is called.
This mechanism balances better performance and flexibility.
```rust
use autograd as ag;

ag::with(|g: &mut ag::Graph<_>| {
    let a: ag::Tensor<f32> = g.ones(&[60]);
    let b: ag::Tensor<f32> = g.ones(&[24]);
    let c: ag::Tensor<f32> = g.reshape(a, &[3, 4, 5]);
    let d: ag::Tensor<f32> = g.reshape(b, &[4, 3, 2]);
    let e: ag::Tensor<f32> = g.tensordot(c, d, &[1, 0], &[0, 1]);
    e.eval(&[]);  // Getting `ndarray::Array` here.
});

```

### Reverse-mode automatic differentiation
There are a lot of [built-in operations](https://docs.rs/autograd/0.9.8/autograd/ops/index.html)
that support *higher-order* derivatives, and
you can also [define your own differentiable ops](https://docs.rs/autograd/0.9.8/autograd/op/trait.Op.html) with ndarrays easily.

Here we are just computing partial derivatives of `z = 2x^2 + 3y + 1`.

```rust
use autograd as ag;

ag::with(|g: ag::Graph<_>| {
    let x = g.placeholder(&[]);
    let y = g.placeholder(&[]);
    let z = 2.*x*x + 3.*y + 1.;

    // dz/dy
    let gy = &g.grad(&[z], &[y])[0];
    println!("{:?}", gy.eval(&[]));   // => Some(3.)

    // dz/dx (requires to fill the placeholder `x`)
    let gx = &ag::grad(&[z], &[x])[0];
    let feed = x.given(ag::ndarray::arr0(2.).view());
    println!("{:?}", gx.eval(&[feed]));  // => Some(8.)
    // ddz/dx (differentiates `z` again)
    let ggx = &g.grad(&[gx], &[x])[0];
    println!("{:?}", ggx.eval(&[]));  // => Some(4.)
});
```

### Neural networks
This crate has various low-level features inspired by tensorflow/theano to train neural networks.
```rust
// This is a softmax regression for MNIST digits classification with Adam.
// This achieves 0.918 test accuracy after 3 epochs (0.11 sec/epoch on 2.7GHz Intel Core i5).
use autograd::{
    with,
    Graph,
    optimizers::adam, 
    ndarray_ext as arr
};

with(|g: &mut Graph<f32>|{
    let w_arr = arr::shared(arr::glorot_uniform(&[28 * 28, 10]));
    let b_arr = arr::shared(arr::zeros(&[1, 10]));
    let adam_state = adam::AdamState::new(&[&w_arr, &b_arr]);

    let w = g.variable(w_arr.clone());
    let b = g.variable(b_arr.clone());
    let x = g.placeholder(&[-1, 28*28]);
    let y = g.placeholder(&[-1]);
    let z = g.matmul(x, w) + b;
    let loss = g.sparse_softmax_cross_entropy(z, y);
    let grads = g.grad(&[loss], &[w, b]);
    let predictions = g.argmax(z, -1, true);
    let accuracy = g.reduce_mean(&ag::equal(predictions, y), &[0], false);
    let state = adam::AdamState::new(&[&w, &b]);
    let optimizer = adam::Adam::<f32>::default();
    let update_ops = optimizer.compute_updates(&[w, b], &grads, &state, g);

    // -- dataset --
    let ((x_train, y_train), (x_test, y_test)) = dataset::load();

    // -- training loop --
    for epoch in 0..max_epoch {
        ...
        g.eval(update_ops, &[x.given(x_batch), y.given(y_batch)]);
    }
});
```

ConvNet, LSTM example can be found in [examples](https://github.com/raskr/rust-autograd/tree/master/examples)

### Hooks
You can register hooks on `ag::Tensor` objects for debugging.
```rust
use autograd as ag;

ag::with(|g| {
    let a: ag::Tensor<f32> = g.zeros(&[4, 2]).show();
    let b: ag::Tensor<f32> = g.ones(&[2, 3]).show_shape();
    let c = g.matmul(a, b).show_with("MatMul:");

    c.eval(&[]);
    // [[0.0, 0.0],
    // [0.0, 0.0],
    // [0.0, 0.0],
    // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
    //
    // [2, 3]
    //
    // MatMul:
    //  [[0.0, 0.0, 0.0],
    //  [0.0, 0.0, 0.0],
    //  [0.0, 0.0, 0.0],
    //  [0.0, 0.0, 0.0]] shape=[4, 3], strides=[3, 1], layout=C (0x1), dynamic ndim=2
});
```

## Why Rust?

- **No need for bridges for fast languages.**
The entire logic including hotspots (kernels etc) is implemented in pure Rust,
without compromising performance.

- **Memory safety.** For example, Rust's lifetime checker makes it possible to implement zero-copy computation graphs without GC.

For more, see [documentation](https://docs.rs/autograd/) or
[examples](https://github.com/raskr/rust-autograd/tree/master/examples)
