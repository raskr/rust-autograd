# autograd

[![Build Status](https://travis-ci.org/raskr/rust-autograd.svg?branch=master)](https://travis-ci.org/raskr/rust-autograd)
[![Crates.io version](http://meritbadge.herokuapp.com/autograd)](https://crates.io/crates/autograd)
[![docs.rs](https://docs.rs/autograd/badge.svg)](https://docs.rs/autograd/)

Differentiable operations and tensors backed by [ndarray](https://github.com/rust-ndarray/ndarray).

## Motivation
Machine learning is one of the field where Rust lagging behind other languages.
The aim of this crate is to show that Rust has the capability to implement efficient and full-featured dataflow graph naturally.
Moreover, the core of this crate is quite small compared to others (due to being implemented in pure Rust and ndarray),
therefore it might be reasonable for those who are not familiar with how this kind of library works.

## Installation

``` toml
[dependencies]
autograd = { version = "1.0.3", features = ["mkl"] }
```
`mkl` feature is recommended to speedup linalg operations using [Intel MKL](https://software.intel.com/en-us/mkl).

#### rustc version
Tested with rustc 1.38 ..= 1.42

## Features
### Lazy, lightweight tensor evaluation
Computation graphs are created on the fly (a.k.a. *define-by-run*), but are not evaluated until `eval` is called.
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
There are a lot of [built-in operations](https://docs.rs/autograd/1.0.3/autograd/struct.Graph.html)
that support *higher-order* derivatives, and
you can also [define your own differentiable ops](https://docs.rs/autograd/1.0.3/autograd/op/trait.Op.html) with ndarrays easily.

Here we are just computing partial derivatives of `z = 2x^2 + 3y + 1`.
 ```rust
ag::with(|g: &mut ag::Graph<_>| {
    let x = g.placeholder(&[]);
    let y = g.placeholder(&[]);
    let z = 2.*x*x + 3.*y + 1.;

    // dz/dy
    let gy = &g.grad(&[z], &[y])[0];
    println!("{:?}", gy.eval(&[]));   // => Ok(3.)

    // dz/dx (requires to fill the placeholder `x`)
    let gx = &g.grad(&[z], &[x])[0];
    let feed = ag::ndarray::arr0(2.);
    println!("{:?}", gx.eval(&[x.given(feed.view())]));  // => Ok(8.)

    // ddz/dx (differentiates `z` again)
    let ggx = &g.grad(&[gx], &[x])[0];
    println!("{:?}", ggx.eval(&[]));  // => Ok(4.)
});
 ```

 ### Neural networks
 This crate has various low-level features inspired by tensorflow/theano to train neural networks.
 Since computation graphs require only bare minimum of heap allocations, the overhead is small, even for complex networks.
 ```rust
 // This is a softmax regression for MNIST digits classification with Adam.
 // This achieves 0.918 test accuracy after 3 epochs (0.11 sec/epoch on 2.7GHz Intel Core i5).
use autograd::{self as ag, Graph, optimizers::adam, ndarray_ext as arr, tensor::Variable};

let rng = ag::ndarray_ext::ArrayRng::<f32>::default();
let w_arr = arr::into_shared(rng.glorot_uniform(&[28 * 28, 10]));
let b_arr = arr::into_shared(arr::zeros(&[1, 10]));
let adam_state = adam::AdamState::new(&[&w_arr, &b_arr]);

let max_epoch = 3;

for epoch in 0..max_epoch {
    ag::with(|g| {
        let w = g.variable(w_arr.clone());
        let b = g.variable(b_arr.clone());
        let x = g.placeholder(&[-1, 28*28]);
        let y = g.placeholder(&[-1]);
        let z = g.matmul(x, w) + b;
        let mean_loss = g.reduce_mean(g.sparse_softmax_cross_entropy(z, &y), &[0], false);
        let grads = &g.grad(&[&mean_loss], &[w, b]);
        let update_ops: &[ag::Tensor<f32>] =
            &adam::Adam::default().compute_updates(&[w, b], grads, &adam_state, g);

        // let batch_size = 200isize;
        // let num_samples = x_train.shape()[0];
        // let num_batches = num_samples / batch_size as usize;
        // for i in get_permutation(num_batches) {
        //     let i = i as isize * batch_size;
        //     let x_batch = x_train.slice(s![i..i + batch_size, ..]).into_dyn();
        //     let y_batch = y_train.slice(s![i..i + batch_size, ..]).into_dyn();
        //     g.eval(update_ops, &[x.given(x_batch), y.given(y_batch)]);
        // }
    });
}
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

For more, see [documentation](https://docs.rs/autograd/) or
[examples](https://github.com/raskr/rust-autograd/tree/master/examples)
