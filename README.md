# autograd

[![Build Status](https://travis-ci.org/raskr/rust-autograd.svg?branch=master)](https://travis-ci.org/raskr/rust-autograd)
[![Crates.io version](http://meritbadge.herokuapp.com/autograd)](https://crates.io/crates/autograd)
[![docs.rs](https://docs.rs/autograd/badge.svg)](https://docs.rs/autograd/)

Tensors and differentiable operations backed by [ndarray](https://github.com/rust-ndarray/ndarray).

## Cargo.toml
If you use basic linalg operations, especially matrix multiplications, `blas` feature would be important to speed them up. 
``` toml
[dependencies]
autograd = {"<version>", features = ["blas", "<blas-implementation-choise>"] }
```

`<blas-implementation-choise>` must be one of the following (See also [blas-src](https://github.com/blas-lapack-rs/blas-src))
- `accelerate` macOS only
- `intel-mkl` Intel/AMD CPU only. Includes Vector Mathematics (VM) ops
- `openblas`

## Features
### Reverse-mode automatic differentiation using lazy tensors
Here we are just computing partial derivatives of `z = 2x^2 + 3y + 1`.
 ```rust
use autograd as ag;
use ag::tensor_ops as T;

ag::run(|ctx: &mut ag::Context<_>| {
    let x = ctx.placeholder("x", &[]);
    let y = ctx.placeholder("y", &[]);
    let z = 2.*x*x + 3.*y + 1.;

    // dz/dy
    let gy = &T::grad(&[z], &[y])[0];
    println!("{:?}", gy.eval(ctx));   // => Ok(3.)

    // dz/dx (requires to fill the placeholder `x`)
    let gx = &T::grad(&[z], &[x])[0];
    let feed = ag::ndarray::arr0(2.);
    println!("{:?}", ctx.evaluator().push(gx).feed(x, feed.view()).run()[0]);  // => Ok(8.)

    // ddz/dx (differentiates `z` again)
    let ggx = &T::grad(&[gx], &[x])[0];
    println!("{:?}", ggx.eval(ctx));  // => Ok(4.)
});
 ```

 ### Neural networks
 This crate has various low-level features inspired by tensorflow/theano to train neural networks.
 Since computation graphs require only bare minimum of heap allocations, the overhead is small, even for complex networks.
 ```rust
// MNIST digits classification model with 0.918 accuracy
use autograd as ag;
use ag::optimizers::adam::Adam;
use ag::tensor_ops::*;
use ag::prelude::*;

let mut env = ag::VariableEnvironment::new();

let rng = ag::ndarray_ext::ArrayRng::<f32>::default();

// Register variables in this env.
env.name("w").set(rng.glorot_uniform(&[28 * 28, 10]));
env.name("b").set(ag::ndarray_ext::zeros(&[1, 10]));

let adam = Adam::default("my_adam", env.default_namespace().current_var_ids(), &mut env);

for epoch in 0..3 {  // 0.11 sec/epoch on 2.7GHz Intel Core i5
    env.run(|ctx| {
        let x = ctx.placeholder("x", &[-1, 28*28]);
        let y = ctx.placeholder("y", &[-1]);
        let w = ctx.variable("w");
        let b = ctx.variable("b");
        let z = matmul(x, w) + b;
        let mean_loss = reduce_mean(sparse_softmax_cross_entropy(z, &y), &[0], false);
        let grads = &grad(&[&mean_loss], &[w, b]);

        // let mut feeder = ag::Feeder::new();
        // feeder.push(x, x_batch).push(y, y_batch);
        // adam.update(&[w, b], grads, ctx, feeder);
    });
}
 ```

For detailed, see [documentation](https://docs.rs/autograd/) or
[examples](https://github.com/raskr/rust-autograd/tree/master/examples)