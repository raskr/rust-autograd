# autograd

[![Build Status](https://travis-ci.org/raskr/rust-autograd.svg?branch=master)](https://travis-ci.org/raskr/rust-autograd)
[![](http://meritbadge.herokuapp.com/autograd)](https://crates.io/crates/autograd)

This library provides differentiable operations and tensors.
The current backend is [rust-ndarray](https://github.com/bluss/rust-ndarray).

## Examples
Here we are computing partial derivatives of `z = 2x^2 + 3y + 1`.

```rust

extern crate ndarray;
extern crate autograd as ag;

let ref x = ag::placeholder(&[]);
let ref y = ag::placeholder(&[]);
let ref z = 2*x*x + 3*y + 1;

// dz/dy
let ref g1 = ag::grad(&[z], &[y])[0];

// dz/dx
let ref g2 = ag::grad(&[z], &[x])[0];

// ddz/dx (differentiates `z` again)
let ref gg = ag::grad(&[g2], &[x])[0];

// evaluation of symbolic gradients
let mut ctx = ag::Context::new();
println!("{}", g1.eval(&mut ctx));   // => 3.
println!("{}", gg.eval(&mut ctx));   // => 4.

// dz/dx requires to fill the placeholder `x`
ag::feed_input(x, ndarray::arr0(2.), &mut ctx);
println!("{}", g2.eval(&mut ctx));   // => 8.
```

Another example: multi layer perceptron for MNIST classification.

```rust
// -- graph def --
let mut ctx = ag::Context::new();
let ref x = ag::placeholder(&[-1, 28*28]);
let ref y = ag::placeholder(&[-1]);
let ref w = ag::variable(ag::ndarray_ext::glorot_uniform(&[28*28, 10]), &mut ctx);
let ref b = ag::variable(ag::ndarray_ext::zeros(&[1, 10]), &mut ctx);
let ref z = ag::matmul(x, w) + b;
let ref loss = ag::sparse_softmax_cross_entropy(z, y);
let ref grads = ag::grad(loss, &[w, b]);
let ref predictions = ag::argmax(z, -1, true);
let ref accuracy = ag::reduce_mean(&ag::equal(predictions, y), &[0], false);

// -- dataset --
let ((x_train, y_train), (x_test, y_test)) = dataset::load();

// -- training method --
let mut optimizer = ag::sgd::optimizers::Adam { ..Default::default() };

// -- training loop --
for epoch in 0..max_epoch {
    ...
}

```
For more, see 
[examples](https://github.com/raskr/rust-autograd/tree/master/examples) or
[tests](https://github.com/raskr/rust-autograd/tree/master/tests). 

Available ops are listed [here](https://docs.rs/autograd/0.5.0/autograd/ops/index.html).

