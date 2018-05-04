# autograd

[![Build Status](https://travis-ci.org/raskr/rust-autograd.svg?branch=master)](https://travis-ci.org/raskr/rust-autograd)
[![](http://meritbadge.herokuapp.com/autograd)](https://crates.io/crates/autograd)

This library provides differentiable operations and tensors.
The current backend is [rust-ndarray](https://github.com/bluss/rust-ndarray).

## Examples
Here we are computing partial derivatives of `z = 2x^2 + 3y + 1`.

```rust

extern crate autograd as ag;

let ref x = ag::placeholder(&[]);
let ref y = ag::placeholder(&[]);
let ref z = 2*x*x + 3*y + 1;

// dz/dy
let ref gy = ag::grad(&[z], &[y])[0];

// dz/dx
let ref gx = ag::grad(&[z], &[x])[0];

// ddz/dx (differentiates `z` again)
let ref ggx = ag::grad(&[gx], &[x])[0];

// evaluation of symbolic gradients
println!("{:?}", gy.eval(&[]));   // => Ok(3.)
println!("{:?}", ggx.eval(&[]));  // => Ok(4.)

// dz/dx requires to fill the placeholder `x`
println!("{:?}", gx.eval(&[(x, &ag::ndarray::arr0(2.))]));  // => Ok(8.)
```

Another example: multi layer perceptron for MNIST digits classification.

```rust
use self::ag::gradient_descent_ops::Optimizer;

// -- graph def --
let ref x = ag::placeholder(&[-1, 28*28]);
let ref y = ag::placeholder(&[-1]);
let ref w = ag::variable(ag::ndarray_ext::glorot_uniform(&[28*28, 10]));
let ref b = ag::variable(ag::ndarray_ext::zeros(&[1, 10]));
let ref z = ag::matmul(x, w) + b;
let ref loss = ag::reduce_mean(&ag::sparse_softmax_cross_entropy(z, y), &[0, 1], false);
let ref params = [w, b]
let ref grads = ag::grad(loss, params);
let ref predictions = ag::argmax(z, -1, true);
let ref accuracy = ag::reduce_mean(&ag::equal(predictions, y), &[0], false);
let mut adam = ag::gradient_descent_ops::Adam::default();
let ref update_ops = adam.compute_updates(params, grads);

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
