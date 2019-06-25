# autograd

[![Build Status](https://travis-ci.org/raskr/rust-autograd.svg?branch=master)](https://travis-ci.org/raskr/rust-autograd)
[![](http://meritbadge.herokuapp.com/autograd)](https://crates.io/crates/autograd)

Provides differentiable operations and tensors.

## Features
* **Lazy, zero-copy and side-effect-free tensors.**
`autograd::Tensor<T>` itself doesn't have its value basically (except for persistent tensors).
It realizes graphs that are eagerly executable at any timing, 
that is, it supports both *run-by-define* and *define-by-run* naturally
in the context of neural networks.

* **Reverse-mode automatic differentiation.**
There are a lot of [built-in operations](https://docs.rs/autograd/0.9.3/autograd/ops/index.html)
that support *higher-order* derivatives, and
you can [define your own ops](https://docs.rs/autograd/0.9.3/autograd/op/trait.Op.html) with ndarrays easily.

* **Pure Rust.**
The graph execution engine is implemented in pure Rust, so it's compilable to WebAssembly.

## Installation
```
[dependencies]
autograd = { version = "0.9.3", features = ["mkl"] }
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
// This achieves 0.918 test accuracy after 3 epochs, 0.11 sec/epoch on 2.7GHz Intel Core i5
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
    ag::eval(update_ops, &[(x, &x_batch), (y, &y_batch)]);
}

```

## Defining your own differentiable operations
Many of well-known ops are pre-defined in `ag::ops`, but you can also
implement custom ops by hand.

```rust
extern crate ndarray;
extern crate autograd as ag;

type NdArray<T: ag::Float> = ndarray::Array<T, ndarray::IxDyn>;

// Implements `Op` trait for `Sigmoid`.
struct Sigmoid;

impl<T: ag::Float> ag::op::Op<T> for Sigmoid {

    fn name(&self) -> &str {
        "Sigmoid"
    }

    // Core function to run this op.
    // Any errors in this function must be reported by *panic*.
    fn compute<'v>(
        &self,
        ctx: ag::runtime::OpComputeContext<'v, T>,
    ) -> ag::op::ComputeResults<'v, T> {
        let xs = ctx.grab_inputs();
        let x = &xs[0];
        // Using `ndarray::Array::mapv` for element-wise computation.
        let half = T::from(0.5).unwrap();
        let y = x.mapv(|a| ((a * half).tanh() * half) + half);
        // In some cases, you can return `ag::ArrRepr::View` for input arrays
        // to reduce unnecessary copies.
        vec![Ok(ag::ArrRepr::Owned(y))]
    }

    fn grad(&self, gy: &ag::Tensor<T>, xs: &[&ag::Tensor<T>], y: &ag::Tensor<T>)
        -> Vec<Option<ag::Tensor<T>>>
    {
        // Symbolic gradient of `x`
        let gx = gy * (y - ag::square(y));
        vec![Some(gx)]
    }
}

// Symbolic `sigmoid` function for end-user.
fn sigmoid<T: ag::Float>(x: &ag::Tensor<T>) -> ag::Tensor<T>
{
    ag::Tensor::builder()
        .set_inputs(vec![x])
        .set_shape(x.shape())
        .build(Sigmoid)
}
```

## Debugging
You can register hooks on `ag::Tensor` objects.
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

For more, see [documentation](https://docs.rs/autograd/) or
[examples](https://github.com/raskr/rust-autograd/tree/master/examples)
