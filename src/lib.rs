//! This library provides differentiable operations and tensors. The current
//! backend is rust-ndarray.
//!
//! # Examples
//!
//! Here we are computing partial derivatives of `z = 2x^2 + 3y + 1`.
//!
//! ```rust
//! extern crate autograd as ag;
//! extern crate ndarray;
//! # fn main() {
//!
//! let ref x = ag::placeholder(&[]);
//! let ref y = ag::placeholder(&[]);
//! let ref z = 2.*x*x + 3.*y + 1.;
//!
//! // dz/dy
//! let gy = &ag::grad(&[z], &[y])[0];
//! println!("{:?}", gy.eval(&[]));   // => Some(3.)
//!
//! // dz/dx (requires to fill the placeholder `x`)
//! let gx = &ag::grad(&[z], &[x])[0];
//! println!("{:?}", gx.eval(&[(x, &ndarray::arr0(2.).into_dyn())]));  // => Some(8.)
//!
//! // ddz/dx (differentiates `z` again)
//! let ggx = &ag::grad(&[gx], &[x])[0];
//! println!("{:?}", ggx.eval(&[]));  // => Some(4.)
//! # }
//! ```
//!
//! Another example: softmax regression for MNIST digits classification.
//!
//! ```rust
//! extern crate autograd as ag;
//! # fn main() {
//! // -- graph def --
//! let ref w = ag::variable(ag::ndarray_ext::glorot_uniform::<f32>(&[28*28, 10]));
//! let ref b = ag::variable(ag::ndarray_ext::zeros::<f32>(&[1, 10]));
//! let ref x = ag::placeholder(&[-1, 28*28]);
//! let ref y = ag::placeholder(&[-1]);
//! let ref z = ag::matmul(x, w) + b;
//! let ref loss = ag::reduce_mean(&ag::sparse_softmax_cross_entropy(z, y), &[0, 1], false);
//! let ref params = [w, b];
//! let ref grads = ag::grad(&[loss], params);
//! let ref predictions = ag::argmax(z, -1, true);
//! let ref accuracy = ag::reduce_mean(&ag::equal(predictions, y), &[0], false);
//! let ref adam = ag::gradient_descent_ops::Adam::default();
//! let mut stateful_params = ag::gradient_descent_ops::Adam::vars_with_states(params);
//! let ref update_ops = adam.compute_updates(&stateful_params, grads);
//!
//! // -- dataset --
//! // let ((x_train, y_train), (x_test, y_test)) = dataset::load();
//! //
//! // -- training loop --
//! // for epoch in 0..30 {
//!     // ...
//!     // ag::run(update_ops, &[(x, &x_batch), (y, &y_batch)]);
//! // }
//! # }
//! ```
#[allow(unused_imports)]
#[macro_use(s)]
extern crate ndarray;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
extern crate libc;
#[cfg(not(feature = "mkl"))]
extern crate matrixmultiply;
extern crate num;
extern crate num_traits;
extern crate rand;
extern crate rayon;

#[macro_use]
#[doc(hidden)]
pub mod test_helper;

pub mod tensor;

#[doc(hidden)]
pub mod runtime;

#[doc(hidden)]
pub mod gradient;

pub mod ops;

pub mod ndarray_ext;

pub mod op;

use std::any::TypeId;
use std::fmt;

pub trait Float:
    num_traits::Float
    + num_traits::NumAssignOps
    + Copy
    + Send
    + Sync
    + fmt::Display
    + fmt::Debug
    + 'static
{
}

pub trait Int:
    num::Integer
    + num_traits::NumAssignOps
    + num_traits::ToPrimitive
    + Copy
    + Send
    + fmt::Display
    + 'static
{
}

impl<T> Float for T where
    T: num::Float
        + num_traits::NumAssignOps
        + Copy
        + Send
        + Sync
        + fmt::Display
        + fmt::Debug
        + 'static
{
}

impl<T> Int for T where
    T: num::Integer
        + num_traits::NumAssignOps
        + num_traits::ToPrimitive
        + Copy
        + Send
        + Sync
        + fmt::Display
        + 'static
{
}

#[doc(hidden)]
#[inline(always)]
/// Return `true` if `A` and `B` are the same type
pub fn same_type<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

pub use ndarray_ext::array_gen;

pub use ops::*;

pub use ops::gradient_descent_ops;

#[doc(hidden)]
pub use ndarray_ext::NdArray;

pub use runtime::{eval, Eval};

pub use tensor::Tensor;
