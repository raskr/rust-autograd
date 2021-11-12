//! Differentiable operations and tensors backed by [ndarray](https://github.com/rust-ndarray/ndarray).
//!
//! ## Enabling blas
//! If you use basic linalg operations, especially matrix multiplications, `blas` feature would be important to speed them up.
//!
//! ```toml
//! [dependencies]
//! autograd = {"<version>", features = ["blas", "<blas-implementation-choice>"] }
//! ```
//! `<blas-implementation-choice>` must be one of the following (See also [blas-src](https://github.com/blas-lapack-rs/blas-src))
//! - `accelerate` macOS only
//! - `intel-mkl` Intel/AMD CPU only. Includes Vector Mathematics (VM) ops
//! - `openblas`
//!
//! ## Features
//! ### Reverse-mode automatic differentiation using lazy tensors
//! Here we are just computing partial derivatives of `z = 2x^2 + 3y + 1`.
//!
//! ```rust
//! use autograd as ag;
//! use ag::tensor_ops as T;
//!
//! # fn main() {
//! ag::run(|ctx: &mut ag::Context<_>| {
//!     let x = ctx.placeholder("x", &[]);
//!     let y = ctx.placeholder("y", &[]);
//!     let z = 2.*x*x + 3.*y + 1.;
//!
//!     // dz/dy
//!     let gy = &T::grad(&[z], &[y])[0];
//!     println!("{:?}", gy.eval(ctx));   // => Ok(3.)
//!
//!     // dz/dx (requires to fill the placeholder `x`)
//!     let gx = &T::grad(&[z], &[x])[0];
//!     let feed = ag::ndarray::arr0(2.);
//!     println!("{:?}", ctx.evaluator().push(gx).feed(x, feed.view()).run()[0]);  // => Ok(8.)
//!
//!     // ddz/dx (differentiates `z` again)
//!     let ggx = &T::grad(&[gx], &[x])[0];
//!     println!("{:?}", ggx.eval(ctx));  // => Ok(4.)
//! });
//! # }
//! ```
//!
//! ### Neural networks
//! This crate has various low-level features inspired by tensorflow/theano to train neural networks.
//! Since computation graphs require only bare minimum of heap allocations, the overhead is small, even for complex networks.
//! ```rust
//! // MNIST digits classification model
//! use autograd as ag;
//! use ag::optimizers::adam::Adam;
//! use ag::tensor_ops::*;
//! use ag::prelude::*;
//!
//! let mut env = ag::VariableEnvironment::new();
//!
//! let rng = ag::ndarray_ext::ArrayRng::<f32>::default();
//!
//! // Register variables in the default namespace
//! env.name("w").set(rng.glorot_uniform(&[28 * 28, 10]));
//! env.name("b").set(ag::ndarray_ext::zeros(&[1, 10]));
//!
//! let adam = Adam::default("my_adam", env.default_namespace().current_var_ids(), &mut env);
//!
//! for epoch in 0..3 { // 0.11 sec/epoch on 2.7GHz Intel Core i5
//!     env.run(|ctx| {
//!         let x = ctx.placeholder("x", &[-1, 28*28]);
//!         let y = ctx.placeholder("y", &[-1]);
//!         let w = ctx.variable("w");
//!         let b = ctx.variable("b");
//!         let z = matmul(x, w) + b;
//!         let mean_loss = reduce_mean(sparse_softmax_cross_entropy(z, &y), &[0], false);
//!         let grads = &grad(&[mean_loss], &[w, b]);
//!
//!         // let mut feeder = ag::Feeder::new();
//!         // feeder.push(x, x_batch).push(y, y_batch);
//!         // adam.update(&[w, b], grads, ctx, feeder);
//!     });
//! }
//! ```
//!
//! ### Other useful features
//! - [Model persistence](variable#model-persistence)
//! - [Variable namespace](variable#variable-and-namespace)
//! - [Hook](hooks)

#[allow(unused_imports)]
// Expose to prevent version conflict
#[macro_use(s)]
/// re-exported for convenience and version-compatibility
pub extern crate ndarray;

#[cfg(all(feature = "blas", feature = "intel-mkl"))]
extern crate intel_mkl_src;

#[cfg(all(feature = "blas", not(feature = "intel-mkl")))]
extern crate blas_src;
#[cfg(feature = "blas")]
extern crate cblas_sys;

extern crate libc;
extern crate matrixmultiply;
extern crate num;
extern crate num_traits;
/// re-exported for convenience and version-compatibility
pub extern crate rand;
extern crate rand_distr;
extern crate rayon;
extern crate rustc_hash;
extern crate serde_json;
pub(crate) extern crate smallvec;
extern crate uuid;
#[macro_use]
extern crate serde_derive;
extern crate approx;
extern crate special;

pub mod evaluation;
mod gradient;
pub(crate) mod graph;
pub mod hooks;
pub mod ndarray_ext;
pub mod op;
pub mod optimizers;
pub mod prelude;
pub mod tensor;
pub mod tensor_ops;
pub mod test_helper;
pub mod variable;

use rustc_hash::{FxHashMap, FxHashSet};
use std::any::TypeId;
use std::fmt;

/// A primitive type in this crate, which is actually a decorated `num_traits::Float`.
pub trait Float:
    num_traits::Float
    + num_traits::NumAssignOps
    + Copy
    + Send
    + Sync
    + fmt::Display
    + fmt::Debug
    + Sized
    + Serialize
    + Deserialize<'static>
    + 'static
{
}

#[doc(hidden)]
/// Internal trait.
pub trait Int:
    num::Integer
    + num_traits::NumAssignOps
    + num_traits::ToPrimitive
    + Copy
    + Send
    + fmt::Display
    + Sized
    + Serialize
    + Deserialize<'static>
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
        + Sized
        + Serialize
        + Deserialize<'static>
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
        + Sized
        + Serialize
        + Deserialize<'static>
        + 'static
{
}

#[inline(always)]
/// Return `true` if `A` and `B` are the same type
pub(crate) fn same_type<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

pub use crate::ndarray_ext::array_gen;

pub use crate::ndarray_ext::{NdArray, NdArrayView, NdArrayViewMut};

pub use crate::evaluation::{Evaluator, Feeder};

pub use crate::tensor::Tensor;

pub(crate) use graph::Graph;
pub(crate) use op::OpOutput;

pub use crate::graph::{run, Context};
pub use crate::variable::VariableEnvironment;
use serde::{Deserialize, Serialize};

/// Error during tensor's evaluation.
#[derive(Debug, PartialEq)]
pub enum EvalError {
    /// Error during `Op`'s computation.
    OpError(op::OpError),
}

impl std::error::Error for EvalError {}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EvalError::OpError(e) => e.fmt(f),
        }
    }
}
