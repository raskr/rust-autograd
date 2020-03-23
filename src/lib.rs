//! Differentiable operations and tensors backed by [ndarray](https://github.com/rust-ndarray/ndarray).
//!
//! ## Features
//! ### Lazy, zero-copy tensor evaluation
//! Computation graphs are created on the fly (a.k.a. *define-by-run*), but are not evaluated until `Tensor::eval` or `ag::eval` is called.
//! This mechanism balances better performance and flexibility.
//! ```rust
//! use autograd as ag;
//!
//! ag::with(|g: &mut ag::Graph<_>| {
//!     let a: ag::Tensor<f32> = g.ones(&[60]);
//!     let b: ag::Tensor<f32> = g.ones(&[24]);
//!     let c: ag::Tensor<f32> = g.reshape(a, &[3, 4, 5]);
//!     let d: ag::Tensor<f32> = g.reshape(b, &[4, 3, 2]);
//!     let e: ag::Tensor<f32> = g.tensordot(c, d, &[1, 0], &[0, 1]);
//!     let result: ag::ndarray::Array<_, _> = e.eval(&[]).unwrap();  // Getting `ndarray::Array` here.
//! });
//! ```
//!
//! ### Reverse-mode automatic differentiation
//! There are a lot of [built-in operations](https://docs.rs/autograd/0.9.8/autograd/ops/index.html)
//! that support *higher-order* derivatives, and
//! you can also [define your own differentiable ops](https://docs.rs/autograd/0.9.8/autograd/op/trait.Op.html) with ndarrays easily.
//!
//! Here we are just computing partial derivatives of `z = 2x^2 + 3y + 1`.
//!
//! ```rust
//! use autograd as ag;
//!
//! # fn main() {
//! ag::with(|g: &mut ag::Graph<_>| {
//!     let x = g.placeholder(&[]);
//!     let y = g.placeholder(&[]);
//!     let z = 2.*x*x + 3.*y + 1.;
//!
//!     // dz/dy
//!     let gy = &g.grad(&[z], &[y])[0];
//!     println!("{:?}", gy.eval(&[]));   // => Some(3.)
//!
//!     // dz/dx (requires to fill the placeholder `x`)
//!     let gx = &g.grad(&[z], &[x])[0];
//!     let feed = ag::ndarray::arr0(2.);
//!     println!("{:?}", gx.eval(&[x.given(feed.view())]));  // => Some(8.)
//!     // ddz/dx (differentiates `z` again)
//!     let ggx = &g.grad(&[gx], &[x])[0];
//!     println!("{:?}", ggx.eval(&[]));  // => Some(4.)
//! });
//! # }
//! ```
//!

#[allow(unused_imports)]
// Expose to prevent version conflict
#[macro_use(s)]
/// re-exported for convenience and version-compatibility
pub extern crate ndarray;
extern crate hashbrown;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
extern crate libc;
#[cfg(not(feature = "mkl"))]
extern crate matrixmultiply;
extern crate num;
extern crate num_traits;
/// re-exported for convenience and version-compatibility
pub extern crate rand;
extern crate rand_distr;
extern crate rayon;
extern crate rustc_hash;
pub(crate) extern crate smallvec;

mod gradient;
pub(crate) mod graph;
mod hook;
pub mod ndarray_ext;
pub mod op;
mod ops;
pub mod optimizers;
mod runtime;
pub mod tensor;
pub mod test_helper;

use rustc_hash::FxHasher;
use std::any::TypeId;
use std::fmt;
use std::hash::BuildHasherDefault;

pub(crate) type FxHashMap<K, V> = hashbrown::HashMap<K, V, BuildHasherDefault<FxHasher>>;

/// Primitive type in this crate, which is actually a decorated `num_traits::Float`.
pub trait Float:
    num_traits::Float
    + num_traits::NumAssignOps
    + Copy
    + Send
    + Sync
    + fmt::Display
    + fmt::Debug
    + Sized
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

pub use crate::runtime::{Eval, Feed};

pub use crate::tensor::Tensor;

pub(crate) use crate::ndarray_ext::ArrRepr;

pub use crate::graph::{with, Graph};

#[inline]
pub(crate) unsafe fn uninitialized_vec<T>(size: usize) -> Vec<T> {
    let mut buf = Vec::with_capacity(size);
    buf.set_len(size);
    buf
}

// Read pointer to type `A` as type `B`.
//
// **Panics** if `A` and `B` are not the same type
#[inline]
pub(crate) fn cast_as<A: 'static + Copy, B: 'static + Copy>(a: &A) -> B {
    unsafe { ::std::ptr::read(a as *const _ as *const B) }
}

/// Error during tensor's evaluation.
#[derive(Debug, PartialEq)]
pub enum EvalError {
    /// Error during `Op`'s computation.
    OpError(op::OpError),
    /// A value of tensor is empty.
    ///
    /// For example, compute results of inplace ops (e.g. optimizers) are not available
    /// and are represented as `Empty`.
    Empty,
}

impl std::error::Error for EvalError {}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EvalError::OpError(e) => e.fmt(f),
            EvalError::Empty => write!(f, "Empty return value from a stateful op"),
        }
    }
}
