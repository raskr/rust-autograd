//! Defining things related to hooks
//!
//! You can register hooks on `ag::Tensor` objects for debugging.
//! ```rust
//! use autograd as ag;
//! use ag::tensor_ops as T;
//!
//! ag::run(|ctx| {
//!    let a: ag::Tensor<f32> = T::zeros(&[4, 2], ctx).show();
//!    let b: ag::Tensor<f32> = T::ones(&[2, 3], ctx).show_shape();
//!    let c = T::matmul(a, b).show_prefixed("MatMul:");
//!
//!    c.eval( ctx);
//!    // [[0.0, 0.0],
//!    // [0.0, 0.0],
//!    // [0.0, 0.0],
//!    // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
//!    //
//!    // [2, 3]
//!    //
//!    // MatMul:
//!    //  [[0.0, 0.0, 0.0],
//!    //  [0.0, 0.0, 0.0],
//!    //  [0.0, 0.0, 0.0],
//!    //  [0.0, 0.0, 0.0]] shape=[4, 3], strides=[3, 1], layout=C (0x1), dynamic ndim=2
//!
//!    // raw hook
//!    let a: ag::Tensor<f32> = T::zeros(&[4, 2], ctx)
//!       .raw_hook(|x| println!("{:?}", x.shape()));
//! });
//!
//! ```
use super::*;
use crate::ndarray_ext::NdArrayView;
use std::marker::PhantomData;

/// Trait for hooks
///
/// hooks can be set using [crate::tensor::Tensor::register_hook()] method.
pub trait Hook<T: Float> {
    fn call(&self, arr: &crate::ndarray::ArrayViewD<T>) -> ();
}

pub struct Print(pub &'static str);

pub struct Show;

pub struct ShowPrefixed(pub &'static str);

pub struct ShowShape;

pub struct ShowPrefixedShape(pub &'static str);

// Calls the given function.
pub struct Raw<T: Float, FUN: Fn(&NdArrayView<T>) -> () + Send + Sync> {
    pub(crate) raw: FUN,
    pub(crate) phantom: PhantomData<T>,
}

impl<T: Float, FUN: Fn(&NdArrayView<T>) -> () + Send + Sync> Hook<T> for Raw<T, FUN> {
    fn call(&self, arr: &crate::ndarray_ext::NdArrayView<T>) {
        (self.raw)(arr)
    }
}

impl<T: Float> Hook<T> for Print {
    fn call(&self, _: &crate::ndarray_ext::NdArrayView<T>) {
        println!("{}", self.0);
    }
}

impl<T: Float> Hook<T> for Show {
    fn call(&self, arr: &crate::ndarray_ext::NdArrayView<T>) {
        println!("{:?}", arr);
    }
}

impl<T: Float> Hook<T> for ShowPrefixed {
    fn call(&self, arr: &crate::ndarray_ext::NdArrayView<T>) {
        println!("{} {:?}", self.0, arr);
    }
}

impl<T: Float> Hook<T> for ShowShape {
    fn call(&self, arr: &crate::ndarray_ext::NdArrayView<T>) {
        println!("{:?}", arr.shape());
    }
}

impl<T: Float> Hook<T> for ShowPrefixedShape {
    fn call(&self, arr: &crate::ndarray_ext::NdArrayView<T>) {
        println!("{}\n{:?}", self.0, arr.shape());
    }
}
