#[allow(unused_imports)]
// Expose to prevent version conflict
#[macro_use(s)]
pub extern crate ndarray;
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
    + Sized
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

#[doc(hidden)]
#[inline(always)]
/// Return `true` if `A` and `B` are the same type
pub fn same_type<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

pub use crate::ndarray_ext::array_gen;

pub use crate::ops::*;

pub use crate::ops::gradient_descent_ops;

pub use crate::ndarray_ext::NdArray;

pub use crate::runtime::{eval, Eval, Feed};

pub use crate::tensor::Tensor;

pub use crate::ops::Hook;

pub use crate::ndarray_ext::ArrRepr;

#[inline]
#[doc(hidden)]
pub fn uninitialized_vec<T>(size: usize) -> Vec<T> {
    let mut buf = Vec::with_capacity(size);
    unsafe {
        buf.set_len(size);
    }
    buf
}
