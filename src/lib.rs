#[allow(unused_imports)]
// Expose to prevent version conflict
#[macro_use(s)]
pub extern crate ndarray;
pub extern crate arrayvec;
extern crate crossbeam;
extern crate hashbrown;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
extern crate libc;
#[cfg(not(feature = "mkl"))]
extern crate matrixmultiply;
extern crate num;
extern crate num_traits;
extern crate rand;
extern crate rayon;
extern crate rustc_hash;

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
pub(crate) type FxHashSet<V> = hashbrown::HashSet<V, BuildHasherDefault<FxHasher>>;

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

pub use crate::ndarray_ext as array;

pub use crate::op::Op;

pub use crate::ndarray_ext::{NdArray, NdArrayView, NdArrayViewMut};

pub use crate::runtime::{Eval, Feed};

pub use crate::tensor::Tensor;

pub use crate::ndarray_ext::ArrRepr;

pub use crate::graph::{with, Graph};

#[inline]
pub(crate) unsafe fn uninitialized_vec<T>(size: usize) -> Vec<T> {
    let mut buf = Vec::with_capacity(size);
    buf.set_len(size);
    buf
}

#[inline]
pub(crate) fn none_vec<T>(len: usize) -> Vec<Option<T>> {
    let mut vec = Vec::with_capacity(len);
    for _ in 0..len {
        vec.push(None);
    }
    vec
}

// Read pointer to type `A` as type `B`.
//
// **Panics** if `A` and `B` are not the same type
#[inline]
pub(crate) fn cast_as<A: 'static + Copy, B: 'static + Copy>(a: &A) -> B {
    unsafe { ::std::ptr::read(a as *const _ as *const B) }
}
