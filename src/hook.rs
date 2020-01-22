//! Defining things used with `Tensor::hook`.
use super::*;
use crate::ndarray_ext::NdArrayView;
use std::marker::PhantomData;

pub(crate) trait Hook<T: Float> {
    /// Calls this hook with the value of the tensor where this hook is set.
    fn call(&self, arr: &crate::ndarray::ArrayViewD<T>) -> ();
}

pub(crate) struct Print(pub &'static str);

pub(crate) struct Show;

pub(crate) struct ShowWith(pub &'static str);

pub(crate) struct ShowShape;

pub(crate) struct ShowShapeWith(pub &'static str);

// Calls the given function.
pub(crate) struct Raw<T: Float, FUN: Fn(&NdArrayView<T>) -> () + Send + Sync> {
    pub(crate) raw: FUN,
    pub(crate) phantom: PhantomData<T>,
}

impl<T: Float, FUN: Fn(&NdArrayView<T>) -> () + Send + Sync> Hook<T> for Raw<T, FUN> {
    fn call(&self, arr: &crate::ndarray_ext::NdArrayView<T>) -> () {
        (self.raw)(arr)
    }
}

impl<T: Float> Hook<T> for Print {
    fn call(&self, _: &crate::ndarray_ext::NdArrayView<T>) -> () {
        println!("{}\n", self.0);
    }
}

impl<T: Float> Hook<T> for Show {
    fn call(&self, arr: &crate::ndarray_ext::NdArrayView<T>) -> () {
        println!("{:?}\n", arr);
    }
}

impl<T: Float> Hook<T> for ShowWith {
    fn call(&self, arr: &crate::ndarray_ext::NdArrayView<T>) -> () {
        println!("{}\n {:?}\n", self.0, arr);
    }
}

impl<T: Float> Hook<T> for ShowShape {
    fn call(&self, arr: &crate::ndarray_ext::NdArrayView<T>) -> () {
        println!("{:?}\n", arr.shape());
    }
}

impl<T: Float> Hook<T> for ShowShapeWith {
    fn call(&self, arr: &crate::ndarray_ext::NdArrayView<T>) -> () {
        println!("{}\n{:?}\n", self.0, arr.shape());
    }
}
