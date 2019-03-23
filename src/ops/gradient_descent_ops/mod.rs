extern crate ndarray;

pub mod adam;
#[allow(dead_code)]
pub mod sgd;

pub use self::adam::Adam;
pub use self::sgd::SGD;

use std::cmp::{Eq, Ordering, PartialEq};
use tensor::Tensor;
use Float;

/// Key to access a state tensor.
/// Stateful optimizers use this.
pub struct StateKey<'a, T: Float + 'a>(pub &'a Tensor<T>);

impl<'a, T: Float> Eq for StateKey<'a, T> {}

impl<'a, T: Float> PartialEq for StateKey<'a, T> {
    #[inline]
    /// Compares addresses of the two tensors.
    /// This can be used for ordering-based data structures (e.g. BinaryTree).
    fn eq(&self, other: &StateKey<'a, T>) -> bool {
        (self.0 as *const _) == (other.0 as *const _)
    }
}

impl<'a, T: Float> Ord for StateKey<'a, T> {
    #[inline]
    /// Compares addresses of the two tensors.
    /// This can be used for ordering-based data structures (e.g. BinaryTree).
    fn cmp(&self, other: &Self) -> Ordering {
        let a = self.0 as *const Tensor<T>;
        let b = other.0 as *const Tensor<T>;
        a.cmp(&b)
    }
}

impl<'a, T: Float> PartialOrd for StateKey<'a, T> {
    #[inline]
    /// Compares addresses of the two tensors.
    /// This can be used for ordering-based data structures (e.g. BinaryTree).
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
