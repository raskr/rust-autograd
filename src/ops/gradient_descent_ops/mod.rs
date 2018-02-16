extern crate ndarray;

pub mod adam;
pub mod sgd;

pub use self::adam::Adam;
pub use self::sgd::SGD;

use std::cmp::{Eq, Ordering, PartialEq};
use tensor::Tensor;

/// Trait for any gradient descent optimizer
pub trait Optimizer<'a>
{
    fn compute_updates<T: AsRef<Tensor>>(
        &mut self,
        param: &[&'a Tensor],
        grad: &[T],
    ) -> Vec<Tensor>;
}

/// Key to access a state tensor.
/// Stateful optimizers use this.
pub struct StateKey<'a>(pub &'a Tensor);

impl<'a> Eq for StateKey<'a> {}

impl<'a> PartialEq for StateKey<'a>
{
    #[inline]
    /// Compares addresses of the two tensors.
    /// This can be used for ordering-based data structures (e.g. BinaryTree).
    fn eq(&self, other: &StateKey<'a>) -> bool
    {
        (self.0 as *const _) == (other.0 as *const _)
    }
}

impl<'a> Ord for StateKey<'a>
{
    #[inline]
    /// Compares addresses of the two tensors.
    /// This can be used for ordering-based data structures (e.g. BinaryTree).
    fn cmp(&self, other: &Self) -> Ordering
    {
        let a = self.0 as *const Tensor;
        let b = other.0 as *const Tensor;
        a.cmp(&b)
    }
}

impl<'a> PartialOrd for StateKey<'a>
{
    #[inline]
    /// Compares addresses of the two tensors.
    /// This can be used for ordering-based data structures (e.g. BinaryTree).
    fn partial_cmp(&self, other: &Self) -> Option<Ordering>
    {
        Some(self.cmp(other))
    }
}
