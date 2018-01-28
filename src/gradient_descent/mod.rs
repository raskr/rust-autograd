pub mod optimizers;

// Expose all optimizers
pub use self::optimizers::*;

use ndarray_ext::NdArray;
use std::cmp::{Eq, Ordering, PartialEq};
use tensor::Tensor;


#[inline]
/// Updates shared variables with its gradients
///
/// This actually runs the computation graph.
/// For the usage, see `examples` dir in repo.
pub fn update<'a, 'b, T, U, O>(
    variables: &[&'b Tensor],
    gradients: &[T],
    optimizer: &mut O,
    feeds: U,
) where
    T: AsRef<Tensor>,
    U: IntoIterator<Item = &'a (&'a Tensor, &'a NdArray)>,
    O: Optimizer<'b>,
{
    assert_eq!(variables.len(), gradients.len());
    // run graph and get gradient arrays
    let grad_refs = gradients.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
    let grad_arrays = ::eval::eval(grad_refs.as_slice(), feeds);
    for (v, g) in variables.iter().zip(grad_arrays) {
        optimizer.update(v, g);
    }
}


/// Trait for any gradient descent optimizer
pub trait Optimizer<'a> {
    #[inline]
    /// Updates the variable tensor
    ///
    /// Updates `param` with `grad`.
    fn update(&mut self, param: &'a Tensor, grad: NdArray);
}


/// An access key for a state tensor.
/// This is used for stateful optimizers.
pub struct StateKey<'a>(pub &'a Tensor);

impl<'a> Tensor {
    #[inline]
    pub fn as_state_key(&'a self) -> StateKey<'a>
    {
        StateKey(self)
    }
}

impl<'a> Eq for StateKey<'a> {}

impl<'a> PartialEq for StateKey<'a> {
    #[inline]
    fn eq(&self, other: &StateKey<'a>) -> bool
    {
        let ptr1 = self.0 as *const _;
        let ptr2 = other.0 as *const _;
        ptr1 == ptr2
    }
}

impl<'a> Ord for StateKey<'a> {
    #[inline]
    // Compare the addresses of two tensors.
    // This can be used for ordering-based data structures (e.g. BinaryTree).
    fn cmp(&self, other: &Self) -> Ordering
    {
        let ptr1 = self.0 as *const _;
        let ptr2 = other.0 as *const _;
        ptr1.cmp(&ptr2)
    }
}

impl<'a> PartialOrd for StateKey<'a> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering>
    {
        Some(self.cmp(other))
    }
}
