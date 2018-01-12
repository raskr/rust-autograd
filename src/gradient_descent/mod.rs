pub mod optimizers;

pub use self::optimizers::*;

use ndarray_ext::NdArray;
use tensor::Tensor;


#[inline]
/// Updates shared variables with its gradients
///
/// This actually runs the computation graph.
/// For the usage, see `examples` dir in repo.
pub fn update<'a, 'b, T, U, O>(variables: &[&Tensor], gradients: &[T], optimizer: &mut O, feeds: U)
where
    T: AsRef<Tensor>,
    U: IntoIterator<Item = &'b (&'b Tensor, &'b NdArray)>,
    O: Optimizer,
{
    assert_eq!(variables.len(), gradients.len());
    // run graph and get gradient arrays
    let gradient_refs = gradients.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
    let grad_arrays = ::eval::eval(gradient_refs.as_slice(), feeds);
    for (v, g) in variables.iter().zip(grad_arrays.iter()) {
        optimizer.update(v, g);
    }
}

/// Trait for any gradient descent optimizer
pub trait Optimizer {
    #[inline]
    /// Updates the variable tensor
    ///
    /// Updates `target` with `grad`.
    fn update(&mut self, target: &Tensor, grad: &NdArray);
}
