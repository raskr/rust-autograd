pub mod optimizers;

pub use self::optimizers::*;

use ndarray_ext::NdArray;
use context::Context;
use tensor::Tensor;


#[inline]
/// Updates shared variables with its gradients
///
/// This actually runs the computation graph.
/// For the usage, see `examples` dir in repo.
pub fn update<T: Optimizer>(
    variables: &[&Tensor],
    gradients: &[Tensor],
    optimizer: &mut T,
    ctx: &mut Context,
)
{
    assert_eq!(variables.len(), gradients.len());
    // run graph and get gradient arrays
    let gradient_refs = gradients.iter().map(|a| a).collect::<Vec<_>>();
    let mut grad_arrays = ::eval::eval_tensors(gradient_refs.as_slice(), &mut ctx.variables,
                                               &mut ctx.outputs);
    ctx.outputs.clear();
    for v in variables {
        // safe unwrap
        assert_eq!(v.op.name(), "Variable", "Can't optimize non-variable");
        let mut v_arr = ctx.variables.get_mut(v).unwrap();
        let g = grad_arrays.remove(0);
        optimizer.update(v, v_arr, g);
    }
}

/// Trait for any gradient descent optimizer
pub trait Optimizer {
    #[inline]
    /// Updates the variable tensor
    ///
    /// Updates `param` with `grad`.
    /// `node` is a symbolic representation of `param`
    fn update(&mut self, node: &Tensor, param: &mut NdArray, grad: NdArray);
}