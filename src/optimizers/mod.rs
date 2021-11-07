//! A collection of gradient descent optimizers
pub mod adam;
pub mod sgd;
pub mod momentum_sgd;

use crate::evaluation::Feeder;
use crate::tensor::Tensor;
use crate::variable::{VariableID, VariableNamespace};
use crate::{Context, Float, NdArray, VariableEnvironment};

/// Differentiates `loss` with all variables in the `namespace`
///
/// Returns a tuple `(variables, gradients)`
pub fn grad_helper<'g, A, F: Float>(
    loss: A,
    namespace: &'g VariableNamespace<F>,
) -> (Vec<Tensor<'g, F>>, Vec<Tensor<'g, F>>)
where
    A: AsRef<Tensor<'g, F>> + Copy,
{
    use crate::tensor_ops as T;

    let loss = loss.as_ref();
    let g = loss.graph;
    let variables: Vec<Tensor<F>> = g.var_tensors_by_name(namespace).map(|(a, b)| b).collect();

    let grads = T::grad(&[loss.as_ref()], &variables);
    (variables, grads)
}

/// Trait for gradient descent optimizers
pub trait Optimizer<F: Float> {
    /// Creates dummy tensors to update `variables`
    ///
    /// It's not supposed to be called directly from the outside (use [get_update_op] instead).
    fn compute_updates<'g, A, B>(
        &self,
        variables: &[A],
        grads: &[B],
        g: &'g Context<F>,
    ) -> Vec<Tensor<'g, F>>
    where
        A: AsRef<Tensor<'g, F>> + Copy,
        B: AsRef<Tensor<'g, F>> + Copy;

    /// Runs the graph and updates the variable arrays.
    ///
    /// Updates `variables` destructively.
    fn update<'g, A, B>(&self, variables: &[A], grads: &[B], g: &'g Context<F>, feeder: Feeder<F>)
    where
        A: AsRef<Tensor<'g, F>> + Copy,
        B: AsRef<Tensor<'g, F>> + Copy,
    {
        let mut evaluator = g.evaluator();
        evaluator.set_feeder(feeder);
        // get updates
        let update_ops = self.compute_updates(variables, grads, g);
        evaluator
            .extend(&update_ops)
            .run() // update runs
            .into_iter()
            .for_each(|r| {
                r.unwrap();
            });
    }

    /// Returns a tensor to update the given parameters
    ///
    /// Note that `variables` will not be updated until the return value is evaluated.
    fn get_update_op<'g, A, B>(
        &self,
        variables: &[A],
        grads: &[B],
        g: &'g Context<F>,
    ) -> Tensor<'g, F>
    where
        A: AsRef<Tensor<'g, F>> + Copy,
        B: AsRef<Tensor<'g, F>> + Copy,
    {
        crate::tensor_ops::add_n(&self.compute_updates(variables, grads, g))
    }
}
