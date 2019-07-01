//! Module defining stochastic gradient descent optimizer.
use crate::op;
use crate::tensor::Tensor;
use crate::Float;

struct SGDOp<T: Float> {
    pub lr: T,
}

impl<T: Float> crate::op::Op<T> for SGDOp<T> {
    fn name(&self) -> &str {
        "SGD"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let xs = ctx.grab_inputs();
        let grad = &xs[1];
        unsafe {
            if let Some(arr) = ctx.node(0).get_persistent_array_mut() {
                arr.scaled_add(-self.lr, grad);
            }
        }
        vec![Err(crate::op::ComputeException::NoOutput)]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

/// Vanilla SGD optimizer
///
/// ```
/// extern crate autograd as ag;
///
/// let sgd = ag::gradient_descent_ops::SGD { lr: 0.1 };
/// // let update_ops = sgd.compute_updates(params, grads)
/// ```
///
/// See also https://github.com/raskr/rust-autograd/blob/master/examples/mlp_mnist.rs
pub struct SGD<T: Float> {
    /// Learning rate
    pub lr: T,
}

impl<'a, T: Float> SGD<T> {
    /// Creates ops to optimize `params` with SGD.
    ///
    /// Evaluated results of the return values will be `None`.
    pub fn compute_updates<A: AsRef<Tensor<T>>>(
        &self,
        params: &[&'a Tensor<T>],
        grads: &[A],
    ) -> Vec<Tensor<T>> {
        params
            .into_iter()
            .zip(grads)
            .map(|(param, grad)| {
                Tensor::builder()
                    .set_inputs(vec![param, grad.as_ref()])
                    .build(SGDOp { lr: self.lr })
            })
            .collect()
    }
}
